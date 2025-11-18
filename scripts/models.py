import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utility / small blocks ----------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Depthwise separable conv: pointwise -> depthwise -> pointwise (optionally)
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

# Asymmetric conv pair 1x3 followed by 3x1
class AsymmetricConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, (1,3), padding=(0,1), bias=False)
        self.conv2 = nn.Conv2d(ch, ch, (3,1), padding=(1,0), bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.act(x)

# Simple 1x1 conv to match channels when needed
class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------- LDA_B block (lightweight block with DW, asymmetric, SE, ADD residual) ----------
class LDA_B(nn.Module):
    """
    Practical LDA-B approximation:
    - Part I: 1x1 conv -> DWConv -> add BN(conv(x))
    - Part II: 1x1 conv -> asymmetric conv (1x3 + 3x1) -> add BN(conv(prev))
    - SE on intermediate
    - Final residual add to input (with 1x1 projection if channels change)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj_in = None
        if in_ch != out_ch:
            self.proj_in = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # Part I
        self.p1_pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.p1_dw = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.p1_bn_convx = nn.BatchNorm2d(out_ch)  # BN(Conv(x))
        self.p1_act = nn.ReLU(inplace=True)

        # Part II
        self.p2_pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.asconv = AsymmetricConv(out_ch)
        self.p2_bn_conv = nn.BatchNorm2d(out_ch)

        # SE and final act
        self.se = SEBlock(out_ch, reduction=8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # stage 1
        y = self.p1_pw(x)
        y = self.p1_dw(y)
        # BN(Conv(x)) part
        convx = F.relu(self.p1_bn_convx(self.p1_pw(x)), inplace=False)  # reuse pw conv on x then BN
        y = y + convx
        y = self.p1_act(y)

        # stage 2
        z = self.p2_pw(y)
        z = self.asconv(z)
        conv_y = F.relu(self.p2_bn_conv(self.p2_pw(y)), inplace=False)
        z = z + conv_y

        z = self.se(z)

        # final residual (project if needed)
        if self.proj_in is not None:
            identity = self.proj_in(identity)
        out = self.act(z + identity)
        return out

# ---------- LMLP (practical approximation) ----------
class LMLP(nn.Module):
    """
    Practical LMLP approximation operating on bottleneck feature map.
    Implements:
      - token shifts abstracted as 1x1 conv
      - MLP along width via depthwise conv with (1,3)
      - DWConv (3x3)
      - MLP along height via depthwise conv (3,1)
      - residual adds and LayerNorm
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Project in->hidden, hidden->out
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.LayerNorm([out_ch, 1, 1])  # will adapt when used via permute
        # width-direction DWConv (1x3)
        self.dw_w = nn.Conv2d(out_ch, out_ch, (1,3), padding=(0,1), groups=out_ch, bias=False)
        # inter conv to mix channels
        self.pw1 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        # height-direction DWConv (3x1)
        self.dw_h = nn.Conv2d(out_ch, out_ch, (3,1), padding=(1,0), groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(out_ch)

        # final projection (identity if same)
        self.proj_out = nn.Conv2d(out_ch, out_ch, 1, bias=False) if out_ch != in_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        x0 = self.proj_in(x)   # (B, out_ch, H, W)
        # Width-MLP approximation
        y = self.dw_w(x0)      # (B, out_ch, H, W)
        y = self.pw1(y)
        y = self.act(y)
        y = y + x0             # ADD 1

        # Height-MLP approximation
        z = self.dw_h(y)
        z = self.pw2(z)
        z = self.act(z)
        z = z + y              # ADD 2

        out = self.proj_out(z)
        return out

# ---------- Up block (same design as your code but uses LDA_B for conv) ----------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_lda=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # after cat, channels = out_ch + skip_ch. The calling code must set in_ch accordingly.
        # We accept conv_in_ch = out_ch + skip_ch at runtime, so we allow a generic conv module.
        # But to simplify, we expect `in_ch` passed here equals concatenated channels.
        # So implement conv as LDA_B(in_ch, out_ch) or a simple ConvBlock.
        self.use_lda = use_lda
        if use_lda:
            self.conv = LDA_B(in_ch, out_ch)
        else:
            self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # align
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ---------- ConvBlock kept for completeness (your original) ----------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# ---------- Modified UNet generator with LDA_B and LMLP bottleneck ----------
class HybridUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32, use_lda=True):
        super().__init__()
        f = base_filters
        # Encoder: replace ConvBlock with LDA modules
        self.enc1 = LDA_B(in_channels, f) if use_lda else ConvBlock(in_channels, f)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc2 = LDA_B(f, f*2) if use_lda else ConvBlock(f, f*2)
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc3 = LDA_B(f*2, f*4) if use_lda else ConvBlock(f*2, f*4)
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc4 = LDA_B(f*4, f*8) if use_lda else ConvBlock(f*4, f*8)

        # Bottleneck: use LMLP
        self.bottleneck = LMLP(f*8, f*16)

        # Decoder (Up blocks). Note: after up we cat skip => conv in_ch = out_ch + skip_ch
        # We set in_ch appropriately during init: for first up, up convtranspose maps f*16 -> f*8 then cat with enc4 (f*8) => in_ch = f*8 + f*8 = f*16
        self.up4 = Up(f*16, f*8, use_lda=use_lda)
        self.up3 = Up(f*8 + f*4, f*4, use_lda=use_lda)   # after previous, conv produces f*8 -> up to f*4, cat with enc3 => f*4 + f*4 = f*8 (we pass in_ch=f*8 to Up)
        self.up2 = Up(f*4 + f*2, f*2, use_lda=use_lda)
        self.up1 = Up(f*2 + f, f, use_lda=use_lda)

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)            # (B, f, H, W)
        e2 = self.enc2_pool(e1)
        e2 = self.enc2(e2)           # (B, f*2, H/2, W/2)
        e3 = self.enc3_pool(e2)
        e3 = self.enc3(e3)           # (B, f*4, H/4, W/4)
        e4 = self.enc4_pool(e3)
        e4 = self.enc4(e4)           # (B, f*8, H/8, W/8)

        b = self.bottleneck(e4)      # (B, f*16, H/8, W/8)

        # Up4: x=b, skip=e4. Up expects in_ch = (after up) + skip channels => we constructed Up to receive concatenated channels
        d4 = self.up4(b, e4)         # expects concatenated in channels inside Up class
        # d4 -> (B, f*8, H/4, W/4)
        d3 = self.up3(d4, e3)        # (B, f*4, H/2, W/2)
        d2 = self.up2(d3, e2)        # (B, f*2, H, W)
        d1 = self.up1(d2, e1)        # (B, f, H, W)
        out = self.out_conv(d1)
        return out

# ---------- Refiner (unchanged from your original, but using ConvBlock) ----------
class Refiner(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownSimple(f, f*2)
        self.enc3 = DownSimple(f*2, f*4)
        self.bottleneck = ConvBlock(f*4, f*8)
        self.up3 = UpSimple(f*8, f*4)
        self.up2 = UpSimple(f*4, f*2)
        self.up1 = UpSimple(f*2, f)
        self.out_conv = nn.Conv2d(f, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.out_conv(d1)
        return out

# Small helper Down and Up variants for Refiner to ensure they are simple and compatible
class DownSimple(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class UpSimple(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch here means features flowing in (not concatenated channels)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)  # after concat in_ch should be out_ch + skip_ch; for simplicity in refiner we assume symmetric sizes
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ---------- Patch Discriminator (same as your original) ----------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_filters=16):
        super().__init__()
        f = base_filters
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, f, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f, f*2, 4, stride=2, padding=1), nn.BatchNorm2d(f*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*2, f*4, 4, stride=2, padding=1), nn.BatchNorm2d(f*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*4, f*8, 4, padding=1), nn.BatchNorm2d(f*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(f*8, 1, 4, padding=1)
        )
    def forward(self, x):
        return self.model(x)
