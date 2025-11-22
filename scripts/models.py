import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Basic building blocks (your style)
# ------------------------------
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Note: matches your original design:
        - in_ch = number of channels arriving into the up block BEFORE transpose (e.g., f*16)
        - up layer produces out_ch channels, and concatenating with skip (out_ch) gives in_ch channels
        - self.conv expects in_ch (after concat) -> out_ch
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary (same as your original)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ------------------------------
# DUCK Block (minimal variant for bottleneck)
# ------------------------------
class DUCKBlock(nn.Module):
    """
    Minimal DUCK block (4-branch multi-scale fusion) designed to
    replace the UNet bottleneck. Input/output channels match the ConvBlock usage.
    Branches:
      - branch1: standard 3x3
      - branch2: depthwise separable (depthwise 3x3 + pointwise 1x1)
      - branch3: dilated conv (3x3, dilation=2)
      - branch4: 1x1 residual conv
    Then concatenate and fuse with 1x1 conv.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Branch 1: standard 3x3 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Branch 2: depthwise separable conv (depthwise 3x3, pointwise 1x1)
        # keep intermediate channels = in_ch to fit grouped conv requirements
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Branch 3: dilated conv (wider receptive field)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 1x1 conv residual/shortcut
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Fuse concatenated outputs (4 * out_ch -> out_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Optional lightweight SE-style channel attention for a small extra gain
        # Keeps the DUCK block compact; you may toggle this behavior if desired.
        self.use_se = True
        if self.use_se:
            self.se_pool = nn.AdaptiveAvgPool2d(1)
            self.se_fc = nn.Sequential(
                nn.Conv2d(out_ch, max(out_ch // 8, 8), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(out_ch // 8, 8), out_ch, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fuse(out)
        if self.use_se:
            w = self.se_pool(out)
            w = self.se_fc(w)
            out = out * w
        return out


# ------------------------------
# UNet (with DUCK at bottleneck only)
# ------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32, use_duck_bottleneck=True):
        super().__init__()
        f = base_filters
        self.use_duck_bottleneck = use_duck_bottleneck

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck: either ConvBlock or DUCKBlock
        if self.use_duck_bottleneck:
            self.bottleneck = DUCKBlock(f * 8, f * 16)
        else:
            self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder (note the in_ch/out_ch convention matches your original code)
        self.up4 = Up(f * 16, f * 8)
        self.up3 = Up(f * 8, f * 4)
        self.up2 = Up(f * 4, f * 2)
        self.up1 = Up(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)        # f
        e2 = self.enc2(e1)       # f*2
        e3 = self.enc3(e2)       # f*4
        e4 = self.enc4(e3)       # f*8
        b = self.bottleneck(e4)  # f*16 (DUCK or ConvBlock)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.out_conv(d1)
        return out


# ------------------------------
# Refiner (left unchanged - minimal DUCK impact)
# ------------------------------
class Refiner(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.bottleneck = ConvBlock(f * 4, f * 8)
        self.up3 = Up(f * 8, f * 4)
        self.up2 = Up(f * 4, f * 2)
        self.up1 = Up(f * 2, f)
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

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


# ------------------------------
# Patch Discriminator (same as yours)
# ------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_filters=16):
        super().__init__()
        f = base_filters
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, f, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(f, f * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(f * 2, f * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(f * 4, f * 8, 4, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(f * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)
