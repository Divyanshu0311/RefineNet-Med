import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f*2)
        self.enc3 = Down(f*2, f*4)
        self.enc4 = Down(f*4, f*8)
        self.bottleneck = ConvBlock(f*8, f*16)
        self.up4 = Up(f*16, f*8)
        self.up3 = Up(f*8, f*4)
        self.up2 = Up(f*4, f*2)
        self.up1 = Up(f*2, f)
        self.out_conv = nn.Conv2d(f, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.out_conv(d1)
        return out

class Refiner(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f, f*2)
        self.enc3 = Down(f*2, f*4)
        self.bottleneck = ConvBlock(f*4, f*8)
        self.up3 = Up(f*8, f*4)
        self.up2 = Up(f*4, f*2)
        self.up1 = Up(f*2, f)
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
