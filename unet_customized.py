import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_block(in_c, out_c, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class UNetCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(conv_block(3, 64), SEBlock(64))
        self.enc2 = nn.Sequential(conv_block(64, 128), SEBlock(128))
        self.enc3 = nn.Sequential(conv_block(128, 256), SEBlock(256))
        self.enc4 = nn.Sequential(conv_block(256, 512, dilation=2), SEBlock(512))  # dilated conv

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = nn.Sequential(conv_block(512+256, 256), SEBlock(256))
        self.dec2 = nn.Sequential(conv_block(256+128, 128), SEBlock(128))
        self.dec1 = nn.Sequential(conv_block(128+64, 64), SEBlock(64))

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.up(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
