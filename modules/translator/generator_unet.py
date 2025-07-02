import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_bn=True, dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_bn))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_bn))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down1 = UNetBlock(in_channels, features, down=True, use_bn=False)
        self.down2 = UNetBlock(features, features * 2)
        self.down3 = UNetBlock(features * 2, features * 4)
        self.down4 = UNetBlock(features * 4, features * 8)
        self.down5 = UNetBlock(features * 8, features * 8)
        self.down6 = UNetBlock(features * 8, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.up1 = UNetBlock(features * 8, features * 8, down=False, dropout=True)
        self.up2 = UNetBlock(features * 16, features * 8, down=False, dropout=True)
        self.up3 = UNetBlock(features * 16, features * 8, down=False)
        self.up4 = UNetBlock(features * 16, features * 4, down=False)
        self.up5 = UNetBlock(features * 8, features * 2, down=False)
        self.up6 = UNetBlock(features * 4, features, down=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bn = self.bottleneck(d6)

        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        return self.final_up(torch.cat([u6, d1], dim=1))
