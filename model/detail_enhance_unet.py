import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if not in_channels==out_channels:
            self.addtion_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.addtion_layer = None


    def forward(self, x):
        if self.addtion_layer is None:
            return x+self.res_conv(x)
        else:
            return self.addtion_layer(x) + self.res_conv(x)


class ResUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ResConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resdown_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.resdown_conv(x)


class FirstResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02, inplace=True),
        )
        if not in_channels==out_channels:
            self.addtion_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.addtion_layer = None


    def forward(self, x):
        if self.addtion_layer is None:
            return x+self.res_conv(x)
        else:
            return self.addtion_layer(x) + self.res_conv(x)


class UNet_Enhancer(nn.Module):
    def __init__(self, n_channels, n_classes, output_confidence=False, output_shadow_mask=True, bilinear=True, **kwargs):
        super(UNet_Enhancer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_1 = FirstResConv(n_channels, 64)  # -> 256x256
        self.down1_1 = ResDown(64, 128)  # -> 128x128
        self.down2_1 = ResDown(128, 256)  # -> 64x64
        self.down3_1 = ResDown(256, 512)  # -> 32x32
        self.down4_1 = ResDown(512, 512)  # -> 16x16
        self.up1_1 = ResUp(1024, 256, bilinear)  # -> 32x32
        self.up2_1 = ResUp(512, 128, bilinear)  # -> 64x64
        self.up3_1 = ResUp(256, 64, bilinear)  # -> 128x128
        self.up4_1 = ResUp(128, 64, bilinear)  # -> 256x256
        self.outc_1 = OutConv(64, n_classes)

        self.output_confidence = output_confidence
        self.output_shadow_mask = output_shadow_mask
        if self.output_confidence:
            self.outc_conf = OutConv(64, 1)
        if self.output_shadow_mask:
            self.outc_shadow = OutConv(64, 1)

        # self.encoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1),
        #                              nn.BatchNorm2d(512),
        #                              nn.ReLU(),
        #                              nn.Conv2d(512, 16*32*4, kernel_size=1, padding=0, stride=1))
        # self.activation = nn.Softplus()

    def forward(self, x):
        x1 = self.inc_1(x)
        x2 = self.down1_1(x1)
        x3 = self.down2_1(x2)
        x4 = self.down3_1(x3)
        x5 = self.down4_1(x4)

        # latent = self.encoder(x5)
        # h = latent.shape[2]
        # w = latent.shape[3]
        # i_l = self.activation(latent[:, :16*32*3].view(-1, 16*32, 3, h*w))
        # weights = F.softmax(latent[:, 16*32*3:].view(-1, 16*32, 1, h*w), dim=3)
        # i_l = (weights * i_l).sum(dim=3)
        # i_l = i_l.view(x.shape[0], 3, 16, 32)

        x = self.up1_1(x5, x4)
        x = self.up2_1(x, x3)
        x = self.up3_1(x, x2)
        x = self.up4_1(x, x1)

        out_color = self.outc_1(x)
        out_color = torch.sigmoid(out_color)
        out_dict = {
            "diffuse": out_color
        }
        
        if self.output_confidence:
            out_conf = self.outc_conf(x)
            out_conf = 1 + torch.exp(out_conf)
            out_dict["confidence"] = out_conf
        
        if self.output_shadow_mask:
            out_shadow = self.outc_shadow(x)
            out_shadow = torch.sigmoid(out_shadow)
            out_dict["shadow_mask"] = out_shadow
        
        return out_dict


if __name__ == "__main__":
    from utils import count_parameters
    model = UNet_Delight(n_channels=3, n_classes=3)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)

    count_parameters(model, detailed=True)
