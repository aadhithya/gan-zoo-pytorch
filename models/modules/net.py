import torch
import torch.nn as nn

from collections import OrderedDict


class NetG(nn.Module):
    """
    NetG DCGAN Generator. Outputs 64x64 images.
    """

    def __init__(
        self,
        z_dim=100,
        out_ch=3,
        norm_layer=nn.BatchNorm2d,
        final_activation=None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation = final_activation

        self.net = nn.Sequential(
            # * Layer 1: 1x1
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias=False),
            norm_layer(512),
            nn.ReLU(),
            # * Layer 2: 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            norm_layer(256),
            nn.ReLU(),
            # * Layer 3: 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            norm_layer(128),
            nn.ReLU(),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            norm_layer(64),
            nn.ReLU(),
            # * Layer 5: 32x32
            nn.ConvTranspose2d(64, self.out_ch, 4, 2, 1, bias=False),
            # * Output: 64x64
        )

    def forward(self, x):
        x = self.net(x)
        return (
            x if self.final_activation is None else self.final_activation(x)
        )


class NetD(nn.Module):
    def __init__(
        self, in_ch=3, norm_layer=nn.BatchNorm2d, final_activation=None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation = final_activation

        self.net = nn.Sequential(
            # * 64x64
            nn.Conv2d(self.in_ch, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # * 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            norm_layer(128, affine=True),
            nn.LeakyReLU(0.2),
            # * 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            norm_layer(256, affine=True),
            nn.LeakyReLU(0.2),
            # * 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            norm_layer(512, affine=True),
            nn.LeakyReLU(0.2),
            # * 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        return (
            x if self.final_activation is None else self.final_activation(x)
        )
