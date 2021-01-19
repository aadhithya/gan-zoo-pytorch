import torch.nn as nn


class PixelNorm(nn.Module):
    """
    PixelNorm PixelNorm from PG GAN
    thanks https://github.com/facebookresearch/pytorch_GAN_zoo/blob/b75dee40918caabb4fe7ec561522717bf096a8cb/models/networks/custom_layers.py#L9

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


class IdentityLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)
