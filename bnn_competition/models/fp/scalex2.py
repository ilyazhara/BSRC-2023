import math

import torch
import torch.nn as nn


class MeanShift(nn.Module):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), sign=-1):
        super(MeanShift, self).__init__()
        mean_shift = sign * rgb_range * torch.Tensor(rgb_mean)
        self.register_buffer("mean_shift", mean_shift.view(1, 3, 1, 1))

    def forward(self, x):
        return x + self.mean_shift


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_features):
        modules = []
        if (scale & (scale - 1)) == 0:  # check if scale == 2^n
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(n_features, 4 * n_features, kernel_size=3, padding=1, bias=True))
                modules.append(nn.PixelShuffle(2))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)


class Scalex2Model(nn.Module):
    """
    Model is based on EDSR model.
    """

    def __init__(self, in_channels=3, out_channels=3, feature_channels=64, n_residual_blocks=16, scale=2, **kwargs):
        super().__init__(**kwargs)

        # define head modules
        modules_head = [nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=True)]

        # define body modules
        modules_body = [ResidualBlock(feature_channels) for _ in range(n_residual_blocks)]
        modules_body.append(nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True))

        # define tail modules
        modules_tail = [
            Upsampler(scale, feature_channels),
            nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1, bias=True),
        ]

        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        x = self.add_mean(x)

        return x
