import math

import torch
import torch.nn as nn

from bnn_competition.libs.binarizers import Binarizer
from bnn_competition.libs.binary_modules import BinaryConv2d
from bnn_competition.libs.sign_approximators import STESignApproximator, STEWithClipSignApproximator
from bnn_competition.libs.transformations import L1Scaling


class LAB(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.fp_depthwise_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=2 * num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=num_channels,
        )
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 1.0

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.fp_depthwise_conv(x)
        x = x.reshape(n, c, 2, h, w)
        x_wo_grad = x.argmax(dim=2) * 2.0 - 1.0
        x_with_grad = self.softmax(self.temperature * x)[:, :, 1, :, :] * 2.0 - 1.0

        return (x_wo_grad - x_with_grad).detach() + x_with_grad


class Scale(nn.Module):
    """
    Scale the input according to the following formula: a * x + b.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.variance = nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        return self.variance * x + self.mean


class MeanShift(nn.Module):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), sign=-1):
        super(MeanShift, self).__init__()
        mean_shift = sign * rgb_range * torch.Tensor(rgb_mean)
        self.register_buffer("mean_shift", mean_shift.view(1, 3, 1, 1))

    def forward(self, x):
        return x + self.mean_shift


class ConvBlock(nn.Module):
    def __init__(self, n_features, group_size):
        super().__init__()
        input_binarizer = Binarizer(
            sign_approximator=STESignApproximator(),
        )
        weight_binarizer = Binarizer(
            sign_approximator=STEWithClipSignApproximator(),
            transformation=L1Scaling(dim=[1, 2, 3]),
            channel_wise=True,
        )

        self.bin_path = nn.Sequential(
            LAB(n_features),
            BinaryConv2d(
                input_binarizer=input_binarizer,
                weight_binarizer=weight_binarizer,
                in_channels=n_features,
                out_channels=n_features,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            Scale(n_features),
        )
        self.act = nn.PReLU(n_features)
        self.fp_depthwise_conv = nn.Conv2d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=n_features // group_size,
            bias=False,
        )
        self.fp_1x1_conv = nn.Conv2d(
            in_channels=n_features, out_channels=n_features, kernel_size=1, padding=0, bias=False
        )

    def forward(self, x):
        path_0 = self.bin_path(x)
        path_1 = self.fp_1x1_conv(x)
        path_2 = self.fp_depthwise_conv(path_1)
        return self.act(path_0 + path_1 + path_2) + x


class ResidualBlock(nn.Module):
    def __init__(self, n_features, group_size):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            ConvBlock(n_features, group_size),
            ConvBlock(n_features, group_size),
        )

    def forward(self, x):
        return self.res_block(x)


class ResidualBlockFP(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlockFP, self).__init__()

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


class BinaryScalex2BaselineV1(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        feature_channels=64,
        n_residual_blocks=16,
        scale=2,
        group_size=16,
        n_blocks_skip=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # define head modules
        modules_head = [nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=True)]

        # define body modules
        modules_body = []
        for id_block in range(n_residual_blocks):
            if n_blocks_skip <= id_block < n_residual_blocks - n_blocks_skip:
                modules_body.append(ResidualBlock(feature_channels, group_size))
            else:
                modules_body.append(ResidualBlockFP(feature_channels))
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
