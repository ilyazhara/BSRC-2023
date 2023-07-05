import math

import torch
import torch.nn as nn

from bnn_competition.libs.binarizers import Binarizer
from bnn_competition.libs.binary_modules import BinaryConv2d
from bnn_competition.libs.sign_approximators import STESignApproximator, STEWithClipSignApproximator
from bnn_competition.libs.transformations import L1Scaling


class Normalize(nn.Module):
    def __init__(self, mean=127.5, std=51.0, sign=-1):
        super(Normalize, self).__init__()
        self.mean_shift = mean
        self.scale_factor = std
        self.sign = sign

    def forward(self, x):
        if self.sign == -1:
            return (x - self.mean_shift) / self.scale_factor
        else:
            return x * self.scale_factor + self.mean_shift


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
        path_1 = self.fp_depthwise_conv(path_1)
        return self.act(path_0 + path_1) + x


class ResidualBlock(nn.Module):
    def __init__(self, n_features, group_size):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            ConvBlock(n_features, group_size),
            ConvBlock(n_features, group_size),
        )

    def forward(self, x):
        return self.res_block(x)


class BinaryScalex4BaselineV1(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, feature_channels=64, n_residual_blocks=5, group_size=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.transform = Normalize(sign=-1)
        self.inverse_transform = Normalize(sign=1)

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=9, stride=1, padding=4), nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(feature_channels, group_size))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_channels),
        )

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(feature_channels, feature_channels * 4, 3, 1, 1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(feature_channels, out_channels, kernel_size=9, stride=1, padding=4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
