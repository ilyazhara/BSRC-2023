import torch
import torch.nn.functional as F


class BinaryConvTranspose2d(torch.nn.ConvTranspose2d):
    """
    Class for binary conv_transpose2d modules.
    """

    def __init__(
        self,
        input_binarizer,
        weight_binarizer,
        **kwargs,
    ):
        """
        Args:
            input_binarizer: binarizer for inputs.
            weight_binarizer: binarizer for weights.
        """
        super().__init__(**kwargs)

        self.input_binarizer = input_binarizer
        self.weight_binarizer = weight_binarizer

    def forward(self, x):
        binarized_x = self.input_binarizer(x)
        binarized_weight = self.weight_binarizer(self.weight)

        return F.conv_transpose2d(
            binarized_x,
            binarized_weight,
            self.bias,
            self.module.stride,
            self.module.padding,
            self.module.output_padding,
            self.module.groups,
            self.module.dilation,
        )
