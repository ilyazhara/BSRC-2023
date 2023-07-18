import torch
import torch.nn.functional as F


class BinaryLinear(torch.nn.Linear):
    """
    Class for binary linear modules.
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

        return F.linear(binarized_x, binarized_weight, self.bias)
