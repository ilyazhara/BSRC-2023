import torch
import torch.nn.functional as F

from bnn_competition.evaluation.utils import is_binary_tensor
from bnn_competition.exceptions import NotBinaryException
from bnn_competition.libs.utils import OUT_CHANNEL_DIM


class Checker(torch.nn.Module):
    def __init__(self, module, name, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.full_name = name
        self.module_name = module._get_name()

    def forward(self, x):
        binarized_x = self.module.input_binarizer(x)
        binarized_weight = self.module.weight_binarizer(self.module.weight)

        if not is_binary_tensor(binarized_weight, out_channel_dim=OUT_CHANNEL_DIM[self.module_name[6:]]):
            raise NotBinaryException(tensor_type="weights", module_name=self.full_name)

        if not is_binary_tensor(binarized_x):
            raise NotBinaryException(tensor_type="inputs", module_name=self.full_name)

        if self.module_name == "BinaryConv2d":
            return F.conv2d(
                binarized_x,
                binarized_weight,
                self.module.bias,
                self.module.stride,
                self.module.padding,
                self.module.dilation,
                self.module.groups,
            )
        if self.module_name == "BinaryConvTranspose2d":
            return F.conv_transpose2d(
                binarized_x,
                binarized_weight,
                self.module.bias,
                self.module.stride,
                self.module.padding,
                self.module.output_padding,
                self.module.groups,
                self.module.dilation,
            )
        if self.module_name == "BinaryLinear":
            return F.linear(binarized_x, binarized_weight, self.module.bias)
