from functools import reduce
from operator import mul


def evaluate_binary_macs(module, output_shape):
    if module._get_name() in ["BinaryConv2d", "BinaryConvTranspose2d"]:
        return module.weight.numel() * reduce(mul, output_shape[-2:]) / module.groups
    if module._get_name() == "BinaryLinear":
        return module.weight.numel()
    else:
        raise ValueError(f"Please, complement `evaluate_binary_macs` function with {module._get_name()} layer")
