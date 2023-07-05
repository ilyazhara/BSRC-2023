import torch

EPS = 1e-8


def is_binary_tensor(x, out_channel_dim=None):
    """
    Checks if tensor x is binary or not.
    x is binary iff x has 2 unique values in each output channel that are equal by absolute value.
    """
    x_abs = x.abs()
    if out_channel_dim is not None:
        dim = [d for d in range(len(x.shape)) if d != out_channel_dim]
        x_min = torch.amin(x, dim=dim, keepdim=True)
        x_max = torch.amax(x, dim=dim, keepdim=True)
        x_abs_max = torch.amax(x_abs, dim=dim, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
        x_abs_max = x_abs.max()

    absolute_difference = (x_abs_max - x_abs).max().item()
    relative_difference = (torch.minimum(x - x_min, x_max - x) / (x_max - x_min)).max().item()
    return absolute_difference < EPS and relative_difference < EPS
