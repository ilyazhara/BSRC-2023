import torch


class L1Scaling(torch.nn.Module):
    """
    Transformation which scales the tensor by multiplying ||W||_1 / n.
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def forward(self, x):
        if self.dim:
            scale = x.abs().mean(dim=self.dim, keepdim=True).detach()
        else:
            scale = x.abs().mean().detach()

        return x, scale

    def inverse(self, x, scale):
        return x * scale
