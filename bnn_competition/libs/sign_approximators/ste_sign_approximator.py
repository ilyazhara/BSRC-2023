import torch


class STESignApproximator(torch.nn.Module):
    """
    STE sign approximator:
    f(x) = x
    f'(x) = 1
    """

    def forward(self, x):
        # torch.sign returns {-1, 0, 1}, so apply it the second time to map 0 --> 1
        x_forward = torch.sign(torch.sign(x) + 0.1)

        return (x_forward - x).detach() + x
