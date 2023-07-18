import torch


class STEWithClipSignApproximator(torch.nn.Module):
    """
    STE sign approximator with clip
    f(x) = 1,   x > 1
         = -1,  x <= -1
         = x,   otherwise
    f'(x) = 1 if -1 < x <= 1, 0 otherwise
    """

    def forward(self, x):
        # torch.sign returns {-1, 0, 1}, so apply it the second time to map 0 --> 1
        x_forward = torch.sign(torch.sign(x) + 0.1)
        x_backward = torch.clamp(x, -1, 1)

        return (x_forward - x_backward).detach() + x_backward
