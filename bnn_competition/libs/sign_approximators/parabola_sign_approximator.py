import torch


class ParabolaSignApproximator(torch.nn.Module):
    """
    Parabola sign approximation:
    f(x) =
        -1,         x < -1
        x^2 + 2*x, -1 <= x < 0;
        -x^2 + 2*x, 0 <= x < 1;
        1,          x >= 1
    f'(x) =
        2*x + 2,   -1 <= x < 0;
        -2*x + 2,   0 <= x < 1;
        0,          otherwise
    """

    def forward(self, x):
        # torch.sign returns {-1, 0, 1}, so apply it the second time to map 0 --> 1
        x_forward = torch.sign(torch.sign(x) + 0.1)
        x_clamped = torch.clamp(x, -1, 1)
        x_backward = torch.where(x_clamped < 0.0, x_clamped ** 2 + 2 * x_clamped, -(x_clamped ** 2) + 2 * x_clamped)

        return (x_forward - x_backward).detach() + x_backward
