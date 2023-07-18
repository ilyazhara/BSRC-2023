import torch


class Binarizer(torch.nn.Module):
    """
    Baseline binarizer.
    """

    def __init__(
        self,
        sign_approximator,
        transformation=None,
        channel_wise: bool = False,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sign_approximator = sign_approximator
        self.transformation = transformation
        self.channel_wise = channel_wise
        self.trainable = trainable

    def forward(self, x):
        if self.transformation is not None:
            x, scale = self.transformation(x)

        x_binarized = self.sign_approximator.forward(x)

        if self.transformation is not None:
            x_binarized = self.transformation.inverse(x_binarized, scale)

        return x_binarized
