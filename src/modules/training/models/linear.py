"""Linear model for 1D Regression"""
import torch
from torch import nn


class Linear(nn.Module):
    """Linear model for tabular regression, baseline model.

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Parameters
    ----------
        n_classes: number of classes.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:  # noqa: FBT001, FBT002
        """Initialize the linear model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        """
        super(Linear, self).__init__()  # noqa: UP008

        self.lin = nn.Linear(in_channels, in_channels // 2)
        self.lin1 = nn.Linear(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.lin(x)
        x = torch.sigmoid(self.lin1(x))
        return x
