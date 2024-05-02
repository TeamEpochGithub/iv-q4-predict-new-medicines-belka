"""CNN1D model for 1D Regression"""
import torch
from torch import nn
import torch.nn.functional as F

from src.utils.logger import logger


class CNN1D(nn.Module):
    """CNN1D model for 1D signal classification, baseline model.

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Parameters
    ----------
        n_classes: number of classes.
    """

    def __init__(self, in_channels: int, out_channels: int, n_len_seg: int, n_classes: int, verbose: bool = False) -> None:  # noqa: FBT001, FBT002
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_len_seg: The length of the segment.
        :param n_classes: The number of classes.
        :param verbose: Whether to print out the shape of the data at each step.
        """
        super(CNN1D, self).__init__()  # noqa: UP008

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=out_channels*2, out_channels=out_channels*3, kernel_size=3, stride=1, padding=0)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(out_channels*3, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(2)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x
