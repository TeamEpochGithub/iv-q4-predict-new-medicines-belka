"""Module containing CNN1D class copied and translated from tensorflow public notebook."""
from torch import Tensor, nn


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

    _embedding: nn.Module

    def __init__(self, n_classes: int, *, embedding: bool = True) -> None:
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_len_seg: The length of the segment.
        :param n_classes: The number of classes.
        :param verbose: Whether to print out the shape of the data at each step.
        """
        super(CNN1D, self).__init__()  # noqa: UP008
        NUM_FILTERS = 32
        hidden_dim = 128
        self.hidden_dim = hidden_dim

        # Embedding layer
        if embedding:
            self._embedding = nn.Embedding(num_embeddings=41, embedding_dim=hidden_dim, padding_idx=0)
        else:
            self._embedding = nn.Linear(4, hidden_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=NUM_FILTERS, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=NUM_FILTERS, out_channels=NUM_FILTERS * 2, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=NUM_FILTERS * 2, out_channels=NUM_FILTERS * 3, kernel_size=3, stride=1)

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Dense and Dropout layers
        self.fc1 = nn.Linear(NUM_FILTERS * 3, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(512, n_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        emb = self._embedding(x).transpose(2, 1)

        x = self.relu(self.conv1(emb))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(2)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)
