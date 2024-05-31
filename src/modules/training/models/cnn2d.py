"""Module containing CNN2D class."""
from torch import Tensor, nn


class CNN2D(nn.Module):
    """CNN2D model to classify molecule.

    :param n_classes: Number of classes to predict
    :param img_size: Image input size
    """

    def __init__(self, n_classes: int = 3, img_size: int = 224, hidden_dim: int = 32) -> None:
        """Initialize the CNN2D model.

        :param n_classes: The number of classes to predict
        :param img_size: The image input size
        """
        super(CNN2D, self).__init__()  # noqa: UP008

        self.img_size = img_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(3, self.hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dense and Dropout layers
        self.fc1 = nn.Linear(self.hidden_dim * img_size // 4 * img_size // 4, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, n_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        x = x.view(x.shape[0], -1, self.img_size, self.img_size)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.hidden_dim * self.img_size // 4 * self.img_size // 4)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)

        return self.fc3(x)
