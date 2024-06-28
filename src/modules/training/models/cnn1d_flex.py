"""Module containing CNN1D class copied and translated from tensorflow public notebook."""
from torch import Tensor, nn


class _ConvBlock(nn.Module):
    enable_pooling: bool

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, *, enable_pooling: bool = False) -> None:
        super().__init__()

        self.enable_pooling = enable_pooling

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-05)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x) if self.enable_pooling else x
        x = self.batch_norm(x)
        return self.activation(x)


class _FCBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.activation(x)
        return self.dropout(x)


class CNN1DFlex(nn.Module):
    """CNN1D model.

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Parameters
    ----------
        n_classes: number of classes.
    """

    embedding: nn.Module

    def __init__(
        self,
        # Fixed Parameters
        n_classes: int,
        n_embeddings: int = 41,
        # Hyperparameters
        embedding_dim: int = 256,
        n_conv_layers: int = 3,
        n_conv_filters: int = 32,
        n_fc_layers: int = 2,
        n_fc_size: int = 512,
        *,
        conv_enable_pooling: bool = False,
        use_embedding_layer: bool = True,
    ) -> None:
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_len_seg: The length of the segment.
        :param n_classes: The number of classes.
        :param verbose: Whether to print out the shape of the data at each step.
        """
        super().__init__()

        self.n_prediction_size = n_fc_size

        # Embedding layer
        self.use_embedding_layer = use_embedding_layer
        if self.use_embedding_layer:
            self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_dim, padding_idx=0)

        # Convolutional layers
        in_channels = embedding_dim if use_embedding_layer else 1
        conv: list[nn.Module] = []
        for idx in range(n_conv_layers):
            if idx == 0:
                conv.append(_ConvBlock(in_channels=in_channels, out_channels=n_conv_filters, enable_pooling=conv_enable_pooling))
            conv.append(_ConvBlock(in_channels=n_conv_filters * (2**idx), out_channels=n_conv_filters * (2 ** (idx + 1)), enable_pooling=conv_enable_pooling))
        conv.append(nn.AdaptiveMaxPool1d(1))
        self.conv = nn.Sequential(*conv)

        # FC layers
        fc: list[nn.Module] = []
        for idx in range(n_fc_layers):
            if idx == 0:
                fc.append(_FCBlock(in_features=n_fc_size, out_features=n_fc_size))
            fc.append(_FCBlock(in_features=n_fc_size, out_features=n_fc_size))
        fc.append(nn.Linear(n_fc_size, n_classes))
        self.fc = nn.Sequential(*fc)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        if self.use_embedding_layer:
            x = self.embedding(x).transpose(2, 1)

        return self.conv(x).squeeze()

        # x = torch.cat([x0, x1], dim=1)
        # x = self.common_backbone(x)

        # # IDEA: Pass the output of mol backbones to the prediction heads concatenated
        # # with the output of the common backbone
        # # x0 = self.prediction_head(torch.cat([x, x0], dim=1))
        # # x1 = self.prediction_head(torch.cat([x, x1], dim=1))
        # x0 = self.fc(x[:, : self.n_prediction_size])
        # x1 = self.fc(x[:, self.n_prediction_size :])
        # x2 = self.similiarity_head(x).squeeze()
