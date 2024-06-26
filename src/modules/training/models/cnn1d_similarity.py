"""Module containing CNN1D class copied and translated from tensorflow public notebook."""
import torch
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


class CNN1DSimilarity(nn.Module):
    """CNN1D model for multi headed predictions. Input are two molecules and output is a prediction for each molecule and a similiarity score.

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
        n_common_layers: int = 2,
        n_common_size: int = 1024,
        n_prediction_layers: int = 2,
        n_prediction_size: int = 512,
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

        self.n_prediction_size = n_prediction_size

        # Embedding layer
        self.use_embedding_layer = use_embedding_layer
        if self.use_embedding_layer:
            self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_dim, padding_idx=0)

        # Molecule Backbone - Responsible for extracting features from each molecule
        in_channels = embedding_dim if use_embedding_layer else 1
        mol_backbone: list[nn.Module] = []
        for idx in range(n_conv_layers):
            if idx == 0:
                mol_backbone.append(_ConvBlock(in_channels=in_channels, out_channels=n_conv_filters, enable_pooling=conv_enable_pooling))
            mol_backbone.append(_ConvBlock(in_channels=n_conv_filters * (2**idx), out_channels=n_conv_filters * (2 ** (idx + 1)), enable_pooling=conv_enable_pooling))
        mol_backbone.append(nn.AdaptiveMaxPool1d(1))
        self.mol_backbone = nn.Sequential(*mol_backbone)

        # Common Fully Connected layers
        common_backbone: list[nn.Module] = []
        for idx in range(n_common_layers):
            if idx == 0:
                common_backbone.append(_FCBlock(in_features=n_conv_filters * (2**n_conv_layers) * 2, out_features=n_common_size))
            common_backbone.append(_FCBlock(in_features=n_common_size, out_features=n_common_size))
        common_backbone.append(_FCBlock(in_features=n_common_size, out_features=n_prediction_size * 2))
        self.common_backbone = nn.Sequential(*common_backbone)

        # Binding Prediction, First Molecule
        prediction_head: list[nn.Module] = []
        for idx in range(n_prediction_layers):
            if idx == 0:
                prediction_head.append(_FCBlock(in_features=n_prediction_size, out_features=n_prediction_size))
            prediction_head.append(_FCBlock(in_features=n_prediction_size, out_features=n_prediction_size))
        prediction_head.append(nn.Linear(n_prediction_size, n_classes))
        self.prediction_head = nn.Sequential(*prediction_head)

        # Molecule Similarity Prediction
        similiarity_head: list[nn.Module] = []
        for idx in range(n_prediction_layers):
            if idx == 0:
                similiarity_head.append(_FCBlock(in_features=n_prediction_size * 2, out_features=n_prediction_size))
            similiarity_head.append(_FCBlock(in_features=n_prediction_size, out_features=n_prediction_size))
        similiarity_head.append(nn.Linear(n_prediction_size, 1))
        self.similiarity_head = nn.Sequential(*similiarity_head)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        if self.use_embedding_layer:
            x0 = self.embedding(x[:, 0, :]).transpose(2, 1)
            x1 = self.embedding(x[:, 1, :]).transpose(2, 1)

        x0 = self.mol_backbone(x0).squeeze()
        x1 = self.mol_backbone(x1).squeeze()

        x = torch.cat([x0, x1], dim=1)
        x = self.common_backbone(x)

        # IDEA: Pass the output of mol backbones to the prediction heads concatenated
        # with the output of the common backbone
        # x0 = self.prediction_head(torch.cat([x, x0], dim=1))
        # x1 = self.prediction_head(torch.cat([x, x1], dim=1))
        x0 = self.prediction_head(x[:, : self.n_prediction_size])
        x1 = self.prediction_head(x[:, self.n_prediction_size :])
        x2 = self.similiarity_head(x).squeeze()

        return x0, x1, x2
