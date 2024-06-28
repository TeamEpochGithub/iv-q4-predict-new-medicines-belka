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


class CNN1DFingerprint(nn.Module):
    """CNN1D model which predicts bindings and the pharmacophore fingerprint (two heads).

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

    def __init__(  # noqa: C901
        self,
        # Fixed Parameters
        n_classes: int,
        n_bits: int,
        n_embeddings: int = 41,
        # Hyperparameters
        embedding_dim: int = 256,
        n_backbone_conv_layers: int = 3,
        n_backbone_conv_filters: int = 64,
        n_backbone_fc_layers: int = 0,
        n_backbone_fc_size: int = 1024,
        n_head_prediction_layers: int = 3,
        n_head_prediction_size: int = 1024,
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

        self.n_prediction_size = n_head_prediction_size
        prev_block_output_size = 1

        # Embedding layer
        self.use_embedding_layer = use_embedding_layer
        if self.use_embedding_layer:
            self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_dim, padding_idx=0)
            prev_block_output_size = embedding_dim

        # Backbone | Conv
        backbone_conv: list[nn.Module] = []
        for idx in range(n_backbone_conv_layers):
            input_size = n_backbone_conv_filters * (2**idx)
            output_size = n_backbone_conv_filters * (2 ** (idx + 1))
            if idx == 0:
                input_size = prev_block_output_size
            backbone_conv.append(_ConvBlock(in_channels=input_size, out_channels=output_size, enable_pooling=conv_enable_pooling))
            prev_block_output_size = output_size
        backbone_conv.append(nn.AdaptiveMaxPool1d(1))
        self.backbone_conv = nn.Sequential(*backbone_conv)

        # Backbone | FC
        backbone_fc: list[nn.Module] = []
        for idx in range(n_backbone_fc_layers):
            input_size = n_backbone_fc_size
            output_size = n_backbone_fc_size
            if idx == 0:
                input_size = prev_block_output_size
            backbone_fc.append(_FCBlock(in_features=input_size, out_features=output_size))
            prev_block_output_size = output_size
        self.backbone_fc = nn.Sequential(*backbone_fc)

        # Head | Predict Binding
        head_binding: list[nn.Module] = []
        for idx in range(n_head_prediction_layers):
            input_size = n_head_prediction_size
            output_size = n_head_prediction_size
            if idx == 0:
                input_size = prev_block_output_size
            if idx == n_head_prediction_layers - 1:
                output_size = n_classes
            head_binding.append(_FCBlock(in_features=input_size, out_features=output_size))
        self.head_binding = nn.Sequential(*head_binding)

        # Head | Predict Molecule Fingerprint
        head_fingerprint: list[nn.Module] = []
        for idx in range(n_head_prediction_layers):
            input_size = n_head_prediction_size
            output_size = n_head_prediction_size
            if idx == 0:
                input_size = prev_block_output_size
            if idx == n_head_prediction_layers - 1:
                output_size = n_bits
            head_fingerprint.append(_FCBlock(in_features=input_size, out_features=output_size))
        self.head_fingerprint = nn.Sequential(*head_fingerprint)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        if self.use_embedding_layer:
            x = self.embedding(x).transpose(2, 1)

        x = self.backbone_conv(x).squeeze(2)
        x = self.backbone_fc(x)

        x0 = self.head_binding(x)
        x1 = self.head_fingerprint(x)

        return x0, x1
