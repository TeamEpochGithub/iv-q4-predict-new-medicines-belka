"""Module containing TransformerEncoder class."""
from torch import Tensor, nn


class TransformerEncoder(nn.Module):
    """TransformerEncoder model for encoding classification.

    :param n_classes: Number of classes to predict.
    :param embedding: Whether to use an embedding or linear layer.
    """

    _embedding: nn.Module

    def __init__(self, n_classes: int, nhead: int = 8, num_encoder_layers: int = 4, *, embedding: bool = True) -> None:
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_len_seg: The length of the segment.
        :param n_classes: The number of classes.
        :param verbose: Whether to print out the shape of the data at each step.
        """
        super(TransformerEncoder, self).__init__()  # noqa: UP008
        hidden_dim = 128
        self.hidden_dim = hidden_dim

        # Embedding layer
        if embedding:
            self._embedding = nn.Embedding(num_embeddings=37, embedding_dim=hidden_dim, padding_idx=0)
        else:
            self._embedding = nn.Linear(4, hidden_dim)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Dense and Dropout layers
        self.fc1 = nn.Linear(142, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(512, n_classes)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        emb = self._embedding(x)  # Transpose batch and sequence dimensions

        # Passing through the transformer encoder
        x = self.transformer_encoder(emb)
        x = self.pool(x).squeeze(2)  # Adapt pooling to sequence data and squeeze the sequence dimension

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)
