import torch
from torch import Tensor, nn
from src.modules.training.models.transformer import PositionalEncoding, Conv1dBnRelu
from transformers import MambaConfig, MambaModel


class Net(nn.Module):
    """Mamba encoder network with convolutional layer."""

    def __init__(
            self,
            n_classes: int,
            num_embeddings: int = 64,
            hidden_dim: int = 16,
            vocab_size: int = 41,
    ) -> None:
        """Initialize Net for using a mamba encoder.

        :param n_classes: Number of classes
        :param num_embeddings: Number of embeddings for transformer
        :param heads: Number of heads of encoder block
        :param hidden_dim: Hidden dimension
        :param vocab_size: Vocabulary size
        :param max_len_pos_enc: Max length of positional encoding
        :param dropout: Dropout rate
        """
        super().__init__()

        # Initializing a Mamba configuration with random weights
        configuration = MambaConfig(vocab_size=vocab_size, hidden_size=num_embeddings, state_size=hidden_dim, num_hidden_layers=2)
        self.mamba_model = MambaModel(configuration)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.norm_f = nn.LayerNorm(hidden_dim, eps=1e-4)

        self.bind = nn.Sequential(
            nn.Linear(num_embeddings, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of net.

        :param x: Input tensor
        :param: Resulting tensor
        """

        x = self.mamba_model(x)
        x = x.last_hidden_state
        z = x.transpose(1, 2)

        pool = self.pool(z).squeeze(2)
        return self.bind(pool)
