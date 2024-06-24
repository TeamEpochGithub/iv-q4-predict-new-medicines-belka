"""Module containing classes related to transformers."""
import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Positional Encoding module."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        """Initialize positional encoding module.

        :param d_model: Model dimension
        :param max_len: Max length of encoding
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of positional encoding.

        :param x: Input Tensor
        :return: Pos encoding with tensor
        """
        return x + self.pe[:, : x.size(1)]


class Net(nn.Module):
    """Transformer encoder network."""

    def __init__(
        self,
        n_classes: int,
        num_embeddings: int = 512,
        heads: int = 8,
        hidden_dim: int = 256,
        vocab_size: int = 41,
        max_len_pos_enc: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Net for using a transformer encoder.

        :param n_classes: Number of classes
        :param num_embeddings: Number of embeddings for transformer
        :param heads: Number of heads of encoder block
        :param hidden_dim: Hidden dimension
        :param vocab_size: Vocabulary size
        :param max_len_pos_enc: Max length of positional encoding
        :param dropout: Dropout rate
        """
        super().__init__()

        self.pe = PositionalEncoding(hidden_dim, max_len=max_len_pos_enc)
        self.embedding = nn.Embedding(vocab_size, num_embeddings, padding_idx=0)
        self.conv_embedding = nn.Sequential(
            Conv1dBnRelu(num_embeddings, hidden_dim, kernel_size=3, stride=1, padding=1, is_bn=True),
        )

        self.tx_encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.bind = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of net.

        :param x: Input tensor
        :param: Resulting tensor
        """
        x = self.embedding(x)
        x = x.permute(0, 2, 1).float()
        x = self.conv_embedding(x)
        x = x.permute(0, 2, 1).contiguous()

        x = self.pe(x)
        z = self.tx_encoder(x)
        z = z.transpose(1, 2)

        pool = self.pool(z).squeeze(2)
        return self.bind(pool)


class Conv1dBnRelu(nn.Module):
    """Conv1dBnRelu module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, *, is_bn: bool = True) -> None:
        """Initialize Conv1dBnRelu block.

        :param in_channels: Number of in channels
        :param out_channels: Number of out channels
        :param kernel_size: Number of kernels
        :param stride: Stride length
        :param padding: Padding size
        :param is_bn:
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.is_bn = is_bn
        if self.is_bn:
            # self.bn1 = partial(nn.BatchNorm1d, eps=5e-3, momentum=0.1)
            self.bn1 = nn.BatchNorm1d(out_channels, eps=5e-3, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward tensor function.

        :param x: Input tensor
        :return: Output tensor
        """
        x = self.conv1(x)
        if self.is_bn:
            x = self.bn1(x)
        return self.relu(x)


# class FlashAttentionTransformerEncoder(nn.Module):
#     def __init__(
#         self,
#         dim_model,
#         num_layers,
#         num_heads=None,
#         dim_feedforward=None,
#         dropout=0.0,
#         norm_first=False,
#         activation=F.gelu,
#         rotary_emb_dim=0,
#     ):
#         super().__init__()
#
#         try:
#             from flash_attn.bert_padding import pad_input, unpad_input
#             from flash_attn.modules.block import Block
#             from flash_attn.modules.mha import MHA
#             from flash_attn.modules.mlp import Mlp
#         except ImportError:
#             raise ImportError('Please install flash_attn from https://github.com/Dao-AILab/flash-attention')
#
#         self._pad_input = pad_input
#         self._unpad_input = unpad_input
#
#         if num_heads is None:
#             num_heads = dim_model // 64
#
#         if dim_feedforward is None:
#             dim_feedforward = dim_model * 4
#
#         if isinstance(activation, str):
#             activation = {
#                 'relu': F.relu,
#                 'gelu': F.gelu
#             }.get(activation)
#
#             if activation is None:
#                 raise ValueError(f'Unknown activation {activation}')
#
#         mixer_cls = partial(
#             MHA,
#             num_heads=num_heads,
#             use_flash_attn=True,
#             rotary_emb_dim=rotary_emb_dim
#         )
#
#         mlp_cls = partial(Mlp, hidden_features=dim_feedforward)
#
#         self.layers = nn.ModuleList([
#             Block(
#                 dim_model,
#                 mixer_cls=mixer_cls,
#                 mlp_cls=mlp_cls,
#                 resid_dropout1=dropout,
#                 resid_dropout2=dropout,
#                 prenorm=norm_first,
#             ) for _ in range(num_layers)
#         ])
#
#     def forward(self, x, src_key_padding_mask=None):
#         batch, seqlen = x.shape[:2]
#
#         if src_key_padding_mask is None:
#             for layer in self.layers:
#                 x = layer(x)
#         else:
#             x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(x, ~src_key_padding_mask)
#
#             for layer in self.layers:
#                 x = layer(x, mixer_kwargs=dict(
#                     cu_seqlens=cu_seqlens,
#                     max_seqlen=max_seqlen_in_batch
#                 ))
#
#             x = self._pad_input(x, indices, batch, seqlen)
#
#         return x
