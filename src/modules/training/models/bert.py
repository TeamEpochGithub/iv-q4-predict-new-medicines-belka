"""Module containing the CNN1D and a pre-trained BERT model."""
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification

from src.modules.training.models.cnn1d import CNN1D


class Net(nn.Module):
    """Pre-trained Bert network with convolutional layers."""

    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        hidden_dim: int,
        filters: int,
    ) -> None:
        """Initialize the pre-trained Bert network.

        :param n_classes: the number of classes
        :param vocab_size: the number of tokens
        :param hidden_dim: the dimensions of the embeddings
        """
        super().__init__()

        self.CNN1D = CNN1D(n_classes=n_classes, num_embeddings=vocab_size, hidden_dim=hidden_dim, filters=filters)
        self.BERT = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=3)

        self._embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=312, padding_idx=0)
        self.BERT.bert.embeddings.word_embeddings = self._embedding

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of model.

        :param x: Input data
        :return: Output data
        """
        y = self.CNN1D(x)
        z = self.BERT(x)
        return z["logits"] + y
