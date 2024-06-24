"""Module containing chemberta class copied from a  public notebook."""
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


class Chemberta(nn.Module):
    """Pre-trained Hugging Face model for molecule tokenizers."""

    def __init__(self, n_classes: int, model_name: str) -> None:
        """Initialize the Hugging Face Model.

        :param model_name: the name of the Hugging Face model
        :param n_classes: the number of classes
        """
        super(Chemberta, self).__init__()  # noqa: UP008
        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes, resume_download=None)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, resume_download=None)

        self.embedding = self.model.embeddings
        self.roberta_1 = self.model.encoder.layer[0]
        self.roberta_2 = self.model.encoder.layer[1]
        self.roberta_3 = self.model.encoder.layer[2]

        # # Freeze the parameters in the embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False

        # Freeze the parameters in the first two roberta
        for param in self.roberta_1.parameters():
            param.requires_grad = False

        # Freeze the parameters in the first two roberta
        for param in self.roberta_2.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation of the model.

        :param x: the input data
        :return: the output data
        """
        x = self.embedding(x.long())
        x = self.roberta_1(x)
        x = self.roberta_2(x[0])
        x = self.roberta_3(x[0])

        x = self.dropout(x[0][:, 0])

        return self.classifier(x)
