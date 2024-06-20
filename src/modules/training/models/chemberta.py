"""Module containing chemberta class copied from a  public notebook."""
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


class Chemberta(nn.Module):
    """Pre-trained Hugging Face model for molecule smiles."""

    def __init__(self, n_classes: int, model_name: str) -> None:
        """Initialize the Hugging Face Model.

        :param model_name: the name of the Hugging Face model
        :param n_classes: the number of classes
        """
        super(Chemberta, self).__init__()  # noqa: UP008
        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes, resume_download=None)
        self.lm = AutoModel.from_pretrained(model_name, add_pooling_layer=False, resume_download=None)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation of the model.

        :param x: the input data
        :return: the output data
        """
        hidden_state = self.lm(x.long()).last_hidden_state
        return self.classifier(self.dropout(hidden_state[:, 0]))