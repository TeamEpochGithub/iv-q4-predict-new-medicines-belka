"""Module containing chemberta class copied from a  public notebook."""
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


class MolFormer(nn.Module):
    """Pre-trained Hugging Face model for molecule tokenizers."""

    def __init__(self, n_classes: int, model_name: str) -> None:
        """Initialize the Hugging Face Model.

        :param model_name: the name of the Hugging Face model
        :param n_classes: the number of classes
        """
        super(MolFormer, self).__init__()  # noqa: UP008
        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)

        embeddings = self.model.embeddings
        first_layers = self.model.encoder.layer[:10]

        # Freeze the parameters in the embedding layer
        for param in embeddings.parameters():
            param.requires_grad = False

        # Freeze the parameters in the first layers
        for param in first_layers.parameters():
            param.requires_grad = False

        self.model.embeddings = embeddings
        self.model.encoder.layer[:10] = first_layers

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation of the model.

        :param x: the input data
        :return: the output data
        """
        x = self.model(x.long())
        aa = 1

        x = self.dropout(x[0][:, 0])
        x = self.classifier(x)

        return x
