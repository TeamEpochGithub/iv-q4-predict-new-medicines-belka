"""Module containing chemberta class copied from a  public notebook."""
from torch import Tensor, nn
from torchvision import models


class EfficientNet(nn.Module):
    """Pre-trained Hugging Face model for molecule tokenizers."""

    def __init__(self, n_classes: int) -> None:
        """Initialize the Hugging Face Model.

        :param model_name: the name of the Hugging Face model
        :param n_classes: the number of classes
        """
        super(EfficientNet, self).__init__()  # noqa: UP008

        # Extract the pre-trained residual network
        self.model = models.resnet50(pretrained=True)

        # Extract the essential layers for the network
        self.conv1 = self.model.conv1
        self.relu = self.model.relu
        self.max_pool = self.model.maxpool

        # Extract the first and last sequential layers
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        # Extract the last essential layers
        self.avg_pool = self.model.avgpool
        self.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)


        # Freeze the weights of the first essential layers
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.relu.parameters():
            param.requires_grad = False

        for param in self.max_pool.parameters():
            param.requires_grad = False

        # Freeze the first two sequential layers
        for param in self.layer1.parameters():
            param.requires_grad = False

        for param in self.layer2.parameters():
            param.requires_grad = False

        for param in self.layer3.parameters():
            param.requires_grad = False


    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation of the model.

        :param x: the input data
        :return: the output data
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        a = x.view(x.size(0), -1)
        return self.fc(a)
