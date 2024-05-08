"""Implementation of Focal Loss."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data."""

    alpha: float = 0.25
    gamma: float = 2.0
    reduction: str = "mean"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: Predictions from the model after sigmoid activation (probabilities between 0 and 1)
        :param targets: Ground truth labels
        """
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = 3*loss.mean()
        return loss
