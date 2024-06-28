"""Module for example training block."""
import gc
from dataclasses import dataclass

import torch
from epochalyst.pipeline.model.training.utils.tensor_functions import batch_to_device
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.training.main_trainer import MainTrainer


@dataclass
class MixedPrecisionTrainer(MainTrainer):
    """Main training block."""

    def __post_init__(self) -> None:
        """Initialize the class."""
        super().__post_init__()
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
        )
        for batch in pbar:
            X_batch, y_batch = batch

            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            # Forward pass
            with torch.cuda.amp.autocast():
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.initialized_optimizer)
            self.scaler.update()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)
