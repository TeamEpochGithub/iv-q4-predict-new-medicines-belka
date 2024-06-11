"""Module for example training block."""
import gc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.utils.tensor_functions import batch_to_device
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.training.main_trainer import MainTrainer


@dataclass
class TwoHeadedTrainer(MainTrainer):
    """Two headed training block."""

    loss1_weight: float = 1.0

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
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})",
        )
        for batch in pbar:
            X_batch, y_batch = batch

            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            protein_labels = batch_to_device(y_batch[:, :3], self.y_tensor_type, self.device)
            ecfp_labels = batch_to_device(y_batch[:, 3:], self.y_tensor_type, self.device)

            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred[0], protein_labels) + self.criterion(y_pred[1], ecfp_labels) * (1 / self.loss1_weight)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Step the scheduler
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch + 1)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def _val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch

                X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
                protein_labels = batch_to_device(y_batch[:, :3], self.y_tensor_type, self.device)
                ecfp_labels = batch_to_device(y_batch[:, 3:], self.y_tensor_type, self.device)

                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred[0], protein_labels) + self.criterion(y_pred[1], ecfp_labels)

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the validation data")
        self.model.eval()
        predictions = []
        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=(
                self.collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
            ),
            **self.dataloader_args,
        )
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)

                y_pred = self.model(X_batch)

                predictions.extend(y_pred[0].cpu().numpy())

        self.log_to_terminal("Done predicting")
        return np.array(predictions)
