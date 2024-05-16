"""Module for example training block."""
import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from epochalyst.pipeline.model.training.utils.tensor_functions import batch_to_device
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.typing.xdata import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""

    representation_to_consider: str = "ECFP"

    def create_datasets(
        self,
        x: XData,
        y: npt.NDArray[np.int8],
        train_indices: list[int],
        test_indices: list[int],
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        if self.representation_to_consider == "ECFP":
            x_array = np.array(x.molecule_ecfp)
        else:
            raise ValueError("Representation does not exist")

        train_dataset = TensorDataset(
            torch.from_numpy(x_array[train_indices]),
            torch.from_numpy(y[train_indices]),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_array[test_indices]),
            torch.from_numpy(y[test_indices]),
        )

        return train_dataset, test_dataset

    def create_prediction_dataset(
        self,
        x: XData,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        x_arr = np.array(x.molecule_ecfp)
        return TensorDataset(torch.from_numpy(x_arr).int() if self.int_type else torch.from_numpy(x_arr).float())

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train the model.

        Overwritten to intercept the fold number and enable two-stage training.

        :param x: The input data.
        :param y: The target variable.
        :return The predictions and the labels.
        """
        self._fold = train_args.get("fold", -1)
        y_pred, y = super().custom_train(x, y, **train_args)
        return y_pred.flatten(), y.flatten()

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using the model.

        :param x: Input data
        :return: predictions
        """
        return super().custom_predict(x).flatten()

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

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
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            # Forward pass
            y_pred = self.model(X_batch).squeeze(1)
            loss = self.criterion(y_pred, y_batch)

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

    # def _train_one_epoch(
    #     self,
    #     dataloader: DataLoader[tuple[Tensor, ...]],
    #     epoch: int,
    # ) -> float:
    #     """Train the model for one epoch.
    #
    #     :param dataloader: Dataloader for the training data.
    #     :param epoch: Epoch number.
    #     :return: Average loss for the epoch.
    #     """
    #     losses = []
    #     self.model.train()
    #     pbar = tqdm(
    #         dataloader,
    #         unit="batch",
    #         desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})",
    #     )
    #     for batch in pbar:
    #         X_batch, y_batch = batch
    #         X_batch = X_batch.to(self.device).int() if self.int_type else X_batch.to(self.device).float()
    #         y_batch = y_batch.to(self.device).float()
    #
    #         # Forward pass
    #         y_pred = self.model(X_batch).squeeze(1)
    #         loss = self.criterion(y_pred, y_batch)
    #
    #         # Backward pass
    #         self.initialized_optimizer.zero_grad()
    #         loss.backward()
    #         self.initialized_optimizer.step()
    #
    #         # Print tqdm
    #         losses.append(loss.item())
    #         pbar.set_postfix(loss=sum(losses) / len(losses))
    #
    #     # Step the scheduler
    #     if self.initialized_scheduler is not None:
    #         self.initialized_scheduler.step(epoch=epoch)
    #
    #     # Remove the cuda cache
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #
    #     return sum(losses) / len(losses)

    # def _val_one_epoch(
    #     self,
    #     dataloader: DataLoader[tuple[Tensor, ...]],
    #     desc: str,
    # ) -> float:
    #     """Compute validation loss of the model for one epoch.
    #
    #     :param dataloader: Dataloader for the testing data.
    #     :param desc: Description for the tqdm progress bar.
    #     :return: Average loss for the epoch.
    #     """
    #     losses = []
    #     self.model.eval()
    #     pbar = tqdm(dataloader, unit="batch")
    #     with torch.no_grad():
    #         for batch in pbar:
    #             X_batch, y_batch = batch
    #             X_batch = X_batch.to(self.device).int() if self.int_type else X_batch.to(self.device).float()
    #             y_batch = y_batch.to(self.device).float()
    #
    #             # Forward pass
    #             y_pred = self.model(X_batch).squeeze(1)
    #             loss = self.criterion(y_pred, y_batch)
    #
    #             # Print losses
    #             losses.append(loss.item())
    #             pbar.set_description(desc=desc)
    #             pbar.set_postfix(loss=sum(losses) / len(losses))
    #     return sum(losses) / len(losses)

    # def predict_on_loader(
    #     self,
    #     loader: DataLoader[tuple[Tensor, ...]],
    # ) -> npt.NDArray[np.float32]:
    #     """Predict on the loader.
    #
    #     :param loader: The loader to predict on.
    #     :return: The predictions.
    #     """
    #     self.log_to_terminal("Predicting on the test data")
    #     self.model.eval()
    #     predictions = []
    #     # Create a new dataloader from the dataset of the input dataloader with collate_fn
    #     loader = DataLoader(
    #         loader.dataset,
    #         batch_size=loader.batch_size,
    #         shuffle=False,
    #         collate_fn=(
    #             collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
    #         ),
    #     )
    #     with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
    #         for data in tepoch:
    #             X_batch = data[0].to(self.device).int() if self.int_type else data[0].to(self.device).float()
    #
    #             y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
    #             predictions.extend(y_pred)
    #
    #     self.log_to_terminal("Done predicting")
    #     return np.array(predictions)
    #


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
