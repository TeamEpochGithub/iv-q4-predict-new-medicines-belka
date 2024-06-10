"""Module for example training block."""
import gc
from copy import deepcopy
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
from src.modules.training.datasets.main_dataset import MainDataset
from src.typing.xdata import XData


@dataclass
class ImageTrainer(TorchTrainer, Logger):
    """Main training block."""

    dataset: MainDataset | None = None  # type: ignore[type-arg]

    def create_datasets(
        self,
        X: XData,
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
        if self.dataset is None:
            x_array = np.array(X.molecule_smiles)
            train_dataset_old = TensorDataset(
                torch.from_numpy(x_array[train_indices]),
                torch.from_numpy(y[train_indices]),
            )
            test_dataset_old = TensorDataset(
                torch.from_numpy(x_array[test_indices]),
                torch.from_numpy(y[test_indices]),
            )
            return train_dataset_old, test_dataset_old

        train_dataset = deepcopy(self.dataset)
        train_dataset.initialize(X, y, train_indices)
        train_dataset.setup_pipeline(use_augmentations=True)

        test_dataset = deepcopy(self.dataset)
        test_dataset.initialize(X, y, test_indices)

        return train_dataset, test_dataset

    def create_prediction_dataset(
        self,
        x: XData,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        if self.dataset is None:
            x_arr = np.array(x.molecule_ecfp)
            return TensorDataset(torch.from_numpy(x_arr))

        dataset = deepcopy(self.dataset)
        dataset.initialize(x)
        return dataset

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train the model.

        Overwritten to intercept the fold number and enable two-stage training.

        :param x: The input data.
        :param y: The target variable.
        :return The predictions and the labels.
        """
        self._fold = train_args.get("fold", -1)
        y_pred, y = super().custom_train(x, y, **train_args)
        return y_pred, y

    def custom_predict(self, x: XData, **pred_args: Any) -> npt.NDArray[np.float64]:
        """Predict using the model.

        :param x: Input data
        :param pred_args: Prediction arguments
        :return: predictions
        """
        return super().custom_predict(x, **pred_args)

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
        self.scaler = torch.cuda.amp.GradScaler()
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

            # Backward pass
            with torch.cuda.amp.autocast():
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            self.initialized_optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            # self.scaler.unscale_(self.initialized_optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.initialized_optimizer)
            self.scaler.update()

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


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
