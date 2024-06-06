"""Module for graph training block."""
import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import (
    Batch,
    Data,  # type: ignore[import-not-found]
)
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.datasets.graph_dataset import GraphDataset
from src.typing.xdata import XData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class GraphTrainer(TorchTrainer, Logger):
    """Graph training block."""

    dataset: GraphDataset | None = None

    def create_datasets(
        self,
        X: XData,
        y: npt.NDArray[np.int8],
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[list[Data], list[Data]] | tuple[GraphDataset, GraphDataset]:
        """Create datasets for graph training."""
        if self.dataset is None:
            if X.molecule_graph is None:
                raise ValueError("x.molecule_graph cannot be None")

            train_graphs = []
            for i in train_indices:
                X.molecule_graph[i].y = torch.from_numpy(y[i])
                train_graphs.append(X.molecule_graph[i])

            test_graphs = []
            for i in validation_indices:
                X.molecule_graph[i].y = torch.from_numpy(y[i])
                test_graphs.append(X.molecule_graph[i])

            return train_graphs, test_graphs

        train_dataset = deepcopy(self.dataset)
        train_dataset.initialize(X, y, train_indices)
        train_dataset.setup_pipeline(use_augmentations=True)

        validation_dataset = deepcopy(self.dataset)
        validation_dataset.initialize(X, y, validation_indices)

        return train_dataset, validation_dataset

    def create_prediction_dataset(self, X: XData) -> GraphDataset | list[Data]:
        """Create datasets for graph prediction.

        :param X: The input data.
        :return: Prediction dataset.
        """
        if self.dataset is None:
            if X.molecule_graph is None:
                raise ValueError("x.molecule_graph cannot be None")
            return X.molecule_graph

        dataset = deepcopy(self.dataset)
        dataset.initialize(X)
        return dataset

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model."""
        self.log_to_terminal("Model Hash: " + self.get_hash())
        return super().custom_train(x, y, **train_args)

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predicts graph prediction."""
        return super().custom_predict(x)

    def create_dataloaders(
        self,
        train_dataset: GraphDataset,
        validation_dataset: GraphDataset,
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=(collate_fn if hasattr(train_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **self.dataloader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(collate_fn if hasattr(validation_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **self.dataloader_args,
        )
        return train_loader, validation_loader

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        loader = GeometricDataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=(
                collate_fn if hasattr(loader.dataset, "__getitems__") else None  # type: ignore[arg-type]
            ),
            **self.dataloader_args,
        )
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for batch in tepoch:
                data = batch.to(self.device)

                y_pred = self.model(data).squeeze(1).cpu().numpy()
                predictions.extend(y_pred)

        self.log_to_terminal("Done predicting")
        return np.array(predictions)

    def save_model_to_external(self) -> None:
        """Save model to external file."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def _training_loop(
        self,
        train_loader: GeometricDataLoader,
        test_loader: GeometricDataLoader,
        train_losses: list[float],
        val_losses: list[float],
        fold: int = -1,
    ) -> None:
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

        self.external_define_metric(f"Training/Train Loss{fold_no}", "epoch")
        self.external_define_metric(f"Validation/Validation Loss{fold_no}", "epoch")

        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            self.log_to_external(
                message={
                    f"Training/Train Loss{fold_no}": train_losses[-1],
                    "epoch": epoch,
                },
            )

            if len(test_loader) > 0:
                self.last_val_loss = self._val_one_epoch(
                    test_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                self.log_to_external(
                    message={
                        f"Validation/Validation Loss{fold_no}": val_losses[-1],
                        "epoch": epoch,
                    },
                )

                self.log_to_external(
                    message={
                        "type": "wandb_plot",
                        "plot_type": "line_series",
                        "data": {
                            "xs": list(range(epoch + 1)),
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": f"Training/Loss{fold_no}",
                            "xname": "Epoch",
                        },
                    },
                )

                if self._early_stopping():
                    self.log_to_external(
                        message={f"Epochs{fold_no}": (epoch + 1) - self.patience},
                    )
                    break

            self.log_to_external(message={f"Epochs{fold_no}": epoch + 1})

    def _train_one_epoch(self, dataloader: GeometricDataLoader, epoch: int) -> float:
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})")
        for batch in pbar:
            data = batch.to(self.device)

            y_pred = self.model(data)

            target = data.y

            if target.shape != y_pred.shape:
                target = target.view(y_pred.shape)

            loss = self.criterion(y_pred, target.float())
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch)

        torch.cuda.empty_cache()
        gc.collect()
        return sum(losses) / len(losses)

    def _val_one_epoch(self, dataloader: GeometricDataLoader, desc: str) -> float:
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                data = batch.to(self.device)
                y_pred = self.model(data)
                y_pred = self.model(data).squeeze(1)

                target = data.y
                if target.shape != y_pred.shape:
                    target = target.view(y_pred.shape)

                loss = self.criterion(y_pred, target.float())
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    return Batch.from_data_list(batch[0])
