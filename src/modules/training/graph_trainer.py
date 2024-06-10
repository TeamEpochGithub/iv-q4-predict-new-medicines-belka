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
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.datasets.graph_dataset import GraphDataset
from src.typing.xdata import XData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class GraphTrainer(TorchTrainer, Logger):
    """Graph training block."""

    dataset: GraphDataset | None = None

    def __post_init__(self) -> None:
        """Initialize the class."""
        super().__post_init__()

        self.collate_fn = collate_fn

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
                X_batch = data.to(self.device)

                y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
                predictions.extend(y_pred)

        self.log_to_terminal("Done predicting")
        return np.array(predictions)

    def save_model_to_external(self) -> None:
        """Save model to external file."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(self.get_model_path())
            wandb.log_artifact(model_artifact)

    def _train_one_epoch(self, dataloader: DataLoader[tuple[Tensor, ...]], epoch: int) -> float:
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})")
        for batch in pbar:
            data = batch.to(self.device)
            target = data.y
            y_pred = self.model(data)

            if target.shape != y_pred.shape:
                target = target.view(y_pred.shape)

            loss = self.criterion(y_pred, target.float())
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def _val_one_epoch(self, dataloader: DataLoader[tuple[Tensor, ...]], desc: str) -> float:
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
