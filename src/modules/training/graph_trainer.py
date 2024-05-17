"""Module for graph training block."""
import gc
import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data  # type: ignore[import-not-found]
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.typing.graph_dataset import GraphDataset
from src.typing.xdata import XData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class GraphTrainer(TorchTrainer, Logger):
    """Graph training block."""

    int_type: bool = False

    def create_datasets(
        self,
        x: XData,
        y: npt.NDArray[np.int8],
        train_indices: list[int],
        test_indices: list[int],
    ) -> tuple[GraphDataset[Data], GraphDataset[Data]]:
        """Create datasets for graph training."""
        if x.molecule_graph is None:
            raise ValueError("x.molecule_graph cannot be None")

        train_graphs = [x.molecule_graph[i] for i in train_indices]
        train_labels = torch.from_numpy(y[train_indices])

        test_graphs = [x.molecule_graph[i] for i in test_indices]
        test_labels = torch.from_numpy(y[test_indices])

        train_dataset: GraphDataset[Data] = GraphDataset(train_graphs, train_labels, device=self.device)
        test_dataset: GraphDataset[Data] = GraphDataset(test_graphs, test_labels, device=self.device)

        return train_dataset, test_dataset

    def create_prediction_dataset(self, x: XData) -> GraphDataset[Data]:
        """Create datasets for graph prediction."""
        if x.molecule_graph is None:
            raise ValueError("x.molecule_graph cannot be None")

        x_array = list(x.molecule_graph)
        return GraphDataset(x_array, None, device=self.device)

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predicts graph prediction."""
        return super().custom_predict(x).flatten()

    def save_model_to_external(self) -> None:
        """Save model to external file."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self._model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],
    ) -> tuple[GeometricDataLoader, GeometricDataLoader]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param test_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = GeometricDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, test_loader

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
            batch_device = batch.to(self.device)
            self.log_to_terminal(f"Batch device: {batch_device.x.device}")

            data = Data(x=batch_device.x, edge_index=batch_device.edge_index, batch=batch_device.batch)
            data = data.to(self.device)  # Ensure data is on the correct device
            y_pred = self.model(data).squeeze(1)
            self.log_to_terminal(f"Prediction device: {y_pred.device}")

            target = batch_device.y.to(self.device)

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
                batch_device = batch.to(self.device)
                data = Data(x=batch_device.x, edge_index=batch_device.edge_index, batch=batch_device.batch)
                data = data.to(self.device)  # Ensure data is on the correct device
                y_pred = self.model(data).squeeze(1)

                target = batch_device.y.to(self.device)
                if target.shape != y_pred.shape:
                    target = target.view(y_pred.shape)

                loss = self.criterion(y_pred, target.float())
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict_on_loader(self, dataloader: GeometricDataLoader) -> npt.NDArray[np.float32]:
        """Predicts on data from data loader."""
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        pbar = tqdm(dataloader, unit="batch")

        with torch.no_grad():
            for batch in pbar:
                batch_device = batch.to(self.device)
                data = Data(x=batch_device.x, edge_index=batch_device.edge_index, batch=batch_device.batch)
                data = data.to(self.device)  # Ensure data is on the correct device
                y_pred = self.model(data).squeeze(1)
                predictions.extend(y_pred.cpu().numpy())

        self.log_to_terminal("Done predicting")
        return np.array(predictions)


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y
