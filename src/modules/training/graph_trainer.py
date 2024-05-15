"""Module for example training block."""
import gc
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import DataLoader as GeometricDataLoader, Data

from src.modules.logging.logger import Logger
from src.typing.graph_dataset import GraphDataset
from src.typing.xdata import XData

from tqdm import tqdm


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
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        train_graphs = [x.molecule_graph[i] for i in train_indices]
        train_labels = torch.from_numpy(y[train_indices])

        test_graphs = [x.molecule_graph[i] for i in test_indices]
        test_labels = torch.from_numpy(y[test_indices])

        train_dataset = GraphDataset(train_graphs, train_labels)
        test_dataset = GraphDataset(test_graphs, test_labels)

        return train_dataset, test_dataset

    def create_prediction_dataset(
            self,
            x: XData,
    ) -> Dataset[tuple[Tensor, ...]] | GraphDataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        x_array = list(x.molecule_graph)
        return GraphDataset(x_array, None)

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

    def _training_loop(
            self,
            train_loader: GeometricDataLoader,
            test_loader: GeometricDataLoader,
            train_losses: list[float],
            val_losses: list[float],
            fold: int = -1,
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the testing data.
        :param test_loader: Dataloader for the training data. (can be empty)
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        """
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

        self.external_define_metric(f"Training/Train Loss{fold_no}", "epoch")
        self.external_define_metric(f"Validation/Validation Loss{fold_no}", "epoch")

        for epoch in range(self.epochs):
            # Train using train_loader
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            self.log_to_external(
                message={
                    f"Training/Train Loss{fold_no}": train_losses[-1],
                    "epoch": epoch,
                },
            )

            # Compute validation loss
            if len(test_loader) > 0:
                self.last_val_loss = self._val_one_epoch(
                    test_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
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
                            "xs": list(
                                range(epoch + 1),
                            ),  # Ensure it's a list, not a range object
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": f"Training/Loss{fold_no}",
                            "xname": "Epoch",
                        },
                    },
                )

                # Early stopping
                if self._early_stopping():
                    self.log_to_external(
                        message={f"Epochs{fold_no}": (epoch + 1) - self.patience},
                    )
                    break

            # Log the trained epochs to wandb if we finished training
            self.log_to_external(message={f"Epochs{fold_no}": epoch + 1})

    def _train_one_epoch(self, dataloader: GeometricDataLoader, epoch: int) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch",
                    desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']})")
        for batch in pbar:
            batch = batch.to(self.device)

            data = Data(x=batch.x, edge_index=batch.edge_index, batch=batch.batch).to(self.device)
            y_pred = self.model(data).squeeze(1)

            target = batch.y.to(self.device)

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

    def _predict_after_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        train_indices: list[int],
        test_indices: list[int],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Predict after training the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_dataset: The training dataset.
        :param test_dataset: The test dataset.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.

        :return: The predictions and the expected output.
        """
        match self.to_predict:
            case "all":
                concat_dataset: Dataset[Any] = self._concat_datasets(
                    train_dataset,
                    test_dataset,
                    train_indices,
                    test_indices,
                )
                pred_dataloader = GeometricDataLoader(
                    concat_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                return self.predict_on_loader(pred_dataloader), y
            case "test":
                train_loader, test_loader = self.create_dataloaders(
                    train_dataset,
                    test_dataset,
                )
                return self.predict_on_loader(test_loader), y[test_indices]
            case "none":
                return x, y
            case _:
                raise ValueError("to_predict should be either 'test', 'all' or 'none")

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

    def _val_one_epoch(self, dataloader: GeometricDataLoader, desc: str) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the testing data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                x = batch.x
                edge_index = batch.edge_index
                target = batch.y
                batch_batch = batch.batch

                data = Data(x=x, edge_index=edge_index, batch=batch_batch).to(self.device)
                y_pred = self.model(data).squeeze(1)

                if target.shape != y_pred.shape:
                    target = target.view(y_pred.shape)

                loss = self.criterion(y_pred, target.float())
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict_on_loader(self, dataloader: GeometricDataLoader) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        pbar = tqdm(dataloader, unit="batch")

        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                x = batch.x
                edge_index = batch.edge_index
                batch_batch = batch.batch

                data = Data(x=x, edge_index=edge_index, batch=batch_batch).to(self.device)
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
