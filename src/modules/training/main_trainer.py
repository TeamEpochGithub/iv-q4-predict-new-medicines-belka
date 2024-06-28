"""Module for example training block."""
import contextlib
import gc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import wandb
from epochalyst._core._pipeline._custom_data_parallel import _CustomDataParallel
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from epochalyst.pipeline.model.training.utils.tensor_functions import batch_to_device
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.objects import TrainObj, TrainPredictObj
from src.modules.training.datasets.main_dataset import MainDataset
from src.modules.training.under_sampler import UnderSampler
from src.typing.xdata import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block."""

    dataset: MainDataset | None = None  # type: ignore[type-arg]
    sample_size: int | None = None
    sample_majority: float = 0.2

    use_mixed_precision: bool = False
    use_data_parallel: bool = True
    compile_model: bool = False

    def _check_arguments(self) -> None:
        """Check the arguments."""
        # Make sure to_predict is either "validation" or "all" or "none"
        if self.to_predict not in ["validation", "all", "none"]:
            raise ValueError("to_predict should be either 'validation', 'all' or 'none'")

        # Make sure n_folds is set
        if self.n_folds == -1:
            raise ValueError(
                "Please specify the number of folds for cross validation or set n_folds to 0 for train full.",
            )

        # Make sure model_name is set and does not contain spaces
        if self.model_name is None:
            raise ValueError("self.model_name is None, please specify a model_name")
        if " " in self.model_name:
            raise ValueError("Spaces in model_name not allowed")

        # Enable data parallel if multiple GPUs are available
        if self.use_data_parallel and torch.cuda.device_count() <= 1:
            self.log_to_terminal("Multiple GPUs not available. Disabling data parallel.")
            self.use_data_parallel = False

        # Check if data parallel and compile model are enabled at the same time
        if self.use_data_parallel and self.compile_model:
            raise ValueError("Cannot use data parallel and compile model at the same time.")

    def __post_init__(self) -> None:
        """Initialize the class."""
        self._check_arguments()

        self.save_model_to_disk = True
        self.best_model_state_dict: dict[Any, Any] = {}

        # Set optimizer
        self.initialized_optimizer = self.optimizer(self.model.parameters())

        # Set scheduler
        self.initialized_scheduler: LRScheduler | None = None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_to_terminal(f"Setting device: {self.device}")

        # Compile the model
        if self.compile_model:
            self.log_to_terminal("Compiling the model")
            self.model: torch.nn.Module = torch.compile(self.model)  # type: ignore[assignment]

        # If multiple GPUs are available, distribute batch size over the GPUs
        if self.use_data_parallel:
            self.log_to_terminal(f"Using {torch.cuda.device_count()} GPUs")
            self.model = _CustomDataParallel(self.model)

        # Move the model to the device
        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

        # Check if mixed precision is available and set it up
        if self.use_mixed_precision:
            self.log_to_terminal("Using mixed precision training.")
            self.scaler = torch.GradScaler(device=self.device.type)
            torch.set_float32_matmul_precision("high")

    def custom_train_override(self, x: XData, y: npt.NDArray[np.int_], **train_args: Any) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments.
            - train_indices: The indices to train on.
            - validation_indices: The indices to validate on.
            - save_model: Whether to save the model.
            - fold: Fold number if running cv.
        :return: The input and output of the system.
        """
        train_indices = train_args.get("train_indices")
        if train_indices is None:
            raise ValueError("train_indices not provided")
        validation_indices = train_args.get("validation_indices")
        if validation_indices is None:
            raise ValueError("validation_indices not provided")
        save_model = train_args.get("save_model", True)
        self._fold = train_args.get("fold", -1)

        self.save_model_to_disk = save_model

        # Print Model Hash
        self.log_to_terminal(f"Model Hash: {self.get_hash()}")

        # Create datasets
        train_dataset, validation_dataset = self.create_datasets(
            x,
            y,
            train_indices,
            validation_indices,
        )

        # Create dataloaders
        train_loader, validation_loader = self.create_dataloaders(train_dataset, validation_dataset, train_labels=y[train_indices])

        # Check if a trained model exists
        if self._model_exists():
            self.log_to_terminal(
                f"Model exists in {self.get_model_path()}. Loading model...",
            )
            self._load_model()

            # Return the predictions
            return self._predict_after_train(
                x,
                y,
                train_dataset,
                validation_dataset,
                train_indices,
                validation_indices,
            )

        # Log the model being trained
        self.log_to_terminal(f"Training model: {self.model.__class__.__name__}")

        # Resume from checkpoint if enabled and checkpoint exists
        start_epoch = 0
        if self.checkpointing_resume_if_exists:
            saved_checkpoints = list(Path(self.trained_models_directory).glob(f"{self.get_hash()}_checkpoint_*.pt"))
            if len(saved_checkpoints) > 0:
                self.log_to_terminal("Resuming training from checkpoint")
                epochs = [int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints]
                self._load_model(saved_checkpoints[np.argmax(epochs)])
                start_epoch = max(epochs) + 1

        # Train the model
        self.log_to_terminal(f"Training model for {self.epochs} epochs{', starting at epoch ' + str(start_epoch) if start_epoch > 0 else ''}")

        train_losses: list[float] = []
        val_losses: list[float] = []

        self.lowest_val_loss = np.inf
        if len(validation_loader) == 0:
            self.log_to_warning(
                f"Doing train full, model will be trained for {self.epochs} epochs",
            )

        self._training_loop(
            train_loader,
            validation_loader,
            train_losses,
            val_losses,
            self._fold,
            start_epoch,
        )
        self.log_to_terminal(
            f"Done training the model: {self.model.__class__.__name__}",
        )

        # Revert to the best model
        if self.best_model_state_dict:
            self.log_to_terminal(
                f"Reverting to model with best validation loss {self.lowest_val_loss}",
            )
            self.model.load_state_dict(self.best_model_state_dict)

        if save_model:
            self._save_model()

        return self._predict_after_train(
            x,
            y,
            train_dataset,
            validation_dataset,
            train_indices,
            validation_indices,
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        validation_dataset: Dataset[tuple[Tensor, ...]],
        train_labels: npt.NDArray[np.int8] | None = None,
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        majority_class = [0, 0, 0]

        # Create sampler
        self.log_to_terminal(f"Sampling majority class with fraction {self.sample_majority}")
        sampler = None
        if train_labels is not None:
            sampler = UnderSampler(train_labels, majority_class, majority_fraction=self.sample_majority)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(train_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            sampler=sampler,
            **self.dataloader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(validation_dataset, "__getitems__") else None),  # type: ignore[arg-type]
            **self.dataloader_args,
        )
        return train_loader, validation_loader

    def create_datasets(
        self,
        X: XData,
        y: npt.NDArray[np.int8],
        train_indices: npt.NDArray[np.int64],
        validation_indices: npt.NDArray[np.int64],
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.
        :return: The training and validation datasets.
        """
        if self.sample_size is not None:
            train_indices = np.random.default_rng().choice(train_indices, self.sample_size, replace=False)
            validation_indices = np.random.default_rng().choice(validation_indices, self.sample_size, replace=False)

        if self.dataset is None:
            x_array = np.array(X.molecule_smiles)
            train_dataset_old = TensorDataset(
                torch.from_numpy(x_array[train_indices]),
                torch.from_numpy(y[train_indices]),
            )
            validation_dataset_old = TensorDataset(
                torch.from_numpy(x_array[validation_indices]),
                torch.from_numpy(y[validation_indices]),
            )
            return train_dataset_old, validation_dataset_old

        train_dataset = deepcopy(self.dataset)
        train_dataset.initialize(X, y, train_indices)
        train_dataset.setup_pipeline(use_augmentations=True)

        validation_dataset = deepcopy(self.dataset)
        validation_dataset.initialize(X, y, validation_indices)

        return train_dataset, validation_dataset

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

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train the model.

        :param x: The input data.
        :param y: The target variable.
        :return The predictions and the labels.
        """
        if train_predict_obj.model is not None:
            self.model = train_predict_obj.model
            self.initialized_optimizer = self.optimizer(self.model.parameters())
            if self.scheduler is not None:
                self.initialized_scheduler = self.scheduler(self.initialized_optimizer)

        y_predictions, y_labels_modified = self.custom_train_override(train_predict_obj.x_data, train_obj.y_labels_original, **train_args)
        train_predict_obj.y_predictions = y_predictions
        train_predict_obj.model = self.model
        train_obj.y_labels_modified = y_labels_modified

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj, **pred_args: Any) -> TrainPredictObj:
        """Predict using the model.

        :param x: Input data
        :param pred_args: Prediction arguments
        :return: predictions
        """
        train_predict_obj.y_predictions = super().custom_predict(train_predict_obj.x_data, **pred_args)
        return train_predict_obj

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(self.get_model_path())
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
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)
