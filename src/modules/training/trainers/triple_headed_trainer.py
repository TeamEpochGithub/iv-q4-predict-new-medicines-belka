"""Module for example training block."""
import gc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.utils.tensor_functions import batch_to_device
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.training.main_trainer import MainTrainer


@dataclass
class TripleHeadedTrainer(MainTrainer):
    """Tripple headed training block."""

    similarity_criterion: nn.Module = field(default=nn.BCEWithLogitsLoss(), init=True, repr=False, compare=False)
    similarity_loss_weight: float = 0.25

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
            labels_binding_mol1 = batch_to_device(y_batch[:, :3], self.y_tensor_type, self.device)
            labels_binding_mol2 = batch_to_device(y_batch[:, 3:6], self.y_tensor_type, self.device)
            labels_similiarity = batch_to_device(y_batch[:, 6], self.y_tensor_type, self.device)

            # Forward pass
            y_pred = self.model(X_batch)
            loss = (
                self.criterion(y_pred[0], labels_binding_mol1) * (1 - self.similarity_loss_weight) / 2
                + self.criterion(y_pred[1], labels_binding_mol2) * (1 - self.similarity_loss_weight) / 2
                + self.similarity_criterion(y_pred[2], labels_similiarity) * self.similarity_loss_weight
            )

            # Backward pass
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

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
                labels_binding_mol1 = batch_to_device(y_batch[:, :3], self.y_tensor_type, self.device)
                labels_binding_mol2 = batch_to_device(y_batch[:, 3:6], self.y_tensor_type, self.device)
                labels_similiarity = batch_to_device(y_batch[:, 6], self.y_tensor_type, self.device)

                # Forward pass
                y_pred = self.model(X_batch)
                loss = (
                    self.criterion(y_pred[0], labels_binding_mol1) * (1 - self.similarity_loss_weight) / 2
                    + self.criterion(y_pred[1], labels_binding_mol2) * (1 - self.similarity_loss_weight) / 2
                    + self.criterion(y_pred[2], labels_similiarity) * self.similarity_loss_weight
                )

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the validation data")

        second_mol_smiles_array = np.array(
            [
                # Binds to BRD4
                "CC12CCC(CNc3nc(NCc4ccc(CN5CCCC5=O)cc4)nc(Nc4c(C(=O)N[Dy])ccc5ccccc45)n3)(C1)OC2",
                # 'Cn1cc(Nc2nc(Nc3ncncc3C#N)nc(N[C@H](CC(=O)N[Dy])c3cccs3)n2)ccc1=O',
                "CC(=O)c1ccc(Nc2nc(NCCS(=O)(=O)Nc3ccccc3)nc(Nc3ccc(C(=O)N[Dy])nc3)n2)c(F)c1",
                # Binds to HSA
                "O=C(N[Dy])C1c2ccccc2CN1c1nc(NCCS(=O)(=O)C2CCOCC2)nc(Nc2ncnc3[nH]cnc23)n1",
                # 'CS(=O)(=O)c1cccc(Nc2nc(Nc3cc(Cl)nc(Cl)c3[N+](=O)[O-])nc(N[C@@H](Cc3cccnc3)C(=O)N[Dy])n2)c1',
                "CCn1cc(Nc2nc(Nc3ccc4c(c3)COC4=O)nc(N[C@@H](CC(=O)N[Dy])Cc3cccs3)n2)c(C)n1",
                # Binds to sEH
                "Cc1cc2cc(CNc3nc(NCc4cccnc4OC(F)F)nc(N[C@H](Cc4cn(C)c5ccccc45)C(=O)N[Dy])n3)ccc2[nH]1",
                # 'CCOC(=O)c1ncccc1Nc1nc(Nc2cccc(-n3cncn3)c2)nc(Nc2nc3cc(C(=O)N[Dy])ccc3[nH]2)n1',
                "COC(=O)c1cncc(Nc2nc(NCC3CCCn4ccnc43)nc(Nc3c(Br)cccc3C(=O)N[Dy])n2)c1",
                # Binds to all three
                # Binds to none
                "CC(C)(C)OC(=O)n1ncc2cc(Nc3nc(NCc4nnc(-c5ccncc5)[nH]4)nc(N[C@H](Cc4ccc(Cl)cc4)C(=O)N[Dy])n3)ccc21",
                # 'O=C(N[Dy])c1cc(Nc2nc(NC[C@@H]3C[C@@H]4O[C@H]3[C@H]3C[C@H]34)nc(NCC3(N4CCOCC4)CC3)n2)ccc1[N+](=O)[O-]',
                "C#Cc1cccc(Nc2nc(Nc3nc(C(F)(F)F)c(C(=O)N[Dy])s3)nc(Nc3c(C)cc(Cl)nc3Cl)n2)c1",
            ],
        )
        loader.dataset.setup_pipeline(use_augmentations=False)  # type: ignore[attr-defined]
        second_mol_encoded_array = loader.dataset._pipeline.train(second_mol_smiles_array, None)[0][:, 0]  # type: ignore[attr-defined] # noqa: SLF001

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

        self.model.eval()
        predictions = np.empty((loader.dataset.__len__(), len(second_mol_encoded_array), 3))  # type: ignore[attr-defined]
        with torch.no_grad():
            batch_size = loader.batch_size if loader.batch_size is not None else 1
            for test_smiles_idx, second_mol_encoded in enumerate(second_mol_encoded_array):
                self.log_to_terminal(f"Predicting on the test smiles index: {test_smiles_idx}")
                for step, data in enumerate(tqdm(loader, unit="batch", disable=False)):
                    X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)
                    X_batch[:, 1] = torch.tensor(second_mol_encoded, device=self.device)

                    y_pred = self.model(X_batch)
                    predictions[step * batch_size : (step + 1) * batch_size, test_smiles_idx] = y_pred[0].cpu().numpy()
        self.log_to_terminal("Done predicting")

        # Average the predictions
        predictions = np.mean(predictions, axis=1)

        # Calculate the standard deviation of the predictions
        std_deviation_per_prediction = np.std(predictions, axis=1)
        average_std = np.mean(std_deviation_per_prediction)
        min_std = np.min(std_deviation_per_prediction)
        max_std = np.max(std_deviation_per_prediction)
        self.log_to_terminal(f"Average std: {average_std}, Min std: {min_std}, Max std: {max_std}")

        return predictions
