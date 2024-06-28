"""Module to encode SMILES into different string representations."""
from dataclasses import dataclass

import deepsmiles  # type: ignore[import-not-found]
import joblib
import numpy as np
import selfies  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

converter = deepsmiles.Converter(rings=True, branches=True)


@dataclass
class StringEncoder(VerboseTransformationBlock):
    """Class that converts smiles into different representations."""

    representation: str = "selfie"
    padding_size: int = 150

    def custom_transform(self, x: XData) -> XData:
        """Convert the smiles into a different string representation.

        :param x: XData containing the molecule smiles.
        :return: XData containing the new string representation
        """

        def selfie_encoder(smile: str) -> list[str]:
            # Convert the molecule smile to selfie
            selfie = selfies.encoder(smile)
            sequence = list(selfies.split_selfies(selfie))

            # Pad the sequence with special token
            return sequence + ["PAD"] * (self.padding_size - len(sequence))

        def deep_encoder(smile: str) -> str:
            return converter.encode(smile)

        smiles = x.molecule_smiles

        if self.representation == "selfie":
            smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(selfie_encoder)(smile) for smile in tqdm(smiles, desc="Encoding SMILES"))
            x.molecule_smiles = np.stack(smiles_enc)

        if self.representation == "deep_smile":
            smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(deep_encoder)(smile) for smile in tqdm(smiles, desc="Encoding SMILES"))
            x.molecule_smiles = np.stack(smiles_enc)

        if self.representation == "smile":
            x.molecule_smiles = smiles

        return x
