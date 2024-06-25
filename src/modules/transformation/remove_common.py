"""Module to encode SMILES into different string representations."""
from dataclasses import dataclass

import numpy as np
from rdkit import Chem  # type: ignore[import-not-found]

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


@dataclass
class RemoveCommon(VerboseTransformationBlock):
    """Class that removes the common subtructure in the first block."""

    def custom_transform(self, x: XData) -> XData:
        """Remove the common substructure in the first block.

        :param x: XData containing the first building block
        :return x: XData containing the new building blocks
        """
        # Compute the molecules of the common substructures
        common1 = Chem.MolFromSmiles("O=COCC1c2ccccc2-c2ccccc21")
        common2 = Chem.MolFromSmiles("CC(C)(C)OC(=O)")

        if x.bb1_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        count_common = 0
        new_smiles = []
        for smile in x.bb1_smiles:
            molecule = Chem.MolFromSmiles(smile)
            if molecule.HasSubstructMatch(common1):
                modified = Chem.rdmolops.DeleteSubstructs(molecule, common1, onlyFrags=False)
                new_smiles.append(Chem.MolToSmiles(modified))
                count_common += 1

            if molecule.HasSubstructMatch(common2):
                modified = Chem.rdmolops.DeleteSubstructs(molecule, common2, onlyFrags=False)
                new_smiles.append(Chem.MolToSmiles(modified))
                count_common += 1

            else:
                new_smiles.append(smile)

        # Print the percentage of molecules with the common structure
        self.log_to_terminal(f"The percentage of molecules with common structure {count_common/len(new_smiles)}.")

        x.bb1_smiles = np.array(new_smiles)
        return x
