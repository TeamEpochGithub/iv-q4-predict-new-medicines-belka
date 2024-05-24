"""Module for sanitizing molecules."""
from rdkit import Chem  # type: ignore[import-not-found]


def sanitize_molecule(molecule: Chem.Mol) -> None:
    """Sanitize molecule to check if it is correct.

    :param molecule
    """
    try:
        Chem.Sanitize(molecule)
    except Chem.MolSanitizeException as e:
        raise ValueError(f"Sanitization of {Chem.MolToSmiles(molecule)} failed") from e
