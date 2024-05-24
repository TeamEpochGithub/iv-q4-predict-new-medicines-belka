"""Module containing function to combine building blocks into molecule."""
from rdkit import Chem  # type: ignore[import-not-found]

from src.modules.chemistry_functions.common_smarts import (
    ACID,
    BOC,
    BORONATE,
    BORONATE_HALIDE_REACTION,
    COOH_BOC_REACTION,
    COOH_FMOC_REACTION,
    DNA,
    FLUORIDE_ACID,
    FMOC,
    FMOC_TRIAZINE_REACTION,
    HALOGEN,
    NH2_TRIAZINE_REACTION1,
    NH2_TRIAZINE_REACTION2,
    TRIAZINE,
)
from src.modules.chemistry_functions.sanitize_molecule import sanitize_molecule


def bbs_to_molecule(bb1: str, bb2: str, bb3: str) -> str:
    """Combine building blocks into larger molecule.

    :param bb1: SMILE of bb1
    :param bb2: SMILE of bb2
    :param bb3: SMILE of bb3
    :return: SMILE of molecule
    """
    BB1 = Chem.MolFromSmiles(bb1)
    BB2 = Chem.MolFromSmiles(bb2)
    BB3 = Chem.MolFromSmiles(bb3)
    result = None

    # If BB1 has FMOC and BB2 don't have BORONATE and BB3 doesn't have acid
    result = None
    bb1_fmoc_substruct_match = BB1.HasSubstructMatch(FMOC)
    bb2_boronate_substruct_match = BB2.HasSubstructMatch(BORONATE)
    bb3_cooh_substruct_match = BB3.HasSubstructMatch(ACID) and not BB3.HasSubstructMatch(FLUORIDE_ACID)

    if bb1_fmoc_substruct_match and not (bb2_boronate_substruct_match and bb3_cooh_substruct_match):
        # Use FMOC_TRIAZINE_REACTION to replace FMOC with triazine
        products = FMOC_TRIAZINE_REACTION.RunReactants([BB1])
        if len(products) == 0:
            raise ValueError(f"No products were generated from fmoc triazine reaction - bb1: {BB1}")
        result = products[0][0]
        sanitize_molecule(result)

    if result is not None and result.HasSubstructMatch(ACID):
        # Replace substructs COOH with CONHDy
        result = Chem.ReplaceSubstructs(result, ACID, DNA)[-1]
        sanitize_molecule(result)
    else:
        raise ValueError("Can't attach DNA to first building block")

    if result is not None and result.HasSubstructMatch(TRIAZINE):
        # Use NH2_TRIAZINE_REACTION1 to add BB2 to triazine
        products = NH2_TRIAZINE_REACTION1.RunReactants((result, BB2))
        if len(products) == 0:
            raise ValueError("No products were generated from nh2 triazine reaction 1")
        result = products[0][0]
        sanitize_molecule(result)

        # Use NH2_TRIAZINE_REACTION2 to add BB3 to triazine
        products = NH2_TRIAZINE_REACTION2.RunReactants((result, BB3))
        if len(products) == 0:
            raise ValueError("No products were generated from nh2 triazine reaction 2")
    elif result is not None and result.HasSubstructMatch(HALOGEN):
        result = boronate_cooh_reactions(result, BB2=BB2, BB3=BB3)
    else:
        raise ValueError("Molecule does not contain triazine or halogn group")

    if result is None:
        raise ValueError(f"No molecule was generated from bb1:{bb1}, bb2:{bb2}, bb3:{bb3}")

    return result.MolToSmiles()


def boronate_cooh_reactions(result: Chem.Mol, BB2: Chem.Mol, BB3: Chem.Mol) -> Chem.Mol:
    """Add BB2 and BB3 to result via boronate and cooh reactions.

    :param result: The molecule result so far
    :param BB2: The second building block
    :param BB3: The third building block
    :return: The resulting molecule.
    """
    # Use BORONATE_HALIDE_REACTION to attach BB2 to BB1
    products = BORONATE_HALIDE_REACTION.RunReactants((result, BB2))
    if len(products) == 0:
        raise ValueError("No products were generated from boronate halide reaction")
    result = products[0][0]
    sanitize_molecule(result)

    # Use COOH_BOC_REACTION or COOH_FMOC_REACTION to attach BB3 to BB1
    if result.HasSubstructMatch(FMOC):
        products = COOH_FMOC_REACTION.RunReactants((result, BB3))
        if len(products) == 0:
            raise ValueError("No products generated from COOH_FMOC_REACTION")
        result = products[0][0]
    elif result.HasSubstructMatch(BOC):
        products = COOH_BOC_REACTION.RunReactants((result, BB3))
        if len(products) == 0:
            raise ValueError("No products were generated from COOH_BOC_REACTION")
        result = products[0][0]
    else:
        raise ValueError("BB1 has no BOC or FMOC group for COOH reaction")
    sanitize_molecule(result)

    return result
