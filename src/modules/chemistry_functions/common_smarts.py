"""File containing the most common reactions and molecules/groups."""
from rdkit import Chem  # type: ignore[import-not-found]

# Take a block with fmoc and attach the block to a triazine core
FMOC_TRIAZINE_REACTION_SMARTS = "[*:1]C(=O)OCC1c2ccccc2-c2ccccc21>>c1nc([*:1])ncn1"
FMOC_TRIAZINE_REACTION = Chem.rdChemReactions.ReactionFromSmarts(FMOC_TRIAZINE_REACTION_SMARTS)

# NH group with triazine core reaction
# 1 = Triazine has one preexisting attachment
# 2 = Triazine has two preexisting attachments
NH2_TRIAZINE_REACTION1_SMARTS = "[cH:1]1nc([*:3])ncn1.[NH2;!$(N-[#66]):2]>>[c:1]([*:3])1ncnc([NH:2])n1"
NH2_TRIAZINE_REACTION2_SMARTS = "[cH:1]1nc([*:3])nc([*:4])n1.[NH2;!$(N-[#66]):2]>>[c:1]([*:3])1nc([*:4])nc([NH:2])n1"
NH2_TRIAZINE_REACTION1 = Chem.rdChemReactions.ReactionFromSmarts(NH2_TRIAZINE_REACTION1_SMARTS)
NH2_TRIAZINE_REACTION2 = Chem.rdChemReactions.ReactionFromSmarts(NH2_TRIAZINE_REACTION2_SMARTS)

# Boronate and halide reaction smart
BORONATE_HALIDE_REACTION_SMARTS = "[*:1]([I,Br,Cl,F]).[*:2]B(O)(O)>>[*:1]([*:2])"
BORONATE_HALIDE_REACTION = Chem.rdChemReactions.ReactionFromSmarts(BORONATE_HALIDE_REACTION_SMARTS)

# COOH BOC and FMOC reactions
COOH_BOC_REACTION_SMARTS = "[*:1]C(=O)OC(C)(C)C.[*:2]C(=O)[O;H1]>>[*:1]C(=O)[*:2]"
COOH_FMOC_REACTION_SMARTS = "[*:1]C(=O)OCC1c2ccccc2-c2ccccc21.[*:2]C(=O)[O;H1]>>[*:1]C(=O)[*:2]"
COOH_BOC_REACTION = Chem.rdChemReactions.ReactionFromSmarts(COOH_BOC_REACTION_SMARTS)
COOH_FMOC_REACTION = Chem.rdChemReactions.ReactionFromSmarts(COOH_FMOC_REACTION_SMARTS)

# Groups
ACID_SMARTS = "C(=O)[O;H1]"
ACID = Chem.MolFromSmarts(ACID_SMARTS)
ESTER_SMARTS = "*C(=O)O*"
ESTER = Chem.MolFromSmarts(ESTER_SMARTS)
BORONATE_SMARTS = "B(O)(O)"
BORONATE = Chem.MolFromSmarts(BORONATE_SMARTS)
HALOGEN_SMARTS = "[I,Br,Cl,F]"
HALOGEN = Chem.MolFromSmarts(HALOGEN_SMARTS)

# Common molecules/substructures
FMOC_SMARTS = "O=COCC1c2ccccc2-c2ccccc21"
FMOC = Chem.MolFromSmarts(FMOC_SMARTS)
BOC_SMARTS = "C(=O)OC(C)(C)C"
BOC = Chem.MolFromSmarts(BOC_SMARTS)
TRIAZINE_SMARTS = "c1ncncn1"
TRIAZINE = Chem.MolFromSmarts(TRIAZINE_SMARTS)
FLUORIDE_ACID_SMARTS = "FC(F)(F)C(=O)[O;H1]"
FLUORIDE_ACID = Chem.MolFromSmarts(FLUORIDE_ACID_SMARTS)
DNA_SMARTS = "C(=O)N[Dy]"
DNA = Chem.MolFromSmiles(DNA_SMARTS)
