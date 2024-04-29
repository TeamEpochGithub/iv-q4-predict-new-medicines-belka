#%%

"""Create the embeddings of the molecules using smiles2vec"""


from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule."""


#%%

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import load_model
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cosine

# Load pre-trained model
model = load_model('model_300dim.pkl')

# Example SMILES
smiles1 = 'CCO'
smiles2 = 'CCOCC'

# Convert SMILES to RDKit molecules
mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)

# Generate molecular sentences
sentence1 = MolSentence(mol2alt_sentence(mol1, 1))
sentence2 = MolSentence(mol2alt_sentence(mol2, 1))

# Generate embeddings
embedding1 = sentences2vec([sentence1], model, unseen='UNK')
embedding2 = sentences2vec([sentence2], model, unseen='UNK')

# Compute cosine similarity
similarity = 1 - cosine(embedding1[0], embedding2[0])

print("Cosine similarity:", similarity)
