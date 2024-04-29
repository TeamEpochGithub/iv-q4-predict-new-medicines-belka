#%%

"""Create the embeddings of the molecules using smiles2vec"""


from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule."""


#%%

from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import Word2Vec
from rdkit import Chem

# load pre-trained mol2vec model
mol2vec_url = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
model = Word2Vec.load(mol2vec_url)

# Example SMILES
smile = 'C#CCCC[C@H](Nc1nc(NCC2CCC(SC)CC2)nc(Nc2ccc(C=C)cc2)n1)C(=O)N[Dy]'
molecule = Chem.MolFromSmiles(smile)

sentence = MolSentence(mol2alt_sentence(molecule, 1))

keys = set(model.wv.key_to_index)

unseen_vec = 'UNK'

embeddings = [model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence]

aa = [embedding for embedding in embeddings if embedding=='UNK']

print(embeddings)
