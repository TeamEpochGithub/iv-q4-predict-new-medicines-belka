#%%
"""Create the embeddings of the molecules using smiles2vec"""
import numpy.typing as npt

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule."""

    def compute_embedding(self, smile:str)-> npt.NDArray[np.float32]:


#%%

from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import Word2Vec
from rdkit import Chem

# load pre-trained mol2vec model
mol2vec_url = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
model = Word2Vec.load(mol2vec_url)


smile = 'C#CCCC[C@H](Nc1nc(NCc2ccc(C)cc2N2CCCC2)nc(Nc2sc(Cl)cc2C(=O)OC)n1)C(=O)N[Dy]'
molecule = Chem.MolFromSmiles(smile)

sentence = MolSentence(mol2alt_sentence(molecule, 1))

keys = set(model.wv.key_to_index)

unseen_vec = 'UNK'

embeddings = [model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence]

aa=0
for embedding in embeddings:
    if type(embedding) == str:
        aa+=1


print(aa/len(embeddings))
