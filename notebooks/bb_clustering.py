from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

from src.setup.setup_data import setup_train_x_data
from src.typing.xdata import XData

dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
          'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}

train = pd.read_csv('data/shrunken/train.csv', dtype = dtypes, nrows=1000000)
train_samples = train.sample(frac = 0.1)

# %%
object_data = setup_train_x_data(Path("q4-detect-medicine/data/shrunken"), train_samples)


# Generate molecules and fingerprints
mols = [Chem.MolFromSmiles(smile) for smile in object_data.molecule_smiles]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in mols]

# Calculate similarity matrix
def tanimoto_similarity(fp1, fp2):
    return np.double(AllChem.DataStructs.TanimotoSimilarity(fp1, fp2))

similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
for i in range(len(fingerprints)):
    for j in range(len(fingerprints)):
        similarity_matrix[i, j] = tanimoto_similarity(fingerprints[i], fingerprints[j])

# %%
# Create a graph
G = nx.Graph()

# Add nodes
for idx, smile in enumerate(object_data.molecule_smiles):
    G.add_node(idx, label=smile)

# Add edges based on a similarity threshold
threshold = 0.5  # Threshold can be adjusted based on desired density of edges
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i, j] > threshold:
            G.add_edge(i, j, weight=similarity_matrix[i, j])

# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes

edges = G.edges(data=True)
weights = [edata['weight'] * 5 for _, _, edata in edges]  # Adjust line thickness by similarity

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', width=weights)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title('SMILES Clustering Graph')
plt.show()