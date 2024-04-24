import torch
import esm

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

model.eval()

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
if torch.cuda.is_available():
    model = model.cuda()


sequence_sEH = "TLRAAVFDLDGVLALPAVFGVLGRTEEALALPRGLLNDAFQKGGPEGATTRLMKGEITLSQWIPLMEENCRKCSETAKVCLPKNFSIKEIFDKAISARKINRPMLQAALMLRKKGFTTAILTNTWLDDRAERDGLAQLMCELKMHFDFLIESCQVGMVKPEPQIYKFLLDTLKASPSEVVFLDDIGANLKPARDLGMVTILVQDTDTALKELEKVTGIQLLNTPAPLPTSCNPSDMSHGYVTVKPRVRLHFVELGSGPAVCLCHGFPESWYSWRYQIPALAQAGYRVLAMDMKGYGESSAPPEIEEYCMEVLCKEMVTFLDKLGLSQAVFIGHDWGGMLVWYMALFYPERVRAVASLNTPFIPANPNMSPLESIKANPVFDYQLYFQEPGVAEAELEQNLSRTFKSLFRASDESVLSMHKVCEAGGLFVNSPEEPSLSRMVTEEEIQFYVQQFKKSGFRGPLNWYRNMERNWKWACKSLGRKILIPALMVTAEKDFVLVPQMSQHMEDWIPHLKRGHIEDCGHWTQMDKPTEVNQILIKWLDSDARNPPVVSKM"
sequence_BRD4 = "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRGRGRKETGTAKPGVSTVPNTTQASTPPQTQTPQPNPPPVQATPHPFPAVTPDLIVQTPVMTVVPPQPLQTPPPVPPQPQPPPAPAPQPVQSHPPIIAATPQPVKTKKGVKRKADTTTPTTIDPIHEPPSLPPEPKTTKLGQRRESSRPVKPPKKDVPDSQQHPAPEKSSKVSEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKMPDE"
sequence_HSA = "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
def generate_embedding(sequence):
    # Convert sequence to a batch format
    batch_labels, batch_strs, batch_tokens = alphabet.get_batch_converter()([(0, sequence)])
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    # Generate embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_embeddings = results['representations'][33]

    # Mean pooling across the token embeddings
    sequence_embedding = token_embeddings.mean(dim=1).cpu().numpy()
    return sequence_embedding

# Generate embeddings
embedding_sEH = generate_embedding(sequence_sEH)
embedding_BRD4 = generate_embedding(sequence_BRD4)
embedding_HSA = generate_embedding(sequence_HSA)

similarity_sEH_BRD4 = cosine_similarity(embedding_sEH.reshape(1, -1), embedding_BRD4.reshape(1, -1))[0][0]
similarity_sEH_HSA = cosine_similarity(embedding_sEH.reshape(1, -1), embedding_HSA.reshape(1, -1))[0][0]
similarity_BRD4_HSA = cosine_similarity(embedding_BRD4.reshape(1, -1), embedding_HSA.reshape(1, -1))[0][0]

print(f"Similarity between sEH and BRD4: {similarity_sEH_BRD4}")
print(f"Similarity between sEH and HSA: {similarity_sEH_HSA}")
print(f"Similarity between BRD4 and HSA: {similarity_BRD4_HSA}")

# %%
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from Bio.Align import substitution_matrices
matrix = substitution_matrices.load("BLOSUM62")
alignment_blosum_seh_hsa = pairwise2.align.globalds(sequence_sEH, sequence_HSA, matrix, -10, -0.5)
alignment_blosum_seh_br4 = pairwise2.align.globalds(sequence_BRD4, sequence_sEH, matrix, -10, -0.5)
alignment_blosum_hsa_br4 = pairwise2.align.globalds(sequence_BRD4, sequence_HSA, matrix, -10, -0.5)

print("BLOSUM62 Score:", alignment_blosum_seh_hsa[0].score)
print("Proportional Score (BLOSUM62) HSA to sEH:", alignment_blosum_seh_hsa[0].score / min(len(sequence_HSA), len(sequence_sEH)))

print("n/BLOSUM62 Score:", alignment_blosum_seh_br4[0].score)
print("Proportional Score (BLOSUM62) sEH to BRD4:", alignment_blosum_seh_br4[0].score / min(len(sequence_BRD4), len(sequence_sEH)))

print("n/BLOSUM62 Score:", alignment_blosum_hsa_br4[0].score)
print("Proportional Score (BLOSUM62) HSA to BRD4:", alignment_blosum_hsa_br4[0].score / min(len(sequence_BRD4), len(sequence_HSA)))

