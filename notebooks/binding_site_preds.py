import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Path to the saved LoRA model
model_path = "AmelieSchreiber/esm2_t33_650M_qlora_binding_16M"
# ESM2 base model
base_model_path = "facebook/esm2_t33_650M_UR50D"

# Load the model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
loaded_model = PeftModel.from_pretrained(base_model, model_path)

# Ensure the model is in evaluation mode
loaded_model.eval()

# Load the tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# %%

# HKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVK

sequence_sEH = "TLRAAVFDLDGVLALPAVFGVLGRTEEALALPRGLLNDAFQKGGPEGATTRLMKGEITLSQWIPLMEENCRKCSETAKVCLPKNFSIKEIFDKAISARKINRPMLQAALMLRKKGFTTAILTNTWLDDRAERDGLAQLMCELKMHFDFLIESCQVGMVKPEPQIYKFLLDTLKASPSEVVFLDDIGANLKPARDLGMVTILVQDTDTALKELEKVTGIQLLNTPAPLPTSCNPSDMSHGYVTVKPRVRLHFVELGSGPAVCLCHGFPESWYSWRYQIPALAQAGYRVLAMDMKGYGESSAPPEIEEYCMEVLCKEMVTFLDKLGLSQAVFIGHDWGGMLVWYMALFYPERVRAVASLNTPFIPANPNMSPLESIKANPVFDYQLYFQEPGVAEAELEQNLSRTFKSLFRASDESVLSMHKVCEAGGLFVNSPEEPSLSRMVTEEEIQFYVQQFKKSGFRGPLNWYRNMERNWKWACKSLGRKILIPALMVTAEKDFVLVPQMSQHMEDWIPHLKRGHIEDCGHWTQMDKPTEVNQILIKWLDSDARNPPVVSKM"
sequence_BRD4 = "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRGRGRKETGTAKPGVSTVPNTTQASTPPQTQTPQPNPPPVQATPHPFPAVTPDLIVQTPVMTVVPPQPLQTPPPVPPQPQPPPAPAPQPVQSHPPIIAATPQPVKTKKGVKRKADTTTPTTIDPIHEPPSLPPEPKTTKLGQRRESSRPVKPPKKDVPDSQQHPAPEKSSKVSEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKMPDE"
sequence_HSA = "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"

# Tokenize the sequence
inputs_sEH = loaded_tokenizer(sequence_sEH, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
inputs_HSA = loaded_tokenizer(sequence_HSA, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
inputs_BRD4 = loaded_tokenizer(sequence_BRD4, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")

# Run the model
with torch.no_grad():
    logits_sEH = loaded_model(**inputs_sEH).logits
    logits_HSA = loaded_model(**inputs_HSA).logits
    logits_BRD4 = loaded_model(**inputs_BRD4).logits


# Get predictions
# sEH
tokens_sEH = loaded_tokenizer.convert_ids_to_tokens(inputs_sEH["input_ids"][0])  # Convert input ids back to tokens
predictions_sEH = torch.argmax(logits_sEH, dim=2)
# HSA
tokens_HSA = loaded_tokenizer.convert_ids_to_tokens(inputs_HSA["input_ids"][0])  # Convert input ids back to tokens
predictions_HSA = torch.argmax(logits_HSA, dim=2)
# BRD4
tokens_BRD4 = loaded_tokenizer.convert_ids_to_tokens(inputs_BRD4["input_ids"][0])  # Convert input ids back to tokens
predictions_BRD4 = torch.argmax(logits_BRD4, dim=2)

# Define labels
id2label = {
    0: "No binding site",
    1: "Binding site",
}


# %%
def group_adjacent_binding_sites(tokens, predictions, id2label):
    binding_sites = set()
    current_site = []

    for token, prediction in zip(tokens, predictions, strict=False):
        if token not in ["<pad>", "<cls>", "<eos>"] and prediction == 1:
            if current_site:
                # Check if the current token is adjacent to the last token in the current_site
                if ord(token) == ord(current_site[-1][0]) + 1:
                    current_site.append((token, id2label[prediction]))
                else:
                    binding_sites.add(tuple(current_site))
                    current_site = [(token, id2label[prediction])]
            else:
                current_site.append((token, id2label[prediction]))
        elif current_site:
            binding_sites.add(tuple(current_site))
            current_site = []

    # Add the last group if not empty
    if current_site:
        binding_sites.add(tuple(current_site))

    return binding_sites


# Example usage with the protein sequences
preds_sEH = group_adjacent_binding_sites(tokens_sEH, predictions_sEH[0].numpy(), id2label)
preds_HSA = group_adjacent_binding_sites(tokens_HSA, predictions_HSA[0].numpy(), id2label)
preds_BRD4 = group_adjacent_binding_sites(tokens_BRD4, predictions_BRD4[0].numpy(), id2label)

print(len(preds_sEH), len(preds_HSA), len(preds_BRD4))

# Calculate intersection
intersection_sEH_HSA = preds_sEH.intersection(preds_HSA)
for binding_site in intersection_sEH_HSA:
    print(", ".join([f"{token}: {label}" for token, label in binding_site]))
print("\n")

intersection_BRD4_HSA = preds_BRD4.intersection(preds_HSA)
for binding_site in intersection_BRD4_HSA:
    print(", ".join([f"{token}: {label}" for token, label in binding_site]))
print("\n")

intersection_sEH_BRD4 = preds_sEH.intersection(preds_BRD4)
for binding_site in intersection_sEH_BRD4:
    print(", ".join([f"{token}: {label}" for token, label in binding_site]))
