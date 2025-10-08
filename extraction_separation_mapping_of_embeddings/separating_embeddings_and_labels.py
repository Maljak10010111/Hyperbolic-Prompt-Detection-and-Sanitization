"""
    SEPARATING EMBEDDINGS AND LABELS IN DIFFERENT FILES
        -> just change the path for the desired dataset (e.g., train, test, validation) and save them under appropriate name
"""

import torch, os
base = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/token_level_embeddings/visu_train/"
# create the directory if it does not exist
if not os.path.exists(base + "separated_token_level/"):
    os.makedirs(base + "separated_token_level/")
data = torch.load(base + 'e4955a99396d31410fea177f134d4bb5_all_embeddings.pt')
embeddings = []
labels = []

for item in data:
    emb = item[0]
    label = item[1]

    embeddings.append(emb)
    labels.append(0 if label == "benign" else 1)

X = torch.stack(embeddings).float()
y = torch.tensor(labels).long()

torch.save(X, base + "separated_token_level/visu_train_token_level_embeddings.pt")
torch.save(y, base + "separated_token_level/visu_train_token_level_labels.pt")

