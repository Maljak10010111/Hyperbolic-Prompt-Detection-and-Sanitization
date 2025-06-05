"""
    SEPARATING EMBEDDINGS AND LABELS IN DIFFERENT FILES
        -> just change the path for the desired dataset (e.g., train, test, validation) and save them under appropriate name
"""

import torch

data = torch.load("C:/Users/Asus/PycharmProjects/Diffusion-Models-Embedding-Space-Defense/embeddings_cache/visu_training/03f7a6e1816195a039adf08998aa1691_all_embeddings.pt")

embeddings = []
labels = []

for item in data:
    emb = item[0]
    label = item[1]

    embeddings.append(emb)
    labels.append(0 if label == "benign" else 1)

X = torch.stack(embeddings).float()
y = torch.tensor(labels).long()

torch.save(X, "training_tensors.pt")
torch.save(y, "training_labels.pt")

