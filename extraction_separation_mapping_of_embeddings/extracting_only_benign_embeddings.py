"""
    EXTRACTING ONLY BENIGN EMBEDDINGS FROM VISU TRAINING DATASET
"""

import torch

data = torch.load("C:/Users/Asus/PycharmProjects/Diffusion-Models-Embedding-Space-Defense/embeddings_cache/visu_training/03f7a6e1816195a039adf08998aa1691_all_embeddings.pt")

benign_embeddings = []

for item in data:
    emb, label = item
    if label == "benign":
        benign_embeddings.append(emb)

X_benign = torch.stack(benign_embeddings).float()
torch.save(X_benign, "benign_training_embeddings.pt")
