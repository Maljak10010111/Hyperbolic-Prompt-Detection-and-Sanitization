"""
    SEPARATING EMBEDDINGS AND LABELS IN DIFFERENT FILES
        -> just change the path for the desired dataset (e.g., train, test, validation) and save them under appropriate name
"""

import torch

data = torch.load("C:\\Users\\lemalak\\PyCharm Projects\\Research\\Diffusion-Models-Embedding-Space-Defense\\embeddings\\custom_2\\2d71250c95d5f635ab6a0d823c2d2bd7_all_embeddings.pt")

embeddings = []
labels = []

for item in data:
    emb = item[0]
    label = item[1]

    embeddings.append(emb)
    labels.append(0 if label == "benign" else 1)

X = torch.stack(embeddings).float()
y = torch.tensor(labels).long()

torch.save(X, "../separated_embeddings/custom2_tensors.pt")
torch.save(y, "../separated_embeddings/custom2_labels.pt")

