"""
    SEPARATING EMBEDDINGS AND LABELS IN DIFFERENT FILES
        -> just change the path for the desired dataset (e.g., train, test, validation) and save them under appropriate name
"""

import torch

data = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings_custom_dataset/custom_testdataset/d04e162e3c8fedd0f50d106625c28374_all_embeddings.pt")

embeddings = []
labels = []

for item in data:
    emb = item[0]
    label = item[1]

    embeddings.append(emb)
    labels.append(label)
    # labels.append(0 if label == "benign" else 1)

X = torch.stack(embeddings).float()
y = torch.tensor(labels).long()

torch.save(X, "C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings_custom_dataset/separated/custom_embeddings.pt")
torch.save(y, "C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings_custom_dataset/separated/custom_labels.pt")

