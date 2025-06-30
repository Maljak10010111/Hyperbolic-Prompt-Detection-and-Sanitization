# get a dataloader of the embedding space dataset

import os
import torch
from torch.utils.data import DataLoader
from LMLR import LorentzMLR  # Assuming LorentzMLR is defined in LorentzMLR.py


embedding_folders = (
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/visu"
)
# get the file that ends like all_embeddings.pt
embedding_files = [
    f for f in os.listdir(embedding_folders) if f.endswith("all_embeddings.pt")
][0]

# load the embeddings
if os.path.exists(embedding_folders + "/" + embedding_files):
    train_data = torch.load(embedding_folders + "/" + embedding_files)
else:
    raise FileNotFoundError(
        f"File {embedding_folders + '/' + embedding_files} does not exist."
    )

validation_folder = '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/extraction_separation_mapping_of_embeddings/embeddings/validation_visu_embeddings'
test_folder = '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/extraction_separation_mapping_of_embeddings/embeddings/test_visu_embeddings'

validation_files = [
    f for f in os.listdir(validation_folder) if f.endswith("_embeddings.pt")
]
print(f"Validation files found: {validation_files}")
test_files = [
    f for f in os.listdir(test_folder) if f.endswith("_embeddings.pt")
]
if os.path.exists(validation_folder + "/" + validation_files[0]):
    val_data = torch.load(validation_folder + "/" + validation_files[0])
else:    raise FileNotFoundError(
        f"File {validation_folder + '/' + validation_files[0]} does not exist."
    )
if os.path.exists(test_folder + "/" + test_files[0]):
    test_data = torch.load(test_folder + "/" + test_files[0])
else:
    raise FileNotFoundError(
        f"File {test_folder + '/' + test_files[0]} does not exist."
    )


# create a dataset
class HyperbolicEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, k=2.3026):
        self.embeddings = embeddings
        self.final_embeddings = []
        self.k = k
        # add time dimension to the hyperbolic coordinates
        for i in range(len(self.embeddings)):
            embedding, label = self.embeddings[i]
            time_component = torch.tensor(
                [torch.sqrt(1 / self.k + embedding.norm() ** 2)], dtype=torch.float32
            )
            embedding = torch.cat((time_component, embedding), dim=0)

            self.final_embeddings.append((embedding, label))
        # convert to tensor
        self.final_embeddings = [
            (torch.tensor(embedding, dtype=torch.float32), label)
            for embedding, label in self.final_embeddings
        ]

    def __len__(self):
        return len(self.final_embeddings)

    def __getitem__(self, idx):
        return (
            self.final_embeddings[idx][0],
            self.final_embeddings[idx][1],
        )  # Assuming each item is a tuple (embedding, label)


# create a dataloader
from sklearn.model_selection import train_test_split

# Prepare your data indices and labels for stratification
train_labels = [1 if item[1] == "malicious" else 0 for item in train_data]
val_labels = [1 if item[1] == "malicious" else 0 for item in val_data]
test_labels = [1 if item[1] == "malicious" else 0 for item in test_data]

train_dataset = HyperbolicEmbeddingDataset(train_data)
val_dataset = HyperbolicEmbeddingDataset(val_data)
test_dataset = HyperbolicEmbeddingDataset(test_data)

from collections import Counter
print("Train class counts:", Counter(train_labels))
print("Val class counts:", Counter(val_labels))
print("Test class counts:", Counter(test_labels))

from geoopt.manifolds.lorentz import Lorentz
import geoopt
import torch.nn.functional as F
import torch.nn as nn
import math


manifold = Lorentz(k=2.3026)  # Lorentz manifold with curvature k
model = LorentzMLR(manifold=manifold, num_features=769, num_classes=1).to('cuda')


training_epochs = 100
optimizer = geoopt.optim.RiemannianSGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4,
    momentum=0.9,
    nesterov=True,
)
criterion = nn.BCEWithLogitsLoss()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def logging_function(epoch, loss, accuracy):
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    # write to a log file or any other logging mechanism if needed
    datetime = __import__('datetime')
    import os
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = f"training_log_{current_time}.txt"
    if not os.path.exists(log_path):
        with open(log_path, "w") as log_file:
            log_file.write("Training Log\n")
            log_file.write("============\n")
    with open(log_path, "a") as log_file:
        log_file.write(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")

# Training loop
for epoch in range(training_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        inputs, labels = batch
        labels = list(labels)
        labels = torch.tensor([int(label == "malicious") for label in labels], dtype=torch.float32)
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('outputs:', outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{training_epochs}], Loss: {avg_loss:.4f}")

    # Validation
    if (epoch + 1) % 2 == 0:  # Validate every 2 epochs
        model.eval()
        val_loss = 0.0
        avg_accuracy = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch
                labels = [int(label == "malicious") for label in labels]
                labels = torch.tensor(labels, dtype=torch.float32).to('cuda')
                inputs = inputs.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
                avg_accuracy += accuracy
                val_loss += loss.item() 
        avg_val_loss = val_loss / len(val_dataloader)
        avg_accuracy /= len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_accuracy:.4f}")


# test the model 
model.eval()
total_test_loss = 0.0

avg_test_accuracy = 0.0


with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        labels = [int(label == "malicious") for label in labels]
        labels = torch.tensor(labels, dtype=torch.float32).to('cuda')
        inputs = inputs.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Calculate accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        avg_test_accuracy += accuracy
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy /= len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")

    #create a ROC curve and AUC score
from sklearn.metrics import roc_curve, auc
import numpy as np
model.eval()

all_labels = []
all_scores = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        labels = torch.tensor([int(label == "malicious") for label in labels], dtype=torch.float32)
        inputs = inputs.to('cuda')
        outputs = model(inputs)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(torch.sigmoid(outputs).cpu().numpy())  # NOT thresholded!

from sklearn.metrics import roc_curve, auc
import numpy as np

print("Collected labels:", set(all_labels))
print("Collected scores min/max:", min(all_scores), max(all_scores))

fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")
# Plot ROC curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig("roc_curve.png")
# save the model
torch.save(model.state_dict(), "hyperbolic_mlr_model.pth")