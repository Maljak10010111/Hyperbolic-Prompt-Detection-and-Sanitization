import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

"""
    TEST ACCURACY 
    -> test dataset is labeled as well
"""


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim = 768
model = MLPClassifier(input_dim)
model.load_state_dict(torch.load("MLP.pth"))
model.eval().to(DEVICE)

# original CLIP test embeddings
test_embeddings = torch.load("C:/Users/Asus/PycharmProjects/Diffusion-Models-Embedding-Space-Defense/clip_embeddings/test_clip_embeddings_visu/test_clip_embeddings.pt").to(DEVICE)

# Hyperbolic CLIP test embeddings
# test_embeddings = torch.load("../extraction_separation_mapping_of_embeddings/test_tensors.pt").to(DEVICE)

# labels
test_labels = torch.load("../../extraction_separation_mapping_of_embeddings/test_labels.pt").long().to(DEVICE)

test_dataset = TensorDataset(test_embeddings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


correct = 0
total = 0

all_probs = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        #preds = (torch.sigmoid(outputs) > 0.5).squeeze().long()
        probs = torch.sigmoid(outputs).squeeze()
        preds = (probs > 0.5).long()

        correct += (preds == y).sum().item()
        total += y.size(0)


        all_probs.extend(probs.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

"""
  ------- ROC-AUC Curve -------
"""

fpr, tpr, _ = roc_curve(all_targets, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

roc_auc_path = "C:/Users/Asus/PycharmProjects/Diffusion-Models-Embedding-Space-Defense/plots/roc_auc"
os.makedirs(roc_auc_path, exist_ok=True)
save_path = os.path.join(roc_auc_path, "original_clip_roc_auc_curve.png")

plt.savefig(save_path)
print(f"ROC curve saved to: {save_path}")