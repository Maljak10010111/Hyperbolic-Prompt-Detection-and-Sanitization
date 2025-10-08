import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.LorentzManifold import LorentzManifold
import torch.nn as nn
import matplotlib.pyplot as plt
import umap
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 2.3026
BATCH_SIZE = 512

test_embeddings = torch.load("/embeddings/separated_embeddings/test_embeddings.pt").to(DEVICE)
test_labels = torch.load("/embeddings/separated_embeddings/test_labels.pt").long().to(DEVICE)

test_labels = 2 * test_labels - 1

manifold = LorentzManifold(k=K)
test_lorentz = manifold.add_time(test_embeddings)
test_tangent = manifold.logmap0(test_lorentz)

test_dataset = TensorDataset(test_tangent, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class TangentSVM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x @ self.w + self.b

input_dim = test_tangent.shape[1]
svm = TangentSVM(dim=input_dim).to(DEVICE)
svm.load_state_dict(torch.load("models/SVM.pt"))
svm.eval()


correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = svm(xb)
        preds = torch.sign(outputs)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")


""" ---- PLOTTING THE CLASSIFICATION USING 2D UMAP ------"""
with torch.no_grad():
    logits = svm(test_tangent)
    preds = torch.sign(logits).cpu()
    gt = test_labels.cpu()

embedding_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(test_tangent.cpu().numpy())

plt.figure(figsize=(10, 5))

# 1. Ground Truth Labels
plt.subplot(1, 2, 1)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=gt, cmap='coolwarm', s=8, alpha=0.8)
plt.title("Ground Truth Labels")
plt.axis("off")

# 2. SVM Predictions
plt.subplot(1, 2, 2)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=preds, cmap='coolwarm', s=8, alpha=0.8)
plt.title("SVM Predictions")
plt.axis("off")

plt.tight_layout()
plot_path = "plots"
os.makedirs(plot_path, exist_ok=True)
save_path = os.path.join(plot_path, "svm_classification_2D.png")
plt.savefig(save_path)


""" ---- PLOTTING THE CLASSIFICATION USING 3D UMAP ------"""
with torch.no_grad():
    logits = svm(test_tangent)
    preds = torch.sign(logits).cpu()
    gt = test_labels.cpu()

umap_3d = umap.UMAP(n_components=3, random_state=42)
embedding_3d = umap_3d.fit_transform(test_tangent.cpu().numpy())

fig = plt.figure(figsize=(14, 6))

# 1. Ground truth labels
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
    c=gt, cmap='coolwarm', s=10, alpha=0.8
)
ax1.set_title("Ground Truth Labels")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# 2. SVM predictions
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(
    embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
    c=preds, cmap='coolwarm', s=10, alpha=0.8
)
ax2.set_title("SVM Predictions")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plot_path = "plots"
os.makedirs(plot_path, exist_ok=True)
save_path = os.path.join(plot_path, "svm_classification_3D.png")
plt.savefig(save_path)