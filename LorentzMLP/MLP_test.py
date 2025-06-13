import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from . import (LorentzReLU, LorentzBatchNorm1d, LorentzFullyConnected)
from code.lib.lorentz.layers.LFC import  LorentzFullyConnectedNoTime
from code.lib.lorentz.manifold import CustomLorentz
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os



class LorentzMLP(nn.Module):
    """MLP in the Lorentz model"""

    def __init__(
        self,
        manifold: CustomLorentz,
        input_dim: int = 768,
        hidden_dim: int = 512,
        use_bias: bool = False,
    ):
        super(LorentzMLP, self).__init__()

        self.manifold = manifold
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self.norm = LorentzBatchNorm1d(self.manifold, self.input_dim)

        self.mlp = nn.Sequential(
            LorentzFullyConnected(self.manifold, self.input_dim, self.hidden_dim, self.use_bias),
            LorentzReLU(self.manifold),
            LorentzFullyConnected(self.manifold, self.hidden_dim, 256, self.use_bias),
            LorentzReLU(self.manifold),
            LorentzFullyConnected(self.manifold, 256, 128, self.use_bias),
            LorentzReLU(self.manifold),
            LorentzFullyConnectedNoTime(self.manifold, 128, 1, self.use_bias)
        )

    def forward(self, x):
        x = self.manifold.add_time(x)
        x = self.mlp(x)
        return x


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

manifold = CustomLorentz(k=2.3026)
model = LorentzMLP(manifold, 768, 512, use_bias=False).to(DEVICE)
model.load_state_dict(torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/HyperbolicMLP/Hyperbolic_MLP_good.pth"))
model.eval().to(DEVICE)


# Hyperbolic CLIP test embeddings
test_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/test_embeddings.pt").to(DEVICE)

# original CLIP test embeddings
#TODO

# labels
test_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/test_labels.pt").long().to(DEVICE)

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

roc_auc_path = "C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/HyperbolicMLP"
os.makedirs(roc_auc_path, exist_ok=True)
save_path = os.path.join(roc_auc_path, "roc_auc_curve.png")

plt.savefig(save_path)
print(f"ROC curve saved to: {save_path}")