import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor
# hypll library requires python>=3.10


def lorentz_to_poincare(x):
    x0 = x[..., :1]   # time component
    x_spatial = x[..., 1:] # spatial component
    return x_spatial / (x0 + 1)


def add_time_component(x_spatial, K):
    spatial_sq = torch.sum(x_spatial**2, dim=1, keepdim=True)
    x0 = torch.sqrt(1.0 / K + spatial_sq)  # x₀ > 0, upper sheet of hyperboloid
    return torch.cat([x0, x_spatial], dim=-1)



class PoincareMLP(nn.Module):
    """MLP in the Poincare ball"""

    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int = 768,
        hidden_dim: int = 512,
        use_bias: bool = False,
    ):
        super(PoincareMLP, self).__init__()

        self.manifold = manifold
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias


        self.mlp = nn.Sequential(
            hnn.HLinear(input_dim, hidden_dim, manifold=self.manifold),
            hnn.HReLU(self.manifold),
            hnn.HLinear(hidden_dim, 256, manifold=self.manifold),
            hnn.HReLU(self.manifold),
            hnn.HLinear(256, 128, manifold=self.manifold),
            hnn.HReLU(self.manifold),
            hnn.HLinear(128, 1, manifold=self.manifold)
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        x = self.mlp(x)
        return x




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim = 768
hidden_dim = 512
manifold = PoincareBall(c=Curvature(value=2.3026, requires_grad=True))
model = PoincareMLP(manifold, input_dim, hidden_dim, use_bias=False).to(DEVICE)
model.load_state_dict(torch.load("models/Poincare_MLP.pt"))
model.eval().to(DEVICE)


# Hyperbolic CLIP test embeddings (VISU dataset)
test_embeddings = torch.load("../embeddings/separated_embeddings/custom_test_tensors.pt").to(DEVICE)

# Hyperbolic CLIP test labels (VISU dataset)
test_labels = torch.load("../embeddings/separated_embeddings/custom_test_labels.pt").long().to(DEVICE)

# adding time component to test embeddings
add_time_test = add_time_component(test_embeddings, 2.3026)

# projecting test embeddings from lorentz model to poincaré ball
test_poincare = lorentz_to_poincare(add_time_test)

test_dataset = TensorDataset(test_poincare, test_labels)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


correct = 0
total = 0

all_probs = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = ManifoldTensor(inputs, manifold=manifold)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs.tensor).squeeze()
        preds = (probs > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

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

roc_auc_path = "plots"
os.makedirs(roc_auc_path, exist_ok=True)
save_path = os.path.join(roc_auc_path, "poincare_mlp_roc_auc_curve_custom2.png")

plt.savefig(save_path)
print(f"ROC curve saved to: {save_path}")