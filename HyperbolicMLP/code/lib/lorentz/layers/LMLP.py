import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math


from . import LorentzReLU, LorentzFullyConnected
from geoopt.optim import RiemannianSGD
# from geoopt.manifolds.lorentz import Lorentz

from code.lib.lorentz.manifold import CustomLorentz



def get_Activation(manifold):
    return LorentzReLU(manifold)


class LorentzMLP(nn.Module):
    """MLP in the Lorentz model"""

    def __init__(
        self,
        manifold: CustomLorentz,
        input_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_bias: bool = False,
    ):
        super(LorentzMLP, self).__init__()

        self.manifold = manifold
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.dropout = dropout

        self.mlp = nn.Sequential(
            LorentzFullyConnected(self.manifold, self.input_dim, 512, self.use_bias),
            get_Activation(manifold),
            nn.Dropout(dropout),
            LorentzFullyConnected(self.manifold, 512, 256, self.use_bias),
            get_Activation(manifold),
            nn.Dropout(dropout),
            LorentzFullyConnected(self.manifold, 256, 128, self.use_bias),
            get_Activation(manifold),
            nn.Dropout(dropout),
            LorentzFullyConnected(self.manifold, 128, 1, self.use_bias)
        )

    def forward(self, x):
        x = self.manifold.projx(x)
        x = self.mlp(x)
        return x

BATCH_SIZE = 256
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_embeddings = torch.load("C:/Users/Asus/KURCINA/Diffusion-Models-Embedding-Space-Defense/training/extraction_separation_mapping_of_embeddings/training_tensors.pt").to(DEVICE)
train_labels = torch.load("C:/Users/Asus/KURCINA/Diffusion-Models-Embedding-Space-Defense/training/extraction_separation_mapping_of_embeddings/training_labels.pt").long().to(DEVICE)
val_embeddings = torch.load("C:/Users/Asus/KURCINA/Diffusion-Models-Embedding-Space-Defense/training/extraction_separation_mapping_of_embeddings/validation_tensors.pt").to(DEVICE)
val_labels = torch.load("C:/Users/Asus/KURCINA/Diffusion-Models-Embedding-Space-Defense/training/extraction_separation_mapping_of_embeddings/validation_labels.pt").long().to(DEVICE)

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_embeddings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



input_dim = train_embeddings.shape[1]
print(input_dim)
manifold = CustomLorentz(k=2.3026)

model = LorentzMLP(manifold, input_dim, 512, dropout=0.1, use_bias=False).to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = RiemannianSGD(model.parameters(), lr=LR)


def evaluate(model, data_loader):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, val_loss / len(data_loader)


best_val_acc = 0.0


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), labels.float())
        loss.backward()

        # --- Print gradients ---
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Grad for {name}: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}")
            else:
                print(f"No grad for {name}")
        optimizer.step()
        total_loss += loss.item() # sum of per-batch averaged losses

    val_acc, val_loss = evaluate(model, val_loader)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")


    # save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "Hyperbolic_MLP.pth")
        print(f"Saved new best model at epoch {epoch + 1} with Validation Accuracy: {val_acc:.4f}")