import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
from geoopt.optim import RiemannianAdam

sys.path.append(os.path.abspath(
    "C:/Users/lemalak/PyCharmProjects/Research/Diffusion-Models-Embedding-Space-Defense"
))

from LorentzMLP.utils.RELU import LorentzReLU
from LorentzMLP.utils.FullyConnectedLayer import LorentzFullyConnected, LorentzFullyConnectedNoTime
from LorentzMLP.utils.LorentzManifold import LorentzManifold


class LorentzMLP(nn.Module):
    def __init__(self, manifold: LorentzManifold, input_dim: int = 768, hidden_dim: int = 512, use_bias: bool = False):
        super(LorentzMLP, self).__init__()
        self.manifold = manifold
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

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
        return self.mlp(x)


class EarlyStopping:
    def __init__(self, patience=10, mode='max', enabled=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.enabled = enabled

    def __call__(self, score):
        if not self.enabled:
            return
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score <= self.best_score) or \
             (self.mode == 'min' and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, val_loss / len(data_loader)


def train():
    BATCH_SIZE = 512
    EPOCHS = 100
    LR = 2e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = "C:/Users/lemalak/PyCharmProjects/Research/Diffusion-Models-Embedding-Space-Defense/Embeddings"
    train_embeddings = torch.load(f"{base_path}/training_embeddings.pt").to(DEVICE)
    train_labels = torch.load(f"{base_path}/training_labels.pt").long().to(DEVICE)
    val_embeddings = torch.load(f"{base_path}/validation_embeddings.pt").to(DEVICE)
    val_labels = torch.load(f"{base_path}/validation_labels.pt").long().to(DEVICE)

    train_loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_embeddings, val_labels), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_embeddings.shape[1]
    manifold = LorentzManifold(k=2.3026, learnable=True)
    model = LorentzMLP(manifold, input_dim=input_dim).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = RiemannianAdam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    early_stopper = EarlyStopping(patience=25, mode='max', enabled=False)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(-1), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, DEVICE)
        early_stopper(val_acc)

        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/assfa.pt")
            print(f"Saved new best model at epoch {epoch + 1} with Validation Accuracy: {val_acc:.4f}")


def main():
    train()


if __name__ == "__main__":
    main()