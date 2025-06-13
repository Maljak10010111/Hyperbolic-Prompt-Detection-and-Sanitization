import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.RELU import LorentzReLU
from utils.FullyConnectedLayer import LorentzFullyConnected, LorentzFullyConnectedNoTime
from utils.LorentzManifold import LorentzManifold
from geoopt.optim import RiemannianAdam


class LorentzMLP(nn.Module):
    """MLP in the Lorentz model"""

    def __init__(
        self,
        manifold: LorentzManifold,
        input_dim: int = 768,
        hidden_dim: int = 512,
        use_bias: bool = False,
    ):
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
        x = self.mlp(x)
        return x


BATCH_SIZE = 256
EPOCHS = 40
LR = 2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/training_embeddings.pt").to(DEVICE)
train_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/training_labels.pt").long().to(DEVICE)
val_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/validation_embeddings.pt").to(DEVICE)
val_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/validation_labels.pt").long().to(DEVICE)

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_embeddings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = train_embeddings.shape[1]
hidden_dim = 512
manifold = LorentzManifold(k=2.3026, learnable=True)
model = LorentzMLP(manifold, input_dim, hidden_dim, use_bias=False).to(DEVICE)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = RiemannianAdam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


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


class EarlyStopping:
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
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


early_stopper = EarlyStopping(12, mode='max')
best_val_acc = 0.0


"""MLP TRAINING"""
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(-1), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # sum of per-batch averaged losses

    scheduler.step()
    train_loss = total_loss / len(train_loader)
    val_acc, val_loss = evaluate(model, val_loader)
    early_stopper(val_acc)
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")


    # save the model with the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "Hyperbolic_MLP_prova.pth")
        print(f"Saved new best model at epoch {epoch + 1} with Validation Accuracy: {val_acc:.4f}")