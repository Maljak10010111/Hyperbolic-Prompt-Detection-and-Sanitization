import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianAdam
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


BATCH_SIZE = 512
EPOCHS = 100
LR = 2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_embeddings = torch.load("/embeddings/separated_embeddings/training_embeddings.pt").to(DEVICE)
train_labels = torch.load("/embeddings/separated_embeddings/training_labels.pt").long().to(DEVICE)
val_embeddings = torch.load("/embeddings/separated_embeddings/validation_embeddings.pt").to(DEVICE)
val_labels = torch.load("/embeddings/separated_embeddings/validation_labels.pt").long().to(DEVICE)


""" PROJECTING EMBEDDINGS FROM LORENTZ MODEL TO POINCARÉ BALL """
# --------------------------------------------------------------------------
# adding time component (dimension) that is missing for our HySac embeddings
add_time_train = add_time_component(train_embeddings, 2.3026)
add_time_val = add_time_component(val_embeddings, 2.3026)

# mapping embeddings from lorentz model to poincaré ball
train_poincare = lorentz_to_poincare(add_time_train)
val_poincare = lorentz_to_poincare(add_time_val)

train_dataset = TensorDataset(train_poincare, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_poincare, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ---------------------------------------------------------------------------

input_dim = train_poincare.shape[1]
hidden_dim = 512
manifold = PoincareBall(c=Curvature(value=2.3026, requires_grad=True))
model = PoincareMLP(manifold, input_dim, hidden_dim, use_bias=False).to(DEVICE)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = RiemannianAdam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)


def evaluate(model, data_loader):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = ManifoldTensor(inputs, manifold=manifold)
            outputs = model(inputs)
            loss = loss_fn(outputs.tensor.squeeze(-1), labels.float())
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs.tensor) > 0.5).squeeze().long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, val_loss / len(data_loader)


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


early_stopper = EarlyStopping(patience=25, mode='max', enabled=False)
best_val_acc = 0.0


"""TRAINING THE MLP in POINCARE BALL"""
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs = ManifoldTensor(inputs, manifold=manifold)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.tensor.squeeze(-1), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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
        torch.save(model.state_dict(), "models/Poincare_MLP_prova.pt")
        print(f"Saved new best model at epoch {epoch + 1} with Validation Accuracy: {val_acc:.4f}")