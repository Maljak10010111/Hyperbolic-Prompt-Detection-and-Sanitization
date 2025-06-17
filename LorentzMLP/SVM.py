import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.LorentzManifold import LorentzManifold


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 2.3026
LR = 2e-2
EPOCHS = 150
BATCH_SIZE = 512
MARGIN = 1
WEIGHT_DECAY = 1e-4


train_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/training_embeddings.pt").to(DEVICE)
train_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/training_labels.pt").long().to(DEVICE)
val_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/validation_embeddings.pt").to(DEVICE)
val_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/training/separated_embeddings/validation_labels.pt").long().to(DEVICE)


# converting labels from {0, 1} to {-1, 1} to be suitable for HINGE loss
train_labels = 2 * train_labels - 1
val_labels = 2 * val_labels - 1

manifold = LorentzManifold(k=K)

# adding missing time component to the embeddings
train_lorentz = manifold.add_time(train_embeddings)
val_lorentz = manifold.add_time(val_embeddings)

# projecting embeddings to the tangent space to be suitable for SVM training
train_tangent = manifold.logmap0(train_lorentz)
val_tangent = manifold.logmap0(val_lorentz)

train_dataset = TensorDataset(train_tangent, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


class TangentSVM(nn.Module):
    """ SVM WORKING IN TANGENT SPACE """
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x @ self.w + self.b

def hinge_loss(output, label, margin=1.0):
    return torch.mean(torch.clamp(margin - label * output, min=0))

svm = TangentSVM(train_tangent.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(svm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

best_val_acc = 0.0

""" --- SVM Training --- """
for epoch in range(EPOCHS):
    svm.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = svm(xb)
        loss = hinge_loss(logits, yb, margin=MARGIN)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    with torch.no_grad():
        svm.eval()
        val_preds = torch.sign(svm(val_tangent))
        val_acc = (val_preds == val_labels).float().mean().item()
        train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # save the model with the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(svm.state_dict(), "models/SVM.pt")
        print(f"Saved new best model at epoch {epoch + 1} with Val Acc: {val_acc:.4f}")
