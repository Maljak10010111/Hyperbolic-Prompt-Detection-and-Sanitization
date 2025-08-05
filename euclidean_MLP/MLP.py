import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings/separated_embeddings/training_embeddings.pt").to(DEVICE)
train_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings/separated_embeddings/training_labels.pt").long().to(DEVICE)
val_embeddings = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings/separated_embeddings/validation_embeddings.pt").to(DEVICE)
val_labels = torch.load("C:/Users/lemalak/PyCharm Projects/Research/Diffusion-Models-Embedding-Space-Defense/embeddings/separated_embeddings/validation_labels.pt").long().to(DEVICE)

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(val_embeddings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


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

if __name__ == "__main__":
    input_dim = train_embeddings.shape[1]
    model = MLPClassifier(input_dim).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)


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
            optimizer.step()
            total_loss += loss.item() # sum of per-batch averaged losses

        val_acc, val_loss = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")


        # save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "MLP.pth")
            print(f"Saved new best model at epoch {epoch + 1} with Validation Accuracy: {val_acc:.4f}")

