import torch
import json
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, f1_score,
    precision_score, recall_score
)

sys.path.append(os.path.abspath(
    "C:/Users/lemalak/PyCharmProjects/Research/Diffusion-Models-Embedding-Space-Defense"
))
from utils.LorentzManifold import LorentzManifold
from LorentzMLP.lorentz_MLP import LorentzMLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
manifold = LorentzManifold(k=2.3026)
model = LorentzMLP(manifold, 768, 512, use_bias=False).to(DEVICE)
model.load_state_dict(torch.load(
    "C:/Users/lemalak/PyCharmProjects/Research/Diffusion-Models-Embedding-Space-Defense/models/Lorentz_MLP.pt"
))
model.eval()


# Evaluation function
def evaluate_and_save_results(name, embedding_path, label_tensor=None):
    print(f"\n--- Evaluating: {name} ---")

    # Load embeddings
    embeddings = torch.load(embedding_path).to(DEVICE)

    # If labels not provided, use all-ones (malicious)
    if label_tensor is None:
        label_tensor = torch.ones(embeddings.size(0)).long().to(DEVICE)

    dataset = TensorDataset(embeddings, label_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    correct = 0
    total = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).long()

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    all_probs_tensor = torch.cat(all_probs)
    all_targets_tensor = torch.cat(all_targets).long()
    all_preds_tensor = (all_probs_tensor > 0.5).long()

    accuracy = correct / total
    precision = precision_score(all_targets_tensor, all_preds_tensor, zero_division=0)
    recall = recall_score(all_targets_tensor, all_preds_tensor, zero_division=0)
    f1 = f1_score(all_targets_tensor, all_preds_tensor, zero_division=0)

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(
        all_targets_tensor.numpy(), all_probs_tensor.numpy()
    )
    pr_auc = auc(recall_curve, precision_curve)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(
        all_targets_tensor.numpy(), all_probs_tensor.numpy()
    )
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/lorentz_MLP_roc_auc_curve_{name}.png")
    plt.close()

    # Save metrics
    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/lorentz_mlp_metrics_{name}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics and ROC curve for {name}.\n")

datasets = {
        "visu": {
            "path": "../Embeddings/test_embeddings_visu.pt",
            "labels": "../Embeddings/test_labels_visu.pt"
        },
        "coco": {
            "path": "../Embeddings/test_embeddings_coco.pt",
            "labels": "../Embeddings/test_labels_coco.pt"
        },
        "i2p": {
            "path": "../Embeddings/test_embeddings_i2p.pt",
            "labels": "../Embeddings/test_labels_i2p.pt"
        }
    }


for name, info in datasets.items():
    label_tensor = (
        torch.load(info["labels"]).long().to(DEVICE) if info["labels"] else None
    )
    evaluate_and_save_results(name, info["path"], label_tensor)



