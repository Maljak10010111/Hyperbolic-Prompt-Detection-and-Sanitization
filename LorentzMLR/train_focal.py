#!/usr/bin/env python3
"""
Improved Hyperbolic Embedding Training Script with Advanced Loss Functions - Save Best Only

This script trains a Lorentzian Multinomial Logistic Regression model
on hyperbolic embeddings for malicious content detection.
It includes options for BCE and Focal Loss to address false negatives.

Author: Merybria99
Date: 2025-01-15
Last Modified: 2025-07-01 07:27:40 UTC
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds.lorentz import Lorentz
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from LMLR import LorentzMLR, create_lorentz_mlr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training the hyperbolic model."""

    # Data paths
    train_embedding_folder: list = field(default_factory=list)
    val_embedding_folder: list = field(default_factory=list)
    test_embedding_folder: list = field(default_factory=list)
    print("Using default train_embedding_folder: %s", train_embedding_folder)
    
    # Model parameters
    curvature: float = 2.3026  # Positive curvature
    num_features: int = 769
    num_classes: int = 1  # Binary classification

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.01
    momentum: float = 0.9
    epochs: int = 100
    validation_frequency: int = 2
    weight_decay: float = 1e-5

    # Loss function parameters
    loss_function: str = "Focal"  # 'BCE' or 'Focal'
    pos_weight: Optional[float] = None
    label_smoothing: float = 0.0
    classification_threshold: float = 0.5  # Lower threshold to reduce false negatives

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_dir: str = "outputs"
    model_name: str = "hyperbolic_mlr_model"

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Best model saving strategy
    save_best_metric: str = "accuracy"  # "accuracy" or "loss"
    cleanup_intermediate: bool = True  # Remove non-best models

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not available, switching to CPU")
            self.device = "cpu"

        if self.curvature <= 0:
            logger.warning(
                f"Curvature {self.curvature} is not positive, setting to 2.3026"
            )
            self.curvature = 2.3026


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces the loss for well-classified examples, focusing on hard-to-classify ones.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.pos_weight is not None:
            weight = torch.ones_like(targets)
            weight[targets == 1] = self.pos_weight.item()
            F_loss = weight * F_loss

        return torch.mean(F_loss)


class HyperbolicEmbeddingDataset(Dataset):
    """Enhanced dataset for hyperbolic embeddings with time dimension."""

    def __init__(self, embeddings: List[Tuple], curvature: float = 2.3026):
        if curvature <= 0:
            raise ValueError(f"Curvature must be positive, got {curvature}")
        self.embeddings = embeddings
        self.curvature = curvature
        self.processed_embeddings = self._process_embeddings()

    def _process_embeddings(self) -> List[Tuple[torch.Tensor, str]]:
        processed = []
        skipped = 0
        for embedding, label in self.embeddings:
            try:
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                if embedding.dim() == 0:
                    embedding = embedding.unsqueeze(0)
                if embedding.shape[0] == 769:
                    hyperbolic_embedding = embedding
                else:
                    spatial_norm_sq = embedding.norm() ** 2
                    time_component = torch.sqrt(1 / self.curvature + spatial_norm_sq)
                    hyperbolic_embedding = torch.cat(
                        [time_component.unsqueeze(0), embedding], dim=0
                    )
                if (
                    torch.isnan(hyperbolic_embedding).any()
                    or torch.isinf(hyperbolic_embedding).any()
                ):
                    skipped += 1
                    continue
                processed.append((hyperbolic_embedding, label))
            except Exception as e:
                logger.warning(f"Error processing embedding: {e}")
                skipped += 1
        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid embeddings")
        logger.info(
            f"Processed {len(processed)} valid embeddings from {len(self.embeddings)} total"
        )
        return processed

    def get_class_distribution(self) -> Dict[str, int]:
        labels = [label for _, label in self.processed_embeddings]
        return Counter(labels)

    def __len__(self) -> int:
        return len(self.processed_embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.processed_embeddings[idx]


class DataManager:
    """Enhanced data manager with better file handling."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _find_embedding_file(self, folder: str, pattern: str) -> str:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist")
        files = [f for f in os.listdir(folder) if pattern in f and f.endswith(".pt")]
        if not files:
            fallback_patterns = [
                "all_embeddings.pt",
                "_embeddings.pt",
                "embeddings.pt",
                "train_embeddings.pt",
                "val_embeddings.pt",
                "test_embeddings.pt",
            ]
            for fallback in fallback_patterns:
                files = [f for f in os.listdir(folder) if f.endswith(fallback)]
                if files:
                    logger.info(f"Using fallback pattern: {fallback}")
                    break
        if not files:
            raise FileNotFoundError(f"No embedding files found in {folder}")
        if len(files) > 1:
            logger.warning(f"Multiple files found, using: {files[0]}")
        return files[0]

    def load_embeddings(self, folders: list, pattern: str) -> List[Tuple]:
        print('Folders is ', folders)
        all_embeddings = []
        for folder in folders:
            
            filename = self._find_embedding_file(folder, pattern)
            filepath = os.path.join(folder, filename)
            logger.info(f"Loading embeddings from: {filepath}")
            embeddings = torch.load(
                filepath, map_location="cpu", weights_only=False
            )
            logger.info(f"Successfully loaded {len(embeddings)} embeddings")
            all_embeddings.extend(embeddings)
        print(f"Total embeddings loaded: {len(all_embeddings)}")
        return all_embeddings

    def create_datasets(self) -> Tuple[HyperbolicEmbeddingDataset, ...]:
        train_data = self.load_embeddings(
            self.config.train_embedding_folder, "all_embeddings"
        )
        val_data = self.load_embeddings(self.config.val_embedding_folder, "_embeddings")
        test_data = self.load_embeddings(
            self.config.test_embedding_folder, "_embeddings"
        )
        train_dataset = HyperbolicEmbeddingDataset(train_data, self.config.curvature)
        val_dataset = HyperbolicEmbeddingDataset(val_data, self.config.curvature)
        test_dataset = HyperbolicEmbeddingDataset(test_data, self.config.curvature)
        self._log_class_distribution_and_compute_weights(
            train_dataset, val_dataset, test_dataset
        )
        return train_dataset, val_dataset, test_dataset

    def _log_class_distribution_and_compute_weights(
        self, train_dataset, val_dataset, test_dataset
    ):
        train_dist = train_dataset.get_class_distribution()
        val_dist = val_dataset.get_class_distribution()
        test_dist = test_dataset.get_class_distribution()
        logger.info(f"Train class distribution: {train_dist}")
        logger.info(f"Validation class distribution: {val_dist}")
        logger.info(f"Test class distribution: {test_dist}")
        if self.config.pos_weight is None:
            neg_count = train_dist.get("benign", 0) + train_dist.get("safe", 0)
            pos_count = train_dist.get("malicious", 0)
            if pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count
                self.config.pos_weight = pos_weight
                logger.info(f"Computed positive class weight: {pos_weight:.4f}")
            else:
                logger.warning("Could not compute pos_weight, classes might be missing")


class ModelTrainer:
    """Enhanced model trainer with best-only model saving and selectable loss."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = create_lorentz_mlr(
            num_features=config.num_features,
            num_classes=1,
            curvature=config.curvature
        )
        self.model.to(self.device)

        self.optimizer = geoopt.optim.RiemannianSGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        pos_weight = (
            torch.tensor(config.pos_weight).to(self.device)
            if config.pos_weight
            else None
        )
        if config.loss_function == "Focal":
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
            logger.info(f"Using Focal Loss with pos_weight: {config.pos_weight}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.loss_function = config.loss_function
        self.train_history = {"loss": [], "epoch": []}
        self.val_history = {"loss": [], "accuracy": [], "epoch": []}
        self.best_val_metric = (
            0.0 if config.save_best_metric == "accuracy" else float("inf")
        )
        self.best_model_path = None
        self.patience_counter = 0
        self.training_start_time = None
        self.training_end_time = None


    def _process_batch_labels(self, labels: List[str]) -> torch.Tensor:
        binary_labels = [1.0 if label == "malicious" else 0.0 for label in labels]
        tensor_labels = torch.tensor(binary_labels, dtype=torch.float32).to(self.device)
        if self.config.label_smoothing > 0:
            tensor_labels = (
                tensor_labels * (1 - self.config.label_smoothing)
                + 0.5 * self.config.label_smoothing
            )
        return tensor_labels

    def _is_better_model(self, current_metric: float) -> bool:
        return (
            current_metric > self.best_val_metric
            if self.config.save_best_metric == "accuracy"
            else current_metric < self.best_val_metric
        )

    def _cleanup_old_best_model(self):
        if (
            self.best_model_path
            and self.config.cleanup_intermediate
            and os.path.exists(self.best_model_path)
        ):
            try:
                os.remove(self.best_model_path)
                logger.info(f"Removed previous best model: {self.best_model_path}")
            except Exception as e:
                logger.warning(
                    f"Could not remove old best model {self.best_model_path}: {e}"
                )

    def _save_best_model(self, val_loss: float, val_accuracy: float, epoch: int):
        current_metric = (
            val_accuracy if self.config.save_best_metric == "accuracy" else val_loss
        )
        if self._is_better_model(current_metric):
            self._cleanup_old_best_model()
            self.best_val_metric = current_metric
            best_filename = f"{self.config.model_name}_best_epoch_{epoch:03d}_acc_{val_accuracy:.4f}.pth"
            self.best_model_path = (
                self.output_dir / str(self.loss_function) / best_filename
            )
            os.makedirs(self.best_model_path.parent, exist_ok=True)
            os.makedirs(self.output_dir / str(self.loss_function), exist_ok=True)
            self.save_model(
                best_filename,
                is_best=True,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
            logger.info(
                f"New best model saved: {best_filename} (Best {self.config.save_best_metric}: {self.best_val_metric:.4f})"
            )
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}"
            )

    def compute_detailed_metrics(
        self, y_true: List[int], y_pred: List[int], y_scores: List[float]
    ) -> Dict[str, float]:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true, y_scores
        )
        pr_auc = average_precision_score(y_true, y_scores)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "matthews_corr_coeff": float(mcc),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_samples": len(y_true),
            "positive_samples": sum(y_true),
            "negative_samples": len(y_true) - sum(y_true),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist(),
        }
        return metrics

    def plot_comprehensive_metrics(self, metrics: Dict, prefix: str = "test"):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Comprehensive Metrics ({self.config.loss_function} Loss, Threshold: {self.config.classification_threshold})",
            fontsize=16,
        )
        ax1.plot(
            metrics["fpr"],
            metrics["tpr"],
            color="blue",
            lw=2,
            label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})',
        )
        ax1.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random")
        ax1.set_title("ROC Curve")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        ax2.plot(
            metrics["recall_curve"],
            metrics["precision_curve"],
            color="green",
            lw=2,
            label=f'PR curve (AUC = {metrics["pr_auc"]:.4f})',
        )
        ax2.axhline(
            y=metrics["positive_samples"] / metrics["total_samples"],
            color="red",
            linestyle="--",
            label="Random",
        )
        ax2.set_title("Precision-Recall Curve")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        bars = ax3.bar(
            ["ROC AUC", "PR AUC"],
            [metrics["roc_auc"], metrics["pr_auc"]],
            color=["blue", "green"],
            alpha=0.7,
        )
        ax3.set_title("AUC Comparison")
        ax3.set_ylabel("AUC Score")
        ax3.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        cm_data = np.array(
            [
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]],
            ]
        )
        im = ax4.imshow(cm_data, interpolation="nearest", cmap="Blues")
        ax4.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax4).set_label("Count")
        thresh = cm_data.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax4.text(
                    j,
                    i,
                    format(cm_data[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm_data[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )
        ax4.set_ylabel("True Label")
        ax4.set_xlabel("Predicted Label")
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Benign", "Malicious"])
        ax4.set_yticklabels(["Benign", "Malicious"])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(self.output_dir / str(self.loss_function), exist_ok=True)
        plot_path = (
            self.output_dir
            / str(self.loss_function)
            / f"{prefix}_comprehensive_metrics_{self.config.loss_function.lower()}.png"
        )

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Comprehensive metrics plot saved to: {plot_path}")

    def save_hyperplane_info(self):
        try:
            if hasattr(self.model, "get_hyperplane_equation"):
                hyperplane_info = self.model.get_hyperplane_equation()
                if hyperplane_info is None:
                    return None
                hyperplane_file = (
                    self.output_dir / f"{self.config.model_name}_hyperplane.json"
                )
                serializable_info = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in hyperplane_info.items()
                }
                with open(hyperplane_file, "w") as f:
                    json.dump(serializable_info, f, indent=2)
                logger.info(f"Hyperplane information saved to: {hyperplane_file}")
                return hyperplane_info
            return None
        except Exception as e:
            logger.error(f"Error saving hyperplane information: {e}")
            return None

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            try:
                inputs = inputs.to(self.device)
                labels = self._process_batch_labels(labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                if outputs.shape != labels.shape:
                    continue
                loss = self.criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            except Exception as e:
                logger.warning(f"Error in training batch: {e}, skipping")
        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs = inputs.to(self.device)
                    labels = self._process_batch_labels(labels)
                    outputs = self.model(inputs)
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(-1)
                    if outputs.shape != labels.shape:
                        continue
                    loss = self.criterion(outputs, labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    predicted = (
                        torch.sigmoid(outputs) > self.config.classification_threshold
                    ).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    total_loss += loss.item()
                except Exception as e:
                    logger.warning(f"Error in validation batch: {e}, skipping")
        return total_loss / max(len(val_loader), 1), correct / max(total, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(
            f"Starting training for {self.config.epochs} epochs using {self.config.loss_function} loss."
        )
        self.training_start_time = datetime.now()
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_history["loss"].append(train_loss)
            self.train_history["epoch"].append(epoch + 1)
            logger.info(
                f"Epoch [{epoch + 1}/{self.config.epochs}], Train Loss: {train_loss:.4f}"
            )
            if (epoch + 1) % self.config.validation_frequency == 0:
                val_loss, val_accuracy = self.validate(val_loader)
                self.val_history["loss"].append(val_loss)
                self.val_history["accuracy"].append(val_accuracy)
                self.val_history["epoch"].append(epoch + 1)
                logger.info(
                    f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                )
                self.scheduler.step(val_loss)
                self._save_best_model(val_loss, val_accuracy, epoch + 1)
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        self.training_end_time = datetime.now()
        logger.info(
            f"Training completed. Best {self.config.save_best_metric}: {self.best_val_metric:.4f}. Duration: {self.training_end_time - self.training_start_time}"
        )

    def test(self, test_loader: DataLoader) -> Dict[str, Union[float, List[float]]]:
        logger.info(
            f"Testing with {self.config.loss_function} loss and threshold {self.config.classification_threshold}..."
        )
        self.model.eval()
        all_labels, all_scores, all_predictions = [], [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                try:
                    inputs = inputs.to(self.device)
                    binary_labels = [
                        1 if label == "malicious" else 0 for label in labels
                    ]
                    outputs = self.model(inputs)
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(-1)
                    scores = torch.sigmoid(outputs).cpu().numpy()
                    predictions = (
                        scores > self.config.classification_threshold
                    ).astype(int)
                    all_labels.extend(binary_labels)
                    all_scores.extend(scores)
                    all_predictions.extend(predictions)
                except Exception as e:
                    logger.warning(f"Error in test batch: {e}, skipping")
        if not all_labels:
            return {}
        metrics = self.compute_detailed_metrics(all_labels, all_predictions, all_scores)
        self.plot_comprehensive_metrics(metrics, "test")
        logger.info(
            "\n"
            + "=" * 60
            + f"\nCOMPREHENSIVE TEST METRICS ({self.config.loss_function} Loss)\n"
            + "=" * 60
        )
        logger.info(
            f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}"
        )
        logger.info(
            f"ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}"
        )
        logger.info(classification_report(all_labels, all_predictions))
        return metrics

    def save_metrics_summary(self, test_metrics: Dict, hyperplane_info: Dict = None):
        try:
            duration = (
                (self.training_end_time - self.training_start_time).total_seconds()
                if self.training_start_time and self.training_end_time
                else None
            )
            summary = {
                "experiment_info": {
                    "model_name": self.config.model_name,
                    "loss_function": self.config.loss_function,
                    "version": "advanced_loss_v1",
                    "best_model_path": str(self.best_model_path),
                    "training_duration_seconds": duration,
                    "modified_by": "Merybria99",
                    "modification_date": "2025-07-01 07:27:40 UTC",
                },
                "model_config": asdict(self.config),
                "training_history": self.train_history,
                "test_metrics": test_metrics,
                "hyperplane_info": hyperplane_info,
            }
            os.makedirs(self.output_dir / str(self.loss_function), exist_ok=True)
            summary_file = (
                self.output_dir
                / str(self.loss_function)
                / f"{self.config.model_name}_complete_results_{self.config.loss_function.lower()}.json"
            )

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Complete results summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving metrics summary: {e}")

    def plot_training_history(self):
        if not self.train_history["loss"] or not self.val_history["loss"]:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        loss_name = self.config.loss_function
        ax1.plot(
            self.train_history["epoch"],
            self.train_history["loss"],
            "b-",
            label=f"Training Loss ({loss_name})",
        )
        ax1.plot(
            self.val_history["epoch"],
            self.val_history["loss"],
            "r-",
            label=f"Validation Loss ({loss_name})",
        )
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(f"{loss_name} Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(
            self.val_history["epoch"],
            self.val_history["accuracy"],
            "g-",
            label="Validation Accuracy",
        )
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        plt.tight_layout()
        os.makedirs(self.output_dir / str(self.loss_function), exist_ok=True)
        history_path = (
            self.output_dir
            / str(self.loss_function)
            / f"training_history_{loss_name.lower()}.png"
        )
        plt.savefig(history_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Training history saved to: {history_path}")

    def save_model(self, filename: str, is_best: bool = False, **kwargs):
        os.makedirs(self.output_dir / str(self.loss_function), exist_ok=True)
        model_path = self.output_dir / str(self.loss_function) / filename
        try:
            save_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "is_best_model": is_best,
                "modified_by": "Merybria99",
                "modification_date": "2025-07-01 07:27:40 UTC",
                **kwargs,
            }
            torch.save(save_dict, model_path)
            logger.info(
                f"{'Best model' if is_best else 'Model'} saved to: {model_path}"
            )
        except Exception as e:
            logger.error(f"Error saving model: {e}")


def create_data_loaders(
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_manager = DataManager(config)
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets()
    num_workers = min(4, os.cpu_count()) if config.device == "cuda" else 0
    common_args = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": config.device == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common_args
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_args)
    return train_loader, val_loader, test_loader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hyperbolic embedding model with advanced loss functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_folder",
        type=list,
        default=[
            "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/visu",
            "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/mma",
        ],
    )
    parser.add_argument(
        "--val_folder",
        type=list,
        default=[
            "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/validation_visu_embeddings"
        ],
    )
    parser.add_argument(
        "--test_folder",
        type=list,
        default=[
            "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/test_visu_embeddings"
        ],
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=2.3026,
        help="Hyperbolic curvature (must be positive)",
    )
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--loss_function",
        type=str,
        default="Focal",
        choices=["BCE", "Focal"],
        help="Loss function to use.",
    )
    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=0.4,
        help="Threshold for classifying as positive.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive class weight (auto-computed if None)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="hyperbolic_mlr_model")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"Parsed arguments: {args}")
    config = TrainingConfig(
        train_embedding_folder=args.train_folder,
        val_embedding_folder=args.val_folder,
        test_embedding_folder=args.test_folder,
        curvature=args.curvature,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        loss_function=args.loss_function,
        pos_weight=args.pos_weight,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=(
            args.device
            if args.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
    )

    logger.info(f"Configuration loaded. Using {config.loss_function} loss.")
    logger.info(f"Script modified by: Merybria99 at 2025-07-01 07:27:40 UTC")

    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
        trainer = ModelTrainer(config)
        trainer.train(train_loader, val_loader)

        if trainer.best_model_path and os.path.exists(trainer.best_model_path):
            logger.info(f"Loading best model from: {trainer.best_model_path}")
            checkpoint = torch.load(
                trainer.best_model_path, map_location=trainer.device
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])

        hyperplane_info = trainer.save_hyperplane_info()
        test_metrics = trainer.test(test_loader)

        if test_metrics:
            trainer.save_metrics_summary(test_metrics, hyperplane_info)
            trainer.save_model(
                f"{config.model_name}_final_with_test_results.pth",
                test_metrics=test_metrics,
                hyperplane_info=hyperplane_info,
            )

        trainer.plot_training_history()
        logger.info(
            "=" * 60
            + "\nTRAINING COMPLETED SUCCESSFULLY!\n"
            + f"All outputs saved to: {trainer.output_dir}\n"
            + "=" * 60
        )

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
