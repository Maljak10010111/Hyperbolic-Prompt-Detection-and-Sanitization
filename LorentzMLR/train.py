#!/usr/bin/env python3
"""
Enhanced Hyperbolic MLR Training Script
Training a Lorentz Multinomial Logistic Regression model on hyperbolic embeddings
"""

import os
import logging
import datetime
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import geoopt
from geoopt.manifolds.lorentz import Lorentz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score

from LMLR import LorentzMLR  # Assuming LorentzMLR is defined in LorentzMLR.py


class Config:
    """Configuration class for training parameters"""
    # Data paths
    EMBEDDING_FOLDER = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/visu"
    VALIDATION_FOLDER = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/validation_visu_embeddings"
    TEST_FOLDER = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/test_visu_embeddings"
    
    # Model parameters
    CURVATURE_K = 2.3026
    NUM_FEATURES = 769
    NUM_CLASSES = 1
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    EPOCHS = 25
    VALIDATION_FREQUENCY = 2  # Validate every N epochs
    
    # Logging
    LOG_FREQUENCY = 1  # Log every N epochs
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging configuration"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"hyperbolic_mlr_training_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting Hyperbolic MLR Training Session ===")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Device: {Config.DEVICE}")
    
    return logger


def load_embeddings(folder_path: str, file_pattern: str) -> torch.Tensor:
    """Load embedding files with proper error handling"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    embedding_files = list(folder_path.glob(file_pattern))
    
    if not embedding_files:
        raise FileNotFoundError(f"No files matching pattern '{file_pattern}' found in {folder_path}")
    
    # Use the first matching file
    embedding_file = embedding_files[0]
    logging.info(f"Loading embeddings from: {embedding_file}")
    
    try:
        data = torch.load(embedding_file, map_location='cpu')
        logging.info(f"Successfully loaded {len(data)} embeddings")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from {embedding_file}: {e}")


class HyperbolicEmbeddingDataset(Dataset):
    """Enhanced dataset class for hyperbolic embeddings"""
    
    def __init__(self, embeddings: List[Tuple], k: float = Config.CURVATURE_K):
        self.embeddings = embeddings
        self.k = k
        self.final_embeddings = []
        
        logging.info(f"Processing {len(embeddings)} embeddings with curvature k={k}")
        
        # Add time dimension to the hyperbolic coordinates
        for i, (embedding, label) in enumerate(embeddings):
            # Calculate time component for Lorentz manifold
            time_component = torch.tensor(
                [torch.sqrt(1 / self.k + embedding.norm() ** 2)], 
                dtype=torch.float32
            )
            # Concatenate time component with spatial embedding
            hyperbolic_embedding = torch.cat((time_component, embedding), dim=0)
            self.final_embeddings.append((hyperbolic_embedding, label))
        
        logging.info(f"Dataset initialized with {len(self.final_embeddings)} processed embeddings")
    
    def __len__(self) -> int:
        return len(self.final_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.final_embeddings[idx]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset"""
        labels = [item[1] for item in self.final_embeddings]
        return Counter(labels)


def create_data_loaders(logger: logging.Logger) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create and return data loaders for train, validation, and test sets"""
    
    logger.info("Loading datasets...")
    
    # Load data
    train_data = load_embeddings(Config.EMBEDDING_FOLDER, "*all_embeddings.pt")
    val_data = load_embeddings(Config.VALIDATION_FOLDER, "*_embeddings.pt")
    test_data = load_embeddings(Config.TEST_FOLDER, "*_embeddings.pt")
    
    # Create datasets
    train_dataset = HyperbolicEmbeddingDataset(train_data)
    val_dataset = HyperbolicEmbeddingDataset(val_data)
    test_dataset = HyperbolicEmbeddingDataset(test_data)
    
    # Log dataset statistics
    logger.info("Dataset Statistics:")
    logger.info(f"Train set: {len(train_dataset)} samples - {train_dataset.get_label_distribution()}")
    logger.info(f"Validation set: {len(val_dataset)} samples - {val_dataset.get_label_distribution()}")
    logger.info(f"Test set: {len(test_dataset)} samples - {test_dataset.get_label_distribution()}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    logger.info(f"Data loaders created with batch size: {Config.BATCH_SIZE}")
    
    return train_loader, val_loader, test_loader


def initialize_model_and_optimizer(logger: logging.Logger) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Initialize model, optimizer, and loss function"""
    
    logger.info("Initializing model and optimizer...")
    
    # Initialize Lorentz manifold and model
    manifold = Lorentz(k=Config.CURVATURE_K)
    model = LorentzMLR(
        manifold=manifold, 
        num_features=Config.NUM_FEATURES, 
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    # Initialize optimizer
    optimizer = geoopt.optim.RiemannianSGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        momentum=Config.MOMENTUM,
        nesterov=True,
    )
    
    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized on device: {Config.DEVICE}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Optimizer: RiemannianSGD (lr={Config.LEARNING_RATE}, wd={Config.WEIGHT_DECAY})")
    
    return model, optimizer, criterion


def convert_labels_to_binary(labels: List[str]) -> torch.Tensor:
    """Convert string labels to binary tensor"""
    return torch.tensor([int(label == "malicious") for label in labels], dtype=torch.float32)


def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Calculate various metrics from model outputs and labels"""
    with torch.no_grad():
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
        
        accuracy = (predictions == labels).float().mean().item()
        
        # Convert to numpy for sklearn metrics
        labels_np = labels.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        
        return {
            'accuracy': accuracy,
            'probabilities': probs_np,
            'labels': labels_np
        }


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, epoch: int, logger: logging.Logger) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Convert labels and move to device
        labels = convert_labels_to_binary(labels).to(Config.DEVICE)
        inputs = inputs.to(Config.DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch progress occasionally
        if batch_idx % (num_batches // 4) == 0:  # Log 4 times per epoch
            logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                   epoch: int, logger: logging.Logger) -> Dict[str, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = convert_labels_to_binary(labels).to(Config.DEVICE)
            inputs = inputs.to(Config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, labels)
            all_labels.extend(metrics['labels'])
            all_probabilities.extend(metrics['probabilities'])
    
    avg_loss = total_loss / len(val_loader)
    overall_accuracy = np.mean([(prob > 0.5) == label for prob, label in zip(all_probabilities, all_labels)])
    
    # Calculate additional metrics for better model selection
    all_predictions = (np.array(all_probabilities) > 0.5).astype(int)
    f1 = f1_score(all_labels, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': overall_accuracy,
        'f1_score': f1,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, 
                   logger: logging.Logger) -> Dict[str, Any]:
    """Comprehensive model evaluation"""
    logger.info("Starting comprehensive model evaluation...")
    
    model.eval()
    all_labels = []
    all_probabilities = []
    all_predictions = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = convert_labels_to_binary(labels).to(Config.DEVICE)
            inputs = inputs.to(Config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            
            metrics = calculate_metrics(outputs, labels)
            all_labels.extend(metrics['labels'])
            all_probabilities.extend(metrics['probabilities'])
            all_predictions.extend((np.array(metrics['probabilities']) > 0.5).astype(int))
    
    # Calculate comprehensive metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and metrics
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Individual metrics
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'predictions': all_predictions
    }
    
    # Log detailed results
    logger.info(f"Test Results:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  PR AUC: {pr_auc:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    
    # Classification report
    from sklearn.metrics import classification_report
    class_report = classification_report(all_labels, all_predictions, target_names=['benign', 'malicious'])
    logger.info(f"Classification Report:\n{class_report}")
    
    return results


def plot_roc_curve(results: Dict[str, Any], save_path: str = "roc_curve.png"):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr'], results['tpr'], color='blue', lw=2, 
             label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(results: Dict[str, Any], save_path: str = "precision_recall_curve.png"):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(results['recall_curve'], results['precision_curve'], color='green', lw=2, 
             label=f'PR curve (AUC = {results["pr_auc"]:.4f})')
    plt.axhline(y=np.mean(results['labels']), color='red', linestyle='--', 
                label=f'Random classifier (AP = {np.mean(results["labels"]):.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Precision-Recall curve saved to {save_path}")


def plot_confusion_matrix(results: Dict[str, Any], save_path: str = "confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Benign', 'Malicious']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion matrix saved to {save_path}")


def plot_metrics_summary(results: Dict[str, Any], save_path: str = "metrics_summary.png"):
    """Plot and save metrics summary bar chart"""
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1_score'],
        'ROC AUC': results['roc_auc'],
        'PR AUC': results['pr_auc']
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'lightsalmon'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title('Model Performance Metrics Summary')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Metrics summary saved to {save_path}")


def save_all_plots(results: Dict[str, Any]):
    """Save all evaluation plots"""
    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all plots
    plot_roc_curve(results, f"plots/roc_curve_{timestamp}.png")
    plot_precision_recall_curve(results, f"plots/precision_recall_curve_{timestamp}.png")
    plot_confusion_matrix(results, f"plots/confusion_matrix_{timestamp}.png")
    plot_metrics_summary(results, f"plots/metrics_summary_{timestamp}.png")
    
    logging.info(f"All plots saved to plots/ directory with timestamp {timestamp}")


def save_model(model: nn.Module, save_path: str = "hyperbolic_mlr_model.pth"):
    """Save the trained model"""
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")


def main():
    """Main training loop"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(logger)
        
        # Initialize model and optimizer
        model, optimizer, criterion = initialize_model_and_optimizer(logger)
        
        # Training loop
        logger.info(f"Starting training for {Config.EPOCHS} epochs...")
        best_val_accuracy = 0.0
        best_val_f1 = 0.0
        best_model_state = None
        
        for epoch in range(Config.EPOCHS):
            # Training
            train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, logger)
            
            # Logging
            if (epoch + 1) % Config.LOG_FREQUENCY == 0:
                logger.info(f"Epoch [{epoch + 1}/{Config.EPOCHS}] - Train Loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % Config.VALIDATION_FREQUENCY == 0:
                val_results = validate_model(model, val_loader, criterion, epoch, logger)
                logger.info(f"Epoch [{epoch + 1}/{Config.EPOCHS}] - "
                           f"Val Loss: {val_results['loss']:.4f}, "
                           f"Val Accuracy: {val_results['accuracy']:.4f}, "
                           f"Val F1: {val_results['f1_score']:.4f}")
                
                # Save best model based on F1 score (better for imbalanced datasets)
                if val_results['f1_score'] > best_val_f1:
                    best_val_f1 = val_results['f1_score']
                    best_val_accuracy = val_results['accuracy']
                    best_model_state = model.state_dict().copy()
                    save_model(model, "best_hyperbolic_mlr_model.pth")
                    logger.info(f"New best model saved! F1: {best_val_f1:.4f}, Accuracy: {best_val_accuracy:.4f}")
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model for final evaluation")
        
        # Final evaluation
        logger.info("Training completed. Starting final evaluation...")
        test_results = evaluate_model(model, test_loader, criterion, logger)
        
        # Save all plots
        save_all_plots(test_results)
        
        # Save final model
        save_model(model, "final_hyperbolic_mlr_model.pth")
        
        logger.info("=== Training Session Completed Successfully ===")
        logger.info(f"Best validation F1 score: {best_val_f1:.4f}")
        logger.info(f"Final test results: Accuracy={test_results['accuracy']:.4f}, "
                   f"F1={test_results['f1_score']:.4f}, AUC={test_results['roc_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()