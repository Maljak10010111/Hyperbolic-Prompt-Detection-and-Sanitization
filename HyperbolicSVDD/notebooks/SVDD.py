"""
Hyperbolic Support Vector Data Description (SVDD) Implementation - Enhanced Training

This version includes best epoch selection, early stopping, and learning rate scheduling.

Author: Merybria99
Date: 2025-07-14
"""

import json
import logging
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperbolic_svdd_enhanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set default tensor type for higher precision
torch.set_default_dtype(torch.float64)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def pairwise_inner(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0):
    """EXACT copy from original notebook"""
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-8) -> Tensor:
    """EXACT copy from original notebook"""
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.1


def elementwise_inner(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0):
    """EXACT copy from original notebook"""
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl


def elementwise_dist(x: Tensor, y: Tensor, curv: Union[float, Tensor] = 1.0, eps: float = 1e-8) -> Tensor:
    """EXACT copy from original notebook"""
    c_xyl = -curv * elementwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.1


def lorentz_inner_product(x, y):
    """EXACT copy from original notebook"""
    return -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)


def batch_hyperbolic_distance(x, y, curv=1.0, eps=1e-5, max_acosh=1e6):
    """EXACT copy from original notebook"""
    ip = lorentz_inner_product(x, y)
    # Clamp both lower and upper bounds
    val = torch.clamp(-ip, min=1.0 + eps, max=max_acosh)
    dist = torch.sqrt(torch.tensor(curv, device=x.device, dtype=x.dtype)) * torch.acosh(val)
    return dist


def load_and_process_mixed_dataset(file_path: str, curvature: float) -> Tuple[Tensor, Tensor, np.ndarray]:
    """
    Load mixed dataset and return data with time components and labels.
    
    Args:
        file_path: Path to the embeddings file
        curvature: Curvature parameter for time component
        
    Returns:
        Tuple of (all_points_with_time, all_labels_tensor, all_labels_numpy)
        - all_points_with_time: All points with time component added
        - all_labels_tensor: Labels as tensor (1 for benign, 0 for malicious)
        - all_labels_numpy: Labels as numpy array (1 for benign, 0 for malicious)
    """
    logger.info(f"Loading mixed dataset from {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data_points = torch.load(file_path)
        logger.info(f"Successfully loaded {len(data_points)} data points")
        
        # Separate points and labels
        all_points = []
        all_labels = []
        
        benign_count = 0
        malicious_count = 0
        
        for point in data_points:
            all_points.append(point[0])
            if point[1] == "benign":
                all_labels.append(1)  # 1 for benign (normal)
                benign_count += 1
            elif point[1] == "malicious":
                all_labels.append(0)  # 0 for malicious (anomaly)
                malicious_count += 1
            else:
                logger.warning(f"Unknown label: {point[1]}")
                continue
        
        if not all_points:
            logger.error("No valid points found")
            return torch.empty(0), torch.empty(0), np.array([])
        
        # Convert to tensors
        all_points_tensor = torch.stack(all_points)
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.int32)
        all_labels_numpy = np.array(all_labels)
        
        # Add time component to all points
        all_points_with_time = torch.cat(
            [
                torch.sqrt(1 / curvature + torch.sum(all_points_tensor**2, dim=-1, keepdim=True)),
                all_points_tensor,
            ],
            dim=-1,
        )
        
        logger.info(f"Processed mixed dataset: {benign_count} benign, {malicious_count} malicious points")
        logger.info(f"Total points with time component: {all_points_with_time.shape}")
        
        return all_points_with_time, all_labels_tensor, all_labels_numpy
        
    except Exception as e:
        logger.error(f"Error loading mixed dataset: {e}")
        raise


def load_and_process_dataset(file_path: str, curvature: float) -> Tuple[Tensor, Tensor]:
    """
    Load dataset and separate benign/malicious points with time components added.
    
    Args:
        file_path: Path to the embeddings file
        curvature: Curvature parameter for time component
        
    Returns:
        Tuple of (benign_points_with_time, malicious_points_with_time)
    """
    logger.info(f"Loading dataset from {file_path}")
    
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data_points = torch.load(file_path)
        logger.info(f"Successfully loaded {len(data_points)} data points")
        
        # Separate benign and malicious points
        benign_points = []
        malicious_points = []
        
        for point in data_points:
            if point[1] == "benign":
                benign_points.append(point[0])
            elif point[1] == "malicious":
                malicious_points.append(point[0])
        
        if benign_points:
            benign_tensor = torch.stack(benign_points)
            # Add time component to benign points
            benign_with_time = torch.cat(
                [
                    torch.sqrt(1 / curvature + torch.sum(benign_tensor**2, dim=-1, keepdim=True)),
                    benign_tensor,
                ],
                dim=-1,
            )
        else:
            benign_with_time = torch.empty(0)
        
        if malicious_points:
            malicious_tensor = torch.stack(malicious_points)
            # Add time component to malicious points
            malicious_with_time = torch.cat(
                [
                    torch.sqrt(1 / curvature + torch.sum(malicious_tensor**2, dim=-1, keepdim=True)),
                    malicious_tensor,
                ],
                dim=-1,
            )
        else:
            malicious_with_time = torch.empty(0)
        
        logger.info(f"Processed {len(benign_points)} benign and {len(malicious_points)} malicious points")
        
        return benign_with_time, malicious_with_time
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def evaluate_on_validation(
    model: 'LorentzHyperbolicSVDD',
    val_benign: Tensor,
    val_malicious: Tensor
) -> Dict[str, float]:
    """
    Evaluate model on validation set and return metrics.
    
    Args:
        model: Trained SVDD model
        val_benign: Validation benign points (with time component)
        val_malicious: Validation malicious points (with time component)
        
    Returns:
        Dictionary with validation metrics
    """
    if model.center_param is None or model.radius_param is None:
        return {"benign_accuracy": 0.0, "malicious_accuracy": 0.0, "overall_accuracy": 0.0}
    
    with torch.no_grad():
        val_metrics = {}
        
        # Evaluate on benign validation data
        if len(val_benign) > 0:
            benign_predictions = model.predict(val_benign)
            benign_accuracy = (benign_predictions == 1).sum().item() / len(benign_predictions)
            val_metrics["benign_accuracy"] = benign_accuracy
        else:
            val_metrics["benign_accuracy"] = 0.0
        
        # Evaluate on malicious validation data
        if len(val_malicious) > 0:
            malicious_predictions = model.predict(val_malicious)
            malicious_accuracy = (malicious_predictions == 0).sum().item() / len(malicious_predictions)
            val_metrics["malicious_accuracy"] = malicious_accuracy
        else:
            val_metrics["malicious_accuracy"] = 0.0
        
        # Calculate overall accuracy
        total_correct = 0
        total_samples = 0
        
        if len(val_benign) > 0:
            benign_correct = (model.predict(val_benign) == 1).sum().item()
            total_correct += benign_correct
            total_samples += len(val_benign)
        
        if len(val_malicious) > 0:
            malicious_correct = (model.predict(val_malicious) == 0).sum().item()
            total_correct += malicious_correct
            total_samples += len(val_malicious)
        
        if total_samples > 0:
            val_metrics["overall_accuracy"] = total_correct / total_samples
        else:
            val_metrics["overall_accuracy"] = 0.0
    
    return val_metrics


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
        
        logger.info(f"Early stopping initialized: patience={patience}, min_delta={min_delta}, mode={mode}")
    
    def __call__(self, score: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            score: Current validation score
            
        Returns:
            True if early stopping should be triggered
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            logger.debug(f"Validation improved to {score:.6f}, resetting patience counter")
        else:
            self.counter += 1
            logger.debug(f"No improvement ({score:.6f} vs {self.best_score:.6f}), patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class ModelState:
    """Utility class to save and restore model state."""
    
    def __init__(self):
        self.center_param = None
        self.radius_param = None
        self.epoch = None
        self.metrics = None
    
    def save_state(self, model: 'LorentzHyperbolicSVDD', epoch: int, metrics: Dict):
        """Save current model state."""
        self.center_param = model.center_param.data.clone() if model.center_param is not None else None
        self.radius_param = model.radius_param.data.clone() if model.radius_param is not None else None
        self.epoch = epoch
        self.metrics = metrics.copy()
        logger.debug(f"Model state saved at epoch {epoch}")
    
    def restore_state(self, model: 'LorentzHyperbolicSVDD'):
        """Restore model to saved state."""
        if self.center_param is not None and model.center_param is not None:
            model.center_param.data.copy_(self.center_param)
        if self.radius_param is not None and model.radius_param is not None:
            model.radius_param.data.copy_(self.radius_param)
        logger.info(f"Model state restored to epoch {self.epoch}")


class LorentzHyperbolicSVDD:
    """
    EXACT implementation matching the original notebook with enhanced training features
    """
    
    def __init__(
        self,
        curvature=1.0,
        radius_init=1.0,
        center_lr=0.02,
        radius_lr=0.01,
        nu=0.1,
        device="cpu",
    ):
        self.curvature = curvature
        self.radius = radius_init
        self.center_lr = center_lr
        self.radius_lr = radius_lr
        self.device = device
        self.nu = nu
        
        self.center_param = None
        self.radius_param = None
        
        logger.info(f"Initialized LorentzHyperbolicSVDD - Enhanced version with early stopping and scheduling")

    def loss_SVDD(self, x, center, radius):
        """EXACT copy from original notebook"""
        center_batch = center.unsqueeze(0).expand(x.shape[0], -1)
        distances_sq = (batch_hyperbolic_distance(x, center_batch, curv=self.curvature) ** 2)
        penalty = torch.relu(distances_sq - radius**2)
        loss = radius**2 + torch.mean(penalty) / self.nu
        return loss

    def fit(
        self, 
        x, 
        epochs: int = 100, 
        batch_size: int = 32, 
        center_lr: float = 0.02, 
        radius_lr: float = 0.01,
        val_benign: Optional[Tensor] = None,
        val_malicious: Optional[Tensor] = None,
        patience: int = 15,
        lr_scheduler_step: int = 20,
        lr_scheduler_gamma: float = 0.8,
        use_early_stopping: bool = True
    ):
        """
        Enhanced training with early stopping and learning rate scheduling.
        """
        logger.info("=== STARTING ENHANCED FIT WITH EARLY STOPPING AND LR SCHEDULING ===")
        logger.info(f"Input data shape: {x.shape}")
        logger.info(f"Curvature: {self.curvature}, Nu: {self.nu}")
        logger.info(f"Patience: {patience}, LR Step: {lr_scheduler_step}, LR Gamma: {lr_scheduler_gamma}")
        
        # Prepare data with time component (in minibatches) - EXACT FROM NOTEBOOK
        mean_center = torch.mean(x, dim=0)
        logger.info(f"Mean center before adding time component: {mean_center.shape}")
        
        # Add time component to data - EXACT FROM NOTEBOOK
        x = torch.cat(
            [torch.sqrt(1 / self.curvature + torch.sum(x**2, dim=-1, keepdim=True)), x],
            dim=-1,
        )
        x = x.to(self.device)
        logger.info(f"Data after adding time component: {x.shape}")
        
        # Add time component to mean center - EXACT FROM NOTEBOOK
        mean_center = torch.cat(
            [
                torch.sqrt(
                    1 / self.curvature + torch.sum(mean_center**2, dim=-1, keepdim=True)
                ),
                mean_center,
            ],
            dim=-1,
        )

        dataloader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

        # Initialize parameters - EXACT FROM NOTEBOOK
        self.center_param = geoopt.ManifoldParameter(
            mean_center.clone().detach().to(self.device),
            manifold=geoopt.Lorentz(k=self.curvature),
        )

        radius_init = torch.tensor(self.radius, device=self.device)
        self.radius_param = torch.nn.Parameter(
            radius_init.clone().detach().to(self.device)
        )

        # Initialize optimizers - EXACT FROM NOTEBOOK
        center_optimizer = geoopt.optim.RiemannianSGD(
            params=[self.center_param], lr=center_lr
        )
        radius_optimizer = torch.optim.SGD(
            [{"params": self.radius_param, "lr": radius_lr}]
        )

        # Initialize learning rate schedulers
        center_scheduler = torch.optim.lr_scheduler.StepLR(
            center_optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )
        radius_scheduler = torch.optim.lr_scheduler.StepLR(
            radius_optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )

        # Initialize early stopping and best model tracking
        early_stopping = EarlyStopping(patience=patience, mode='max') if use_early_stopping else None
        best_model_state = ModelState()
        best_val_score = -float('inf')
        best_epoch = 0

        training_losses = []
        validation_metrics = []
        learning_rates = {'center': [], 'radius': []}
        
        logger.info("Starting training loop...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_inside = 0
            total_seen = 0
            
            # Training phase
            for batch in dataloader:
                batch_x = batch[0]
                center_optimizer.zero_grad()
                radius_optimizer.zero_grad()
                loss = self.loss_SVDD(batch_x, self.center_param, self.radius_param)
                loss.backward()
                center_optimizer.step()
                radius_optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

                # Minibatch stats - EXACT FROM NOTEBOOK
                center_batch = self.center_param.unsqueeze(0).expand(batch_x.shape[0], -1)
                distances = batch_hyperbolic_distance(batch_x, center_batch, curv=self.curvature)
                inside_count = torch.sum(distances <= self.radius_param).item()
                total_inside += inside_count
                total_seen += batch_x.size(0)

            avg_loss = epoch_loss / total_seen
            training_losses.append(avg_loss)
            
            # Record learning rates
            learning_rates['center'].append(center_optimizer.param_groups[0]['lr'])
            learning_rates['radius'].append(radius_optimizer.param_groups[0]['lr'])
            
            # Validation evaluation
            current_val_score = 0.0
            if val_benign is not None or val_malicious is not None:
                val_metrics = evaluate_on_validation(self, val_benign, val_malicious)
                validation_metrics.append(val_metrics)
                current_val_score = val_metrics['overall_accuracy']
                
                # Check for best model
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    best_epoch = epoch
                    best_model_state.save_state(self, epoch, val_metrics)
                    logger.debug(f"New best model saved at epoch {epoch+1} with val_acc: {current_val_score:.4f}")
                
                # Enhanced logging with LR info
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                    f"Inside: {total_inside}/{total_seen}, "
                    f"Center norm: {self.center_param.norm().item():.4f}, "
                    f"Radius: {self.radius_param.item():.4f}, "
                    f"Val Acc: {current_val_score:.4f} "
                    f"(Benign: {val_metrics['benign_accuracy']:.4f}, "
                    f"Malicious: {val_metrics['malicious_accuracy']:.4f}), "
                    f"LR_center: {learning_rates['center'][-1]:.6f}, "
                    f"LR_radius: {learning_rates['radius'][-1]:.6f}"
                )
                
                # Early stopping check
                if early_stopping is not None:
                    if early_stopping(current_val_score):
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        logger.info(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch+1}")
                        break
            else:
                validation_metrics.append({})
                # EXACT logging from notebook with LR info
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                    f"Inside: {total_inside}/{total_seen}, "
                    f"Center norm: {self.center_param.norm().item():.4f}, "
                    f"Radius: {self.radius_param.item():.4f}, "
                    f"LR_center: {learning_rates['center'][-1]:.6f}, "
                    f"LR_radius: {learning_rates['radius'][-1]:.6f}"
                )
            
            # Update learning rates
            center_scheduler.step()
            radius_scheduler.step()
        
        # Restore best model if validation was used
        if val_benign is not None or val_malicious is not None:
            if best_model_state.center_param is not None:
                best_model_state.restore_state(self)
                logger.info(f"Restored best model from epoch {best_epoch+1} with validation accuracy: {best_val_score:.4f}")
        
        training_info = {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'early_stopped': early_stopping.early_stop if early_stopping else False,
            'final_epoch': epoch + 1,
            'learning_rates': learning_rates
        }
        
        return training_losses, validation_metrics, training_info

    def fit_alternatively(
        self,
        x,
        epochs: int = 100,
        batch_size: int = 1024,
        epoch_center: int = 10,
        epoch_radius: int = 5,
        center_lr: float = 0.02,
        radius_lr: float = 0.01,
        val_benign: Optional[Tensor] = None,
        val_malicious: Optional[Tensor] = None,
        patience: int = 15,
        lr_scheduler_step: int = 20,
        lr_scheduler_gamma: float = 0.8,
        use_early_stopping: bool = True
    ):
        """
        Enhanced alternating training with early stopping and learning rate scheduling.
        """
        logger.info("=== STARTING ENHANCED ALTERNATING FIT WITH EARLY STOPPING AND LR SCHEDULING ===")
        logger.info(f"Alternating pattern: {epoch_center} center + {epoch_radius} radius epochs")
        logger.info(f"Patience: {patience}, LR Step: {lr_scheduler_step}, LR Gamma: {lr_scheduler_gamma}")
        
        # Compute mean center before time component - EXACT FROM NOTEBOOK
        mean_center = torch.mean(x, dim=0)
        logger.info(f"Mean center before adding time component: {mean_center.shape}")
        
        # Add time component to dataset and mean center - EXACT FROM NOTEBOOK
        x = torch.cat(
            [torch.sqrt(1 / self.curvature + torch.sum(x**2, dim=-1, keepdim=True)), x],
            dim=-1,
        )
        x = x.to(self.device)
        logger.info(f"Data after adding time component: {x.shape}")
        
        mean_center = torch.cat(
            [
                torch.sqrt(
                    1 / self.curvature + torch.sum(mean_center**2, dim=-1, keepdim=True)
                ),
                mean_center,
            ],
            dim=-1,
        )

        dataloader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
        
        # Use mean center as initialization - EXACT FROM NOTEBOOK
        self.center_param = geoopt.ManifoldParameter(
            mean_center.clone().detach().to(self.device),
            manifold=geoopt.Lorentz(k=self.curvature),
        )
        radius_init = torch.tensor(self.radius, device=self.device)
        self.radius_param = torch.nn.Parameter(
            radius_init.clone().detach().to(self.device)
        )

        center_optimizer = geoopt.optim.RiemannianSGD(
            params=[self.center_param], lr=center_lr
        )
        radius_optimizer = torch.optim.SGD(
            [{"params": self.radius_param, "lr": radius_lr}]
        )

        # Initialize learning rate schedulers
        center_scheduler = torch.optim.lr_scheduler.StepLR(
            center_optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )
        radius_scheduler = torch.optim.lr_scheduler.StepLR(
            radius_optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )

        # Initialize early stopping and best model tracking
        early_stopping = EarlyStopping(patience=patience, mode='max') if use_early_stopping else None
        best_model_state = ModelState()
        best_val_score = -float('inf')
        best_epoch = 0

        training_losses = []
        validation_metrics = []
        learning_rates = {'center': [], 'radius': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_inside = 0
            total_seen = 0
            
            optimize_center = epoch % (epoch_center + epoch_radius) < epoch_center
            
            if optimize_center:
                # Optimize center only - EXACT FROM NOTEBOOK
                for batch in dataloader:
                    batch_x = batch[0]
                    center_optimizer.zero_grad()
                    loss = self.loss_SVDD(batch_x, self.center_param, self.radius_param)
                    loss.backward()
                    center_optimizer.step()
                    epoch_loss += loss.item() * batch_x.size(0)
                    # Minibatch stats
                    center_batch = self.center_param.unsqueeze(0).expand(batch_x.shape[0], -1)
                    distances = batch_hyperbolic_distance(batch_x, center_batch, curv=self.curvature)
                    inside_count = torch.sum(distances <= self.radius_param).item()
                    total_inside += inside_count
                    total_seen += batch_x.size(0)
            else:
                # Optimize radius only - EXACT FROM NOTEBOOK
                for batch in dataloader:
                    batch_x = batch[0]
                    radius_optimizer.zero_grad()
                    loss = self.loss_SVDD(batch_x, self.center_param, self.radius_param)
                    loss.backward()
                    radius_optimizer.step()
                    epoch_loss += loss.item() * batch_x.size(0)
                    # Minibatch stats
                    center_batch = self.center_param.unsqueeze(0).expand(batch_x.shape[0], -1)
                    distances = batch_hyperbolic_distance(batch_x, center_batch, curv=self.curvature)
                    inside_count = torch.sum(distances <= self.radius_param).item()
                    total_inside += inside_count
                    total_seen += batch_x.size(0)

            avg_loss = epoch_loss / total_seen
            training_losses.append(avg_loss)
            
            # Record learning rates
            learning_rates['center'].append(center_optimizer.param_groups[0]['lr'])
            learning_rates['radius'].append(radius_optimizer.param_groups[0]['lr'])
            
            # Validation evaluation
            current_val_score = 0.0
            if val_benign is not None or val_malicious is not None:
                val_metrics = evaluate_on_validation(self, val_benign, val_malicious)
                validation_metrics.append(val_metrics)
                current_val_score = val_metrics['overall_accuracy']
                
                # Check for best model
                if current_val_score > best_val_score:
                    best_val_score = current_val_score
                    best_epoch = epoch
                    best_model_state.save_state(self, epoch, val_metrics)
                    logger.debug(f"New best model saved at epoch {epoch+1} with val_acc: {current_val_score:.4f}")
                
                # Gradient norms for logging
                center_grad_norm = (
                    self.center_param.grad.norm().item()
                    if self.center_param.grad is not None
                    else 0.0
                )
                radius_grad_norm = (
                    self.radius_param.grad.norm().item()
                    if self.radius_param.grad is not None
                    else 0.0
                )
                
                optimization_mode = "center" if optimize_center else "radius"
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] ({optimization_mode}), Loss: {avg_loss:.4f}, "
                    f"Center: {self.center_param.norm().item():.4f}, "
                    f"Radius: {self.radius_param.item():.4f}, "
                    f"Inside: {total_inside}/{total_seen}, "
                    f"Val Acc: {current_val_score:.4f} "
                    f"(B: {val_metrics['benign_accuracy']:.4f}, "
                    f"M: {val_metrics['malicious_accuracy']:.4f}), "
                    f"LR_c: {learning_rates['center'][-1]:.6f}, "
                    f"LR_r: {learning_rates['radius'][-1]:.6f}, "
                    f"Grad_c: {center_grad_norm:.4f}, "
                    f"Grad_r: {radius_grad_norm:.4f}"
                )
                
                # Early stopping check
                if early_stopping is not None:
                    if early_stopping(current_val_score):
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        logger.info(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch+1}")
                        break
            else:
                validation_metrics.append({})
                # Original logging without validation
                center_grad_norm = (
                    self.center_param.grad.norm().item()
                    if self.center_param.grad is not None
                    else 0.0
                )
                radius_grad_norm = (
                    self.radius_param.grad.norm().item()
                    if self.radius_param.grad is not None
                    else 0.0
                )
                optimization_mode = "center" if optimize_center else "radius"
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] ({optimization_mode}), Loss: {avg_loss:.4f}, "
                    f"Center: {self.center_param.norm().item():.4f}, "
                    f"Radius: {self.radius_param.item():.4f}, "
                    f"Inside: {total_inside}/{total_seen}, "
                    f"LR_c: {learning_rates['center'][-1]:.6f}, "
                    f"LR_r: {learning_rates['radius'][-1]:.6f}, "
                    f"Grad_c: {center_grad_norm:.4f}, "
                    f"Grad_r: {radius_grad_norm:.4f}"
                )
            
            # Update learning rates
            center_scheduler.step()
            radius_scheduler.step()
        
        # Restore best model if validation was used
        if val_benign is not None or val_malicious is not None:
            if best_model_state.center_param is not None:
                best_model_state.restore_state(self)
                logger.info(f"Restored best model from epoch {best_epoch+1} with validation accuracy: {best_val_score:.4f}")
        
        training_info = {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score,
            'early_stopped': early_stopping.early_stop if early_stopping else False,
            'final_epoch': epoch + 1,
            'learning_rates': learning_rates
        }
        
        return training_losses, validation_metrics, training_info

    def predict(self, x):
        """EXACT copy from original notebook"""
        with torch.no_grad():
            distances = batch_hyperbolic_distance(x, self.center_param, curv=self.curvature)
            predictions = (distances <= self.radius_param).int()
        return predictions

    def predict_with_scores(self, x):
        """Added for metrics calculation"""
        with torch.no_grad():
            distances = batch_hyperbolic_distance(x, self.center_param, curv=self.curvature)
            predictions = (distances <= self.radius_param).int()
        return predictions, distances


class MetricsCalculator:
    """
    Handles calculation and visualization of evaluation metrics.
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize metrics calculator."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Results directory: {self.results_dir}")
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        logger.info("Calculating evaluation metrics")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
        }
        
        # Calculate precision manually to avoid confusion
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp > 0:
            metrics['precision'] = tp / (tp + fp)
        else:
            metrics['precision'] = 0.0
        
        if y_scores is not None:
            # For ROC AUC, we need to flip scores since smaller distances = normal
            y_scores_flipped = -y_scores
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores_flipped)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None
        
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    
    def plot_training_curves_enhanced(
        self,
        training_losses: List[float],
        validation_metrics: List[Dict],
        learning_rates: Dict[str, List[float]],
        training_info: Dict,
        title_prefix: str = "",
        save_path: Optional[str] = None
    ):
        """Plot enhanced training curves with LR and best epoch."""
        epochs = range(1, len(training_losses) + 1)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training Loss
        ax1.plot(epochs, training_losses, 'b-', linewidth=2, label='Training Loss')
        if training_info.get('best_epoch') is not None:
            best_epoch = training_info['best_epoch'] + 1
            ax1.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title_prefix} - Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy
        if validation_metrics and validation_metrics[0]:
            val_epochs = range(1, len(validation_metrics) + 1)
            overall_acc = [m.get('overall_accuracy', 0) for m in validation_metrics]
            benign_acc = [m.get('benign_accuracy', 0) for m in validation_metrics]
            malicious_acc = [m.get('malicious_accuracy', 0) for m in validation_metrics]
            
            ax2.plot(val_epochs, overall_acc, 'g-', linewidth=2, marker='o', label='Overall')
            ax2.plot(val_epochs, benign_acc, 'b-', linewidth=2, marker='s', label='Benign')
            ax2.plot(val_epochs, malicious_acc, 'r-', linewidth=2, marker='^', label='Malicious')
            
            if training_info.get('best_epoch') is not None:
                best_epoch = training_info['best_epoch'] + 1
                ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'{title_prefix} - Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
        else:
            ax2.text(0.5, 0.5, 'No Validation Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{title_prefix} - Validation Accuracy (No Data)')
        
        # Plot 3: Learning Rates
        if learning_rates.get('center') and learning_rates.get('radius'):
            ax3.plot(epochs, learning_rates['center'], 'b-', linewidth=2, label='Center LR')
            ax3.plot(epochs, learning_rates['radius'], 'r-', linewidth=2, label='Radius LR')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title(f'{title_prefix} - Learning Rate Schedule')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # Plot 4: Training Info Summary
        ax4.axis('off')
        info_text = f"Training Summary:\n\n"
        info_text += f"Total Epochs: {training_info.get('final_epoch', 'N/A')}\n"
        info_text += f"Best Epoch: {training_info.get('best_epoch', 'N/A') + 1 if training_info.get('best_epoch') is not None else 'N/A'}\n"
        info_text += f"Best Val Score: {training_info.get('best_val_score', 'N/A'):.4f}\n"
        info_text += f"Early Stopped: {'Yes' if training_info.get('early_stopped', False) else 'No'}\n"
        info_text += f"Final Train Loss: {training_losses[-1]:.4f}\n"
        
        if validation_metrics and validation_metrics[-1]:
            final_val = validation_metrics[-1]
            info_text += f"\nFinal Validation:\n"
            info_text += f"Overall Acc: {final_val.get('overall_accuracy', 0):.4f}\n"
            info_text += f"Benign Acc: {final_val.get('benign_accuracy', 0):.4f}\n"
            info_text += f"Malicious Acc: {final_val.get('malicious_accuracy', 0):.4f}\n"
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Enhanced training curves saved: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Plot ROC curve."""
        # Flip scores for ROC calculation
        y_scores_flipped = -y_scores
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores_flipped)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved: {save_path}")
            
            plt.show()
            
            return fpr, tpr, roc_auc
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
            return None, None, None
    
    def plot_precision_recall_curve(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Plot Precision-Recall curve."""
        # Flip scores for PR calculation
        y_scores_flipped = -y_scores
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores_flipped)
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, 
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"PR curve saved: {save_path}")
            
            plt.show()
            
            return precision, recall, pr_auc
            
        except Exception as e:
            logger.error(f"Error plotting PR curve: {e}")
            return None, None, None
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Anomaly', 'Normal'], 
                   yticklabels=['Anomaly', 'Normal'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(
        self, 
        metrics_dict: Dict[str, Dict[str, float]], 
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None
    ):
        """Plot comparison of metrics across different experiments."""
        experiments = list(metrics_dict.keys())
        
        # Filter out None values
        filtered_metrics = {}
        for exp, metrics in metrics_dict.items():
            filtered_metrics[exp] = {k: v for k, v in metrics.items() if v is not None}
        
        metric_names = list(next(iter(filtered_metrics.values())).keys())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, exp in enumerate(experiments):
            values = [filtered_metrics[exp][metric] for metric in metric_names]
            ax.bar(x + i * width, values, width, label=exp, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, exp in enumerate(experiments):
            values = [filtered_metrics[exp][metric] for metric in metric_names]
            for j, v in enumerate(values):
                ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved: {save_path}")
        
        plt.show()
    
    def save_metrics_json(self, metrics: Dict, filepath: str):
        """Save metrics to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for k, v in metrics.items():
            if v is not None:
                if hasattr(v, 'item'):
                    json_metrics[k] = v.item()
                else:
                    json_metrics[k] = v
            else:
                json_metrics[k] = None
        
        with open(filepath, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to JSON: {filepath}")


class ModelCheckpointer:
    """Enhanced model checkpointing with training info."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpointer."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self, 
        model: LorentzHyperbolicSVDD, 
        experiment_name: str,
        training_info: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ) -> str:
        """Save enhanced model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include best epoch in filename if available
        epoch_str = ""
        if training_info and training_info.get('best_epoch') is not None:
            epoch_str = f"_best_epoch_{training_info['best_epoch']+1}"
        
        filename = f"{experiment_name}_{timestamp}{epoch_str}.pkl"
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state': {
                'center_param': model.center_param.data if model.center_param is not None else None,
                'radius_param': model.radius_param.data if model.radius_param is not None else None,
                'curvature': model.curvature,
                'nu': model.nu,
            },
            'config': {
                'curvature': model.curvature,
                'radius_init': model.radius,
                'center_lr': model.center_lr,
                'radius_lr': model.radius_lr,
                'nu': model.nu,
                'device': model.device,
            },
            'training_info': training_info,
            'metrics': metrics,
            'timestamp': timestamp,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Enhanced checkpoint saved: {filepath}")
        return str(filepath)


def test_svdd_fit_enhanced(
    hyper_points, 
    nu, 
    curvature=1.0, 
    epochs=500, 
    val_benign=None, 
    val_malicious=None,
    patience=15,
    lr_scheduler_step=20,
    lr_scheduler_gamma=0.8,
    use_early_stopping=True
):
    """Enhanced test function with early stopping and learning rate scheduling."""
    logger.info("=== STARTING ENHANCED test_svdd_fit ===")
    logger.info(f"Training parameters: epochs={epochs}, patience={patience}, lr_step={lr_scheduler_step}")
    
    num_tot = hyper_points.shape[0]
    model = LorentzHyperbolicSVDD(
        curvature=curvature, center_lr=0.1, radius_lr=0.2, nu=nu
    )

    logger.info("Starting enhanced training...")
    training_losses, validation_metrics, training_info = model.fit(
        hyper_points, 
        epochs=epochs, 
        val_benign=val_benign, 
        val_malicious=val_malicious,
        patience=patience,
        lr_scheduler_step=lr_scheduler_step,
        lr_scheduler_gamma=lr_scheduler_gamma,
        use_early_stopping=use_early_stopping
    )

    logger.info("Enhanced training completed")
    logger.info(f"Training info: {training_info}")
    logger.info(f"Final center norm: {model.center_param.norm().item():.4f}")
    logger.info(f"Final radius: {model.radius_param.item():.4f}")

    # Evaluate final model - EXACT FROM NOTEBOOK
    hyper_points_eval = torch.cat(
        [
            torch.sqrt(
                1 / model.curvature + torch.sum(hyper_points**2, dim=-1, keepdim=True)
            ),
            hyper_points,
        ],
        dim=-1,
    )
    center_batch = model.center_param.expand(hyper_points_eval.shape[0], -1)

    dists = batch_hyperbolic_distance(hyper_points_eval, center_batch, curv=model.curvature)
    logger.info(f"Distance statistics - Min: {dists.min().item():.4f}, Max: {dists.max().item():.4f}, Mean: {dists.mean().item():.4f}")
    
    inner_points = (dists <= model.radius_param.item()).float()
    count_inner = inner_points.sum().item()
    logger.info(f"Points inside radius: {count_inner}/{num_tot} ({count_inner/num_tot*100:.2f}%)")

    return model, training_losses, validation_metrics, training_info


def test_svdd_fit_alternatively_enhanced(
    hyper_points, 
    nu, 
    curvature=1.0, 
    epochs=500, 
    val_benign=None, 
    val_malicious=None,
    patience=15,
    lr_scheduler_step=20,
    lr_scheduler_gamma=0.8,
    use_early_stopping=True
):
    """Enhanced alternating test function."""
    logger.info("=== STARTING ENHANCED test_svdd_fit_alternatively ===")
    logger.info(f"Training parameters: epochs={epochs}, patience={patience}, lr_step={lr_scheduler_step}")
    
    num_tot = hyper_points.shape[0]
    model = LorentzHyperbolicSVDD(
        curvature=curvature, center_lr=0.1, radius_lr=0.2, nu=nu
    )

    logger.info("Starting enhanced alternating training...")
    training_losses, validation_metrics, training_info = model.fit_alternatively(
        hyper_points, 
        epochs=epochs, 
        val_benign=val_benign, 
        val_malicious=val_malicious,
        patience=patience,
        lr_scheduler_step=lr_scheduler_step,
        lr_scheduler_gamma=lr_scheduler_gamma,
        use_early_stopping=use_early_stopping
    )

    logger.info("Enhanced alternating training completed")
    logger.info(f"Training info: {training_info}")
    logger.info(f"Final center norm: {model.center_param.norm().item():.4f}")
    logger.info(f"Final radius: {model.radius_param.item():.4f}")

    # Evaluate final model - EXACT FROM NOTEBOOK
    hyper_points_eval = torch.cat(
        [
            torch.sqrt(
                1 / model.curvature + torch.sum(hyper_points**2, dim=-1, keepdim=True)
            ),
            hyper_points,
        ],
        dim=-1,
    )
    center_batch = model.center_param.expand(hyper_points_eval.shape[0], -1)

    dists = batch_hyperbolic_distance(hyper_points_eval, center_batch, curv=model.curvature)
    logger.info(f"Distance statistics - Min: {dists.min().item():.4f}, Max: {dists.max().item():.4f}, Mean: {dists.mean().item():.4f}")
    
    inner_points = (dists <= model.radius_param.item()).float()
    count_inner = inner_points.sum().item()
    logger.info(f"Points inside radius: {count_inner}/{num_tot} ({count_inner/num_tot*100:.2f}%)")

    return model, training_losses, validation_metrics, training_info


def evaluate_model_on_mixed_test(
    model: LorentzHyperbolicSVDD,
    test_data: Tensor,
    test_labels: np.ndarray,
    metrics_calc: MetricsCalculator,
    experiment_name: str,
    test_name: str = "Test"
) -> Dict[str, float]:
    """
    Comprehensive model evaluation on mixed test set.
    """
    logger.info(f"Comprehensive evaluation on {test_name} set (mixed data)")
    logger.info(f"Test set contains {len(test_data)} samples")
    logger.info(f"Label distribution: {np.sum(test_labels == 1)} benign, {np.sum(test_labels == 0)} malicious")
    
    # Get predictions and scores
    predictions, distances = model.predict_with_scores(test_data)
    
    # Convert to numpy for sklearn metrics
    y_true = test_labels
    y_pred = predictions.cpu().numpy()
    y_scores = distances.cpu().numpy()
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_scores)
    
    # Create save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{test_name.lower()}_{timestamp}"
    
    # Plot ROC curve
    if metrics['roc_auc'] is not None:
        roc_path = metrics_calc.results_dir / f"{base_name}_roc_curve.png"
        metrics_calc.plot_roc_curve(
            y_true, y_scores, 
            title=f"ROC Curve - {test_name} Set ({experiment_name})",
            save_path=str(roc_path)
        )
    
    # Plot Precision-Recall curve
    pr_path = metrics_calc.results_dir / f"{base_name}_pr_curve.png"
    precision, recall, pr_auc = metrics_calc.plot_precision_recall_curve(
        y_true, y_scores,
        title=f"Precision-Recall Curve - {test_name} Set ({experiment_name})",
        save_path=str(pr_path)
    )
    
    if pr_auc is not None:
        metrics['pr_auc'] = pr_auc
    
    # Plot confusion matrix
    cm_path = metrics_calc.results_dir / f"{base_name}_confusion_matrix.png"
    metrics_calc.plot_confusion_matrix(
        y_true, y_pred,
        title=f"Confusion Matrix - {test_name} Set ({experiment_name})",
        save_path=str(cm_path)
    )
    
    # Save metrics to JSON
    json_path = metrics_calc.results_dir / f"{base_name}_metrics.json"
    metrics_calc.save_metrics_json(metrics, str(json_path))
    
    # Additional detailed analysis
    benign_mask = (y_true == 1)
    malicious_mask = (y_true == 0)
    
    benign_accuracy = accuracy_score(y_true[benign_mask], y_pred[benign_mask]) if np.sum(benign_mask) > 0 else 0
    malicious_accuracy = accuracy_score(y_true[malicious_mask], y_pred[malicious_mask]) if np.sum(malicious_mask) > 0 else 0
    
    logger.info(f"{test_name} evaluation completed.")
    logger.info(f"Overall metrics: {metrics}")
    logger.info(f"Benign accuracy: {benign_accuracy:.4f}")
    logger.info(f"Malicious accuracy: {malicious_accuracy:.4f}")
    
    # Add per-class accuracies to metrics
    metrics['benign_accuracy'] = benign_accuracy
    metrics['malicious_accuracy'] = malicious_accuracy
    
    return metrics


def main():
    """
    Enhanced main function with early stopping, LR scheduling, and best epoch selection
    """
    logger.info("Starting ENHANCED experiment with early stopping and LR scheduling")
    
    # Enhanced Configuration
    train_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/visu/03f7a6e1816195a039adf08998aa1691_all_embeddings.pt"
    validation_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/validation_visu_embeddings/validation_visu_embeddings.pt"
    test_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/EMBEDDINGS/hyperbolic_safe_clip/test_visu_embeddings/test_visu_embeddings.pt"
    
    # Model parameters
    curvature = 2.3026
    nu = 0.05
    
    # Enhanced training parameters
    max_epochs_standard = 100  # Increased from 10
    max_epochs_alternative = 150  # Increased from 50
    patience = 20  # Early stopping patience
    alternate_patience = 40  # Patience for alternating optimization
    lr_scheduler_step = 25  # LR decay every 25 epochs
    lr_scheduler_gamma = 0.7  # LR decay factor
    use_early_stopping = True
    
    # Initialize utility classes
    checkpointer = ModelCheckpointer()
    metrics_calc = MetricsCalculator()
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = metrics_calc.results_dir / f"enhanced_experiment_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)
    
    logger.info(f"Enhanced experiment configuration:")
    logger.info(f"  - Max epochs: Standard={max_epochs_standard}, Alternative={max_epochs_alternative}")
    logger.info(f"  - Patience: {patience}")
    logger.info(f"  - Alternate patience: {alternate_patience}")
    logger.info(f"  - LR scheduler: step={lr_scheduler_step}, gamma={lr_scheduler_gamma}")
    logger.info(f"  - Early stopping: {use_early_stopping}")
    
    try:
        # Load training data - EXACT from notebook
        logger.info(f"Loading training embeddings from {train_path}")
        hyperbolic_points = torch.load(train_path)
        
        # get only the points whose class is 'benign' - EXACT from notebook
        bening_point = []  # Note: keeping original typo to match exactly
        for point in hyperbolic_points:
            if point[1] == "benign":
                bening_point.append(point[0])

        benign_points = torch.stack(bening_point)
        logger.info(f"Number of benign training points: {benign_points.shape}")
        
        # Load validation data
        logger.info("Loading validation data...")
        val_benign, val_malicious = load_and_process_dataset(validation_path, curvature)
        logger.info(f"Validation set - Benign: {len(val_benign)}, Malicious: {len(val_malicious)}")
        
        # Load test data (MIXED)
        logger.info("Loading mixed test data...")
        test_data, test_labels_tensor, test_labels_numpy = load_and_process_mixed_dataset(test_path, curvature)
        logger.info(f"Test set - Total: {len(test_data)}, Benign: {np.sum(test_labels_numpy == 1)}, Malicious: {np.sum(test_labels_numpy == 0)}")

        # Enhanced training with standard optimization
        logger.info("=" * 60)
        logger.info("ENHANCED TRAINING WITH STANDARD OPTIMIZATION")
        logger.info("=" * 60)
        
        benign_model, standard_losses, standard_val_metrics, standard_training_info = test_svdd_fit_enhanced(
            hyper_points=benign_points, 
            curvature=curvature, 
            nu=nu, 
            epochs=max_epochs_standard,
            val_benign=val_benign,
            val_malicious=val_malicious,
            patience=patience,
            lr_scheduler_step=lr_scheduler_step,
            lr_scheduler_gamma=lr_scheduler_gamma,
            use_early_stopping=use_early_stopping
        )
        
        # Save enhanced model checkpoint
        standard_checkpoint = checkpointer.save_checkpoint(
            benign_model, 
            "enhanced_standard_optimization",
            training_info=standard_training_info,
            metrics={
                "final_loss": standard_losses[-1] if standard_losses else None,
                "best_val_accuracy": standard_training_info.get('best_val_score', 0)
            }
        )
        
        # Plot enhanced training curves
        curves_path = experiment_dir / "standard_enhanced_training_curves.png"
        metrics_calc.plot_training_curves_enhanced(
            standard_losses, 
            standard_val_metrics, 
            standard_training_info.get('learning_rates', {}),
            standard_training_info,
            title_prefix="Standard Optimization",
            save_path=str(curves_path)
        )

        # Enhanced training with alternating optimization
        logger.info("=" * 60)
        logger.info("ENHANCED TRAINING WITH ALTERNATING OPTIMIZATION")
        logger.info("=" * 60)
        
        benign_alt_model, alt_losses, alt_val_metrics, alt_training_info = test_svdd_fit_alternatively_enhanced(
            hyper_points=benign_points, 
            curvature=curvature, 
            nu=nu, 
            epochs=max_epochs_alternative,
            val_benign=val_benign,
            val_malicious=val_malicious,
            patience=alternate_patience,
            lr_scheduler_step=lr_scheduler_step,
            lr_scheduler_gamma=lr_scheduler_gamma,
            use_early_stopping=use_early_stopping
        )
        
        # Save enhanced model checkpoint
        alt_checkpoint = checkpointer.save_checkpoint(
            benign_alt_model, 
            "enhanced_alternating_optimization",
            training_info=alt_training_info,
            metrics={
                "final_loss": alt_losses[-1] if alt_losses else None,
                "best_val_accuracy": alt_training_info.get('best_val_score', 0)
            }
        )
        
        # Plot enhanced training curves
        alt_curves_path = experiment_dir / "alternating_enhanced_training_curves.png"
        metrics_calc.plot_training_curves_enhanced(
            alt_losses, 
            alt_val_metrics, 
            alt_training_info.get('learning_rates', {}),
            alt_training_info,
            title_prefix="Alternating Optimization",
            save_path=str(alt_curves_path)
        )

        # Evaluate on MIXED TEST SET (final evaluation)
        logger.info("=" * 60)
        logger.info("FINAL ENHANCED TEST SET EVALUATION")
        logger.info("=" * 60)
        
        # Standard model on mixed test set
        if len(test_data) > 0:
            standard_test_metrics = evaluate_model_on_mixed_test(
                benign_model, test_data, test_labels_numpy,
                metrics_calc, "enhanced_standard_optimization", "test"
            )
        else:
            standard_test_metrics = {}
        
        # Alternating model on mixed test set
        if len(test_data) > 0:
            alt_test_metrics = evaluate_model_on_mixed_test(
                benign_alt_model, test_data, test_labels_numpy,
                metrics_calc, "enhanced_alternating_optimization", "test"
            )
        else:
            alt_test_metrics = {}

        # Create comprehensive comparison
        logger.info("=" * 60)
        logger.info("ENHANCED TEST SET COMPARISON")
        logger.info("=" * 60)
        
        all_test_metrics = {}
        if standard_test_metrics:
            all_test_metrics["Enhanced Standard"] = standard_test_metrics
        if alt_test_metrics:
            all_test_metrics["Enhanced Alternating"] = alt_test_metrics
        
        if all_test_metrics:
            # Plot test metrics comparison
            test_comparison_path = experiment_dir / "enhanced_test_metrics_comparison.png"
            metrics_calc.plot_metrics_comparison(
                all_test_metrics,
                title="Enhanced Mixed Test Set - SVDD Methods Comparison",
                save_path=str(test_comparison_path)
            )
        
        # Plot enhanced training comparison
        plt.figure(figsize=(15, 10))
        
        # Training loss comparison
        plt.subplot(2, 2, 1)
        plt.plot(standard_losses, label='Standard Optimization', linewidth=2, color='blue')
        plt.plot(alt_losses, label='Alternating Optimization', linewidth=2, color='red')
        
        # Mark best epochs
        if standard_training_info.get('best_epoch') is not None:
            plt.axvline(x=standard_training_info['best_epoch'], color='blue', linestyle='--', alpha=0.7)
        if alt_training_info.get('best_epoch') is not None:
            plt.axvline(x=alt_training_info['best_epoch'], color='red', linestyle='--', alpha=0.7)
        
        plt.title('Enhanced Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation accuracy comparison
        plt.subplot(2, 2, 2)
        if standard_val_metrics and alt_val_metrics:
            std_overall = [m.get('overall_accuracy', 0) for m in standard_val_metrics]
            alt_overall = [m.get('overall_accuracy', 0) for m in alt_val_metrics]
            
            plt.plot(range(1, len(std_overall) + 1), std_overall, 
                    label='Standard Optimization', linewidth=2, marker='o', color='blue')
            plt.plot(range(1, len(alt_overall) + 1), alt_overall, 
                    label='Alternating Optimization', linewidth=2, marker='s', color='red')
            
            # Mark best epochs
            if standard_training_info.get('best_epoch') is not None:
                plt.axvline(x=standard_training_info['best_epoch']+1, color='blue', linestyle='--', alpha=0.7)
            if alt_training_info.get('best_epoch') is not None:
                plt.axvline(x=alt_training_info['best_epoch']+1, color='red', linestyle='--', alpha=0.7)
            
            plt.title('Enhanced Validation Accuracy Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
        
        # Learning rate schedules
        plt.subplot(2, 2, 3)
        if standard_training_info.get('learning_rates') and alt_training_info.get('learning_rates'):
            std_lr = standard_training_info['learning_rates']
            alt_lr = alt_training_info['learning_rates']
            
            plt.plot(std_lr['center'], label='Standard Center LR', linewidth=2, color='blue', linestyle='-')
            plt.plot(std_lr['radius'], label='Standard Radius LR', linewidth=2, color='blue', linestyle='--')
            plt.plot(alt_lr['center'], label='Alternating Center LR', linewidth=2, color='red', linestyle='-')
            plt.plot(alt_lr['radius'], label='Alternating Radius LR', linewidth=2, color='red', linestyle='--')
            
            plt.title('Learning Rate Schedules')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        # Training summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = "Enhanced Training Summary:\n\n"
        summary_text += f"Standard Optimization:\n"
        summary_text += f"  Total Epochs: {standard_training_info.get('final_epoch', 'N/A')}\n"
        summary_text += f"  Best Epoch: {standard_training_info.get('best_epoch', 'N/A')+1 if standard_training_info.get('best_epoch') is not None else 'N/A'}\n"
        summary_text += f"  Best Val Score: {standard_training_info.get('best_val_score', 'N/A'):.4f}\n"
        summary_text += f"  Early Stopped: {'Yes' if standard_training_info.get('early_stopped', False) else 'No'}\n\n"
        
        summary_text += f"Alternating Optimization:\n"
        summary_text += f"  Total Epochs: {alt_training_info.get('final_epoch', 'N/A')}\n"
        summary_text += f"  Best Epoch: {alt_training_info.get('best_epoch', 'N/A')+1 if alt_training_info.get('best_epoch') is not None else 'N/A'}\n"
        summary_text += f"  Best Val Score: {alt_training_info.get('best_val_score', 'N/A'):.4f}\n"
        summary_text += f"  Early Stopped: {'Yes' if alt_training_info.get('early_stopped', False) else 'No'}\n"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        enhanced_comparison_path = experiment_dir / "enhanced_training_comparison.png"
        plt.savefig(enhanced_comparison_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Save comprehensive enhanced results summary
        enhanced_summary = {
            "experiment_info": {
                "type": "enhanced_with_early_stopping_and_lr_scheduling",
                "timestamp": timestamp,
                "curvature": curvature,
                "nu": nu,
                "max_epochs_standard": max_epochs_standard,
                "max_epochs_alternative": max_epochs_alternative,
                "patience": patience,
                "lr_scheduler_step": lr_scheduler_step,
                "lr_scheduler_gamma": lr_scheduler_gamma,
                "use_early_stopping": use_early_stopping,
                "train_benign_samples": len(benign_points),
                "val_benign_samples": len(val_benign),
                "val_malicious_samples": len(val_malicious),
                "test_total_samples": len(test_data),
                "test_benign_samples": int(np.sum(test_labels_numpy == 1)),
                "test_malicious_samples": int(np.sum(test_labels_numpy == 0)),
            },
            "model_checkpoints": {
                "enhanced_standard": standard_checkpoint,
                "enhanced_alternating": alt_checkpoint,
            },
            "training_info": {
                "standard": standard_training_info,
                "alternating": alt_training_info
            },
            "final_validation_metrics": {
                "standard": standard_val_metrics[-1] if standard_val_metrics else {},
                "alternating": alt_val_metrics[-1] if alt_val_metrics else {}
            },
            "test_metrics": all_test_metrics,
            "training_losses": {
                "standard": standard_losses,
                "alternating": alt_losses
            },
            "validation_curves": {
                "standard": standard_val_metrics,
                "alternating": alt_val_metrics
            },
            "learning_rate_schedules": {
                "standard": standard_training_info.get('learning_rates', {}),
                "alternating": alt_training_info.get('learning_rates', {})
            }
        }
        
        enhanced_summary_path = experiment_dir / "enhanced_experiment_summary.json"
        with open(enhanced_summary_path, 'w') as f:
            # Convert any tensor values to regular numbers for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_summary = convert_for_json(enhanced_summary)
            json.dump(json_summary, f, indent=2)
        
        logger.info(f"Enhanced experiment summary saved: {enhanced_summary_path}")
        
        # Enhanced final summary
        logger.info("=" * 60)
        logger.info("ENHANCED EXPERIMENT FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Results saved in: {experiment_dir}")
        
        # Training efficiency summary
        logger.info("Training Efficiency:")
        logger.info(f"Standard Optimization:")
        logger.info(f"  - Trained for {standard_training_info.get('final_epoch', 'N/A')} epochs")
        logger.info(f"  - Best model at epoch {standard_training_info.get('best_epoch', 'N/A')+1 if standard_training_info.get('best_epoch') is not None else 'N/A'}")
        logger.info(f"  - Early stopped: {'Yes' if standard_training_info.get('early_stopped', False) else 'No'}")
        logger.info(f"  - Best validation score: {standard_training_info.get('best_val_score', 'N/A'):.4f}")
        
        logger.info(f"Alternating Optimization:")
        logger.info(f"  - Trained for {alt_training_info.get('final_epoch', 'N/A')} epochs")
        logger.info(f"  - Best model at epoch {alt_training_info.get('best_epoch', 'N/A')+1 if alt_training_info.get('best_epoch') is not None else 'N/A'}")
        logger.info(f"  - Early stopped: {'Yes' if alt_training_info.get('early_stopped', False) else 'No'}")
        logger.info(f"  - Best validation score: {alt_training_info.get('best_val_score', 'N/A'):.4f}")
        
        # Final validation results
        if standard_val_metrics and alt_val_metrics:
            logger.info("Final Validation Results (using best epoch models):")
            
            std_best_val = standard_val_metrics[standard_training_info.get('best_epoch', -1)] if standard_training_info.get('best_epoch') is not None else standard_val_metrics[-1]
            alt_best_val = alt_val_metrics[alt_training_info.get('best_epoch', -1)] if alt_training_info.get('best_epoch') is not None else alt_val_metrics[-1]
            
            logger.info("Standard optimization (best epoch):")
            logger.info(f"  - Overall: {std_best_val.get('overall_accuracy', 0):.4f}")
            logger.info(f"  - Benign: {std_best_val.get('benign_accuracy', 0):.4f}")
            logger.info(f"  - Malicious: {std_best_val.get('malicious_accuracy', 0):.4f}")
            
            logger.info("Alternating optimization (best epoch):")
            logger.info(f"  - Overall: {alt_best_val.get('overall_accuracy', 0):.4f}")
            logger.info(f"  - Benign: {alt_best_val.get('benign_accuracy', 0):.4f}")
            logger.info(f"  - Malicious: {alt_best_val.get('malicious_accuracy', 0):.4f}")
        
        # Mixed test set results
        if all_test_metrics:
            logger.info("Mixed Test Set Results (using best epoch models):")
            for exp_name, metrics in all_test_metrics.items():
                logger.info(f"  {exp_name}:")
                logger.info(f"    - Overall Accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"    - F1 Score: {metrics.get('f1_score', 0):.4f}")
                logger.info(f"    - Recall: {metrics.get('recall', 0):.4f}")
                logger.info(f"    - Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"    - ROC AUC: {metrics.get('roc_auc', 'N/A')}")
                logger.info(f"    - PR AUC: {metrics.get('pr_auc', 'N/A')}")
                logger.info(f"    - Benign Accuracy: {metrics.get('benign_accuracy', 0):.4f}")
                logger.info(f"    - Malicious Accuracy: {metrics.get('malicious_accuracy', 0):.4f}")
        
        # Final comparison and recommendation
        logger.info("=" * 60)
        logger.info("ENHANCED PERFORMANCE COMPARISON & RECOMMENDATION")
        logger.info("=" * 60)
        
        if standard_test_metrics and alt_test_metrics:
            std_overall = standard_test_metrics.get('accuracy', 0)
            alt_overall = alt_test_metrics.get('accuracy', 0)
            
            std_val_score = standard_training_info.get('best_val_score', 0)
            alt_val_score = alt_training_info.get('best_val_score', 0)
            
            logger.info("Enhanced Test Set Performance Summary:")
            logger.info(f"Enhanced Standard Optimization:")
            logger.info(f"  - Test Accuracy: {std_overall:.4f}")
            logger.info(f"  - Best Validation Score: {std_val_score:.4f}")
            logger.info(f"  - Training Epochs: {standard_training_info.get('final_epoch', 'N/A')}")
            logger.info(f"  - Early Stopped: {'Yes' if standard_training_info.get('early_stopped', False) else 'No'}")
            
            logger.info(f"Enhanced Alternating Optimization:")
            logger.info(f"  - Test Accuracy: {alt_overall:.4f}")
            logger.info(f"  - Best Validation Score: {alt_val_score:.4f}")
            logger.info(f"  - Training Epochs: {alt_training_info.get('final_epoch', 'N/A')}")
            logger.info(f"  - Early Stopped: {'Yes' if alt_training_info.get('early_stopped', False) else 'No'}")
            
            # Determine best performing method
            if std_overall > alt_overall:
                logger.info(f"🏆 BEST METHOD: Enhanced Standard Optimization")
                logger.info(f"   Superior test accuracy: {std_overall:.4f} vs {alt_overall:.4f}")
            elif alt_overall > std_overall:
                logger.info(f"🏆 BEST METHOD: Enhanced Alternating Optimization")
                logger.info(f"   Superior test accuracy: {alt_overall:.4f} vs {std_overall:.4f}")
            else:
                logger.info(f"🤝 RESULT: Both enhanced methods perform equally")
                logger.info(f"   Equal test accuracy: {std_overall:.4f}")
            
            # Additional insights
            logger.info("\n📊 Training Efficiency Insights:")
            if standard_training_info.get('early_stopped', False) and not alt_training_info.get('early_stopped', False):
                logger.info("   - Standard optimization converged faster (early stopped)")
            elif alt_training_info.get('early_stopped', False) and not standard_training_info.get('early_stopped', False):
                logger.info("   - Alternating optimization converged faster (early stopped)")
            
            std_epochs = standard_training_info.get('final_epoch', float('inf'))
            alt_epochs = alt_training_info.get('final_epoch', float('inf'))
            if std_epochs < alt_epochs:
                logger.info(f"   - Standard optimization was more efficient ({std_epochs} vs {alt_epochs} epochs)")
            elif alt_epochs < std_epochs:
                logger.info(f"   - Alternating optimization was more efficient ({alt_epochs} vs {std_epochs} epochs)")
        
        logger.info("=" * 60)
        logger.info("ENHANCED EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Key enhancements applied:")
        logger.info("✅ Early stopping with patience")
        logger.info("✅ Learning rate scheduling")
        logger.info("✅ Best epoch model selection")
        logger.info("✅ Enhanced training monitoring")
        logger.info("✅ Comprehensive evaluation pipeline")
        
    except Exception as e:
        logger.error(f"Error in enhanced main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()