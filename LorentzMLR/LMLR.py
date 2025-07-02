"""
Lorentz Multinomial Logistic Regression Module

This module implements multinomial logistic regression in the Lorentz (hyperbolic) model
for classification tasks in hyperbolic space.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds.lorentz import Lorentz


class LorentzMLR(nn.Module):
    """
    Multinomial Logistic Regression (MLR) in the Lorentz model.
    
    This module performs classification in hyperbolic space using hyperplanes
    defined in the Lorentz model of hyperbolic geometry.
    
    Args:
        manifold: Lorentz manifold instance
        num_features: Number of input features (including time component)
        num_classes: Number of output classes
        
    Attributes:
        manifold: The Lorentz manifold
        a: Learnable parameter controlling hyperplane position
        z: Learnable spatial components of hyperplane normal vectors
    """
    
    def __init__(self, manifold: Lorentz, num_features: int, num_classes: int):
        super().__init__()
        
        # Validate inputs
        if num_features < 2:
            raise ValueError("num_features must be at least 2 (time + spatial)")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
            
        self.manifold = manifold
        self.num_features = num_features
        self.num_classes = num_classes
        self.curvature = manifold.k.abs()
        
        # Initialize learnable parameters
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        # Hyperplane position parameter
        self.a = nn.Parameter(torch.empty(self.num_classes))
        
        # Spatial components of hyperplane normal vectors
        # Shape: (num_classes, num_features - 1)
        spatial_dim = self.num_features - 1
        self.z = nn.Parameter(torch.empty(self.num_classes, spatial_dim))
        
        # Apply weight initialization
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:
        """Reset parameters using uniform initialization."""
        stdv = 1.0 / math.sqrt(self.num_features - 1)
        nn.init.uniform_(self.a, -stdv, stdv)
        nn.init.uniform_(self.z, -stdv, stdv)
        
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if x.dim() < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {x.dim()}")
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Input last dimension must be {self.num_features}, got {x.size(-1)}"
            )
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input contains infinite values")
            
    def _check_for_numerical_issues(self, tensor: torch.Tensor, name: str) -> None:
        """Check tensor for NaN or infinite values."""
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Infinite values detected in {name}")
            
    def _compute_hyperplane_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute hyperplane parameters.
        
        Returns:
            Tuple of (sqrt_mK, w_t, w_s) where:
            - sqrt_mK: Square root of negative curvature
            - w_t: Time components of hyperplane normals
            - w_s: Spatial components of hyperplane normals
        """
       
        sqrt_mK = 1 / self.curvature.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        
        w_t = torch.sinh(sqrt_mK * self.a) * norm_z
        w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
        
        return sqrt_mK, w_t, w_s
        
    def _extract_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract time and spatial components from Lorentz points.
        
        Args:
            x: Input tensor with shape (..., num_features)
            
        Returns:
            Tuple of (time_component, spatial_component)
        """
        time_component = x.narrow(-1, 0, 1)  # First component
        # print(f"time_component shape: {time_component.shape}, values: {time_component}")
        spatial_component = x.narrow(-1, 1, x.shape[-1] - 1)  # Remaining components
        # print(f"spatial_component shape: {spatial_component.shape}, values: {spatial_component}")
        return time_component, spatial_component
        
    def _compute_distance_to_hyperplane(
        self, 
        x: torch.Tensor, 
        sqrt_mK: torch.Tensor, 
        w_t: torch.Tensor, 
        w_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute signed distance from points to hyperplanes.
        
        Args:
            x: Input Lorentz points
            sqrt_mK: Square root of negative curvature
            w_t: Time components of hyperplane normals
            w_s: Spatial components of hyperplane normals
            
        Returns:
            Signed distances to hyperplanes
        """
        # Extract time and spatial components
        time_comp, spatial_comp = self._extract_components(x)
        # Compute beta (hyperplane parameter)
        beta = torch.sqrt(-(w_t**2) + torch.norm(w_s, dim=-1) ** 2)
        self._check_for_numerical_issues(beta, "beta calculation")
        
        # Compute alpha (signed distance numerator)
        alpha = (-w_t * time_comp + 
                torch.cosh(sqrt_mK * self.a) * torch.inner(spatial_comp, self.z))
        self._check_for_numerical_issues(alpha, "alpha calculation")
        
        # Compute distance
        distance_arg = sqrt_mK * alpha / beta
        d = self.curvature.sqrt() * torch.abs(torch.asinh(distance_arg))
        self._check_for_numerical_issues(d, "distance calculation")
        
        return torch.sign(alpha) * beta * d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Lorentz MLR model.
        
        Args:
            x: Input tensor of Lorentz points with shape (..., num_features)            
        Returns:
            Logits tensor with shape (..., num_classes)
        """
        # Validate input
        self._validate_input(x)
        
        # Compute hyperplane parameters
        sqrt_mK, w_t, w_s = self._compute_hyperplane_params()
        
        # Compute signed distances (logits)
        logits = self._compute_distance_to_hyperplane(x, sqrt_mK, w_t, w_s)
        self._check_for_numerical_issues(logits, "logits calculation")
        
        return logits.squeeze(-1)
        
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'num_features={self.num_features}, '
                f'num_classes={self.num_classes}, '
                f'curvature={self.manifold.k.item().abs():.6f}')


# Example usage and helper functions
def create_lorentz_manifold(curvature: float = -1.0) -> Lorentz:
    """
    Create a Lorentz manifold with specified curvature.
    
    Args:
        curvature: Negative curvature value (default: -1.0)
        
    Returns:
        Lorentz manifold instance
    """
    if curvature <= 0:
        raise ValueError("Curvature must be negative for Lorentz manifold")
    return Lorentz(k=-torch.tensor(curvature))


def create_lorentz_mlr(
    num_features: int, 
    num_classes: int, 
    curvature: float = -1.0
) -> LorentzMLR:
    """
    Factory function to create a LorentzMLR model.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        curvature: Manifold curvature (default: -1.0)
        
    Returns:
        LorentzMLR model instance
    """
    manifold = create_lorentz_manifold(curvature)
    return LorentzMLR(manifold, num_features, num_classes)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_lorentz_mlr(num_features=10, num_classes=5)
    
    # Create sample input (batch_size=32, features=10)
    batch_size = 32
    x = torch.randn(batch_size, 10)
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model: {model}")