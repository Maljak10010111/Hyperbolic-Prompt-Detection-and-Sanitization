import torch
import torch.nn as nn
from .LorentzManifold import LorentzManifold
import torch.nn.functional as F


class LorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. """

    def __init__(self, manifold: LorentzManifold):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        x_space = F.relu(x[..., 1:])  # apply ReLU on space
        # Recompute time component to project back to manifold
        k = self.manifold.k
        # x_time = torch.sqrt((x_space ** 2).sum(dim=-1, keepdim=True) + k + 1e-6)
        x_time = torch.sqrt(torch.norm(x_space, dim=-1, keepdim=True) ** 2 + k)
        return torch.cat([x_time, x_space], dim=-1)
