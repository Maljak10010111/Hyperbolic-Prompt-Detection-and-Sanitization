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
        x_space = F.relu(x[..., 1:])  # applying ReLU only on space components of vector x
        k = self.manifold.k
        x_time = torch.sqrt((x_space ** 2).sum(dim=-1, keepdim=True) + k)  # recomputing time component to project back to manifold
        return torch.cat([x_time, x_space], dim=-1)