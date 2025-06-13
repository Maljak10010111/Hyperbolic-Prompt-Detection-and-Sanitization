import torch
import torch.nn as nn

from .LorentzManifold import LorentzManifold


class LorentzFullyConnected(nn.Module):
    def __init__(
        self,
        manifold: LorentzManifold,
        in_features,
        out_features,
        bias=False,
    ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features  # includes time
        self.out_features = out_features  # includes time

        self.linear = nn.Linear(in_features, out_features, bias=bias)  # Euclidean FC for tangent space
        self.reset_parameters()

    def forward(self, x):
        tangent_vec = self.manifold.logmap0(x)  # projecting input to tangent space at origin

        tangent_space = tangent_vec[..., 1:]  # remove tangent "time", taking only spatial components of hyperbolic vector
        tangent_transformed = self.linear(tangent_space) # applying FC only on spatial dims

        # reconstructing tangent vector with 0 time component (origin)
        tangent_full = torch.cat([torch.zeros_like(tangent_transformed[..., :1]), tangent_transformed], dim=-1)

        # mapping back to manifold at origin
        output = self.manifold.expmap0(tangent_full)

        # return output
        return self.manifold.projx(output)
        #  I should return this, because output in theory should lie on hyperboloid, but due to flotting-point errors, it drifts off the manifold.
        #  projx() re-normalizes the point back to the manifold

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)


# RESOLVE THIS SHIT
class LorentzFullyConnectedNoTime(nn.Module):
    def __init__(self, manifold, in_features, out_features, bias=False):
        """
        Applies a manifold-aware FC layer to spatial components after mapping from Lorentz space.

        Args:
            manifold: CustomLorentz manifold object.
            in_features: Input dimension including time.
            out_features: Output Euclidean dimension.
        """
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        """
        x: [batch_size, in_features] -- on Lorentz manifold (includes time dim)
        returns: [batch_size, out_features] -- logits or Euclidean projections
        """
        # projecting back to tangent space at origin
        tangent_vec = self.manifold.logmap0(x)  # shape: [batch, in_features]

        # removing the time dim (first component)
        tangent_space = tangent_vec[..., 1:]  # taking spatial components only

        out = self.linear(tangent_space)
        return out


