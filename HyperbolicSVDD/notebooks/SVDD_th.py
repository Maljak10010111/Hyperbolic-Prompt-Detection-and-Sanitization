import math
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import geoopt


def elementwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl


def elementwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    c_xyl = -curv * elementwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def project_to_lorentz(x, curv=1.0):
    space = x[..., 0:]
    t = torch.sqrt(1.0 / curv + torch.sum(space**2, dim=-1, keepdim=True))
    return torch.cat([t, space], dim=-1)


class LorentzHyperbolicOriginSVDD:
    def __init__(
        self,
        curvature=1.0,
        radius_init=0.1,
        radius_lr=0.01,
        nu=0.1,
        device="cpu",
        center_init: str = "origin",
        dimension: int = 768,
    ):
        self.curvature = curvature
        self.radius = radius_init
        self.radius_lr = radius_lr
        self.device = device
        self.nu = nu
        self.center_init = center_init
        self.dimension = dimension
        self.center = self.get_center_init()
        self.best_radius = None
        self.best_val_score = float('inf')
        self.early_stop_patience = 10
        self.no_improve_count = 0

    def get_center_init(self):
        if self.center_init == "mean":
            print("")
            # mean_center = torch.mean(x, dim=0)
            # mean_center = project_to_lorentz(
            #     mean_center.unsqueeze(0), self.curvature
            # ).squeeze(0)
            # return mean_center
        elif self.center_init == "origin":
            root = torch.zeros((1, self.dimension))
            root = project_to_lorentz(root, self.curvature)
            return root
        else:
            raise ValueError(f"Unknown center_init value: {self.center_init}")


    def predict_xai(self, x):
        distances = elementwise_dist(
            x[:, 1:], self.center[0][1:], curv=self.curvature
        )
        return distances

    def predict_xai_no_grad(self, x):
        with torch.no_grad():
            distances = elementwise_dist(
                x[:, 1:], self.center[0][1:], curv=self.curvature
            )
        return distances
    
   
    def load(self, path):
        """Load model parameters from file"""
        checkpoint = torch.load(path)
        self.curvature = checkpoint['curvature']
        self.nu = checkpoint['nu']
        self.center = checkpoint['center']
        self.radius_param = torch.nn.Parameter(torch.tensor(checkpoint['radius']))
        return self



if __name__ == "__main__":
    print("")