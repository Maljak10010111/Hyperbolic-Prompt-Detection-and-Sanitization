# %%
import math
import torch
from torch import Tensor

# import Dataloader
from torch.utils.data import DataLoader, TensorDataset
import geoopt


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


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


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN or Inf detected in input to exp_map0")

    x_norm = torch.norm(x, dim=-1, keepdim=True)
    rc_xnorm = curv**0.5 * x_norm

    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    rc_xnorm_clamped = torch.clamp(rc_xnorm, min=eps)

    _output = torch.sinh(sinh_input) * x / rc_xnorm_clamped

    if torch.isnan(_output).any() or torch.isinf(_output).any():
        print("NaN or Inf detected in output of exp_map0")

    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-5
) -> Tensor:
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5):
    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def lorentz_inner_product(x, y):
    # x: (..., d+1), y: (..., d+1) or (1, d+1)
    lip = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    return lip


def batch_hyperbolic_distance(x, y, curv=1.0, eps=1e-5, max_acosh=1e6):
    # evaluate the points are on the manifold
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    assert is_lorentz_point(x, curv)
    assert is_lorentz_point(y, curv)
    ip = lorentz_inner_product(x, y)
    val = -ip * torch.sqrt(torch.tensor(curv, device=x.device, dtype=x.dtype))

    dist = torch.sqrt(torch.tensor(curv, device=x.device, dtype=x.dtype)) * torch.acosh(
        val
    )
    return dist


def is_lorentz_point(x, curv=1.0, tol=1e-4):
    # Returns True if x is (almost) on the Lorentz hyperboloid
    norm = -x[..., 0] ** 2 + torch.sum(x[..., 1:] ** 2, dim=-1)
    diff = (torch.abs(norm + 1.0 / curv) < tol).all()
    return diff


def project_to_lorentz(x, curv=1.0):
    space = x[..., 0:]
    t = torch.sqrt(1.0 / curv + torch.sum(space**2, dim=-1, keepdim=True))
    return torch.cat([t, space], dim=-1)

# %%
curvature = 2.3026
root = torch.zeros((1, 768))
root = project_to_lorentz(root, curvature)
print(root.shape)


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
    def to(self, device):
        self.device = device
        self.center = self.center.to(device)
        return self

    def get_center_init(self):
        if self.center_init == "mean":
            mean_center = torch.mean(x, dim=0)
            mean_center = project_to_lorentz(
                mean_center.unsqueeze(0), self.curvature
            ).squeeze(0)
            return mean_center
        elif self.center_init == "origin":
            root = torch.zeros((1, self.dimension))
            root = project_to_lorentz(root, self.curvature)
            return root
        else:
            raise ValueError(f"Unknown center_init value: {self.center_init}")

    def loss_SVDD(self, x, radius):
        penalty = []

        for x_i in x:
            distance_from_center = elementwise_dist(
                x_i[1:], self.center[0][1:], curv=self.curvature
            )
            penalty.append(torch.relu(distance_from_center - radius))

        loss = (torch.mean(torch.stack(penalty)) / self.nu) + 0.5 * radius**2
        return loss

    def evaluate(self, validation_data):
        """
        Evaluate model performance on validation data
        Returns validation loss and metrics like F1 score
        """
        with torch.no_grad():
            val_loss = self.loss_SVDD(validation_data, self.radius_param)
            
            # Calculate performance metrics
            distances = elementwise_dist(
                validation_data[:, 1:], self.center[0][1:], curv=self.curvature
            )
            predictions = (distances <= self.radius_param).int()
            
            # Here we're using validation loss as our score
            # In a real scenario, you might want to use F1 score or other metrics
            
            return val_loss.item(), predictions

    def load_validation_data(self, validation_path):
        """
        Load validation data from the specified path
        """
        validation_points = torch.load(validation_path)
        
        # Extract only benign validation points
        benign_validation = []
        for point in validation_points:
            if point[1] == "benign":
                benign_validation.append(point[0])
        
        benign_validation = torch.stack(benign_validation) if benign_validation else torch.tensor([])
        
        # Project to Lorentz space
        if len(benign_validation) > 0:
            benign_validation = project_to_lorentz(benign_validation, self.curvature).to(self.device)
        
        return benign_validation

    def fit(
        self,
        x,
        epochs: int = 100,
        batch_size: int = 32,
        radius_lr: float = 0.01,
        validation_path: str = None,
        early_stopping_patience: int = 10,
    ):
        self.early_stop_patience = early_stopping_patience
        x = project_to_lorentz(x, self.curvature).to(self.device)
        
        # Load validation data if provided
        val_data = None
        if validation_path:
            val_data = self.load_validation_data(validation_path)
            print(f"Loaded validation data: {val_data.shape if val_data is not None else 'None'}")

        dataloader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

        radius_init = torch.tensor(self.radius, device=self.device)
        self.radius_param = torch.nn.Parameter(radius_init)

        radius_optimizer = torch.optim.SGD(
            [{"params": self.radius_param, "lr": radius_lr}]
        )

        # Initialize best model tracking
        self.best_radius = self.radius_param.item()
        self.best_val_score = float('inf')
        self.no_improve_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_inside = 0
            total_seen = 0
            print(f"Epoch [{epoch+1}/{epochs}], radius parameter: {self.radius_param.item():.4f}")
            
            # Training loop
            for batch in dataloader:
                batch_x = batch[0]
                radius_optimizer.zero_grad()
                loss = self.loss_SVDD(batch_x, self.radius_param)
                loss.backward()
                radius_optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

                # Minibatch stats
                distances = elementwise_dist(
                    batch_x[:, 1:], self.center[0][1:], curv=self.curvature
                )
                inside_count = torch.sum(distances <= self.radius_param).item()
                total_inside += inside_count
                total_seen += batch_x.size(0)

            avg_loss = epoch_loss / total_seen
            print(
                f"Epoch [{epoch+1}/{epochs}], Avg Train Loss: {avg_loss:.4f}, "
                f"Points inside radius: {total_inside}/{total_seen}, "
                f"Current radius: {self.radius_param.item():.4f}"
            )
            
            # Validation step if validation data is available
            if val_data is not None and len(val_data) > 0:
                val_loss, val_predictions = self.evaluate(val_data)
                val_inside = torch.sum(val_predictions).item()
                val_accuracy = val_inside / len(val_data) if len(val_data) > 0 else 0
                
                print(
                    f"Validation Loss: {val_loss:.4f}, "
                    f"Validation Points inside: {val_inside}/{len(val_data)}, "
                    f"Validation Accuracy: {val_accuracy:.4f}"
                )
                
                # Early stopping logic
                if val_loss < self.best_val_score:
                    self.best_val_score = val_loss
                    self.best_radius = self.radius_param.item()
                    self.no_improve_count = 0
                    print(f"New best model! Saving radius = {self.best_radius:.4f}")
                else:
                    self.no_improve_count += 1
                    print(f"No improvement for {self.no_improve_count} epochs")
                    
                    if self.no_improve_count >= self.early_stop_patience:
                        print(f"Early stopping triggered! Restoring best radius = {self.best_radius:.4f}")
                        with torch.no_grad():
                            self.radius_param.copy_(torch.tensor(self.best_radius))
                        break
        
        # Restore best model
        if val_data is not None and len(val_data) > 0:
            print(f"Training finished. Final radius = {self.radius_param.item():.4f}")
            if self.best_radius != self.radius_param.item():
                print(f"Restoring best radius from validation = {self.best_radius:.4f}")
                with torch.no_grad():
                    self.radius_param.copy_(torch.tensor(self.best_radius))
        
        return self

    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.center.device)

            distances = elementwise_dist(
                x[:, 1:], self.center[0][1:], curv=self.curvature
            )
            predictions = (distances <= self.radius_param).int()
        return predictions
    
    def save(self, path):
        """Save model parameters to file"""
        torch.save({
            'radius': self.radius_param.item(),
            'center': self.center,
            'curvature': self.curvature,
            'nu': self.nu,
        }, path)
        
    def load(self, path):
        """Load model parameters from file"""
        checkpoint = torch.load(path)
        self.curvature = checkpoint['curvature']
        self.nu = checkpoint['nu']
        self.center = checkpoint['center']
        self.radius_param = torch.nn.Parameter(torch.tensor(checkpoint['radius']))
        return self

# %%
