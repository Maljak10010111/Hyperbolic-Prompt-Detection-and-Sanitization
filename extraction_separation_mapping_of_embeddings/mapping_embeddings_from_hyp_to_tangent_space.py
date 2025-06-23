import torch
from HySAC.hysac.lorentz import log_map0

hyperbolic_embeddings = torch.load(
    "benign_training_embeddings.pt")  # Shape: (N, D), assumed



curvature = 1.0
tangent_embeddings = log_map0(hyperbolic_embeddings, curv=curvature) # map hyperbolic embeddings to tangent space


torch.save(tangent_embeddings, "../../train_benign_embeddings_tangent_space.pt")
print("Tangent space embeddings saved!")
