import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import geoopt  # For hyperbolic geometry operations

def hyperbolic_distance(x, y, c=1.0):
    """
    Compute the hyperbolic distance in the Poincaré ball
    between points x and y with curvature c
    """
    sqrt_c = np.sqrt(c)
    
    # Compute the Möbius addition
    x_norm_sq = np.sum(x**2)
    y_norm_sq = np.sum(y**2)
    xy_dot = np.sum(x * y)
    
    # Avoid numerical issues
    x_norm_sq = np.clip(x_norm_sq, 0, 1 - 1e-10)
    y_norm_sq = np.clip(y_norm_sq, 0, 1 - 1e-10)
    
    num = 2 * np.sqrt(x_norm_sq * y_norm_sq) + 2 * xy_dot
    denom = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq
    
    return (2 / sqrt_c) * np.arctanh(sqrt_c * np.sqrt(num / denom))

def project_to_3d_hyperbolic(embeddings, random_state=42):
    """
    Project high-dimensional hyperbolic embeddings to 3D
    while preserving hyperbolic geometry as much as possible
    """
    # First convert to tangent space at origin
    # For the Poincaré ball, this is approximately a logarithmic map
    
    # Creating a PyTorch tensor from numpy array
    if not isinstance(embeddings, torch.Tensor):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings_tensor = embeddings
        
    # Create a Poincaré ball manifold
    manifold = geoopt.PoincareBall()
    
    # Map to tangent space at origin
    # We're assuming embeddings are already in the Poincaré ball model
    tangent_points = manifold.logmap0(embeddings_tensor).detach().numpy()
    
    # Add alternative UMAP implementation if available
    try:
        from umap import UMAP
        print("Using UMAP for dimensionality reduction (better visualization)")
        umap = UMAP(n_components=3, random_state=random_state)
        reduced_tangent = umap.fit_transform(tangent_points)
    except ImportError:
        print("UMAP not available, falling back to PCA (install umap-learn package for better visualization)")
        # Use PCA as fallback
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=random_state)
        reduced_tangent = pca.fit_transform(tangent_points)
    
    # Scale down to ensure all points will be within the Poincaré ball after projection
    scale_factor = 0.9 / max(1e-10, np.max(np.linalg.norm(reduced_tangent, axis=1)))
    reduced_tangent *= scale_factor
    
    # Map back to Poincaré ball (approximately exponential map)
    reduced_tangent_tensor = torch.tensor(reduced_tangent, dtype=torch.float32)
    reduced_points = manifold.expmap0(reduced_tangent_tensor).detach().numpy()
    
    return reduced_points

def visualize_hyperbolic_embeddings(embeddings, captions):
    """
    Visualize hyperbolic embeddings in 3D Poincaré ball
    with color coding based on captions
    """
    # Encode captions to numeric labels
    if isinstance(captions, torch.Tensor):
        captions = captions.cpu().numpy()
    
    # Handle captions as numeric labels or convert text captions to labels
    if captions.shape[1] if len(captions.shape) > 1 else 1 == 1:
        if isinstance(captions, np.ndarray) and captions.dtype in [np.float32, np.float64, np.int32, np.int64]:
            labels = captions.flatten().astype(int)
        else:
            # Convert text captions to numeric labels
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(captions.flatten())
    else:
        # For one-hot encoded labels
        labels = np.argmax(captions, axis=1)
    
    # Project embeddings to 3D while preserving hyperbolic structure
    reduced_points = project_to_3d_hyperbolic(embeddings)
    
    # Create colormap for visualization
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)
    colormap = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    custom_cmap = ListedColormap(colormap)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with colors based on labels
    scatter = ax.scatter(
        reduced_points[:, 0], reduced_points[:, 1], reduced_points[:, 2],
        c=labels, cmap=custom_cmap, s=50, alpha=0.8
    )
    
    # Draw wireframe for Poincaré ball boundary
    r = 1.0  # Radius of Poincaré ball
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=custom_cmap(i), markersize=10,
                       label=f'Class {i}') for i in range(n_colors)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Hyperbolic Embeddings\nPoincaré Ball Model')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Class Labels')
    
    plt.tight_layout()
    plt.savefig('hyperUMAP.png')


# Example usage
if __name__ == "__main__":
    import pickle
    # You can replace this with your actual data
    # embedding_tensor, captions = generate_sample_data(n_samples=9000, dim=768, n_classes=10)
    # print(type(embedding_tensor))
    with open('/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/embeddings.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
        #print(loaded_data)
    tangent_embeddings = embedding_tensor = loaded_data[0].numpy()
    
    # print(type(embedding_tensor))
    # print(embedding_tensor[0])
    captions = loaded_data[1]
   
    num_captions=len(set(captions))
    from sklearn.preprocessing import LabelEncoder
    import torch

    le = LabelEncoder()
    labels = le.fit_transform(captions)  # e.g., ['cat', 'dog', 'fish'] → [0, 1, 2]
    captions = torch.tensor(labels, dtype=torch.long)
    print(captions)
    # Scale to ensure projections are within Poincaré ball
    norms = np.linalg.norm(tangent_embeddings, axis=1, keepdims=True)
    scale_factor = 0.9 / np.max(norms)  # Ensure points are within unit ball
    tangent_embeddings *= scale_factor
    
    # Convert to PyTorch tensors for hyperbolic operations
    tangent_tensor = torch.tensor(tangent_embeddings, dtype=torch.float32)
    
    # Create manifold and map points to Poincaré ball
    manifold = geoopt.PoincareBall()
    hyperbolic_embeddings = manifold.expmap0(tangent_tensor)
    
    # Simulate caption labels (3 classes for this example)
    labels = captions
    
    # Visualize
    visualize_hyperbolic_embeddings(hyperbolic_embeddings, labels)