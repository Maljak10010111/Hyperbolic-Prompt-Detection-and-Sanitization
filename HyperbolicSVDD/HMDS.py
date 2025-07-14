from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import geoopt  # For hyperbolic geometry operations


def hyperbolic_distance_batch(x, y, c=1.0):
    """
    Vectorized hyperbolic distance using the Poincaré ball model.
    x: [N, D], y: [M, D]
    Returns: [N, M] matrix of distances
    """
    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)
    y_norm_sq = np.sum(y**2, axis=1, keepdims=True)
    xy_dot = x @ y.T

    x_norm_sq = np.clip(x_norm_sq, 0, 1 - 1e-10)
    y_norm_sq = np.clip(y_norm_sq, 0, 1 - 1e-10)

    num = 2 * np.sqrt(x_norm_sq) @ np.sqrt(y_norm_sq).T + 2 * xy_dot
    denom = 1 + 2 * xy_dot + (x_norm_sq @ y_norm_sq.T)

    sqrt_c = np.sqrt(c)
    dist = (2 / sqrt_c) * np.arctanh(sqrt_c * np.sqrt(np.clip(num / denom, 0, 1 - 1e-10)))
    return dist

def project_to_3d_hyperbolic(embeddings, curvature=1.0):
    """
    Project high-dimensional hyperbolic embeddings to 3D using Hyperbolic MDS
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.detach().cpu().numpy()

    # Compute full pairwise distance matrix
    print("Computing hyperbolic pairwise distances...")
    D = hyperbolic_distance_batch(embeddings, embeddings, c=curvature)

    # Apply MDS using precomputed distances
    print("Running MDS...")
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=300)
    reduced = mds.fit_transform(D)

    # Scale and re-embed in Poincaré ball (optional)
    scale_factor = 0.9 / np.max(np.linalg.norm(reduced, axis=1))
    reduced *= scale_factor

    # Map to Poincaré ball via expmap0
    manifold = geoopt.PoincareBall(c=curvature)
    reduced_tensor = torch.tensor(reduced, dtype=torch.float32)
    reprojected = manifold.expmap0(reduced_tensor).detach().numpy()

    return reprojected

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
    plt.savefig('hyperMDS.png')


# Example usage
if __name__ == "__main__":
    import pickle
    import umap
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
    
    
    hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
                              random_state=42).fit(embedding_tensor)
    plt.scatter(hyperbolic_mapper.embedding_.T[0],
                hyperbolic_mapper.embedding_.T[1],
                c=captions, cmap='Spectral')
    
    fig = plt.figure()
    # from sklearn.preprocessing import LabelEncoder
    # import torch

    # le = LabelEncoder()
    # labels = le.fit_transform(captions)  # e.g., ['cat', 'dog', 'fish'] → [0, 1, 2]
    # captions = torch.tensor(labels, dtype=torch.long)
    # print(captions)
    # # Scale to ensure projections are within Poincaré ball
    # norms = np.linalg.norm(tangent_embeddings, axis=1, keepdims=True)
    # scale_factor = 0.9 / np.max(norms)  # Ensure points are within unit ball
    # tangent_embeddings *= scale_factor
    
    # # Convert to PyTorch tensors for hyperbolic operations
    # tangent_tensor = torch.tensor(tangent_embeddings, dtype=torch.float32)
    
    # # Create manifold and map points to Poincaré ball
    # manifold = geoopt.PoincareBall()
    # hyperbolic_embeddings = manifold.expmap0(tangent_tensor)
    
    # # Simulate caption labels (3 classes for this example)
    # labels = captions
    
    # # Visualize
    # visualize_hyperbolic_embeddings(hyperbolic_embeddings, labels)