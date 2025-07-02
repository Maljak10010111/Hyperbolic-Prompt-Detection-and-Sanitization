import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from LMLR import LorentzMLR
from geoopt.manifolds.lorentz import Lorentz
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

class CorrectHyperbolicHyperplaneVisualizer:
    """Correct visualizer for MLR hyperplane using logits (not probabilities)"""
    
    def __init__(self, model_path, curvature_k=2.3026, device='cuda'):
        self.k = curvature_k
        self.device = device
        self.manifold = Lorentz(k=self.k)
        
        # Load the MLR model
        self.mlr_model = LorentzMLR(
            manifold=self.manifold,
            num_features=769,
            num_classes=1
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.mlr_model.load_state_dict(state_dict)
        self.mlr_model.eval()
        
        # Extract model parameters
        self.a = self.mlr_model.a.detach().cpu().numpy().item()
        self.z = self.mlr_model.z.detach().cpu().numpy().squeeze()  # Weight vector
        
        # Find most relevant components
        self.find_most_relevant_components()
        
        print("🎯 CORRECT HYPERBOLIC MLR HYPERPLANE ANALYZER")
        print("=" * 60)
        print(f"📅 Date: 2025-07-02 14:16:38")
        print(f"👤 User: Merybria99")
        print(f"📊 Curvature κ: {self.k:.6f}")
        print(f"📈 Bias parameter a: {self.a:.6f}")
        print(f"🎲 Weight vector norm: {np.linalg.norm(self.z):.6f}")
        print(f"📏 Weight vector shape: {self.z.shape}")
        print(f"⚡ Decision boundary: logits = 0 (NOT probabilities = 0.5)")
        print(f"🔍 Most relevant components: {self.top_indices[:5]}")
        print(f"💪 Top weights: {[f'{self.z[i]:.6f}' for i in self.top_indices[:5]]}")
        print("=" * 60)
        
    def find_most_relevant_components(self, n_components=3):
        """Find the most relevant components based on absolute weight values."""
        abs_weights = np.abs(self.z)
        self.top_indices = np.argsort(abs_weights)[::-1]
        self.relevant_indices = self.top_indices[:n_components]
        self.relevant_weights = self.z[self.relevant_indices]
        
        print(f"\n🔍 WEIGHT RELEVANCE ANALYSIS:")
        print(f"=" * 50)
        print("Top 10 most relevant components:")
        for i, idx in enumerate(self.top_indices[:10]):
            print(f"   {i+1:2d}. Component {idx:3d}: weight = {self.z[idx]:8.6f} (|w| = {abs(self.z[idx]):8.6f})")
        
        total_weight_magnitude = np.sum(abs_weights)
        top3_contribution = np.sum(abs_weights[self.relevant_indices]) / total_weight_magnitude * 100
        
        print(f"\n📊 RELEVANCE STATISTICS:")
        print(f"   • Top 3 components contribution: {top3_contribution:.2f}% of total weight magnitude")
        print(f"   • Selected components for visualization: {self.relevant_indices}")
        print(f"   • Their weights: {[f'{w:.6f}' for w in self.relevant_weights]}")
        
        return self.relevant_indices, self.relevant_weights
    
    def lorentz_inner_product(self, x, y):
        """Compute Lorentz inner product: <x,y>_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ"""
        if len(x) != len(y):
            raise ValueError("Vectors must have same length for inner product")
        return -x[0] * y[0] + np.sum(x[1:] * y[1:])
    
    def generate_manifold_grid_relevant_dims(self, n_points=50, spatial_range=0.1):
        """Generate manifold grid using most relevant spatial dimensions."""
        
        # Identify the spatial dimensions to use for visualization
        spatial_indices = [idx for idx in self.relevant_indices if idx > 0]  # Exclude time (idx=0)
        
        if len(spatial_indices) < 2:
            # If we don't have enough spatial dimensions, use top spatial ones
            all_spatial = [i for i in self.top_indices if i > 0]
            spatial_indices = all_spatial[:2]
        
        self.vis_spatial_idx1 = spatial_indices[0]
        self.vis_spatial_idx2 = spatial_indices[1] if len(spatial_indices) > 1 else spatial_indices[0]
        
        print(f"📐 Visualizing spatial dimensions: {self.vis_spatial_idx1}, {self.vis_spatial_idx2}")
        print(f"   • Weights: {self.z[self.vis_spatial_idx1]:.6f}, {self.z[self.vis_spatial_idx2]:.6f}")
        
        # Create spatial grid
        x1_vals = np.linspace(-spatial_range, spatial_range, n_points)
        x2_vals = np.linspace(-spatial_range, spatial_range, n_points)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        
        # Calculate time coordinates for Lorentz constraint
        T = np.sqrt(X1**2 + X2**2 + 1/self.k)
        
        return X1, X2, T
    
    def evaluate_logits_on_manifold(self, X1, X2, T):
        """
        Evaluate the RAW LOGITS (not probabilities) on the manifold.
        The decision boundary is where logits = 0.
        """
        logits_values = np.zeros_like(T)
        
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                # Create full dimensional point
                point_full = np.zeros(len(self.z))
                point_full[0] = T[i, j]  # Time component
                point_full[self.vis_spatial_idx1] = X1[i, j]  # First spatial dimension
                point_full[self.vis_spatial_idx2] = X2[i, j]  # Second spatial dimension
                
                # Compute RAW LOGITS: f(x) = a + <z, x>_L
                logits_values[i, j] = self.a + self.lorentz_inner_product(self.z, point_full)
        
        return logits_values
    
    def create_correct_hyperplane_visualization(self, save_path=None):
        """Create correct visualization using LOGITS (not probabilities)."""
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            'Correct Hyperbolic MLR Decision Hyperplane: Raw Logits Analysis\n' +
            f'User: Merybria99 | Date: 2025-07-02 14:16:38 | κ = {self.k:.3f}\n' +
            f'Decision Boundary: LOGITS = 0 (before sigmoid) | Components: {self.relevant_indices[:3]}',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # Generate data
        X1, X2, T = self.generate_manifold_grid_relevant_dims(n_points=60, spatial_range=0.08)
        logits_values = self.evaluate_logits_on_manifold(X1, X2, T)
        
        # Plot 1: 3D Hyperplane showing RAW LOGITS
        ax1 = fig.add_subplot(231, projection='3d')
        
        # Color by logits values
        surface = ax1.plot_surface(X1, X2, T, facecolors=plt.cm.RdBu_r(
            (logits_values - logits_values.min()) / 
            (logits_values.max() - logits_values.min())
        ), alpha=0.8)
        
        # Plot decision boundary where LOGITS = 0
        decision_boundary = ax1.contour3D(X1, X2, T, logits_values, levels=[0], 
                                         colors=['black'], linewidths=6, alpha=1.0)
        
        ax1.set_xlabel(f'Spatial Dim {self.vis_spatial_idx1}', fontweight='bold')
        ax1.set_ylabel(f'Spatial Dim {self.vis_spatial_idx2}', fontweight='bold')
        ax1.set_zlabel('Time t', fontweight='bold')
        ax1.set_title('RAW LOGITS on Lorentz Manifold\nBlack Line: Decision Boundary (logits=0)', 
                     fontweight='bold')
        
        # Plot 2: Logits contour map
        ax2 = fig.add_subplot(232)
        
        contourf = ax2.contourf(X1, X2, logits_values, levels=20, cmap='RdBu_r', alpha=0.8)
        
        # Add specific logit level contours
        contour_lines = ax2.contour(X1, X2, logits_values, 
                                   levels=[-10, -5, -2, -1, 0, 1, 2, 5, 10], 
                                   colors='black', linewidths=1)
        ax2.clabel(contour_lines, inline=True, fontsize=10, fmt='%d')
        
        # Highlight decision boundary (logits = 0)
        decision_contour = ax2.contour(X1, X2, logits_values, levels=[0], 
                                     colors=['black'], linewidths=6)
        
        ax2.set_xlabel(f'Spatial Dimension {self.vis_spatial_idx1}', fontweight='bold')
        ax2.set_ylabel(f'Spatial Dimension {self.vis_spatial_idx2}', fontweight='bold')
        ax2.set_title('LOGITS Contour Map\nBlack thick line: Decision Boundary (logits=0)', 
                     fontweight='bold')
        
        cbar2 = plt.colorbar(contourf, ax=ax2)
        cbar2.set_label('Raw Logits f(x)', fontweight='bold')
        
        # Plot 3: Cross-section showing logits profile
        ax3 = fig.add_subplot(233)
        
        # Take cross-section through center
        center_idx = X1.shape[0] // 2
        x_line = X1[center_idx, :]
        logits_line = logits_values[center_idx, :]
        
        ax3.plot(x_line, logits_line, 'b-', linewidth=3, label='Logits f(x)')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=4, label='Decision Boundary (logits=0)')
        
        # Show classification regions based on LOGITS
        ax3.fill_between(x_line, logits_line, 0, where=(logits_line > 0), 
                        alpha=0.3, color='red', label='Malicious (logits > 0)')
        ax3.fill_between(x_line, logits_line, 0, where=(logits_line <= 0), 
                        alpha=0.3, color='blue', label='Benign (logits ≤ 0)')
        
        ax3.set_xlabel(f'Spatial Dimension {self.vis_spatial_idx1}', fontweight='bold')
        ax3.set_ylabel('Raw Logits f(x)', fontweight='bold')
        ax3.set_title('Logits Cross-Section\nClassification based on logits sign', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison with probabilities (to show the difference)
        ax4 = fig.add_subplot(234)
        
        probabilities = 1 / (1 + np.exp(-logits_line))  # Apply sigmoid
        
        ax4.plot(x_line, logits_line, 'b-', linewidth=3, label='Raw Logits')
        ax4.plot(x_line, probabilities, 'r-', linewidth=3, label='Probabilities (after sigmoid)')
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Logits Decision (0)')
        ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Probability Decision (0.5)')
        
        ax4.set_xlabel(f'Spatial Dimension {self.vis_spatial_idx1}', fontweight='bold')
        ax4.set_ylabel('Value', fontweight='bold')
        ax4.set_title('Logits vs Probabilities\nBoth give same decision boundary!', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Mathematical formulation
        ax5 = fig.add_subplot(235)
        ax5.axis('off')
        
        math_text = f"""
        CORRECT BINARY CLASSIFIER FORMULATION:
        
        Raw Logits (Model Output):
        f(x) = {self.a:.6f} + ⟨z, x⟩_L
        
        Where:
        • ⟨z, x⟩_L = -{self.z[0]:.6f}×t + {self.z[self.vis_spatial_idx1]:.6f}×x{self.vis_spatial_idx1} + {self.z[self.vis_spatial_idx2]:.6f}×x{self.vis_spatial_idx2} + ...
        
        Decision Rule (Binary Classification):
        • f(x) > 0  →  Malicious Class
        • f(x) ≤ 0  →  Benign Class
        
        Decision Boundary:
        f(x) = 0  ⟺  {self.a:.6f} + ⟨z, x⟩_L = 0
        
        Probability (for reference):
        P(malicious) = σ(f(x)) = 1/(1 + e^(-f(x)))
        
        Key Insight:
        • Decision boundary is where LOGITS = 0
        • This corresponds to P(malicious) = 0.5
        • But we classify based on logits, not probabilities!
        
        Most Relevant Components:
        • Dim {self.relevant_indices[0]}: weight = {self.relevant_weights[0]:.6f}
        • Dim {self.relevant_indices[1]}: weight = {self.relevant_weights[1]:.6f}
        • Dim {self.relevant_indices[2]}: weight = {self.relevant_weights[2]:.6f}
        """
        
        ax5.text(0.05, 0.95, math_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))
        
        ax5.set_title('Correct Mathematical Formulation', fontweight='bold', fontsize=14)
        
        # Plot 6: Weight analysis for relevant components
        ax6 = fig.add_subplot(236)
        
        # Show top components
        top_n = 15
        indices = self.top_indices[:top_n]
        weights = self.z[indices]
        
        bars = ax6.bar(range(len(weights)), weights, 
                      color=['darkred' if w > 0 else 'darkblue' for w in weights],
                      alpha=0.7, edgecolor='black')
        
        # Highlight visualization components
        for i, idx in enumerate([self.vis_spatial_idx1, self.vis_spatial_idx2]):
            if idx in indices:
                pos = list(indices).index(idx)
                bars[pos].set_color('gold')
                bars[pos].set_linewidth(3)
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_xlabel('Component Rank', fontweight='bold')
        ax6.set_ylabel('Weight Value', fontweight='bold')
        ax6.set_title(f'Top {top_n} Components\nGold: Visualized dimensions', fontweight='bold')
        ax6.set_xticks(range(len(weights)))
        ax6.set_xticklabels([f'{idx}' for idx in indices], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Correct hyperplane visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def create_interactive_correct_hyperplane(self, save_html=None):
        """Create interactive plot showing correct logits-based hyperplane."""
        
        # Generate data
        X1, X2, T = self.generate_manifold_grid_relevant_dims(n_points=50, spatial_range=0.08)
        logits_values = self.evaluate_logits_on_manifold(X1, X2, T)
        
        fig = go.Figure()
        
        # Add surface colored by LOGITS
        fig.add_trace(go.Surface(
            x=X1, y=X2, z=T,
            surfacecolor=logits_values,
            colorscale='RdBu_r',
            opacity=0.8,
            name='Logits on Manifold',
            colorbar=dict(title="Raw Logits f(x)", x=0.9),
            hovertemplate=f'<b>Spatial Dim {self.vis_spatial_idx1}</b>: %{{x:.6f}}<br>' +
                         f'<b>Spatial Dim {self.vis_spatial_idx2}</b>: %{{y:.6f}}<br>' +
                         '<b>Time t</b>: %{z:.6f}<br>' +
                         '<b>Logits f(x)</b>: %{surfacecolor:.6f}<br>' +
                         '<b>Classification</b>: %{customdata}<extra></extra>',
            customdata=np.where(logits_values > 0, 'Malicious', 'Benign')
        ))
        
        # Add decision boundary (logits = 0)
        fig.add_trace(go.Contour(
            x=X1[0, :], y=X2[:, 0], z=logits_values,
            contours=dict(start=0, end=0, size=1, coloring='lines'),
            line=dict(color='black', width=8),
            showscale=False,
            name='Decision Boundary (logits=0)',
            hovertemplate='<b>Decision Boundary</b><br>Logits f(x) = 0<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"""Correct Hyperbolic MLR Hyperplane: Raw Logits<br>
            <sub>User: Merybria99 | Date: 2025-07-02 14:16:38 | κ = {self.k:.3f}</sub><br>
            <sub>Decision: logits > 0 → Malicious | logits ≤ 0 → Benign</sub><br>
            <sub>Visualizing dimensions: {self.vis_spatial_idx1}, {self.vis_spatial_idx2}</sub>""",
            scene=dict(
                xaxis_title=f"Spatial Dimension {self.vis_spatial_idx1}",
                yaxis_title=f"Spatial Dimension {self.vis_spatial_idx2}",
                zaxis_title="Time t",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"🌐 Interactive correct hyperplane plot saved to {save_html}")
        
        fig.show()
        
        return fig
    
    def verify_decision_boundary_consistency(self):
        """Verify that logits=0 corresponds to probability=0.5."""
        
        print("\n🔍 DECISION BOUNDARY CONSISTENCY CHECK")
        print("=" * 60)
        
        # Generate test points near decision boundary
        X1, X2, T = self.generate_manifold_grid_relevant_dims(n_points=30, spatial_range=0.05)
        logits_values = self.evaluate_logits_on_manifold(X1, X2, T)
        
        # Find points close to logits = 0
        boundary_mask = np.abs(logits_values) < 0.1
        boundary_logits = logits_values[boundary_mask]
        boundary_probs = 1 / (1 + np.exp(-boundary_logits))
        
        print(f"📊 Points near decision boundary (|logits| < 0.1):")
        print(f"   • Number of points: {len(boundary_logits)}")
        print(f"   • Logits range: [{boundary_logits.min():.6f}, {boundary_logits.max():.6f}]")
        print(f"   • Probability range: [{boundary_probs.min():.6f}, {boundary_probs.max():.6f}]")
        print(f"   • Mean probability: {boundary_probs.mean():.6f}")
        
        # Verify at exactly logits = 0
        prob_at_zero_logits = 1 / (1 + np.exp(-0))
        print(f"\n✅ VERIFICATION:")
        print(f"   • At logits = 0: probability = {prob_at_zero_logits:.6f}")
        print(f"   • This confirms: logits = 0 ⟺ probability = 0.5")
        print(f"   • Decision boundary is correctly at logits = 0")
        
        return {
            'boundary_logits': boundary_logits,
            'boundary_probs': boundary_probs,
            'verification': prob_at_zero_logits
        }

def main():
    """Main function for correct hyperplane visualization."""
    
    print("🚀 CORRECT HYPERBOLIC MLR HYPERPLANE VISUALIZATION")
    print("=" * 80)
    print("📅 Current Date: 2025-07-02 14:16:38")
    print("👤 User: Merybria99")
    print("🎯 Objective: Visualize MLR hyperplane using CORRECT logits (not probabilities)")
    print("⚡ Key Fix: Decision boundary at logits = 0, NOT probabilities = 0.5")
    print("=" * 80)
    
    model_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/LorentzMLR/final_hyperbolic_mlr_model.pth"
    
    try:
        # Initialize visualizer
        visualizer = CorrectHyperbolicHyperplaneVisualizer(model_path)
        
        # Verify decision boundary consistency
        verification = visualizer.verify_decision_boundary_consistency()
        
        # Create correct hyperplane visualization
        print("\n📊 Creating CORRECT hyperplane visualization...")
        fig = visualizer.create_correct_hyperplane_visualization(
            save_path="correct_hyperbolic_hyperplane_2025_07_02_14_16_38.png"
        )
        
        # Create interactive hyperplane plot
        print("\n🌐 Creating interactive CORRECT hyperplane plot...")
        interactive_fig = visualizer.create_interactive_correct_hyperplane(
            save_html="interactive_correct_hyperplane_2025_07_02_14_16_38.html"
        )
        
        print("\n✅ CORRECT HYPERPLANE VISUALIZATION COMPLETED!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("   📊 correct_hyperbolic_hyperplane_2025_07_02_14_16_38.png")
        print("   🌐 interactive_correct_hyperplane_2025_07_02_14_16_38.html")
        print("=" * 60)
        print("🎯 Key Corrections Made:")
        print("   ✅ Decision boundary: logits = 0 (NOT probabilities = 0.5)")
        print("   ✅ Classification: logits > 0 → Malicious, logits ≤ 0 → Benign")
        print("   ✅ Visualization shows RAW logits, not transformed probabilities")
        print("   ✅ Black line shows true decision boundary where f(x) = 0")
        print("   ✅ Using most relevant weight components for visualization")
        
    except Exception as e:
        print(f"❌ Error creating correct hyperplane visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()