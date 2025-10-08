# %%
import math
import torch
from torch import Tensor
from HyperbolicSVDD.notebooks.SVDD import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
import seaborn as sns

def project_to_lorentz(x, curv=1.0):
    space = x[..., 0:]
    t = torch.sqrt(1.0 / curv + torch.sum(space**2, dim=-1, keepdim=True))
    return torch.cat([t, space], dim=-1)


# %%
curvature = 2.3026
root = torch.zeros((1, 768))
root = project_to_lorentz(root, curvature)
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.patheffects as pe

ls = LightSource(azdeg=0, altdeg=65)

# Set up a publication-style without LaTeX
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 18,
    'figure.figsize': (18, 11),
    'figure.dpi': 300
})

# More pastel yellow to green gradient colors
colors = ["#939CC3", "#686EA5", "#817BBE", "#A076F4", "#C59AF9"]
# Create a softer yellow-to-green gradient colormap
pastel_cmap = LinearSegmentedColormap.from_list("soft_blue_purple", colors)

# Parameters
K = 2.3026  # Positive constant (the absolute value of curvature)
factor = 1 / np.sqrt(K)  # Scale factor

# Create a meshgrid for parameters u and v
u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)

# Hyperboloid equations (upper sheet)
x = np.sinh(u) * np.cos(v) * factor
y = np.sinh(u) * np.sin(v) * factor
z = np.cosh(u) * factor

# Create figure
fig = plt.figure(figsize=(25, 18))
ax = fig.add_subplot(111, projection="3d")
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def truncate_colormap(cmap, minval=0.25, maxval=0.9, n=256):
    """Truncate a colormap to avoid extremes (dark/white ends)."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Create softened magma_r colormap
cmap = truncate_colormap(cm.Purples, 0.3, 0.8)
# Smooth pastel shading
rgb = ls.shade(z, cmap=pastel_cmap, blend_mode='soft')
surf = ax.plot_surface(
    x, y, z,
    facecolors=rgb,
    alpha=0.5,
    linewidth=0,
    antialiased=True,
    # set on the background
    zorder=0
)

t = np.linspace(-2, 2, 100)

# Enhanced Geodesic 1: In the xz-plane with gradient effect
g1_x = np.sinh(t) * factor
g1_y = np.zeros_like(t)
g1_z = np.cosh(t) * factor

# Enhanced Geodesic 2: In the yz-plane with gradient effect
g2_x = np.zeros_like(t)
g2_y = np.sinh(t) * factor
g2_z = np.cosh(t) * factor

# Plot geodesics with simple styling
ax.plot(g1_x, g1_y, g1_z, color='grey', linewidth=0.8)
ax.plot(g2_x, g2_y, g2_z, color='grey', linewidth=0.8)

# Mark the center of the hyperboloid with a simple marker
ax.scatter([3], [3], [4], color="#00B73A", s=100, edgecolor="#00B73A", linewidth=10,
           zorder=100,)
ax.scatter([1.8], [3], [4], color="#FF0101", s=100, edgecolor="#FF0101", linewidth=10,
           zorder=100,)
# Add single equidistant curve
r_hyper = 1.6
v_circle = np.linspace(0, 2*np.pi, 200)
x_circle = np.sinh(r_hyper) * np.cos(v_circle) * factor
y_circle = np.sinh(r_hyper) * np.sin(v_circle) * factor
z_circle = np.cosh(r_hyper) * factor
ax.plot(x_circle, y_circle, z_circle, color="#8548E8", lw=4, zorder=20)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# Hide all the ticks in the axes
ax.set_xticks([-1,0,1,2])
ax.set_yticks([-1,0,1,2])
ax.set_zticks([-1, 0,1,2,3])
# hide the numbers
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# add names to the axes

# Set view limits for a consistent look
ax.set_xlim(-2.5 * factor, 2.5 * factor)
ax.set_ylim(-2.5 * factor, 2.5 * factor)
ax.set_zlim(0, 5 * factor)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

ax.view_init(elev=25, azim=45)  # Same POV as original
ax.set_box_aspect([1, 1, 1])
# remove the background when saving
# save the file with transparent background

# # Simple text label
# ax.text(0, 0, 0.5 * factor, "Root feature", color="black", fontsize=18, zorder=20, ha='center')

# save a vectorial pdf version
plt.savefig('hyperboloid_transparent.pdf', format='pdf', dpi=300, bbox_inches='tight', transparent=True)
# a svg version
plt.savefig('hyperboloid_transparent.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
# save high resolution image version
plt.savefig('hyperboloid_transparent.png', format='png', dpi=600, bbox_inches='tight', transparent=True)

# Show the plot
plt.show()

# %%

import scienceplots
plt.style.use("science")

# -----------------------------
# Unified Style
# -----------------------------
sns.set_theme(style="whitegrid", font="serif", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 600,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# -----------------------------
# Colors and Light
# -----------------------------
colors = {
    'equator': 'red',
    'meridian': 'blue',
    'equidistant': 'darkorange',
    'center': 'black',
    'g1': 'crimson',
    'g2': 'royalblue'
}
ls = LightSource(azdeg=0, altdeg=65)
pastel_cmap = LinearSegmentedColormap.from_list(
    "pastel_sky_yellow_pink", ["#6EC6FF", "#FFD54F", "#FF80AB"]
)

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(25, 10))
gs = GridSpec(1, 2, figure=fig, wspace=0.05, left=0.03, right=0.97)


# -----------------------------
# 2. Sphere
# -----------------------------
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
phi = np.linspace(0, 2*np.pi, 120)
theta = np.linspace(0, np.pi, 120)
phi, theta = np.meshgrid(phi, theta)
r_sphere = 2
x_sphere = r_sphere*np.sin(theta)*np.cos(phi)
y_sphere = r_sphere*np.sin(theta)*np.sin(phi)
z_sphere = r_sphere*np.cos(theta)

surf1 = ax1.plot_surface(x_sphere, y_sphere, z_sphere, cmap=cm.viridis_r,
                         alpha=0.7, linewidth=0, rstride=5, cstride=5, antialiased=True)

# Sphere geodesics
t = np.linspace(0, 2*np.pi, 200)
for alpha, lw in zip([0.15,0.35,0.6,0.9],[6,4,2,2]):
    ax1.plot(r_sphere*np.cos(t), r_sphere*np.sin(t), np.zeros_like(t),
             color=colors['equator'], alpha=alpha, linewidth=lw)
    ax1.plot(r_sphere*np.sin(t), np.zeros_like(t), r_sphere*np.cos(t),
             color=colors['meridian'], alpha=alpha, linewidth=lw)

ax1.set_xlabel('X', labelpad=10); ax1.set_ylabel('Y', labelpad=10); ax1.set_zlabel('Z', labelpad=10)
# set the fontsize of the axis labels
ax1.xaxis.label.set_size(35)
ax1.yaxis.label.set_size(35)
ax1.zaxis.label.set_size(35)
ax1.set_box_aspect([1,1,1])
ax1.set_xlim(-3,3); ax1.set_ylim(-3,3); ax1.set_zlim(-3,3)
ax1.set_xticks([ax1.get_xticks()[0], ax1.get_xticks()[-1]])
ax1.set_yticks([ax1.get_yticks()[0], ax1.get_yticks()[-1]])
ax1.set_zticks([-2,-1,0,1,2])
for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]: axis.pane.fill=False


# Colorbar
norm1 = Normalize(vmin=z_sphere.min(), vmax=z_sphere.max())
cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm1, cmap=cm.viridis_r), ax=ax1, shrink=0.7, pad=0.03)
cbar1.set_label('Height (Z)', rotation=90, labelpad=10, fontsize=35)
cbar1.ax.tick_params(labelsize=20)
cbar1.outline.set_visible(False)

# -----------------------------
# 3. Cut Hyperboloid
# -----------------------------
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
z_max = 1.8
factor_hyper = 1/np.sqrt(K)
u_max = np.arccosh(z_max / factor_hyper)
u = np.linspace(-u_max, u_max, 120)
v = np.linspace(0, 2*np.pi, 120)
u, v = np.meshgrid(u, v)
x_hyper = np.sinh(u)*np.cos(v)*factor_hyper
y_hyper = np.sinh(u)*np.sin(v)*factor_hyper
z_hyper = np.cosh(u)*factor_hyper -0.5

surf2 = ax2.plot_surface(x_hyper, y_hyper, z_hyper, cmap=cm.summer,
                         alpha=0.35, linewidth=0, rstride=5, cstride=5, antialiased=True)

# Equidistant curve
r_hyper = 1.3
v_circle = np.linspace(0,2*np.pi,200)
x_circle = np.sinh(r_hyper)*np.cos(v_circle)*factor_hyper
y_circle = np.sinh(r_hyper)*np.sin(v_circle)*factor_hyper
z_circle = np.cosh(r_hyper)*factor_hyper -0.5
ax2.plot(x_circle, y_circle, z_circle, color=colors['equidistant'], lw=3, zorder=20)
# set title
ax2.set_xlabel('X', labelpad=10); ax2.set_ylabel('Y', labelpad=10); ax2.set_zlabel('Z', labelpad=10)
ax2.set_box_aspect([1,1,1])
ax2.set_xlim(-2.5*factor_hyper,2.5*factor_hyper)
ax2.set_ylim(-2.5*factor_hyper,2.5*factor_hyper)
# disable x ticks and y ticks and z ticks
ax2.set_xticks([ax2.get_xticks()[0], ax2.get_xticks()[-1]])
ax2.set_yticks([ax2.get_yticks()[0], ax2.get_yticks()[-1]])
ax2.set_zticks([ -1,0, 0.5, 1, 1.5])
# set tick fontsize
ax2.xaxis.label.set_size(35)
ax2.yaxis.label.set_size(35)
ax2.zaxis.label.set_size(35)
ax2.set_zlim(0,z_max)

ax2.view_init(elev=30, azim=30)
for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]: axis.pane.fill=False

# Colorbar
norm2 = Normalize(vmin=z_hyper.min(), vmax=z_hyper.max())
cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cm.summer), ax=ax2, shrink=0.7, pad=0.03)
cbar2.set_label('Height (Z)', rotation=90, labelpad=20, fontsize=35)
cbar2.ax.tick_params(labelsize=20)
cbar2.outline.set_visible(False)

# -----------------------------
# Unified Legend
# -----------------------------
legend_elements = [
    plt.Line2D([0],[0], color=colors['equator'], lw=5, label='Equator'),
    plt.Line2D([0],[0], color=colors['meridian'], lw=5, label='Meridian'),
    plt.Line2D([0],[0], color=colors['equidistant'], lw=5, label='Equidistant curve'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.9),
           fontsize=30, frameon=True)

# -----------------------------
# Save and show
# -----------------------------
plt.savefig('three_plots_side_by_side_bars.pdf', dpi=600)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.patheffects as pe
ls = LightSource(azdeg=0, altdeg=65)

# Set up a publication-style without LaTeX
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 18,
    'figure.figsize': (18, 11),
    'figure.dpi': 300
})
colors = ["#ECFF6E", "#FFD54F", "#FF80AB"]
# sky blue → sunflower yellow → candy pink
pastel_cmap = LinearSegmentedColormap.from_list("pastel_sky_yellow_pink", colors)


# Parameters
K = 2.3026  # Positive constant (the absolute value of curvature)
factor = 1 / np.sqrt(K)  # Scale factor

# Create a meshgrid for parameters u and v
u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)

# Hyperboloid equations (upper sheet)
x = np.sinh(u) * np.cos(v) * factor
y = np.sinh(u) * np.sin(v) * factor
z = np.cosh(u) * factor

# Create figure
fig = plt.figure(figsize=(15, 10))


ax = fig.add_subplot(111, projection="3d")

rgb = ls.shade(z, cmap=pastel_cmap, vert_exag=0.5, blend_mode='soft')
surf = ax.plot_surface(
    x, y, z,
    facecolors=rgb,
    alpha=0.5,
    linewidth=0,
    antialiased=True
)


t = np.linspace(-2, 2, 100)

# Geodesic 1: In the xz-plane (y=0) - using a bright, high-contrast color
g1_x = np.sinh(t) * factor
g1_y = np.zeros_like(t)
g1_z = np.cosh(t) * factor

# Geodesic 2: In the yz-plane (x=0) - using another bright, contrasting color
g2_x = np.zeros_like(t)
g2_y = np.sinh(t) * factor
g2_z = np.cosh(t) * factor

ax.plot(g1_x, g1_y, g1_z, color='crimson', linewidth=2.2)
ax.scatter([g1_x[-1]], [g1_y[-1]], [g1_z[-1]], color='crimson', s=30, zorder=10)

ax.plot(g2_x, g2_y, g2_z, color='royalblue', linewidth=2.2)
ax.scatter([g2_x[-1]], [g2_y[-1]], [g2_z[-1]], color='royalblue', s=30, zorder=10)

# Mark the center of the hyperboloid with a distinctive marker
ax.scatter([0], [0], [factor], color='black', s=70, edgecolor='black', linewidth=0.5,
           label='Root feature', zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# add horizontal equidistant curve  onto the hyperboloid surface
r_hyper = 1.3
v_circle = np.linspace(0,2*np.pi,200)
x_circle = np.sinh(r_hyper)*np.cos(v_circle)*factor_hyper
y_circle = np.sinh(r_hyper)*np.sin(v_circle)*factor_hyper
z_circle = np.cosh(r_hyper)*factor_hyper -0.5
ax.plot(x_circle, y_circle, z_circle, color=colors['equidistant'], lw=3, zorder=20)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Publication-style labels
ax.set_xlabel('X', labelpad=10)
ax.set_ylabel('Y', labelpad=10)
ax.set_zlabel('Z', labelpad=10)

# Set view limits for a consistent look
ax.set_xlim(-2.5 * factor, 2.5 * factor)
ax.set_ylim(-2.5 * factor, 2.5 * factor)
ax.set_zlim(0, 5 * factor)
# Professional grid styling
ax.grid(False)

# Subtle transparent grid colors for each axis
ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)
ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)
ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([0, 1, 2, 3, 4])


plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

ax.view_init(elev=25, azim=45)  # Lower elevation and rotated more for better geodesic visibility
ax.set_box_aspect([1,1,1])

#put root feature label above the point and above everyting

ax.text(0, 0, 0.5*factor, "Root feature", color="black", fontsize=18, zorder=20, ha='center')

# Save with publication-quality settings
plt.savefig('hyperboloid_transparent.pdf', format='pdf', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.patheffects as pe
ls = LightSource(azdeg=0, altdeg=65)

# Set up a publication-style without LaTeX
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 18,
    'figure.figsize': (18, 11),
    'figure.dpi': 300
})
colors = ["#6EC6FF", "#FFD54F", "#FF80AB"]
# sky blue → sunflower yellow → candy pink
pastel_cmap = LinearSegmentedColormap.from_list("pastel_sky_yellow_pink", colors)


# Parameters
K = 2.3026  # Positive constant (the absolute value of curvature)
factor = 1 / np.sqrt(K)  # Scale factor

# Create a meshgrid for parameters u and v
u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)

# Hyperboloid equations (upper sheet)
x = np.sinh(u) * np.cos(v) * factor
y = np.sinh(u) * np.sin(v) * factor
z = np.cosh(u) * factor

# Create figure
fig = plt.figure(figsize=(15, 10))


ax = fig.add_subplot(111, projection="3d")

rgb = ls.shade(z, cmap=pastel_cmap, vert_exag=0.5, blend_mode='soft')
surf = ax.plot_surface(
    x, y, z,
    facecolors=rgb,
    alpha=0.5,
    linewidth=0,
    antialiased=True
)


t = np.linspace(-2, 2, 100)

# Geodesic 1: In the xz-plane (y=0) - using a bright, high-contrast color
g1_x = np.sinh(t) * factor
g1_y = np.zeros_like(t)
g1_z = np.cosh(t) * factor

# Geodesic 2: In the yz-plane (x=0) - using another bright, contrasting color
g2_x = np.zeros_like(t)
g2_y = np.sinh(t) * factor
g2_z = np.cosh(t) * factor

ax.plot(g1_x, g1_y, g1_z, color='crimson', linewidth=2.2)
ax.scatter([g1_x[-1]], [g1_y[-1]], [g1_z[-1]], color='crimson', s=30, zorder=10)

ax.plot(g2_x, g2_y, g2_z, color='royalblue', linewidth=2.2)
ax.scatter([g2_x[-1]], [g2_y[-1]], [g2_z[-1]], color='royalblue', s=30, zorder=10)

# Mark the center of the hyperboloid with a distinctive marker
ax.scatter([0], [0], [factor], color='black', s=70, edgecolor='black', linewidth=0.5,
           label='Root feature', zorder=10, path_effects=[pe.withStroke(linewidth=2, foreground="white")])


ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Publication-style labels
ax.set_xlabel('X', labelpad=10)
ax.set_ylabel('Y', labelpad=10)
ax.set_zlabel('Z', labelpad=10)

# Set view limits for a consistent look
ax.set_xlim(-2.5 * factor, 2.5 * factor)
ax.set_ylim(-2.5 * factor, 2.5 * factor)
ax.set_zlim(0, 5 * factor)
# Professional grid styling
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, color="#666666")

# Subtle transparent grid colors for each axis
ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)
ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)
ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.4)

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([0, 1, 2, 3, 4])


plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


ax.view_init(elev=25, azim=45)  # Lower elevation and rotated more for better geodesic visibility
ax.set_box_aspect([1,1,1])

# Move legend to upper left (away from geodesics with new viewing angle)
ax.text(2.5, 0, 6.8*factor, "xz geodesic", color="black", fontsize=18)
ax.text(0.3, 2.5, 6.7*factor, "yz geodesic", color="black", fontsize=18)
#put root feature label above the point and above everyting

ax.text(0, 0, 0.5*factor, "Root feature", color="black", fontsize=18, zorder=20, ha='center')

from matplotlib import cm
from matplotlib.colors import Normalize

# Normalizer: maps Z range to colormap
norm = Normalize(vmin=z.min(), vmax=z.max())

# Add colorbar using the same colormap
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=pastel_cmap), ax=ax, shrink=0.8)
cbar.set_label('Height (Z)', rotation=90, labelpad=10)
cbar.outline.set_visible(False)
cbar.ax.tick_params(size=0, labelsize=10)


# Save with publication-quality settings
plt.savefig('hyperboloid_transparent.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# %%
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import seaborn as sns

# -----------------------------
# Seaborn + custom style
# -----------------------------
sns.set_theme(style="whitegrid", font="serif", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 25,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# -----------------------------
# Colors
# -----------------------------
colors = {
    'equator': 'red',
    'meridian': 'blue',
    'xz_geodesic': '#4daf4a',
    'yz_geodesic': '#ff7f00',
    'equidistant': 'orange',


    'center': 'black'
}

# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(25, 10))


gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

# -----------------------------
# Sphere subplot
# -----------------------------
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
phi = np.linspace(0, 2*np.pi, 120)
theta = np.linspace(0, np.pi, 120)
phi, theta = np.meshgrid(phi, theta)
r_sphere = 1
x_sphere = r_sphere * np.sin(theta) * np.cos(phi)
y_sphere = r_sphere * np.sin(theta) * np.sin(phi)
z_sphere = r_sphere * np.cos(theta)

surf1 = ax1.plot_surface(
    x_sphere, y_sphere, z_sphere, cmap=cm.viridis_r,
    alpha=0.7, linewidth=0, rstride=5, cstride=5, antialiased=True
)

# Geodesics with glow
t = np.linspace(0, 2*np.pi, 200)
for alpha, lw in zip([0.15, 0.35, 0.6, 0.9], [6, 4, 2, 2]):
    ax1.plot(r_sphere*np.cos(t), r_sphere*np.sin(t), np.zeros_like(t),
             color=colors['equator'], alpha=alpha, linewidth=lw)
    ax1.plot(r_sphere*np.sin(t), np.zeros_like(t), r_sphere*np.cos(t),
             color=colors['meridian'], alpha=alpha, linewidth=lw)
ax1.set_xlabel(r'$X$', labelpad=15)
ax1.set_ylabel(r'$Y$', labelpad=15)
ax1.set_zlabel(r'$Z$', labelpad=15)
ax1.set_title('   (a) Sphere in 3D Euclidean Space', pad=5)


ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_zlim(-1.5, 1.5)
ax1.set_box_aspect([1, 1, 1])

# Clean background
for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor('lightgrey')

cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.45, aspect=15, pad=0.08)
cbar1.set_label('Height ($Z$)', rotation=90, labelpad=10, fontsize=16)
cbar1.ax.tick_params(labelsize=12)

# -----------------------------
# Hyperboloid subplot
# -----------------------------
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
K_hyper = 2.3026
factor_hyper = 1/np.sqrt(K_hyper)

# Calculate max u to cut at z = 1.5
z_max = 2
u_max = np.arccosh(z_max / factor_hyper)
u = np.linspace(-u_max, u_max, 120)
v = np.linspace(0, 2*np.pi, 120)
u, v = np.meshgrid(u, v)
x_hyper = np.sinh(u)*np.cos(v)*factor_hyper
y_hyper = np.sinh(u)*np.sin(v)*factor_hyper
z_hyper = np.cosh(u)*factor_hyper -0.5



surf2 = ax2.plot_surface(
    x_hyper, y_hyper, z_hyper, cmap=cm.summer, alpha=0.35, linewidth=0, rstride=5, cstride=5, antialiased=True
)



# Equidistant curve
r_hyper = 1.3
v_circle = np.linspace(0, 2*np.pi, 200)
x_circle = np.sinh(r_hyper)*np.cos(v_circle)*factor_hyper
y_circle = np.sinh(r_hyper)*np.sin(v_circle)*factor_hyper
z_circle = np.cosh(r_hyper)*factor_hyper * np.ones_like(v_circle) - 0.5


z_circle[z_circle > z_max] = z_max 


ax2.plot(x_circle, y_circle, z_circle, color=colors['equidistant'], linewidth=3, zorder=20)

# Axis labels with extra padding
ax2.set_xlabel(r'$X$', labelpad=15)
ax2.set_ylabel(r'$Y$', labelpad=15)
ax2.set_zlabel(r'$Z$', labelpad=15)

# Title with more padding
ax2.set_title('(b) Lorentz Locus of Equidistant Points', pad=5)

ax2.set_xlim(-2.5*factor_hyper, 2.5*factor_hyper)
ax2.set_ylim(-2.5*factor_hyper, 2.5*factor_hyper)
ax2.set_zlim(0, z_max)
ax2.view_init(elev=30, azim=30)
ax2.set_box_aspect([1, 1, 1])

# Clean background
for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor('lightgrey')

cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.45, aspect=15, pad=0.08)


cbar2.set_label('Height ($Z$)', rotation=90, labelpad=10, fontsize=16)
cbar2.ax.tick_params(labelsize=12)

# -----------------------------
# Unified legend
# -----------------------------
legend_elements = [
    plt.Line2D([0], [0], color=colors['equator'], lw=5, label='Equator'),
    plt.Line2D([0], [0], color=colors['meridian'], lw=5, label='Meridian'),
    plt.Line2D([0], [0], color=colors['equidistant'], lw=5, label='Equidistant curve'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02),
           fontsize=18, frameon=False)

# -----------------------------
# Save high-quality PDFs
# -----------------------------
plt.savefig('sphere_hyperboloid_serif_cut.pdf', dpi=600)
plt.savefig('sphere_hyperboloid_serif_cut.png', dpi=600)

plt.show()




# %%
