# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme and style

file_paths = [
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-original/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-sanitized-TLLM/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-sanitized-TWR/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-semantic-TLLM/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-semantic-TWR/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-WR/results_cos.csv"
]
methods = [
    "Original",
    "Sanitized-TLLM",
    "Sanitized-TWR",
    "Semantic-TLLM",
    "Semantic-TWR",
    "Word Removal"
]

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["font.size"] = 13

# Set seaborn theme and style
sns.set_theme(style="whitegrid", font_scale=1.2)

dfs = []
for i, file in enumerate(file_paths):
    df = pd.read_csv(file)
    df["source_file"] = methods[i]
    if "sd_clip_san" not in df.columns:
        df["sd_clip_san"] = float('nan')
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

if all_data["classification"].astype(str).str.startswith("tensor").any():
    all_data["classification"] = all_data["classification"].astype(str).str.extract(r'tensor\(\[(\d+)\]', expand=False).astype(float)

melted = all_data.melt(
    id_vars=["source_file"],
    value_vars=["sd_clip_mal", "sd_clip_ben", "sd_clip_san"],
    var_name="clip_type",
    value_name="score"
)

# Pastel palette (always the same colors)
box_palette = {
    "sd_clip_mal": "#A3C1DA",  # light pastel blue
    "sd_clip_ben": "#F7CAC9",  # light pastel pink
    "sd_clip_san": "#B6D7A8"   # light pastel green
}

plt.figure(figsize=(17, 7))
sns.boxplot(
    x="source_file",
    y="score",
    hue="clip_type",
    data=melted,
    palette=box_palette,
    linewidth=2.2,
    fliersize=3
)

plt.title("CLIP Score Distributions by Method", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Method", fontsize=15)
plt.ylabel("CLIP Score", fontsize=15)
plt.xticks(fontsize=13, rotation=15)
plt.yticks(fontsize=12)
plt.legend(title="CLIP Type", title_fontsize=14, fontsize=13, loc='lower right', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Set scienceplots style
plt.style.use("science")

# Set seaborn theme and style

file_paths = [
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-original/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-semantic-TLLM/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-semantic-TWR/results_cos.csv",
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/visu-WR/results_cos.csv"
]
methods = [
    "ViSU prompts",
    "Thesaurus+LLM",
    "Thesaurus+WR",
    "Word Removal"
]


dfs = []
for i, file in enumerate(file_paths):
    df = pd.read_csv(file)
    df["source_file"] = methods[i]
    if "sd_clip_san" not in df.columns:
        df["sd_clip_san"] = float('nan')
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

if all_data["classification"].astype(str).str.startswith("tensor").any():
    all_data["classification"] = all_data["classification"].astype(str).str.extract(r'tensor\(\[(\d+)\]', expand=False).astype(float)

melted = all_data.melt(
    id_vars=["source_file"],
    value_vars=["sd_clip_mal"],#, "sd_clip_ben", "sd_clip_san"],
    var_name="clip_type",
    value_name="score"
)

# Pastel palette (always the same colors)
box_palette = {"sd_clip_mal": "#8cafebbd"}#, "sd_clip_ben": "#3498db", "sd_clip_san": "#2ecc71"}

plt.figure(figsize=(16, 12))
sns.boxplot(
    x="source_file",
    y="score",
    hue="clip_type",
    data=melted,
    palette=box_palette,
    linewidth=2.2,
    fliersize=3,
)

plt.xlabel("", fontsize=30)
plt.ylabel("CLIP Score", fontsize=60)
plt.xticks(fontsize=48, rotation=15)
plt.yticks(fontsize=48)
# define legend with different names
new_labels = {
    "sd_clip_mal": "Harmful Prompts",
    # "sd_clip_ben": "Benign",
    # "sd_clip_san": "Sanitized"
}
handles, labels = plt.gca().get_legend_handles_labels()
labels = [new_labels[label] for label in labels]
# add dashed red line on VIsu prompt at the mean of malicious CLIP scores
visu_mal_mean = all_data[all_data["source_file"] == "ViSU prompts"]["sd_clip_mal"].mean()
plt.axhline(y=visu_mal_mean, color="#eb958cbe", linestyle='--', linewidth=2,zorder=0)

# plt.legend(handles, labels, fontsize=45, loc='lower right', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
#disable legend
plt.legend().set_visible(False)
# save the figure
plt.savefig("/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/Image_gen/generated_images/CLIP_score_boxplot_methods.pdf", dpi=600)
plt.show()
# %%

# Compute statistics
stats = all_data.groupby("source_file")[["sd_clip_mal"]].agg(["mean", "std", "min", "max", "median", "count"])

# Reformat for nicer LaTeX output
stats = stats.swaplevel(axis=1)
stats = stats.sort_index(axis=1, level=0)

# Create LaTeX table string
latex_table = stats.to_latex(float_format="%.3f", na_rep="-", multirow=True, column_format="lcccccc")

print("\n--- LaTeX Table for CLIP Score Statistics ---\n")
print(latex_table)

# %%
import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns
#, 'no-latex'
plt.style.use(['science'])

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["font.size"] = 13
# Data
ks = [1, 2, 3, 4, 5]
num_total = 4820

num_benign_fidelity = [1801, 2911, 3499, 3825, 4042]
num_benign_thesaurus = [1920, 2796, 3254, 3508, 3657]
num_benign_thesaurus_llm = [1923, 2573, 2877, 3065, 3149]

percent_fidelity = [n / num_total * 100 for n in num_benign_fidelity]
percent_thesaurus = [n / num_total * 100 for n in num_benign_thesaurus]
percent_thesaurus_llm = [n / num_total * 100 for n in num_benign_thesaurus_llm]

sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")
markers = ['o', 's', '^']
labels = ['Word Removal', 'Thesaurus + Word Removal', 'Thesaurus + LLM']

plt.figure(figsize=(16, 12))  # Single, wide figure
# plt.figure(figsize = (23, 18))

plt.plot(ks, percent_fidelity, label=labels[0], marker=markers[0],
         markersize=35, markerfacecolor=palette[0], markeredgecolor='black', markeredgewidth=1.5, color=palette[0], linewidth=2)
plt.plot(ks, percent_thesaurus, label=labels[1], marker=markers[1],
         markersize=35, markerfacecolor=palette[1], markeredgecolor='black', markeredgewidth=1.5, color=palette[1], linewidth=2)
plt.plot(ks, percent_thesaurus_llm, label=labels[2], marker=markers[2],
         markersize=35, markerfacecolor=palette[2], markeredgecolor='black', markeredgewidth=1.5, color=palette[2], linewidth=2)


plt.xlabel('Harmful Words Substituted', fontsize=50, fontweight='bold', family='Latin Modern Roman')
plt.ylabel('Percentage of Neutralized Prompts', fontsize=50, fontweight='bold', family='Latin Modern Roman')
plt.yticks(fontsize=50)
plt.xticks(fontsize=50)
plt.legend(fontsize=43, frameon=True, loc='lower right')
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(ks)


plt.tight_layout()
plt.savefig("sanitization_results_curve3.pdf",  dpi=600,  bbox_inches='tight', pad_inches=0.05)
plt.close()
# %%
