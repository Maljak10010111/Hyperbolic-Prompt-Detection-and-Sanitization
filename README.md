# Diffusion-Models-Embedding-Space-Defense

This repository explores how to use diffusion models and CLIP-based embeddings for defending against adversarial content generation, particularly in sensitive domains.

---

## 🔧 Setup

Before running any scripts in this repository, you must first set up the environment by following the instructions in the [HySAC](./HySAC/README.md) package.

### Step 1: Run HySAC Setup

```bash
cd ../HySAC
```
Follow all instructions in README.md to install dependencies and prepare the model

## 🧠 Embedding Extraction
After setting up HySAC, run the Hyperclip-inference.py script to generate embeddings from selected datasets using a CLIP model (HySAC or other).
```
python Hyperclip-inference.py \
  --datasets i2p mscoco mma \
  --model_id aimagelab/hysac \
  --clip_backbone openai/clip-vit-large-patch14 \
  --device_id 0 \
  --output_file embeddings.pkl
```

## 📊 Visualization
After generating the embeddings, you can visualize and analyze them using the provided notebook in the plots/ directory.

### Step 1: Navigate to the Plots Folder
```bash
cd plots/
```
### Step 2: Open and Run the Notebook
Open the notebook (e.g., plot_embeddings.ipynb) in Jupyter Notebook or Visual Studio Code. Run all cells to visualize the embedding space and explore how categories such as "sex", "violence", or "porn" are represented.