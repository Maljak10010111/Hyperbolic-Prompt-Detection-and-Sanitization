"""
    This script computes word-level attributions for malicious prompts of ViSU test dataset using LIME and a hyperbolic MLP model.
"""

import json
import torch
from transformers import CLIPTokenizer
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import sys
import os
import re
sys.path.append(os.path.abspath(
    "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.models import HySAC
from Lorentz_and_Poincare_MLP.lorentz_MLP import LorentzMLP
from Lorentz_and_Poincare_MLP.utils.LorentzManifold import LorentzManifold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
dataset_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/visu_testset.json"
output_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/lime_word_attributions_MLP.jsonl"
mlp_model_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/Lorentz_MLP.pt"


tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()
manifold = LorentzManifold(k=2.3026)
mlp = LorentzMLP(manifold=manifold, input_dim=768, hidden_dim=512, use_bias=False).to(DEVICE)
mlp.load_state_dict(torch.load(
    mlp_model_path,
    map_location=DEVICE
))
mlp = mlp.to(DEVICE)
mlp.eval()


def lime_predict(texts):
    input_ids = tokenizer(
        texts,
        return_tensors='pt',
        padding="max_length",
        truncation=True,
        max_length=77
    ).input_ids.to(DEVICE)
    with torch.no_grad():
        embeddings = hyperbolic_clip.encode_text(input_ids, project=True)
        outputs = mlp(embeddings)
        probs = torch.sigmoid(outputs).cpu().numpy()
    return probs.reshape(-1, 1)

explainer = LimeTextExplainer(class_names=['malicious_score'])


with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)


with open(output_path, "w", encoding="utf-8") as outf:
    for pair in tqdm(data):
        nsfw = pair["nsfw"]
        malicious_prompt = re.sub(r'\.? *$', '', nsfw) # removing the dot and space at the end of nsfw prompt
        try:
            exp = explainer.explain_instance(
                malicious_prompt,
                lime_predict,
                num_features=20,
                num_samples=1000,
                labels=[0]
            )
            word_attributions = exp.as_list(label=0)
            word_attributions.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Error for prompt: {nsfw[:50]}... {str(e)}")
            word_attributions = []

        result = {
            "malicious_prompt": malicious_prompt,
            "word_attributions": word_attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")