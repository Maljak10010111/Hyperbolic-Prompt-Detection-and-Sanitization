"""
    This script computes word-level attributions for malicious prompts of ViSU test dataset using LIME and a hyperbolic MLR model.
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
    "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.models import HySAC
from LorentzMLR.LMLR import LorentzMLR
from LorentzMLR.classify_embeddings import convert_to_hyperbolic_embedding
from geoopt.manifolds.lorentz import Lorentz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
dataset_path = "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/visu_testset.json"
output_path = "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/lime_word_attributions.jsonl"
mlr_model_path = "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/final_hyperbolic_mlr_model.pth"


tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()
manifold = Lorentz(k=2.3026)
mlr = LorentzMLR(manifold=manifold, num_features=769, num_classes=1)
mlr.load_state_dict(torch.load(
    mlr_model_path,
    map_location=DEVICE
))
mlr = mlr.to(DEVICE)
mlr.eval()


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
        embeddings = convert_to_hyperbolic_embedding(embeddings, manifold.k)
        outputs = mlr(embeddings)
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
            "malicious_prompt": nsfw,
            "word_attributions": word_attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")