"""
    This script computes token and word-level attributions for malicious prompts of ViSU test dataset using SHAP and a hyperbolic MLP model.
"""

import torch
from transformers import CLIPTokenizer
import sys
import torch.nn as nn
import shap
import numpy as np
import os
import re
import json
from tqdm import tqdm
sys.path.append(os.path.abspath(
    "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.models import HySAC
from Lorentz_and_Poincare_MLP.lorentz_MLP import LorentzMLP
from Lorentz_and_Poincare_MLP.utils.LorentzManifold import LorentzManifold


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
dataset_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/visu_testset.json"
output_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/shap_word_attributions_MLP.jsonl"
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


class FullModel(nn.Module):
    def __init__(self, hyperbolic_clip, mlp, manifold):
        super().__init__()
        self.hyperbolic_clip = hyperbolic_clip
        self.mlp = mlp
        self.manifold = manifold

    def forward(self, input_ids_tensor):
        embedding = self.hyperbolic_clip.encode_text(input_ids_tensor, project=True)
        output = self.mlp(embedding)
        return torch.sigmoid(output)


full_model = FullModel(hyperbolic_clip, mlp, manifold)


def predict_fn(texts):
    input_ids = [tokenizer(t, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids for t in texts]
    input_ids = torch.cat(input_ids, dim=0)
    with torch.no_grad():
        outputs = full_model(input_ids)
        return outputs.cpu().numpy().flatten()


def shap_predict(token_lists):
    texts = [" ".join(toks) for toks in token_lists]
    return predict_fn(texts)


def tokens_to_words(tokens, scores):
    word_attributions = []
    token_buffer = []
    score_buffer = []
    for token, score in zip(tokens, scores):
        if token in ["<|startoftext|>", "<|endoftext|>", "<|endoftext|>", "</w>"]:
            continue
        if token.endswith("</w>"):
            clean_token = token.replace("</w>", "")
            token_buffer.append(clean_token)
            score_buffer.append(score)
            word = "".join(token_buffer)
            word_score = sum(score_buffer)
            if word:  # skip empty
                word_attributions.append((word, word_score))
            token_buffer = []
            score_buffer = []
        else:
            token_buffer.append(token)
            score_buffer.append(score)
    if token_buffer:
        word = "".join(token_buffer)
        word_score = sum(score_buffer)
        if word:
            word_attributions.append((word, word_score))
    return word_attributions


with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)


with open(output_path, "w", encoding="utf-8") as outf:
    for pair in tqdm(data):
        nsfw = pair["nsfw"]
        benign = pair["safe"]
        malicious_prompt = re.sub(r'\.? *$', '', nsfw) # removing the dot and space at the end of nsfw prompt
        benign_prompt = benign
        malicious_tokens = tokenizer.tokenize(malicious_prompt)
        benign_tokens = tokenizer.tokenize(benign_prompt)

        # Pad or crop benign_tokens to match length
        if len(benign_tokens) < len(malicious_tokens):
            benign_tokens += ["<pad>"] * (len(malicious_tokens) - len(benign_tokens))
        elif len(benign_tokens) > len(malicious_tokens):
            benign_tokens = benign_tokens[:len(malicious_tokens)]

        explainer = shap.KernelExplainer(shap_predict, np.array([benign_tokens], dtype=object))
        try:
            shap_values = explainer.shap_values(np.array([malicious_tokens], dtype=object), nsamples=1000)
            word_attributions = tokens_to_words(malicious_tokens, shap_values[0]) # converting tokens to words with their scores
            word_attributions.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Error for prompt: {nsfw[:50]}... {str(e)}")
            word_attributions = []

        result = {
            "malicious_prompt": malicious_prompt,
            "word_attributions": word_attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")