"""
    This script computes word-level attributions for malicious prompts of ViSU test dataset using Layer Integrated Gradients (LIG) with a hyperbolic MLR model.
"""

import json
from transformers import CLIPTokenizer
import torch
from captum.attr import LayerIntegratedGradients
import os
import sys
from tqdm import tqdm
import string
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
output_path = "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense/XAI/xai/lig_word_attributions.jsonl"
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

embedding_layer = hyperbolic_clip.textual.text_model.embeddings
lig = LayerIntegratedGradients(
    lambda input_ids: torch.sigmoid(
        mlr(convert_to_hyperbolic_embedding(
            hyperbolic_clip.encode_text(input_ids, project=True), manifold.k
        ))
    ),
    embedding_layer
)

def is_punctuation(word):
    return all(char in string.punctuation for char in word)

def get_token_attributions(mal_prompt, benign_prompt):
    input_ids = tokenizer(mal_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    baseline_ids = tokenizer(benign_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        return_convergence_delta=True,
        n_steps=500
    )
    token_importances = attributions.sum(dim=-1).squeeze(0) # [77]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    filtered_scores = [
        (tok, score.item()) for tok, score in zip(tokens, token_importances)
        if tok not in tokenizer.all_special_tokens
    ]
    sorted_tokens = sorted(filtered_scores, key=lambda x: (x[1]), reverse=True)
    return sorted_tokens

def get_word_attributions(mal_prompt, benign_prompt):
    input_ids = tokenizer(mal_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    baseline_ids = tokenizer(benign_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        return_convergence_delta=True,
        n_steps=500
    )
    token_importances = attributions.sum(dim=-1).squeeze(0) # [77]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    word_attributions = []
    token_buffer = []
    score_buffer = []

    for token, score in zip(tokens, token_importances):
        if token in ["<|startoftext|>", "<|endoftext|>", "</w>"]:
            continue
        if token.endswith("</w>"):
            clean_token = token.replace("</w>", "")
            token_buffer.append(clean_token)
            score_buffer.append(score.item())

            word = "".join(token_buffer)
            word_score = sum(score_buffer)

            # Skip empty words and punctuations
            if word and not is_punctuation(word):
                word_attributions.append((word, word_score))

            token_buffer = []
            score_buffer = []
        else:
            token_buffer.append(token)
            score_buffer.append(score.item())

    if token_buffer:
        word = "".join(token_buffer)
        word_score = sum(score_buffer)
        if word and not is_punctuation(word):
            word_attributions.append((word, word_score))

    word_attributions.sort(key=lambda x: (x[1]), reverse=True)
    return word_attributions

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

""" TOKEN-LEVEL ATTRIBUTION SCORES """
# with open(output_path, "w", encoding="utf-8") as outf:
#     for pair in tqdm(data):
#         safe = pair["safe"]
#         nsfw = pair["nsfw"]
#         malicious_prompt = re.sub(r'\.? *$', '', nsfw)
#         try:
#             attributions = get_token_attributions(malicious_prompt, safe)
#         except Exception as e:
#             print("Error!")
#             attributions = []
#         result = {
#             "malicious_prompt": nsfw,
#             "token_attributions": attributions
#         }
#         outf.write(json.dumps(result, ensure_ascii=False) + "\n")


""" WORD-LEVEL ATTRIBUTION SCORES """
with open(output_path, "w", encoding="utf-8") as outf:
    for pair in tqdm(data):
        safe = pair["safe"]
        nsfw = pair["nsfw"]
        malicious_prompt = re.sub(r'\.? *$', '', nsfw)
        try:
            attributions = get_word_attributions(malicious_prompt, safe)
        except Exception as e:
            print("Error!")
            attributions = []
        result = {
            "malicious_prompt": nsfw,
            "word_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")