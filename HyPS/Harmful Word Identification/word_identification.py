"""
    This script computes token and word-level attributions for different harmful prompts using Layer Integrated Gradients (LIG)
    with the Hyperbolic SVDD detector on multiple datasets:
    - ViSU test dataset
    - SneakyPrompt dataset
    - MMA dataset
"""


import json
import os
import sys
import torch
import string
import re
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients
from transformers import CLIPTokenizer
import pandas as pd
import gc

sys.path.append(os.path.abspath(
    "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense"
))

from HySAC.hysac.models import HySAC
from HyperbolicSVDD.SVDD_th import LorentzHyperbolicOriginSVDD, project_to_lorentz


os.environ["HF_HOME"] = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/.cache/huggingface"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATASET PATHS
visu_dataset_path = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/datasets/visu_testset.json"
output_path_word_visu = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_word_attributions_svdd_visu.jsonl"
output_path_token_visu = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_token_attributions_svdd_visu.jsonl"

sneaky_200_dataset_path = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/datasets/nsfw_200.txt"
output_path_word_sneaky = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_word_attributions_svdd_nsfw_200.jsonl"
output_path_token_sneaky = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_token_attributions_svdd_nsfw_200.jsonl"

mma_dataset_path = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/datasets/mma_adjusted.csv"
output_path_word_mma = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_word_attributions_svdd_mma.jsonl"
output_path_token_mma = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/SVDD/results/lig_token_attributions_svdd_mma.jsonl"
# --------------------------------------------------------------------------------------------------------------------------------------------------------------


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()


hsvdd = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
hsvdd.load("/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/models/best_hyperbolic_svdd_model.pth")
hsvdd.center = hsvdd.center.to(DEVICE)


def hsvdd_predict(input_ids):
    emb = hyperbolic_clip.encode_text(input_ids, project=True)
    emb = project_to_lorentz(emb, hsvdd.curvature)
    distance  = hsvdd.predict_xai(emb)
    return distance


embedding_layer = hyperbolic_clip.textual.text_model.embeddings
lig = LayerIntegratedGradients(
    hsvdd_predict,
    embedding_layer
)


def is_punctuation(word):
    return all(char in string.punctuation for char in word)


def get_token_attributions(mal_prompt, baseline):
    input_ids = tokenizer(mal_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    baseline_ids = tokenizer(baseline, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        return_convergence_delta=True,
        n_steps=500
    )
    token_importances = attributions.sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    filtered_scores = [
        (tok, score.item()) for tok, score in zip(tokens, token_importances)
        if tok not in tokenizer.all_special_tokens
    ]
    sorted_tokens = sorted(filtered_scores, key=lambda x: (x[1]), reverse=True)
    return sorted_tokens


def get_word_attributions(mal_prompt, baseline):
    input_ids = tokenizer(mal_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    baseline_ids = tokenizer(baseline, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
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


def load_prompts_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    nsfw = df["NSFW prompts"].dropna().astype(str).tolist()
    sfw = df["SFW prompts"].dropna().astype(str).tolist()
    return nsfw, sfw


# periodic cleanup to free memory
def periodic_cleanup(i, N=100):
    if (i + 1) % N == 0:
        gc.collect()
        torch.cuda.empty_cache()


# function to check if a prompt is classified as malicious by the HSVDD model
def is_malicious(prompt):
    input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    emb = hyperbolic_clip.encode_text(input_ids, project=True)
    emb = project_to_lorentz(emb, hsvdd.curvature)
    pred = hsvdd.predict(emb)
    return pred[0].item() == 0 # return True if "malicious"


""" ------------ ViSU DATASET --------------- """
with open(visu_dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)


#TOKEN-LEVEL ATTRIBUTIONS
with open(output_path_token_visu, "w", encoding="utf-8") as outf:
    for i, prompt in enumerate(tqdm(data, desc="ViSU dataset -> TOKEN-LEVEL ATTRIBUTIONS")):
        nsfw = prompt["nsfw"]
        sfw = prompt["safe"]
        malicious_prompt = re.sub(r'\.? *$', '', nsfw)
        benign_prompt = re.sub(r'\.? *$', '', sfw)
        try:
            attributions = get_token_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "token_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)


# WORD-LEVEL ATTRIBUTIONS
with open(output_path_word_visu, "w", encoding="utf-8") as outf:
    for i, prompt in enumerate(tqdm(data, desc="ViSU dataset -> WORD-LEVEL ATTRIBUTIONS")):
        nsfw = prompt["nsfw"]
        sfw = prompt["safe"]
        malicious_prompt = re.sub(r'\.? *$', '', nsfw)
        benign_prompt = re.sub(r'\.? *$', '', sfw)
        if not is_malicious(malicious_prompt):
            continue
        try:
            attributions = get_word_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "word_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)



""" ----------- SneakyPrompt DATASET --------------------"""
sneakyp_prompts = load_prompts_from_csv(sneaky_200_dataset_path)


#TOKEN-LEVEL ATTRIBUTIONS
with open(output_path_token_sneaky, "w", encoding="utf-8") as outf:
    for i, (malicious_prompt, benign_prompt) in enumerate(tqdm(sneakyp_prompts, desc="SneakyPrompt dataset -> TOKEN-LEVEL ATTRIBUTIONS")):
        benign_prompt = re.sub(r'\.? *$', '', benign_prompt)
        malicious_prompt = re.sub(r'\.? *$', '', malicious_prompt)
        try:
            attributions = get_token_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "token_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)


# WORD-LEVEL ATTRIBUTIONS
with open(output_path_word_sneaky, "w", encoding="utf-8") as outf:
    for i, (malicious_prompt, benign_prompt) in enumerate(tqdm(sneakyp_prompts, desc="SneakyPrompt dataset -> WORD-LEVEL ATTRIBUTIONS")):
        benign_prompt = re.sub(r'\.? *$', '', benign_prompt)
        malicious_prompt = re.sub(r'\.? *$', '', prompt)
        if not is_malicious(malicious_prompt):
            continue
        try:
            attributions = get_word_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "word_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)


""" ----------- MMA DATASET  --------------------"""
mma_prompts = load_prompts_from_csv(mma_dataset_path)


# TOKEN-LEVEL ATTRIBUTIONS
with open(output_path_token_mma, "w", encoding="utf-8") as outf:
    for i, (malicious_prompt, benign_prompt) in enumerate(tqdm(mma_prompts, desc="MMA dataset -> TOKEN-LEVEL ATTRIBUTIONS")):
        benign_prompt = re.sub(r'\.? *$', '', benign_prompt)
        malicious_prompt = re.sub(r'\.? *$', '', malicious_prompt)
        try:
            attributions = get_token_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "token_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)


# WORD-LEVEL ATTRIBUTIONS
with open(output_path_word_mma, "w", encoding="utf-8") as outf:
    for i, (malicious_prompt, benign_prompt) in enumerate(tqdm(mma_prompts, desc="MMA dataset -> WORD-LEVEL ATTRIBUTIONS")):
        benign_prompt = re.sub(r'\.? *$', '', benign_prompt)
        malicious_prompt = re.sub(r'\.? *$', '', malicious_prompt)
        if not is_malicious(malicious_prompt):
            continue
        try:
            attributions = get_word_attributions(malicious_prompt, benign_prompt)
        except Exception as e:
            print("Error!", e)
            attributions = []
        result = {
            "malicious_prompt": malicious_prompt,
            "word_attributions": attributions
        }
        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
        del attributions, result
        periodic_cleanup(i, N=50)