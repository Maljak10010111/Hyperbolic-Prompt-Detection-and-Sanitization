"""
    This script computes the FIDELITY for LIG, LIG with pad baseline, LIME, and SHAP
    by removing the top N influential words (malicious) and checking whether the model's prediction changes.
    Each method's attribution file is loaded in sequence and fidelity results are saved to corresponding outputs.
"""

import json
import re
import torch
from transformers import CLIPTokenizer
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(
    "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.models import HySAC
from LorentzMLR.LMLR import LorentzMLR
from LorentzMLR.classify_embeddings import convert_to_hyperbolic_embedding
from geoopt.manifolds.lorentz import Lorentz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()
manifold = Lorentz(k=2.3026)
mlr = LorentzMLR(manifold=manifold, num_features=769, num_classes=1)
mlr_model_path = "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense/Lorentz_and_Poincare_MLP/models/final_hyperbolic_mlr_model.pth"
mlr.load_state_dict(torch.load(
    mlr_model_path,
    map_location=DEVICE
))
mlr = mlr.to(DEVICE)
mlr.eval()

def predict_malicious(prompt):
    input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
    embedding = hyperbolic_clip.encode_text(input_ids, project=True)
    embedding = convert_to_hyperbolic_embedding(embedding, manifold.k)
    output = mlr(embedding)
    prob = torch.sigmoid(output).item()
    return prob

def remove_topN_influential_words(prompt, attributions, N):
    positive_attributions = [t for t in attributions if t[1] > 0]
    words_to_remove = [t[0] for t in positive_attributions[:N]]
    prompt_masked = prompt
    for word in words_to_remove:
        if len(word) == 1 and not word.isalnum():
            prompt_masked = prompt_masked.replace(word, "")
        else:
            pattern = re.compile(r'\b{}\b'.format(re.escape(word)), flags=re.IGNORECASE)
            prompt_masked = pattern.sub('', prompt_masked)
    prompt_masked = re.sub(r'\s+', ' ', prompt_masked).strip()
    return prompt_masked, words_to_remove

METHODS = [
    ("lig_word_attributions.jsonl",     "fidelity_results_lig.jsonl"),
    ("lig_word_attributions_pad_baseline.jsonl", "fidelity_results_lig_pad_baseline.jsonl"),
    ("lime_word_attributions.jsonl",    "fidelity_results_lime.jsonl"),
    ("shap_word_attributions.jsonl",    "fidelity_results_shap.jsonl"),
]

for input_jsonl, output_jsonl in METHODS:
    print(f"\nProcessing {input_jsonl} -> {output_jsonl}")
    with open(input_jsonl, "r", encoding="utf-8") as f, open(output_jsonl, "w", encoding="utf-8") as out_f:
        for line in tqdm(f, desc=f"FIDELITY {input_jsonl}"):
            obj = json.loads(line)
            if "word_attributions" not in obj:
                print(f"Skipping line without 'word_attributions': {line[:80]}...")
                continue
            orig_prompt = obj.get("malicious_prompt", "")
            attributions = obj["word_attributions"]
            orig_probability = predict_malicious(orig_prompt)
            orig_pred = "malicious" if orig_probability >= 0.5 else "benign"
            results = {
                "original_prompt": orig_prompt,
                "original_probability": orig_probability,
                "original_pred": orig_pred,
            }
            positive_attributions = [t for t in attributions if t[1] > 0] # only consider words with positive attribution scores
            for N in range(1, 6): # removing top 1 to 5 influential words
                masked_prompt, removed_words = remove_topN_influential_words(orig_prompt, attributions, N)
                masked_score = predict_malicious(masked_prompt)
                masked_pred = "malicious" if masked_score >= 0.5 else "benign"
                results[f"masked_prompt_top_{N}"] = masked_prompt
                results[f"removed_words_top_{N}"] = removed_words
                results[f"masked_probability_top_{N}"] = masked_score
                results[f"masked_pred_top_{N}"] = masked_pred
            out_f.write(json.dumps(results, ensure_ascii=False) + "\n")