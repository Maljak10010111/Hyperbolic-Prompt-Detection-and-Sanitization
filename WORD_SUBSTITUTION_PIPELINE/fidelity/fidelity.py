"""
    This script computes the FIDELITY (word removal) for LIG, LIME, and SHAP by removing the top N influential words (malicious)
    and checking whether the model's prediction changes, for both MLR and MLP models.
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
    "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.models import HySAC
from LorentzMLR.LMLR import LorentzMLR
from LorentzMLR.classify_embeddings import convert_to_hyperbolic_embedding
from geoopt.manifolds.lorentz import Lorentz
from Lorentz_and_Poincare_MLP.lorentz_MLP import LorentzMLP
from Lorentz_and_Poincare_MLP.utils.LorentzManifold import LorentzManifold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()


# ----------- MLR -----------
class MLRModel:
    def __init__(self):
        self.manifold = Lorentz(k=2.3026)
        self.model = LorentzMLR(manifold=self.manifold, num_features=769, num_classes=1)
        self.model_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/final_hyperbolic_mlr_model.pth"
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
    def predict(self, prompt):
        input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
        embedding = convert_to_hyperbolic_embedding(embedding, self.manifold.k)
        output = self.model(embedding)
        prob = torch.sigmoid(output).item()
        return prob

# ----------- MLP -----------
class MLPModel:
    def __init__(self):
        self.manifold = LorentzManifold(k=2.3026)
        self.model = LorentzMLP(manifold=self.manifold, input_dim=768, hidden_dim=512)
        self.model_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/Lorentz_MLP.pt"
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
    def predict(self, prompt):
        input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
        output = self.model(embedding)
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


def fidelity_run(input_jsonl, output_jsonl, model_predict_fn):
    print(f"\nProcessing {input_jsonl} -> {output_jsonl}")
    with open(input_jsonl, "r", encoding="utf-8") as f, open(output_jsonl, "w", encoding="utf-8") as out_f:
        for line in tqdm(f, desc=f"FIDELITY {input_jsonl}"):
            obj = json.loads(line)
            if "word_attributions" not in obj:
                print(f"Skipping line without 'word_attributions': {line[:80]}...")
                continue
            orig_prompt = obj.get("malicious_prompt", "")
            orig_prompt = re.sub(r'\.? *$', '', orig_prompt)
            attributions = obj["word_attributions"]
            orig_probability = model_predict_fn(orig_prompt)
            orig_pred = "malicious" if orig_probability >= 0.5 else "benign"
            results = {
                "original_prompt": orig_prompt,
                "original_probability": orig_probability,
                "original_pred": orig_pred,
            }
            for N in range(1, 6): # removing from top 1 up to top5 most influential words
                masked_prompt, removed_words = remove_topN_influential_words(orig_prompt, attributions, N)
                masked_score = model_predict_fn(masked_prompt)
                masked_pred = "malicious" if masked_score >= 0.5 else "benign"
                results[f"masked_prompt_top_{N}"] = masked_prompt
                results[f"removed_words_top_{N}"] = removed_words
                results[f"masked_probability_top_{N}"] = masked_score
                results[f"masked_pred_top_{N}"] = masked_pred
            out_f.write(json.dumps(results, ensure_ascii=False) + "\n")


def main():
    # MLR fidelity
    mlr_model = MLRModel()
    mlr_methods = [
        ("lig_MLR_cleaned_word_attributions.jsonl",     "fidelity_results_lig_MLR.jsonl"),
        ("lime_MLR_cleaned_word_attributions.jsonl",    "fidelity_results_lime_MLR.jsonl"),
        ("shap_MLR_cleaned_word_attributions.jsonl",    "fidelity_results_shap_MLR.jsonl"),
    ]
    for input_jsonl, output_jsonl in mlr_methods:
        fidelity_run(input_jsonl, output_jsonl, mlr_model.predict)

    # MLP fidelity
    mlp_model = MLPModel()
    mlp_methods = [
        ("lig_MLP_cleaned_word_attributions.jsonl",     "fidelity_results_lig_MLP.jsonl"),
        ("lime_MLP_cleaned_word_attributions.jsonl",    "fidelity_results_lime_MLP.jsonl"),
        ("shap_MLP_cleaned_word_attributions.jsonl",    "fidelity_results_shap_MLP.jsonl"),
    ]
    for input_jsonl, output_jsonl in mlp_methods:
        fidelity_run(input_jsonl, output_jsonl, mlp_model.predict)


if __name__ == "__main__":
    main()