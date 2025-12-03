"""
This script implements a Word Removal prompt sanitization pipeline to remove harmful intent within prompts.
    The pipeline looks like this:
    1. Remove the top N influential words in the prompt based on provided attributions.
    2. See whether the new prompt with removed words is benign.
    Datasets sanitized:
        - ViSU test dataset
        - SneakyPrompt dataset
        - MMA dataset
"""

import json
import re
import torch
from transformers import CLIPTokenizer
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(
    "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense"
))

from HySAC.hysac.models import HySAC
from HyperbolicSVDD.notebooks.SVDD_th import LorentzHyperbolicOriginSVDD, project_to_lorentz


os.environ["HF_HOME"] = "/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/.cache/huggingface"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()


# ---------- HyPE ---------
class HSVDD:
    def __init__(self):
        self.model = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
        self.model.load("/leonardo_scratch/large/userexternal/imaljkov/Diffusion-Models-Embedding-Space-Defense/models/best_hyperbolic_svdd_model.pth")
        self.model.center = self.model.center.to(DEVICE)
    def predict(self,  prompt):
        input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
        emb = hyperbolic_clip.encode_text(input_ids, project=True)
        emb = project_to_lorentz(emb, self.model.curvature)
        prediction = self.model.predict(emb)
        return prediction


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


def word_removal_run(input_jsonl, output_jsonl, model_predict_fn):
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
            results = {
                "original_prompt": orig_prompt,
            }
            for N in range(1, 6): # removing from top 1 up to top 5 most influential words
                masked_prompt, removed_words = remove_topN_influential_words(orig_prompt, attributions, N)
                m_pred = model_predict_fn(masked_prompt)
                masked_pred = "malicious" if m_pred[0].item() == 0 else "benign" # 0 -> malicious, 1 -> benign
                results[f"masked_prompt_top_{N}"] = masked_prompt
                results[f"removed_words_top_{N}"] = removed_words
                results[f"masked_pred_top_{N}"] = masked_pred
            out_f.write(json.dumps(results, ensure_ascii=False) + "\n")


def main():
    hsvdd_model = HSVDD()
    hsvdd_results = [
        ("visu_lig_SVDD_cleaned_word_attributions.jsonl",        "fidelity_results_lig_SVDD_visu.jsonl"),
        ("mma_lig_SVDD_cleaned_word_attributions.jsonl",         "fidelity_results_lig_SVDD_mma.jsonl"),
        ("nsfw_200_lig_SVDD_cleaned_word_attributions.jsonl",    "fidelity_results_lig_SVDD_nsfw200.jsonl"),
    ]
    for input_jsonl, output_jsonl in hsvdd_results:
        word_removal_run(input_jsonl, output_jsonl, hsvdd_model.predict)


if __name__ == "__main__":
    main()