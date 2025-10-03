"""
This script implements a Thesaurus + Word Removal prompt sanitization pipeline to remove harmful intent within prompts.
    The pipeline looks like this:
    1. First try to find antonyms for the top N influential words in the prompt using a thesaurus API.
    2. Substitute the top N influential words with their antonyms in the original prompt.
    3. If thesaurus doesn't have antonyms for provided words, discard them.
    4. See whether the new prompt with substituted/discarded words is benign.
    Datasets sanitized:
        - ViSU test dataset
        - SneakyPrompt dataset
        - MMA dataset
"""


import torch
from transformers import CLIPTokenizer
import sys
import os
import re
import json
from tqdm import tqdm
import requests

sys.path.append(os.path.abspath(
    "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))

from HySAC.hysac.models import HySAC
from HyperbolicSVDD.SVDD_th import LorentzHyperbolicOriginSVDD, project_to_lorentz


MERRIAM_WEBSTER_API_KEY = "10cc0688-e590-4b23-b41e-930b84406801"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()


# ---------- HyPE ---------
class HSVDD:
    def __init__(self):
        self.model = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
        self.model.load("/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/best_hyperbolic_svdd_model.pth")
        self.model.center = self.model.center.to(DEVICE)
    def predict(self,  prompt):
        input_ids = clip_tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
        emb = hyperbolic_clip.encode_text(input_ids, project=True)
        emb = project_to_lorentz(emb, self.model.curvature)
        prediction = self.model.predict(emb)
        return prediction


def get_thesaurus_antonyms(word, api_key=MERRIAM_WEBSTER_API_KEY):
    url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching antonyms for '{word}': {e}")
        return []
    antonyms = []
    for entry in data:
        if isinstance(entry, dict) and 'meta' in entry and 'ants' in entry['meta']:
            for ants_list in entry['meta']['ants']:
                antonyms.extend(ants_list)
    antonyms = list(dict.fromkeys([ant.replace("_", " ") for ant in antonyms]))
    return antonyms


def get_clip_embedding(text):
    input_ids = clip_tokenizer(text, return_tensors='pt', padding="max_length", truncation=True,
                               max_length=77).input_ids.to(DEVICE)
    with torch.no_grad():
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
    return embedding[0]


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def choose_best_antonym(nsfw_word, antonyms):
    """
        For each antonym, encode it using CLIP.
        Compute similarity between the NSFW word embedding and each antonym embedding.
        Pick the antonym whose embedding has the highest cosine similarity to the NSFW word.
    """
    nsfw_emb = get_clip_embedding(nsfw_word)
    antonym_embs = [get_clip_embedding(antonym) for antonym in antonyms]
    similarities = [cosine_similarity(nsfw_emb, antonym_emb) for antonym_emb in antonym_embs]
    best_idx = similarities.index(max(similarities))
    return antonyms[best_idx]


def substitute_word(prompt, orig_word, new_word):
    pattern = re.compile(r'\b{}\b'.format(re.escape(orig_word)), flags=re.IGNORECASE)
    substituted = pattern.sub(new_word, prompt)
    substituted = re.sub(r'\s+', ' ', substituted).strip()
    return substituted


def get_top_k_influential_words(word_attributions, k=1):
    attributions = [t for t in word_attributions if t[1] > 0]
    if not attributions:
        return []
    attributions.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in attributions[:k]]


def substitute_multiple_words(prompt, top_words, antonyms_list):
    substituted_prompt = prompt
    for word, antonym in zip(top_words, antonyms_list):
        if antonym:
            substituted_prompt = substitute_word(substituted_prompt, word, antonym)
    return substituted_prompt


def remove_word(text, word):
    pattern = r'\b{}\b[,.!?;:]*\s*'.format(re.escape(word))
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    return text


def process_prompt(harmful_prompt, word_attributions, k, model_predict_fn):
    """
       For top-k influential words in the prompt, try to substitute each with its best antonym from the thesaurus.
       If a word is not found in the thesaurus, remove it from the prompt.
    """
    top_harmful_words = get_top_k_influential_words(word_attributions, k=k)
    result = {
        "original_prompt": harmful_prompt,
        "top_influential_words": top_harmful_words,
    }
    antonyms_list = []
    for harmful_word in top_harmful_words:
        antonyms = get_thesaurus_antonyms(harmful_word) if harmful_word else []
        if antonyms:
            chosen_antonym = choose_best_antonym(harmful_word, antonyms)
        else:
            chosen_antonym = None
        antonyms_list.append(chosen_antonym)
    # build the new prompt: substitute with antonym if available, else remove word
    new_prompt = harmful_prompt
    for harmful_word, antonym_word in zip(top_harmful_words, antonyms_list):
        if antonym_word:
            new_prompt = substitute_word(new_prompt, harmful_word, antonym_word)
        else:
            new_prompt = remove_word(new_prompt, harmful_word)
    antonym_prompt = new_prompt
    pred = model_predict_fn(antonym_prompt)
    antonym_pred = "malicious" if pred[0].item() == 0 else "benign"
    result.update({
        "antonym_words": antonyms_list,
        "antonym_prompt": antonym_prompt,
        "antonym_pred": antonym_pred
    })
    return result


def main():
    hsvdd_input_files = [
        "visu_lig_SVDD_cleaned_word_attributions.jsonl",
        "mma_lig_SVDD_cleaned_word_attributions.jsonl",
        "sneakyprompt_lig_SVDD_cleaned_word_attributions.jsonl",
    ]
    hsvdd_output_prefixes = [
        "sanitized_prompts_thesaurus_SVDD_lig_visu",
        "sanitized_prompts_thesaurus_SVDD_lig_mma",
        "sanitized_prompts_thesaurus_SVDD_lig_nsfw200",
    ]
    model = HSVDD()
    for input_jsonl, output_prefix in zip(hsvdd_input_files, hsvdd_output_prefixes):
        with open(input_jsonl, "r", encoding="utf-8") as f:
            input_lines = [json.loads(line) for line in f]
        for k in range(1, 6):
            output_jsonl = f"{output_prefix}_k{k}.jsonl"
            with open(output_jsonl, "w", encoding="utf-8") as out_f:
                for obj in tqdm(input_lines, desc=f"Sanitizing {input_jsonl} (k={k}, model=SVDD)"):
                    prompt = obj.get("malicious_prompt", "")
                    word_attributions = obj.get("word_attributions", [])
                    pred = model.predict(prompt)
                    orig_pred = "malicious" if pred[0].item() == 0 else "benign" # 0 -> malicious, 1 -> benign
                    if orig_pred == "malicious":
                        result = {
                            "original_prompt": prompt,
                            "original_pred": orig_pred
                        }
                        process_result = process_prompt(prompt, word_attributions, k, model.predict)
                        result.update(process_result)
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()