"""
This script processes prompts to replace malicious words with benign alternatives.
    The pipeline looks like this:
    1. First try to find antonyms for the top N influential words in the prompt using a thesaurus API.
    2. Substitute the top N influential words with their antonyms in the original prompt.
    3. If thesaurus doesn't have antonyms for provided words, discard them.
    4. See whether the new prompt with substituted/discarded words is benign.
    5. We apply this pipeline to two models: MLR and MLP.
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
from LorentzMLR.LMLR import LorentzMLR
from LorentzMLR.classify_embeddings import convert_to_hyperbolic_embedding
from geoopt.manifolds.lorentz import Lorentz
from LorentzMLP.LorentzMLP_train import LorentzMLP
from LorentzMLP.utils.LorentzManifold import LorentzManifold

MERRIAM_WEBSTER_API_KEY = "10cc0688-e590-4b23-b41e-930b84406801"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
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
        input_ids = clip_tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True,
                                   max_length=77).input_ids.to(DEVICE)
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
        embedding = convert_to_hyperbolic_embedding(embedding, self.manifold.k)
        output = self.model(embedding)
        prob = torch.sigmoid(output).item()
        return prob


# ----------- MLP -----------
class MLPModel:
    def __init__(self):
        self.manifold = LorentzManifold(k=2.3026)
        self.model = LorentzMLP(manifold=self.manifold, input_dim=768, hidden_dim=512, use_bias=False).to(DEVICE)
        self.model_path = "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/models/LorentzMLP.pt"
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def predict(self, prompt):
        input_ids = clip_tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True,
                                   max_length=77).input_ids.to(DEVICE)
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
        output = self.model(embedding)
        prob = torch.sigmoid(output).item()
        return prob


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


"""
choose_best_antonym_in_context:
For each antonym, substitute it into the entire prompt (replacing the NSFW word).
Encode the whole prompt using CLIP.
Compute similarity between the original prompt embedding and each substituted-prompt embedding.
Pick the antonym whose substituted-prompt embedding has the lowest cosine similarity (i.e., pointing in the most opposite direction) to the original prompt.
"""
def choose_best_antonym_in_context(prompt, nsfw_word, antonyms):
    context_sentences = [substitute_word(prompt, nsfw_word, ant) for ant in antonyms]
    orig_emb = get_clip_embedding(prompt)
    ant_embs = [get_clip_embedding(sent) for sent in context_sentences]
    similarities = [cosine_similarity(orig_emb, ant_emb) for ant_emb in ant_embs]
    best_idx = similarities.index(min(similarities))
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
    top_harmful_words = get_top_k_influential_words(word_attributions, k=k)
    result = {
        "original_prompt": harmful_prompt,
        "top_influential_words": top_harmful_words,
    }
    antonyms_list = []
    for harmful_word in top_harmful_words:
        antonyms = get_thesaurus_antonyms(harmful_word) if harmful_word else []
        if antonyms:
            chosen_antonym = choose_best_antonym_in_context(harmful_prompt, harmful_word, antonyms)
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
    antonym_score = model_predict_fn(antonym_prompt)
    antonym_pred = "malicious" if antonym_score >= 0.5 else "benign"
    result.update({
        "antonym_words": antonyms_list,
        "antonym_prompt": antonym_prompt,
        "antonym_score": antonym_score,
        "antonym_pred": antonym_pred
    })
    return result


def main():
    # ----------- MLR Run -----------
    mlr_input_files = [
        "lig_MLR_cleaned_word_attributions.jsonl",
        "lime_MLR_cleaned_word_attributions.jsonl",
        "shap_MLR_cleaned_word_attributions.jsonl",
    ]
    mlr_output_prefixes = [
        "sanitized_prompts_thesaurus_MLR_lig",
        "sanitized_prompts_thesaurus_MLR_lime",
        "sanitized_prompts_thesaurus_MLR_shap",
    ]
    mlr_model = MLRModel()
    for input_jsonl, output_prefix in zip(mlr_input_files, mlr_output_prefixes):
        with open(input_jsonl, "r", encoding="utf-8") as f:
            input_lines = [json.loads(line) for line in f]
        for k in range(1, 6):
            output_jsonl = f"{output_prefix}_k{k}.jsonl"
            with open(output_jsonl, "w", encoding="utf-8") as out_f:
                for obj in tqdm(input_lines, desc=f"Sanitizing {input_jsonl} (k={k}, model=MLR)"):
                    prompt = obj.get("malicious_prompt", "")
                    word_attributions = obj.get("word_attributions", [])
                    result = process_prompt(prompt, word_attributions, k, mlr_model.predict)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    # ----------- MLP Run -----------
    mlp_input_files = [
        "lig_MLP_cleaned_word_attributions.jsonl",
        "lime_MLP_cleaned_word_attributions.jsonl",
        "shap_MLP_cleaned_word_attributions.jsonl",
    ]
    mlp_output_prefixes = [
        "sanitized_prompts_thesaurus_MLP_lig",
        "sanitized_prompts_thesaurus_MLP_lime",
        "sanitized_prompts_thesaurus_MLP_shap",
    ]
    mlp_model = MLPModel()
    for input_jsonl, output_prefix in zip(mlp_input_files, mlp_output_prefixes):
        with open(input_jsonl, "r", encoding="utf-8") as f:
            input_lines = [json.loads(line) for line in f]
        for k in range(1, 6):
            output_jsonl = f"{output_prefix}_k{k}.jsonl"
            with open(output_jsonl, "w", encoding="utf-8") as out_f:
                for obj in tqdm(input_lines, desc=f"Sanitizing {input_jsonl} (k={k}, model=MLP)"):
                    prompt = obj.get("malicious_prompt", "")
                    word_attributions = obj.get("word_attributions", [])
                    result = process_prompt(prompt, word_attributions, k, mlp_model.predict)
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()