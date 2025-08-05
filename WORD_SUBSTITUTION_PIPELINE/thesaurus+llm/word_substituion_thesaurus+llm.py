"""
This script processes prompts to replace malicious words with benign alternatives.
    The pipeline looks like this:
    1. First try to find antonyms for the top N influential words in the prompt using a thesaurus API.
    2. Substitute the top N influential words with their antonyms in the original prompt.
    3. If thesaurus doesn't have antonyms for provided words, use LLM for word substitution.
    4. See whether the new prompt with substituted words is benign.
    5. Check if the new prompt is benign according to two models: MLR and MLP.
"""

import torch
from transformers import CLIPTokenizer, AutoTokenizer, AutoModelForCausalLM
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
from Lorentz_and_Poincare_MLP.lorentz_MLP import LorentzMLP
from Lorentz_and_Poincare_MLP.utils.LorentzManifold import LorentzManifold

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
        input_ids = clip_tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
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
        input_ids = clip_tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
        embedding = hyperbolic_clip.encode_text(input_ids, project=True)
        output = self.model(embedding)
        prob = torch.sigmoid(output).item()
        return prob


qwen_model_name = "Qwen/Qwen2.5-32B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(
    qwen_model_name,
    token="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"
)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    device_map="auto",
    torch_dtype="auto",
    token="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"
)


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
    input_ids = clip_tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=77).input_ids.to(DEVICE)
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


def get_first_sentence(text):
    text = text.strip()
    if '.' in text:
        return text.split('.', 1)[0].strip() + '.'
    return text


def llm_rewrite_prompt(original_prompt, top_words):
    words_str = "', '".join(top_words)
    instruction = (
        f"Rewrite the following sentence to ensure it is safe, appropriate, and benign for all audiences.\n"
        f"Primarily, replace only the harmful words ('{words_str}') with suitable alternatives. If simply replacing these words is not sufficient to ensure safety or natural language flow, you may also modify the surrounding context as needed.\n"
        f"The rewritten sentence must preserve the original sentence's structure, logic, and core semantic meaning as closely as possible.\n"
        f"Limit your output to a maximum of 70 tokens.\n"
        f"Only output the rewritten sentence itself, without any explanations, headers, or additional information.\n"
        f"Original sentence: {original_prompt}\n"
        f"Rewritten sentence:"
    )
    inputs = qwen_tokenizer(instruction, return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=70,
            do_sample=True,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    rewritten = qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Rewritten sentence:" in rewritten:
        rewritten = rewritten.split("Rewritten sentence:")[-1].strip()
    rewritten = get_first_sentence(rewritten)
    return rewritten



def process_prompt(prompt, word_attributions, k, model_predict_fn):
    """
    For top-k influential words in the prompt, try to substitute each with its best antonym from the thesaurus.
    If a word is not found in the thesaurus, use Qwen2-32B to rewrite the prompt, changing only that word (and its surroundings if needed).
    The prompt structure must be kept as much as possible.

    If LLM is used, the output dictionary contains llm_prompt, llm_score, and llm_pred.
    Otherwise, it contains antonym_words, antonym_prompt, antonym_score, and antonym_pred.
    """
    top_words = get_top_k_influential_words(word_attributions, k=k)
    result = {
        "original_prompt": prompt,
        "top_influential_words": top_words,
        "antonym_words": []
    }
    antonyms_list = []
    LLM_USED = False
    substituted_prompt = prompt

    for word in top_words:
        antonyms = get_thesaurus_antonyms(word) if word else []
        chosen_antonym = None

        if antonyms:
            chosen_antonym = choose_best_antonym_in_context(substituted_prompt, word, antonyms)
            substituted_prompt = substitute_word(substituted_prompt, word, chosen_antonym)
        else:
            LLM_USED = True
            llm_rewrite = llm_rewrite_prompt(substituted_prompt, [word])
            substituted_prompt = llm_rewrite  # update prompt for subsequent words

        antonyms_list.append(chosen_antonym)

    if LLM_USED:
        llm_prompt = get_first_sentence(substituted_prompt)
        llm_score = model_predict_fn(llm_prompt)
        llm_pred = "malicious" if llm_score >= 0.5 else "benign"
        result.update({
            "llm_prompt": llm_prompt,
            "llm_score": llm_score,
            "llm_pred": llm_pred
        })
    else:
        antonym_prompt = substituted_prompt
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
        "sanitized_prompts_thesaurus_llm_MLR_lig",
        "sanitized_prompts_thesaurus_llm_MLR_lime",
        "sanitized_prompts_thesaurus_llm_MLR_shap",
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
        "sanitized_prompts_thesaurus_llm_MLP_lig",
        "sanitized_prompts_thesaurus_llm_MLP_lime",
        "sanitized_prompts_thesaurus_llm_MLP_shap",
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