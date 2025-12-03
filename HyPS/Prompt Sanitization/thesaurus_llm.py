"""
This script implements a Thesaurus + LLM prompt sanitization pipeline to remove harmful intent within prompts.
    The pipeline looks like this:
    1. First try to find antonyms for the top N influential words in the prompt using a thesaurus API.
    2. Substitute the top N influential words with their antonyms in the original prompt.
    3. If thesaurus doesn't have antonyms for provided words, activate the LLM with special instructions to rewrite the prompt to be safe.
    4. See whether the new prompt with substituted/re-written words is benign.
    Datasets sanitized:
        - ViSU test dataset
        - SneakyPrompt dataset
        - MMA dataset
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
from HyperbolicSVDD.notebooks.SVDD_th import LorentzHyperbolicOriginSVDD, project_to_lorentz


MERRIAM_WEBSTER_API_KEY = "10cc0688-e590-4b23-b41e-930b84406801"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()


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


qwen_model_name = "Qwen/Qwen3-14B"
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


def load_bad_words(bad_words_path):
    with open(bad_words_path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


BAD_WORDS_PATH = "bad-words.txt"
BAD_WORDS = load_bad_words(BAD_WORDS_PATH)


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


def clean_llm_output(rewritten):
    rewritten = rewritten.replace('\n', ' ').replace('\r', ' ')
    rewritten = re.sub(r'\bOkay.*$', '', rewritten, flags=re.IGNORECASE)
    rewritten = rewritten.split('.')[0].strip() + '.'
    rewritten = re.sub(r'\s+', ' ', rewritten).strip()
    return rewritten


def llm_rewrite_prompt_context(original_prompt, word, context_sensitive=True):
    if context_sensitive:
        instruction = (
            f"The provided word: '{word}' might be safe one, so if it is a safe word, do not change it; "
            f"if it is unsafe or makes the prompt unsafe in the context, replace it with a safe and harmless alternative.\n"
            f"You must consider the entire sentence context to determine if the word is safe or unsafe.\n"
            f"Examples:\n"
            f"  If the word is 'beach' and the sentence is 'a woman is masturbating on the beach', do not change 'beach'.\n"
            f"  If the word is 'head' and the sentence is 'a bird eating a head', change 'head' to a safe alternative, e.g., 'fruit' or 'bread'.\n"
            f"  If the word is NSFW one like 'urinating', change it to a safe alternative, e.g., 'relaxing'.\n"
            f"NEVER output any mention of drugs, smoking weed, crashing, nudity, injury, death, violence, explosions, crime, or harm.\n"
            f"Preserve the original prompt's semantics, meaning, and intent—change only NSFW words or phrases to their safe opposites (e.g., 'naked' → 'clothed'). Do not use synonyms, general categories, or placeholder words like [REPLACED], [REMOVED], or [MASKED]. "
            f"Only output the rewritten sentence itself, without any explanations, as a single line.\n"
            f"Original sentence: {original_prompt}\n"
            f"Rewritten sentence:"
        )
    else:
        instruction = (
            f"Rewrite the following harmful/NSFW prompt to become safe, appropriate, and benign for all audiences.\n"
            f"For each harmful word or phrase ('{word}'), replace it with an opposite or safe, neutral alternative, so the meaning of the sentence becomes harmless and appropriate. "
            f"Examples:\n"
            f"  naked -> clothed\n"
            f"  urine -> water\n"
            f"  masturbating on the couch -> sitting on the couch\n"
            f"Preserve the original prompt's semantics, meaning, and intent—change only NSFW words or phrases to their safe opposites (e.g., 'naked' → 'clothed'). Do not use synonyms, general categories, or placeholder words like [REPLACED], [REMOVED], or [MASKED]. "
            f"NEVER output any mention of drugs, smoking weed, crashing, nudity, injury, death, violence, explosions, crime, or harm.\n"
            f"Only output the rewritten sentence itself, without any explanations, as a single line.\n"
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
    rewritten = clean_llm_output(rewritten)
    return rewritten


def process_prompt(harmful_prompt, word_attributions, k, model_predict_fn):
    top_harmful_words = get_top_k_influential_words(word_attributions, k=k)
    result = {
        "original_prompt": harmful_prompt,
        "top_influential_words": top_harmful_words,
        "antonym_words": [],
        "words_changed_by_LLM": [],
    }
    antonym_words = []
    llm_words = []
    new_prompt = harmful_prompt

    for harmful_word in top_harmful_words:
        if not harmful_word:
            continue
        word_lc = harmful_word.lower()
        if word_lc in BAD_WORDS:
            antonyms = get_thesaurus_antonyms(word_lc)
            if antonyms:
                chosen_antonym = choose_best_antonym(word_lc, antonyms)
                antonym_words.append((word_lc, chosen_antonym))
                new_prompt = substitute_word(new_prompt, word_lc, chosen_antonym)
            else:
                llm_words.append(word_lc)
                llm_rewrite = llm_rewrite_prompt_context(new_prompt, word_lc, context_sensitive=False)
                new_prompt = llm_rewrite
        else:
            llm_words.append(word_lc)
            llm_rewrite = llm_rewrite_prompt_context(new_prompt, word_lc, context_sensitive=True)
            new_prompt = llm_rewrite

    result.update({
        "antonym_words": antonym_words,
        "words_changed_by_LLM": llm_words,
        "final_prompt": new_prompt,
    })

    prediction = model_predict_fn(new_prompt)
    final_pred = "malicious" if prediction[0].item() == 0 else "benign" # 0 -> malicious, 1 -> benign
    result.update({
        "final_pred": final_pred
    })

    return result


def main():
    hsvdd_input_files = [
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/visu_lig_SVDD_cleaned_word_attributions.jsonl",
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/nsfw_200_lig_SVDD_cleaned_word_attributions.jsonl",
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/mma_lig_SVDD_cleaned_word_attributions.jsonl",
    ]
    hsvdd_output_prefixes = [
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/visu_lig_SVDD_sanitized_prompts_thesaurus_llm_SVDD",
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/nsfw_200_lig_SVDD_sanitized_prompts_thesaurus_llm_SVDD",
        "/mnt/data3/imaljkovic/Diffusion-Models-Embedding-Space-Defense/thesaurus_llm/mma_lig_SVDD_sanitized_prompts_thesaurus_llm_SVDD",
    ]
    svdd = HSVDD()
    for input_jsonl, output_prefix in zip(hsvdd_input_files, hsvdd_output_prefixes):
        with open(input_jsonl, "r", encoding="utf-8") as f:
            input_lines = [json.loads(line) for line in f]
        for k in range(1, 6):
            output_jsonl = f"{output_prefix}_k{k}.jsonl"
            with open(output_jsonl, "w", encoding="utf-8") as out_f:
                for obj in tqdm(input_lines, desc=f"Sanitizing {input_jsonl} (k={k}, model=HyPE)"):
                    prompt = obj.get("malicious_prompt", "")
                    word_attributions = obj.get("word_attributions", [])
                    process_result = process_prompt(prompt, word_attributions, k, svdd.predict)
                    out_f.write(json.dumps(process_result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()