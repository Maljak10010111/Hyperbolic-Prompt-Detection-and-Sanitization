import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from transformers import CLIPTokenizer, CLIPModel
from HySAC.hysac.utils.distributed import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VISU test embeddings using original CLIP")
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="clip_embeddings")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument('--dataset', type=str, default="4chain", choices=["mma", "4chain", 'lexica', 'pattern'])
    return parser.parse_args()


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def process_batch_clip(prompts, tokenizer, model, device):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    embeddings = model.get_text_features(**inputs)
    return embeddings.cpu()

def parse_mma_dataset():
    mma_path = '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/mma-diffusion-nsfw-adv-prompts.csv'
    # Load the dataset from the CSV file - malicious prompts
    df = pd.read_csv(mma_path)
    prompts = df['target_prompt'].tolist()
    # Load the dataset from the CSV file - benign prompts
    benign_prompts = df['clean_prompt'].tolist()
    # merge into 1 single list
    prompts.extend(benign_prompts)
    return prompts
def parse_4_chain_dataset():
    chain_path = '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/4chain.csv'

    df = pd.read_csv(chain_path)
    prompts = df['target_prompts'].tolist()
    categories = ['malicious'] * len(prompts)
    # Load the dataset from the CSV file - benign prompts
    benign_prompts = df['sanitized_prompts'].tolist()
    # merge into 1 single list
    prompts.extend(benign_prompts)
    categories.extend(['benign'] * len(benign_prompts))
    return prompts#, categories

def process_dataset(args, dataset_name, tokenizer, model, device):
    dataset_digest = {
        'mma': parse_mma_dataset,
        '4chain': parse_4_chain_dataset,
        'lexica': 'Lexica Dataset',
        'pattern': 'Pattern Dataset'
    }
    split_dir = os.path.join(args.cache_dir, f"_clip_embeddings_{dataset_name}")
    ensure_directory(split_dir)

    full_embeddings_file = os.path.join(split_dir, f"full_clip_embeddings.pt")

    if os.path.exists(full_embeddings_file) and not args.force_recompute:
        print(f"Embeddings already exist at: {full_embeddings_file} — skipping.")
        return
    all_prompts = dataset_digest[dataset_name]()
    print(f"Found {len(all_prompts)} prompts in {dataset_name} dataset.")
    batch_size = 128
    all_embeddings = []

    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
        batch_embeddings = process_batch_clip(batch_prompts, tokenizer, model, device)
        all_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, full_embeddings_file)
    print(f"Saved embeddings to {full_embeddings_file}")

def main():
    args = parse_args()
    load_dotenv()

    device = get_device(args.device_id)
    print(f"\nExtracting embeddings using original CLIP on device {device}")
    print(f"Model: {args.model_id}, Output Dir: {args.cache_dir}")

    ensure_directory(args.cache_dir)

    model = CLIPModel.from_pretrained(args.model_id).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id)

    process_dataset(args, args.dataset, tokenizer, model, device)

    print("\nFinished extracting test embeddings.")


if __name__ == "__main__":
    main()
