import os
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC as HySAC_model
from HySAC.hysac.dataset.utils import get_dataloader_and_dataset
from HySAC.hysac.dataset.datasetsEnum import DatasetName
from HySAC.hysac.utils.distributed import get_device
from HySAC.hysac.utils.embedder import process_batch_embeddings
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
    parser.add_argument("--model_id", type=str, default="aimagelab/hysac")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="clip_embeddings")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument('--dataset', type=str, default="mma", choices=["mma", "4chain", 'lexica', 'pattern'])
    return parser.parse_args()


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def parse_mma_dataset():
    mma_path = '/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/mma-diffusion-nsfw-adv-prompts.csv'
    # Load the dataset from the CSV file - malicious prompts
    df = pd.read_csv(mma_path)
    prompts = df['target_prompt'].tolist()
    categories = ['malicious'] * len(prompts)
    # Load the dataset from the CSV file - benign prompts
    benign_prompts = df['clean_prompt'].tolist()
    # merge into 1 single list
    prompts.extend(benign_prompts)
    categories.extend(['benign'] * len(benign_prompts))
    return prompts, categories

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
    return prompts, categories

def process_dataset(args, dataset_name, tokenizer, model, device):
    dataset_digest = {
        'mma': parse_mma_dataset,
        '4chain': parse_4_chain_dataset,
        #'lexica': parse_lexica_dataset,
        #'pattern': parse_pattern_dataset
    }
    split_dir = os.path.join(args.cache_dir, f"_hyperclip_embeddings_{dataset_name}")
    ensure_directory(split_dir)

    full_embeddings_file = os.path.join(split_dir, f"full_hyperclip_embeddings.pt")

    if os.path.exists(full_embeddings_file) and not args.force_recompute:
        print(f"Embeddings already exist at: {full_embeddings_file} — skipping.")
        return
    all_prompts, all_categories = dataset_digest[dataset_name]()

    print(f"Found {len(all_prompts)} prompts in {dataset_name} dataset.")
    all_embeddings = []
    batch_size = 128

    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_cache_file = os.path.join(split_dir, f"batch_{batch_idx}.pt")

        if os.path.exists(batch_cache_file) and not args.force_recompute:
            batch_embeddings = torch.load(batch_cache_file)
            print(f"Loaded batch {batch_idx} from cache.")
        else:
            batch_embeddings = process_batch_embeddings(
                args.model_id,
                batch_idx,
                batch_size,
                all_prompts,
                all_categories,
                tokenizer,
                model,
                device,
                batch_cache_file
            )


            if not batch_embeddings:
                raise RuntimeError(f"Failed to process batch {batch_idx}")
            print(f"Processed and cached batch {batch_idx}")

        all_embeddings.extend(batch_embeddings)

    torch.save(all_embeddings, full_embeddings_file)
    print(f"Saved  embeddings to {full_embeddings_file}")
    print(f"Found {len(all_prompts)} prompts in {dataset_name} dataset.")
   


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VISU validation and test embeddings using Hyperbolic-CLIP")

    parser.add_argument(
        "--model_id",
        type=str,
        default="aimagelab/hysac",
        help="HySAC model ID. Default: aimagelab/hysac"
    )

    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP backbone used with HySAC. Default: openai/clip-vit-large-patch14"
    )

    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID to use. Default: 0"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="embeddings",
        help="Base directory to store embeddings. Default: embeddings_cache"
    )

    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists."
    )

    parser.add_argument('--dataset', type=str, default="4chain", choices=["mma", "4chain", 'lexica', 'pattern'])
    return parser.parse_args()

    


def main():
    args = parse_args()
    load_dotenv()

    device = get_device(args.device_id)
    print(f"\nRunning HySAC embedding extraction")
    print(f"Model ID: {args.model_id}")
    print(f"Backbone: {args.clip_backbone}")
    print(f"Output base dir: {args.cache_dir}")

    ensure_directory(args.cache_dir)

    print("Loading model and tokenizer...")
    model = HySAC_model.from_pretrained(args.model_id, device=device).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_backbone)

    print("Model and tokenizer loaded successfully.")

    process_dataset(args, args.dataset, tokenizer, model, device)

    print("\nFinished all splits.")


if __name__ == "__main__":
    main()
