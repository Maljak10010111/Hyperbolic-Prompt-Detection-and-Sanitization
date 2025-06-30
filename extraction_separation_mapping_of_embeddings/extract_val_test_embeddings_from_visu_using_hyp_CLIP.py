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

    return parser.parse_args()


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def process_visu_split(split, args, model, tokenizer, device):
    print(f"\nProcessing VISU split: {split}")

    split_dir = os.path.join(args.cache_dir, f"{split}_visu_embeddings")
    ensure_directory(split_dir)

    full_embeddings_file = os.path.join(split_dir, f"{split}_visu_embeddings.pt")

    # if full embeddings file exists and no force recompute -> skip
    if os.path.exists(full_embeddings_file) and not args.force_recompute:
        print(f"Embeddings for '{split}' already exist at: {full_embeddings_file} — skipping.")
        return

    dataset_args = {
        "cache_dir": os.getenv("VISU_CACHE_DIR"),
        "split": split
    }

    dataloader, dataset = get_dataloader_and_dataset(
        dataset_name=DatasetName("visu"),
        dataset_args=dataset_args,
        batch_size=int(os.getenv("visu_batch_size", 32))
    )

    all_prompts, all_categories = dataset.get_all_prompt_and_categories()
    print(f"  {len(all_prompts)} prompts found in {split} split.")

    batch_size = int(os.getenv("visu_batch_size", 32))
    all_embeddings = []

    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_cache_file = os.path.join(split_dir, f"{split}_batch_{batch_idx}.pt")

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
    print(f"Saved {split} embeddings to {full_embeddings_file}")


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

    for split in ["validation", "test"]:
        process_visu_split(split, args, model, tokenizer, device)

    print("\nFinished all splits.")


if __name__ == "__main__":
    main()
