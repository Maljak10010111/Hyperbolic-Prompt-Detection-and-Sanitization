import os
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import CLIPTokenizer, CLIPModel
from HySAC.hysac.dataset.utils import get_dataloader_and_dataset
from HySAC.hysac.dataset import DatasetName
from HySAC.hysac.utils.distributed import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VISU test embeddings using original CLIP")
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="clip_embeddings")
    parser.add_argument("--force_recompute", action="store_true")
    return parser.parse_args()


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def process_batch_clip(prompts, tokenizer, model, device):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    embeddings = model.get_text_features(**inputs)
    return embeddings.cpu()


def process_visu_split(split, args, model, tokenizer, device):
    print(f"\nProcessing VISU split: {split}")

    split_dir = os.path.join(args.cache_dir, f"{split}_clip_embeddings_visu")
    ensure_directory(split_dir)

    full_embeddings_file = os.path.join(split_dir, f"{split}_clip_embeddings.pt_visu")

    if os.path.exists(full_embeddings_file) and not args.force_recompute:
        print(f"Embeddings already exist at: {full_embeddings_file} — skipping.")
        return

    dataset_args = {
        "cache_dir": os.getenv("VISU_CACHE_DIR"),
        "split": split
    }

    dataloader, dataset = get_dataloader_and_dataset(
        dataset_name=DatasetName("visu_validation"),
        dataset_args=dataset_args,
        #batch_size=int(os.getenv("visu_batch_size", 32))
    )

    all_prompts, _ = dataset.get_all_prompt_and_categories()
    print(f"Found {len(all_prompts)} prompts in {split} split.")

    #batch_size = int(os.getenv("visu_batch_size", 32))
    batch_size = 128
    all_embeddings = []

    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
        batch_embeddings = process_batch_clip(batch_prompts, tokenizer, model, device)
        all_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, full_embeddings_file)
    print(f"Saved {split} embeddings to {full_embeddings_file}")


def main():
    args = parse_args()
    load_dotenv()

    device = get_device(args.device_id)
    print(f"\nExtracting embeddings using original CLIP on device {device}")
    print(f"Model: {args.model_id}, Output Dir: {args.cache_dir}")

    ensure_directory(args.cache_dir)

    model = CLIPModel.from_pretrained(args.model_id).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id)

    # for split in ["validation", "test"]:
    #     process_visu_split(split, args, model, tokenizer, device)

    process_visu_split("test", args, model, tokenizer, device)

    print("\nFinished extracting test embeddings.")


if __name__ == "__main__":
    main()
