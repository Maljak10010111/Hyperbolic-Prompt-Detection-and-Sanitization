from transformers import (
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
)
import os
import sys
sys.path.append(os.path.abspath(
    "/data/imaljkovic/Diffusion-Models-Embedding-Space-Defense"
))
from HySAC.hysac.dataset.utils import DatasetName
from HySAC.hysac.models import HySAC, CLIPBaseline, CLIPWrapper
from HySAC.hysac.utils.distributed import get_device
from HySAC.hysac.utils.logger import get_cache_filename
from HySAC.hysac.dataset.utils import get_dataloader_and_dataset
from HySAC.hysac.utils.embedder import process_batch_embeddings
from dotenv import load_dotenv
import torch
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process embeddings from selected datasets using specified CLIP model."
    )

    # Dataset selection arguments
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["i2p", "mscoco", "mma", 'visu_validation'],
        help="List of datasets to process. Default: i2p, mscoco, mma, visu_validation",
    )

    # CLIP model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="aimagelab/hysac",
        help="HySAC model ID. Default: aimagelab/hysac",
    )
    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP backbone model. Default: openai/clip-vit-large-patch14",
    )

    # Other arguments
    parser.add_argument(
        "--device_id", type=int, default=0, help="GPU device ID to use. Default: 0"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="token_level_embeddings",
        help="Directory to store embeddings cache. Default: embeddings_cache",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    load_dotenv()
    device = get_device(args.device_id)

    # Section on clip backbone instanciation and weights loading
    model_id = args.model_id
    clip_backbone = args.clip_backbone

    print(f"Using model ID: {model_id}")
    print(f"Using CLIP backbone: {clip_backbone}")

    if "hysac" in model_id:
        print("loading hyperbolic CLIP model")

        model = HySAC.from_pretrained(model_id, device=device).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    else:
        print("loading standard clip model")
       
        tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)
        clip_text_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()
        clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            clip_backbone
        ).to(device).eval()
        model = CLIPWrapper(te = clip_text_model, ve = clip_vision_model).to(device).eval()

    # Section about dataset loading
    datasets = args.datasets
    print(f"Processing datasets: {', '.join(datasets)}")

    dataset_kwargs = {
        "visu_validation": {
            "cache_dir": os.getenv("VISU_CACHE_DIR"),
            "split": "test",
        }
    }

    dataloaders = []
    embedding_paths = []
    global_embeddings = []
    global_captions = []

    for idx, dataset in enumerate(datasets):
        print(f"\nProcessing dataset: {dataset}")

        # Check if dataset is supported
        if dataset not in dataset_kwargs:
            print(f"Warning: Dataset {dataset} is not configured. Skipping.")
            continue

        batch_size = int(
            os.getenv(f"{dataset}_batch_size", 512)
        )  # Default to 32 if not specified
        dataloader, dataset_torch = get_dataloader_and_dataset(
            dataset_name=DatasetName(dataset),
            dataset_args=dataset_kwargs[dataset],
            batch_size=batch_size,
        )
        

        dataloaders.append(dataloader)
        embedding_paths.append(f"{dataset}_embeddings_")
        all_prompts, all_categories = dataset_torch.get_all_prompt_and_categories()

        # taking only 10% of the dataset
        subset_size = int(0.1 * len(all_prompts))
        all_prompts = all_prompts[:subset_size]
        all_categories = all_categories[:subset_size]

        print(f"Number of prompts: {len(all_prompts)}")
        print(all_prompts[:3])
        print(f"Number of categories: {len(all_categories)}")

        full_embeddings_file = get_cache_filename(
            model_id, clip_backbone, dataset, "test", cache_dir=args.cache_dir
        )
        # if the folder does not exist, create it
        os.makedirs(os.path.dirname(full_embeddings_file), exist_ok=True)

        if os.path.exists(full_embeddings_file) and not args.force_recompute:
            print(f"Loading pre-computed embeddings from {full_embeddings_file}")
            all_embeddings = torch.load(full_embeddings_file)
        else:
            print("Computing embeddings...")
            all_embeddings = []

            for batch_idx in range(0, len(all_prompts), batch_size):
                batch_cache_file = get_cache_filename(
                    model_id, clip_backbone, dataset, "test", batch_idx, cache_dir=args.cache_dir
                )

                # Check if this batch has already been processed
                if os.path.exists(batch_cache_file) and not args.force_recompute:
                    print(f"Loading batch {batch_idx} from cache")
                    batch_embeddings = torch.load(batch_cache_file)
                else:
                    batch_embeddings = process_batch_embeddings(
                        model_id,
                        batch_idx,
                        batch_size,
                        all_prompts,
                        all_categories,
                        tokenizer,
                        model,
                        device,
                        batch_cache_file,
                    )
                    if not batch_embeddings:
                        msg = f"Failed to process batch {batch_idx}. Skipping."
                        raise RuntimeError(msg)
                    else:
                        msg = f"Batch {batch_idx} correctly processed"
                        print(msg)
                # Add the batch embeddings to the full list
                all_embeddings.extend(batch_embeddings)

            # Save all embeddings to a single file for future use
            torch.save(all_embeddings, full_embeddings_file)
            print(f"Saved all embeddings to {full_embeddings_file}")

        global_embeddings.extend(all_embeddings)
        global_captions.extend(all_categories)

        print(f"Total embeddings collected: {len(global_embeddings)}")
        if len(global_embeddings) <= 0:
            raise RuntimeError("Embeddings not correctly computed")
        print(f"Total captions collected: {len(global_captions)}")


    print("Process completed successfully!")


if __name__ == "__main__":
    main()
