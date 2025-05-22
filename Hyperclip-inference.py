from transformers import CLIPTokenizer
from HySAC.hysac.dataset import *
from HySAC.hysac.dataset.utils import DATASET_CLASS_MAP
from HySAC.hysac.models import HySAC
from HySAC.hysac.utils.distributed import get_device
from HySAC.hysac.utils.logger import get_cache_filename
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import os
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from HySAC.hysac.dataset.utils import get_dataloader_and_dataset
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Process embeddings from selected datasets using specified CLIP model.")
    
    # Dataset selection arguments
    parser.add_argument("--datasets", nargs="+", default=["i2p", "mscoco", "mma"],
                        help="List of datasets to process. Default: i2p, mscoco, mma")
    
    # CLIP model arguments
    parser.add_argument("--model_id", type=str, default="aimagelab/hysac",
                        help="HySAC model ID. Default: aimagelab/hysac")
    parser.add_argument("--clip_backbone", type=str, default="openai/clip-vit-large-patch14",
                        help="CLIP backbone model. Default: openai/clip-vit-large-patch14")
    
    # Other arguments
    parser.add_argument("--device_id", type=int, default=0,
                        help="GPU device ID to use. Default: 0")
    parser.add_argument("--output_file", type=str, default="embeddings.pkl",
                        help="Output file path for embeddings. Default: embeddings.pkl")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputation of embeddings even if cache exists")
    
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

    model = HySAC.from_pretrained(model_id, device=device).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_backbone)

    # Section about dataset loading
    datasets = args.datasets
    print(f"Processing datasets: {', '.join(datasets)}")
    
    dataset_kwargs = {
        "i2p": {"split": "train"},
        "mscoco": {"annotation_path": os.getenv("mscoco_path")},
        "mma": {'csv_file': os.getenv('mma_csv_file')}
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
            
        batch_size = int(os.getenv(f"{dataset}_batch_size", 32))  # Default to 32 if not specified
        dataloader, dataset_torch = get_dataloader_and_dataset(
            dataset_name=DatasetName(dataset),
            dataset_args=dataset_kwargs[dataset],
            batch_size=batch_size,
        )
        
        dataloaders.append(dataloader)
        embedding_paths.append(f"{dataset}_embeddings")
        all_prompts, all_categories = dataset_torch.get_all_prompt_and_categories()
        print(f'Number of prompts: {len(all_prompts)}')
        print(all_prompts[:3])
        print(f'Number of categories: {len(all_categories)}')
        
        full_embeddings_file = get_cache_filename(model_id, clip_backbone, dataset, "train")

        if os.path.exists(full_embeddings_file) and not args.force_recompute:
            print(f"Loading pre-computed embeddings from {full_embeddings_file}")
            all_embeddings = torch.load(full_embeddings_file)
        else:
            print("Computing embeddings...")
            all_embeddings = []
            
            for batch_idx in range(0, len(all_prompts), batch_size):
                batch_cache_file = get_cache_filename(
                    model_id, clip_backbone, dataset, "train", batch_idx
                )

                # Check if this batch has already been processed
                if os.path.exists(batch_cache_file) and not args.force_recompute:
                    print(f"Loading batch {batch_idx} from cache")
                    batch_embeddings = torch.load(batch_cache_file)
                else:
                    print(f"Computing batch {batch_idx} to {batch_idx + batch_size}")
                    batch_embeddings = []
                    batch_prompts = all_prompts[batch_idx: batch_idx + batch_size]
                    
                    batch_categories = all_categories[batch_idx: batch_idx + batch_size]
                    try:
                        for prompt_idx, first_text_prompt in enumerate(tqdm(batch_prompts)):
                            try:
                                # Tokenize the prompt
                                first_text_prompt_tokens = tokenizer(
                                    first_text_prompt,
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=True,
                                )

                                # Get input IDs and attention mask
                                first_text_input_ids = first_text_prompt_tokens[
                                    "input_ids"
                                ].to(device)

                                # Generate embeddings with no gradient tracking
                                with torch.no_grad():
                                    first_text_prompt_encoding = model.encode_text(
                                        first_text_input_ids, project=True
                                    )

                                # Process and store embeddings
                                flattened_first_text_prompt_encoding = (
                                    first_text_prompt_encoding.squeeze(0).to("cpu")
                                )
                                # save the couple embedding, categories associated to it
                                batch_embeddings.append(
                                    (
                                        flattened_first_text_prompt_encoding,
                                        batch_categories[prompt_idx],
                                    )
                                )

                                # Clean up to free memory
                                del first_text_input_ids
                                del first_text_prompt_encoding
                                torch.cuda.empty_cache()

                            except Exception as e:
                                print(
                                    f"Error processing prompt {batch_idx + prompt_idx}: '{first_text_prompt}'"
                                )
                                print(f"Error details: {str(e)}")
                            continue

                        # Save the batch embeddings
                        torch.save(batch_embeddings, batch_cache_file)
                        print(f"Saved batch {batch_idx} to {batch_cache_file}")

                    except Exception as e:
                        print(f"Error processing batch {batch_idx}:")
                        print(f"Error details: {str(e)}")

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

    # Save the joint embeddings in a pickle file
    print(f"Saving embeddings to {args.output_file}")
    with open(args.output_file, "wb") as f:
        pickle.dump([global_embeddings, global_captions], f)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()