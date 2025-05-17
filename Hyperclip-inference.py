from HySAC.hysac.dataset import i2p
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import os 
import hashlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


load_dotenv()
def _get_device(index = 0):
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Total GPUs: {torch.cuda.device_count()}")
        device = 'cuda:'+ str(index)  # Make sure this GPU exists
        # Verify GPU 2 exists
        if torch.cuda.device_count() <= 2:
            print(f"Warning: GPU 2 may not exist. Available GPUs: {torch.cuda.device_count()}")
            device = f'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    return device 

# Function to generate a unique identifier for embeddings cache
def _get_cache_filename(model_id, clip_backbone, dataset_name, split, batch_idx=None):
    # Create a hash of the model and dataset configuration
    config_string = f"{model_id}_{clip_backbone}_{dataset_name}_{split}"
    hash_obj = hashlib.md5(config_string.encode())
    hash_str = hash_obj.hexdigest()
    
    # Create cache directory if it doesn't exist
    cache_dir = "embeddings_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    if batch_idx is not None:
        return os.path.join(cache_dir, f"{hash_str}_batch_{batch_idx}.pt")
    else:
        return os.path.join(cache_dir, f"{hash_str}_all_embeddings.pt")

dataset_name='i2p'
dataset_available={'i2p': i2p.I2P}

model_id = "aimagelab/hysac"
clip_backbone = 'openai/clip-vit-large-patch14'
split = 'train'
device = _get_device(2)

dataset = dataset_available[dataset_name] ('train')
batch_size = int(os.getenv(f"{dataset_name}_batch_size"))
embeddings_path = os.getenv(f"{dataset_name}_embeddings")
all_prompts = dataset.get_all_prompts()
print(len(all_prompts))


model = HySAC.from_pretrained(model_id, device=device).to(device).eval()
tokenizer =  CLIPTokenizer.from_pretrained(clip_backbone)
all_embeddings = []

full_embeddings_file = _get_cache_filename(model_id, clip_backbone, dataset_name, split)
if os.path.exists(full_embeddings_file):
    print(f"Loading pre-computed embeddings from {full_embeddings_file}")
    all_embeddings = torch.load(full_embeddings_file)
else:
    
    print("Computing embeddings...")
    all_embeddings = []
    
    for batch_idx in range(0, len(all_prompts), batch_size):
        batch_cache_file = _get_cache_filename(model_id, clip_backbone, dataset_name, split, batch_idx)
        
        # Check if this batch has already been processed
        if os.path.exists(batch_cache_file):
            print(f"Loading batch {batch_idx} from cache")
            batch_embeddings = torch.load(batch_cache_file)
        else:
            print(f"Computing batch {batch_idx} to {batch_idx + batch_size}")
            batch_embeddings = []
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            
            try:
                for prompt_idx, first_text_prompt in enumerate(tqdm(batch_prompts)):
                    try:
                        # Tokenize the prompt
                        first_text_prompt_tokens = tokenizer(
                            first_text_prompt, 
                            return_tensors='pt', 
                            padding='max_length', 
                            truncation=True
                        )
                        
                        # Get input IDs and attention mask
                        first_text_input_ids = first_text_prompt_tokens['input_ids'].to(device)
                        
                        # Generate embeddings with no gradient tracking
                        with torch.no_grad():
                            first_text_prompt_encoding = model.encode_text(first_text_input_ids, project=True)
                        
                        # Process and store embeddings
                        flattened_first_text_prompt_encoding = first_text_prompt_encoding.squeeze(0).to('cpu')
                        batch_embeddings.append(flattened_first_text_prompt_encoding)
                        
                        # Clean up to free memory
                        del first_text_input_ids
                        del first_text_prompt_encoding
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing prompt {batch_idx + prompt_idx}: '{first_text_prompt}'")
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

print(f"Total embeddings collected: {len(all_embeddings)}")

if len(all_embeddings) < 0:
    raise RuntimeError('Embeddings not correctly computed')


# Perform PCA (3D)
embeddings_tensor = torch.stack(all_embeddings)
print(f"Original Embeddings Shape: {embeddings_tensor.shape}")

if embeddings_tensor.shape[1] > 3 :
    print("Performing PCA to reduce embeddings to 3D...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_tensor)

    # Visualization Directory
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Save Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='blue', alpha=0.6)
    ax.set_title("3D Visualization of Embeddings (PCA)")
    plot_path = os.path.join(plot_dir, "embeddings_pca_3d.png")
    plt.savefig(plot_path)
    print(f"3D plot saved at {plot_path}")
else:
    print("Embeddings are not of dimension 768. Cannot perform PCA.")
