import torch
import pandas as pd
from LMLR import LorentzMLR
from geoopt.manifolds.lorentz import Lorentz
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC
import numpy as np

class Config:
    """Configuration matching the training script"""
    CURVATURE_K = 2.3026
    NUM_FEATURES = 769
    NUM_CLASSES = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_models():
    """Initialize models with proper configuration"""
    print("Setting up models...")
    
    # Initialize HySAC model
    model_id = "aimagelab/hysac"
    hysac_model = HySAC.from_pretrained(model_id, device=Config.DEVICE).to(Config.DEVICE)
    
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    # Initialize Lorentz MLR with same configuration as training
    manifold = Lorentz(k=Config.CURVATURE_K)
    lorentz_mlr = LorentzMLR(
        manifold=manifold, 
        num_features=Config.NUM_FEATURES, 
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    print(f"Models initialized on device: {Config.DEVICE}")
    print(f"Lorentz manifold curvature: {manifold.k.item()}")
    
    return hysac_model, tokenizer, lorentz_mlr

def convert_to_hyperbolic_embedding(embedding, manifold_k):
    """
    Convert spatial embedding to hyperbolic embedding by adding time dimension.
    This function matches the preprocessing in the training script.
    """
    # Ensure embedding is 2D (batch_size, embedding_dim)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    elif embedding.dim() == 3:
        embedding = embedding.squeeze(1)
    
    # Calculate time component using the SAME formula as in training
    # Training formula: torch.sqrt(1 / self.k + embedding.norm() ** 2)
    spatial_norm_squared = torch.sum(embedding ** 2, dim=-1, keepdim=True)
    time_component = torch.sqrt(1 / manifold_k + spatial_norm_squared)
    
    # Concatenate time component with spatial embedding
    hyperbolic_embedding = torch.cat([time_component, embedding], dim=-1)
    
    return hyperbolic_embedding

def validate_lorentz_embedding(embedding, manifold_k, tolerance=1e-6):
    """
    Validate that embedding satisfies Lorentz manifold constraint:
    -x0^2 + x1^2 + ... + xn^2 = -1/k
    """
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    time_component = embedding[:, 0]
    spatial_component = embedding[:, 1:]
    
    # Calculate Lorentz inner product
    lorentz_product = -time_component**2 + torch.sum(spatial_component**2, dim=-1)
    expected_value = -1 / manifold_k
    
    constraint_violation = torch.abs(lorentz_product - expected_value)
    
    is_valid = constraint_violation < tolerance
    
    return is_valid, constraint_violation

def load_trained_model(model, model_path):
    """Load trained model with proper error handling"""
    try:
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model from: {model_path}")
        
        # Print model parameters for debugging
        print(f"Model curvature: {model.manifold.k.item()}")
        if hasattr(model, 'a'):
            print(f"Model parameter 'a': {model.a}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def classify_embeddings(model, embeddings, description=""):
    """Classify embeddings with proper error handling and validation"""
    if embeddings is None or embeddings.numel() == 0:
        print(f"Warning: Empty embeddings for {description}")
        return None, None
    
    print(f"\nClassifying {description}...")
    print(f"Input embeddings shape: {embeddings.shape}")
    
    # Validate embeddings are on Lorentz manifold
    is_valid, violations = validate_lorentz_embedding(embeddings, model.manifold.k.item())
    print(f"Manifold constraint validation - Valid: {is_valid.all().item()}")
    if not is_valid.all():
        print(f"Warning: Some embeddings violate manifold constraint. Max violation: {violations.max().item():.6f}")
    
    try:
        with torch.no_grad():
            logits = model(embeddings)
            probabilities = torch.sigmoid(logits)
            
            print(f"Logits shape: {logits.shape}")
            print(f"Logits: {logits.flatten()}")
            print(f"Probabilities: {probabilities.flatten()}")
            
            return logits, probabilities
    
    except Exception as e:
        print(f"Error during classification: {e}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings stats - mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
        return None, None

def process_saved_embeddings(lorentz_mlr, embeddings_base_path):
    """Process saved embeddings from attack experiments"""
    print("=" * 60)
    print("PROCESSING SAVED EMBEDDINGS")
    print("=" * 60)
    
    original_emb_list, attacked_emb_list = [], []
    
    for i in range(1, 11):
        emb_path = f"{embeddings_base_path}/{i}_embeddings.pt"
        
        try:
            embeddings = torch.load(emb_path, map_location=Config.DEVICE)
            print(f"\nProcessing file: {emb_path}")
            
            att = embeddings.get("N1", None)
            orig = embeddings.get("original_prompt", None)
            
            if att is not None:
                print(f"Attacked embedding shape before processing: {att.shape}")
                att_hyperbolic = convert_to_hyperbolic_embedding(att, lorentz_mlr.manifold.k.item())
                attacked_emb_list.append(att_hyperbolic)
                print(f"Attacked embedding shape after processing: {att_hyperbolic.shape}")
            
            if orig is not None:
                print(f"Original embedding shape before processing: {orig.shape}")
                orig_hyperbolic = convert_to_hyperbolic_embedding(orig, lorentz_mlr.manifold.k.item())
                original_emb_list.append(orig_hyperbolic)
                print(f"Original embedding shape after processing: {orig_hyperbolic.shape}")
                
        except Exception as e:
            print(f"Error processing file {emb_path}: {e}")
            continue
    
    if not original_emb_list or not attacked_emb_list:
        print("Error: No valid embeddings found!")
        return
    
    # Stack embeddings
    original_emb = torch.cat(original_emb_list, dim=0)
    attacked_emb = torch.cat(attacked_emb_list, dim=0)
    
    print(f"\nFinal shapes - Original: {original_emb.shape}, Attacked: {attacked_emb.shape}")
    
    # Classify embeddings
    orig_logits, orig_probabilities = classify_embeddings(lorentz_mlr, original_emb, "Original")
    att_logits, att_probabilities = classify_embeddings(lorentz_mlr, attacked_emb, "Attacked")
    
    if orig_probabilities is None or att_probabilities is None:
        print("Error: Classification failed!")
        return
    
    # Calculate accuracy (assuming original=benign=0, attacked=malicious=1)
    original_correct = (orig_probabilities < 0.5).sum().item()
    attacked_correct = (att_probabilities >= 0.5).sum().item()
    
    total_original = orig_probabilities.numel()
    total_attacked = att_probabilities.numel()
    
    original_accuracy = original_correct / total_original
    attacked_accuracy = attacked_correct / total_attacked
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Original Accuracy (should classify as benign): {original_accuracy * 100:.2f}%")
    print(f"Attacked Accuracy (should classify as malicious): {attacked_accuracy * 100:.2f}%")
    
    print(f"\n{'Original Probability':<25} {'Attacked Probability':<25}")
    print("-" * 50)
    for orig, att in zip(orig_probabilities.flatten(), att_probabilities.flatten()):
        print(f"{orig.item():<25.4f} {att.item():<25.4f}")

def process_text_prompts(hysac_model, tokenizer, lorentz_mlr, csv_path, max_prompts=5):
    """Process text prompts through the classification pipeline"""
    print("=" * 60)
    print("PROCESSING TEXT PROMPTS")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_path)
        prompts = df["prompt"].tolist()[:max_prompts]  # Limit for testing
        
        print(f"Processing {len(prompts)} prompts from: {csv_path}")
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
            print(f"Text: {prompt}")
            
            try:
                # Tokenize prompt
                tokenized = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )["input_ids"].to(Config.DEVICE)
                
                print(f"Tokenized shape: {tokenized.shape}")
                
                # Get embedding from HySAC
                with torch.no_grad():
                    embedding = hysac_model.encode_text(
                        tokenized,
                        project=True
                    )
                    
                    print(f"HySAC embedding shape: {embedding.shape}")
                    print(f"HySAC embedding stats - mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
                    
                    # Convert to hyperbolic embedding
                    hyperbolic_embedding = convert_to_hyperbolic_embedding(embedding, lorentz_mlr.manifold.k.item())
                    
                    # Classify
                    logits, probabilities = classify_embeddings(lorentz_mlr, hyperbolic_embedding, f"Prompt {i+1}")
                    
                    if probabilities is not None:
                        classification = "Malicious" if probabilities.item() >= 0.5 else "Benign"
                        confidence = probabilities.item() if probabilities.item() >= 0.5 else 1 - probabilities.item()
                        print(f"Classification: {classification} (confidence: {confidence:.4f})")
                    
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
                continue
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")

def main():
    """Main execution function"""
    print("Starting Hyperbolic MLR Classification Script")
    print(f"Using device: {Config.DEVICE}")
    
    # Paths
    embeddings_base_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/COMPOSITIONAL_ATTACK/HyperbolicSD/out-images/visu_sdv14_hyperclip/N1/prompts"
    state_dict_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/LorentzMLR/final_hyperbolic_mlr_model.pth"
    csv_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/COMPOSITIONAL_ATTACK/HyperbolicSD/data/mma_clean_prompts.csv"
    
    # Setup models
    hysac_model, tokenizer, lorentz_mlr = setup_models()
    
    # Load trained model
    if not load_trained_model(lorentz_mlr, state_dict_path):
        print("Failed to load trained model. Exiting.")
        return
    
    # Process saved embeddings
    process_saved_embeddings(lorentz_mlr, embeddings_base_path)
    
    # Process text prompts
    process_text_prompts(hysac_model, tokenizer, lorentz_mlr, csv_path)
    
    print("\nClassification script completed!")

if __name__ == "__main__":
    main()