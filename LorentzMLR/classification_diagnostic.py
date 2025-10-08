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
    
    model_id = "aimagelab/hysac"
    hysac_model = HySAC.from_pretrained(model_id, device=Config.DEVICE).to(Config.DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    manifold = Lorentz(k=Config.CURVATURE_K)
    lorentz_mlr = LorentzMLR(
        manifold=manifold, 
        num_features=Config.NUM_FEATURES, 
        num_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)
    
    return hysac_model, tokenizer, lorentz_mlr

def estimate_training_embedding_scale():
    """
    Estimate the scale of embeddings used during training.
    Based on your debug output, training embeddings likely had much smaller norms.
    """
    # From your debug output:
    # - Simple embeddings (norm ~0.1): reasonable logits (-10 to -83)
    # - HySAC embeddings (norm ~2-3): extreme logits (+56 to +61)
    
    # Let's try to find the scale that would produce reasonable time components
    # For training, time components were probably around 0.66-1.0 range
    
    # If time_component = sqrt(1/k + norm^2), and we want time_component ≈ 0.66-1.0
    # Then: 0.66^2 = 1/k + norm^2
    # norm^2 = 0.66^2 - 1/k = 0.4356 - 1/2.3026 = 0.4356 - 0.4343 ≈ 0.001
    # So training embeddings likely had norms around 0.03-0.1
    
    return 0.05  # Estimated typical training embedding norm

def normalize_embedding_to_training_scale(embedding, target_norm=None):
    """
    Normalize HySAC embedding to match the scale used during training.
    """
    if target_norm is None:
        target_norm = estimate_training_embedding_scale()
    
    # Calculate current norm
    current_norm = torch.norm(embedding, dim=1, keepdim=True)
    
    # Normalize to target scale
    normalized_embedding = embedding * (target_norm / current_norm)
    
    print(f"Normalized embedding: original norm {current_norm.item():.6f} → target norm {target_norm:.6f}")
    
    return normalized_embedding

def convert_to_hyperbolic_with_scale_correction(embedding, k, normalize=True):
    """
    Convert to hyperbolic embedding with proper scale correction.
    """
    # Ensure 2D
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    elif embedding.dim() == 3:
        embedding = embedding.squeeze(1)
    
    if embedding.shape[1] == 769:
        return embedding
    elif embedding.shape[1] != 768:
        raise ValueError(f"Expected 768-dimensional embedding, got {embedding.shape[1]}")
    
    # Normalize to training scale if requested
    if normalize:
        embedding = normalize_embedding_to_training_scale(embedding)
    
    # Calculate time component
    embedding_norms = torch.norm(embedding, dim=1, keepdim=True)
    time_component = torch.sqrt(1 / k + embedding_norms ** 2)
    
    print(f"Spatial norm: {embedding_norms.item():.6f}, Time component: {time_component.item():.6f}")
    
    # Concatenate
    hyperbolic_embedding = torch.cat([time_component, embedding], dim=1)
    
    return hyperbolic_embedding

def load_trained_model(model, model_path):
    """Load trained model"""
    try:
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def classify_embedding(model, embedding, description=""):
    """Classify a single embedding with detailed output"""
    print(f"\nClassifying: {description}")
    print(f"Input shape: {embedding.shape}")
    
    try:
        with torch.no_grad():
            logits = model(embedding)
            probabilities = torch.sigmoid(logits)
            
            prediction = "Malicious" if probabilities.item() >= 0.5 else "Benign"
            confidence = probabilities.item() if probabilities.item() >= 0.5 else 1 - probabilities.item()
            
            print(f"Logits: {logits.item():.4f}")
            print(f"Probability: {probabilities.item():.6f}")
            print(f"Prediction: {prediction} (confidence: {confidence:.4f})")
            
            return logits, probabilities
    except Exception as e:
        print(f"Error in classification: {e}")
        return None, None

def test_different_normalizations(hysac_model, tokenizer, lorentz_mlr, test_prompt):
    """Test different normalization scales to find the right one"""
    print("=" * 80)
    print("TESTING DIFFERENT NORMALIZATION SCALES")
    print("=" * 80)
    
    # Get HySAC embedding
    tokenized = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(Config.DEVICE)
    
    with torch.no_grad():
        output = hysac_model.textual(input_ids=tokenized)
        spatial_embedding = output.text_embeds
    
    print(f"Test prompt: {test_prompt}")
    print(f"Original HySAC embedding norm: {torch.norm(spatial_embedding).item():.6f}")
    
    # Test different target norms
    target_norms = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    
    for target_norm in target_norms:
        print(f"\n--- Testing target norm: {target_norm:.3f} ---")
        
        # Normalize embedding
        normalized_embedding = normalize_embedding_to_training_scale(spatial_embedding, target_norm)
        
        # Convert to hyperbolic
        hyperbolic_embedding = convert_to_hyperbolic_with_scale_correction(
            normalized_embedding, lorentz_mlr.manifold.k.item(), normalize=False
        )
        
        # Classify
        classify_embedding(lorentz_mlr, hyperbolic_embedding, f"Norm {target_norm:.3f}")

def process_saved_embeddings_with_normalization(lorentz_mlr, embeddings_base_path):
    """Process saved embeddings with proper normalization"""
    print("=" * 80)
    print("PROCESSING SAVED EMBEDDINGS WITH NORMALIZATION")
    print("=" * 80)
    
    original_emb_list, attacked_emb_list = [], []
    
    for i in range(1, 6):  # Test fewer files first
        emb_path = f"{embeddings_base_path}/{i}_embeddings.pt"
        
        try:
            embeddings = torch.load(emb_path, map_location=Config.DEVICE)
            
            att = embeddings.get("N1", None)
            orig = embeddings.get("original_prompt", None)
            
            if att is not None:
                # Convert to hyperbolic with normalization
                att_hyperbolic = convert_to_hyperbolic_with_scale_correction(att, lorentz_mlr.manifold.k.item())
                attacked_emb_list.append(att_hyperbolic)
            
            if orig is not None:
                orig_hyperbolic = convert_to_hyperbolic_with_scale_correction(orig, lorentz_mlr.manifold.k.item())
                original_emb_list.append(orig_hyperbolic)
                
        except Exception as e:
            print(f"Error processing file {emb_path}: {e}")
            continue
    
    if not original_emb_list or not attacked_emb_list:
        print("No valid embeddings found!")
        return
    
    # Stack and classify
    original_emb = torch.cat(original_emb_list, dim=0)
    attacked_emb = torch.cat(attacked_emb_list, dim=0)
    
    print(f"\nProcessing {len(original_emb)} original and {len(attacked_emb)} attacked embeddings")
    
    with torch.no_grad():
        orig_logits = lorentz_mlr(original_emb)
        orig_probabilities = torch.sigmoid(orig_logits)
        
        att_logits = lorentz_mlr(attacked_emb)
        att_probabilities = torch.sigmoid(att_logits)
    
    # Calculate accuracy
    original_correct = (orig_probabilities < 0.5).sum().item()
    attacked_correct = (att_probabilities >= 0.5).sum().item()
    
    original_accuracy = original_correct / len(orig_probabilities)
    attacked_accuracy = attacked_correct / len(att_probabilities)
    
    print(f"\nRESULTS:")
    print(f"Original Accuracy (should be benign): {original_accuracy * 100:.2f}%")
    print(f"Attacked Accuracy (should be malicious): {attacked_accuracy * 100:.2f}%")
    
    print(f"\n{'Original':<15} {'Attacked':<15}")
    print("-" * 30)
    for orig, att in zip(orig_probabilities.flatten(), att_probabilities.flatten()):
        print(f"{orig.item():<15.6f} {att.item():<15.6f}")

def main():
    """Main function with scale correction"""
    print("SCALE-CORRECTED HYPERBOLIC MLR CLASSIFICATION")
    print("=" * 80)
    
    # Paths
    state_dict_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/LorentzMLR/final_hyperbolic_mlr_model.pth"
    embeddings_base_path = "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/COMPOSITIONAL_ATTACK/HyperbolicSD/out-images/visu_sdv14_hyperclip/N1/prompts"
    
    # Setup
    hysac_model, tokenizer, lorentz_mlr = setup_models()
    
    if not load_trained_model(lorentz_mlr, state_dict_path):
        return
    
    # Test different normalizations
    test_prompt = "A harmless image of a cat."
    test_different_normalizations(hysac_model, tokenizer, lorentz_mlr, test_prompt)
    
    # Process saved embeddings
    process_saved_embeddings_with_normalization(lorentz_mlr, embeddings_base_path)

if __name__ == "__main__":
    main()