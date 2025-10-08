import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import gc
import os

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=False)
        )

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return attention_weights

class MLPWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attended_out = torch.sum(attention_weights * x, dim=1)
        return self.classifier(attended_out), attention_weights

def create_data_chunks(embeddings_path, labels_path, chunk_size=5000):
    """Split large dataset into manageable chunks"""
    print(f"Creating chunks from {embeddings_path}")
    
    # Load and chunk the data
    embeddings = torch.load(embeddings_path, map_location='cpu')
    labels = torch.load(labels_path, map_location='cpu')
    
    num_samples = len(embeddings)
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    chunk_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        chunk_embeddings = embeddings[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx]
        
        chunk_emb_file = f"temp_chunk_{i}_embeddings.pt"
        chunk_lbl_file = f"temp_chunk_{i}_labels.pt"
        
        torch.save(chunk_embeddings, chunk_emb_file)
        torch.save(chunk_labels, chunk_lbl_file)
        
        chunk_files.append((chunk_emb_file, chunk_lbl_file))
        print(f"Created chunk {i+1}/{num_chunks} with {len(chunk_embeddings)} samples")
    
    # Clean up original data from memory
    del embeddings, labels
    gc.collect()
    
    return chunk_files

def cleanup_chunks(chunk_files):
    """Remove temporary chunk files"""
    for emb_file, lbl_file in chunk_files:
        if os.path.exists(emb_file):
            os.remove(emb_file)
        if os.path.exists(lbl_file):
            os.remove(lbl_file)

def train_on_chunks(model, chunk_files, optimizer, loss_fn, device, batch_size=16):
    """Train model on data chunks"""
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    for emb_file, lbl_file in chunk_files:
        # Load chunk
        chunk_embeddings = torch.load(emb_file, map_location='cpu')
        chunk_labels = torch.load(lbl_file, map_location='cpu').long()
        
        # Create data loader for this chunk
        chunk_dataset = TensorDataset(chunk_embeddings, chunk_labels)
        chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        # Train on this chunk
        for inputs, labels in chunk_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            # Clean up
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        
        # Clean up chunk data
        del chunk_embeddings, chunk_labels, chunk_dataset, chunk_loader
        gc.collect()
    
    return total_loss / total_batches

def evaluate_on_chunks(model, chunk_files, loss_fn, device, batch_size=16):
    """Evaluate model on data chunks"""
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for emb_file, lbl_file in chunk_files:
            # Load chunk
            chunk_embeddings = torch.load(emb_file, map_location='cpu')
            chunk_labels = torch.load(lbl_file, map_location='cpu').long()
            
            # Create data loader for this chunk
            chunk_dataset = TensorDataset(chunk_embeddings, chunk_labels)
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            # Evaluate on this chunk
            for inputs, labels in chunk_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs, _ = model(inputs)
                loss = loss_fn(outputs.squeeze(), labels.float())
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).squeeze().long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Clean up
                del inputs, labels, outputs, preds, loss
                torch.cuda.empty_cache()
            
            # Clean up chunk data
            del chunk_embeddings, chunk_labels, chunk_dataset, chunk_loader
            gc.collect()
    
    return correct / total, val_loss / total

if __name__ == '__main__':
    BATCH_SIZE = 8  # Even smaller batch size
    EPOCHS = 10
    LR = 1e-3
    CHUNK_SIZE = 2000  # Adjust based on your available memory
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Creating training data chunks...")
    train_chunk_files = create_data_chunks(
        "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/token_level_embeddings/visu_train/separated_token_level/visu_train_token_level_embeddings.pt",
        "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/token_level_embeddings/visu_train/separated_token_level/visu_train_token_level_labels.pt",
        CHUNK_SIZE
    )
    
    print("Creating validation data chunks...")
    val_chunk_files = create_data_chunks(
        "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/token_level_embeddings/visu_validation/separated_token_level/visu_val_token_level_embeddings.pt",
        "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/token_level_embeddings/visu_validation/separated_token_level/visu_val_token_level_labels.pt",
        CHUNK_SIZE
    )
    
    # Get input dimension from first chunk
    first_chunk = torch.load(train_chunk_files[0][0], map_location='cpu')
    input_dim = first_chunk.shape[2]
    del first_chunk
    gc.collect()
    
    # Initialize model
    model = MLPWithAttention(input_dim).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    best_val_acc = 0.0
    try:
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            # Train on chunks
            avg_loss = train_on_chunks(model, train_chunk_files, optimizer, loss_fn, DEVICE, BATCH_SIZE)
            
            # Evaluate on chunks
            val_acc, val_loss = evaluate_on_chunks(model, val_chunk_files, loss_fn, DEVICE, BATCH_SIZE)
            
            print(f"Training Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "MLP_with_attention.pth")
                print(f"Saved new best model with Validation Accuracy: {val_acc:.4f}")
    
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        cleanup_chunks(train_chunk_files)
        cleanup_chunks(val_chunk_files)