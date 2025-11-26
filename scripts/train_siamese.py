import sys
from pathlib import Path

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# --- End of fix ---

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time

# Import our new modules
from src.triplet_dataset import TripletDataset
from src.model import ResNetEncoder  # <-- NEW: Import the ResNetEncoder
# from src.model import Encoder # (We are no longer using the simple Encoder)

# --- 1. CONFIGURATION ---
PROCESSED_DATA_PATH = project_root / "data" / "processed"
MODELS_PATH = project_root / "models"
MODELS_PATH.mkdir(exist_ok=True)
NEW_MODEL_NAME = "best_resnet_encoder.pth" # <-- NEW: Save file name

# Training Hyperparameters
NUM_EPOCHS = 30 # ResNet might learn faster, but 30 is a safe start
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MARGIN = 1.0

def main():
    """Main function to run the Siamese training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. SETUP DATA LOADERS ---
    # The TripletDataset will now automatically use the updated ScribeDataset
    train_dataset = TripletDataset(PROCESSED_DATA_PATH / 'train', is_train=True)
    val_dataset = TripletDataset(PROCESSED_DATA_PATH / 'val', is_train=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 3. INITIALIZE MODEL, OPTIMIZER, AND LOSS ---
    print("Loading pre-trained ResNetEncoder...")
    model = ResNetEncoder().to(device) # <-- NEW: Use the ResNetEncoder
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    # --- 4. TRAINING LOOP ---
    print("\nStarting Siamese network (Triplet) training with ResNet...")
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for anchor_img, positive_img, negative_img, _ in train_loader:
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            
            anchor_embedding = model(anchor_img)
            positive_embedding = model(positive_img)
            negative_embedding = model(negative_img)
            
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # --- 5. VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor_img, positive_img, negative_img, _ in val_loader:
                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                negative_img = negative_img.to(device)
                
                anchor_embedding = model(anchor_img)
                positive_embedding = model(positive_img)
                negative_embedding = model(negative_img)
                
                loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        duration = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Duration: {duration:.2f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_PATH / NEW_MODEL_NAME)
            print(f"  > New best model saved with Val Loss: {avg_val_loss:.4f}")

    print("\nTraining finished!")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Best encoder model saved to {MODELS_PATH / NEW_MODEL_NAME}")

if __name__ == '__main__':
    main()
