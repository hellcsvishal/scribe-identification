import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from src.triplet_dataset import TripletDataset
from src.model import Encoder

PROCESSED_DATA_PATH = project_root / "data" / "processed"
MODELS_PATH = project_root / "models"
MODELS_PATH.mkdir(exist_ok=True)

# Training Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MARGIN = 1.0  # The "margin" for the triplet loss

def main():
    """Main function to run the Siamese training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---  SETUP DATA LOADERS ---
    train_dataset = TripletDataset(PROCESSED_DATA_PATH / 'train', is_train=True)
    val_dataset = TripletDataset(PROCESSED_DATA_PATH / 'val', is_train=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---  INITIALIZE MODEL, OPTIMIZER, AND LOSS ---
    model = Encoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # This is the new loss function, designed for triplets
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    # ---  TRAINING LOOP ---
    print("\nStarting Siamese network (Triplet) training...")
    
    best_val_loss = float('inf')  # We want to minimize the loss
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0.0
        
        for anchor_img, positive_img, negative_img, _ in train_loader:
            # Move images to the device
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            
            # --- Forward pass ---
            # Pass all three images through the SAME model to get their "fingerprints"
            anchor_embedding = model(anchor_img)
            positive_embedding = model(positive_img)
            negative_embedding = model(negative_img)
            
            # Calculate the triplet loss
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # --- Backward pass and optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # ---  VALIDATION ---
        model.eval() # Set model to evaluation mode
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
            torch.save(model.state_dict(), MODELS_PATH / "best_encoder.pth")
            print(f"  > New best model saved with Val Loss: {avg_val_loss:.4f}")

    print("\nTraining finished!")
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}")
    print(f"Best encoder model saved to {MODELS_PATH / 'best_encoder.pth'}")

if __name__ == '__main__':
   main()
