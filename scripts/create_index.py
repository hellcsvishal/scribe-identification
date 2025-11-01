import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import faiss
import numpy as np
import pickle
import time

from src.model import Encoder

PROCESSED_DATA_PATH = project_root / "data" / "processed"
MODELS_PATH = project_root / "models"
MODEL_FILE = MODELS_PATH / "best_encoder.pth"
INDEX_FILE = MODELS_PATH / "scribe_index.faiss"
PATHS_FILE = MODELS_PATH / "image_paths.pkl"
LABELS_FILE = MODELS_PATH / "image_labels.pkl" 
BATCH_SIZE = 64

# A SIMPLE DATASET FOR INFERENCE 
class InferenceDataset(Dataset):
    """A simple dataset to load processed images for inference."""
    
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') 
        image = self.transform(image)
        return image

def main():
    print("Starting to build the search index")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load the trained Encoder model ---
    print(f"Loading trained encoder from {MODEL_FILE}...")
    model = Encoder().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    # --- Step 2: Find all image patches AND their labels ---
    print(f"Finding all image patches in {PROCESSED_DATA_PATH}...")
    all_image_paths = list(PROCESSED_DATA_PATH.rglob('*.[jJ][pP][gG]'))
    
    # --- NEW: Extract labels (parent folder name) ---
    all_labels = [p.parent.name for p in all_image_paths]
    print(f"Found {len(all_image_paths)} total patches from {len(set(all_labels))} scribes.")

    # --- Step 3: Create Dataset and DataLoader ---
    dataset = InferenceDataset(all_image_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 4: Generate "fingerprints" (embeddings) for all images ---
    print("Generating 'fingerprints' for all patches...")
    all_embeddings = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            
    embeddings_np = np.vstack(all_embeddings)
    
    # --- Step 5: Build the FAISS Search Index ---
    print("Building FAISS index...")
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)     
    
    print(f"Writing FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, str(INDEX_FILE))

    # --- Step 6: Save the path and label lists ---
    all_image_paths_str = [str(p) for p in all_image_paths]
    
    print(f"Saving image path list to {PATHS_FILE}...")
    with open(PATHS_FILE, 'wb') as f:
        pickle.dump(all_image_paths_str, f)
        
    # --- NEW: Save the labels list ---
    print(f"Saving image label list to {LABELS_FILE}...")
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(all_labels, f)

    print("\n Index creation complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print(f"Total time taken: {duration:.2f} seconds.")


