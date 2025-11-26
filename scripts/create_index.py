import sys
from pathlib import Path

# --- Add project root to Python path ---
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

# --- NEW: Import the correct encoder ---
from src.model import ResNetEncoder

# --- 1. CONFIGURATION ---
PROCESSED_DATA_PATH = project_root / "data" / "processed"
MODELS_PATH = project_root / "models"
MODEL_FILE = MODELS_PATH / "best_resnet_encoder.pth"  # <-- NEW: Use the ResNet model
INDEX_FILE = MODELS_PATH / "scribe_index.faiss"
PATHS_FILE = MODELS_PATH / "image_paths.pkl"
LABELS_FILE = MODELS_PATH / "image_labels.pkl"
BATCH_SIZE = 32  # Smaller batch size for the larger model, just in case

# --- NEW: ResNet-compatible transformations ---
# Must be identical to the validation set's transforms
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    imagenet_normalize
])

# --- 2. A SIMPLE DATASET FOR INFERENCE ---
class InferenceDataset(Dataset):
    """A simple dataset to load processed images for inference."""
    
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
        self.transform = inference_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # --- NEW: Load as 3-channel RGB ---
        # We load the binarized (grayscale) patch,
        # but convert it to 3-channel RGB for ResNet.
        image = Image.open(img_path).convert('RGB') 
        
        image = self.transform(image)
        return image

# --- 3. MAIN SCRIPT ---
def main():
    print("Starting to build the search index with ResNet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load the trained ResNetEncoder model ---
    print(f"Loading trained encoder from {MODEL_FILE}...")
    model = ResNetEncoder().to(device) # <-- NEW: Load ResNetEncoder
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    # --- Step 2: Find all image patches AND their labels ---
    print(f"Finding all image patches in {PROCESSED_DATA_PATH}...")
    all_image_paths = list(PROCESSED_DATA_PATH.rglob('*.[jJ][pP][gG]'))
    all_labels = [p.parent.name for p in all_image_paths]
    print(f"Found {len(all_image_paths)} total patches from {len(set(all_labels))} scribes.")

    # --- Step 3: Create Dataset and DataLoader ---
    dataset = InferenceDataset(all_image_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 4: Generate "fingerprints" for all images ---
    print("Generating 'fingerprints' for all patches... (this may take a moment)")
    all_embeddings = []
    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            if i % 20 == 0:
                print(f"  > Processed batch {i}/{len(loader)}")
            
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
        
    print(f"Saving image label list to {LABELS_FILE}...")
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(all_labels, f)

    print("\n✅ ResNet Index creation complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print(f"Total time taken: {duration:.2f} seconds.")

