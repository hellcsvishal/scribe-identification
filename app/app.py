import streamlit as st
import sys
from pathlib import Path
import torch
import faiss
import numpy as np
import pickle
from PIL import Image
import cv2
from torchvision import transforms
from collections import Counter # To count the "votes"

# --- 1. SETUP: ADD PROJECT ROOT TO PATH ---
# This ensures we can import from the 'src' folder
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from src.model import Encoder
except ImportError:
    st.error("Error: Could not import the Encoder model from 'src/model.py'. "
             "Make sure the script is run from the project root and 'src' is accessible.")
    st.stop()

# --- 2. CONFIGURATION ---
# Define paths to our saved model and index
MODELS_PATH = project_root / "models"
MODEL_FILE = MODELS_PATH / "best_encoder.pth"
INDEX_FILE = MODELS_PATH / "scribe_index.faiss"
PATHS_FILE = MODELS_PATH / "image_paths.pkl"
LABELS_FILE = MODELS_PATH / "image_labels.pkl" # The new labels file

# Define the same pre-processing settings used during training
PATCH_SIZE = 256
STRIDE = 128
TEXT_THRESHOLD = 5.0  # 5%

# Define the image transformation
# This must be IDENTICAL to the validation transform
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. LOAD MODELS (with Caching) ---
# @st.cache_resource ensures these heavy models are loaded only ONCE.
@st.cache_resource
def load_models():
    """Loads the Encoder, FAISS index, image paths, and labels list."""
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Encoder model
    model = Encoder().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    # Load FAISS index
    index = faiss.read_index(str(INDEX_FILE))
    
    # Load the image paths list
    with open(PATHS_FILE, 'rb') as f:
        image_paths = pickle.load(f)
        
    # Load the image labels list
    with open(LABELS_FILE, 'rb') as f:
        image_labels = pickle.load(f)
    
    return model, index, image_paths, image_labels, device

# --- 4. HELPER FUNCTION: PROCESS UPLOADED IMAGE ---
def process_uploaded_image(image_pil, device):
    """
    Takes an uploaded PIL image, replicates the entire data preparation pipeline
    (patching, binarizing, filtering), and returns a tensor of valid patches.
    """
    # Convert to grayscale NumPy array
    img_np = np.array(image_pil.convert('L'))
    height, width = img_np.shape
    
    valid_patches = []
    
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            # Crop the patch
            patch_np = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Binarize
            _, binary_patch = cv2.threshold(
                patch_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            # Quality control filter
            total_pixels = PATCH_SIZE * PATCH_SIZE
            white_pixels = cv2.countNonZero(binary_patch)
            text_percentage = (white_pixels / total_pixels) * 100
            
            if text_percentage > TEXT_THRESHOLD:
                # If it's a good patch, convert back to PIL, apply transforms
                patch_pil = Image.fromarray(binary_patch)
                transformed_patch = inference_transform(patch_pil)
                valid_patches.append(transformed_patch)
                
    if not valid_patches:
        return None
        
    # Stack all valid patches into a single batch tensor
    return torch.stack(valid_patches).to(device)

# --- 5. STREAMLIT APP UI ---
st.title("📜 Scribe's Journey: Manuscript Search")
st.write("Upload an image of a manuscript page to identify the most likely scribe.")

# Load models once
try:
    model, index, image_paths, image_labels, device = load_models()
except FileNotFoundError as e:
    st.error(f"Error: Missing model files in {MODELS_PATH}. "
             "Did you run 'scripts/create_index.py' first?")
    st.info(f"Details: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the models: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a manuscript image...", type=["jpg", "jpeg", "png"])

# Slider for number of results
num_results = st.slider("Number of patches to compare (for voting):", 5, 25, 15)

if uploaded_file is not None:
    # 1. Display the uploaded (query) image
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Your Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image and searching the archive..."):
        # 2. Process the uploaded image
        patches_tensor = process_uploaded_image(query_image, device)
        
        if patches_tensor is None:
            st.warning("Could not find any text in the uploaded image. "
                       "Please try a different image or a clearer crop.")
        else:
            # 3. Generate "fingerprints" for all patches
            with torch.no_grad():
                embeddings = model(patches_tensor).cpu().numpy()
            
            # 4. Create an average "fingerprint" for the entire image
            query_vector = np.mean(embeddings, axis=0, keepdims=True)
            
            # 5. Search the FAISS index
            # D = distances, I = indices
            D, I = index.search(query_vector, num_results)
            
            # --- 6. TALLY THE VOTES (The New Feature) ---
            result_indices = I[0]
            result_labels = [image_labels[i] for i in result_indices]
            
            st.subheader("Analysis of Results")
            votes = Counter(result_labels)
            
            # Find the winner
            winner, vote_count = votes.most_common(1)[0]
            confidence = (vote_count / num_results) * 100
            
            # Display the winner in a metric card
            st.metric(
                label="Predicted Scribe (based on top matches)",
                value=winner.upper(),
                delta=f"{confidence:.0f}% Confidence ({vote_count} of {num_results} patches)"
            )
            
            # --- 7. Display the individual patch results ---
            st.subheader(f"Top {num_results} Most Similar Patches Found:")
            
            # Organize patches in columns for a cleaner look
            num_cols = 5
            cols = st.columns(num_cols)
            for i, (idx, label) in enumerate(zip(result_indices, result_labels)):
                with cols[i % num_cols]:
                    result_image = Image.open(image_paths[idx])
                    st.image(result_image, 
                             caption=f"Match {i+1} | Scribe: {label}", 
                             use_column_width=True)


