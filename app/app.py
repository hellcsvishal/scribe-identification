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
from collections import Counter

# --- 1. SETUP: ADD PROJECT ROOT TO PATH ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    # --- NEW: Import the ResNetEncoder ---
    from src.model import ResNetEncoder
except ImportError:
    st.error("Error: Could not import the Encoder model from 'src/model.py'.")
    st.stop()

# --- 2. CONFIGURATION ---
MODELS_PATH = project_root / "models"
# --- NEW: Point to the ResNet model file ---
MODEL_FILE = MODELS_PATH / "best_resnet_encoder.pth" 
INDEX_FILE = MODELS_PATH / "scribe_index.faiss"
PATHS_FILE = MODELS_PATH / "image_paths.pkl"
LABELS_FILE = MODELS_PATH / "image_labels.pkl"

PATCH_SIZE = 256
STRIDE = 128
TEXT_THRESHOLD = 5.0

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

# --- 3. LOAD MODELS (with Caching) ---
@st.cache_resource
def load_models():
    """Loads the ResNetEncoder, FAISS index, image paths, and labels list."""
    print("Loading ResNet models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- NEW: Load the ResNetEncoder ---
    model = ResNetEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    
    index = faiss.read_index(str(INDEX_FILE))
    
    with open(PATHS_FILE, 'rb') as f:
        image_paths = pickle.load(f)
        
    with open(LABELS_FILE, 'rb') as f:
        image_labels = pickle.load(f)
    
    return model, index, image_paths, image_labels, device

# --- 4. HELPER FUNCTION: PROCESS UPLOADED IMAGE (ResNet compatible) ---
def process_uploaded_image(image_pil, device):
    """
    Takes an uploaded PIL image, replicates the ResNet data pipeline
    (patching, binarizing, filtering, 3-channel conversion), 
    and returns a tensor of valid patches.
    """
    img_np = np.array(image_pil.convert('L'))
    height, width = img_np.shape
    valid_patches = []
    
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch_np = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            _, binary_patch = cv2.threshold(
                patch_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            total_pixels = PATCH_SIZE * PATCH_SIZE
            white_pixels = cv2.countNonZero(binary_patch)
            text_percentage = (white_pixels / total_pixels) * 100
            
            if text_percentage > TEXT_THRESHOLD:
                # --- NEW: Convert to 3-Channel RGB for ResNet ---
                patch_pil = Image.fromarray(binary_patch).convert('RGB')
                
                transformed_patch = inference_transform(patch_pil)
                valid_patches.append(transformed_patch)
                
    if not valid_patches:
        return None
    return torch.stack(valid_patches).to(device)

# --- 5. STREAMLIT APP UI ---
st.title("📜 Scribe's Journey: Manuscript Search (ResNet v2.0)")
st.write("Upload an image of a manuscript page to identify the most likely scribe.")

try:
    model, index, image_paths, image_labels, device = load_models()
except FileNotFoundError as e:
    st.error(f"Error: Missing model files in {MODELS_PATH}. "
             "Did you run 'scripts/create_index.py' first?")
    st.info(e)
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a manuscript image...", type=["jpg", "jpeg", "png"])

# --- Sliders for settings ---
st.sidebar.title("Search Settings")
distance_threshold = st.sidebar.slider(
    "Confidence Threshold (Lower = Stricter)",
    min_value=0.1, max_value=20.0, value=10.0, step=0.1  # Note: ResNet distances may be different, so we increase the max
)
num_results = st.sidebar.slider("Number of patches to compare (for voting):", 5, 25, 15)


if uploaded_file is not None:
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Your Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image with ResNet encoder..."):
        patches_tensor = process_uploaded_image(query_image, device)
        
        if patches_tensor is None:
            st.warning("Could not find any text in the uploaded image.")
        else:
            with torch.no_grad():
                embeddings = model(patches_tensor).cpu().numpy()
            
            query_vector = np.mean(embeddings, axis=0, keepdims=True)
            D, I = index.search(query_vector, num_results)
            
            avg_distance = np.mean(D[0])
            
            st.subheader("Analysis of Results")
            st.metric(label="Average Match Distance", value=f"{avg_distance:.4f}")
            st.caption(f"Confidence Threshold is set to: {distance_threshold}")

            # 1. CHECK CONFIDENCE THRESHOLD
            if avg_distance > distance_threshold:
                st.warning(f"""
                **Prediction: UNKNOWN SCRIBE**
                
                The average match distance ({avg_distance:.4f}) is higher than your
                threshold ({distance_threshold}). This handwriting style does not
                closely match any scribe in our database.
                """)

            # 2. IF IT PASSES, TALLY THE VOTES
            else:
                st.success(f"""
                **Prediction: KNOWN SCRIBE**
                
                The average match distance ({avg_distance:.4f}) is below your
                threshold. Proceeding with identification...
                """)
                
                result_indices = I[0]
                result_labels = [image_labels[i] for i in result_indices]
                votes = Counter(result_labels)
                winner, vote_count = votes.most_common(1)[0]
                confidence = (vote_count / num_results) * 100
                
                st.metric(
                    label="Predicted Scribe",
                    value=winner.upper(),
                    delta=f"{confidence:.0f}% Confidence ({vote_count} of {num_results} patches)"
                )
            
            # 3. ALWAYS SHOW THE TOP MATCHES FOR DEBUGGING
            st.subheader(f"Top {num_results} Most Similar Patches Found:")
            num_cols = 5
            cols = st.columns(num_cols)
            for i, (idx, label, dist) in enumerate(zip(result_indices, result_labels, D[0])):
                with cols[i % num_cols]:
                    result_image = Image.open(image_paths[idx])
                    st.image(result_image, 
                             caption=f"Match {i+1} | {label}\nDist: {dist:.4f}", 
                             use_column_width=True)

