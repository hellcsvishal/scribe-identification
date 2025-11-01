import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import cv2      # We use OpenCV for image processing
import numpy as np # OpenCV works with NumPy arrays

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

# Patching Settings
PATCH_SIZE = 256
STRIDE = 128

#  Quality Control Setting ---
# Patches with less than this percentage of text (white pixels) will be discarded.
# This prevents the model from learning from empty margins or blank spaces.
TEXT_THRESHOLD = 5.0  # (5.0 means 5%)

# Dataset Splitting Settings
VALIDATION_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# ---  HELPER FUNCTION ---
def create_patches_for_set(image_paths: list, output_dir: Path):
    """
    Takes a list of source image paths, creates patches, binarizes them,
    filters them for quality, and saves them to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_patches_created = 0
    total_patches_discarded = 0
    
    for image_path in image_paths:
        with Image.open(image_path).convert('L') as img:
            width, height = img.size
            patch_count = 0
            for y in range(0, height - PATCH_SIZE + 1, STRIDE):
                for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                    box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                    patch = img.crop(box)
                    
                    patch_np = np.array(patch)
                    
                    # Binarize the patch (text becomes white, background becomes black)
                    _, binary_patch = cv2.threshold(
                        patch_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    
                    # --- Quality Control Check ---
                    total_pixels = PATCH_SIZE * PATCH_SIZE
                    white_pixels = cv2.countNonZero(binary_patch)
                    text_percentage = (white_pixels / total_pixels) * 100
                    
                    # Only save the patch if it contains enough text
                    if text_percentage > TEXT_THRESHOLD:
                        patch_filename = f'{image_path.stem}_patch_{patch_count}.jpg'
                        cv2.imwrite(str(output_dir / patch_filename), binary_patch)
                        total_patches_created += 1
                        patch_count += 1
                    else:
                        total_patches_discarded += 1
            
    return total_patches_created, total_patches_discarded

def main():
    """
    Executes the end-to-end data preparation pipeline with binarization
    and quality filtering.
    """
    print("Starting Filtered Data Preparation Pipeline...")
    pipeline_start_time = time.time()

    if PROCESSED_DATA_PATH.exists():
        shutil.rmtree(PROCESSED_DATA_PATH)
    print(f"Cleaned up old processed data directory.")

    for scribe_dir in [d for d in RAW_DATA_PATH.iterdir() if d.is_dir()]:
        print(f"\nProcessing scribe: {scribe_dir.name}")
        
        source_images = list(scribe_dir.glob('*.[jJ][pP][gG]'))
        
        train_image_files, val_image_files = train_test_split(
            source_images,
            test_size=VALIDATION_SPLIT_SIZE,
            random_state=RANDOM_STATE
        )
        print(f"  > Split {len(source_images)} source images into {len(train_image_files)} for training and {len(val_image_files)} for validation.")
        
        train_output_dir = PROCESSED_DATA_PATH / 'train' / scribe_dir.name
        val_output_dir = PROCESSED_DATA_PATH / 'val' / scribe_dir.name
        
        # Process the training images
        num_train, discarded_train = create_patches_for_set(train_image_files, train_output_dir)
        print(f"  > Generated {num_train} training patches (discarded {discarded_train} blank patches).")
        
        # Process the validation images
        num_val, discarded_val = create_patches_for_set(val_image_files, val_output_dir)
        print(f"  > Generated {num_val} validation patches (discarded {discarded_val} blank patches).")

    duration = time.time() - pipeline_start_time
    print(f"\n Pipeline finished successfully in {duration:.2f} seconds!")
    print(f"   Your filtered, binarized dataset is now available in: {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    main()





