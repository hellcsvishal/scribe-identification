import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import random

class ScribeDataset(Dataset):
    """
    Custom Dataset for loading scribe image patches.
    Applies random augmentations (blur, erode, dilate) to the training set.
    """
    
    def __init__(self, data_path: Path, is_train: bool = False):
        """
        Args:
            data_path (Path): Path to the train or val directory.
            is_train (bool): If True, apply augmentations.
        """
        self.image_paths = list(data_path.glob('*/*.jpg'))
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.is_train = is_train

        # We only define the normalization transform here.
        # ToTensor will be handled manually.
        # Normalizing to -1 and 1 (from 0 and 1) is a standard practice.
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def apply_augmentations(self, img_np):
        """Applies random augmentations to a NumPy image array."""
        
        # 50% chance of applying any augmentation
        if random.random() < 0.5:
            # Pick one of the three augmentations
            aug_type = random.choice(['blur', 'erode', 'dilate'])
            
            if aug_type == 'blur':
                # Apply a 3x3 Gaussian blur
                img_np = cv2.GaussianBlur(img_np, (3,3), 0)
            
            elif aug_type == 'erode':
                # Erode to make text thinner. Kernel size (2,2) is small.
                kernel = np.ones((2,2), np.uint8)
                img_np = cv2.erode(img_np, kernel, iterations=1)
            
            elif aug_type == 'dilate':
                # Dilate to make text thicker
                kernel = np.ones((2,2), np.uint8)
                img_np = cv2.dilate(img_np, kernel, iterations=1)
                
        return img_np

    def __getitem__(self, idx):
        """
        Gets one image, applies augmentations (if training), converts to a normalized
        tensor, and returns with its label.
        """
        img_path = self.image_paths[idx]
        
        # Open as PIL Image and get the label
        image_pil = Image.open(img_path)
        label = self.class_to_idx[img_path.parent.name]
        
        # Convert to NumPy array for processing
        image_np = np.array(image_pil)
        
        if self.is_train:
            # Apply random augmentations
            image_np = self.apply_augmentations(image_np)

        # --- Efficient NumPy to Tensor Conversion ---
        
        # 1. Add a channel dimension (H, W) -> (1, H, W)
        #    PyTorch expects the channel to be first.
        image_np = np.expand_dims(image_np, axis=0)
        
        # 2. Convert from NumPy array to PyTorch Tensor
        #    and scale from 0-255 to 0.0-1.0
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        
        # 3. Apply normalization (transforms 0.0-1.0 to -1.0-1.0)
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


