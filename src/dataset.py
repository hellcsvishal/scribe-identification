import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import random
import cv2
from torchvision import transforms

# --- NEW: ImageNet standard normalization ---
# ResNet models were trained on ImageNet. We must use the same normalization.
# We will apply this to our 3-channel grayscale images.
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class ScribeDataset(Dataset):
    """
    Custom Dataset for loading scribe image patches.
    This version is compatible with ResNet.
    """
    
    def __init__(self, data_path: Path, is_train: bool):
        """
        Args:
            data_path (Path): Path to the train or val directory.
            is_train (bool): If True, apply augmentations.
        """
        self.image_paths = list(data_path.glob('*/*.[jJ][pP][gG]'))
        self.is_train = is_train
        
        self.classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # --- UPDATED: Transformations ---
        # 1. Convert to Tensor
        # 2. Apply standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ])

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)
    
    def apply_augmentations(self, img_np):
        """
        Applies random "damage" to the image to force the model to learn
        handwriting features, not binarization artifacts.
        """
        # Randomly choose a kernel for dilation/erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        aug_type = random.randint(0, 3) # 4 possible augmentations
        
        if aug_type == 1:
            # Erode (make text thinner)
            img_np = cv2.erode(img_np, kernel, iterations=1)
        elif aug_type == 2:
            # Dilate (make text thicker)
            img_np = cv2.dilate(img_np, kernel, iterations=1)
        elif aug_type == 3:
            # Gaussian Blur (make text blurrier)
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        # if aug_type == 0, do nothing (pass the original)
            
        return img_np

    def __getitem__(self, idx):
        """Gets one image and its label."""
        img_path = self.image_paths[idx]
        
        # We load the image as a NumPy array using OpenCV
        # cv2.IMREAD_GRAYSCALE ensures it's 1-channel
        image_np = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations ONLY if it's the training set
        if self.is_train:
            image_np = self.apply_augmentations(image_np)
            
        # --- UPDATED: Convert to 3-Channel RGB ---
        # We convert our 1-channel image back to a PIL Image
        # and then use .convert('RGB') to stack the channel 3 times.
        # This makes it (256, 256, 3) instead of (256, 256, 1)
        image_pil = Image.fromarray(image_np).convert('RGB')
        
        # Apply the ToTensor and Normalize transforms
        image_tensor = self.transform(image_pil)
        
        # Get the class label
        class_name = img_path.parent.name
        label = self.class_to_idx[class_name]
        
        return image_tensor, torch.tensor(label, dtype=torch.long)
