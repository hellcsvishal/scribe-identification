import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import random
import cv2

# Import the ScribeDataset to reuse its augmentation and loading logic
from src.dataset import ScribeDataset

class TripletDataset(Dataset):
    """
    A dataset that returns triplets (Anchor, Positive, Negative) for training.
    
    It first creates a "base" dataset (our ScribeDataset) and then builds
    a map of which images belong to which class.
    """
    
    def __init__(self, data_path: Path, is_train: bool):
        """
        Args:
            data_path (Path): Path to the train or val directory.
            is_train (bool): If True, apply augmentations.
        """
        # 1. Create a base ScribeDataset instance
        # We will use this dataset to get the processed images
        self.base_dataset = ScribeDataset(data_path, is_train=is_train)
        
        # 2. Build a map of {label -> list of image indices}
        # e.g., {0: [0, 1, 5, 10...], 1: [2, 3, 4, 6...]}
        self.label_to_indices = {label: [] for label in range(len(self.base_dataset.classes))}
        for i in range(len(self.base_dataset)):
            # We get the label for each image at index 'i'
            # We call the base_dataset's __getitem__ but only need the label
            _, label_tensor = self.base_dataset[i]
            label = label_tensor.item()
            self.label_to_indices[label].append(i)
            
        self.labels_set = set(self.label_to_indices.keys())

    def __len__(self):
        """Returns the total number of images (anchors) in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, index):
        """
        Gets one triplet (Anchor, Positive, Negative).
        
        Args:
            index (int): The index of the ANCHOR image.
        """
        
        # 1. Get the Anchor image and its label
        anchor_img, anchor_label_tensor = self.base_dataset[index]
        anchor_label = anchor_label_tensor.item()
        
        # 2. Get a Positive image
        # - Pick a random index from the list of images with the SAME label
        positive_index = index
        while positive_index == index:
            # Keep picking until we find an index that is NOT the anchor
            positive_index = random.choice(self.label_to_indices[anchor_label])
        
        positive_img, _ = self.base_dataset[positive_index]

        # 3. Get a Negative image
        # - Pick a random label that is NOT the anchor's label
        negative_label = random.choice(list(self.labels_set - {anchor_label}))
        
        # - Pick a random index from that negative label's list
        negative_index = random.choice(self.label_to_indices[negative_label])
        
        negative_img, _ = self.base_dataset[negative_index]
        
        return anchor_img, positive_img, negative_img, anchor_label

