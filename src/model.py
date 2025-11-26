import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- This is our original, simple Encoder ---
# We can leave this here for reference, it's not hurting anything.
class Encoder(nn.Module):
    """
    This is the "Encoder" network. It's the same as SimpleCNN,
    but we stop before the final classification layer.
    Its job is to turn an image patch into a "fingerprint" (embedding).
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_dim = 32 * 64 * 64
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_dim)
        x = F.relu(self.fc1(x)) 
        return x

# --- THIS IS THE NEW, CORRECTED ResNetEncoder ---
class ResNetEncoder(nn.Module):
    """
    A powerful Encoder based on a pre-trained ResNet18 model.
    """
    
    def __init__(self, embedding_dim=512):
        super(ResNetEncoder, self).__init__()
        
        # 1. Load a pre-trained ResNet18 model
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # --- NO INPUT SURGERY ---
        # We are *not* changing resnet.conv1.
        # It correctly expects a 3-channel image, and our dataset
        # (in src/dataset.py) correctly provides one.
        
        # 2. Perform "Output Surgery"
        # Get the number of input features for the final layer
        num_ftrs = resnet.fc.in_features
        
        # We still replace the final classification layer with our "fingerprint" layer
        # We use nn.Identity() as a placeholder to effectively remove the final layer.
        resnet.fc = nn.Identity()
        
        # Now, self.encoder is the full ResNet *except* its final layer.
        self.encoder = resnet
        
        # 3. Add our own new "fingerprint" layer
        # This will take the output from the ResNet (num_ftrs)
        # and turn it into our 512-dimension embedding.
        self.embedding_layer = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        """
        The forward pass for the encoder.
        It returns the 512-dimension "fingerprint" vector.
        """
        # 1. Pass the 3-channel image through the ResNet body
        x = self.encoder(x)
        # 2. Pass the ResNet output through our new fingerprint layer
        x = self.embedding_layer(x)
        return x



