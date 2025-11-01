import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: sees 1-channel (grayscale) 256x256 images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features
        # Input 256 -> Pool -> 128 -> Pool -> 64. So, 32 channels * 64 * 64
        self.fc1_input_dim = 32 * 64 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes) # The final classification layer
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        """Passes input through the convolutional 'feature extraction' layers."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the image tensor for the fully connected layers
        x = x.view(-1, self.fc1_input_dim)
        return x

    def forward(self, x):
        """The full forward pass for the classifier."""
        x = self.forward_features(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # The final prediction
        return x

# --- NEW CLASS FOR THE SIAMESE NETWORK ---
class Encoder(nn.Module):
    """
    This is the "Encoder" network. It's the same as SimpleCNN,
    but we stop before the final classification layer.
    Its job is to turn an image patch into a "fingerprint" (embedding).
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        # We build the encoder using the parts from our proven SimpleCNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_dim = 32 * 64 * 64
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        
    def forward(self, x):
        """
        The forward pass for the encoder.
        It returns the 512-dimension "fingerprint" vector.
        """
        # Pass input through the feature extractor
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_dim)
        
        # Pass through the first fully connected layer
        # This 512-dimension vector is our "fingerprint"
        x = F.relu(self.fc1(x)) 
        return x

