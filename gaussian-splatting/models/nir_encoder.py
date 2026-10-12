import torch
import torch.nn as nn
import torch.nn.functional as F

class NIREncoder(nn.Module):
    """
    Lightweight NIR Feature Encoder (Untrained/Random).
    Takes a 1-channel NIR image and outputs a d-dimensional feature map
    downsampled by a factor of 4 (H/4, W/4).
    
    This network is designed to be frozen at initialization. The random convolutional
    filters provide a strong spatial inductive bias for the Gaussian splats to match
    without risking representation collapse in a cosine-similarity loss.
    """
    def __init__(self, d=32):
        super(NIREncoder, self).__init__()
        
        self.d = d
        
        # Block 1: stride=2 (downsample to H/2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: stride=2 (downsample to H/4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 3: stride=1 (increase receptive field, maintain H/4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Output Head: project to exactly d channels
        self.head = nn.Conv2d(64, d, kernel_size=1, stride=1)
        
        # Freeze all parameters immediately
        self._freeze()
        
    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        x: (B, 1, H, W)
        returns: (B, d, H/4, W/4)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x)
        
        # L2 Normalize the features across the channel dimension
        # (Important since this acts as the target for cosine similarity)
        x = F.normalize(x, p=2, dim=1)
        
        return x
