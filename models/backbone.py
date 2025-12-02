import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetFaceEmbedder(nn.Module):
    """
    A ResNet-18 backbone modified for Face Recognition tasks.
    
    Modifications:
    1. Accepts 112x112 inputs (Standard for aligned face datasets).
    2. Removes the final Classification Layer.
    3. Outputs a fixed embedding vector (dim=512).
    """
    def __init__(self, embedding_dim=512, pretrained=True):
        super(ResNetFaceEmbedder, self).__init__()
        
        # Load standard ResNet18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
            
        # Modification 1: Remove the final FC layer (fc)
        # We want the output of the penultimate layer (avgpool)
        self.feature_dim = self.backbone.fc.in_features # Typically 512 for ResNet18
        
        # Replace the final FC with an Identity or Projection layer
        # Here we add a projection to the desired embedding dimension
        self.backbone.fc = nn.Linear(self.feature_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Input: [Batch, 3, 112, 112]
        Output: [Batch, Embedding_Dim]
        """
        x = self.backbone(x)
        
        # L2 Normalization is crucial for Face Recognition (Projecting to hypersphere)
        # This helps the Deep Optimizer learn direction rather than magnitude
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x