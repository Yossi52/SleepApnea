"""
Baseline Model: Single EfficientNet-B0
For single transformation method (e.g., RP, Scalogram) or early fusion
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineModel(nn.Module):
    """Single EfficientNet-B0 model for baseline or early fusion"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(BaselineModel, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get number of features
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

