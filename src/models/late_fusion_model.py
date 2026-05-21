"""
Late Fusion Model: Three EfficientNet-B0 branches with feature concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LateFusionModel(nn.Module):
    """Late fusion with 3 EfficientNet-B0 branches"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(LateFusionModel, self).__init__()
        
        # Three feature extractors (one per transformation method)
        self.feature_extractors = nn.ModuleList([
            self._create_feature_extractor(pretrained) for _ in range(3)
        ])
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_classifier()
    
    def _create_feature_extractor(self, pretrained):
        """Create EfficientNet-B0 feature extractor"""
        if pretrained:
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        else:
            model = models.efficientnet_b0(weights=None)
        
        # Return only features (remove classifier)
        return model.features
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, C, H, W)
               3 branches, each with C channels (usually 3)
        """
        # Extract features from each branch
        features = []
        for i in range(3):
            feat = self.feature_extractors[i](x[:, i])  # (B, 1280, H', W')
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 1280)
            features.append(feat)
        
        # Concatenate features
        combined = torch.cat(features, dim=1)  # (B, 1280*3)
        
        # Classification
        return self.classifier(combined)

