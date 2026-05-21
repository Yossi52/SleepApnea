"""
Transformer Fusion Model: Three EfficientNet-B0 branches with Transformer encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class TransformerFusionModel(nn.Module):
    """Transformer-based fusion model"""
    
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_layers=3, 
                 dropout=0.1, pretrained=True):
        super(TransformerFusionModel, self).__init__()
        
        self.d_model = d_model
        
        # Three feature extractors
        self.feature_extractors = nn.ModuleList([
            self._create_feature_extractor(pretrained) for _ in range(3)
        ])
        
        # Channel embeddings (learnable position embeddings for each branch)
        self.channel_embeddings = nn.Parameter(torch.randn(3, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _create_feature_extractor(self, pretrained):
        """Create EfficientNet-B0 feature extractor"""
        if pretrained:
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        else:
            model = models.efficientnet_b0(weights=None)
        return model.features
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.channel_embeddings, mean=0, std=0.02)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, C, H, W)
               3 branches, each with C channels
        """
        batch_size = x.size(0)
        
        # Extract features from each branch
        features = []
        for i in range(3):
            feat = self.feature_extractors[i](x[:, i])  # (B, 1280, H', W')
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 1280)
            features.append(feat)
        
        # Stack features: (B, 3, 1280)
        channel_features = torch.stack(features, dim=1)
        
        # Add channel embeddings
        channel_features = channel_features + self.channel_embeddings.unsqueeze(0)
        
        # Transformer encoding
        encoded_features = self.transformer(channel_features)  # (B, 3, 1280)
        
        # Global average pooling across channels
        fused_features = encoded_features.mean(dim=1)  # (B, 1280)
        
        # Classification
        return self.classifier(fused_features)

