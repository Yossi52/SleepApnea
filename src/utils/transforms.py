"""Data augmentation transforms"""

import torch
from torchvision import transforms


class AdjustBrightness:
    """Adjust brightness for normalized images"""
    def __init__(self, brightness_range=0.1):
        self.brightness_range = brightness_range
        
    def __call__(self, tensor):
        factor = torch.empty(1).uniform_(-self.brightness_range, self.brightness_range).item()
        return tensor + factor


class AdjustContrast:
    """Adjust contrast for normalized images"""
    def __init__(self, contrast_range=0.1):
        self.contrast_range = contrast_range
        
    def __call__(self, tensor):
        factor = torch.empty(1).uniform_(1 - self.contrast_range, 1 + self.contrast_range).item()
        return tensor * factor


class AddGaussianNoise:
    """Add Gaussian noise"""
    def __init__(self, mean=0., std=0.001):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def get_train_transform(image_size=(224, 224)):
    """Get training transforms"""
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.RandomAffine(
            degrees=2,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        AdjustBrightness(brightness_range=0.05),
        AdjustContrast(contrast_range=0.05),
        AddGaussianNoise(mean=0., std=0.001)
    ])


def get_val_transform(image_size=(224, 224)):
    """Get validation/test transforms"""
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True)
    ])

