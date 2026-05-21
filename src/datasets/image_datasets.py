"""
Dataset classes for different fusion strategies

- SingleChannelDataset: For baseline and early fusion (single 3-channel input)
- MultiBranchDataset: For late and transformer fusion (3 separate inputs)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob


class SingleChannelDataset(Dataset):
    """
    Dataset for baseline or early fusion models
    - Baseline: Single transformation method (e.g., RP)
    - Early Fusion: 3-channel selection from 6-channel fusion image
    """
    
    def __init__(self, file_list, transform=None, is_fusion=False, 
                 use_methods=None, mode='train', stats_file='normalization_stats.npz'):
        self.file_list = file_list
        self.transform = transform
        self.labels = [0 if 'label0' in f else 1 for f in file_list]
        self.is_fusion = is_fusion
        self.use_methods = use_methods
        self.mode = mode
        self.stats_file = stats_file
        
        # Fusion channel mapping
        self.fusion_channel_mapping = {
            'gadf': 0, 'gasf': 1, 'mtf': 2,
            'rp': 3, 'scalogram': 4, 'spectrogram': 5
        }
        
        # Compute or load normalization stats
        self.mean, self.std = self._compute_stats()
    
    def _compute_stats(self):
        """Compute or load channel-wise mean and std"""
        if self.mode == 'train':
            channel_sum = np.zeros(3)
            channel_sum_sq = np.zeros(3)
            pixel_count = 0
            
            for file_path in self.file_list:
                img_array = np.load(file_path)
                
                if self.is_fusion:
                    # Select 3 channels from 6-channel fusion image
                    selected_indices = [self.fusion_channel_mapping[m] for m in self.use_methods]
                    img_array = img_array[:, :, selected_indices]
                
                img_array = img_array.astype(np.float32)
                channel_sum += img_array.sum(axis=(0, 1))
                channel_sum_sq += (img_array ** 2).sum(axis=(0, 1))
                pixel_count += img_array.shape[0] * img_array.shape[1]
            
            channel_mean = channel_sum / pixel_count
            channel_std = np.sqrt(channel_sum_sq / pixel_count - channel_mean ** 2)
            
            # Save stats
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            np.savez(self.stats_file, mean=channel_mean, std=channel_std)
            
            return channel_mean, channel_std
        else:
            # Load stats
            stats = np.load(self.stats_file)
            return stats['mean'], stats['std']
    
    def _normalize_image(self, img):
        """Channel-wise normalization"""
        return (img.astype(np.float32) - self.mean) / (self.std + 1e-8)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image_array = np.load(file_path)
        label = self.labels[idx]
        
        if self.is_fusion:
            # Select 3 channels
            selected_indices = [self.fusion_channel_mapping[m] for m in self.use_methods]
            image_array = image_array[:, :, selected_indices]
        
        # Normalize
        image_array = self._normalize_image(image_array)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MultiBranchDataset(Dataset):
    """
    Dataset for late fusion or transformer fusion models
    Loads 3 separate images (one per transformation method)
    """
    
    def __init__(self, file_list, transform=None, use_methods=('rp', 'scalogram', 'gadf'),
                 data_paths=None, mode='train', stats_file='normalization_stats.npz'):
        self.file_list = file_list
        self.transform = transform
        self.labels = [0 if 'label0' in f else 1 for f in file_list]
        self.use_methods = use_methods
        self.data_paths = data_paths  # Dict mapping method name to directory
        self.mode = mode
        self.stats_file = stats_file
        
        # Compute or load stats for each branch
        self.mean, self.std = self._compute_branch_stats()
    
    def _compute_branch_stats(self):
        """Compute stats for each branch separately"""
        if self.mode == 'train':
            branch_means = []
            branch_stds = []
            
            for method in self.use_methods:
                channel_sum = np.zeros(3)
                channel_sum_sq = np.zeros(3)
                pixel_count = 0
                
                for file_path in self.file_list:
                    # Get corresponding file for this method
                    basename = os.path.basename(file_path)
                    method_path = os.path.join(self.data_paths[method], basename)
                    
                    if not os.path.exists(method_path):
                        continue
                    
                    img_array = np.load(method_path).astype(np.float32)
                    channel_sum += img_array.sum(axis=(0, 1))
                    channel_sum_sq += (img_array ** 2).sum(axis=(0, 1))
                    pixel_count += img_array.shape[0] * img_array.shape[1]
                
                branch_mean = channel_sum / pixel_count
                branch_std = np.sqrt(channel_sum_sq / pixel_count - branch_mean ** 2)
                branch_means.append(branch_mean)
                branch_stds.append(branch_std)
            
            # Save stats
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            np.savez(self.stats_file, mean=np.array(branch_means), std=np.array(branch_stds))
            
            return np.array(branch_means), np.array(branch_stds)
        else:
            # Load stats
            stats = np.load(self.stats_file)
            return stats['mean'], stats['std']
    
    def _normalize_image(self, img, branch_idx):
        """Normalize image for specific branch"""
        return (img.astype(np.float32) - self.mean[branch_idx]) / (self.std[branch_idx] + 1e-8)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        basename = os.path.basename(file_path)
        label = self.labels[idx]
        
        # Load image for each branch
        images = []
        for branch_idx, method in enumerate(self.use_methods):
            method_path = os.path.join(self.data_paths[method], basename)
            img_array = np.load(method_path)
            
            # Normalize
            img_array = self._normalize_image(img_array, branch_idx)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            
            if self.transform:
                img_tensor = self.transform(img_tensor)
            
            images.append(img_tensor)
        
        # Stack images: (3, C, H, W)
        images = torch.stack(images, dim=0)
        
        return images, label


def load_data(base_dir, file_ext='.npy', exclude_c05=False):
    """Load all data files from directory"""
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(file_ext) and 'segment_info' not in file:
                if exclude_c05 and 'c05' in file:
                    continue
                all_files.append(os.path.join(root, file))
    return sorted(all_files)

