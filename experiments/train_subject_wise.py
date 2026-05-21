"""
Subject-wise Split Training Script

Splits data by subjects: files containing 'x' are used as test set,
remaining files are split into train/validation.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import json
import re

from src.utils.config_loader import load_config
from src.utils.logger import ResultLogger
from src.utils.transforms import get_train_transform, get_val_transform
from src.datasets.image_datasets import SingleChannelDataset, MultiBranchDataset, load_data
from src.models.baseline_model import BaselineModel
from src.models.late_fusion_model import LateFusionModel
from src.models.transformer_fusion_model import TransformerFusionModel
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def get_model(config):
    """Create model based on config"""
    if config.model.type == 'baseline':
        return BaselineModel(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        )
    elif config.model.type == 'late':
        return LateFusionModel(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        )
    elif config.model.type == 'transformer':
        return TransformerFusionModel(
            num_classes=config.model.num_classes,
            d_model=config.model.transformer.d_model,
            nhead=config.model.transformer.nhead,
            num_layers=config.model.transformer.num_layers,
            dropout=config.model.transformer.dropout,
            pretrained=config.model.pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")


def extract_subject_id(file_path):
    """Extract subject ID from filename (e.g., a01, x15)"""
    basename = os.path.basename(file_path)
    match = re.search(r'([axbc]\d{2})', basename)
    if match:
        return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser(description='Subject-wise Split Training')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--gpu', default='0', help='GPU ID')
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Model type: {config.model.type}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Device: {device}")
    
    # Get transforms
    train_transform = get_train_transform(tuple(config.training.image_size))
    val_transform = get_val_transform(tuple(config.training.image_size))
    
    # Load data files
    if config.model.type == 'baseline':
        all_files = load_data(config.dataset.data_dir, file_ext='.npy',
                             exclude_c05=config.dataset.get('exclude_c05', False))
    else:
        all_files = load_data(os.path.join(config.dataset.data_dir, 'fusion'),
                             file_ext='.npy',
                             exclude_c05=config.dataset.get('exclude_c05', False))
    
    # Extract labels
    y = [0 if 'label0' in f else 1 for f in all_files]
    
    print(f"\nTotal files: {len(all_files)}")
    print(f"Label 0: {y.count(0)}, Label 1: {y.count(1)}")
    
    # Run for each random state
    for random_state in config.experiment.random_states:
        print("\n" + "="*100)
        print(f"Random State: {random_state}")
        print("="*100)
        
        # Split by subject: 'x' files are test set
        test_files = [f for f in all_files if 'x' in os.path.basename(f).lower()]
        test_y = [0 if 'label0' in f else 1 for f in test_files]
        
        remaining_files = [f for f in all_files if f not in test_files]
        remaining_y = [0 if 'label0' in f else 1 for f in remaining_files]
        
        # Split remaining into train/val
        train_files, valid_files, train_y, valid_y = train_test_split(
            remaining_files, remaining_y, 
            test_size=config.experiment.val_ratio,
            stratify=remaining_y, random_state=random_state
        )
        
        print(f"Train: {len(train_files)}, Val: {len(valid_files)}, Test: {len(test_files)}")
        
        # Compute loss weights
        weight_0 = len(train_y) / (2 * train_y.count(0))
        weight_1 = len(train_y) / (2 * train_y.count(1))
        loss_weights = torch.tensor([weight_0, weight_1]).to(device)
        
        # Create datasets
        stats_file = f"stats/{config.logging.project_name}_rs{random_state}.npz"
        
        if config.model.type in ['late', 'transformer']:
            data_paths = {method: os.path.join(config.dataset.data_dir, method)
                         for method in config.model.fusion.use_methods}
            
            train_dataset = MultiBranchDataset(train_files, transform=train_transform,
                                               use_methods=tuple(config.model.fusion.use_methods),
                                               data_paths=data_paths, mode='train',
                                               stats_file=stats_file)
            valid_dataset = MultiBranchDataset(valid_files, transform=val_transform,
                                               use_methods=tuple(config.model.fusion.use_methods),
                                               data_paths=data_paths, mode='test',
                                               stats_file=stats_file)
            test_dataset = MultiBranchDataset(test_files, transform=val_transform,
                                              use_methods=tuple(config.model.fusion.use_methods),
                                              data_paths=data_paths, mode='test',
                                              stats_file=stats_file)
        else:
            is_fusion = hasattr(config.model, 'fusion')
            use_methods = tuple(config.model.fusion.use_methods) if is_fusion else None
            
            train_dataset = SingleChannelDataset(train_files, transform=train_transform,
                                                 is_fusion=is_fusion, use_methods=use_methods,
                                                 mode='train', stats_file=stats_file)
            valid_dataset = SingleChannelDataset(valid_files, transform=val_transform,
                                                 is_fusion=is_fusion, use_methods=use_methods,
                                                 mode='test', stats_file=stats_file)
            test_dataset = SingleChannelDataset(test_files, transform=val_transform,
                                                is_fusion=is_fusion, use_methods=use_methods,
                                                mode='test', stats_file=stats_file)
        
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                                shuffle=True, num_workers=8, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.training.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                               shuffle=False, num_workers=8, pin_memory=True)
        
        # Create model, optimizer, scheduler, criterion
        model = get_model(config).to(device)
        
        optimizer = optim.AdamW(model.parameters(),
                               lr=config.training.optimizer.lr,
                               weight_decay=config.training.optimizer.weight_decay)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.training.scheduler.max_lr,
            steps_per_epoch=len(train_loader), epochs=config.training.epochs
        )
        
        criterion = nn.CrossEntropyLoss(weight=loss_weights,
                                      label_smoothing=config.training.loss.label_smoothing)
        
        # Create trainer
        trainer = Trainer(model, optimizer, criterion, scheduler, device,
                        max_norm=config.training.gradient.max_norm,
                        early_stop_patience=config.training.early_stop)
        
        # Train
        exp_name = f"{config.logging.project_name}_rs{random_state}"
        model_save_path = Path(config.logging.save_dir) / 'models' / f"{exp_name}.pth"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        history = trainer.train(train_loader, valid_loader,
                              config.training.epochs, str(model_save_path))
        
        # Test
        evaluator = Evaluator(model, device)
        test_results = evaluator.evaluate(test_loader, str(model_save_path))
        
        # Save results
        logger = ResultLogger(config.logging.save_dir, exp_name)
        logger.log_test_results(test_results)
        logger.save(subdir='test_results')
        
        print(f"\nRandom State {random_state} Results:")
        print(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"AUC: {test_results['metrics']['auc']:.4f}")
    
    print("\n" + "="*100)
    print("Training Complete!")
    print("="*100)


if __name__ == '__main__':
    main()

