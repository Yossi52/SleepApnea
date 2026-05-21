"""
10-Fold Cross-Validation Training Script

Supports:
- Baseline models (single transformation)
- Early fusion models (3-channel selection from fusion image)
- Late fusion models (3 separate branches)
- Transformer fusion models (3 branches + transformer)
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
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import json

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


def main():
    parser = argparse.ArgumentParser(description='10-Fold Cross-Validation Training')
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
        # Single method directory
        all_files = load_data(config.dataset.data_dir, file_ext='.npy', 
                             exclude_c05=config.dataset.get('exclude_c05', False))
    else:
        # Multi-branch: load from base directory
        all_files = load_data(os.path.join(config.dataset.data_dir, 'fusion'), 
                             file_ext='.npy',
                             exclude_c05=config.dataset.get('exclude_c05', False))
    
    # Extract labels
    y = [0 if 'label0' in f else 1 for f in all_files]
    
    print(f"\nTotal files: {len(all_files)}")
    print(f"Label 0: {y.count(0)}, Label 1: {y.count(1)}")
    
    # Run 10-fold CV for each random state
    for random_state in config.experiment.random_states:
        print("\n" + "="*100)
        print(f"Random State: {random_state}")
        print("="*100)
        
        skf = StratifiedKFold(n_splits=config.experiment.n_folds, 
                            shuffle=True, random_state=random_state)
        
        fold_results = []
        
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(all_files, y)):
            print(f"\n{'='*80}")
            print(f"Fold {fold+1}/{config.experiment.n_folds}")
            print(f"{'='*80}")
            
            # Split data
            test_files = [all_files[i] for i in test_idx]
            test_y = [y[i] for i in test_idx]
            
            train_val_files = [all_files[i] for i in train_val_idx]
            train_val_y = [y[i] for i in train_val_idx]
            
            train_files, valid_files, train_y, valid_y = train_test_split(
                train_val_files, train_val_y, test_size=1/9, 
                stratify=train_val_y, random_state=random_state + fold
            )
            
            print(f"Train: {len(train_files)}, Val: {len(valid_files)}, Test: {len(test_files)}")
            
            # Compute loss weights
            weight_0 = len(train_y) / (2 * train_y.count(0))
            weight_1 = len(train_y) / (2 * train_y.count(1))
            loss_weights = torch.tensor([weight_0, weight_1]).to(device)
            
            # Create datasets and dataloaders
            stats_file = f"stats/{config.logging.project_name}_rs{random_state}_fold{fold+1}.npz"
            
            if config.model.type in ['late', 'transformer']:
                # Multi-branch dataset
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
                # Single channel dataset
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
            
            # Create model
            model = get_model(config)
            model = model.to(device)
            
            # Create optimizer
            if config.training.optimizer.name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), 
                                       lr=config.training.optimizer.lr,
                                       weight_decay=config.training.optimizer.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {config.training.optimizer.name}")
            
            # Create scheduler
            if config.training.scheduler.name == 'OneCycleLR':
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=config.training.scheduler.max_lr,
                    steps_per_epoch=len(train_loader), epochs=config.training.epochs
                )
            else:
                scheduler = None
            
            # Create criterion
            criterion = nn.CrossEntropyLoss(weight=loss_weights, 
                                          label_smoothing=config.training.loss.label_smoothing)
            
            # Create trainer
            trainer = Trainer(model, optimizer, criterion, scheduler, device,
                            max_norm=config.training.gradient.max_norm,
                            early_stop_patience=config.training.early_stop)
            
            # Train
            exp_name = f"{config.logging.project_name}_rs{random_state}_fold{fold+1}"
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
            
            fold_results.append(test_results['metrics'])
            
            print(f"\nFold {fold+1} Results:")
            print(f"Accuracy: {test_results['metrics']['accuracy']:.4f}")
            print(f"AUC: {test_results['metrics']['auc']:.4f}")
        
        # Compute average results
        print("\n" + "="*100)
        print(f"Average Results (Random State {random_state})")
        print("="*100)
        
        avg_metrics = {}
        for key in fold_results[0].keys():
            if key != 'confusion_matrix':
                values = [r[key] for r in fold_results]
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
                avg_metrics[f'std_{key}'] = float(np.std(values))
        
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Save summary
        summary_path = Path(config.logging.save_dir) / 'test_results' / f"summary_rs{random_state}.json"
        with open(summary_path, 'w') as f:
            json.dump({'avg_metrics': avg_metrics, 'fold_results': fold_results}, f, indent=2)
    
    print("\n" + "="*100)
    print("Training Complete!")
    print("="*100)


if __name__ == '__main__':
    main()

