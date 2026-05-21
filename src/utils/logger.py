"""JSON logger for experiment results"""

import json
import os
from pathlib import Path
from datetime import datetime


class ResultLogger:
    """Logger for saving experiment results to JSON"""
    
    def __init__(self, save_dir, exp_name):
        self.save_dir = Path(save_dir)
        self.exp_name = exp_name
        self.history = {
            'experiment_name': exp_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_history': [],
            'val_history': [],
        }
    
    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch"""
        metrics['epoch'] = epoch
        
        # Separate train and val metrics
        train_metrics = {k: v for k, v in metrics.items() if 'train' in k or k == 'epoch'}
        val_metrics = {k: v for k, v in metrics.items() if 'val' in k or 'valid' in k or k == 'epoch'}
        
        if train_metrics:
            self.history['train_history'].append(train_metrics)
        if val_metrics:
            self.history['val_history'].append(val_metrics)
    
    def log_test_results(self, results):
        """Log test results"""
        self.history['test_results'] = results
    
    def log_attention_analysis(self, analysis):
        """Log attention analysis (for Transformer models)"""
        self.history['attention_analysis'] = analysis
    
    def save(self, subdir='histories'):
        """Save history to JSON file"""
        save_path = self.save_dir / subdir
        save_path.mkdir(parents=True, exist_ok=True)
        
        filepath = save_path / f"{self.exp_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return filepath

