"""
Trainer class for model training with early stopping and checkpointing
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score
import numpy as np


class Trainer:
    """Model trainer with mixed precision and early stopping"""
    
    def __init__(self, model, optimizer, criterion, scheduler=None, 
                 device='cuda', max_norm=0.5, early_stop_patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.max_norm = max_norm
        self.early_stop_patience = early_stop_patience
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        
        # Training state
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.early_stop_counter = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        train_true = []
        train_pred_prob = []
        
        for X, Y in tqdm(train_loader, desc='Training'):
            X, Y = X.to(self.device), Y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast(self.device, dtype=torch.float16):
                outputs = self.model(X)
                loss = self.criterion(outputs, Y)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()
            
            # Step scheduler if OneCycleLR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
            
            # Compute metrics
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item()
            running_acc += (preds == Y).float().mean().item()
            
            train_true.extend(Y.cpu().numpy())
            train_pred_prob.extend(probabilities.detach().cpu().numpy())
        
        # Compute epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        epoch_auc = roc_auc_score(train_true, np.array(train_pred_prob)[:, 1])
        
        return {
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'train_auc': epoch_auc
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        val_true = []
        val_pred_prob = []
        
        with torch.no_grad():
            for X, Y in tqdm(val_loader, desc='Validation'):
                X, Y = X.to(self.device), Y.to(self.device)
                
                with torch.amp.autocast(self.device, dtype=torch.float16):
                    outputs = self.model(X)
                    loss = self.criterion(outputs, Y)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item()
                running_acc += (preds == Y).float().mean().item()
                
                val_true.extend(Y.cpu().numpy())
                val_pred_prob.extend(probabilities.cpu().numpy())
        
        # Compute epoch metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = running_acc / len(val_loader)
        epoch_auc = roc_auc_score(val_true, np.array(val_pred_prob)[:, 1])
        
        return {
            'val_loss': epoch_loss,
            'val_acc': epoch_acc,
            'val_auc': epoch_auc
        }
    
    def train(self, train_loader, val_loader, epochs, save_path):
        """Full training loop"""
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("="*80)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Step scheduler (if not OneCycleLR)
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}, "
                  f"Train AUC: {train_metrics['train_auc']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}, "
                  f"Val AUC: {val_metrics['val_auc']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for improvement
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.early_stop_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Model saved (Val Loss: {self.best_loss:.4f})")
            else:
                self.early_stop_counter += 1
                print(f"No improvement ({self.early_stop_counter}/{self.early_stop_patience})")
            
            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nBest model from epoch {self.best_epoch+1} with Val Loss: {self.best_loss:.4f}")
        return {'best_epoch': self.best_epoch}

