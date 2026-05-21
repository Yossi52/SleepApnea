"""
Evaluator class for model testing
"""

import torch
from tqdm import tqdm
import numpy as np
from ..utils.metrics import calculate_metrics


class Evaluator:
    """Model evaluator"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader, model_path=None):
        """
        Evaluate model on test set
        
        Args:
            test_loader: DataLoader for test set
            model_path: Path to saved model weights (if None, use current model)
            
        Returns:
            Dictionary of test metrics
        """
        # Load model if path provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.to(self.device)
        self.model.eval()
        
        true_labels = []
        pred_labels = []
        pred_probs = []
        
        with torch.no_grad():
            for X, Y in tqdm(test_loader, desc='Testing'):
                X, Y = X.to(self.device), Y.to(self.device)
                
                with torch.amp.autocast(self.device, dtype=torch.float16):
                    outputs = self.model(X)
                
                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Store results
                true_labels.extend(Y.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                pred_probs.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        pred_probs = np.array(pred_probs)
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, pred_labels, pred_probs)
        
        # Print results
        print("\n" + "="*80)
        print("Test Results:")
        print("-"*80)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        if 'sensitivity' in metrics:
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"AUC:         {metrics['auc']:.4f}")
        print("="*80)
        
        return {
            'metrics': metrics,
            'predictions': {
                'true_labels': true_labels.tolist(),
                'pred_labels': pred_labels.tolist(),
                'pred_probs': pred_probs.tolist()
            }
        }

