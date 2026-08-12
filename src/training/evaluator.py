"""
Evaluator class for model testing
"""

import csv
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from ..utils.metrics import calculate_metrics

# <record>_segment<NNN>_label<0|1>.npy
SEGMENT_NAME_RE = re.compile(r'^(?P<record>.+?)_segment(?P<segment>\d+)_label(?P<label>\d+)')


def save_segment_predictions(csv_path, file_list, true_labels, pred_labels, pred_probs):
    """Write one row per test segment.

    This is the input of `analysis/ahi_from_segment_csv.py`: recording-level AHI needs to
    know which recording and which minute each prediction belongs to, which only the
    filename carries.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset_index', 'file_path', 'record_id', 'segment_index',
                    'label_from_filename', 'y_true', 'y_pred', 'prob_class0', 'prob_class1'])
        for i, path in enumerate(file_list):
            m = SEGMENT_NAME_RE.match(os.path.basename(path))
            record = m['record'] if m else ''
            segment = int(m['segment']) if m else ''
            label = int(m['label']) if m else ''
            w.writerow([i, path, record, segment, label,
                        int(true_labels[i]), int(pred_labels[i]),
                        float(pred_probs[i][0]), float(pred_probs[i][1])])
    print(f"Segment predictions saved: {csv_path} ({len(file_list)} rows)")
    return csv_path


class Evaluator:
    """Model evaluator"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader, model_path=None, segment_csv_path=None):
        """
        Evaluate model on test set

        Args:
            test_loader: DataLoader for test set (must not be shuffled)
            model_path: Path to saved model weights (if None, use current model)
            segment_csv_path: If given, also write a per-segment prediction CSV there
                (input for analysis/ahi_from_segment_csv.py)

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

        if segment_csv_path:
            save_segment_predictions(segment_csv_path, test_loader.dataset.file_list,
                                     true_labels, pred_labels, pred_probs)

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

