"""Evaluation metrics calculation"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics for binary classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (N, 2) array
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Binary classification metrics
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred_proba[:, 1])),
            'confusion_matrix': cm.tolist()
        }
    else:
        # Multi-class metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')),
            'confusion_matrix': cm.tolist()
        }
    
    return metrics

