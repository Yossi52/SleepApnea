# Source Code

## Structure

```
src/
├── datasets/        # Dataset classes
├── models/          # Model architectures
├── training/        # Training and evaluation
└── utils/           # Utility functions
```

## Datasets (`datasets/`)

### `image_datasets.py`
- `SingleChannelDataset`: For baseline or early fusion models
- `MultiBranchDataset`: For late fusion or transformer fusion models
- `load_data`: Helper function to load image files

## Models (`models/`)

### `baseline_model.py`
Single EfficientNet-B0 for baseline experiments (single transformation method)

### `late_fusion_model.py`
Three EfficientNet-B0 branches with feature concatenation

### `transformer_fusion_model.py`
Three EfficientNet-B0 branches with Transformer encoder for attention-based fusion

## Training (`training/`)

### `trainer.py`
- `Trainer`: Training loop with mixed precision, early stopping, and checkpointing

### `evaluator.py`
- `Evaluator`: Model evaluation and metric calculation

## Utils (`utils/`)

### `config_loader.py`
- `load_config`: Load YAML configuration files

### `logger.py`
- `ResultLogger`: Save experiment results to JSON

### `metrics.py`
- `calculate_metrics`: Comprehensive metric calculation

### `transforms.py`
- Data augmentation transforms for training and validation

## Usage

These modules are imported by the experiment scripts (`experiments/train_10fold.py`, etc.).
Direct usage is not recommended.

