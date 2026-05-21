# Experiments

## Scripts

### `train_10fold.py`
10-fold cross-validation training for all model types.

**Usage:**
```bash
python experiments/train_10fold.py --config config/physionet_10fold_transformer.yaml --gpu 0
```

**What it does:**
1. Loads configuration from YAML file
2. Splits data into 10 folds (stratified)
3. For each fold:
   - Trains model on 8 folds
   - Validates on 1 fold
   - Tests on 1 fold
4. Computes average performance across folds
5. Saves results to JSON files

### `train_subject_wise.py`
Subject-wise split training (files with 'x' are test set).

**Usage:**
```bash
python experiments/train_subject_wise.py --config config/physionet_subject_wise_transformer.yaml --gpu 0
```

**What it does:**
1. Splits data by subject IDs ('x' files = test set)
2. Randomly splits remaining files into train/validation
3. Trains and evaluates model
4. Saves results to JSON files

## Configuration

All experiments are configured through YAML files in `config/` directory.

Key configuration parameters:
- `dataset.data_dir`: Path to image data
- `model.type`: Model architecture (baseline/late/transformer)
- `training.batch_size`, `training.epochs`, etc.

See example configs in `config/` directory.

## Output

Results are saved to `results/` directory:
- `results/models/`: Trained model weights (.pth files)
- `results/test_results/`: Test metrics and predictions (JSON files)
- `stats/`: Normalization statistics (.npz files)

## Notes

- GPU is automatically used if available
- Mixed precision training is enabled for faster training
- Early stopping is used to prevent overfitting
- All random seeds are fixed for reproducibility

