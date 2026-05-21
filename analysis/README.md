# Analysis

## CKA (Centered Kernel Alignment) Analysis

Quantifies the representational similarity between models trained on different image transformation methods (GADF, GASF, MTF, RP, Scalogram, Spectrogram) using CKA.

### What it does

1. **Segment Matching**: Finds the same ECG segments across all 6 transformation methods using the validation set from stratified 10-fold CV.
2. **Feature Extraction**: Extracts layer-wise activations (Early, Mid, Late) from each method's trained EfficientNet-B0 model.
3. **CKA Computation**: Computes 6x6 CKA similarity matrices for each layer.
4. **Visualization**: Generates heatmaps, trend plots, and comparison figures.
5. **Robustness Analysis**: Compares Linear CKA and RBF Kernel CKA for result validation.

### Usage

```bash
# From the repository root
python analysis/cka_analysis.py \
    --data_dir /path/to/images \
    --model_dir /path/to/models \
    --stats_file /path/to/normalization_stats.npz \
    --output_dir results/analysis
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Root directory with per-method image subdirectories |
| `--model_dir` | (required) | Directory containing trained model checkpoints (.pth) |
| `--stats_file` | None | Path to normalization stats (.npz). If not provided, raw pixel values are scaled to [0, 1] |
| `--output_dir` | `results/analysis` | Directory to save figures and raw CKA matrices |
| `--random_state` | 374 | Random seed for reproducible data splits |
| `--fold_num` | 1 | Fold number to select validation set from |
| `--max_samples` | 500 | Maximum segments to use (0 = all) |
| `--batch_size` | 32 | Batch size for feature extraction |
| `--cka_types` | `linear rbf` | CKA types to compute (`linear`, `rbf`, or both) |
| `--norm_type` | `both` | Feature normalization: `zscore`, `l2`, or `both` |
| `--viz_mode` | `comparison` | Visualization mode (see below) |

### Visualization Modes

- **`single`**: One CKA type, 3 layer heatmaps side-by-side
- **`comparison`**: Linear vs RBF CKA heatmaps (3x2 grid)
- **`correlation`**: Scatter plot of Linear vs RBF CKA values per layer
- **`difference`**: Heatmap of (Linear - RBF) difference per layer
- **`all`**: Generate all of the above

### Output

```
results/analysis/
├── cka_linear_heatmap.png      # Linear CKA heatmaps (single mode)
├── cka_rbf_heatmap.png         # RBF CKA heatmaps (single mode)
├── cka_comparison.png          # Linear vs RBF comparison
├── cka_correlation.png         # Linear-RBF correlation scatter
├── cka_difference.png          # Linear-RBF difference heatmap
├── cka_layer_trend_linear.png  # Off-diagonal mean CKA bar chart
├── cka_layer_trend_rbf.png
└── cka_results.npz             # Raw CKA matrices for further analysis
```

### Feature Normalization

CKA is sensitive to feature scale differences across transformation methods. Three normalization options are provided:

- **`zscore`**: Feature-wise z-score normalization. Best for Linear CKA.
- **`l2`**: Sample-wise L2 normalization (projects each sample onto a unit sphere). Best for RBF Kernel CKA.
- **`both`**: Z-score followed by L2 normalization. Most robust option (recommended).

### Example

```bash
# Compute both Linear and RBF CKA, generate all visualizations
python analysis/cka_analysis.py \
    --data_dir ./data/physionet/images \
    --model_dir ./results/models \
    --stats_file ./results/stats/normalization_stats.npz \
    --cka_types linear rbf \
    --norm_type both \
    --viz_mode all \
    --max_samples 500

# Quick analysis with Linear CKA only
python analysis/cka_analysis.py \
    --data_dir ./data/physionet/images \
    --model_dir ./results/models \
    --cka_types linear \
    --viz_mode single \
    --max_samples 200
```
