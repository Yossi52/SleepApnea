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

---

## Attention Analysis (Transformer Fusion)

Which transformation branch does the transformer actually attend to, and how peaked is that
attention? Two steps: extract, then aggregate.

```bash
# 1. Attention weights of every test segment, per (seed, fold)
python analysis/extract_attention.py --config config/physionet_10fold_transformer.yaml
#    -> <save_dir>/attention_analysis/<project>_rs<seed>_fold<k>_attention.npz
#       attention (N, num_layers, nhead, 3, 3), labels, logits, channel_names

# 2. Aggregate over folds and seeds
python analysis/analyze_attention.py --attn_dir results/attention_analysis
#    -> results/attention_analysis/attention_summary.json + a printed table
```

`extract_attention.py` re-uses the training config, so the fold split, dataset and
normalization statistics are the ones used at training time. `nn.TransformerEncoderLayer`
discards its attention weights, so the script wraps each layer's `self_attn` to keep them
(and disables the fused fast path, which would bypass that wrapper). Check that mechanism
without any data or checkpoints:

```bash
python analysis/extract_attention.py --self_test
```

Reported by `analyze_attention.py`:

| Metric | Meaning |
|---|---|
| `branch_importance_offdiag_last_layer` | attention each branch receives from the *other* branches, last layer |
| `received_attention_by_layer` | attention received per branch and layer (uniform = 0.3333) |
| `layer_entropy_norm` | attention entropy per layer, normalised by log 3 (1.0 = uniform) |
| `layer_l1_dev_uniform` | L1 distance from the uniform 1/3 distribution (max 4/3) |
| `layer_self_attention` | diagonal share, i.e. how much a branch attends to itself |

---

## Recording-level AHI (Apnea-ECG withheld set)

Per-minute segment predictions aggregated into an apnea-hypopnea index per recording, then
compared with the ground-truth AHI of the withheld records (x01-x35): MAE/RMSE, Bland-Altman
and scatter plots, and severity-class agreement.

Input is the per-segment CSV written by `experiments/train_subject_wise.py`
(`<save_dir>/test_results/<exp_name>_test_segment_predictions.csv`), so the predictions come
from exactly the preprocessing path used at training time. Several CSVs can be passed at
once; each becomes one (model type, seed) group and metrics are reported as mean +- sd.

```bash
python analysis/ahi_from_segment_csv.py \
    --pred_csv "results/test_results/*_test_segment_predictions.csv" \
    --addinfo /path/to/apnea-ecg/additional-information.txt \
    --annos   /path/to/apnea-ecg/test-dataset-annos.txt \
    --output_dir results/ahi
```

Both annotation files ship with the PhysioNet Apnea-ECG database.

---

## Error Complementarity

Do different transformations fail on different segments -- i.e. is there anything for fusion
to exploit? Read-only, consumes the test-result JSONs the training scripts already wrote.

```bash
python analysis/error_complementarity.py \
    --pred rp=results/test_results/*rp*_rs*_fold*.json \
    --pred gadf=results/test_results/*gadf*_rs*_fold*.json \
    --pred scalogram=results/test_results/*scalogram*_rs*_fold*.json \
    --pred gasf=... --pred mtf=... --pred spectrogram=... \
    --pred early_fusion=... --pred late_fusion=... --pred hybrid_fusion=... \
    --branches rp,gadf,scalogram \
    --out_dir results/error_complementarity
```

Files are grouped by the `rs<seed>` / `fold<n>` tokens in their names; folds are concatenated
per seed and the results reported as mean +- sd across seeds. True labels must agree across
models inside a group, otherwise the group is excluded (logged).

Output:

| File | Content |
|---|---|
| `pair_metrics_raw.csv` | per seed and pair: disagreement, double fault, Yule's Q, rho, 2-model oracle |
| `pair_metrics_summary.csv` | the same, mean +- sd over seeds |
| `oracle_bounds.csv` | best single branch, 3-model oracle, majority/soft vote, measured fusion accuracies |
| `difficulty_histogram.csv` | fraction of segments that k models get wrong, and the apnea share of each bucket |
| `run_log.txt` | full log, including every excluded or incomplete group |

The names `early_fusion`, `late_fusion` and `hybrid_fusion` are treated as reference models:
they appear in the oracle table but not in the pairwise analysis.
