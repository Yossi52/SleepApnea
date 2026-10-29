# Analysis

All scripts read what `experiments/train.py` wrote (`results/runs/*.json`, checkpoints,
statistics, segment CSVs). None of them retrains a network. Scripts that rerun a model
rebuild the run's test set from the config stored in its JSON.

| Script | Question |
|---|---|
| `select_fusion_inputs.py` | Which three transforms go into fusion? (validation data only) |
| `branch_ablation.py` | How much does each branch contribute at test time? |
| `branch_probe.py` | How much does each fusion branch encode on its own, compared with a single model? |
| `extract_attention.py`, `analyze_attention.py` | Where does the hybrid model's attention go? |
| `cka_analysis.py` | Do single models trained on different transforms learn similar features? |
| `error_complementarity.py` | Do the transforms fail on different segments? |
| `ahi_from_segment_csv.py` | How accurate is the recording-level AHI on the withheld set? |

## Fusion input selection

```bash
python analysis/select_fusion_inputs.py --runs "results/runs/withheld_*_single_rs*.json"
```

For each transform, the minimum validation loss over epochs is averaged over seeds; the three
lowest are the fusion inputs, and their rank order is the early-fusion channel order (R, G, B).
Prints `--channel_order a,b,c` for `train.py` and writes `results/fusion_inputs.json`. Test
metrics are not read. In the paper (10 seeds): scalogram 0.329, GADF 0.354, spectrogram 0.355,
RP 0.357, GASF 0.380, MTF 0.393.

## Branch ablation

```bash
python analysis/branch_ablation.py --runs "results/runs/*_hybrid_*.json" --out results/branch_ablation.csv
```

Each branch of an early/late/hybrid run is replaced at test time in two ways: `zero` (0 after
normalization, i.e. the training mean) and `shuffle` (the same transform from another test
segment, one fixed derangement with seed 20261026). One row per run x branch x replacement,
plus the unmodified `none` row and the accuracy recorded at training time as a check.

## Branch linear probe

```bash
python analysis/branch_probe.py \
    --runs "results/runs/10fold_*_late_*.json" "results/runs/10fold_*_hybrid_*.json" \
           "results/runs/10fold_*_single_*.json" --out results/branch_probe.csv
```

GAP features (1,280-d) of every branch, for the run's own train/valid/test segments; for
hybrid models, before the channel embedding is added. Probe: StandardScaler (train) -> L2
logistic regression, C from {0.01, 0.1, 1, 10} by validation AUC -> test AUC and accuracy.
Pair a fusion branch with the single model of the same transform, seed and fold.

## Attention (hybrid fusion)

```bash
python analysis/extract_attention.py --runs "results/runs/*_hybrid_*.json" --out results/attention_analysis
python analysis/analyze_attention.py --attn_dir results/attention_analysis
python analysis/extract_attention.py --self_test      # no data needed
```

`extract_attention.py` saves the attention of every test segment per run
(`attention` (N, layers, heads, 3, 3), rows = query branch). `analyze_attention.py` reports,
per layer and averaged over runs:

| Metric | Meaning |
|---|---|
| `received_attention_by_layer` | attention each branch receives (uniform = 1/3) |
| `branch_importance_offdiag_last_layer` | attention a branch receives from the other branches, last layer |
| `layer_entropy_norm` | entropy / log 3 (1 = uniform) |
| `layer_l1_dev_uniform` | L1 distance from uniform (max 4/3) |
| `layer_self_attention` | diagonal share |

## CKA

```bash
python analysis/cka_analysis.py --data_dir /path/to/images --model_dir results/checkpoints \
    --stats_file /path/to/stats.npz --cka_types linear rbf --viz_mode all --output_dir results/cka
```

Layer-wise (early/mid/late) 6x6 CKA between the single-transform models of one seed and fold,
on the validation segments of that fold. `--norm_type` (`zscore`, `l2`, `both`) normalizes the
features first; `both` is the default and works for linear and RBF CKA. `--max_samples` caps
the number of segments (default 500). Figures and `cka_results.npz` go to `--output_dir`.

## Error complementarity

```bash
python analysis/error_complementarity.py \
    --pred scalogram="results/runs/10fold_scalogram_single_*.json" \
    --pred gadf="results/runs/10fold_gadf_single_*.json" \
    --pred spectrogram="results/runs/10fold_spectrogram_single_*.json" \
    --pred hybrid_fusion="results/runs/10fold_*_hybrid_*.json" \
    --branches scalogram,gadf,spectrogram --out_dir results/error_complementarity
```

Pairwise disagreement, double fault, Yule's Q and oracle accuracy; folds are pooled per seed,
results are mean +- sd over seeds. `early_fusion`, `late_fusion` and `hybrid_fusion` are only
reported as reference accuracies.

## Recording-level AHI

```bash
python analysis/ahi_from_segment_csv.py \
    --pred_csv "results/predictions/withheld_*_test_segment_predictions.csv" \
    --addinfo /path/to/apnea-ecg/additional-information.txt \
    --annos   /path/to/apnea-ecg/test-dataset-annos.txt --output_dir results/ahi
```

Predicted apnea minutes per hour of recording (x01-x35) are compared with the reference AHI
(MAE, RMSE, Bland-Altman, severity agreement). Both annotation files ship
with the PhysioNet Apnea-ECG database.
