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
4. Saves results to JSON files, plus a per-segment prediction CSV
   (`<exp_name>_test_segment_predictions.csv`) used by `analysis/ahi_from_segment_csv.py`

### `train_grouped_ucddb.py`
Record-wise (subject-grouped) 10-fold cross-validation on UCDDB.

**Why it exists:** segment-wise 10-fold puts segments of the same recording in both train and
test, so its accuracy is optimistic. This script splits on the subject group instead:
`StratifiedGroupKFold` for train+val / test, plus a subject-disjoint validation hold-out
inside train+val. Train, validation and test recordings never overlap - asserted every fold.
Everything else (architectures, optimizer, scheduler, augmentation, training numerics) is the
same as the segment-wise runs, so the two protocols are directly comparable.

**Usage:**
```bash
# Print the fold/subject split and class balance, then exit (no training)
python experiments/train_grouped_ucddb.py --data_dir /path/to/ucddb/images \
    --model_type single --image_trans scalogram --dry_run

# One model, one seed, all 10 folds
python experiments/train_grouped_ucddb.py --data_dir /path/to/ucddb/images \
    --model_type single --image_trans scalogram --random_state 128 --gpu 0

# Fusion models: --channel_order picks the three branches
python experiments/train_grouped_ucddb.py --data_dir /path/to/ucddb/images \
    --model_type hybrid --channel_order rp,gadf,scalogram --random_state 128 --gpu 0
```

**`--data_dir` layout** (produced by `image_transformation/ucddb/`):
```
<data_dir>/
├── gadf_drop/ gasf_drop/ mtf_drop/ rp_drop/ scalogram_drop/ spectrogram_drop/
├── fusion_drop/          # 6-channel fusion image
└── gadf/ ... fusion/     # same without quality control, used by --no-seg_drop
```
Filenames must be `<subject>_segment<NNN>_label<0|1>.npy` - the subject prefix is the CV group.

**Model types:** `single` (one transformation), `early` (3 channels of the fusion image),
`late` / `hybrid` (three single-channel branches, concatenated / transformer-fused),
`mb_late` / `mb_hybrid` (three branches fed whole RGB images, one per transformation).

**Key options:**

| Option | Default | Meaning |
|---|---|---|
| `--random_state` | 128 | seeds the fold split and the validation hold-out |
| `--n_splits` | 10 | number of folds |
| `--pretrained` / `--no-pretrained` | on | ImageNet weights (ablation "+Pre") |
| `--seg_drop` / `--no-seg_drop` | on | use the quality-controlled `*_drop` dirs (ablation "+SQC") |
| `--resume` / `--no-resume` | on | skip finished folds, restore `last.pt` of an interrupted one |
| `--dry_run` | off | print the split and exit |
| `--max_folds` | all | run only the first N folds (smoke tests) |
| `--output_dir` | `results/ucddb_grouped` | results, logs, stats, state |
| `--ckpt_dir` | `results/ucddb_grouped/checkpoints` | per-fold `best.pt` / `last.pt` |

**Output tree:**
```
results/ucddb_grouped/
├── results/ucddb_grouped_10fold.csv      # one row per fold: accuracy, sens, spec, F1, AUC, runtime
├── results/fold_subject_mapping.csv      # which subject was in train/valid/test per fold
├── results/segment_predictions/<job>.csv # per-segment y_true / y_prob / y_pred
├── history/<job>_fold<k>.json            # per-epoch loss/acc/AUC
├── stats/<job>_fold<k>.npz               # normalization statistics (train split only)
├── state/run_state_<job>.json            # fold completion state, used by --resume
├── logs/<job>.log
└── env/                                  # torch/CUDA/platform snapshot + resolved config
```

Each invocation is one job (`<transform-or-channels>_<model_type>_rs<seed>`), and jobs write
to separate state and stats files, so several can run concurrently on different GPUs. SIGINT
and SIGTERM checkpoint the current epoch and exit; rerunning with `--resume` continues.


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
