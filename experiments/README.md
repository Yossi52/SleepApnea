# Experiments

## `train.py`: withheld and segment-wise 10-fold

```bash
python experiments/train.py --data_dir DATA --protocol {withheld,10fold} --model_type TYPE \
    [--image_trans T | --channel_order a,b,c] [--seeds ...] [--folds ...]
```

`DATA` holds the folders written by `image_transformation/` (`<transform>_drop/`, `fusion_drop/`).

| Protocol | Test set | Validation |
|---|---|---|
| `withheld` | Apnea-ECG x01-x35 | stratified 20% of the remaining segments, seeded |
| `10fold` | fold k of `StratifiedKFold(10, shuffle, seed)` | stratified 1/9 of the other folds, seed + k - 1 |

| `--model_type` | Input | lr / OneCycle max_lr | grad clip | batch |
|---|---|---|---|---|
| `single` | one transform, viridis RGB | 5e-4 / 1e-3 | 1.0 | 16 |
| `early` | three fusion channels as one RGB image | 5e-4 / 1e-3 | 1.0 | 32 |
| `late` | one fusion channel per EfficientNet-B0 branch, features concatenated | 5e-6 / 1e-4 | 0.5 | 32 |
| `hybrid` | as `late`, fused by a 3-layer Pre-LN transformer (d=1280, 8 heads, dropout 0.2) | 5e-6 / 1e-4 | 0.5 | 32 |
| `mb_late`, `mb_hybrid` | as `late`/`hybrid`, each branch fed the viridis RGB image (encoding control) | 5e-6 / 1e-4 | 0.5 | 32 |

Common settings: ImageNet-pretrained EfficientNet-B0 (`--no-pretrained` to ablate), AdamW
(weight decay 1e-5), OneCycleLR (5% warm-up), weighted cross-entropy with label smoothing 0.1,
fp16 mixed precision, up to 100 epochs, early stopping after 10 epochs without a lower
validation loss. Normalized batches are divided by 255 whenever their maximum exceeds 1, as in
the paper's training code; checkpoints trained this way need the same step at inference.
`--channel_order` sets R, G, B (= branches 0, 1, 2); the default is the order
chosen by `analysis/select_fusion_inputs.py` in the paper.

Outputs under `--out_dir` (default `results/`): `runs/<job>.json` (config, per-epoch history,
test metrics and predictions), `predictions/<job>_test_segment_predictions.csv`, `stats/<job>.npz`,
`checkpoints/<job>.pt`. Runs that already have a JSON are skipped.

Segments are taken in sorted filename order. The paper's runs used `os.listdir` order, so the
same seed gives the same withheld test set but different 10-fold memberships.

## `train_grouped_ucddb.py`: record-wise 10-fold on UCDDB

Segment-wise folds put segments of one recording in both train and test. This script splits
on the subject instead: `StratifiedGroupKFold` for test, plus a subject-disjoint validation
hold-out (about 1/9 of the training subjects, at least 2). Overlap is asserted every fold.

```bash
# print the split and exit
python experiments/train_grouped_ucddb.py --data_dir DATA --model_type single --image_trans scalogram --dry_run
# one model, one seed, all folds
python experiments/train_grouped_ucddb.py --data_dir DATA --model_type hybrid \
    --channel_order scalogram,gadf,spectrogram --random_state 128 --gpu 0
```

Same models and optimizer as `train.py`, except that every model type uses batch 32 and
OneCycle `final_div_factor` 1e4, as in the paper's grouped runs.

| Option | Default | |
|---|---|---|
| `--random_state` | 128 | seeds the folds and the validation hold-out |
| `--pretrained` / `--seg_drop` | on | turn off for the ablation |
| `--resume` | on | skip finished folds, continue an interrupted one from `last.pt` |
| `--output_dir` | `results/ucddb_grouped` | results, history, stats, state, logs |
| `--ckpt_dir` | `results/ucddb_grouped/checkpoints` | per-fold `best.pt` / `last.pt` |

Results: `results/ucddb_grouped_10fold.csv` (one row per fold), `results/fold_subject_mapping.csv`,
`results/segment_predictions/<job>.csv`. Jobs keep separate state and stats files, so several
can run at once; SIGINT/SIGTERM save the current epoch.
