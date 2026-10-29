# Image Transformation and Fusion Strategies for ECG-Based Sleep Apnea Detection

Code for the paper. Single-lead ECG minutes are turned into six 2D images (GADF, GASF, MTF,
recurrence plot, scalogram, spectrogram), classified with EfficientNet-B0, and three of the
transforms are combined by early, late or hybrid (transformer) fusion. Evaluated on PhysioNet
Apnea-ECG and UCDDB under three protocols: the Apnea-ECG withheld set, segment-wise 10-fold,
and record-wise (subject-grouped) 10-fold on UCDDB.

## Layout

```
preprocessing/          filtering, quality control, 60 s segments (Apnea-ECG, UCDDB)
image_transformation/   segment -> six viridis images, and the 6-channel grayscale fusion image
src/                    models.py, data.py (splits, datasets, augmentation), engine.py (training loop)
experiments/            train.py (withheld, 10-fold), train_grouped_ucddb.py (record-wise)
analysis/               input selection, branch ablation and probe, attention, CKA, AHI, ...
```

## Setup

Python 3.9+, `pip install -r requirements.txt`. Data (PhysioNet):
- Apnea-ECG: https://physionet.org/content/apnea-ecg/1.0.0/
- UCDDB: https://physionet.org/content/ucddb/1.0.0/

## Pipeline

```bash
# 1. segments and images (UCDDB: preprocessing/ucddb/..., --fs 128)
python preprocessing/physionet/filter_and_segment.py --data_dir RAW --output_dir SEG
python image_transformation/image_trans_single_method.py --input_dir SEG --output_dir IMG --fs 100
python image_transformation/make_fusion_image.py --image_dir IMG

# 2. single transforms on the withheld split, then pick the fusion inputs from validation loss
for t in gadf gasf mtf rp scalogram spectrogram; do
  python experiments/train.py --data_dir IMG --protocol withheld --model_type single --image_trans $t \
      --seeds 43 128 154 239 374 451 562 687 716 895
done
python analysis/select_fusion_inputs.py --runs "results/runs/withheld_*_single_*.json"

# 3. fusion with the selected inputs (paper: scalogram,gadf,spectrogram)
python experiments/train.py --data_dir IMG --protocol 10fold --model_type hybrid \
    --channel_order scalogram,gadf,spectrogram --seeds 128 374 562
python experiments/train_grouped_ucddb.py --data_dir UCDDB_IMG --model_type hybrid --random_state 128

# 4. post-hoc analyses (no retraining)
python analysis/branch_ablation.py --runs "results/runs/*_hybrid_*.json"
python analysis/branch_probe.py --runs "results/runs/10fold_*_hybrid_*.json" "results/runs/10fold_*_single_*.json"
```

Details: [experiments/README.md](experiments/README.md) (protocols, model types,
hyperparameters, outputs) and [analysis/README.md](analysis/README.md).

## How the paper's experiments map to the code

| Experiment | Command |
|---|---|
| Single transforms, pretraining and quality-control ablation | `train.py --model_type single [--no-pretrained] [--no-seg_drop]` |
| Fusion inputs (top 3 by withheld validation loss) | `analysis/select_fusion_inputs.py` |
| Early / late / hybrid fusion | `train.py --model_type {early,late,hybrid}` |
| Early-fusion channel order | `train.py --model_type early --channel_order <permutation>` |
| Grayscale vs viridis branch inputs | `train.py --model_type {mb_late,mb_hybrid}` |
| Record-wise evaluation on UCDDB | `train_grouped_ucddb.py` |
| Scalogram / spectrogram parameter sensitivity | `image_trans_single_method.py --sc_bandwidth / --sp_window` |
| Branch removal and replacement, linear probe | `analysis/branch_ablation.py`, `analysis/branch_probe.py` |
| Attention, CKA, error complementarity, AHI | `analysis/` |

## Citation

```bibtex
@article{kang_image,
  title   = {Image Transformation and Fusion Strategies for ECG-Based Sleep Apnea Detection: A Representation Similarity Analysis},
  journal = {TBD},
  year    = {TBD}
}
```
