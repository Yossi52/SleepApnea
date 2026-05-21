# Multi-Channel Image Fusion for Sleep Apnea Detection

This repository contains the implementation of multi-channel image fusion methods for sleep apnea detection using ECG signals, supporting both Physionet Apnea-ECG and UCDDB datasets.

## Features

- **Multiple Datasets**: Physionet Apnea-ECG and UCDDB
- **Two Evaluation Protocols**: 10-fold cross-validation and subject-wise split
- **Various Model Architectures**:
  - Baseline (single transformation method)
  - Early Fusion (3-channel fusion image)
  - Late Fusion (multi-branch architecture)
  - Transformer Fusion (attention-based fusion)
- **Six Image Transformation Methods**: GADF, GASF, MTF, RP, Scalogram, Spectrogram

## Directory Structure

```
image_fusion_for_git/
├── config/                    # YAML configuration files
├── preprocessing/             # Data preprocessing scripts
│   ├── physionet/
│   └── ucddb/
├── image_transformation/      # Image transformation scripts
│   ├── physionet/
│   └── ucddb/
├── src/                      # Source code
│   ├── datasets/             # Dataset classes
│   ├── models/               # Model architectures
│   ├── training/             # Training and evaluation
│   └── utils/                # Utility functions
├── experiments/              # Training scripts
├── analysis/                 # Post-hoc analysis (CKA similarity)
└── results/                  # Output directory
```

## Datasets

This project uses two publicly available sleep apnea detection datasets:

### 1. PhysioNet Apnea-ECG Database

**Description**: Contains ECG recordings from 70 subjects (35 with apnea, 35 normal). Each recording is approximately 8 hours long.

**Access**: 
- **URL**: https://physionet.org/content/apnea-ecg/1.0.0/
- **Download**: `wget -r -N -c -np https://physionet.org/files/apnea-ecg/1.0.0/`


### 2. UCD Sleep Apnea Database (UCDDB)

**Description**: Contains polysomnographic recordings from 25 subjects with suspected sleep-disordered breathing. Includes ECG, respiratory effort, and oxygen saturation.

**Access**:
- **URL**: https://physionet.org/content/ucddb/1.0.0/
- **Download**: `wget -r -N -c -np https://physionet.org/files/ucddb/1.0.0/`

**Important Note**: UCDDB data files (`.rec` format) may require format conversion to be compatible with standard signal processing libraries. You may need to convert these files to an appropriate format before preprocessing.

**Note**: Please cite these datasets if you use this code in your research.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd image_fusion_for_git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

#### For Physionet Apnea-ECG:
```bash
# Preprocessing (filtering and segmentation)
python preprocessing/physionet/filter_and_segment.py \
    --input_dir /path/to/physionet/raw \
    --output_dir /path/to/physionet/preprocessed

# Image transformation
python image_transformation/physionet/image_trans_single_method.py \
    --input_dir /path/to/physionet/preprocessed \
    --output_dir /path/to/physionet/images

# Create fusion images
python image_transformation/physionet/make_fusion_image.py \
    --input_dir /path/to/physionet/images \
    --output_dir /path/to/physionet/images/fusion
```

#### For UCDDB:
```bash
# Step 0: Generate labels (UCDDB only)
python preprocessing/ucddb/ucddb_labeling.py \
    --data_dir /path/to/ucddb/raw \
    --output_dir /path/to/ucddb/labels

# Step 1-3: Same as Physionet
python preprocessing/ucddb/filter_and_segment.py --input_dir ... --output_dir ...
python image_transformation/ucddb/image_trans_single_method.py --input_dir ... --output_dir ...
python image_transformation/ucddb/make_fusion_image.py --input_dir ... --output_dir ...
```

### 2. Training

#### 10-Fold Cross-Validation
```bash
# Baseline model
python experiments/train_10fold.py --config config/physionet_10fold_baseline.yaml

# Transformer Fusion model
python experiments/train_10fold.py --config config/physionet_10fold_transformer.yaml
```

#### Subject-wise Split
```bash
python experiments/train_subject_wise.py --config config/physionet_subject_wise_transformer.yaml
```

## Configuration

Edit YAML files in `config/` directory to customize experiments:

```yaml
# Example: config/physionet_10fold_transformer.yaml
dataset:
  name: physionet
  data_dir: ./images/physionet

model:
  type: transformer  # baseline, late, or transformer
  backbone: efficientnet_b0
  pretrained: true
  fusion:
    use_methods: [rp, gadf, scalogram]

training:
  batch_size: 32
  epochs: 100
  early_stop: 10
```

See `config/` directory for more examples.

## Results

Results are saved in JSON format under `results/` directory:
- `results/histories/`: Training history for each fold
- `results/test_results/`: Test metrics for each fold
- `results/attention_analysis/`: Attention analysis (Transformer models only)

## Post-hoc Analysis

### CKA (Centered Kernel Alignment) Similarity

Analyze representational similarity between models trained on different transformation methods. Computes layer-wise (Early/Mid/Late) CKA matrices to investigate whether different input transformations lead to similar or distinct internal representations.

```bash
python analysis/cka_analysis.py \
    --data_dir /path/to/images \
    --model_dir /path/to/models \
    --stats_file /path/to/normalization_stats.npz \
    --cka_types linear rbf \
    --viz_mode all \
    --output_dir results/analysis
```

See [`analysis/README.md`](analysis/README.md) for detailed usage and options.

## Model Types

### Baseline
Single EfficientNet-B0 with one transformation method (e.g., RP, Scalogram)

### Early Fusion
Single EfficientNet-B0 with 3-channel fusion image selected from 6 methods

### Late Fusion
Three independent EfficientNet-B0 branches, features concatenated before classification

### Transformer Fusion
Three EfficientNet-B0 branches with Transformer encoder for attention-based fusion

