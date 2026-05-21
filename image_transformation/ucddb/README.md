# Image Transformation

## Overview

Transform preprocessed 1D ECG signals into 2D images using six different methods.

## Transformation Methods

1. **GADF (Gramian Angular Difference Field)**
2. **GASF (Gramian Angular Summation Field)**
3. **MTF (Markov Transition Field)**
4. **RP (Recurrence Plot)**
5. **Scalogram (Continuous Wavelet Transform)**
6. **Spectrogram (Short-Time Fourier Transform)**

All images are resized to 224x224 pixels and converted to RGB format using viridis colormap.

## Usage

### Step 1: Transform to Single-Method Images

```bash
# For Physionet (100Hz sampling)
python image_trans_single_method.py \
    --input_dir /path/to/preprocessed/physionet \
    --output_dir /path/to/images/physionet \
    --fs 100

# For UCDDB (128Hz sampling)
python image_trans_single_method.py \
    --input_dir /path/to/preprocessed/ucddb \
    --output_dir /path/to/images/ucddb \
    --fs 128
```

**Output structure:**
```
output_dir/
├── gadf/
│   ├── record_seg0001_label0.npy
│   └── ...
├── gasf/
├── mtf/
├── rp/
├── scalogram/
└── spectrogram/
```

### Step 2: Create Fusion Images

```bash
python make_fusion_image.py \
    --input_dir /path/to/images/physionet \
    --output_dir /path/to/images/physionet/fusion
```

**Output:**
- 6-channel fusion images (H, W, 6)
- Channel order: [GADF, GASF, MTF, RP, Scalogram, Spectrogram]

## Image Format

- **Single-method images**: (224, 224, 3) - RGB format
- **Fusion images**: (224, 224, 6) - 6-channel format
- All saved as `.npy` files (NumPy arrays)

## Notes

- Processing can be memory-intensive for long recordings
- Images are saved with the same filename as input segments
- Failed transformations are skipped with error messages

