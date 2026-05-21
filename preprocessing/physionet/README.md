# Preprocessing Scripts

## Overview

This directory contains scripts for preprocessing raw ECG signals from Physionet Apnea-ECG and UCDDB datasets.

## UCDDB Preprocessing

UCDDB requires two steps:

### Step 1: Generate Labels

```bash
python ucddb_labeling.py \
    --data_dir /path/to/ucddb/raw \
    --output_dir /path/to/ucddb/labels
```

**Required files in data_dir:**
- `ucddb001_respevt.txt` ~ `ucddb027_respevt.txt`
- `SubjectDetails.xls`

**Output:**
- Minute-level labels with 5-second overlap criterion

### Step 2: Filter and Segment

```bash
python filter_and_segment.py \
    --data_dir /path/to/ucddb/raw \
    --label_dir /path/to/ucddb/labels \
    --output_dir /path/to/ucddb/preprocessed
```

**Options:**
- `--no_drop`: Keep all segments (don't drop based on quality)

## Physionet Apnea-ECG Preprocessing

```bash
python filter_and_segment.py \
    --data_dir /path/to/physionet/raw \
    --output_dir /path/to/physionet/preprocessed \
    --exclude_c05
```

**Options:**
- `--no_drop`: Keep all segments (don't drop based on quality)
- `--exclude_c05`: Exclude record c05 from processing

## Preprocessing Steps

Both scripts apply the following preprocessing:

1. **45Hz Low-pass Filter**: Remove high-frequency noise
2. **0.5-10Hz Bandpass Filter**: Extract relevant ECG frequency range
3. **Median Filter**: Remove impulse noise
4. **Segment Dropping**: Remove segments with extreme kurtosis (>10) or skewness (>5)

## Output Format

Each processed segment is saved as a `.npy` file:
- Filename format: `{record}_{seg_id}_label{0/1}.npy`
- Contains 60-second ECG signal (6000 samples for Physionet @100Hz, 7680 samples for UCDDB @128Hz)
- Accompanied by `segment_info.csv` with metadata

