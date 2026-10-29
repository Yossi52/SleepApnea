# Preprocessing

Raw ECG -> filtered, quality-controlled 60-second segments, one `.npy` per segment.

Filtering: 45 Hz low-pass, 0.5-10 Hz band-pass, median filter. Quality control drops segments
with |kurtosis| > 10 or |skewness| > 5 (`--no_drop` keeps everything).

## Apnea-ECG (100 Hz)

```bash
python preprocessing/physionet/filter_and_segment.py \
    --data_dir /path/to/apnea-ecg --output_dir /path/to/segments/physionet
```

The paper keeps record c05 (`--exclude_c05` drops it).

## UCDDB (128 Hz)

Labels first (needs `ucddbNNN_respevt.txt` and `SubjectDetails.xls`), then segments:

```bash
python preprocessing/ucddb/ucddb_labeling.py --data_dir /path/to/ucddb --output_dir /path/to/ucddb/labels
python preprocessing/ucddb/filter_and_segment.py \
    --data_dir /path/to/ucddb --label_dir /path/to/ucddb/labels --output_dir /path/to/segments/ucddb
```

A minute is labeled apnea when it overlaps a respiratory event by at least 5 seconds.

## Output

`<record>_segment<NNN>_label<0|1>.npy` (6,000 samples for Apnea-ECG, 7,680 for UCDDB) plus a
`segment_info.csv` per record. The record prefix is the subject group used by the
record-wise split, and `x*` records are the Apnea-ECG withheld test set.
