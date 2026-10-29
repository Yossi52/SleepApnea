# Image transformation

```bash
# six viridis RGB images per segment -> <images>/<transform>_drop/
python image_transformation/image_trans_single_method.py \
    --input_dir /path/to/segments/physionet --output_dir /path/to/images/physionet --fs 100
# (UCDDB: --fs 128)

# 6-channel grayscale fusion image -> <images>/fusion_drop/
python image_transformation/make_fusion_image.py --image_dir /path/to/images/physionet
```

| Transform | Settings |
|---|---|
| GADF / GASF | `pyts` GramianAngularField, method `d` / `s` |
| MTF | `pyts` MarkovTransitionField, 5 bins |
| RP | `pyts` RecurrencePlot, defaults |
| Scalogram | complex Morlet `cmor1.0-1.0`, 1,500 log-spaced frequencies from 0.5 Hz to fs/2 |
| Spectrogram | STFT, Blackman window of fs/4 samples, 50% overlap, PSD scaling |

Each matrix is min-max normalized, mapped through viridis and resized to 224x224 (Lanczos).
The fusion image stores the PIL luma (`convert('L')`) of each viridis image, channels ordered
gadf, gasf, mtf, rp, scalogram, spectrogram. Early, late and hybrid fusion use three of these
channels; the `mb_*` encoding control feeds the viridis RGB images instead.

Sensitivity analysis: `--sp_window` (STFT window in samples, default fs/4) and
`--sc_bandwidth` (Morlet bandwidth, default 1.0). Write each setting to its own `--output_dir`.

`--suffix ''` names the folders without `_drop`, for segments made with `--no_drop`.
