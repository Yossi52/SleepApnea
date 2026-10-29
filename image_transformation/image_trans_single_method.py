"""ECG segments -> six 224x224 viridis RGB images (GADF, GASF, MTF, RP, scalogram, spectrogram).

Reads the per-segment .npy files written by preprocessing/ and saves
<output_dir>/<transform><suffix>/<record>_segment<NNN>_label<0|1>.npy.

--sp_window and --sc_bandwidth change the STFT window length and the Morlet bandwidth; the
defaults are the paper's settings and the other values were used for the sensitivity analysis.

Usage:
    python image_transformation/image_trans_single_method.py --input_dir SEGMENTS --output_dir IMAGES --fs 100
    python image_transformation/image_trans_single_method.py ... --fs 128          # UCDDB
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib
import numpy as np
import pywt
import scipy.signal
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from tqdm import tqdm


def to_image(matrix, size=(224, 224)):
    """Min-max normalize, map through viridis, resize with Lanczos."""
    norm = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
    rgb = (matplotlib.colormaps['viridis'](norm)[:, :, :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize(size, Image.LANCZOS))


def scalogram(signal, fs, bandwidth):
    freqs = np.geomspace(0.5, fs // 2, num=1500)
    coef, _ = pywt.cwt(signal, fs / (2 * freqs * np.pi), f'cmor{bandwidth}-1.0')
    return np.flip(np.abs(coef) ** 2, axis=0)


def spectrogram(signal, fs, window):
    stft = scipy.signal.ShortTimeFFT(win=scipy.signal.windows.blackman(window, sym=True), hop=window // 2,
                                     fs=fs, scale_to='psd')
    return np.flip(stft.spectrogram(signal), axis=0)


def transform_all(signal, fs, sp_window, sc_bandwidth):
    x = signal.reshape(1, -1)
    return {
        'gadf': GramianAngularField(method='d').transform(x)[0],
        'gasf': GramianAngularField(method='s').transform(x)[0],
        'mtf': MarkovTransitionField(n_bins=5).transform(x)[0],
        'rp': RecurrencePlot().transform(x)[0],
        'scalogram': scalogram(signal, fs, sc_bandwidth),
        'spectrogram': spectrogram(signal, fs, sp_window),
    }


def process(path, output_dir, suffix, fs, sp_window, sc_bandwidth):
    name = os.path.basename(path)
    for method, matrix in transform_all(np.load(path), fs, sp_window, sc_bandwidth).items():
        np.save(os.path.join(output_dir, f'{method}{suffix}', name), to_image(matrix))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--fs', type=int, default=100, help='100 for Apnea-ECG, 128 for UCDDB')
    p.add_argument('--suffix', default='_drop', help="'_drop' for quality-controlled segments, '' otherwise")
    p.add_argument('--sp_window', type=int, default=None, help='STFT window in samples (default fs // 4)')
    p.add_argument('--sc_bandwidth', type=float, default=1.0, help='complex Morlet bandwidth')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    files = sorted(os.path.join(r, f) for r, _, fs_ in os.walk(args.input_dir) for f in fs_
                   if f.endswith('.npy') and '_segment' in f)
    for method in ('gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram'):
        os.makedirs(os.path.join(args.output_dir, f'{method}{args.suffix}'), exist_ok=True)
    job = partial(process, output_dir=args.output_dir, suffix=args.suffix, fs=args.fs,
                  sp_window=args.sp_window or args.fs // 4, sc_bandwidth=args.sc_bandwidth)
    with ProcessPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(job, files, chunksize=16), total=len(files)))


if __name__ == '__main__':
    main()
