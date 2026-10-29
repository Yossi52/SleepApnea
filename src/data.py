"""Segment images, train/valid/test splits, normalization statistics and augmentation.

Images are per-segment .npy files named <record>_segment<NNN>_label<0|1>.npy under
<data_dir>/<transform>[_drop]/ (viridis RGB, 224x224x3) and <data_dir>/fusion[_drop]/
(6 grayscale channels in TRANSFORMS order). `_drop` = segments kept by signal quality control.
"""
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

TRANSFORMS = ('gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram')
PROTOCOLS = ('withheld', '10fold')


def label_of(name):
    return int(name[-5])


def image_reader(data_dir, model_type, image_trans=None, channel_order=None, seg_drop=True):
    """Return (directory whose filenames define the segments, name -> uint8 array)."""
    sfx = '_drop' if seg_drop else ''
    if model_type == 'single':
        d = Path(data_dir) / f'{image_trans}{sfx}'
        return d, lambda name: np.load(d / name)
    if model_type.startswith('mb_'):
        dirs = [Path(data_dir) / f'{t}{sfx}' for t in channel_order]
        return dirs[0], lambda name: np.stack([np.load(d / name) for d in dirs])
    d = Path(data_dir) / f'fusion{sfx}'
    idx = [TRANSFORMS.index(t) for t in channel_order]
    return d, lambda name: np.load(d / name)[:, :, idx]


def list_segments(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith('.npy'))


def make_split(names, protocol, seed, fold=None):
    """train/valid/test filenames, as in the paper.

    withheld: Apnea-ECG x01-x35 are the test set; 20% of the rest (stratified) is validation.
    10fold:   StratifiedKFold(10, shuffle, seed), fold 1-10 is the test set; 1/9 of the rest
              (stratified, seed + fold - 1) is validation.
    """
    y = [label_of(n) for n in names]
    if protocol == 'withheld':
        rest = [n for n in names if not n.startswith('x')]
        train, valid = train_test_split(rest, test_size=int(len(rest) * 0.2),
                                        stratify=[label_of(n) for n in rest], random_state=seed)
        return train, valid, [n for n in names if n.startswith('x')]
    trainval_idx, test_idx = list(StratifiedKFold(10, shuffle=True, random_state=seed).split(names, y))[fold - 1]
    trainval = [names[i] for i in trainval_idx]
    train, valid = train_test_split(trainval, test_size=1 / 9, stratify=[y[i] for i in trainval_idx],
                                    random_state=seed + fold - 1)
    return train, valid, [names[i] for i in test_idx]


def run_inputs(cfg):
    """(reader, train, valid, test) for a run config; training and the analyses share this."""
    d, read = image_reader(cfg['data_dir'], cfg['model_type'], cfg.get('image_trans'),
                           cfg.get('channel_order'), cfg['seg_drop'])
    return (read, *make_split(list_segments(d), cfg['protocol'], cfg['seed'], cfg.get('fold')))


def compute_stats(read, names):
    """Per-channel mean/std over the training images (per branch for mb_* models)."""
    total = total_sq = 0.0
    for n in names:
        img = read(n).astype(np.float32)
        total = total + img.sum(axis=(-3, -2)).astype(np.float64)
        total_sq = total_sq + (img ** 2).sum(axis=(-3, -2)).astype(np.float64)
    pixels = len(names) * img.shape[-3] * img.shape[-2]
    mean = total / pixels
    return mean, np.sqrt(total_sq / pixels - mean ** 2)


class SegmentDataset(Dataset):
    """z-scored images as (C, H, W), or (3, C, H, W) for mb_* models."""

    def __init__(self, names, read, mean, std, transform=None):
        self.names, self.read, self.transform = names, read, transform
        self.mean, self.std = np.expand_dims(mean, (-3, -2)), np.expand_dims(std, (-3, -2))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = (self.read(self.names[i]).astype(np.float32) - self.mean) / self.std
        x = torch.from_numpy(np.moveaxis(img, -1, -3)).float()
        if self.transform:
            x = torch.stack([self.transform(b) for b in x]) if x.dim() == 4 else self.transform(x)
        return x, label_of(self.names[i])


class AdjustBrightness:
    def __init__(self, brightness_range):
        self.r = brightness_range

    def __call__(self, x):
        return x + torch.empty(1).uniform_(-self.r, self.r).item()


class AdjustContrast:
    def __init__(self, contrast_range):
        self.r = contrast_range

    def __call__(self, x):
        return x * torch.empty(1).uniform_(1 - self.r, 1 + self.r).item()


class AddGaussianNoise:
    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


train_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02),
                            interpolation=transforms.InterpolationMode.BILINEAR),
    AdjustBrightness(0.05),
    AdjustContrast(0.05),
    AddGaussianNoise(0.001),
])
valid_transform = transforms.Resize((224, 224), antialias=True)


if __name__ == '__main__':
    names = [f'{r}{i:02d}_segment{s:03d}_label{(i + s) % 2}.npy'
             for r in 'axbc' for i in range(1, 6) for s in range(20)]
    tr, va, te = make_split(names, 'withheld', seed=43)
    assert set(te) == {n for n in names if n[0] == 'x'} and not set(tr) & set(va)
    folds = [make_split(names, '10fold', 128, f)[2] for f in range(1, 11)]
    assert sorted(sum(folds, [])) == sorted(names)
    imgs = {n: np.full((4, 4, 3), label_of(n) * 10, np.uint8) for n in names[:4]}
    mean, std = compute_stats(imgs.__getitem__, names[:4])
    assert np.allclose(mean, 5) and np.allclose(std, 5)
    print('data OK: splits are disjoint and cover every segment')
