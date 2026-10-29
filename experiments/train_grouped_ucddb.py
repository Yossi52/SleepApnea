"""Record-wise (subject-grouped) 10-fold cross-validation on UCDDB.

Segment-wise 10-fold puts segments of the same recording in both train and test, so the
reported accuracy is optimistic. This script instead splits on the *subject* group:

  1. StratifiedGroupKFold on subject for the train+val / test split, plus a
     subject-disjoint validation hold-out inside train+val (subject_valid_holdout).
     Train, validation and test recordings never overlap (asserted every fold).
  2. Results, checkpoints and logs are written incrementally to a resumable output tree,
     so an interrupted job (`--resume`, on by default) continues where it stopped.

One invocation = 1 model x 1 seed x all folds. Models, augmentation and optimizer settings
match the segment-wise runs; only the split differs.

Inputs are the per-segment .npy images produced by `image_transformation/`, laid out as

    <data_dir>/
        gadf_drop/ gasf_drop/ mtf_drop/ rp_drop/ scalogram_drop/ spectrogram_drop/
        fusion_drop/                       # 6-channel fusion image
        gadf/ ... fusion/                  # same, without quality control (--no-seg_drop)

with filenames `<subject>_segment<NNN>_label<0|1>.npy`.

Usage:
    python experiments/train_grouped_ucddb.py --data_dir /path/to/ucddb/images \
        --model_type single --image_trans scalogram --random_state 128 --gpu 0

    python experiments/train_grouped_ucddb.py --data_dir /path/to/ucddb/images \
        --model_type hybrid --channel_order scalogram,gadf,spectrogram --dry_run
"""
import argparse
import csv
import json
import os
import platform
import random
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import TRANSFORMS, train_transform, valid_transform  # noqa: E402
from src.models import build_model  # noqa: E402

# UCDDB: 002_segment000_label0.npy   Apnea-ECG: a01_segment000_label0.npy
FNAME_RE = re.compile(r'^(?P<subject>.+?)_segment(?P<segment>\d+)_label(?P<label>\d+)\.npy$')


def parse_name(fname):
    """Filename -> (subject, segment index, label). The subject is the CV group."""
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f'unparseable filename: {fname}')
    return m['subject'], int(m['segment']), int(m['label'])


# Data: row-indexed image directories

FUSION_CHANNEL_MAPPING = {t: i for i, t in enumerate(TRANSFORMS)}


def is_multibranch(model_type):
    """mb_late / mb_hybrid: one whole RGB image per transform, one array per transform."""
    return model_type.startswith('mb_')


def build_fusion_image(fusion_arr, i, channel_order):
    """Select one channel per transform from the 6-channel fusion image, in the given order."""
    sel = [FUSION_CHANNEL_MAPPING[t] for t in channel_order]
    return np.array(fusion_arr[i])[:, :, sel]


class GroupedDataset(Dataset):
    """Indexes the per-transform image directories by row position.

    model_type == 'single': one packed array ({transform}_drop), all 3 channels used as-is.
    model_type in {early, late, hybrid}: the packed 6-channel fusion_drop array, with one
        channel per transform selected via FUSION_CHANNEL_MAPPING in channel_order -- the
        same construction the segment-wise dataset performs.
    model_type in {mb_late, mb_hybrid}: one packed array per transform in channel_order, kept
        as a full RGB image -> (3 branches, 3 RGB, H, W). mean/std are (3 branches, 3)
        because each branch has its own colormap and therefore its own statistics.
    """

    def __init__(self, arrays, channel_order, model_type, row_indices, labels,
                 mean, std, transform=None):
        self.arrays = arrays  # dict: key -> memmap array (H,W,3) uint8
        self.channel_order = channel_order
        self.model_type = model_type
        self.row_indices = row_indices  # positions into the packed arrays
        self.labels = labels  # aligned with row_indices
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.row_indices)

    def _build_image(self, i):
        if self.model_type == 'single':
            return np.array(self.arrays['single'][i])
        if is_multibranch(self.model_type):
            return np.stack([np.array(self.arrays[t][i]) for t in self.channel_order])
        return build_fusion_image(self.arrays['fusion'], i, self.channel_order)

    def __getitem__(self, idx):
        i = self.row_indices[idx]
        img = self._build_image(i).astype(np.float32)
        if is_multibranch(self.model_type):
            # (3,H,W,3) -> per-branch z-score -> (3,3,H,W); augmentation drawn per branch.
            img = (img - self.mean[:, None, None, :]) / self.std[:, None, None, :]
            image = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
            if self.transform:
                image = torch.stack([self.transform(b) for b in image])
            return image, int(self.labels[idx])
        img = (img - self.mean) / self.std
        image = torch.from_numpy(img.transpose(2, 0, 1)).float()
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])


def compute_stats(arrays, channel_order, model_type, row_indices, stats_path):
    """Per-channel mean/std over the given rows, computed the same way as the existing
    segment-wise dataset (full pass, biased-variance formula), then cached to disk."""
    def stats_over(images):
        channel_sum = np.zeros(3)
        channel_sum_sq = np.zeros(3)
        pixel_count = 0
        for img in images:
            channel_sum += img.sum(axis=(0, 1))
            channel_sum_sq += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]
        m = channel_sum / pixel_count
        return m, np.sqrt(channel_sum_sq / pixel_count - m ** 2)

    if model_type == 'single':
        mean, std = stats_over(np.array(arrays['single'][i]).astype(np.float32)
                               for i in row_indices)
    elif is_multibranch(model_type):
        per_branch = [stats_over(np.array(arrays[t][i]).astype(np.float32)
                                 for i in row_indices) for t in channel_order]
        mean = np.stack([m for m, _ in per_branch])   # (3 branches, 3)
        std = np.stack([s for _, s in per_branch])
    else:
        mean, std = stats_over(build_fusion_image(arrays['fusion'], i, channel_order)
                               .astype(np.float32) for i in row_indices)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mean=mean, std=std)
    return mean, std


def load_stats(stats_path):
    stats = np.load(stats_path)
    return stats['mean'], stats['std']


class ImageDirArray:
    """Array-like (len / integer __getitem__ / .shape) view over a directory of .npy files.

    Reads the images in place -- no regeneration, no copy. Row order is the filename sort
    order, which is what makes a row index mean the same segment in every transform dir.
    """

    def __init__(self, directory):
        self.dir = Path(directory)
        self.names = sorted(f.name for f in self.dir.iterdir() if f.suffix == '.npy')
        if not self.names:
            raise RuntimeError(f'no .npy under {self.dir}')
        probe = np.load(self.dir / self.names[0])
        self.shape = (len(self.names), *probe.shape)
        self.dtype = probe.dtype

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return np.load(self.dir / self.names[i])


def _rows_from_names(names):
    """Filename list -> row records (subject / segment / label) used by the CV split."""
    rows = []
    for i, n in enumerate(names):
        subject, segment, label = parse_name(n)
        rows.append({'idx': str(i), 'filename': n, 'subject': str(subject),
                     'segment': str(segment), 'label': str(label)})
    return rows


def load_arrays(data_dir, model_type, image_trans, channel_order, seg_drop=True):
    """Returns (arrays dict, reference rows list) for the requested model type."""
    # early/late/hybrid select channels of the grayscale fusion image; channel 0 of the viridis
    # RGB .npy is a different image. *_drop dirs hold the segments kept by quality control.
    suffix = '_drop' if seg_drop else ''

    def read(name):
        arr = ImageDirArray(Path(data_dir) / name)
        return arr, _rows_from_names(arr.names)

    # mb_*: one RGB array per transform; the row order must match across them (asserted).
    if is_multibranch(model_type):
        arrays, rows = {}, None
        for t in channel_order:
            arr, r = read(f'{t}{suffix}')
            assert arr.shape[-1] == 3, f'{t}{suffix} has {arr.shape[-1]} channels, expected 3'
            if rows is None:
                rows = r
            else:
                assert [x['filename'] for x in r] == [x['filename'] for x in rows], \
                    f'{t}{suffix} row order differs from {channel_order[0]}{suffix}'
            arrays[t] = arr
        return arrays, rows

    name = f'{image_trans}{suffix}' if model_type == 'single' else f'fusion{suffix}'
    key = 'single' if model_type == 'single' else 'fusion'
    arr, rows = read(name)

    if key == 'fusion':
        assert arr.shape[-1] == len(FUSION_CHANNEL_MAPPING), (
            f'{name} has {arr.shape[-1]} channels, expected {len(FUSION_CHANNEL_MAPPING)}')
    return {key: arr}, rows


# Subject-wise split

def subject_valid_holdout(trainval_idx, subjects, labels, random_state, fold, n_trials=500):
    """Hold out ~1/9 of trainval subjects (>=2) as validation, subject-disjoint from train.
    Picks the seeded candidate whose resulting valid-set segment class ratio is closest to
    the trainval segment class ratio."""
    trainval_subjects = sorted(set(subjects[i] for i in trainval_idx))
    subj_segments = defaultdict(list)
    for i in trainval_idx:
        subj_segments[subjects[i]].append(i)

    n_valid_subj = max(2, round(len(trainval_subjects) / 9))
    n_valid_subj = min(n_valid_subj, len(trainval_subjects) - 1)  # leave >=1 for train
    trainval_ratio = float(np.mean([labels[i] for i in trainval_idx]))

    rng = random.Random(random_state + fold)
    best_subjs, best_dev = None, None
    for _ in range(n_trials):
        cand = rng.sample(trainval_subjects, n_valid_subj)
        valid_idx = [i for s in cand for i in subj_segments[s]]
        if not valid_idx:
            continue
        ratio = float(np.mean([labels[i] for i in valid_idx]))
        dev = abs(ratio - trainval_ratio)
        if best_dev is None or dev < best_dev:
            best_dev, best_subjs = dev, cand

    valid_subjects = sorted(best_subjs)
    train_subjects = sorted(set(trainval_subjects) - set(valid_subjects))
    train_idx = [i for s in train_subjects for i in subj_segments[s]]
    valid_idx = [i for s in valid_subjects for i in subj_segments[s]]
    return train_idx, valid_idx, train_subjects, valid_subjects


def build_folds(ref_rows, n_splits, random_state):
    X = np.arange(len(ref_rows))
    y = np.array([int(r['label']) for r in ref_rows])
    subjects = np.array([r['subject'] for r in ref_rows])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(sgkf.split(X, y, groups=subjects)), y, subjects


def class_dist(indices, labels):
    lbl = [labels[i] for i in indices]
    return {'n': len(lbl), 'n_normal': lbl.count(0), 'n_apnea': lbl.count(1)}


# Checkpointing / resume / run state

def load_json(path, default):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return default


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def get_fold_status(state, job_id, fold):
    return state.get(job_id, {}).get(str(fold), {}).get('status', 'pending')


def set_fold_status(state, job_id, fold, status):
    state.setdefault(job_id, {}).setdefault(str(fold), {})['status'] = status


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric,
                     patience_counter, history):
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'patience_counter': patience_counter,
        'history': history,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    # weights_only=False: the payload (written by save_checkpoint) includes numpy RNG state.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    # Best effort: a failed RNG restore must not abort a resumed fold. set_rng_state needs CPU tensors.
    try:
        random.setstate(ckpt['rng_python'])
        np.random.set_state(ckpt['rng_numpy'])
        torch.set_rng_state(ckpt['rng_torch'].cpu())
        cuda_state = ckpt.get('rng_cuda')
        if cuda_state is not None and torch.cuda.is_available() \
                and len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
    except Exception as e:  # noqa: BLE001
        print(f'[warn] RNG state not restored ({type(e).__name__}: {e}); '
              f'continuing with fresh RNG.', flush=True)
    return ckpt['epoch'], ckpt['best_metric'], ckpt['patience_counter'], ckpt['history']


_STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# Train / test loops

def check_nan_inf(tensor, name, log):
    if torch.isnan(tensor).any():
        log(f'{name} contains NaN values.')
    if torch.isinf(tensor).any():
        log(f'{name} contains Inf values.')


def train_one_fold(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                    device, epochs, max_norm, early_stopping_epochs, best_path, last_path,
                    resume, log):
    amp_on = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)

    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'train_auc': [],
               'val_loss': [], 'val_acc': [], 'val_auc': [], 'best_epoch': -1}

    if resume and Path(last_path).exists():
        start_epoch, best_loss, early_stop_counter, history = load_checkpoint(
            last_path, model, optimizer, scheduler, device)
        start_epoch += 1
        log(f'Resumed from {last_path} at epoch {start_epoch}')

    for e in range(start_epoch, epochs):
        running_loss = running_accuracy = 0.0
        train_true, train_pred_prob = [], []
        model.train()

        for X, Y in tqdm(train_loader, desc=f'Epoch {e + 1:3d}/{epochs}'):
            try:
                X = X.to(device, dtype=torch.float32)
                Y = Y.to(device, dtype=torch.long)
                if X.max() > 1.0:
                    X = X / 255.0

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=amp_on):
                    outputs = model(X)
                    outputs = torch.clamp(outputs, min=-100, max=100)
                    loss = criterion(outputs, Y)

                if torch.isnan(loss) or torch.isinf(loss):
                    log(f'Warning: NaN/Inf loss at epoch {e + 1}')
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()

                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    try:
                        scheduler.step()
                    except ValueError as err:
                        log(f'Scheduler step error ignored: {err}')

                probabilities = torch.softmax(outputs, dim=1)
                eps = 1e-7
                probabilities = torch.clamp(probabilities, eps, 1 - eps)
                train_true.extend(Y.cpu().numpy())
                train_pred_prob.extend(probabilities.detach().cpu().numpy())
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_accuracy += torch.sum(preds == Y).float().item() / X.size(0)
            except RuntimeError as error:
                log(f'Error during training: {error}')
                continue

            if _STOP_REQUESTED:
                break

        train_true = np.array(train_true)
        train_pred_prob = np.array(train_pred_prob)
        epoch_train_auc = (roc_auc_score(train_true, train_pred_prob[:, 1])
                            if len(np.unique(train_true)) == 2 else float('nan'))

        model.eval()
        val_loss = val_accuracy = 0.0
        n_skipped_val = 0
        val_true, val_pred_prob = [], []
        with torch.no_grad():
            for X, Y in tqdm(valid_loader, desc='Validation   '):
                X = X.to(device, dtype=torch.float32)
                Y = Y.to(device, dtype=torch.long)
                if X.max() > 1.0:
                    X = X / 255.0
                with torch.amp.autocast(device.type, enabled=amp_on):
                    outputs = model(X)
                    loss = criterion(outputs, Y)
                if torch.isnan(loss) or torch.isinf(loss):
                    n_skipped_val += 1
                    continue
                probabilities = torch.softmax(outputs, dim=1)
                val_true.extend(Y.cpu().numpy())
                val_pred_prob.extend(probabilities.cpu().numpy())
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_accuracy += torch.sum(preds == Y).float().item() / X.size(0)

        if n_skipped_val:
            log(f'Warning: {n_skipped_val}/{len(valid_loader)} validation batches had '
                f'NaN/Inf loss and were dropped from val metrics')

        val_true = np.array(val_true)
        val_pred_prob = np.array(val_pred_prob)
        epoch_val_auc = (roc_auc_score(val_true, val_pred_prob[:, 1])
                          if len(np.unique(val_true)) == 2 else float('nan'))

        current_val_loss = val_loss / max(len(valid_loader), 1)
        current_lr = optimizer.param_groups[0]['lr']
        log(f'Epoch {e + 1}/{epochs} train_loss={running_loss / max(len(train_loader), 1):.3f} '
            f'val_loss={current_val_loss:.3f} train_acc={running_accuracy / max(len(train_loader), 1):.3f} '
            f'val_acc={val_accuracy / max(len(valid_loader), 1):.3f} train_auc={epoch_train_auc:.3f} '
            f'val_auc={epoch_val_auc:.3f} lr={current_lr}')

        history['train_loss'].append(running_loss / max(len(train_loader), 1))
        history['train_acc'].append(running_accuracy / max(len(train_loader), 1))
        history['train_auc'].append(float(epoch_train_auc))
        history['val_loss'].append(current_val_loss)
        history['val_acc'].append(val_accuracy / max(len(valid_loader), 1))
        history['val_auc'].append(float(epoch_val_auc))

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            history['best_epoch'] = e
            torch.save(model.state_dict(), best_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        save_checkpoint(last_path, model, optimizer, scheduler, e, best_loss,
                         early_stop_counter, history)

        if _STOP_REQUESTED:
            log('Stop requested (signal). Checkpoint saved at last.pt; exiting.')
            return history, 'interrupted'

        if early_stopping_epochs > 0 and early_stop_counter >= early_stopping_epochs:
            log(f'Early stopping at epoch {e + 1}')
            return history, 'early_stop'

    return history, 'max_epochs'


def test_fold(model, test_loader, device, best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    true_labels, pred_labels, pred_probs = [], [], []
    # fp32: some segments overflow fp16 inside EfficientNet and give NaN logits.
    with torch.no_grad():
        for X, Y in tqdm(test_loader, desc='Testing'):
            X = X.to(device, dtype=torch.float32)
            Y = Y.to(device, dtype=torch.long)
            if X.max() > 1.0:
                X = X / 255.0
            with torch.amp.autocast(device.type, enabled=False):
                outputs = model(X)
                outputs = torch.clamp(outputs, min=-100, max=100)
                probabilities = torch.softmax(outputs, dim=1)
            eps = 1e-7
            probabilities = torch.clamp(probabilities, eps, 1 - eps)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(Y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            pred_probs.extend(probabilities.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    assert np.isfinite(pred_probs).all(), (
        f'{int((~np.isfinite(pred_probs)).any(axis=1).sum())} of {len(pred_probs)} test '
        f'segments produced non-finite logits; metrics would be meaningless')

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-7
    metrics = {
        'test_accuracy': float(accuracy_score(true_labels, pred_labels)),
        'test_sensitivity': float(tp / (tp + fn + eps)),
        'test_specificity': float(tn / (tn + fp + eps)),
        'test_f1': float(f1_score(true_labels, pred_labels, zero_division=0)),
        'test_auc': float(roc_auc_score(true_labels, pred_probs[:, 1])),
    }
    return metrics, true_labels, pred_labels, pred_probs


# CLI / orchestration

def sanitize(s):
    return re.sub(r'[ :,()]+', '', str(s))


# model_type -> (lr, max_lr, max_norm). The grouped runs used final_div_factor 1e4 and batch 32
# for every model type, unlike src/engine.py (segment-wise runs).
HYPERPARAMS = {
    'single': (5e-4, 1e-3, 1.0),
    'early': (5e-4, 1e-3, 1.0),
    'late': (5e-6, 1e-4, 0.5),
    'hybrid': (5e-6, 1e-4, 0.5),
    'mb_late': (5e-6, 1e-4, 0.5),
    'mb_hybrid': (5e-6, 1e-4, 0.5),
}


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True,
                   help='directory holding the per-transform image dirs '
                        '(scalogram_drop/, fusion_drop/, ...)')
    p.add_argument('--image_trans', choices=['scalogram', 'rp', 'gasf', 'gadf', 'mtf', 'spectrogram'],
                   default=None, help='single-transform only')
    p.add_argument('--model_type',
                   choices=['single', 'early', 'late', 'hybrid', 'mb_late', 'mb_hybrid'],
                   required=True)
    p.add_argument('--channel_order', default='scalogram,gadf,spectrogram',
                   help='comma-separated, early/late/hybrid/mb_* only')
    p.add_argument('--random_state', type=int, default=128)
    p.add_argument('--n_splits', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                   help='ImageNet pretrained weights ("+Pre")')
    p.add_argument('--seg_drop', action=argparse.BooleanOptionalAction, default=True,
                   help='statistical quality control: use the *_drop segment set ("+SQC")')
    p.add_argument('--output_dir', default='results/ucddb_grouped')
    p.add_argument('--ckpt_dir', default='results/ucddb_grouped/checkpoints')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                   help='skip folds already done and restore last.pt (default: on; '
                        'use --no-resume to force a clean rerun)')
    p.add_argument('--max_folds', type=int, default=None, help='limit folds run (smoke tests)')
    p.add_argument('--dry_run', action='store_true',
                   help='print fold/subject split + class distribution, then exit without training')
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.model_type == 'single' and args.image_trans is None:
        raise SystemExit('--image_trans is required for --model_type single')
    channel_order = tuple(c.strip() for c in args.channel_order.split(','))
    if args.model_type != 'single' and len(channel_order) != 3:
        raise SystemExit('--channel_order must have exactly 3 entries for early/late/hybrid')

    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.ckpt_dir)

    tag = sanitize(args.image_trans) if args.model_type == 'single' else sanitize('-'.join(channel_order))
    # Ablation settings go into the id so their stats/checkpoints/predictions do not collide.
    abl = '' if (args.pretrained and args.seg_drop) else \
        f"_pre{int(args.pretrained)}sqc{int(args.seg_drop)}"
    job_id = f'{tag}_{args.model_type}_rs{args.random_state}{abl}'

    for sub in ('results/segment_predictions', 'history', 'logs', 'state', 'env', 'stats'):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / 'logs' / f'{job_id}.log', 'a')

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    log(f'job_id={job_id} model_type={args.model_type} random_state={args.random_state} '
        f'n_splits={args.n_splits} channel_order={channel_order}')

    pretrained, seg_drop = args.pretrained, args.seg_drop
    lr, max_lr, max_norm = HYPERPARAMS[args.model_type]

    arrays, ref_rows = load_arrays(args.data_dir, args.model_type, args.image_trans,
                                   channel_order, seg_drop=seg_drop)
    folds, labels, subjects = build_folds(ref_rows, args.n_splits, args.random_state)

    fold_mapping_path = output_dir / 'results' / 'fold_subject_mapping.csv'

    def append_fold_mapping(rows):
        write_header = not fold_mapping_path.exists()
        with open(fold_mapping_path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(['fold', 'split', 'subject', 'n_segments', 'n_apnea'])
            w.writerows(rows)

    if args.dry_run:
        for fold, (trainval_idx, test_idx) in enumerate(folds):
            train_idx, valid_idx, train_subj, valid_subj = subject_valid_holdout(
                list(trainval_idx), subjects, labels, args.random_state, fold)
            test_subj = sorted(set(subjects[i] for i in test_idx))
            assert not (set(train_subj) & set(valid_subj))
            assert not (set(train_subj) & set(test_subj))
            assert not (set(valid_subj) & set(test_subj))
            print(f'--- Fold {fold + 1}/{args.n_splits} ---')
            print(f'  train subjects ({len(train_subj)}): {train_subj}  {class_dist(train_idx, labels)}')
            print(f'  valid subjects ({len(valid_subj)}): {valid_subj}  {class_dist(valid_idx, labels)}')
            print(f'  test  subjects ({len(test_subj)}): {test_subj}  {class_dist(list(test_idx), labels)}')
        print('[dry_run] disjointness assertions passed for all folds. Exiting without training.')
        return

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    log(f'device={device}')

    env_lines = [
        f'torch: {torch.__version__}',
        f'cuda: {torch.version.cuda}',
        f'gpu: {torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu"}',
        f'platform: {platform.platform()}',
        f'numpy: {np.__version__}',
    ]
    (output_dir / 'env' / 'environment.txt').write_text('\n'.join(env_lines) + '\n')
    save_json(output_dir / 'env' / f'config_{job_id}.json', {
        **vars(args), 'channel_order': channel_order, 'job_id': job_id,
        'pretrained': pretrained, 'seg_drop': seg_drop, 'lr': lr, 'max_lr': max_lr,
        'max_norm': max_norm,
    })

    results_csv = output_dir / 'results' / 'ucddb_grouped_10fold.csv'
    results_header = ['fold', 'random_state', 'image_trans', 'channel_order', 'pretrained',
                       'seg_drop', 'model_type', 'test_accuracy', 'test_sensitivity',
                       'test_specificity', 'test_f1', 'test_auc', 'Runtime', 'gpu_model',
                       'concurrent_jobs', 'n_train_subj', 'n_valid_subj', 'n_test_subj',
                       'n_test_seg', 'seed_note']

    seg_pred_path = output_dir / 'results' / 'segment_predictions' / f'{job_id}.csv'

    # One state file per job: concurrent jobs would overwrite a shared one.
    state_path = output_dir / 'state' / f'run_state_{job_id}.json'
    state = load_json(state_path, {})

    gpu_model = torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'cpu'
    concurrent_jobs = int(os.environ.get('CONCURRENT_JOBS', 1))

    n_folds_to_run = args.n_splits if args.max_folds is None else min(args.n_splits, args.max_folds)

    for fold in range(n_folds_to_run):
        trainval_idx, test_idx = folds[fold]
        if args.resume and get_fold_status(state, job_id, fold) == 'done':
            log(f'Fold {fold}: already done, skipping.')
            continue

        set_fold_status(state, job_id, fold, 'running')
        save_json(state_path, state)

        train_idx, valid_idx, train_subj, valid_subj = subject_valid_holdout(
            list(trainval_idx), subjects, labels, args.random_state, fold)
        test_subj = sorted(set(subjects[i] for i in test_idx))
        assert not (set(train_subj) & set(valid_subj)), 'train/valid subject overlap'
        assert not (set(train_subj) & set(test_subj)), 'train/test subject overlap'
        assert not (set(valid_subj) & set(test_subj)), 'valid/test subject overlap'

        append_fold_mapping(
            [[fold, 'train', s, len(subj_seg(train_idx, subjects, s)),
              sum(labels[i] for i in subj_seg(train_idx, subjects, s))] for s in train_subj] +
            [[fold, 'valid', s, len(subj_seg(valid_idx, subjects, s)),
              sum(labels[i] for i in subj_seg(valid_idx, subjects, s))] for s in valid_subj] +
            [[fold, 'test', s, len(subj_seg(list(test_idx), subjects, s)),
              sum(labels[i] for i in subj_seg(list(test_idx), subjects, s))] for s in test_subj]
        )

        stats_path = output_dir / 'stats' / f'{job_id}_fold{fold}.npz'
        if not stats_path.exists():
            mean, std = compute_stats(arrays, channel_order, args.model_type, train_idx, stats_path)
        else:
            mean, std = load_stats(stats_path)
        log(f'Fold {fold}: stats mean={mean} std={std}')

        train_labels = [labels[i] for i in train_idx]
        weight_0 = len(train_labels) / (2 * max(train_labels.count(0), 1))
        weight_1 = len(train_labels) / (2 * max(train_labels.count(1), 1))
        loss_weights = torch.tensor([weight_0, weight_1])

        train_ds = GroupedDataset(arrays, channel_order, args.model_type, train_idx,
                                  [labels[i] for i in train_idx], mean, std, train_transform)
        valid_ds = GroupedDataset(arrays, channel_order, args.model_type, valid_idx,
                                  [labels[i] for i in valid_idx], mean, std, valid_transform)
        test_ds = GroupedDataset(arrays, channel_order, args.model_type, list(test_idx),
                                 [labels[i] for i in test_idx], mean, std, valid_transform)

        persistent = args.num_workers > 0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, prefetch_factor=4 if persistent else None,
                                   pin_memory=True, persistent_workers=persistent)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, prefetch_factor=4 if persistent else None,
                                   pin_memory=True, persistent_workers=persistent)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, prefetch_factor=4 if persistent else None,
                                  pin_memory=True, persistent_workers=persistent)

        model = build_model(args.model_type, pretrained).to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights, reduction='mean',
                                         label_smoothing=0.1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5,
                                 betas=(0.9, 0.999), eps=1e-6)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=max(len(train_loader), 1),
            epochs=args.epochs, pct_start=0.05, div_factor=25.0, final_div_factor=10000.0,
            anneal_strategy='cos')

        fold_ckpt_dir = ckpt_dir / job_id / f'fold{fold}'
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = fold_ckpt_dir / 'best.pt'
        last_path = fold_ckpt_dir / 'last.pt'

        t0 = time.time()
        history, stop_reason = train_one_fold(
            model, train_loader, valid_loader, criterion, optimizer, scheduler, device,
            args.epochs, max_norm, args.patience, best_path, last_path, args.resume, log)
        runtime = time.time() - t0

        save_json(output_dir / 'history' / f'{job_id}_fold{fold}.json', history)

        if stop_reason == 'interrupted':
            set_fold_status(state, job_id, fold, 'pending')
            save_json(state_path, state)
            log(f'Fold {fold}: interrupted, marked pending. Exiting.')
            log_file.close()
            sys.exit(0)

        metrics, true_labels, pred_labels, pred_probs = test_fold(model, test_loader, device, best_path)

        write_header = not seg_pred_path.exists()
        with open(seg_pred_path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(['subject', 'segment', 'y_true', 'y_prob', 'y_pred', 'fold'])
            for k, i in enumerate(test_idx):
                w.writerow([ref_rows[i]['subject'], ref_rows[i]['segment'],
                            int(true_labels[k]), float(pred_probs[k][1]), int(pred_labels[k]), fold])

        write_header = not results_csv.exists()
        with open(results_csv, 'a', newline='') as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(results_header)
            w.writerow([
                fold, args.random_state, args.image_trans or '', '-'.join(channel_order),
                pretrained, seg_drop, args.model_type,
                metrics['test_accuracy'], metrics['test_sensitivity'], metrics['test_specificity'],
                metrics['test_f1'], metrics['test_auc'], runtime, gpu_model, concurrent_jobs,
                len(train_subj), len(valid_subj), len(test_subj), len(test_idx), stop_reason,
            ])

        log(f'Fold {fold}: done. acc={metrics["test_accuracy"]:.4f} auc={metrics["test_auc"]:.4f} '
            f'runtime={runtime:.1f}s ({stop_reason})')

        set_fold_status(state, job_id, fold, 'done')
        save_json(state_path, state)
        if last_path.exists():
            last_path.unlink()

    log_file.close()


def subj_seg(indices, subjects, s):
    return [i for i in indices if subjects[i] == s]


if __name__ == '__main__':
    main()
