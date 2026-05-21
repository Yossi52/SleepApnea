"""
CKA (Centered Kernel Alignment) Analysis for Multi-Channel Image Fusion

Computes layer-wise CKA similarity between representations learned from
different image transformation methods (GADF, GASF, MTF, RP, Scalogram, Spectrogram).

Supports:
  - Linear CKA and RBF Kernel CKA
  - Layer-wise analysis (Early, Mid, Late layers of EfficientNet-B0)
  - Robustness analysis (Linear vs RBF comparison)
  - Multiple visualization modes (single, comparison, correlation, difference)

Usage:
  python analysis/cka_analysis.py \
      --data_dir /path/to/images \
      --model_dir /path/to/models \
      --stats_file /path/to/normalization_stats.npz \
      --output_dir results/analysis
"""

import os
import sys
import argparse
import random
import gc
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# 1. CKA Implementation
# ============================================================

def gram_linear(x):
    """Compute Gram matrix for linear kernel."""
    return x @ x.T


def gram_rbf(x, threshold=1.0):
    """Compute Gram matrix for RBF kernel with fixed bandwidth."""
    dot_products = x @ x.T
    sq_norms = (x ** 2).sum(axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_products
    return torch.exp(-sq_dists / (2 * threshold ** 2))


def gram_rbf_adaptive(x):
    """Compute Gram matrix for RBF kernel with adaptive (median) bandwidth."""
    sq_dists = torch.cdist(x, x, p=2) ** 2
    sigma = torch.median(sq_dists[sq_dists > 0]).sqrt()
    return torch.exp(-sq_dists / (2 * sigma ** 2))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix."""
    if not unbiased:
        means = gram.mean(dim=0)
        means -= means.mean() / 2
        gram -= means[:, None]
        gram -= means[None, :]
    else:
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = gram.sum(dim=0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA given two centered Gram matrices."""
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = gram_x.ravel().dot(gram_x.ravel())
    normalization_y = gram_y.ravel().dot(gram_y.ravel())
    return scaled_hsic / torch.sqrt(normalization_x * normalization_y)


def linear_CKA(X, Y):
    """Compute Linear CKA between two feature matrices.

    Args:
        X: Feature matrix of shape (N, D1).
        Y: Feature matrix of shape (N, D2).

    Returns:
        CKA similarity value (float).
    """
    return cka(gram_linear(X), gram_linear(Y)).item()


def kernel_CKA(X, Y, threshold=None):
    """Compute RBF Kernel CKA between two feature matrices.

    Args:
        X: Feature matrix of shape (N, D1).
        Y: Feature matrix of shape (N, D2).
        threshold: RBF bandwidth. If None, uses adaptive median bandwidth.

    Returns:
        CKA similarity value (float).
    """
    if threshold is None:
        gx = gram_rbf_adaptive(X)
        gy = gram_rbf_adaptive(Y)
    else:
        gx = gram_rbf(X, threshold)
        gy = gram_rbf(Y, threshold)
    return cka(gx, gy).item()


# ============================================================
# 2. Model & Feature Extraction
# ============================================================

LAYER_INDICES = {
    'early': (0, 3),    # EfficientNet-B0 features[0:3]
    'mid':   (0, 5),    # EfficientNet-B0 features[0:5]
    'late':  (0, None),  # EfficientNet-B0 features (all)
}


def build_model(num_classes=2, pretrained=True):
    """Build EfficientNet-B0 model."""
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


def extract_layer_activations(model, x, layer_indices):
    """Extract layer-wise activations from EfficientNet-B0.

    Args:
        model: EfficientNet-B0 model (torchvision).
        x: Input tensor of shape (N, 3, 224, 224).
        layer_indices: Dict mapping layer name to (start, end) index range.

    Returns:
        Dict mapping layer name to flattened activation tensor (N, D).
    """
    activations = {}
    for name, (_, end_idx) in layer_indices.items():
        feat = x
        if end_idx is None:
            for layer in model.features:
                feat = layer(feat)
        else:
            for layer in model.features[0:end_idx]:
                feat = layer(feat)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        activations[name] = torch.flatten(pooled, 1)
    return activations


def find_checkpoint(method, model_dir, random_state, fold_num):
    """Find model checkpoint file matching the given configuration.

    Args:
        method: Transformation method name.
        model_dir: Directory containing model checkpoints.
        random_state: Random seed used in training.
        fold_num: Fold number.

    Returns:
        Path to the checkpoint file, or None if not found.
    """
    patterns = [
        f"{method}, efficientnet_b0, rs:{random_state}, fold:{fold_num}, seg_drop: True ds: False, optim: AdamW, channel_weights: None, channel_order: None_train.pth",
        f"{method}, efficientnet_b0, rs:{random_state}, fold:{fold_num}, seg_drop: True ds: False, *_train.pth",
    ]
    for pattern in patterns:
        matches = glob(os.path.join(model_dir, pattern))
        # Filter out models with "pretrained: False"
        matches = [f for f in matches if 'pretrained: False' not in os.path.basename(f)]
        if matches:
            return matches[0]
    return None


# ============================================================
# 3. Segment Matching
# ============================================================

def get_matched_segments(base_dir, methods, random_state=128, fold_num=1,
                         patient_ids=None, segment_range=None, max_samples=None):
    """Find matched segments across all transformation methods.

    Each segment represents the same ECG epoch transformed by different methods.
    Uses the validation set from stratified 10-fold CV to ensure fair evaluation.

    Args:
        base_dir: Root directory containing per-method image subdirectories.
        methods: List of transformation method names.
        random_state: Random seed for reproducible splits.
        fold_num: Fold number to select validation set from.
        patient_ids: Optional list of patient IDs to include (e.g., ['a01']).
        segment_range: Optional (start, end) tuple for segment number filtering.
        max_samples: Maximum number of segments to use (random sampling).

    Returns:
        Dict mapping segment_id to {method: filepath, ..., 'label': int}.
    """
    first_method = methods[0]
    target_dir = os.path.join(base_dir, first_method)
    all_files = sorted(glob(os.path.join(target_dir, '*.npy')))
    all_files = [f for f in all_files if 'c05' not in f]

    y = [int(f[-5]) for f in all_files]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    valid_files = None
    for fold, (train_val_idx, _) in enumerate(skf.split(all_files, y)):
        if fold + 1 == fold_num:
            train_val_files = [all_files[i] for i in train_val_idx]
            train_val_y = [y[i] for i in train_val_idx]
            _, valid_files, _, _ = train_test_split(
                train_val_files, train_val_y,
                test_size=1 / 9, stratify=train_val_y,
                shuffle=True, random_state=random_state + fold,
            )
            break

    matched = {}
    for filepath in valid_files:
        filename = os.path.basename(filepath)
        parts = filename.replace('.npy', '').split('_')
        patient_id = parts[0]
        segment_id = f"{parts[0]}_{parts[1]}"
        label = int(parts[-1].replace('label', ''))

        if patient_ids and patient_id not in patient_ids:
            continue
        if segment_range is not None:
            seg_num = int(parts[1].replace('segment', ''))
            if not (segment_range[0] <= seg_num < segment_range[1]):
                continue

        seg_dict = {'label': label}
        all_exist = True
        for method in methods:
            method_path = os.path.join(base_dir, method, filename)
            if not os.path.exists(method_path):
                all_exist = False
                break
            seg_dict[method] = method_path

        if all_exist:
            matched[segment_id] = seg_dict

    if max_samples and len(matched) > max_samples:
        random.seed(random_state)
        keys = random.sample(list(matched.keys()), max_samples)
        matched = {k: matched[k] for k in keys}

    print(f"Matched segments: {len(matched)}")
    return matched


# ============================================================
# 4. Feature Normalization
# ============================================================

def normalize_features(f, norm_type='both'):
    """Normalize feature tensors before CKA computation.

    Args:
        f: Feature tensor of shape (N, D).
        norm_type: One of 'zscore', 'l2', or 'both'.

    Returns:
        Normalized feature tensor.
    """
    if norm_type in ('zscore', 'both'):
        f = (f - f.mean(dim=0, keepdim=True)) / (f.std(dim=0, keepdim=True) + 1e-8)
    if norm_type in ('l2', 'both'):
        f = f / (torch.norm(f, p=2, dim=1, keepdim=True) + 1e-8)
    return f


# ============================================================
# 5. CKA Matrix Computation
# ============================================================

def compute_cka_matrices(all_features, methods, cka_types, norm_type, device):
    """Compute CKA similarity matrices for all layer-method pairs.

    Args:
        all_features: Nested dict {method: {layer: tensor(N, D)}}.
        methods: List of method names.
        cka_types: List of CKA types ('linear' and/or 'rbf').
        norm_type: Feature normalization type.
        device: Torch device.

    Returns:
        Nested dict {cka_type: {layer_name: np.ndarray(M, M)}}.
    """
    n = len(methods)
    results = {ct: {} for ct in cka_types}

    for ct in cka_types:
        print(f"\nComputing {ct.upper()} CKA...")
        for layer_name in LAYER_INDICES:
            matrix = np.zeros((n, n))
            for i, m1 in enumerate(methods):
                for j, m2 in enumerate(methods):
                    if i > j:
                        matrix[i, j] = matrix[j, i]
                        continue

                    f1 = normalize_features(all_features[m1][layer_name].to(device), norm_type)
                    f2 = normalize_features(all_features[m2][layer_name].to(device), norm_type)

                    if ct == 'linear':
                        val = linear_CKA(f1, f2)
                    else:
                        val = kernel_CKA(f1, f2, threshold=None)

                    matrix[i, j] = val
                    del f1, f2
                    torch.cuda.empty_cache()

            results[ct][layer_name] = matrix
            print(f"  {layer_name} layer done")

    return results


# ============================================================
# 6. Visualization
# ============================================================

METHOD_LABELS = ['GADF', 'GASF', 'MTF', 'RP', 'Scalo', 'Spec']


def plot_single(cka_results, cka_type, output_dir=None):
    """Plot CKA heatmaps for a single CKA type across all layers."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, layer_name in enumerate(['early', 'mid', 'late']):
        ax = axes[idx]
        sns.heatmap(
            cka_results[cka_type][layer_name],
            annot=True, fmt='.3f',
            xticklabels=METHOD_LABELS, yticklabels=METHOD_LABELS,
            cmap='Blues', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'CKA Similarity'}, square=True,
        )
        ax.set_title(f'{layer_name.capitalize()} Layer', fontsize=14, fontweight='bold')
        ax.set_xlabel('Transformation Method', fontsize=12)
        ax.set_ylabel('Transformation Method', fontsize=12)

    fig.suptitle(f'{cka_type.upper()} CKA Similarity Across Layers',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'cka_{cka_type}_heatmap.png'),
                    dpi=200, bbox_inches='tight')
        print(f"Saved: cka_{cka_type}_heatmap.png")
    plt.show()


def plot_comparison(cka_results, output_dir=None):
    """Plot Linear vs RBF CKA comparison (3x2 grid)."""
    cka_types = list(cka_results.keys())[:2]
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, layer_name in enumerate(['early', 'mid', 'late']):
        for col, ct in enumerate(cka_types):
            ax = axes[row, col]
            sns.heatmap(
                cka_results[ct][layer_name],
                annot=True, fmt='.3f',
                xticklabels=METHOD_LABELS, yticklabels=METHOD_LABELS,
                cmap='Blues', vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'CKA Similarity'}, square=True,
            )
            ax.set_title(f'{layer_name.capitalize()} Layer - {ct.upper()} CKA',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Transformation Method', fontsize=11)
            ax.set_ylabel('Transformation Method', fontsize=11)

    fig.suptitle('Linear CKA vs RBF Kernel CKA Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'cka_comparison.png'),
                    dpi=200, bbox_inches='tight')
        print("Saved: cka_comparison.png")
    plt.show()


def plot_correlation(cka_results, output_dir=None):
    """Plot scatter-based correlation between Linear and RBF CKA."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, layer_name in enumerate(['early', 'mid', 'late']):
        ax = axes[idx]
        linear_mat = cka_results['linear'][layer_name]
        rbf_mat = cka_results['rbf'][layer_name]

        mask = np.triu(np.ones_like(linear_mat, dtype=bool), k=1)
        lv = linear_mat[mask]
        rv = rbf_mat[mask]

        ax.scatter(lv, rv, alpha=0.6, s=100)
        pr, pp = pearsonr(lv, rv)
        sr, sp = spearmanr(lv, rv)

        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y = x')
        ax.set_xlabel('Linear CKA', fontsize=12)
        ax.set_ylabel('RBF Kernel CKA', fontsize=12)
        ax.set_title(f'{layer_name.capitalize()} Layer\n'
                     f'Pearson r={pr:.3f} (p={pp:.2e})\n'
                     f'Spearman rho={sr:.3f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle('Linear CKA vs RBF Kernel CKA Correlation',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'cka_correlation.png'),
                    dpi=200, bbox_inches='tight')
        print("Saved: cka_correlation.png")
    plt.show()


def plot_difference(cka_results, output_dir=None):
    """Plot difference heatmap (Linear - RBF)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, layer_name in enumerate(['early', 'mid', 'late']):
        ax = axes[idx]
        diff = cka_results['linear'][layer_name] - cka_results['rbf'][layer_name]
        vmax = np.abs(diff).max()

        sns.heatmap(
            diff, annot=True, fmt='.3f',
            xticklabels=METHOD_LABELS, yticklabels=METHOD_LABELS,
            cmap='RdBu_r', vmin=-vmax, vmax=vmax, center=0, ax=ax,
            cbar_kws={'label': 'Difference (Linear - RBF)'}, square=True,
        )
        ax.set_title(f'{layer_name.capitalize()} Layer\nLinear - RBF',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Transformation Method', fontsize=11)
        ax.set_ylabel('Transformation Method', fontsize=11)

    fig.suptitle('Difference Between Linear and RBF Kernel CKA',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'cka_difference.png'),
                    dpi=200, bbox_inches='tight')
        print("Saved: cka_difference.png")
    plt.show()


def plot_layer_trend(cka_results, cka_type, methods, output_dir=None):
    """Plot bar chart of mean off-diagonal CKA per layer."""
    n = len(methods)
    mask = ~np.eye(n, dtype=bool)
    means = {}
    for layer_name in ['early', 'mid', 'late']:
        means[layer_name] = cka_results[cka_type][layer_name][mask].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#e67e22', '#e74c3c']
    bars = ax.bar(list(means.keys()), list(means.values()), color=colors,
                  alpha=0.7, edgecolor='black')
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean CKA Similarity (off-diagonal)', fontsize=14, fontweight='bold')
    ax.set_title('Layer-wise Representation Similarity\n(Higher = More Similar)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'cka_layer_trend_{cka_type}.png'),
                    dpi=200, bbox_inches='tight')
        print(f"Saved: cka_layer_trend_{cka_type}.png")
    plt.show()


# ============================================================
# 7. Robustness Analysis
# ============================================================

def robustness_analysis(cka_results, methods):
    """Compare Linear and RBF CKA to assess robustness of findings."""
    print("=" * 70)
    print("Robustness Analysis: Linear CKA vs RBF Kernel CKA")
    print("=" * 70)

    all_lv, all_rv = [], []

    for layer_name in ['early', 'mid', 'late']:
        linear_mat = cka_results['linear'][layer_name]
        rbf_mat = cka_results['rbf'][layer_name]

        mask = np.triu(np.ones_like(linear_mat, dtype=bool), k=1)
        lv = linear_mat[mask]
        rv = rbf_mat[mask]
        all_lv.extend(lv)
        all_rv.extend(rv)

        pr, pp = pearsonr(lv, rv)
        sr, sp = spearmanr(lv, rv)
        abs_diff = np.abs(lv - rv)

        print(f"\n--- {layer_name.upper()} Layer ---")
        print(f"  Pearson r  = {pr:.4f} (p = {pp:.2e})")
        print(f"  Spearman rho = {sr:.4f} (p = {sp:.2e})")
        print(f"  Mean |diff|  = {abs_diff.mean():.4f}")
        print(f"  Max  |diff|  = {abs_diff.max():.4f}")

    all_lv = np.array(all_lv)
    all_rv = np.array(all_rv)
    pr, _ = pearsonr(all_lv, all_rv)
    sr, _ = spearmanr(all_lv, all_rv)

    print(f"\n{'=' * 70}")
    print(f"Overall  Pearson r  = {pr:.4f}")
    print(f"Overall  Spearman rho = {sr:.4f}")
    print(f"Overall  Mean |diff|  = {np.abs(all_lv - all_rv).mean():.4f}")

    if pr > 0.9:
        print("Interpretation: EXCELLENT consistency between Linear and RBF CKA.")
    elif pr > 0.7:
        print("Interpretation: GOOD consistency with minor variations.")
    elif pr > 0.5:
        print("Interpretation: MODERATE consistency — some discrepancies.")
    else:
        print("Interpretation: LOW consistency — significant differences.")
    print("=" * 70)


# ============================================================
# 8. Summary Statistics
# ============================================================

def print_summary(cka_results, cka_type, methods, all_labels):
    """Print analysis summary including off-diagonal stats and cluster analysis."""
    n = len(methods)
    mask_offdiag = ~np.eye(n, dtype=bool)

    print(f"\n{'=' * 70}")
    print("CKA Analysis Summary")
    print(f"{'=' * 70}")
    print(f"  Samples : {len(all_labels)}")
    print(f"  Label 0 : {(all_labels == 0).sum()}, Label 1 : {(all_labels == 1).sum()}")
    print(f"  Methods : {', '.join(m.upper() for m in methods)}")
    print(f"  CKA Type: {cka_type.upper()}")

    off_means = {}
    for layer in ['early', 'mid', 'late']:
        off_means[layer] = cka_results[cka_type][layer][mask_offdiag].mean()
        print(f"  {layer.capitalize():5s} Layer mean off-diag CKA: {off_means[layer]:.4f}")

    if off_means['late'] > off_means['early'] + 0.05:
        print("\nKey Finding: Representations converge in late layers.")
        print("  Different transformations yield similar semantic features at depth.")
    else:
        print("\nKey Finding: Representations remain distinct across layers.")
        print("  Transformation-specific information is preserved through the network.")

    # Cluster analysis for late layer
    late = cka_results[cka_type]['late']
    idx_map = {m: i for i, m in enumerate(methods)}
    pairs = [
        ('scalogram', 'spectrogram', 'Scalogram-Spectrogram'),
        ('gadf', 'gasf', 'GADF-GASF'),
        ('rp', 'mtf', 'RP-MTF'),
    ]
    print(f"\nLate Layer Cluster Analysis:")
    for m1, m2, label in pairs:
        if m1 in idx_map and m2 in idx_map:
            val = late[idx_map[m1], idx_map[m2]]
            print(f"  {label:25s}: {val:.4f}")

    print(f"{'=' * 70}\n")

    return off_means


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='CKA Analysis for Image Fusion')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Root directory containing per-method image subdirectories')
    p.add_argument('--model_dir', type=str, required=True,
                   help='Directory containing trained model checkpoints')
    p.add_argument('--stats_file', type=str, default=None,
                   help='Path to normalization_stats.npz (optional)')
    p.add_argument('--output_dir', type=str, default='results/analysis',
                   help='Directory to save output figures')

    p.add_argument('--random_state', type=int, default=374)
    p.add_argument('--fold_num', type=int, default=1)
    p.add_argument('--max_samples', type=int, default=500,
                   help='Max number of segments to use (0 = all)')
    p.add_argument('--batch_size', type=int, default=32)

    p.add_argument('--cka_types', nargs='+', default=['linear', 'rbf'],
                   choices=['linear', 'rbf'],
                   help='CKA types to compute')
    p.add_argument('--norm_type', type=str, default='both',
                   choices=['zscore', 'l2', 'both'],
                   help='Feature normalization type')

    p.add_argument('--viz_mode', type=str, default='comparison',
                   choices=['single', 'comparison', 'correlation', 'difference', 'all'],
                   help='Visualization mode')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    methods = ['gadf', 'gasf', 'mtf', 'rp', 'scalogram', 'spectrogram']
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True)])
    max_samples = args.max_samples if args.max_samples > 0 else None

    # --- Load normalization stats ---
    norm_mean, norm_std = None, None
    if args.stats_file and os.path.exists(args.stats_file):
        stats = np.load(args.stats_file)
        norm_mean, norm_std = stats['mean'], stats['std']
        print(f"Loaded normalization stats from {args.stats_file}")

    # --- 1. Segment Matching ---
    print("\n[Step 1] Matching segments across methods...")
    matched = get_matched_segments(
        args.data_dir, methods,
        random_state=args.random_state,
        fold_num=args.fold_num,
        max_samples=max_samples,
    )
    segment_ids = sorted(matched.keys())

    # --- 2. Feature Extraction ---
    print("\n[Step 2] Extracting layer-wise features...")
    all_features = {m: {l: [] for l in LAYER_INDICES} for m in methods}
    all_labels = []

    for method in methods:
        print(f"\n  Processing: {method}")
        ckpt = find_checkpoint(method, args.model_dir, args.random_state, args.fold_num)
        if ckpt is None:
            print(f"  WARNING: Checkpoint not found for {method}, skipping.")
            continue
        print(f"  Checkpoint: {os.path.basename(ckpt)}")

        model = build_model(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval().to(device)

        for batch_start in tqdm(range(0, len(segment_ids), args.batch_size),
                                desc=f"  {method}"):
            batch_ids = segment_ids[batch_start:batch_start + args.batch_size]
            images, labels = [], []
            for sid in batch_ids:
                seg = matched[sid]
                img = np.load(seg[method])

                if norm_mean is not None:
                    img = (img.astype(np.float32) - norm_mean) / norm_std
                else:
                    img = img.astype(np.float32) / 255.0

                t = torch.from_numpy(img.transpose(2, 0, 1)).float()
                t = transform(t)
                images.append(t)
                labels.append(seg['label'])

            X = torch.stack(images).to(device)
            with torch.no_grad():
                acts = extract_layer_activations(model, X, LAYER_INDICES)
                for lname, act in acts.items():
                    all_features[method][lname].append(act.cpu())

            if method == methods[0]:
                all_labels.extend(labels)

        for lname in LAYER_INDICES:
            all_features[method][lname] = torch.cat(all_features[method][lname], dim=0)
            print(f"    {lname}: {all_features[method][lname].shape}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    all_labels = np.array(all_labels)
    print(f"\nTotal samples: {len(all_labels)} "
          f"(label 0: {(all_labels == 0).sum()}, label 1: {(all_labels == 1).sum()})")

    # --- 3. CKA Computation ---
    print("\n[Step 3] Computing CKA matrices...")
    cka_results = compute_cka_matrices(
        all_features, methods, args.cka_types, args.norm_type, device,
    )

    # --- 4. Visualization ---
    print("\n[Step 4] Generating visualizations...")
    out = args.output_dir

    if args.viz_mode == 'single' or args.viz_mode == 'all':
        for ct in args.cka_types:
            plot_single(cka_results, ct, output_dir=out)

    if args.viz_mode == 'comparison' or args.viz_mode == 'all':
        if len(args.cka_types) >= 2:
            plot_comparison(cka_results, output_dir=out)

    if args.viz_mode == 'correlation' or args.viz_mode == 'all':
        if len(args.cka_types) >= 2:
            plot_correlation(cka_results, output_dir=out)

    if args.viz_mode == 'difference' or args.viz_mode == 'all':
        if len(args.cka_types) >= 2:
            plot_difference(cka_results, output_dir=out)

    for ct in args.cka_types:
        plot_layer_trend(cka_results, ct, methods, output_dir=out)

    # --- 5. Robustness Analysis ---
    if len(args.cka_types) >= 2:
        print("\n[Step 5] Robustness analysis...")
        robustness_analysis(cka_results, methods)

    # --- 6. Summary ---
    print_summary(cka_results, args.cka_types[0], methods, all_labels)

    # --- Save raw results ---
    if out:
        os.makedirs(out, exist_ok=True)
        save_path = os.path.join(out, 'cka_results.npz')
        save_dict = {}
        for ct in args.cka_types:
            for lname in LAYER_INDICES:
                save_dict[f'{ct}_{lname}'] = cka_results[ct][lname]
        np.savez(save_path, **save_dict)
        print(f"Saved raw CKA matrices to {save_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
