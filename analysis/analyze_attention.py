"""Aggregate the .npz files written by extract_attention.py over folds and seeds.

Two questions:
  (a) branch importance -- how much attention each transformation branch *receives*,
      per layer and (as used in the paper) off-diagonal in the last layer;
  (b) how peaked the attention is -- per-layer entropy (normalised by log 3), L1 deviation
      from the uniform 1/3 distribution, and the self-attention (diagonal) share.

Writes `<attn_dir>/attention_summary.json` and prints the same numbers as a table.

Usage:
    python analysis/analyze_attention.py --attn_dir results/attention_analysis
"""

import argparse
import glob
import json
import os
import re

import numpy as np


def fold_stats(path):
    """(N, L, H, 3, 3) attention -> per-run summary. Rows = query, columns = key."""
    z = np.load(path)
    a = z['attention'].astype(np.float64)
    names = [str(s) for s in z['channel_names']]
    L = a.shape[1]

    # (a) branch importance: attention *received* by each branch (column mean), averaged
    # over samples, heads and queries.
    received = a.mean(axis=(0, 2, 3))              # (L, 3)
    # Off-diagonal variant on the last layer: how much attention a branch receives from the
    # other branches only.
    offdiag = a[:, -1].copy()
    idx = np.arange(3)
    offdiag[:, :, idx, idx] = 0
    importance_offdiag = offdiag.sum(axis=-1).mean(axis=(0, 1))  # (3,)

    # (b) per-layer entropy (per query row, normalised by log 3) and deviation from uniform
    ent = -(a * np.log(a + 1e-12)).sum(axis=-1)    # (N, L, H, 3)
    ent_norm = ent.mean(axis=(0, 2, 3)) / np.log(3)                    # (L,)
    dev_l1 = np.abs(a - 1 / 3).sum(axis=-1).mean(axis=(0, 2, 3))       # (L,) max 4/3
    diag = a[..., idx, idx].mean(axis=(0, 2, 3))                       # (L,) self-attention

    return dict(names=names, n=int(a.shape[0]), n_layers=L,
                received=received, importance_offdiag=importance_offdiag,
                entropy=ent_norm, dev_l1=dev_l1, diag=diag)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--attn_dir', default='results/attention_analysis',
                    help='directory of *_attention.npz files from extract_attention.py')
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.attn_dir, '*_attention.npz'))):
        m = re.search(r'rs(\d+)_fold(\d+)', os.path.basename(p))
        if not m:
            print(f'  [skip] no rs/fold in name: {os.path.basename(p)}')
            continue
        s = fold_stats(p)
        s.update(rs=int(m.group(1)), fold=int(m.group(2)))
        rows.append(s)
        print(f'  loaded rs{s["rs"]} fold{s["fold"]} (N={s["n"]})')
    if not rows:
        raise SystemExit(f'no *_attention.npz under {args.attn_dir}')

    names = rows[0]['names']
    L = rows[0]['n_layers']
    recv = np.stack([r['received'] for r in rows])            # (K, L, 3)
    imp = np.stack([r['importance_offdiag'] for r in rows])   # (K, 3)
    ent = np.stack([r['entropy'] for r in rows])              # (K, L)
    dev = np.stack([r['dev_l1'] for r in rows])
    dia = np.stack([r['diag'] for r in rows])
    seeds = sorted({r['rs'] for r in rows})

    print(f'\n{len(rows)} runs ({len(seeds)} seeds x fold), layers={L}, channels={names}')
    print('\n=== (a) Branch importance ===')
    print('last layer, off-diagonal:')
    for i, n in enumerate(names):
        print(f'  {n:<9} {imp[:, i].mean():.4f} +- {imp[:, i].std():.4f}')
    print('received attention per layer (column mean; uniform = 0.3333):')
    for li in range(L):
        cells = '  '.join(f'{n}={recv[:, li, i].mean():.4f}+-{recv[:, li, i].std():.4f}'
                          for i, n in enumerate(names))
        print(f'  layer{li + 1}: {cells}')
    print('  per seed (last layer, off-diagonal):')
    for s in seeds:
        sel = [k for k, r in enumerate(rows) if r['rs'] == s]
        print(f'    rs{s}: ' + '  '.join(f'{n}={imp[sel, i].mean():.4f}'
                                         for i, n in enumerate(names)))

    # Per-run argmax: supports (or refutes) statements like "branch X dominates in every run"
    print(f'  per-run winner (argmax per layer, n={len(rows)}):')
    for li in range(L):
        win = np.bincount(recv[:, li].argmax(axis=1), minlength=3)
        margin = np.sort(recv[:, li], axis=1)
        print(f'    layer{li + 1}: '
              + ', '.join(f'{n}={win[i]}/{len(rows)}' for i, n in enumerate(names))
              + f'  | min gap 1st-2nd={np.min(margin[:, -1] - margin[:, -2]):.4f}'
              + f', min 1st value={margin[:, -1].min():.4f}')

    print('\n=== (b) Attention distribution per layer ===')
    print('  layer  entropy(/log3)      L1 dev from uniform(max 1.333)   self-attn(diag)')
    for li in range(L):
        print(f'  {li + 1:<6} {ent[:, li].mean():.4f} +- {ent[:, li].std():.4f}      '
              f'{dev[:, li].mean():.4f} +- {dev[:, li].std():.4f}                '
              f'{dia[:, li].mean():.4f} +- {dia[:, li].std():.4f}')

    out = {
        'n_runs': len(rows), 'seeds': seeds, 'channel_names': names,
        'branch_importance_offdiag_last_layer': {
            n: {'mean': float(imp[:, i].mean()), 'std': float(imp[:, i].std())}
            for i, n in enumerate(names)},
        'received_attention_by_layer': [
            {n: {'mean': float(recv[:, li, i].mean()), 'std': float(recv[:, li, i].std())}
             for i, n in enumerate(names)} for li in range(L)],
        'layer_entropy_norm': [{'mean': float(ent[:, li].mean()), 'std': float(ent[:, li].std())}
                               for li in range(L)],
        'layer_l1_dev_uniform': [{'mean': float(dev[:, li].mean()), 'std': float(dev[:, li].std())}
                                 for li in range(L)],
        'layer_self_attention': [{'mean': float(dia[:, li].mean()), 'std': float(dia[:, li].std())}
                                 for li in range(L)],
        'per_run': [{'rs': r['rs'], 'fold': r['fold'], 'n': r['n'],
                     'importance_offdiag': r['importance_offdiag'].tolist(),
                     'received_by_layer': r['received'].tolist(),
                     'entropy': r['entropy'].tolist(), 'dev_l1': r['dev_l1'].tolist(),
                     'self_attention': r['diag'].tolist()}
                    for r in rows],
    }
    out_path = os.path.join(args.attn_dir, 'attention_summary.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
