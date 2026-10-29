"""Linear probe of branch encoder features (no retraining of the networks).

For every run, the GAP features (1,280-d) of each branch are extracted for the run's own
train/valid/test segments: the three EfficientNet-B0 branches of late/hybrid models (before
the channel embedding), or the single backbone of single-transform models. A probe per
branch: StandardScaler fit on train -> L2 logistic regression, C chosen by validation AUC
from {0.01, 0.1, 1, 10} -> applied to test. Comparing a fusion branch with the single model of
the same transform, seed and fold shows how much the branch learned on its own.

Usage:
    python analysis/branch_probe.py --runs "results/runs/10fold_*_hybrid_*.json" \
        "results/runs/10fold_*_single_*.json" --out results/branch_probe.csv
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SegmentDataset, label_of, run_inputs, valid_transform  # noqa: E402
from src.engine import logits_of, prepare  # noqa: E402
from src.models import build_model  # noqa: E402

CS = (0.01, 0.1, 1, 10)


@torch.no_grad()
def gap_features(model, loader, device, taps):
    """Run the model and return one (N, 1280) array per tapped module (GAP of its output)."""
    captured = {}
    hooks = [m.register_forward_hook(lambda mod, i, out, k=k: captured.__setitem__(k, out)) for k, m in enumerate(taps)]
    feats = [[] for _ in taps]
    model.eval()
    for x, _ in loader:
        logits_of(model, prepare(x, device))
        for k in range(len(taps)):
            feats[k].append(F.adaptive_avg_pool2d(captured[k].float(), 1).flatten(1).cpu().numpy())
    for h in hooks:
        h.remove()
    return [np.concatenate(f) for f in feats]


def probe(train, valid, test):
    """Each argument is (features, labels). Returns the chosen C and the valid/test scores."""
    scaler = StandardScaler().fit(train[0])
    xs = [scaler.transform(f) for f, _ in (train, valid, test)]
    fits = [LogisticRegression(C=c, max_iter=5000).fit(xs[0], train[1]) for c in CS]
    valid_auc = [roc_auc_score(valid[1], m.predict_proba(xs[1])[:, 1]) for m in fits]
    best = int(np.argmax(valid_auc))
    p = fits[best].predict_proba(xs[2])[:, 1]
    return {'C': CS[best], 'valid_auc': valid_auc[best], 'test_auc': roc_auc_score(test[1], p),
            'test_accuracy': float(((p >= 0.5) == test[1]).mean())}


def probe_run(run, device, batch_size, num_workers):
    cfg = run['config']
    read, *splits = run_inputs(cfg)
    stats = np.load(cfg['stats'])
    model = build_model(cfg['model_type'], pretrained=False)
    model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu'))
    model.to(device)
    single = cfg['model_type'] == 'single'
    taps = [model.features] if single else list(model.feature_extractors)
    branches = [cfg['image_trans']] if single else cfg['channel_order']

    per_split = []
    for names in splits:
        loader = DataLoader(SegmentDataset(names, read, stats['mean'], stats['std'], valid_transform),
                            batch_size=batch_size, num_workers=num_workers)
        per_split.append((gap_features(model, loader, device, taps), np.array([label_of(n) for n in names])))
    return [{'run': Path(cfg['checkpoint']).stem, 'source': cfg['model_type'], 'branch': b, 'seed': cfg['seed'],
             'fold': cfg['fold'], **probe(*[(f[k], y) for f, y in per_split]),
             'n_train': len(splits[0]), 'n_test': len(splits[2])} for k, b in enumerate(branches)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='globs of runs/*.json (late/hybrid/single)')
    p.add_argument('--out', default='results/branch_probe.csv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    paths = sorted({f for g in args.runs for f in glob.glob(g)})
    with open(args.out, 'w', newline='') as f:
        writer = None
        for path in paths:
            run = json.load(open(path))
            if run['config']['model_type'] not in ('single', 'late', 'hybrid'):
                continue
            rows = probe_run(run, device, args.batch_size, args.num_workers)
            writer = writer or csv.DictWriter(f, fieldnames=list(rows[0]))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            print(f"{rows[0]['run']}: " + ' '.join(f"{r['branch']}={r['test_auc']:.4f} (C={r['C']})" for r in rows))


if __name__ == '__main__':
    main()
