"""Branch ablation at test time for trained fusion runs (no retraining).

For each run and each input branch (the early-fusion channel, or the late/hybrid branch):
  zero     the branch is set to 0 after normalization, i.e. to the training-set mean
  shuffle  the branch is taken from another test segment of the same transform; one fixed
           derangement of the test set (seed 20261026) is shared by all branches
The other two branches keep their own segment. The unmodified condition ('none') is checked
against the accuracy the run recorded at training time.

Usage:
    python analysis/branch_ablation.py --runs "results/runs/*_hybrid_rs*.json" --out results/branch_ablation.csv
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SegmentDataset, run_inputs, valid_transform  # noqa: E402
from src.engine import metrics, predict  # noqa: E402
from src.models import build_model  # noqa: E402

PERM_SEED = 20261026


def derangement(n, seed=PERM_SEED):
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


class Paired(Dataset):
    """(own input, input of the permutation partner, label); the partner is stacked as an extra leading dim."""

    def __init__(self, base, perm):
        self.base, self.perm = base, perm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return torch.stack([x, self.base[int(self.perm[i])][0]]), y


def ablate(run, device, batch_size, num_workers):
    cfg = run['config']
    read, _, _, test = run_inputs(cfg)
    stats = np.load(cfg['stats'])
    model = build_model(cfg['model_type'], pretrained=False)
    model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu'))
    model.to(device)
    base = SegmentDataset(test, read, stats['mean'], stats['std'], valid_transform)
    loader = DataLoader(Paired(base, derangement(len(base))), batch_size=batch_size, num_workers=num_workers)

    def edit(branch, mode):
        # x is the prepared pair batch (B, 2, ...): index 0 = own input, 1 = partner
        def f(x):
            own = x[:, 0].clone()
            if mode == 'zero':
                own[:, branch] = 0
            elif mode == 'shuffle':
                own[:, branch] = x[:, 1, branch]
            return own
        return f

    conditions = [('none', 'none')] + [(b, m) for m in ('zero', 'shuffle') for b in cfg['channel_order']]
    rows = []
    for branch, mode in conditions:
        idx = None if branch == 'none' else cfg['channel_order'].index(branch)
        y, prob, pred = predict(model, loader, device, edit(idx, mode))
        rows.append({'run': Path(cfg['checkpoint']).stem, 'protocol': cfg['protocol'], 'model_type': cfg['model_type'],
                     'seed': cfg['seed'], 'fold': cfg['fold'], 'branch': branch, 'replacement': mode,
                     **metrics(y, prob, pred), 'recorded_accuracy': run['test_results']['metrics']['accuracy']})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True, help='glob of runs/*.json of early/late/hybrid models')
    p.add_argument('--out', default='results/branch_ablation.csv')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    with open(args.out, 'w', newline='') as f:
        writer = None
        for path in sorted(glob.glob(args.runs)):
            run = json.load(open(path))
            if run['config']['model_type'] == 'single':
                continue
            rows = ablate(run, device, args.batch_size, args.num_workers)
            writer = writer or csv.DictWriter(f, fieldnames=list(rows[0]))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            base = rows[0]
            print(f"{base['run']}: accuracy {base['accuracy']:.4f} (recorded {base['recorded_accuracy']:.4f}); "
                  + ' '.join(f"{r['replacement']}:{r['branch']}={r['auc'] - base['auc']:+.3f}" for r in rows[1:]))


if __name__ == '__main__':
    assert sorted(derangement(500)) == list(range(500)) and (derangement(500) != np.arange(500)).all()
    main()
