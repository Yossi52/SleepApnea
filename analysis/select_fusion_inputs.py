"""Pick the three fusion inputs from the single-transform runs, using validation data only.

Rule (as in the paper): for every transform, take the validation loss of the best epoch
(minimum over epochs) of each seed on the withheld protocol, average over seeds, and keep the
three lowest. Their rank order is the early-fusion channel order (R, G, B) and the branch order.
No test metric is read.

Usage:
    python analysis/select_fusion_inputs.py --runs "results/runs/withheld_*_single_rs*.json"
    -> prints the ranking and writes results/fusion_inputs.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def rank(runs):
    """runs: iterable of run dicts -> [(transform, mean, sd, seeds)] sorted by mean best val_loss."""
    best = defaultdict(dict)
    for r in runs:
        cfg = r['config']
        if cfg['model_type'] != 'single' or cfg['protocol'] != 'withheld':
            continue
        best[cfg['image_trans']][cfg['seed']] = min(r['history']['val_loss'])
    seeds = {t: sorted(s) for t, s in best.items()}
    if len({tuple(s) for s in seeds.values()}) > 1:
        raise SystemExit(f'transforms were run with different seeds: {seeds}')
    rows = [(t, float(np.mean(list(s.values()))), float(np.std(list(s.values()), ddof=1)) if len(s) > 1 else 0.0,
             seeds[t]) for t, s in best.items()]
    return sorted(rows, key=lambda r: r[1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', required=True, help='glob of runs/*.json from experiments/train.py')
    p.add_argument('--out', default='results/fusion_inputs.json')
    args = p.parse_args()

    rows = rank(json.load(open(f)) for f in sorted(glob.glob(args.runs)))
    if len(rows) < 3:
        raise SystemExit(f'need at least 3 transforms, found {[r[0] for r in rows]}')
    for i, (t, mean, sd, seeds) in enumerate(rows, 1):
        print(f'{i}. {t:12s} best-epoch val_loss {mean:.5f} +- {sd:.5f} (n={len(seeds)})')
    top = [r[0] for r in rows[:3]]
    print(f'gap between rank 3 and 4: {rows[3][1] - rows[2][1]:.5f}' if len(rows) > 3 else '')
    print(f'--channel_order {",".join(top)}')
    with open(args.out, 'w') as f:
        json.dump({'rule': 'mean over seeds of min-epoch validation loss, withheld protocol, ascending',
                   'fusion_inputs': top,
                   'ranking': [{'transform': t, 'val_loss_mean': m, 'val_loss_sd': s, 'seeds': sd}
                               for t, m, s, sd in rows]}, f, indent=2)


if __name__ == '__main__':
    main()
