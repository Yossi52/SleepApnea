"""Self-attention weights of trained hybrid (transformer) fusion runs on their test segments.

Output, one .npz per run (input of analyze_attention.py):
    attention     (N, num_layers, nhead, 3, 3)  rows = query branch, cols = key branch
    labels        (N,)
    logits        (N, 2)
    channel_names (3,)  branch order

Usage:
    python analysis/extract_attention.py --runs "results/runs/*_hybrid_*.json" --out results/attention_analysis
    python analysis/extract_attention.py --self_test        # no data or checkpoints needed
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SegmentDataset, run_inputs, valid_transform  # noqa: E402
from src.engine import prepare  # noqa: E402
from src.models import build_model  # noqa: E402


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    attn, labels, logits = [], [], []
    for x, y in loader:
        out, weights = model(prepare(x, device), return_attention_weights=True)
        attn.append(torch.stack(weights, dim=1).float().cpu())
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(attn).numpy(), torch.cat(labels).numpy(), torch.cat(logits).numpy()


def self_test():
    model = build_model('hybrid', pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    a, _, _ = extract(model, [(x, torch.zeros(2))], torch.device('cpu'))
    assert a.shape == (2, 3, 8, 3, 3), a.shape
    assert np.allclose(a.sum(-1), 1, atol=1e-5), 'attention rows do not sum to 1'
    print('self_test OK: attention captured for every layer, rows sum to 1')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', help='glob of runs/*.json of hybrid models')
    p.add_argument('--out', default='results/attention_analysis')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--self_test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.runs:
        p.error('--runs is required (or use --self_test)')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(args.runs)):
        cfg = json.load(open(path))['config']
        if cfg['model_type'] not in ('hybrid', 'mb_hybrid'):
            continue
        read, _, _, test = run_inputs(cfg)
        stats = np.load(cfg['stats'])
        model = build_model(cfg['model_type'], pretrained=False)
        model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu'))
        loader = DataLoader(SegmentDataset(test, read, stats['mean'], stats['std'], valid_transform),
                            batch_size=args.batch_size, num_workers=4)
        attention, labels, logits = extract(model.to(device), loader, device)
        name = Path(cfg['checkpoint']).stem
        np.savez_compressed(out_dir / f'{name}_attention.npz', attention=attention, labels=labels, logits=logits,
                            channel_names=np.array(cfg['channel_order']), files=np.array(test))
        print(f'{name}: n={len(labels)} acc={(logits.argmax(1) == labels).mean():.4f}')


if __name__ == '__main__':
    main()
