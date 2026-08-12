"""Extract self-attention weights from trained Transformer Fusion checkpoints.

Re-uses the training configuration (`experiments/train_10fold.py`) so the fold split,
dataset and normalization statistics are exactly the ones used at training time; the model
is then run over each fold's test set with the attention weights captured.

The model itself is unchanged: `nn.TransformerEncoderLayer` discards its attention weights,
so this script wraps each layer's `self_attn` to ask for them (see `capture_attention`).

Output, one .npz per (random_state, fold):
    attention     (N, num_layers, nhead, 3, 3)  rows = query branch, cols = key branch
    labels        (N,)
    logits        (N, 2)
    channel_names (3,)   branch order, i.e. config.model.fusion.use_methods

Usage:
    python analysis/extract_attention.py --config config/physionet_10fold_transformer.yaml
    python analysis/extract_attention.py --config ... --rs 374 --folds 1 2
    python analysis/extract_attention.py --self_test        # no data/checkpoints needed
"""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_loader import load_config
from src.utils.transforms import get_val_transform
from src.datasets.image_datasets import MultiBranchDataset, load_data
from src.models.transformer_fusion_model import TransformerFusionModel


@contextmanager
def capture_attention(model):
    """Yield a list that collects the per-layer attention weights of every forward pass.

    `nn.TransformerEncoderLayer` calls `self_attn(..., need_weights=False)` and throws the
    weights away. Wrapping `self_attn.forward` flips that flag and keeps the result. The
    fused fast path would bypass `self_attn.forward` entirely, so it is disabled here.
    """
    store = []
    originals = []
    for layer in model.transformer.layers:
        mha = layer.self_attn
        originals.append((mha, mha.forward))

        def forward(query, key, value, *a, _orig=mha.forward, **kw):
            kw['need_weights'] = True
            kw['average_attn_weights'] = False  # keep the heads separate
            out, weights = _orig(query, key, value, *a, **kw)
            store.append(weights)  # (B, nhead, 3, 3)
            return out, weights

        mha.forward = forward

    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield store
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for mha, orig in originals:
            mha.forward = orig


@torch.no_grad()
def extract(model, loader, device, num_layers):
    """Run the loader and return (attention, labels, logits) as numpy arrays."""
    model.eval()
    attn, labels, logits = [], [], []
    with capture_attention(model) as store:
        for X, Y in loader:
            store.clear()
            X = X.to(device, dtype=torch.float32)
            out = model(X)
            assert len(store) == num_layers, (
                f'captured {len(store)} attention maps, expected {num_layers} - the '
                f'transformer fast path was not disabled')
            attn.append(torch.stack(store, dim=1).float().cpu())  # (B, L, H, 3, 3)
            logits.append(out.float().cpu())
            labels.append(Y)
    return (torch.cat(attn).numpy(), torch.cat(labels).numpy(), torch.cat(logits).numpy())


def build_model(config, ckpt_path, device):
    model = TransformerFusionModel(
        num_classes=config.model.num_classes,
        d_model=config.model.transformer.d_model,
        nhead=config.model.transformer.nhead,
        num_layers=config.model.transformer.num_layers,
        dropout=config.model.transformer.dropout,
        pretrained=False,  # weights come from the checkpoint
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def self_test():
    """Randomly initialised model: the capture must fire once per layer and be a softmax."""
    # d_model is fixed by the EfficientNet-B0 branch feature size (1280).
    model = TransformerFusionModel(num_classes=2, d_model=1280, nhead=4, num_layers=3,
                                   pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 3, 32, 32)
    with capture_attention(model) as store:
        model(x)
    assert len(store) == 3, f'expected 3 attention maps, got {len(store)}'
    a = torch.stack(store, dim=1)
    assert a.shape == (2, 3, 4, 3, 3), a.shape
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5), \
        'attention rows do not sum to 1'
    print('self_test OK: attention captured for every layer, rows sum to 1')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='the config the models were trained with')
    ap.add_argument('--rs', type=int, nargs='*', default=None,
                    help='random states (default: all in the config)')
    ap.add_argument('--folds', type=int, nargs='*', default=None,
                    help='1-based fold numbers (default: all)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', default=None,
                    help='output dir (default: <save_dir>/attention_analysis)')
    ap.add_argument('--self_test', action='store_true',
                    help='check the attention capture without data or checkpoints')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.config:
        raise SystemExit('--config is required (or use --self_test)')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)
    if config.model.type != 'transformer':
        raise SystemExit(f'attention only exists for model.type=transformer, '
                         f'got {config.model.type}')

    methods = tuple(config.model.fusion.use_methods)
    out_dir = Path(args.out or Path(config.logging.save_dir) / 'attention_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same file list and the same StratifiedKFold seeding as train_10fold.py, so fold k
    # here is fold k there.
    all_files = load_data(os.path.join(config.dataset.data_dir, 'fusion'), file_ext='.npy',
                          exclude_c05=config.dataset.get('exclude_c05', False))
    y = [0 if 'label0' in f else 1 for f in all_files]
    data_paths = {m: os.path.join(config.dataset.data_dir, m) for m in methods}
    val_transform = get_val_transform(tuple(config.training.image_size))

    random_states = args.rs if args.rs else config.experiment.random_states
    folds = args.folds if args.folds else list(range(1, config.experiment.n_folds + 1))

    for rs in random_states:
        skf = StratifiedKFold(n_splits=config.experiment.n_folds, shuffle=True, random_state=rs)
        splits = list(skf.split(all_files, y))
        for fold in folds:
            exp_name = f"{config.logging.project_name}_rs{rs}_fold{fold}"
            ckpt = Path(config.logging.save_dir) / 'models' / f"{exp_name}.pth"
            if not ckpt.exists():
                print(f'[skip] no checkpoint: {ckpt}')
                continue

            test_files = [all_files[i] for i in splits[fold - 1][1]]
            dataset = MultiBranchDataset(
                test_files, transform=val_transform, use_methods=methods,
                data_paths=data_paths, mode='test',
                stats_file=f"stats/{config.logging.project_name}_rs{rs}_fold{fold}.npz")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

            model = build_model(config, ckpt, device)
            attention, labels, logits = extract(model, loader, device,
                                                config.model.transformer.num_layers)

            acc = float((logits.argmax(1) == labels).mean())
            out_path = out_dir / f'{exp_name}_attention.npz'
            np.savez_compressed(out_path, attention=attention, labels=labels, logits=logits,
                                channel_names=np.array(methods), files=np.array(test_files))
            print(f'{out_path.name}: n={len(labels)} acc={acc:.4f} '
                  f'attention{attention.shape}')


if __name__ == '__main__':
    main()
