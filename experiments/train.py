"""Train and test one model type on the withheld or segment-wise 10-fold protocol.

One run = one model x one seed (x one fold). Each run writes, under --out_dir:
  runs/<job>.json                               config, per-epoch history, test metrics and predictions
  predictions/<job>_test_segment_predictions.csv per-segment test output (input of the AHI analysis)
  stats/<job>.npz                               normalization statistics of the training split
  checkpoints/<job>.pt                          weights of the best validation-loss epoch
Finished runs are skipped, so an interrupted sweep can be restarted with the same command.

Examples:
  # single transforms on the withheld split (inputs of select_fusion_inputs.py)
  python experiments/train.py --data_dir DATA --protocol withheld --model_type single \
      --image_trans scalogram --seeds 43 128 154 239 374 451 562 687 716 895

  # hybrid fusion, 10-fold, using the selected inputs as R,G,B
  python experiments/train.py --data_dir DATA --protocol 10fold --model_type hybrid \
      --channel_order scalogram,gadf,spectrogram --seeds 128 374 562
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (PROTOCOLS, TRANSFORMS, SegmentDataset, compute_stats, label_of,  # noqa: E402
                      run_inputs, train_transform, valid_transform)
from src.engine import HYPERPARAMS, fit, metrics, predict  # noqa: E402
from src.models import MODEL_TYPES, build_model  # noqa: E402


def job_name(cfg):
    tag = cfg['image_trans'] if cfg['model_type'] == 'single' else '-'.join(cfg['channel_order'])
    name = f"{cfg['protocol']}_{tag}_{cfg['model_type']}_rs{cfg['seed']}"
    if cfg['protocol'] == '10fold':
        name += f"_fold{cfg['fold']}"
    if not (cfg['pretrained'] and cfg['seg_drop']):
        name += f"_pre{int(cfg['pretrained'])}sqc{int(cfg['seg_drop'])}"
    return name


def write_segment_csv(path, names, y, prob, pred):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset_index', 'file_path', 'record_id', 'segment_index', 'label_from_filename',
                    'y_true', 'y_pred', 'prob_class0', 'prob_class1'])
        for i, n in enumerate(names):
            record, segment, _ = n[:-4].split('_')
            w.writerow([i, n, record, int(segment[len('segment'):]), label_of(n), y[i], pred[i], 1 - prob[i], prob[i]])


def run(cfg, args, device):
    out = Path(args.out_dir)
    job = job_name(cfg)
    run_path = out / 'runs' / f'{job}.json'
    if run_path.exists():
        print(f'[skip] {job} already finished')
        return
    for sub in ('runs', 'predictions', 'stats', 'checkpoints'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, 'stats': str(out / 'stats' / f'{job}.npz'), 'checkpoint': str(out / 'checkpoints' / f'{job}.pt')}

    read, train, valid, test = run_inputs(cfg)
    print(f'{job}: train {len(train)}, valid {len(valid)}, test {len(test)}')
    mean, std = compute_stats(read, train)
    np.savez(cfg['stats'], mean=mean, std=std)

    def loader(names, transform, shuffle):
        return DataLoader(SegmentDataset(names, read, mean, std, transform), batch_size=cfg['batch_size'],
                          shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    y_train = np.array([label_of(n) for n in train])
    class_weights = torch.tensor(len(y_train) / (2 * np.bincount(y_train, minlength=2)), dtype=torch.float32)
    model = build_model(cfg['model_type'], cfg['pretrained']).to(device)
    history = fit(model, loader(train, train_transform, True), loader(valid, valid_transform, False),
                  cfg['model_type'], args.epochs, args.patience, class_weights, cfg['checkpoint'], device)

    model.load_state_dict(torch.load(cfg['checkpoint'], map_location=device))
    y, prob, pred = predict(model, loader(test, valid_transform, False), device)
    result = metrics(y, prob, pred)
    print(f'{job}: ' + ' '.join(f'{k}={v:.4f}' for k, v in result.items()))
    write_segment_csv(out / 'predictions' / f'{job}_test_segment_predictions.csv', test, y, prob, pred)
    run_path.write_text(json.dumps({
        'config': cfg, 'n_train': len(train), 'n_valid': len(valid), 'n_test': len(test), 'history': history,
        'test_results': {'metrics': result, 'predictions': {
            'true_labels': y.tolist(), 'pred_labels': pred.tolist(),
            'pred_probs': np.stack([1 - prob, prob], 1).tolist()}}}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True, help='holds <transform>_drop/ and fusion_drop/ image dirs')
    p.add_argument('--protocol', choices=PROTOCOLS, required=True)
    p.add_argument('--model_type', choices=MODEL_TYPES, required=True)
    p.add_argument('--image_trans', choices=TRANSFORMS, help='single only')
    p.add_argument('--channel_order', default='scalogram,gadf,spectrogram',
                   help='fusion inputs in R,G,B (= branch 0,1,2) order')
    p.add_argument('--seeds', type=int, nargs='+', default=[128, 374, 562])
    p.add_argument('--folds', type=int, nargs='+', default=list(range(1, 11)), help='10fold only')
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seg_drop', action=argparse.BooleanOptionalAction, default=True,
                   help='use the quality-controlled *_drop segment set')
    p.add_argument('--batch_size', type=int, default=None, help='default: 16 for single, 32 otherwise')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out_dir', default='results')
    args = p.parse_args()
    if args.model_type == 'single' and not args.image_trans:
        p.error('--image_trans is required for --model_type single')
    order = args.channel_order.split(',')
    if len(order) != 3 or not set(order) <= set(TRANSFORMS):
        p.error(f'--channel_order needs three of {TRANSFORMS}')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    folds = args.folds if args.protocol == '10fold' else [None]
    for seed in args.seeds:
        for fold in folds:
            torch.manual_seed(seed)
            run({'data_dir': str(Path(args.data_dir).resolve()), 'protocol': args.protocol,
                 'model_type': args.model_type, 'image_trans': args.image_trans,
                 'channel_order': None if args.model_type == 'single' else order, 'seed': seed, 'fold': fold,
                 'pretrained': args.pretrained, 'seg_drop': args.seg_drop,
                 'batch_size': args.batch_size or HYPERPARAMS[args.model_type][4]}, args, device)


if __name__ == '__main__':
    main()
