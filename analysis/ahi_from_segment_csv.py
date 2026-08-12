"""Recording-level AHI analysis from segment-prediction CSVs (Apnea-ECG withheld set x01-x35).

No model loading, no inference. This script consumes the per-segment CSVs written by the
patched `test_model()` in subject_wise_split.py / subject_wise_split_fusion_models.py
(`*_test_segment_predictions.csv`), so the predictions are guaranteed to come from the
exact preprocessing path used at training time.

Expected CSV columns:
    dataset_index, file_path, record_id, segment_index,
    label_from_filename, y_true, y_pred, prob_class0, prob_class1

Usage:
    python ahi_from_segment_csv.py \
        --pred_csv "/path/to/histories/*_test_segment_predictions.csv" \
        --addinfo /path/to/raw_data/additional-information.txt \
        --annos   /path/to/raw_data/test-dataset-annos.txt \
        --output_dir ./ahi_out

Multiple CSVs (different models / seeds) can be passed at once; each file becomes one
(model_type, seed) group and metrics are reported as mean +/- sd across seeds.
"""
import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

RECORD_IDS = [f'x{i:02d}' for i in range(1, 36)]
PRIOR_WORK_MAE_RANGE = (2.593, 2.71)
REQUIRED_COLUMNS = ['record_id', 'y_true', 'y_pred']


# ============================================================================
# 1. Ground truth
# ============================================================================

def parse_test_dataset_annos(path):
    """test-dataset-annos.txt -> {record_id: per-minute 0/1 array}.

    Format: bare record-id line, then indented '<hour_idx> <A/N string>' lines.
    Hour lines are not always 60 characters (final block is a partial hour), so we simply
    concatenate in order.
    """
    recs, cur = {}, None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            if not line.startswith(' '):
                cur = line.strip()
                recs[cur] = []
            else:
                parts = line.strip().split(' ', 1)
                recs[cur].append(parts[1] if len(parts) == 2 else parts[0])
    return {rec: np.array([1 if c == 'A' else 0 for c in ''.join(chunks)], dtype=np.int64)
            for rec, chunks in recs.items()}


def parse_additional_information(path):
    """additional-information.txt -> DataFrame indexed by withheld record id."""
    rows = []
    with open(path) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            while fields and fields[-1] == '':
                fields.pop()
            if len(fields) < 8:
                continue
            rec = fields[0].strip()
            if not re.match(r'^x\d{2}$', rec):
                continue
            try:
                rows.append({
                    'record': rec,
                    'length_min': float(fields[1]),
                    'non_apnea_min': float(fields[2]),
                    'apnea_min': float(fields[3]),
                    'hours_w_apnea': float(fields[4]),
                    'AI': float(fields[5]),
                    'HI': float(fields[6]),
                    'AHI': float(fields[7]),
                    'age': fields[8] if len(fields) > 8 else None,
                    'sex': fields[9] if len(fields) > 9 else None,
                })
            except ValueError:
                continue
    if not rows:
        raise ValueError(f'no x-records parsed from {path}')
    return pd.DataFrame(rows).set_index('record')


def build_ground_truth_table(annos, addinfo):
    rows, discrepancies = [], []
    for rec in RECORD_IDS:
        if rec not in addinfo.index:
            discrepancies.append(f'{rec}: missing from additional-information.txt')
            continue
        info = addinfo.loc[rec]
        arr = annos.get(rec) if annos else None
        annos_minutes = len(arr) if arr is not None else None
        annos_apnea = int(arr.sum()) if arr is not None else None
        if annos_apnea is not None and abs(annos_apnea - info['apnea_min']) > 1:
            discrepancies.append(
                f'{rec}: annos apnea minutes={annos_apnea} vs additional-information '
                f'apnea_min={info["apnea_min"]:.0f}')
        if annos_minutes is not None and abs(annos_minutes - info['length_min']) > 2:
            discrepancies.append(
                f'{rec}: annos total minutes={annos_minutes} vs length_min={info["length_min"]:.0f}')
        rows.append({
            'record': rec,
            'annos_total_minutes': annos_minutes,
            'annos_apnea_minutes': annos_apnea,
            'real_length_min': info['length_min'],
            'real_apnea_min': info['apnea_min'],
            'real_AI': info['AI'],
            'real_HI': info['HI'],
            'real_AHI': info['AHI'],
        })
    return pd.DataFrame(rows).set_index('record'), discrepancies


# ============================================================================
# 2. Segment CSV loading
# ============================================================================

def infer_model_type(name):
    low = os.path.basename(name).lower()
    if 'transformer' in low or 'hybrid' in low:
        return 'hybrid'
    if 'late' in low:
        return 'late'
    if 'fusion' in low:          # early fusion = single backbone over a 3-channel fused image
        return 'early'
    return 'single'


def infer_seed(name):
    m = re.search(r'rs[:_ ]?(\d+)', os.path.basename(name))
    return int(m.group(1)) if m else -1


def load_prediction_csvs(patterns, model_type=None, seed=None):
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise FileNotFoundError(f'no CSV matched: {pat}')
        paths.extend(hits)

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'{path}: missing column(s) {missing}')
        if 'prob_class1' in df.columns:
            df['y_prob'] = df['prob_class1']
        elif 'y_prob' in df.columns:
            pass
        else:
            df['y_prob'] = np.nan
        df['model_type'] = model_type or infer_model_type(path)
        df['seed'] = seed if seed is not None else infer_seed(path)
        df['source_csv'] = os.path.basename(path)
        frames.append(df)
        print(f'[load] {os.path.basename(path)}: n={len(df)} '
              f'model_type={df["model_type"].iloc[0]} seed={df["seed"].iloc[0]}')
    return pd.concat(frames, ignore_index=True)


def sanity_check_segments(seg_df, gt_table):
    """Checks that must pass before any AHI number is trusted."""
    problems = []

    if 'label_from_filename' in seg_df.columns:
        mism = seg_df['label_from_filename'].notna() & \
               (seg_df['label_from_filename'] != seg_df['y_true'])
        if mism.any():
            problems.append(f'{int(mism.sum())} rows where the filename label disagrees with '
                            f'y_true -> DataLoader ordering is misaligned, stop here')

    unknown = sorted(set(seg_df['record_id'].dropna()) - set(RECORD_IDS))
    if unknown:
        problems.append(f'record ids outside the withheld set: {unknown}')

    for (mt, seed), g in seg_df.groupby(['model_type', 'seed']):
        dup = g.duplicated(subset=['record_id', 'segment_index']).sum() \
            if 'segment_index' in g.columns else 0
        if dup:
            problems.append(f'{mt}/seed{seed}: {int(dup)} duplicated (record_id, segment_index) rows')
        for rec, gg in g.groupby('record_id'):
            if rec not in gt_table.index:
                continue
            n_min = gt_table.loc[rec, 'real_length_min']
            if len(gg) > n_min + 2:
                problems.append(f'{mt}/seed{seed}/{rec}: {len(gg)} segments scored but recording is '
                                f'only {n_min:.0f} minutes long')
    return problems


def segment_level_metrics(seg_df):
    """Recompute segment-level metrics so they can be checked against the manuscript table.

    If these do not reproduce the reported withheld-test numbers, the CSV is wrong and the
    AHI analysis below is meaningless.
    """
    rows = []
    for (mt, seed), g in seg_df.groupby(['model_type', 'seed']):
        y, p = g['y_true'].to_numpy(int), g['y_pred'].to_numpy(int)
        tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        pr = g['y_prob'].to_numpy(float)
        auc = float('nan')
        if np.isfinite(pr).all() and len(np.unique(y)) == 2:
            order = np.argsort(pr, kind='mergesort')
            ranks = np.empty(len(pr), dtype=float)
            ranks[order] = np.arange(1, len(pr) + 1)
            # average ranks for ties
            s = pd.Series(pr)
            ranks = s.rank(method='average').to_numpy()
            n1, n0 = int(y.sum()), int((1 - y).sum())
            auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append({
            'model_type': mt, 'seed': seed, 'n': len(g),
            'accuracy': (tp + tn) / max(len(g), 1),
            'sensitivity': tp / (tp + fn) if (tp + fn) else float('nan'),
            'specificity': tn / (tn + fp) if (tn + fp) else float('nan'),
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan'),
            'auc': auc,
            'pred_positive_rate': float(p.mean()),
            'true_positive_rate': float(y.mean()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 3. Recording-level aggregation & metrics
# ============================================================================

def aggregate_recording_level(seg_df, gt_table):
    rows = []
    for (mt, seed, rec), g in seg_df.groupby(['model_type', 'seed', 'record_id']):
        if rec not in gt_table.index:
            continue
        duration_hours = gt_table.loc[rec, 'real_length_min'] / 60.0
        pred_apnea_min = int(g['y_pred'].sum())
        true_apnea_min = int(g['y_true'].sum())
        rows.append({
            'model_type': mt, 'seed': seed, 'record_id': rec,
            'n_segments_scored': len(g),
            'duration_hours': duration_hours,
            'pred_apnea_minutes': pred_apnea_min,
            'true_apnea_minutes_available': true_apnea_min,
            'pred_apnea_index': pred_apnea_min / duration_hours,
            'true_minute_apnea_index': true_apnea_min / duration_hours,
            'real_AHI': gt_table.loc[rec, 'real_AHI'],
            'real_AI': gt_table.loc[rec, 'real_AI'],
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    diff = y_pred - y_true
    mae = float(np.abs(diff).mean())
    rmse = float(math.sqrt((diff ** 2).mean()))
    if n > 1:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        rank_t = pd.Series(y_true).rank(method='average').to_numpy()
        rank_p = pd.Series(y_pred).rank(method='average').to_numpy()
        rho = float(np.corrcoef(rank_t, rank_p)[0, 1])  # scipy-free Spearman
        ss_res = float((diff ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float('nan')
        sd = float(diff.std(ddof=1))
    else:
        r = rho = r2 = sd = float('nan')
    bias = float(diff.mean())
    return {'MAE': mae, 'RMSE': rmse, 'pearson_r': r, 'spearman_rho': rho, 'R2': r2,
            'bland_altman_bias': bias,
            'bland_altman_loa_lower': bias - 1.96 * sd,
            'bland_altman_loa_upper': bias + 1.96 * sd,
            'n': n}


def binary_class_metrics(index_values, real_ahi, threshold):
    y_true = (np.asarray(real_ahi) >= threshold).astype(int)
    y_pred = (np.asarray(index_values) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    auc = float('nan')
    if len(np.unique(y_true)) == 2:
        ranks = pd.Series(index_values).rank(method='average').to_numpy()
        n1, n0 = int(y_true.sum()), int((1 - y_true).sum())
        auc = (ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {'threshold': threshold,
            'accuracy': (tp + tn) / max(len(y_true), 1),
            'sensitivity': tp / (tp + fn) if (tp + fn) else float('nan'),
            'specificity': tn / (tn + fp) if (tn + fp) else float('nan'),
            'auc': auc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def severity_class(ahi):
    if ahi < 5:
        return 'normal'
    if ahi < 15:
        return 'mild'
    if ahi < 30:
        return 'moderate'
    return 'severe'


def abc_class(ahi):
    if ahi >= 10:
        return 'A'
    if ahi >= 5:
        return 'B'
    return 'C'


# ============================================================================
# 4. Plots
# ============================================================================

def make_plots(df, x_col, y_col, label_col, title, out_prefix, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)
    labels = df[label_col].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.7)
    for xi, yi, li in zip(x, y, labels):
        ax.annotate(li, (xi, yi), fontsize=6, alpha=0.7)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', label='identity')
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, m * xs + b, 'r-', label=f'fit (y={m:.2f}x+{b:.2f})')
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{out_prefix}_scatter.png'), dpi=150)
    plt.close(fig)

    mean_xy = (x + y) / 2
    diff = y - x
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean_xy, diff, alpha=0.7)
    ax.axhline(bias, color='k', label=f'bias={bias:.2f}')
    ax.axhline(bias + 1.96 * sd, color='r', linestyle='--', label=f'+1.96SD={bias + 1.96 * sd:.2f}')
    ax.axhline(bias - 1.96 * sd, color='r', linestyle='--', label=f'-1.96SD={bias - 1.96 * sd:.2f}')
    ax.set_xlabel(f'mean({x_col}, {y_col})')
    ax.set_ylabel(f'{y_col} - {x_col}')
    ax.set_title(f'Bland-Altman: {title}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{out_prefix}_bland_altman.png'), dpi=150)
    plt.close(fig)


# ============================================================================
# 5. Report
# ============================================================================

def df_to_md(df):
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(
            (f'{v:.3f}' if isinstance(v, float) else str(v)) for v in row) + ' |')
    return '\n'.join(lines)


def write_report(out_dir, gt_table, discrepancies, problems, seg_metrics, comparisons,
                 cls_results, severity_counts, sources):
    L = []
    L.append('# Recording-level AHI analysis from segment-prediction CSVs\n')
    L.append('Predictions were exported by the training/testing script itself '
             '(`test_model(..., per_sample_save_path=...)`), so the preprocessing path is '
             'identical to the one used to produce the manuscript results. No checkpoint was '
             'reloaded outside that pipeline.\n')
    L.append('## Source files\n')
    for s in sources:
        L.append(f'- `{s}`')
    L.append('')

    L.append('## Integrity checks\n')
    if problems:
        L.append('**FAILED - do not use the numbers below until these are resolved:**\n')
        for p in problems:
            L.append(f'- {p}')
    else:
        L.append('All checks passed: filename labels agree with `y_true`, no duplicated '
                 '(record, segment) pairs, no record scored beyond its recording length.')
    L.append('')

    L.append('## Segment-level metrics recomputed from the CSV\n')
    L.append('These must reproduce the withheld-test values reported in the manuscript. '
             'If they do not, the export is faulty and every AHI number below is meaningless.\n')
    L.append(df_to_md(seg_metrics))
    L.append('')

    L.append('## Ground truth\n')
    L.append(df_to_md(gt_table.reset_index()))
    L.append('')
    if discrepancies:
        L.append('**Discrepancies between the two ground-truth files:**\n')
        for d in discrepancies:
            L.append(f'- {d}')
    else:
        L.append('No material discrepancy between `test-dataset-annos.txt` and '
                 '`additional-information.txt`.')
    L.append('')

    L.append('## Three comparisons (keep them separate)\n')
    L.append('- **(i) predicted apnea index vs minute-label apnea index** - agreement with the '
             'proxy label the model was actually trained on.')
    L.append('- **(ii) predicted apnea index vs real PSG AHI** - the clinically meaningful '
             f'comparison; prior work reports MAE {PRIOR_WORK_MAE_RANGE[0]}-{PRIOR_WORK_MAE_RANGE[1]}.')
    L.append('- **(iii) minute-label index vs real PSG AHI** - model-independent. This bounds how '
             'well any perfect segment classifier could recover AHI from minute labels.\n')

    for (mt, key), res in comparisons.items():
        L.append(f'### {mt} - {key}\n')
        m, s = res['mean'], res['std']
        L.append('| metric | mean | sd (across seeds) |')
        L.append('|---|---|---|')
        for k in ['MAE', 'RMSE', 'pearson_r', 'spearman_rho', 'R2',
                  'bland_altman_bias', 'bland_altman_loa_lower', 'bland_altman_loa_upper']:
            L.append(f'| {k} | {m.get(k, float("nan")):.3f} | {s.get(k, float("nan")):.3f} |')
        L.append('')

    L.append('## Recording-level screening at an apnea-index threshold\n')
    for (mt, thr), res in cls_results.items():
        L.append(f'### {mt} - threshold >= {thr} events/hour\n')
        L.append('| metric | mean | sd |')
        L.append('|---|---|---|')
        for k in ['accuracy', 'sensitivity', 'specificity', 'auc']:
            L.append(f'| {k} | {res["mean"].get(k, float("nan")):.3f} | '
                     f'{res["std"].get(k, float("nan")):.3f} |')
        L.append('')

    L.append('## Severity 4-class distribution (reference only)\n')
    L.append('| class | n |')
    L.append('|---|---|')
    for c, n in severity_counts.items():
        L.append(f'| {c} | {n} |')
    small = [c for c, n in severity_counts.items() if n < 10]
    if small:
        L.append(f'\n**WARNING**: class(es) {small} have n<10 out of 35 recordings. Report for '
                 'reference only; no per-class statistical conclusion is supportable at this n.')
    L.append('')

    path = os.path.join(out_dir, 'ahi_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(L))
    return path


# ============================================================================
# 6. Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred_csv', nargs='+', required=True,
                   help='segment prediction CSV path(s); glob patterns allowed')
    p.add_argument('--addinfo', required=True, help='additional-information.txt')
    p.add_argument('--annos', default=None,
                   help='test-dataset-annos.txt (optional, used only for cross-checking)')
    p.add_argument('--output_dir', default='./ahi_out')
    p.add_argument('--model_type', default=None,
                   help='override the model label inferred from the filename')
    p.add_argument('--seed', type=int, default=None,
                   help='override the seed inferred from the filename')
    p.add_argument('--thresholds', default='5,10')
    p.add_argument('--no_plots', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    annos = parse_test_dataset_annos(args.annos) if args.annos else None
    addinfo = parse_additional_information(args.addinfo)
    gt_table, discrepancies = build_ground_truth_table(annos, addinfo)
    gt_table['official_class'] = gt_table['real_AHI'].apply(abc_class)
    gt_table['severity'] = gt_table['real_AHI'].apply(severity_class)
    gt_table.to_csv(os.path.join(args.output_dir, 'ground_truth_table.csv'))
    print(f'\n=== Ground truth ({len(gt_table)} withheld records) ===')
    print(gt_table.to_string())
    for d in discrepancies:
        print(' [gt discrepancy]', d)

    seg_df = load_prediction_csvs(args.pred_csv, args.model_type, args.seed)

    problems = sanity_check_segments(seg_df, gt_table)
    print('\n=== Integrity checks ===')
    if problems:
        for pb in problems:
            print(' [FAIL]', pb)
        print(' -> AHI numbers below are NOT trustworthy until these are fixed.')
    else:
        print(' all passed')

    seg_metrics = segment_level_metrics(seg_df)
    seg_metrics.to_csv(os.path.join(args.output_dir, 'segment_level_metrics.csv'), index=False)
    print('\n=== Segment-level metrics (compare against the manuscript table) ===')
    print(seg_metrics.to_string(index=False))

    rec_df = aggregate_recording_level(seg_df, gt_table)
    rec_df.to_csv(os.path.join(args.output_dir, 'ahi_recording_level.csv'), index=False)
    print(f'\n[recording-level] {len(rec_df)} rows')

    comp_defs = {
        'pred_vs_true_minute_index': ('true_minute_apnea_index', 'pred_apnea_index'),
        'pred_vs_real_ahi': ('real_AHI', 'pred_apnea_index'),
        'true_minute_index_vs_real_ahi': ('real_AHI', 'true_minute_apnea_index'),
    }
    comparisons, per_seed_rows = {}, []
    for mt, g_mt in rec_df.groupby('model_type'):
        for key, (gt_col, pred_col) in comp_defs.items():
            per_seed = []
            for seed, g in g_mt.groupby('seed'):
                m = regression_metrics(g[gt_col], g[pred_col])
                m.update({'model_type': mt, 'comparison': key, 'seed': seed})
                per_seed.append(m)
                per_seed_rows.append(m)
            pdf = pd.DataFrame(per_seed)
            comparisons[(mt, key)] = {
                'mean': pdf.mean(numeric_only=True).to_dict(),
                'std': pdf.std(numeric_only=True).to_dict(),
            }
            print(f'\n=== {mt} / {key} (over {len(pdf)} seed(s)) ===')
            print(pdf[['MAE', 'RMSE', 'pearson_r', 'spearman_rho', 'R2',
                       'bland_altman_bias']].mean().to_string())
    pd.DataFrame(per_seed_rows).to_csv(
        os.path.join(args.output_dir, 'ahi_regression_metrics_per_seed.csv'), index=False)

    cls_results = {}
    thresholds = [float(t) for t in args.thresholds.split(',') if t.strip()]
    for mt, g_mt in rec_df.groupby('model_type'):
        for thr in thresholds:
            per_seed = [binary_class_metrics(g['pred_apnea_index'].to_numpy(),
                                             g['real_AHI'].to_numpy(), thr)
                        for _, g in g_mt.groupby('seed')]
            pdf = pd.DataFrame(per_seed)
            cls_results[(mt, thr)] = {'mean': pdf.mean(numeric_only=True).to_dict(),
                                      'std': pdf.std(numeric_only=True).to_dict()}

    if not args.no_plots:
        for mt, g_mt in rec_df.groupby('model_type'):
            plot_seed = sorted(g_mt['seed'].unique())[0]
            plot_df = g_mt[g_mt['seed'] == plot_seed]
            for key, (gt_col, pred_col) in comp_defs.items():
                make_plots(plot_df, gt_col, pred_col, 'record_id',
                           title=f'{key} (model={mt}, seed={plot_seed})',
                           out_prefix=f'{mt}_{key}', out_dir=args.output_dir)

    severity_counts = gt_table['severity'].value_counts().to_dict()
    for c in ['normal', 'mild', 'moderate', 'severe']:
        severity_counts.setdefault(c, 0)

    report = write_report(args.output_dir, gt_table, discrepancies, problems, seg_metrics,
                          comparisons, cls_results, severity_counts,
                          sorted(seg_df['source_csv'].unique()))
    print(f'\n[done] {report}')


if __name__ == '__main__':
    main()
