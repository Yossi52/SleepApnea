"""Error-complementarity analysis: do different transformations fail on different segments?

Read-only. Consumes the per-segment predictions already stored in the test-result JSONs
written by `experiments/train_10fold.py` / `train_subject_wise.py` -- no training, no
inference.

Three questions:
  (A) pairwise complementarity between models: disagreement, double fault (both wrong),
      Yule's Q, correlation rho, and the 2-model oracle;
  (B) oracle bounds over the branch set the fusion models actually use: best single branch,
      3-model oracle, majority vote, soft vote, and the measured fusion accuracies;
  (C) segment difficulty: how many of the models get each segment wrong.

Every model is given as `--pred NAME=GLOB`. Files are grouped by the `rs<seed>` and
optional `fold<n>` tokens in their filename, so a model's folds are concatenated per seed
and the results are reported as mean +- sd across seeds.

Usage:
    python analysis/error_complementarity.py \
        --pred rp=results/test_results/*rp*_rs*_fold*.json \
        --pred gadf=results/test_results/*gadf*_rs*_fold*.json \
        --pred scalogram=results/test_results/*scalogram*_rs*_fold*.json \
        --pred gasf=... --pred mtf=... --pred spectrogram=... \
        --pred early_fusion=... --pred late_fusion=... --pred hybrid_fusion=... \
        --branches rp,gadf,scalogram \
        --out_dir results/error_complementarity
"""

import argparse
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

# Models used for the pairwise / difficulty analysis; the fusion models below are only
# reported as reference accuracies in the oracle table.
FUSION_MODELS = ('early_fusion', 'late_fusion', 'hybrid_fusion')

GROUP_RE = re.compile(r'rs(?P<seed>\d+)(?:.*?fold(?P<fold>\d+))?')

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def load(path):
    """Test-result JSON -> (y_true, y_pred, y_prob). Accepts both key spellings."""
    with open(path) as fh:
        j = json.load(fh)
    p = j.get('test_results', j)
    p = p.get('predictions', p)
    true = p.get('true_labels', p.get('true_label'))
    pred = p.get('pred_labels', p.get('predicted_labels'))
    prob = p.get('pred_probs', p.get('prediction_probabilities'))
    if true is None or pred is None:
        raise ValueError(f'{path}: no true/pred labels found')
    return (np.asarray(true, dtype=int), np.asarray(pred, dtype=int),
            np.asarray(prob, dtype=float))


def group_key(path):
    """(seed, fold) parsed from the filename; fold is None for single-split runs."""
    m = GROUP_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f'{path}: filename has no rs<seed> token')
    return int(m['seed']), (int(m['fold']) if m['fold'] else None)


def collect(specs):
    """{NAME: GLOB} -> {(seed, fold): {model: path}}"""
    groups = defaultdict(dict)
    for model, pattern in specs.items():
        hits = sorted(glob.glob(pattern))
        if not hits:
            raise FileNotFoundError(f'{model}: no file matched {pattern}')
        for path in hits:
            key = group_key(path)
            if model in groups[key]:
                raise ValueError(f'{model}: two files map to seed/fold {key}: '
                                 f'{groups[key][model]} and {path}')
            groups[key][model] = path
        log(f'[collect] {model}: {len(hits)} file(s)')
    return groups


def verify_and_stack(groups, models):
    """Check that the true labels agree inside each (seed, fold), then concatenate folds."""
    seeds = sorted({s for s, _ in groups})
    folds = sorted({f for _, f in groups}, key=lambda f: (f is None, f))
    ok = bad = 0
    per_seed = {}

    for seed in seeds:
        seed_data = defaultdict(list)
        seed_ok = True
        for fold in folds:
            g = groups.get((seed, fold))
            if not g:
                log(f'  !! MISSING group seed={seed} fold={fold}')
                bad += 1
                seed_ok = False
                continue
            missing = [m for m in models if m not in g]
            if missing:
                log(f'  !! seed={seed} fold={fold} missing models: {missing}')
                bad += 1
                seed_ok = False
                continue
            data = {m: load(p) for m, p in g.items()}
            ref = data[models[0]][0]
            mism = [m for m, (t, _, _) in data.items()
                    if t.shape != ref.shape or not np.array_equal(t, ref)]
            if any(m in models for m in mism):
                log(f'  !! TRUE-LABEL MISMATCH among the analysed models, seed={seed} '
                    f'fold={fold} -> {mism}; GROUP EXCLUDED')
                bad += 1
                seed_ok = False
                continue
            if mism:
                # Only reference models (fusion) disagree -> drop those, keep the group.
                log(f'  !! TRUE-LABEL MISMATCH seed={seed} fold={fold} -> {mism} '
                    f'(n={[len(data[m][0]) for m in mism]} vs ref n={len(ref)}); '
                    f'those models excluded, group kept')
                for m in mism:
                    data.pop(m)
            ok += 1
            for m, v in data.items():
                seed_data[m].append(v)
        if not seed_ok:
            continue
        merged = {}
        for m, parts in seed_data.items():
            if len(parts) != len(folds):
                log(f'  ~ seed={seed} model={m}: {len(parts)}/{len(folds)} folds present '
                    f'-> dropped')
                continue
            merged[m] = (np.concatenate([p[0] for p in parts]),
                         np.concatenate([p[1] for p in parts]),
                         np.concatenate([p[2] for p in parts]))
        per_seed[seed] = merged
    log(f'[verify] true-label alignment: {ok} groups passed, {bad} failed/incomplete')
    return per_seed


def pair_stats(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    n11 = int(np.sum(ca & cb)); n10 = int(np.sum(ca & ~cb))
    n01 = int(np.sum(~ca & cb)); n00 = int(np.sum(~ca & ~cb))
    n = n11 + n10 + n01 + n00
    ad, bc = n11 * n00, n01 * n10
    q = (ad - bc) / (ad + bc) if (ad + bc) else float("nan")
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    rho = (ad - bc) / np.sqrt(den) if den else float("nan")
    acc_a, acc_b = (n11 + n10) / n, (n11 + n01) / n
    oracle = (n11 + n10 + n01) / n
    return dict(n_total=n, n11=n11, n10=n10, n01=n01, n00=n00,
                acc_a=acc_a, acc_b=acc_b,
                disagreement=(n10 + n01) / n, double_fault=n00 / n,
                q_stat=q, rho=rho, oracle_2=oracle,
                oracle_gain=oracle - max(acc_a, acc_b))


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6f}"
    return str(v)


def write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(fmt(r.get(h)) for h in header) + "\n")
    log(f"wrote {path} ({len(rows)} rows)")


def ms(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
    return f"{100 * v.mean():6.2f} +- {100 * sd:4.2f} (n={len(v)})" if len(v) else "n/a"


def summarize(models, branches, pair_sum, oracle_rows, hist_rows):
    log("\n=== SUMMARY ===")
    rows = sorted(pair_sum, key=lambda r: r["double_fault_mean"])
    log(f"\ndouble_fault, lowest 3 / highest 3 of the "
        f"{len(rows)} pairs (%)")
    for tag, sel in (("LOW ", rows[:3]), ("HIGH", rows[-3:])):
        for r in sel:
            log(f"  {tag} {r['model_a']:>10}-{r['model_b']:<10} "
                f"df={100 * r['double_fault_mean']:5.2f}+-{100 * r['double_fault_sd']:.2f}  "
                f"disag={100 * r['disagreement_mean']:5.2f}  Q={r['q_stat_mean']:.3f}  "
                f"oracle_gain={100 * r['oracle_gain_mean']:.2f}")

    log(f"\noracle bounds, branch_set={'+'.join(branches)} (%)")
    for k in ["best_single_of_3", "oracle_3", "majority_vote_3", "soft_vote_3",
              "early_fusion_acc", "late_fusion_acc", "hybrid_fusion_acc"]:
        log(f"  {k:<18} {ms([r[k] for r in oracle_rows])}")
    log(f"  ORACLE HEADROOM (oracle_3 - best_single_of_3) "
        f"{ms([r['oracle_3'] - r['best_single_of_3'] for r in oracle_rows])} pp")

    log("\nsegment difficulty (fraction of segments, apnea share)")
    for k in range(len(models) + 1):
        h = [r for r in hist_rows if r["n_models_wrong"] == k]
        log(f"  {k} models wrong: {ms([r['fraction'] for r in h])}   "
            f"apnea={ms([r['apnea_fraction'] for r in h])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', action='append', required=True, metavar='NAME=GLOB',
                    help='model name and glob of its test-result JSONs (repeatable). '
                         'Names in {early,late,hybrid}_fusion are treated as reference '
                         'fusion models and excluded from the pairwise analysis.')
    ap.add_argument('--branches', default='rp,gadf,scalogram',
                    help='the 3 branch models the fusion models were trained on')
    ap.add_argument('--out_dir', default='results/error_complementarity')
    args = ap.parse_args()

    specs = {}
    for item in args.pred:
        if '=' not in item:
            raise SystemExit(f'--pred expects NAME=GLOB, got {item!r}')
        name, pattern = item.split('=', 1)
        specs[name.strip()] = pattern.strip()

    branches = tuple(b.strip() for b in args.branches.split(','))
    models = [m for m in specs if m not in FUSION_MODELS]
    missing = [b for b in branches if b not in specs]
    if missing:
        raise SystemExit(f'--branches names models that were not given via --pred: {missing}')
    if len(models) < 2:
        raise SystemExit('need at least 2 non-fusion models')

    os.makedirs(args.out_dir, exist_ok=True)
    log("=== error complementarity run log ===")
    log(f"models: {models}")
    log(f"reference fusion models: {[m for m in specs if m in FUSION_MODELS]}")
    log(f"branch set for the oracle bounds: {branches}")
    log()

    per_seed = verify_and_stack(collect(specs), models)
    if not per_seed:
        raise SystemExit('no complete (seed, fold) group survived the checks')

    log("\naccuracy per model (%, mean over seeds):")
    for m in models + [f for f in FUSION_MODELS if f in specs]:
        accs = [100 * np.mean(d[m][1] == d[m][0]) for d in per_seed.values() if m in d]
        log(f"    {m:<14} n_seed={len(accs)}  {np.mean(accs):6.2f}" if accs
            else f"    {m:<14} n_seed=0")

    # (A) pairwise
    pair_raw, pair_sum = [], []
    for seed, d in sorted(per_seed.items()):
        for a, b in itertools.combinations(models, 2):
            pair_raw.append(dict(seed=seed, model_a=a, model_b=b,
                                 **pair_stats(d[a][0], d[a][1], d[b][1])))
    for a, b in itertools.combinations(models, 2):
        rows = [r for r in pair_raw if r["model_a"] == a and r["model_b"] == b]
        agg = dict(model_a=a, model_b=b, n_seeds=len(rows))
        for k in ["disagreement", "double_fault", "q_stat", "rho", "oracle_2", "oracle_gain"]:
            v = np.array([r[k] for r in rows], dtype=float)
            agg[k + "_mean"] = float(np.mean(v))
            agg[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        pair_sum.append(agg)

    # (B) oracle bounds over the branch set the fusion models use
    oracle_rows = []
    for seed, d in sorted(per_seed.items()):
        y = d[branches[0]][0]
        preds = np.stack([d[t][1] for t in branches])
        probs = np.stack([d[t][2] for t in branches])
        correct = preds == y
        row = dict(seed=seed, branch_set="+".join(branches),
                   best_single_of_3=float(correct.mean(axis=1).max()),
                   oracle_3=float(np.mean(correct.any(axis=0))),
                   majority_vote_3=float(np.mean((preds.sum(axis=0) >= 2).astype(int) == y)),
                   soft_vote_3=float(np.mean(probs.mean(axis=0).argmax(axis=1) == y)))
        for key in FUSION_MODELS:
            row[key + '_acc'] = float(np.mean(d[key][1] == d[key][0])) if key in d else None
        oracle_rows.append(row)

    # (C) difficulty histogram
    hist_rows = []
    for seed, d in sorted(per_seed.items()):
        y = d[models[0]][0]
        wrong = np.sum([d[m][1] != y for m in models], axis=0)
        for k in range(len(models) + 1):
            sel = wrong == k
            n = int(sel.sum())
            hist_rows.append(dict(seed=seed, n_models_wrong=k, n_segments=n,
                                  fraction=n / len(y),
                                  apnea_fraction=float(y[sel].mean()) if n else float("nan")))

    write_csv(os.path.join(args.out_dir, "pair_metrics_raw.csv"),
              ["seed", "model_a", "model_b", "n_total", "n11", "n10", "n01", "n00",
               "acc_a", "acc_b", "disagreement", "double_fault", "q_stat", "rho",
               "oracle_2", "oracle_gain"], pair_raw)
    write_csv(os.path.join(args.out_dir, "pair_metrics_summary.csv"),
              ["model_a", "model_b", "n_seeds"] +
              [k + s for k in ["disagreement", "double_fault", "q_stat", "rho",
                               "oracle_2", "oracle_gain"] for s in ["_mean", "_sd"]],
              pair_sum)
    write_csv(os.path.join(args.out_dir, "oracle_bounds.csv"),
              ["seed", "branch_set", "best_single_of_3", "oracle_3", "majority_vote_3",
               "soft_vote_3", "early_fusion_acc", "late_fusion_acc", "hybrid_fusion_acc"],
              oracle_rows)
    write_csv(os.path.join(args.out_dir, "difficulty_histogram.csv"),
              ["seed", "n_models_wrong", "n_segments", "fraction", "apnea_fraction"],
              hist_rows)

    summarize(models, branches, pair_sum, oracle_rows, hist_rows)

    with open(os.path.join(args.out_dir, "run_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
