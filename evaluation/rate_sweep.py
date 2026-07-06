#!/usr/bin/env python3
"""Sinusoidal-rate anytime sweep for ClusTree / LiarTree.

Tests the anytime mechanism's ability to track a data stream whose arrival
rate oscillates sinusoidally between dense and sparse (instead of arriving at
a constant rate). The budget parameter (max distance computations per
insertion -- see src/clustree.py) is *not* changed in the source: it is
simply reassigned before every learn_one() call, since learn_one already
re-reads self.budget fresh each time. High instantaneous rate -> less time
available per point -> lower budget; low rate -> higher budget.

    rate(i)   = 1 + magnitude * sin(2*pi*frequency*i/n)
    budget(i) = clip(round(base_budget / rate(i)), budget_min, budget_max)

frequency is expressed in cycles over the whole stream (not points per
cycle), so it is comparable across datasets of different lengths.

Two sweep axes:
  --axis magnitude   fixed --frequency, list of --magnitudes
  --axis frequency   fixed --magnitude, list of --frequencies

lam/eps/min_samples/noise_threshold default to the values already tuned by
param_sweep.py (see param_sweep_results_default/summary.csv) per
(algorithm, dataset), so the sweep isolates the effect of the rate schedule.
Pass --lam/--eps/--min-samples/--noise-threshold/etc. to override them
uniformly across whichever combos are run.

Examples
--------
# Fixed frequency (3 cycles over the stream), vary the oscillation magnitude
python rate_sweep.py --datasets complex9 stream_densired \\
    --algorithms liartree clustree --axis magnitude --frequency 3 \\
    --magnitudes 0.0 0.2 0.4 0.6 0.8 0.95

# Fixed magnitude, vary the oscillation frequency
python rate_sweep.py --datasets complex9 stream_densired \\
    --algorithms liartree clustree --axis frequency --magnitude 0.6 \\
    --frequencies 0.5 1 2 5 10 20 50

# Override a tuned hyperparameter for every combo in the run
python rate_sweep.py --datasets complex9 --algorithms clustree \\
    --axis magnitude --magnitudes 0.0 0.5 --lam 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Tuned defaults (param_sweep_results_default/summary.csv) ───────────────
# Only the two datasets requested; add more rows here (or pass explicit CLI
# overrides) to cover others.
CLUSTREE_DEFAULTS = {
    "complex9": dict(lam=0.000055, eps=0.035953, min_samples=15),
    "stream_densired": dict(lam=0.004300, eps=0.014404, min_samples=8),
}
LIARTREE_DEFAULTS = {
    "complex9": dict(lam=0.000495, noise_threshold=1.062794,
                     eps=0.033401, min_samples=12),
    "stream_densired": dict(lam=2.466295	, noise_threshold=1.108716,
                            eps=0.009972	, min_samples=1),
}
# One shared budget per dataset, used by both algorithms -- budget is meant
# to be an algorithm-agnostic unit (see the "budget" docstring in
# src/clustree.py: "The LiarTree uses the same unit, so the two stay
# comparable"), so ClusTree and LiarTree must be driven by the identical
# schedule to make any cross-algorithm comparison meaningful. Full-depth
# insertion under these tuned hyperparameters only ever costs ~12-27 distance
# computations on these (2D, max_fanout=3) datasets -- see the calibration
# probe in the PR/session notes. The run_clustree.py CLI convention defaults
# (36/72) sit above that natural ceiling, so budget would never actually bind
# and the rate schedule would have no effect. These are picked near the
# measured median cost so the sinusoid's swing crosses both the "buffered
# above a leaf" and "reaches a leaf" regimes.
DEFAULT_BASE_BUDGET = {
    "complex9": 18,
    "stream_densired": 21,
}

STRUCT_PARAM_NAMES = {"max_fanout", "leaf_capacity_max", "descent", "horizon",
                       "max_height", "overlap_reduction"}
TREE_PARAM_NAMES = {"lam", "noise_threshold"} | STRUCT_PARAM_NAMES
OFFLINE_PARAM_NAMES = {"eps", "min_samples"}

CMM_K = 5
DEFAULT_MAX_CMM_N = 2000


# ── Budget schedule ──────────────────────────────────────────────────────

def sinusoidal_budget_schedule(n, base_budget, magnitude, frequency,
                                budget_min=1, budget_max=None):
    """Per-point (budget, rate) arrays of length n for a sinusoidal arrival rate.

    magnitude in [0, 1): amplitude of the rate oscillation as a fraction of the
    baseline rate. frequency: cycles over the whole stream (i=0..n-1).
    """
    if budget_max is None:
        budget_max = base_budget * 5
    i = np.arange(n)
    rate = 1.0 + magnitude * np.sin(2 * np.pi * frequency * i / n)
    rate = np.clip(rate, 1e-3, None)   # guard against divide-by-zero near troughs
    budget = np.round(base_budget / rate).astype(int)
    budget = np.clip(budget, budget_min, budget_max)
    return budget, rate


# ── Per-run worker (executed inside a joblib process) ───────────────────

def _tree_class(algorithm):
    from src.clustree import ClusTree
    from src.liartree import LiarTree
    return {"liartree": LiarTree, "clustree": ClusTree}[algorithm]


def _tuned_defaults(algorithm, dataset):
    table = LIARTREE_DEFAULTS if algorithm == "liartree" else CLUSTREE_DEFAULTS
    if dataset not in table:
        raise SystemExit(
            f"no tuned defaults for {algorithm}/{dataset}; pass explicit "
            f"--lam/--eps/--min-samples (and --noise-threshold for liartree)")
    return dict(table[dataset])


def _resolve_base_budget(dataset, override):
    if override is not None:
        return override
    if dataset not in DEFAULT_BASE_BUDGET:
        raise SystemExit(
            f"no calibrated base-budget default for {dataset}; pass "
            f"--base-budget explicitly")
    return DEFAULT_BASE_BUDGET[dataset]


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return None


def _snapshot(tree, offline, X, y, eval_idx_full, n_seen, budget, rate):
    from param_sweep import _compute_purity, _compute_f1, _nearest_summary_labels
    from baseframework.metrics.cmm import get_cmm

    summaries = tree.get_summary()
    row = dict(step=n_seen, budget=int(budget), rate=float(rate),
               n_microclusters=len(summaries))
    idx = eval_idx_full[eval_idx_full < n_seen]
    if not summaries or len(idx) == 0:
        row.update(n_clusters=0, cmm=None, f1=None, purity=None)
        return row

    labels_by_id = offline.predict_summaries(summaries)
    X_eval, y_eval = X[idx], y[idx]
    pred = _nearest_summary_labels(X_eval, summaries, labels_by_id)
    n_clusters = len(set(pred.tolist()) - {-1})
    row["n_clusters"] = n_clusters
    if n_clusters < 1:
        row.update(cmm=None, f1=None, purity=None)
        return row

    w_eval = np.ones(len(idx))
    row["cmm"] = _safe(lambda: get_cmm(X_eval, y_eval, pred, w_eval, CMM_K))
    row["f1"] = _safe(lambda: _compute_f1(y_eval, pred, X_eval))
    row["purity"] = _safe(lambda: _compute_purity(y_eval, pred))
    return row


def _run_one(algorithm, dataset, axis, value, frequency_fixed, magnitude_fixed,
             base_budget, overrides, snapshot_every_frac, max_cmm_n,
             budget_min, budget_max_multiplier, seed):
    """Entry point executed inside a worker process."""
    sys.path.insert(0, PROJECT_ROOT)
    import pandas as pd
    from src.stream_runner import load_dataset
    from src.offline import OfflineClusterer

    X, y = load_dataset(dataset, scale=True)
    n = len(X)

    magnitude = value if axis == "magnitude" else magnitude_fixed
    frequency = frequency_fixed if axis == "magnitude" else value
    budget_max = base_budget * budget_max_multiplier
    schedule, rate = sinusoidal_budget_schedule(
        n, base_budget, magnitude, frequency, budget_min, budget_max)

    merged = _tuned_defaults(algorithm, dataset)
    merged.update({k: v for k, v in overrides.items() if v is not None})

    tree_kwargs = dict(
        max_fanout=merged.get("max_fanout", 3),
        leaf_capacity_max=merged.get("leaf_capacity_max", 3),
        descent=merged.get("descent", "depthfirst"),
        lam=merged["lam"], horizon=merged.get("horizon", 1000),
        max_height=merged.get("max_height", 0), budget=int(base_budget),
    )
    if algorithm == "liartree":
        tree_kwargs["noise_threshold"] = merged["noise_threshold"]
        tree_kwargs["overlap_reduction"] = merged.get("overlap_reduction", True)

    tree = _tree_class(algorithm)(**tree_kwargs)
    offline = OfflineClusterer(method="dbscan", eps=merged["eps"],
                               min_samples=merged["min_samples"],
                               weighted=False, random_state=seed)

    rng = np.random.default_rng(seed)
    cap = n if not max_cmm_n else min(max_cmm_n, n)
    eval_idx_full = (np.sort(rng.choice(n, size=cap, replace=False))
                     if cap < n else np.arange(n))

    snapshot_every = max(1, int(round(n * snapshot_every_frac)))
    rows = []
    for i in range(n):
        tree.budget = int(schedule[i])
        tree.learn_one(X[i], t=i + 1)
        n_seen = i + 1
        if n_seen % snapshot_every == 0 or n_seen == n:
            rows.append(_snapshot(tree, offline, X, y, eval_idx_full,
                                  n_seen, schedule[i], rate[i]))

    ts_df = pd.DataFrame(rows)
    last = rows[-1]
    summary_row = dict(
        algorithm=algorithm, dataset=dataset, axis=axis, value=value,
        frequency=frequency, magnitude=magnitude, base_budget=base_budget,
        final_n_microclusters=last["n_microclusters"], final_n_clusters=last["n_clusters"],
        final_cmm=last["cmm"], final_f1=last["f1"], final_purity=last["purity"],
        **{f"param_{k}": v for k, v in merged.items()},
    )
    return ts_df, summary_row


# ── Orchestration ────────────────────────────────────────────────────────

def run_rate_sweep(datasets, algorithms, axis, frequency, magnitudes, magnitude,
                    frequencies, base_budget_overrides, overrides, snapshot_every_frac,
                    max_cmm_n, budget_min, budget_max_multiplier, seed, n_jobs,
                    output_dir):
    from joblib import Parallel, delayed
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    values = magnitudes if axis == "magnitude" else frequencies
    tasks = [(algo, ds, v) for algo in algorithms for ds in datasets for v in values]
    base_budgets = {(algo, ds): _resolve_base_budget(algo, ds, base_budget_overrides[algo])
                    for algo in algorithms for ds in datasets}

    print(f"Dispatching {len(tasks)} runs across {n_jobs} worker processes "
          f"(axis={axis}, {len(algorithms)} algorithm(s), {len(datasets)} dataset(s), "
          f"{len(values)} value(s))")
    print("base budgets: " + ", ".join(f"{a}/{d}={b}" for (a, d), b in base_budgets.items()))
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one)(algo, ds, axis, v, frequency, magnitude, base_budgets[(algo, ds)],
                          overrides, snapshot_every_frac, max_cmm_n, budget_min,
                          budget_max_multiplier, seed)
        for algo, ds, v in tasks
    )
    print(f"Sweep finished in {time.time() - t0:.1f}s")

    summary_rows = []
    for (algo, ds, v), (ts_df, summary_row) in zip(tasks, results):
        fname = f"{algo}_{ds}_{axis}_{v}_timeseries.csv"
        ts_df.to_csv(os.path.join(output_dir, fname), index=False)
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nPer-run time series -> {output_dir}/<algo>_<dataset>_{axis}_<value>_timeseries.csv")
    print(f"Summary -> {summary_path}")
    return summary


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["complex9", "stream_densired"])
    p.add_argument("--algorithms", nargs="+", default=["liartree", "clustree"],
                   choices=["liartree", "clustree"])
    p.add_argument("--axis", required=True, choices=["magnitude", "frequency"],
                   help="which parameter to sweep over a list; the other is held fixed")

    p.add_argument("--frequency", type=float, default=3.0,
                   help="fixed frequency (cycles over the whole stream) when "
                        "--axis magnitude")
    p.add_argument("--magnitudes", type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
                   help="magnitude values to sweep when --axis magnitude")

    p.add_argument("--magnitude", type=float, default=0.6,
                   help="fixed magnitude (fraction of base rate, in [0,1)) when "
                        "--axis frequency")
    p.add_argument("--frequencies", type=float, nargs="+",
                   default=[0.5, 1, 2, 5, 10, 20, 50],
                   help="frequency values (cycles over the whole stream) to sweep "
                        "when --axis frequency")

    p.add_argument("--clustree-base-budget", type=int, default=None,
                   help="override the calibrated clustree base budget for every "
                        "dataset run (default: per-dataset value in "
                        "DEFAULT_BASE_BUDGET, chosen near that dataset's natural "
                        "full-depth insertion cost)")
    p.add_argument("--liartree-base-budget", type=int, default=None,
                   help="override the calibrated liartree base budget for every "
                        "dataset run (see --clustree-base-budget)")
    p.add_argument("--budget-min", type=int, default=1)
    p.add_argument("--budget-max-multiplier", type=float, default=5.0,
                   help="hard cap on budget as a multiple of base-budget, so a "
                        "near-zero-rate trough can't make an insertion unboundedly slow")

    p.add_argument("--snapshot-every-frac", type=float, default=0.02,
                   help="fraction of the stream between quality snapshots "
                        "(0.02 = 50 snapshots)")
    p.add_argument("--max-cmm-n", type=int, default=DEFAULT_MAX_CMM_N,
                   help="subsample cap for the CMM/F1/purity metrics; 0 = no cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--output-dir", default="rate_sweep_results")

    # Hyperparameter overrides: None means "use the tuned default for that
    # (algorithm, dataset)"; any value passed here applies to every combo run.
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--noise-threshold", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--max-fanout", type=int, default=None)
    p.add_argument("--leaf-max", type=int, default=None, dest="leaf_capacity_max")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--max-height", type=int, default=None)
    p.add_argument("--descent", default=None,
                   choices=["depthfirst", "breadthfirst", "bestfirst", "iterativedf"])
    p.add_argument("--no-overlap-reduction", dest="overlap_reduction",
                   action="store_const", const=False, default=None,
                   help="liartree only: disable overlap reduction")
    return p


def main():
    args = build_arg_parser().parse_args()
    overrides = dict(
        lam=args.lam, noise_threshold=args.noise_threshold, eps=args.eps,
        min_samples=args.min_samples, max_fanout=args.max_fanout,
        leaf_capacity_max=args.leaf_capacity_max, horizon=args.horizon,
        max_height=args.max_height, descent=args.descent,
        overlap_reduction=args.overlap_reduction,
    )
    base_budget_overrides = {"clustree": args.clustree_base_budget,
                             "liartree": args.liartree_base_budget}

    run_rate_sweep(
        args.datasets, args.algorithms, args.axis, args.frequency, args.magnitudes,
        args.magnitude, args.frequencies, base_budget_overrides, overrides,
        args.snapshot_every_frac, args.max_cmm_n, args.budget_min,
        args.budget_max_multiplier, args.seed, args.n_jobs, args.output_dir,
    )


if __name__ == "__main__":
    main()
