#!/usr/bin/env python3
"""True-multiprocess Optuna hyperparameter sweep for ClusTree / LiarTree.

`optuna.Study.optimize(n_jobs=...)` parallelizes trials with *threads*, which
doesn't help here: each trial streams a whole dataset through a pure-Python
tree (CPU-bound), so the GIL serializes them anyway. This script instead
farms trial-chunks out to a joblib "loky" process pool -- real OS processes,
no GIL sharing -- and has them all append to the same Optuna study through a
JournalStorage file, so parallel workers really do explore the search space
concurrently.

It sweeps every (algorithm x dataset) combination you ask for, splitting
every study's trials into chunks and flattening all combos' chunks into one
task list, so all worker processes stay busy regardless of how many datasets
or algorithms you passed.

Examples
--------
# Sweep both algorithms over two datasets, using all cores
python param_sweep.py --datasets complex9 stream_densired \\
    --algorithms liartree clustree --n-trials 60

# Just ClusTree on one dataset, capped at 4 worker processes
python param_sweep.py --datasets complex9 --algorithms clustree \\
    --n-trials 40 --n-jobs 4

# Add more trials to a previous sweep's studies instead of overwriting them
python param_sweep.py --datasets complex9 --algorithms liartree \\
    --n-trials 20 --resume

# Sweep every dataset found under baseframework/data/
python param_sweep.py --datasets all --algorithms liartree clustree --n-trials 60
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import optuna
from joblib import Parallel, delayed
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Search spaces ────────────────────────────────────────────────────────
# name -> (kind, low, high, log). lam/noise_threshold are tree constructor
# kwargs; eps/min_samples are DBSCAN offline-phase kwargs (see
# TREE_PARAM_NAMES below for the split).
SEARCH_SPACES = {
    "liartree": {
        "lam": ("float", 1e-5, 4.0, True),          # ClusTree Sec. 3.2 decay rate
        "noise_threshold": ("float", 0.0, 1.5, False),  # LiarTree Def. 2
        "eps": ("float", 0.005, 0.5, False),
        "min_samples": ("int", 1, 20, False),
    },
    "clustree": {
        "lam": ("float", 1e-5, 4.0, True),
        "eps": ("float", 0.005, 0.5, False),
        "min_samples": ("int", 1, 20, False),
    },
}
TREE_PARAM_NAMES = {"lam", "noise_threshold"}

# Structural params fixed per the paper / repo CLI defaults -- see
# `param sweep/*_param_sweep.ipynb` for the paper-provenance table. Only the
# params in SEARCH_SPACES above are actually tuned.
BASE_TREE_KWARGS = {
    "liartree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                     budget=10**9, max_height=0, horizon=1000,
                     overlap_reduction=True),
    "clustree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                     budget=10**9, max_height=0, horizon=1000),
}

OFFLINE_METHOD = "dbscan"
CMM_K = 5
SIZE_PENALTY_WEIGHT = 0.3   # see objective() docstring
DEFAULT_MAX_CMM_N = 2000    # CMM/F1/purity subsample cap for large datasets


# ── Metrics helpers (run inside worker processes) ───────────────────────

def _compute_purity(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for label in set(y_pred.tolist()):
        if label == -1:
            continue
        mask = y_pred == label
        if mask.any():
            correct += Counter(y_true[mask].tolist()).most_common(1)[0][1]
    return correct / len(y_true)


def _compute_f1(gt, clu, X):
    """Macro F1 using CMM's own ground-truth-cluster mapping, so F1 and CMM
    score the same predicted-to-true correspondence."""
    from sklearn.metrics import f1_score
    from baseframework.metrics.cmm import (cluster_mapping, get_clu_clusters,
                                           get_gt_clusters)
    gt_dist = get_gt_clusters(gt, X)
    clu_dist = get_clu_clusters(clu, gt)
    mapping = cluster_mapping(gt_dist, clu_dist, clu, gt)
    mapped = np.array([mapping.get(c, -1) for c in clu])
    return f1_score(gt, mapped, average="macro", zero_division=0)


def _nearest_summary_labels(X, summaries, labels_by_id):
    if not summaries:
        return np.full(len(X), -1)
    ids = list(summaries.keys())
    centers = np.array([summaries[mid]["position"] for mid in ids])
    label_arr = np.array([labels_by_id.get(mid, -1) for mid in ids])
    diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    return label_arr[np.argmin(dists, axis=1)]


def _load_dataset_cmm(dataset, max_cmm_n, cmm_seed=42):
    """Load a dataset and take a deterministic CMM/F1/purity subsample.

    Every worker (and every chunk of the same study) calls this itself from
    inside its own process instead of receiving the arrays over IPC -- the
    fixed rng seed makes the subsample identical across processes, and
    reloading these small benchmark datasets from disk is cheap.
    """
    from src.stream_runner import load_dataset
    X_full, y_full = load_dataset(dataset, scale=True)
    true_k = len(np.unique(y_full))

    rng = np.random.default_rng(cmm_seed)
    n = len(X_full)
    cap = n if max_cmm_n is None else min(max_cmm_n, n)
    if cap < n:
        idx = np.sort(rng.choice(n, size=cap, replace=False))
    else:
        idx = np.arange(n)
    X_cmm, y_cmm = X_full[idx], y_full[idx]
    w_cmm = np.ones(len(idx))
    return X_full, y_full, X_cmm, y_cmm, w_cmm, true_k


def _tree_class(algorithm):
    from src.clustree import ClusTree
    from src.liartree import LiarTree
    return {"liartree": LiarTree, "clustree": ClusTree}[algorithm]


def _make_objective(algorithm, X_full, X_cmm, y_cmm, w_cmm, true_k, seed):
    """Composite objective: CMM/F1 are 0-1 higher-is-better metrics, so a
    plain average is meaningful. Cluster count is scored as *closeness to
    the true number of classes* rather than maximized/minimized directly --
    a trivial near-singleton clustering can score a perfect CMM despite being
    meaningless, so it's penalized instead of left to game the objective.
    """
    from baseframework.metrics.cmm import get_cmm
    from src.offline import OfflineClusterer

    tree_cls = _tree_class(algorithm)
    base_kwargs = BASE_TREE_KWARGS[algorithm]
    space = SEARCH_SPACES[algorithm]

    def objective(trial):
        sampled = {}
        for name, (kind, low, high, log) in space.items():
            if kind == "float":
                sampled[name] = trial.suggest_float(name, low, high, log=log)
            else:
                sampled[name] = trial.suggest_int(name, int(low), int(high))

        tree_kwargs = dict(base_kwargs)
        tree_kwargs.update({k: v for k, v in sampled.items() if k in TREE_PARAM_NAMES})

        tree = tree_cls(**tree_kwargs)
        for i in range(len(X_full)):
            tree.learn_one(X_full[i], t=i + 1)
        summaries = tree.get_summary()

        # weighted=False: decayed CF weights are all << 1 at these lam/horizon
        # scales, so weighting would starve DBSCAN's min_samples check
        # regardless of eps.
        offline = OfflineClusterer(
            method=OFFLINE_METHOD,
            eps=sampled["eps"], min_samples=sampled["min_samples"],
            weighted=False, random_state=seed,
        )
        labels_by_id = offline.predict_summaries(summaries)
        pred = _nearest_summary_labels(X_cmm, summaries, labels_by_id)

        n_clusters = len(set(pred.tolist()) - {-1})
        if n_clusters < 2:
            raise optuna.TrialPruned()

        cmm = float(get_cmm(X_cmm, y_cmm, pred, w_cmm, CMM_K))
        f1 = float(_compute_f1(y_cmm, pred, X_cmm))
        purity = _compute_purity(y_cmm, pred)

        quality = f1
        size_penalty = abs(n_clusters - true_k) / true_k

        trial.set_user_attr("cmm", cmm)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("purity", purity)
        trial.set_user_attr("n_clusters", n_clusters)
        return quality - SIZE_PENALTY_WEIGHT * size_penalty

    return objective


# ── Study bookkeeping ────────────────────────────────────────────────────

def _storage_path(output_dir, algorithm, dataset):
    return os.path.join(output_dir, f"optuna_journal_{algorithm}_{dataset}.log")


def _study_name(algorithm, dataset):
    return f"{algorithm}_{dataset}"


def _open_storage(output_dir, algorithm, dataset):
    return JournalStorage(JournalFileBackend(_storage_path(output_dir, algorithm, dataset)))


def _run_chunk(algorithm, dataset, output_dir, n_trials_chunk, max_cmm_n, seed):
    """Entry point executed inside a worker process. Only plain strings/ints
    cross the process boundary; data is (re)loaded locally per worker."""
    sys.path.insert(0, PROJECT_ROOT)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=_study_name(algorithm, dataset),
                              storage=_open_storage(output_dir, algorithm, dataset))
    X_full, _y_full, X_cmm, y_cmm, w_cmm, true_k = _load_dataset_cmm(dataset, max_cmm_n)
    objective = _make_objective(algorithm, X_full, X_cmm, y_cmm, w_cmm, true_k, seed)
    # catch=(Exception,): a bug triggered by one (algorithm, dataset, params)
    # combo -- e.g. the high-dimensional-data OverflowError HAR used to hit --
    # should only fail that trial, not take down this worker process and the
    # rest of the sweep with it (Optuna's default is to propagate and stop).
    study.optimize(objective, n_trials=n_trials_chunk, catch=(Exception,))


def _chunk_sizes(n_trials, n_chunks):
    base, extra = divmod(n_trials, n_chunks)
    sizes = [base + (1 if i < extra else 0) for i in range(n_chunks)]
    return [s for s in sizes if s > 0]


# ── Orchestration ────────────────────────────────────────────────────────

def run_sweep(datasets, algorithms, n_trials, n_jobs, output_dir,
              seed=0, resume=False, max_cmm_n=DEFAULT_MAX_CMM_N):
    os.makedirs(output_dir, exist_ok=True)
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    combos = [(algo, ds) for algo in algorithms for ds in datasets]

    # Create (or reattach to) each study up front and sequentially in the
    # main process -- avoids every worker racing to create the same journal
    # file the moment the pool starts.
    for algo, ds in combos:
        path = _storage_path(output_dir, algo, ds)
        if not resume and os.path.exists(path):
            os.remove(path)
        optuna.create_study(
            study_name=_study_name(algo, ds), storage=_open_storage(output_dir, algo, ds),
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
            load_if_exists=resume,
        )

    # Flatten every (algorithm, dataset) combo's trial-chunks into one task
    # list so joblib's loky pool keeps all n_jobs processes busy regardless
    # of how many combos or trials each one has.
    tasks = []
    for algo, ds in combos:
        n_chunks = min(n_trials, max(1, n_jobs))
        for size in _chunk_sizes(n_trials, n_chunks):
            tasks.append((algo, ds, size))

    print(f"Dispatching {len(tasks)} trial-chunks across {n_jobs} worker processes "
          f"({len(combos)} dataset x algorithm combo(s), {n_trials} trials each)")
    t0 = time.time()
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_chunk)(algo, ds, output_dir, size, max_cmm_n, seed)
        for algo, ds, size in tasks
    )
    print(f"Sweep finished in {time.time() - t0:.1f}s")

    return _collect_results(combos, output_dir)


def _collect_results(combos, output_dir):
    import pandas as pd

    summary_rows = []
    for algo, ds in combos:
        study = optuna.load_study(study_name=_study_name(algo, ds),
                                  storage=_open_storage(output_dir, algo, ds))

        df = study.trials_dataframe()
        df.insert(0, "dataset", ds)
        df.insert(0, "algorithm", algo)
        csv_path = os.path.join(output_dir, f"{algo}_{ds}_trials.csv")
        df.to_csv(csv_path, index=False)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\n[{algo} / {ds}] {len(completed)}/{len(study.trials)} valid trials -> {csv_path}")
        if completed:
            bt = study.best_trial
            print(f"  best score={study.best_value:.4f}  "
                  f"cmm={bt.user_attrs['cmm']:.4f}  f1={bt.user_attrs['f1']:.4f}  "
                  f"purity={bt.user_attrs['purity']:.4f}  clusters={bt.user_attrs['n_clusters']}")
            print(f"  best params: {study.best_params}")
            summary_rows.append(dict(
                algorithm=algo, dataset=ds, score=study.best_value,
                cmm=bt.user_attrs["cmm"], f1=bt.user_attrs["f1"],
                purity=bt.user_attrs["purity"], n_clusters=bt.user_attrs["n_clusters"],
                **{f"param_{k}": v for k, v in study.best_params.items()},
            ))
        else:
            print("  no valid configuration found in any trial")

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary_path = os.path.join(output_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary -> {summary_path}")
    return summary


def _discover_datasets():
    """Every dataset name available under baseframework/data/ -- the directory
    datahandler.load_data actually reads from -- as <name> for each
    <name>_data.npy that has a matching <name>_label.npy alongside it."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    names = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith("_data.npy"):
            name = fname[: -len("_data.npy")]
            if os.path.exists(os.path.join(data_dir, f"{name}_label.npy")):
                names.append(name)
    return names


def _expand_datasets(datasets):
    if any(d.lower() == "all" for d in datasets):
        discovered = _discover_datasets()
        if not discovered:
            raise SystemExit("--datasets all: no <name>_data.npy/<name>_label.npy "
                             "pairs found under baseframework/data/")
        return discovered
    return datasets


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="dataset names understood by baseframework.datahandler "
                        "(e.g. complex9 stream_densired rbf3 har, or any "
                        "<name>_data.npy/<name>_label.npy pair under "
                        "baseframework/data/), or 'all' to sweep every dataset "
                        "found there")
    p.add_argument("--algorithms", nargs="+", default=["liartree", "clustree"],
                   choices=sorted(SEARCH_SPACES.keys()),
                   help="algorithms to sweep (default: both)")
    p.add_argument("--n-trials", type=int, default=80,
                   help="Optuna trials per (algorithm, dataset) combo")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="worker processes for the loky pool; -1 = all cores")
    p.add_argument("--max-cmm-n", type=int, default=DEFAULT_MAX_CMM_N,
                   help="subsample cap for the CMM/F1/purity metrics "
                        "(full dataset streams regardless); use 0 for no cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="param_sweep_results",
                   help="where journal storage files, per-combo trial CSVs, "
                        "and summary.csv are written")
    p.add_argument("--resume", action="store_true",
                   help="add trials to existing studies in --output-dir "
                        "instead of overwriting them")
    return p


def main():
    args = build_arg_parser().parse_args()
    max_cmm_n = None if args.max_cmm_n == 0 else args.max_cmm_n
    datasets = _expand_datasets(args.datasets)
    if datasets != args.datasets:
        print(f"--datasets all -> {len(datasets)} datasets: {datasets}")
    run_sweep(datasets, args.algorithms, args.n_trials, args.n_jobs,
             args.output_dir, seed=args.seed, resume=args.resume, max_cmm_n=max_cmm_n)


if __name__ == "__main__":
    main()
