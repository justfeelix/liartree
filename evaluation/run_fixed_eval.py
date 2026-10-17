#!/usr/bin/env python3
"""Re-run both algorithms with best-trial params from summary.csv using the FIXED code.

Outputs a JSON file with before/after results for comparison.
Run from liartree/liartree/:
    python evaluation/run_fixed_eval.py
"""
import sys, os, json, multiprocessing
from collections import Counter

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baseframework"))
sys.path.insert(0, os.path.join(ROOT, "baseframework", "data"))
sys.path.insert(0, os.path.join(ROOT, "baseframework", "metrics"))

from baseframework.metrics.cmm import get_cmm, cluster_mapping, get_clu_clusters, get_gt_clusters
from src.stream_runner import load_dataset
from src.offline import OfflineClusterer
from src.clustree import ClusTree
from src.liartree import LiarTree
from sklearn.metrics import f1_score

# ── config ───────────────────────────────────────────────────────────────
SUMMARY_CSV = os.path.join(ROOT, "evaluation",
    "param_sweep_param_disableable_noise_threshold_objective_cmm_f1",
    "summary.csv")

BASE_TREE_KWARGS = {
    "liartree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                     budget=10**9, max_height=0, horizon=1000,
                     overlap_reduction=True),
    "clustree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                     budget=10**9, max_height=0, horizon=1000),
}

CMM_K = 5
CMM_SEED = 42
MAX_CMM_N = 2000

TREE_CLASSES = {"liartree": LiarTree, "clustree": ClusTree}

# Only run the complex9 family (matches evaluation slides).
# stream_densired at 10-20k pts × unlimited budget is too slow for interactive use.
DATASETS_TO_RUN = [
    "complex9", "complex9_noise10", "complex9_noise20",
    "complex9_noise30", "complex9_noise40", "complex9_noise50",
    "stream_densired",
]


# ── helpers ──────────────────────────────────────────────────────────────

def compute_purity(y_true, y_pred):
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


def compute_f1(gt, clu, X):
    gt_dist  = get_gt_clusters(gt, X)
    clu_dist = get_clu_clusters(clu, gt)
    mapping  = cluster_mapping(gt_dist, clu_dist, clu, gt)
    mapped   = np.array([mapping.get(c, -1) for c in clu])
    return float(f1_score(gt, mapped, average="macro", zero_division=0))


def nearest_labels(X, summaries, labels_by_id):
    if not summaries:
        return np.full(len(X), -1)
    ids     = list(summaries.keys())
    centers = np.array([summaries[mid]["position"] for mid in ids])
    larr    = np.array([labels_by_id.get(mid, -1) for mid in ids])
    diffs   = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists   = np.sqrt(np.sum(diffs ** 2, axis=2))
    return larr[np.argmin(dists, axis=1)]


def load_summary(path):
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def run_one_worker(args):
    """Worker: (algorithm, dataset, params) -> metrics dict. Multiprocessing-safe."""
    algorithm, dataset, params = args

    # Re-add paths inside worker process
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [root,
              os.path.join(root, "baseframework"),
              os.path.join(root, "baseframework", "data"),
              os.path.join(root, "baseframework", "metrics")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from baseframework.metrics.cmm import (get_cmm, cluster_mapping,
                                            get_clu_clusters, get_gt_clusters)
    from src.stream_runner import load_dataset
    from src.offline import OfflineClusterer
    from src.clustree import ClusTree
    from src.liartree import LiarTree
    from sklearn.metrics import f1_score
    import numpy as np
    from collections import Counter

    TREE_CLASSES_W = {"liartree": LiarTree, "clustree": ClusTree}
    BASE_TREE_KWARGS_W = {
        "liartree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                         budget=10**9, max_height=0, horizon=1000,
                         overlap_reduction=True),
        "clustree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                         budget=10**9, max_height=0, horizon=1000),
    }

    def _purity(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        correct = 0
        for label in set(y_pred.tolist()):
            if label == -1: continue
            mask = y_pred == label
            if mask.any():
                correct += Counter(y_true[mask].tolist()).most_common(1)[0][1]
        return correct / max(len(y_true), 1)

    def _f1(gt, clu, X):
        gd = get_gt_clusters(gt, X)
        cd = get_clu_clusters(clu, gt)
        mp = cluster_mapping(gd, cd, clu, gt)
        mapped = np.array([mp.get(c, -1) for c in clu])
        return float(f1_score(gt, mapped, average="macro", zero_division=0))

    def _near(X, summaries, labels_by_id):
        if not summaries: return np.full(len(X), -1)
        ids = list(summaries.keys())
        centers = np.array([summaries[m]["position"] for m in ids])
        larr = np.array([labels_by_id.get(m, -1) for m in ids])
        diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        return larr[np.argmin(np.sqrt(np.sum(diffs**2, axis=2)), axis=1)]

    try:
        X_full, y_full = load_dataset(dataset, scale=True)
        rng = np.random.default_rng(42)
        n = len(X_full)
        cap = min(2000, n)
        idx = np.sort(rng.choice(n, size=cap, replace=False)) if cap < n else np.arange(n)
        X_cmm, y_cmm = X_full[idx], y_full[idx]
        w_cmm = np.ones(len(idx))

        tree_kwargs = dict(BASE_TREE_KWARGS_W[algorithm])
        tree_kwargs["lam"] = float(params["param_lam"])
        if algorithm == "liartree" and params.get("param_noise_threshold"):
            tree_kwargs["noise_threshold"] = float(params["param_noise_threshold"])

        tree = TREE_CLASSES_W[algorithm](**tree_kwargs)
        for i in range(len(X_full)):
            tree.learn_one(X_full[i], t=i + 1)

        summaries = tree.get_summary()
        offline = OfflineClusterer(
            method="dbscan",
            eps=float(params["param_eps"]),
            min_samples=int(params["param_min_samples"]),
            weighted=False, random_state=42,
        )
        labels_by_id = offline.predict_summaries(summaries)
        pred = _near(X_cmm, summaries, labels_by_id)

        n_clusters = len(set(pred.tolist()) - {-1})
        cmm     = float(get_cmm(X_cmm, y_cmm, pred, w_cmm, 5))
        f1      = _f1(y_cmm, pred, X_cmm) if n_clusters >= 2 else 0.0
        purity  = _purity(y_cmm, pred)

        return dict(algorithm=algorithm, dataset=dataset,
                    cmm=cmm, f1=f1, purity=purity,
                    n_clusters=n_clusters, n_microclusters=len(summaries),
                    error=None)
    except Exception as e:
        return dict(algorithm=algorithm, dataset=dataset,
                    cmm=float("nan"), f1=float("nan"),
                    purity=float("nan"), n_clusters=0, n_microclusters=0,
                    error=str(e))


# ── main ─────────────────────────────────────────────────────────────────

def main():
    rows = load_summary(SUMMARY_CSV)
    # Filter to the datasets we want to run
    tasks = [(r["algorithm"], r["dataset"], r)
             for r in rows if r["dataset"] in DATASETS_TO_RUN]

    print(f"Running {len(tasks)} evaluations with {multiprocessing.cpu_count()} cores …")

    with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(run_one_worker, tasks)):
            algo, ds = res["algorithm"], res["dataset"]
            if res["error"]:
                print(f"  [{i+1}/{len(tasks)}] ERR  {algo:10s} {ds}: {res['error']}")
            else:
                print(f"  [{i+1}/{len(tasks)}] OK   {algo:10s} {ds:30s} "
                      f"CMM={res['cmm']:.3f}  F1={res['f1']:.3f}  "
                      f"purity={res['purity']:.3f}  k={res['n_clusters']}")
            results.append(res)

    out_path = os.path.join(ROOT, "evaluation", "results_after_fixes.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
