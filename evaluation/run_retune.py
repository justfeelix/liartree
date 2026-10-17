#!/usr/bin/env python3
"""Focused re-tuning of LiarTree and ClusTree on complex9 using FIXED code.

Strategy:
- Start from the known-good params found by the original Optuna sweep
- Run a 1-D sweep over `lam` (the parameter most affected by Bug D)
  while holding eps/min_samples/noise_threshold fixed at the known-good values
- Also run the cross-param comparison: each algorithm with the other's best params

Bug D (premature _on_descend in look-ahead) was adding spurious extra decay to
subtrees. With it fixed, the optimal lam should be LOWER (less decay needed since
we're no longer discounting peeked-at subtrees). We scan lam to find this new sweet-spot.

Outputs:
  param_sweep_fixed_code/lam_sweep_results.json   — all lam sweep results
  param_sweep_fixed_code/best_params.json          — best params found
  param_sweep_fixed_code/cross_param.json          — cross-param comparison

Run from liartree/liartree/:
    python evaluation/run_retune.py
"""
from __future__ import annotations
import sys, os, json, time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT,
          os.path.join(ROOT, "baseframework"),
          os.path.join(ROOT, "baseframework", "data"),
          os.path.join(ROOT, "baseframework", "metrics")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Config ────────────────────────────────────────────────────────────────
DATASET   = "complex9"
SEED      = 42
MAX_CMM_N = 300          # subsample for fast CMM
CMM_K     = 5
OUTPUT_DIR = os.path.join(ROOT, "evaluation", "param_sweep_fixed_code")

# Known-good params from original Optuna sweep (on buggy code)
ORIG_LIARTREE = dict(lam=0.24602, eps=0.022021, min_samples=6,
                     noise_threshold=1.16085)
ORIG_CLUSTREE = dict(lam=0.00145, eps=0.024218, min_samples=1)

# lam grid: log-spaced around the original value and lower values
# (Bug D fix should shift optimal lam downward)
LAM_GRID_LIARTREE = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.246, 0.4, 0.8]
LAM_GRID_CLUSTREE = [0.0005, 0.00145, 0.003, 0.01, 0.03, 0.1]

BASE_TREE_KWARGS = {
    "liartree": dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                     budget=10**9, max_height=0, horizon=1000,
                     overlap_reduction=True),
    "clustree":  dict(max_fanout=3, leaf_capacity_max=3, descent="depthfirst",
                      budget=10**9, max_height=0, horizon=1000),
}


# ── Metric helpers ────────────────────────────────────────────────────────

def _nearest_labels(X, summaries, labels_by_id):
    if not summaries:
        return np.full(len(X), -1)
    ids     = list(summaries.keys())
    centers = np.array([summaries[m]["position"] for m in ids])
    larr    = np.array([labels_by_id.get(m, -1) for m in ids])
    diffs   = X[:, None, :] - centers[None, :, :]
    return larr[np.argmin(np.sqrt((diffs**2).sum(2)), axis=1)]


def _compute_f1(y_cmm, pred, X_cmm):
    from sklearn.metrics import f1_score
    from baseframework.metrics.cmm import (cluster_mapping,
                                           get_clu_clusters, get_gt_clusters)
    gd = get_gt_clusters(y_cmm, X_cmm)
    cd = get_clu_clusters(pred, y_cmm)
    mp = cluster_mapping(gd, cd, pred, y_cmm)
    mapped = np.array([mp.get(c, -1) for c in pred])
    return float(f1_score(y_cmm, mapped, average="macro", zero_division=0))


def _compute_purity(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    correct = 0
    for lbl in set(y_pred.tolist()):
        if lbl == -1: continue
        mask = y_pred == lbl
        if mask.any():
            correct += Counter(y_true[mask].tolist()).most_common(1)[0][1]
    return correct / max(len(y_true), 1)


def _run_one(algorithm, params, X_full, X_cmm, y_cmm, w_cmm):
    from src.clustree import ClusTree
    from src.liartree import LiarTree
    from src.offline import OfflineClusterer
    from baseframework.metrics.cmm import get_cmm

    CLASSES = {"liartree": LiarTree, "clustree": ClusTree}
    tree_kw = dict(BASE_TREE_KWARGS[algorithm])
    for k in ("lam", "noise_threshold"):
        if k in params:
            tree_kw[k] = params[k]

    t0 = time.time()
    tree = CLASSES[algorithm](**tree_kw)
    for i, x in enumerate(X_full):
        tree.learn_one(x, t=i + 1)
    summaries = tree.get_summary()

    offline = OfflineClusterer(method="dbscan",
                               eps=params["eps"],
                               min_samples=int(params["min_samples"]),
                               weighted=False, random_state=SEED)
    labels_by_id = offline.predict_summaries(summaries)
    pred = _nearest_labels(X_cmm, summaries, labels_by_id)

    n_clusters = len(set(pred.tolist()) - {-1})
    if n_clusters < 2:
        return dict(cmm=0.0, f1=0.0, purity=0.0, n_clusters=n_clusters,
                    n_microclusters=len(summaries), elapsed=time.time()-t0,
                    params=dict(params))

    cmm    = float(get_cmm(X_cmm, y_cmm, pred, w_cmm, CMM_K))
    f1     = _compute_f1(y_cmm, pred, X_cmm)
    purity = _compute_purity(y_cmm, pred)
    return dict(cmm=cmm, f1=f1, purity=purity,
                n_clusters=n_clusters, n_microclusters=len(summaries),
                elapsed=time.time()-t0, params=dict(params))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from src.stream_runner import load_dataset
    X_full, y_full = load_dataset(DATASET, scale=True)
    rng = np.random.default_rng(SEED)
    n = len(X_full)
    cap = min(MAX_CMM_N, n)
    idx = np.sort(rng.choice(n, size=cap, replace=False))
    X_cmm, y_cmm = X_full[idx], y_full[idx]
    w_cmm = np.ones(len(idx))
    true_k = len(np.unique(y_full))
    print(f"Dataset: {DATASET}  n={n}  d={X_full.shape[1]}  k={true_k}  "
          f"CMM subsample={len(idx)}", flush=True)

    all_results = {}

    # ── LiarTree: sweep lam, hold other params fixed ──────────────────────
    print("\n[LiarTree] lam sweep (fixed eps/min_samples/noise_threshold):", flush=True)
    lt_results = []
    for lam in LAM_GRID_LIARTREE:
        params = dict(ORIG_LIARTREE, lam=lam)
        r = _run_one("liartree", params, X_full, X_cmm, y_cmm, w_cmm)
        lt_results.append(r)
        print(f"  lam={lam:.4g}  cmm={r['cmm']:.4f}  f1={r['f1']:.4f}  "
              f"purity={r['purity']:.4f}  k={r['n_clusters']}  "
              f"({r['elapsed']:.1f}s)", flush=True)
    all_results["liartree_lam_sweep"] = lt_results
    best_lt = max(lt_results, key=lambda r: r["f1"])
    print(f"  → best lam={best_lt['params']['lam']:.4g}  "
          f"cmm={best_lt['cmm']:.4f}  f1={best_lt['f1']:.4f}", flush=True)

    # ── ClusTree: sweep lam, hold other params fixed ──────────────────────
    print("\n[ClusTree] lam sweep (fixed eps/min_samples):", flush=True)
    ct_results = []
    for lam in LAM_GRID_CLUSTREE:
        params = dict(ORIG_CLUSTREE, lam=lam)
        r = _run_one("clustree", params, X_full, X_cmm, y_cmm, w_cmm)
        ct_results.append(r)
        print(f"  lam={lam:.5g}  cmm={r['cmm']:.4f}  f1={r['f1']:.4f}  "
              f"purity={r['purity']:.4f}  k={r['n_clusters']}  "
              f"({r['elapsed']:.1f}s)", flush=True)
    all_results["clustree_lam_sweep"] = ct_results
    best_ct = max(ct_results, key=lambda r: r["f1"])
    print(f"  → best lam={best_ct['params']['lam']:.5g}  "
          f"cmm={best_ct['cmm']:.4f}  f1={best_ct['f1']:.4f}", flush=True)

    # ── Save lam sweep ────────────────────────────────────────────────────
    out1 = os.path.join(OUTPUT_DIR, "lam_sweep_results.json")
    with open(out1, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nLam sweep saved → {out1}")

    best_params = {
        "liartree": best_lt,
        "clustree":  best_ct,
    }
    out2 = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(out2, "w") as f:
        json.dump(best_params, f, indent=2)

    # ── Cross-param comparison ────────────────────────────────────────────
    print("\n─── Cross-param comparison ──────────────────────────────────", flush=True)
    print("  (each algorithm evaluated under the OTHER algorithm's best params)", flush=True)
    cross = []

    # For cross-param we need compatible params:
    # LiarTree can accept clustree params (just add noise_threshold from orig)
    # ClusTree ignores noise_threshold, so LiarTree's params work with it removed

    for eval_algo in ["liartree", "clustree"]:
        for param_src in ["liartree", "clustree"]:
            src = dict(best_params[param_src]["params"])
            if eval_algo == "clustree":
                src.pop("noise_threshold", None)
            try:
                r = _run_one(eval_algo, src, X_full, X_cmm, y_cmm, w_cmm)
            except Exception as e:
                r = dict(cmm=float("nan"), f1=float("nan"), purity=float("nan"),
                         n_clusters=0, n_microclusters=0, elapsed=0,
                         params=src, error=str(e))
            r["algorithm"] = eval_algo
            r["param_src"] = param_src
            cross.append(r)
            if not np.isnan(r["cmm"]):
                print(f"  {eval_algo:10s} @ {param_src:10s} params  "
                      f"CMM={r['cmm']:.4f}  F1={r['f1']:.4f}  "
                      f"k={r['n_clusters']}", flush=True)
            else:
                print(f"  {eval_algo:10s} @ {param_src:10s} params  FAILED ({r.get('error', '?')})", flush=True)

    out3 = os.path.join(OUTPUT_DIR, "cross_param.json")
    with open(out3, "w") as f:
        json.dump(cross, f, indent=2)
    print(f"\nCross-param saved → {out3}")


if __name__ == "__main__":
    main()
