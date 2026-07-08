#!/usr/bin/env python3
"""Runner for the LiarTree stream-clustering algorithm.

Mirrors run_clustree.py but uses LiarTree with its noise-handling and
overlap-reduction extensions.  Requires the baseframework submodule
(baseframework/) to be initialised.

Examples
--------
# 3-cluster dataset, default settings
python run_liartree.py --dataset complex9 --n-clusters 9

# Deterministic distance-budget interruption (reproducible runs)
python run_liartree.py --dataset rbf3 --interrupt budget --budget 72 --n-clusters 3

# Disable noise handling (compare with ClusTree baseline)
python run_liartree.py --dataset complex9 --noise-threshold 2.0

# Disable overlap reduction
python run_liartree.py --dataset complex9 --no-overlap

# DBSCAN offline phase
python run_liartree.py --dataset complex9 --offline dbscan --eps 0.05

# Time-limited run (anytime: stop after 2 s)
python run_liartree.py --dataset complex9 --stop-after 2
"""
import argparse

from src.liartree import LiarTree
from src.offline import OfflineClusterer
from src.stream_runner import StreamRunner, load_dataset


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="LiarTree anytime stream clustering")

    # data
    p.add_argument("--dataset", default="complex9",
                   choices=["complex9", "rbf3", "har", "stream_densired"])
    p.add_argument("--no-scale", action="store_true", help="skip MinMax scaling")

    # anytime interruption / stream speed
    p.add_argument("--interrupt", default="time", choices=["time", "budget"],
                   help="how each insertion is cut off: 'time' (paper, real "
                        "wall-clock) or 'budget' (deterministic distance-count)")
    p.add_argument("--time-slice", type=float, default=None,
                   help="per-object time allowance in seconds for --interrupt "
                        "time (default: no limit unless --pps paces the stream)")
    p.add_argument("--budget", type=int, default=72,
                   help="distance computations per object "
                        "(low = fast stream, coarse model). "
                        "LiarTree uses ~2× more dist calls than ClusTree per "
                        "insertion (overlap look-ahead), so 72 is the minimum "
                        "that avoids buffering at default tree depth.")
    p.add_argument("--pps", type=float, default=0.0,
                   help="points per second (0 = unbounded)")
    p.add_argument("--stop-after", type=float, default=-1.0,
                   help="stop after X seconds (-1 = whole dataset; "
                        "Ctrl+C also stops)")

    # tree / ClusTree params
    p.add_argument("--max-fanout", type=int, default=3)
    p.add_argument("--leaf-max", type=int, default=3)
    p.add_argument("--lambda", dest="lam", type=float, default=0.001)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--max-height", type=int, default=0,
                   help="cap tree height (0 = unbounded)")
    p.add_argument("--aggregate", action="store_true",
                   help="aggregate similar objects before insertion (Sec. 3.3 "
                        "speed-up for very fast streams)")
    p.add_argument("--max-aggregates", type=int, default=20,
                   help="cap on the number of open aggregates")
    p.add_argument("--descent", default="depthfirst",
                   choices=["depthfirst", "breadthfirst",
                            "bestfirst", "iterativedf"])

    # LiarTree-specific params
    p.add_argument("--noise-threshold", type=float, default=0.5,
                   help="noise probability threshold θ (Def. 2); "
                        "lower = more aggressive noise detection. "
                        "Values > 1 effectively disable noise handling.")
    p.add_argument("--no-overlap", action="store_true",
                   help="disable local overlap reduction (Sec. 2.2); "
                        "falls back to ClusTree's greedy entry selection")

    # offline phase
    p.add_argument("--offline", default="kmeans", choices=["kmeans", "dbscan"])
    p.add_argument("--n-clusters", type=int, default=9,
                   help="k for k-Means offline phase")
    p.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps")
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--unweighted", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # metrics / output
    p.add_argument("--no-cmm", action="store_true", help="skip CMM metric")
    p.add_argument("--no-ts", action="store_true",
                   help="skip Temporal Silhouette metric")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-batch progress lines")
    return p


def main():
    args = build_arg_parser().parse_args()

    X, y = load_dataset(args.dataset, scale=not args.no_scale)

    tree = LiarTree(
        noise_threshold=args.noise_threshold,
        overlap_reduction=not args.no_overlap,
        max_fanout=args.max_fanout,
        leaf_capacity_max=args.leaf_max,
        lam=args.lam,
        horizon=args.horizon,
        descent=args.descent,
        interrupt=args.interrupt,
        time_slice=args.time_slice,
        budget=args.budget,
        max_height=args.max_height,
    )
    offline = OfflineClusterer(
        method=args.offline,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        weighted=not args.unweighted,
        random_state=args.seed,
    )
    sim = StreamRunner(
        tree, offline,
        stop_after=args.stop_after,
        pps=args.pps,
        compute_cmm=not args.no_cmm,
        compute_ts=not args.no_ts,
        progress=not args.quiet,
        aggregate=args.aggregate,
        max_aggregates=args.max_aggregates,
    )

    print("dataset=%s  points=%d  dim=%d  interrupt=%s  offline=%s"
          % (args.dataset, len(X), X.shape[1], args.interrupt, args.offline))
    print("noise_threshold=%.2f  overlap_reduction=%s"
          % (args.noise_threshold, not args.no_overlap))
    sim.run(X, y)


if __name__ == "__main__":
    main()
