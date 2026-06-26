#!/usr/bin/env python3
"""Runner for the ClusTree anytime stream-clustering algorithm.

Example:

    python run_clustree.py --dataset complex9 --budget 36 --offline kmeans \\
        --n-clusters 9 --stop-after -1
"""
import argparse

from src.clustree import ClusTree
from src.offline import OfflineClusterer
from src.stream_runner import StreamSimulator, load_dataset


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="ClusTree anytime stream clustering (built on baseframework)")

    # data
    p.add_argument("--dataset", default="complex9",
                   choices=["complex9", "rbf3", "har", "stream_densired"])
    p.add_argument("--no-scale", action="store_true", help="skip MinMax scaling")

    # anytime load / stream speed
    p.add_argument("--budget", type=int, default=36,
                   help="distance computations allowed per object (low = fast "
                        "stream, coarse model)")
    p.add_argument("--pps", type=float, default=0.0,
                   help="points per second to feed in real time (0 = unbounded)")
    p.add_argument("--stop-after", type=float, default=-1.0,
                   help="stop after X seconds and run the offline phase "
                        "(-1 = whole dataset; Ctrl+C also stops)")

    # tree
    p.add_argument("--max-fanout", type=int, default=3)
    p.add_argument("--leaf-max", type=int, default=3)
    p.add_argument("--lambda", dest="lam", type=float, default=0.001)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--max-height", type=int, default=0,
                   help="cap tree height (0 = unbounded); once reached the model "
                        "size stops growing")
    p.add_argument("--aggregate", action="store_true",
                   help="aggregate similar objects before insertion (Sec. 3.3 "
                        "speed-up for very fast streams)")
    p.add_argument("--max-aggregates", type=int, default=20,
                   help="cap on the number of open aggregates")
    p.add_argument("--descent", default="depthfirst",
                   choices=["depthfirst", "breadthfirst", "bestfirst", "iterativedf"])

    # offline phase
    p.add_argument("--offline", default="kmeans", choices=["kmeans", "dbscan"])
    p.add_argument("--n-clusters", type=int, default=5, help="k for k-Means")
    p.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps")
    p.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    p.add_argument("--unweighted", action="store_true",
                   help="do not weight the offline clustering by CF weight")
    p.add_argument("--seed", type=int, default=0)

    # metrics
    p.add_argument("--no-cmm", action="store_true", help="skip the CMM metric")
    p.add_argument("--no-ts", action="store_true",
                   help="skip the Temporal Silhouette metric")
    p.add_argument("--quiet", action="store_true",
                   help="do not print progress lines while streaming")
    return p


def main():
    args = build_arg_parser().parse_args()

    X, y = load_dataset(args.dataset, scale=not args.no_scale)
    tree = ClusTree(max_fanout=args.max_fanout, leaf_capacity_max=args.leaf_max,
                    lam=args.lam, horizon=args.horizon, descent=args.descent,
                    budget=args.budget, max_height=args.max_height,
                    aggregate=args.aggregate, max_aggregates=args.max_aggregates)
    offline = OfflineClusterer(method=args.offline, n_clusters=args.n_clusters,
                               eps=args.eps, min_samples=args.min_samples,
                               weighted=not args.unweighted, random_state=args.seed)
    sim = StreamSimulator(tree, offline, stop_after=args.stop_after, pps=args.pps,
                          compute_cmm=not args.no_cmm, compute_ts=not args.no_ts,
                          progress=not args.quiet)

    print("dataset=%s points=%d dim=%d budget=%d offline=%s"
          % (args.dataset, len(X), X.shape[1], args.budget, args.offline))
    sim.run(X, y)


if __name__ == "__main__":
    main()
