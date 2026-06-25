#!/usr/bin/env python3
"""Runner for the ClusTree anytime stream-clustering algorithm.

Example:

    python run_clustree.py --dataset complex9 --speed 36 --offline kmeans \\
        --n-clusters 9 --stop-after -1
"""
import argparse


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="ClusTree anytime stream clustering (built on baseframework)")

    # data
    p.add_argument("--dataset", default="complex9",
                   choices=["complex9", "rbf3", "har", "stream_densired"],
                   help="baseframework dataset to stream")

    # anytime load / speed
    p.add_argument("--speed", type=int, default=36,
                   help="anytime load: max distance computations allowed per "
                        "object (low = fast stream, high = slow stream)")
    p.add_argument("--stop-after", type=float, default=-1.0,
                   help="stop after X seconds of processing and run the offline "
                        "phase (-1 = process the whole dataset)")

    # offline phase
    p.add_argument("--offline", default="kmeans", choices=["kmeans", "dbscan"],
                   help="offline (macro) clustering method")
    p.add_argument("--n-clusters", type=int, default=5, help="k for k-Means")
    p.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps")
    p.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")

    # TODO: tree params, decay, descent, seed, scale, metric toggles
    return p


def main():
    args = build_arg_parser().parse_args()
    # TODO: build the tree + offline clusterer and run the simulator
    raise SystemExit("run_clustree.py is not wired up yet.")


if __name__ == "__main__":
    main()
