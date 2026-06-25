"""Offline phase: cluster the ClusTree leaf summaries into the final clusters.

The online tree keeps fine micro-clusters; the macro clustering is done offline
on those summaries (Sec. 3.5), using scikit-learn and switchable between k-Means
and DBSCAN.
"""
from __future__ import annotations

import numpy as np

import stream_clustering   # baseframework, made importable in src/__init__


class OfflineClusterer(stream_clustering.OfflinePhase_Clusterer):
    """k-Means or DBSCAN over the micro-cluster summaries.

    method picks the algorithm ("kmeans" / "dbscan"); n_clusters is k for
    k-Means, eps/min_samples are the DBSCAN params, weighted passes the CF
    weights as sample_weight.
    """

    def __init__(self, method="kmeans", n_clusters=5, eps=0.1, min_samples=5,
                 weighted=True, random_state=0):
        raise NotImplementedError

    def predict_summaries(self, summaries: dict) -> dict:
        """Cluster the summaries and return {summary_id: label} (noise = -1),
        using each summary's position (and weight for k-Means). The base class
        predict_assignment is reused as-is."""
        raise NotImplementedError
