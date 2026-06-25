"""Offline phase: cluster the ClusTree leaf summaries into the final clusters.

The online tree keeps fine micro-clusters; the macro clustering is done offline
on those summaries (Sec. 3.5), using scikit-learn and switchable between k-Means
and DBSCAN.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, DBSCAN

import stream_clustering   # baseframework, made importable in src/__init__


class OfflineClusterer(stream_clustering.OfflinePhase_Clusterer):
    """k-Means or DBSCAN over the micro-cluster summaries.

    method picks the algorithm ("kmeans" / "dbscan"); n_clusters is k for
    k-Means, eps/min_samples are the DBSCAN params, weighted passes the CF
    weights as sample_weight.
    """

    def __init__(self, method="kmeans", n_clusters=5, eps=0.1, min_samples=5,
                 weighted=True, random_state=0):
        method = method.lower()
        if method not in ("kmeans", "dbscan"):
            raise ValueError("method must be 'kmeans' or 'dbscan'")
        self.method = method
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.weighted = weighted
        self.random_state = random_state

    def predict_summaries(self, summaries: dict) -> dict:
        """Cluster the summaries and return {summary_id: label} (noise = -1),
        using each summary's position (and weight for k-Means). The base class
        predict_assignment is reused as-is."""
        if not summaries:
            return {}
        ids = list(summaries.keys())
        X = np.array([summaries[i]["position"] for i in ids], dtype=float)
        weights = np.array([summaries[i]["weight"] for i in ids], dtype=float)
        sample_weight = weights if self.weighted else None

        if self.method == "kmeans":
            k = max(1, min(self.n_clusters, len(ids)))
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        else:
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = model.fit(X, sample_weight=sample_weight).labels_
        return {i: int(label) for i, label in zip(ids, labels)}
