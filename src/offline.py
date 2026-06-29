"""Offline phase: cluster the ClusTree leaf summaries into the final clusters.

The online tree keeps fine micro-clusters; the macro clustering is done offline
on those summaries (Sec. 3.5), using scikit-learn and switchable between k-Means
and DBSCAN.
"""
from __future__ import annotations

import numpy as np

import stream_clustering   # baseframework, made importable in src/__init__

try:
    from sklearn.cluster import KMeans, DBSCAN
except ImportError:  # pragma: no cover - exercised in minimal local test envs
    KMeans = None
    DBSCAN = None


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
        rows = self._valid_summary_rows(summaries)
        if not rows:
            return {}
        ids = [row[0] for row in rows]
        X = np.array([row[1] for row in rows], dtype=float)
        weights = np.array([row[2] for row in rows], dtype=float)
        sample_weight = weights if self.weighted else None

        if self.method == "kmeans":
            self._require_clusterer(KMeans, "KMeans")
            k = max(1, min(self.n_clusters, len(ids)))
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        else:
            self._require_clusterer(DBSCAN, "DBSCAN")
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = model.fit(X, sample_weight=sample_weight).labels_
        return {i: int(label) for i, label in zip(ids, labels)}

    def macro_clusters(self, summaries: dict, labels_by_id: dict) -> dict:
        """Return macro-cluster metadata keyed by non-noise label.

        Each value contains the weighted center, total summary weight, member
        summary IDs, and a covering radius over member centers plus their own
        micro-cluster radii.
        """
        rows = self._valid_summary_rows(summaries)
        grouped = {}
        for summary_id, position, weight, radius in rows:
            label = labels_by_id.get(summary_id)
            if label is None or int(label) == -1:
                continue
            grouped.setdefault(int(label), []).append(
                (summary_id, position, weight, radius)
            )

        macro = {}
        for label in sorted(grouped):
            members = grouped[label]
            weights = np.array([member[2] for member in members], dtype=float)
            positions = np.array([member[1] for member in members], dtype=float)
            total_weight = float(weights.sum())
            if total_weight <= 0:
                continue
            center = np.average(positions, axis=0, weights=weights)
            radius = max(
                float(np.linalg.norm(position - center) + member_radius)
                for _summary_id, position, _weight, member_radius in members
            )
            macro[label] = {
                "center": center,
                "weight": total_weight,
                "member_ids": [member[0] for member in members],
                "radius": radius,
            }
        return macro

    @staticmethod
    def _require_clusterer(clusterer, name):
        if clusterer is None:
            raise ImportError(
                "scikit-learn is required for OfflineClusterer.%s" % name
            )

    def _valid_summary_rows(self, summaries: dict):
        if not summaries:
            return []

        rows = []
        dim = None
        for summary_id in self._ordered_ids(summaries.keys()):
            summary = summaries[summary_id]
            if not isinstance(summary, dict) or "position" not in summary:
                continue
            try:
                position = np.asarray(summary["position"], dtype=float)
                weight = float(summary["weight"])
            except (KeyError, TypeError, ValueError):
                continue
            if position.ndim != 1 or position.size == 0:
                continue
            if not np.all(np.isfinite(position)):
                continue
            if not np.isfinite(weight) or weight <= 0:
                continue
            if dim is None:
                dim = position.size
            if position.size != dim:
                continue
            radius = self._valid_radius(summary)
            rows.append((summary_id, position, weight, radius))
        return rows

    @staticmethod
    def _valid_radius(summary):
        try:
            radius = float(summary.get("radius", 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(radius) or radius < 0:
            return 0.0
        return radius

    @staticmethod
    def _ordered_ids(ids):
        try:
            return sorted(ids)
        except TypeError:
            return sorted(ids, key=lambda item: (type(item).__name__, repr(item)))
