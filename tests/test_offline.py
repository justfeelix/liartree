from __future__ import annotations

import unittest

import numpy as np

from tests._stubs import install_baseframework_stubs

install_baseframework_stubs()

import src.offline as offline_module
from src.offline import OfflineClusterer


class FakeKMeans:
    instances = []

    def __init__(self, n_clusters, random_state, n_init):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        FakeKMeans.instances.append(self)

    def fit(self, X, sample_weight=None):
        self.X = X
        self.sample_weight = sample_weight
        self.labels_ = np.arange(len(X)) % self.n_clusters
        return self


class FakeDBSCAN:
    instances = []

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        FakeDBSCAN.instances.append(self)

    def fit(self, X, sample_weight=None):
        self.X = X
        self.sample_weight = sample_weight
        self.labels_ = np.array([-1, 0][:len(X)], dtype=int)
        return self


class OfflineClustererTests(unittest.TestCase):
    def setUp(self):
        self._old_kmeans = offline_module.KMeans
        self._old_dbscan = offline_module.DBSCAN
        FakeKMeans.instances = []
        FakeDBSCAN.instances = []
        offline_module.KMeans = FakeKMeans
        offline_module.DBSCAN = FakeDBSCAN

    def tearDown(self):
        offline_module.KMeans = self._old_kmeans
        offline_module.DBSCAN = self._old_dbscan

    def test_empty_summaries_return_empty_labels(self):
        clusterer = OfflineClusterer(method="kmeans")

        self.assertEqual(clusterer.predict_summaries({}), {})

    def test_kmeans_filters_invalid_summaries_clamps_k_and_uses_weights(self):
        clusterer = OfflineClusterer(method="kmeans", n_clusters=5, weighted=True)
        summaries = {
            2: {"position": [2.0, 0.0], "weight": 2.5},
            1: {"position": [0.0, 0.0], "weight": 1.5},
            3: {"position": [np.inf, 0.0], "weight": 1.0},
            4: {"position": [3.0, 0.0], "weight": 0.0},
            5: {"weight": 1.0},
        }

        labels = clusterer.predict_summaries(summaries)

        self.assertEqual(set(labels), {1, 2})
        self.assertTrue(all(isinstance(label, int) for label in labels.values()))
        model = FakeKMeans.instances[-1]
        self.assertEqual(model.n_clusters, 2)
        np.testing.assert_allclose(model.sample_weight, [1.5, 2.5])

    def test_dbscan_can_return_noise_labels_without_crashing(self):
        clusterer = OfflineClusterer(method="dbscan", eps=0.3, min_samples=2)
        summaries = {
            "a": {"position": [0.0], "weight": 1.0},
            "b": {"position": [0.2], "weight": 1.0},
        }

        labels = clusterer.predict_summaries(summaries)

        self.assertEqual(labels, {"a": -1, "b": 0})
        self.assertEqual(FakeDBSCAN.instances[-1].eps, 0.3)

    def test_macro_clusters_aggregate_weight_center_members_and_radius(self):
        clusterer = OfflineClusterer(method="kmeans")
        summaries = {
            1: {"position": [0.0, 0.0], "weight": 1.0, "radius": 0.5},
            2: {"position": [2.0, 0.0], "weight": 3.0, "radius": 1.0},
            3: {"position": [10.0, 0.0], "weight": 1.0, "radius": 2.0},
            4: {"position": [5.0, 0.0], "weight": -1.0, "radius": 1.0},
        }
        labels = {1: 0, 2: 0, 3: -1, 4: 0}

        macros = clusterer.macro_clusters(summaries, labels)

        self.assertEqual(set(macros), {0})
        macro = macros[0]
        np.testing.assert_allclose(macro["center"], [1.5, 0.0])
        self.assertEqual(macro["weight"], 4.0)
        self.assertEqual(macro["member_ids"], [1, 2])
        self.assertAlmostEqual(macro["radius"], 2.0)

    def test_all_noise_dbscan_has_no_non_noise_macro_clusters(self):
        clusterer = OfflineClusterer(method="dbscan")
        summaries = {
            1: {"position": [0.0], "weight": 1.0},
            2: {"position": [1.0], "weight": 1.0},
        }

        self.assertEqual(clusterer.macro_clusters(summaries, {1: -1, 2: -1}), {})


if __name__ == "__main__":
    unittest.main()
