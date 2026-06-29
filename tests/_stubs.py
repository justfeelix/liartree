from __future__ import annotations

import sys
import types

import numpy as np


def install_baseframework_stubs():
    if "stream_clustering" not in sys.modules:
        stream_clustering = types.ModuleType("stream_clustering")
        stream_clustering.OnlinePhase_StreamClusterer = object
        stream_clustering.OfflinePhase_Clusterer = object
        sys.modules["stream_clustering"] = stream_clustering

    if "utils" not in sys.modules:
        utils = types.ModuleType("utils")

        def distance_calc(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.ndim == 2 and b.ndim == 2:
                return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            return float(np.linalg.norm(a - b))

        utils.distance_calc = distance_calc
        sys.modules["utils"] = utils
