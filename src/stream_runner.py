"""Stream runner: baseframework integration plus the anytime stream workflow.

All baseframework access is here so we reuse its data loading and metrics rather
than reimplementing them: load_dataset -> datahandler.load_data, compute_cmm ->
metrics/cmm, compute_temporal_silhouette -> metrics/TSindex. StreamSimulator
feeds the dataset as an anytime stream.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from . import BASEFRAMEWORK_DIR, setup_baseframework_path


@contextlib.contextmanager
def _in_baseframework_dir():
    # datahandler reads relative ./data paths, so run it from the submodule dir.
    prev = os.getcwd()
    os.chdir(BASEFRAMEWORK_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def load_dataset(name, batch=None, batch_max=None, scale=True):
    """Load a baseframework dataset (complex9 / rbf3 / har / stream_densired)."""
    setup_baseframework_path()
    import datahandler
    with _in_baseframework_dir():
        return datahandler.load_data(name, batch=batch, batch_max=batch_max,
                                     scale=scale)


def compute_cmm(X, gt, clu, weights=None, k=10):
    setup_baseframework_path()
    import cmm
    if weights is None:
        weights = np.ones(len(X))
    return cmm.get_cmm(np.asarray(X), np.asarray(gt), np.asarray(clu),
                       np.asarray(weights), k)


def compute_temporal_silhouette(t, X, labels, s=200, kn=200, c=1):
    setup_baseframework_path()
    import TSindex
    _, _, ts = TSindex.tempsil(np.asarray(t), np.asarray(X), np.asarray(labels),
                               s=s, kn=kn, c=c)
    return ts


class StreamSimulator:
    """Feed a dataset into a ClusTree as an anytime stream: a per-object distance
    budget sets the speed, processing stops after N seconds (-1 = all) or on a
    stdin keypress, then the offline phase and metrics run.
    """

    def __init__(self, online, offline=None, stop_after=-1.0):
        raise NotImplementedError

    def run(self, X, y=None):
        raise NotImplementedError
