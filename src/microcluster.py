"""Micro-cluster / Cluster Feature for the ClusTree.

In the paper the cluster feature CF = (n, LS, SS) is the micro-cluster
(Sec. 3.1). We use the same object as an entry summary, as
an entry buffer (hitchhikers) and, for the LiarTree, as a node's noise buffer.
Time-weighting (Sec. 3.2) decays n, LS, SS by omega(t) = 2^(-lambda*t).
"""
from __future__ import annotations

import numpy as np


class MicroCluster:
    """Cluster Feature CF = (n, LS, SS) with time decay.

    n is the weighted point count, ls/ss the weighted linear/squared sums, and
    timestamp the last-update time used for decay.
    """

    def __init__(self, n=0.0, ls=None, ss=None, timestamp=0):
        raise NotImplementedError

    @classmethod
    def from_point(cls, x, timestamp):
        """CF for a single point x (n=1, LS=x, SS=x**2)."""
        raise NotImplementedError

    @classmethod
    def empty(cls, dim, timestamp=0):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def radius(self):
        """Standard deviation of the CF (the radius r_e used in Def. 2)."""
        raise NotImplementedError

    def decay(self, t, lam, beta=2.0):
        """Decay n, LS, SS by omega(t - timestamp), then set timestamp to t."""
        raise NotImplementedError

    def add(self, other):
        """Add another CF in place (CFs are additive, Sec. 3.2)."""
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def is_empty(self):
        raise NotImplementedError

    # LiarTree only (Def. 3): volume V_e = C_d * r^d, density rho_e = n / V_e
    def volume(self):
        raise NotImplementedError

    def density(self):
        raise NotImplementedError
