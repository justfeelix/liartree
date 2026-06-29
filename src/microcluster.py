"""Micro-cluster / Cluster Feature for the ClusTree.

In the paper the cluster feature CF = (n, LS, SS) is the micro-cluster
(Sec. 3.1). We use the same object as an entry summary, as an entry buffer
(hitchhikers) and, for the LiarTree, as a node's noise buffer.
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
        self.n = float(n)
        self.ls = np.zeros(0) if ls is None else np.asarray(ls, dtype=float)
        self.ss = np.zeros(0) if ss is None else np.asarray(ss, dtype=float)
        self.timestamp = timestamp

    @classmethod
    def from_point(cls, x, timestamp):
        """CF for a single point x (n=1, LS=x, SS=x**2)."""
        x = np.asarray(x, dtype=float)
        return cls(1.0, x.copy(), x * x, timestamp)

    @classmethod
    def empty(cls, dim, timestamp=0):
        return cls(0.0, np.zeros(dim), np.zeros(dim), timestamp)

    def mean(self):
        if self.n == 0:
            return np.zeros_like(self.ls)
        return self.ls / self.n

    def variance(self):
        if self.n == 0:
            return np.zeros_like(self.ls)
        var = self.ss / self.n - (self.ls / self.n) ** 2
        return np.maximum(var, 0.0)  # clamp tiny negative values from rounding

    def radius(self):
        """Standard deviation of the CF (the radius r_e used in Def. 2)."""
        return float(np.sqrt(self.variance().sum()))

    def decay(self, t, lam, beta=2.0):
        """Decay n, LS, SS by omega(t - timestamp), then set timestamp to t."""
        omega = beta ** (-lam * (t - self.timestamp))
        self.n *= omega
        self.ls *= omega
        self.ss *= omega
        self.timestamp = t

    def add(self, other):
        """Add another CF in place (CFs are additive, Sec. 3.2)."""
        if self.ls.size == 0:
            self.ls = np.zeros_like(other.ls)
            self.ss = np.zeros_like(other.ss)
        self.n += other.n
        self.ls += other.ls
        self.ss += other.ss
        self.timestamp = max(self.timestamp, other.timestamp)

    def subtract(self, other):
        """Remove another CF's contribution in place (used by the discard rule)."""
        self.n -= other.n
        self.ls = self.ls - other.ls
        self.ss = self.ss - other.ss

    def copy(self):
        return MicroCluster(self.n, self.ls.copy(), self.ss.copy(), self.timestamp)

    def is_empty(self):
        return self.n == 0

    # LiarTree only (Def. 3): volume V_e = C_d * r^d, density rho_e = n / V_e
    def volume(self):
        # V_e = C_d * r^d, where C_d = pi^(d/2) / Gamma(d/2 + 1)
        # r = self.radius(), d = number of dimensions
        import math
        d = self.ls.size
        if d == 0:
            return 0.0
        r = self.radius()
        C_d = (math.pi ** (d / 2)) / math.gamma(d / 2 + 1)
        return C_d * (r ** d) # TODO be careful for high-dimensional data r^d can overflow or underflip

    def density(self):
        v = self.volume()
        if v == 0:
            return 0.0
        return self.n / v