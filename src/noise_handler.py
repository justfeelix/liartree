"""Noise handling math for the LiarTree.

Pure functions — no tree state. Import and call from LiarTree.

Implements Definitions 2, 3, 4 and the Gompertz sigmoid from Sec. 2.3.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Def. 2 — Noise probability
# ---------------------------------------------------------------------------

def noise_probability(x: np.ndarray, node_entries, dist_fn) -> float:
    """Noise probability of point x w.r.t. the entries of a node (Def. 2).

    np(o) = min over entries { dist(o, mu_e) / r_e }, capped at 1.0.

    A value of 0 means x sits inside the closest entry; 1 means it is at
    or beyond every entry's radius. Returns 1.0 when no valid entries exist
    (all empty or single-point with radius 0).

    Parameters
    ----------
    x : np.ndarray
        Raw coordinate vector of the point. Pass x_cf.mean() from the caller.
    node_entries : list[Entry]
        Regular entries of the node (noise buffer excluded).
    dist_fn : callable(a, b) -> float
        Budget-aware distance function; pass self._dist from LiarTree so
        noise-probability calls count against the insertion budget.
    """
    min_ratio = np.inf
    for e in node_entries:
        if e.summary.is_empty():
            continue
        r = e.summary.radius()
        if r == 0.0:
            continue   # single-point entry: radius undefined
        d = dist_fn(x, e.summary.mean())
        min_ratio = min(min_ratio, d / r)
    if not np.isfinite(min_ratio):
        return 1.0
    return float(min(1.0, min_ratio))


# ---------------------------------------------------------------------------
# Gompertz sigmoid (Sec. 2.3)
# ---------------------------------------------------------------------------

def gompertz_params(n_avg: float,
                    t0: float = 1e-4,
                    t1: float = 0.97):
    """Fit the Gompertz shape parameters b and c to two anchor points.

    The function g(n) = exp(-b * exp(-c * n)) is anchored so that:
      g(2)     ≈ t0  (≈ 0 when only 2 noise points — no cluster yet)
      g(n_avg) ≈ t1  (≈ 1 when noise buffer matches avg entry weight)

    Derivation from the two constraints:
      -b * exp(-2c)        = ln(t0)   ... (i)
      -b * exp(-c * n_avg) = ln(t1)   ... (ii)
    Dividing: exp(c*(n_avg-2)) = ln(t0)/ln(t1)
      c = ln(ln(t0) / ln(t1)) / (n_avg - 2)
      b = -ln(t0) * exp(2c)

    Returns (None, None) when n_avg <= 2 (cannot fit the curve; the caller
    should skip the noise-to-cluster check).
    """
    if n_avg <= 2.0:
        return None, None
    c = float(np.log(np.log(t0) / np.log(t1)) / (n_avg - 2.0))
    b = float(-np.log(t0) * np.exp(2.0 * c))
    return b, c


def gompertz(n_nb: float,
             n_avg: float,
             t0: float = 1e-4,
             t1: float = 0.97) -> float:
    """Gompertz sigmoid: g(n_nb) = exp(-b * exp(-c * n_nb))  (Sec. 2.3).

    b and c are fitted from anchor points (t0, t1) and n_avg.
    Returns 0.0 for the degenerate case n_avg <= 2.

    Parameters
    ----------
    n_nb  : weight of the noise buffer (n^(t)_nb).
    n_avg : average weight of the node's regular entries.
    """
    b, c = gompertz_params(n_avg, t0, t1)
    if b is None:
        return 0.0
    return float(np.exp(-b * np.exp(-c * n_nb)))


# ---------------------------------------------------------------------------
# Def. 4 — Noise-to-cluster event test
# ---------------------------------------------------------------------------

def noise_to_cluster_test(n_nb: float,
                           rho_nb: float,
                           n_avg: float,
                           rho_avg: float,
                           t0: float = 1e-4,
                           t1: float = 0.97) -> bool:
    """Return True if the noise buffer should graduate to a cluster (Def. 4).

    The event fires when:
        gompertz(n_nb, n_avg) * rho_nb  >=  rho_avg

    The Gompertz factor acts as a weight gate: a noise buffer that has
    barely two points (n_nb ≈ 2) gets a factor ≈ 0 and cannot pass the
    test no matter how dense it is. Once n_nb approaches n_avg the gate
    opens and density alone decides.

    Parameters
    ----------
    n_nb    : weight of the noise buffer CF.
    rho_nb  : density of the noise buffer CF  (n_nb / volume, Def. 3).
    n_avg   : average weight of the node's regular entries.
    rho_avg : average density of the node's regular entries.
    """
    if rho_avg == 0.0:
        return False
    g = gompertz(n_nb, n_avg, t0, t1)
    return bool(g * rho_nb >= rho_avg)
