"""LiarTree scaffold.

Extends the ClusTree with noise handling and top-down novelty growth. Only the
structure and TODOs are here, no implementation yet.
"""
from __future__ import annotations

from .clustree import ClusTree


class LiarTree(ClusTree):
    # TODO: __init__ with a noise_threshold param [Sec. 2.2]
    # TODO: override learn_one to follow Algorithm 1 [Sec. 2.1]
    # TODO: override _calc_closest_entry, add noise marking + overlap look-ahead [Sec. 2.2, Def. 2, Fig. 1]
    # TODO: noise_proc, noise buffer -> new cluster via the gompertz test [Sec. 2.3, Def. 3-4]
    # TODO: liar_proc, grow the subtree top-down [Alg. 2, Sec. 2.4]
    # TODO: override _leaf_proc to allow encouraged splits [Sec. 2.5]
    pass
