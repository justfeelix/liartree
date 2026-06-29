"""LiarTree scaffold.

Extends ClusTree with per-node noise buffers and Gompertz-based
noise-to-cluster events (Sec. 2.2–2.3 of the paper).

Tasks
--------------
Task 1 (Nils)  — ClusTree base:         done, see clustree.py
Task 2 (Felix) — noise handling:         this file + noise_handler.py
Task 3 (Anton) — liar node mechanism:    _liar_proc stub below
Task 4 (Paul)  — overlap look-ahead:     _calc_closest_entry stub below
                 encouraged leaf splits: _leaf_proc stub below
"""
from __future__ import annotations

import numpy as np

from .clustree import ClusTree
from .microcluster import MicroCluster
from .node import Entry, Node
from .noise_handler import noise_probability, noise_to_cluster_test


class LiarTree(ClusTree):
    """ClusTree extended with per-node noise buffers and liar-node novelty growth.

    Extra parameter
    ---------------
    noise_threshold : float in [0, 1]
        A point is classified as noise when its noise probability (Def. 2)
        is >= noise_threshold.  0 = nothing is noise; 1 = everything outside
        every entry radius is noise.
    """

    def __init__(self, noise_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.noise_threshold = noise_threshold


    # Main LiarTree addition to ClusTree =========================================================================
    # Descent override — wires in liar check, enc_split, and noise check
    # =========================================================================

    def _insert_depth_first(self, x_cf: MicroCluster, t: int):
        """Algorithm 1 depth-first descent with liar and noise handling.

        Compared to ClusTree._insert_depth_first, this adds:
          • liar hook at the top of each iteration
          • the enc_split flag tracked across levels (wires it in via
            _calc_closest_entry; defaults to True until then)
          • noise check after finding the closest entry
        """
        node = self.root
        h = None        # hitchhiker CF
        path = []       # entries chosen on the way down (for discard rule)
        events = []
        assigned = -1
        # enc_split accumulates across levels: once True, stays True.
        # TODO Paul: _calc_closest_entry will set self._enc_split; for now
        # default to True so the noise-to-cluster test actually fires.
        self._enc_split = True

        while True:
            self._on_descend(node, t)

            # ── Task 3 (Anton): liar hook ─────────────────────────────────
            if node.is_liar:
                self._liar_proc(node, x_cf, t)
                break
            # ─────────────────────────────────────────────────────────────

            if node.is_leaf:
                assigned = self._leaf_proc(node, x_cf, h, path, events)
                break

            ex = self._calc_closest_entry(node, x_cf)
            if ex is None:
                break

            # ── Task 2 (Felix): noise check ──────────────────────────────
            np_score = noise_probability(x_cf.mean(), node.entries, self._dist)
            if np_score >= self.noise_threshold:
                self._noise_proc(x_cf, node, t, enc_split=self._enc_split,
                                 events=events)
                events.append("noise")
                break   # point consumed by noise buffer; stop descent
            # ─────────────────────────────────────────────────────────────

            # hitchhiker logic (unchanged from ClusTree)
            if h is not None:
                eh = self._calc_closest_entry(node, h)
                if eh is ex:
                    ex.summary.add(h)
                else:
                    self._deposit(eh, h)
                    h = None

            ex.summary.add(x_cf)
            path.append(ex)

            if self._time_is_up():
                ex.buffer.add(x_cf)
                if h is not None:
                    ex.buffer.add(h)
                events.append("buffered")
                break

            picked = ex.buffer
            if not picked.is_empty():
                h = picked.copy() if h is None else (h.add(picked) or h)
            ex.buffer = MicroCluster.empty(self._dim, t)
            node = ex.child

        return assigned, -1, ";".join(events)
    

    # =========================================================================
    # Task 3  — Anton 
    # =========================================================================

    def _liar_proc(self, node: Node, x_cf: MicroCluster, t: int) -> None:
        """Grow the liar subtree top-down by one level (Alg. 2, Sec. 2.4).
        """
        # TODO Anton
        pass

    # =========================================================================
    # Task 4  — Paul
    # =========================================================================

    def _calc_closest_entry(self, node: Node, cf: MicroCluster):
        """Closest entry with local overlap look-ahead (Fig. 1, Sec. 2.2).
        """
        # TODO Paul: check for irrelevant entries and set self._enc_split = True
        # TODO Paul: if top-2 entries overlap, peek one level and swap sub-entries
        # reutrn super
        pass

    def _leaf_proc(self, node: Node, x_cf: MicroCluster, h, path, events,
                   can_split=None):
        """Leaf insertion with encouraged splits (Sec. 2.5).
        """
        # TODO Paul: when self._enc_split is True, allow hitchhiker to overflow
        # reutrn super
        pass


    # =========================================================================
    # Task 2: noise procedure (Sec. 2.3)
    # =========================================================================

    def _noise_proc(self, x_cf: MicroCluster, node: Node, t: int,
                    enc_split: bool = True, events: list | None = None) -> None:
        """Handle a noise-classified point (Sec. 2.3).

        1. Initialise or decay the node's noise buffer and add x_cf to it.
        2. If enc_split is True, run the noise-to-cluster test (Def. 4).
        3. If the test fires, graduate the noise buffer to a new entry.

        The point is consumed here — it is NOT added to any regular entry.

        Parameters
        ----------
        enc_split : True when an irrelevant entry was found on the path down
            (set by _calc_closest_entry, Paul's task). The paper only checks
            the Gompertz test when this flag is True.
        events : running event log from the descent loop.
        """
        if events is None:
            events = []

        # Step 1: update noise buffer
        if node.noise_buffer is None:
            node.noise_buffer = MicroCluster.empty(self._dim, t)
        node.noise_buffer.decay(t, self.lam)
        node.noise_buffer.add(x_cf)

        # Step 2: check noise-to-cluster only when a split is encouraged
        if not enc_split:
            return

        valid_entries = [e for e in node.entries if not e.summary.is_empty()]
        if not valid_entries:
            return   # nothing to compare against yet

        n_avg  = float(np.mean([e.summary.n       for e in valid_entries]))
        rho_avg = float(np.mean([e.summary.density() for e in valid_entries]))
        n_nb   = node.noise_buffer.n
        rho_nb = node.noise_buffer.density()

        # Step 3: graduate if test passes
        if noise_to_cluster_test(n_nb, rho_nb, n_avg, rho_avg):
            self._noise_to_cluster(node, t, events)

    def _noise_to_cluster(self, node: Node, t: int, events: list) -> None:
        """Graduate the noise buffer into a new cluster entry (Sec. 2.3).

        Creates a new Entry whose child is an empty liar node (Anton grows it
        top-down via _liar_proc). If appending overflows the node, splits
        immediately so tree invariants are maintained.
        """
        cf = node.noise_buffer.copy()

        # Empty liar child — Anton's _liar_proc will grow the subtree here
        liar_child = Node(is_leaf=True, parent=node, timestamp=t)
        liar_child.is_liar = True
        liar_child.is_liar_root = True

        new_entry = Entry(
            summary=cf,
            buffer=MicroCluster.empty(self._dim, t),
            child=liar_child,
            entry_id=self._next_id,
        )
        self._next_id += 1
        liar_child.parent_entry = new_entry

        node.entries.append(new_entry)
        node.noise_buffer = None   # reset; fresh buffer starts on next noise point
        events.append("noise_to_cluster")

        # Handle inner-node overflow: split if we now exceed max_fanout
        if len(node.entries) > self.max_fanout:
            self._split(node, events)