"""LiarTree online phase.

Extends ClusTree with per-node noise buffers, Gompertz-based
noise-to-cluster events, and local overlap-reduction look-ahead during entry
selection.
"""
from __future__ import annotations

import numpy as np

from .clustree import ClusTree
from .microcluster import MicroCluster
from .node import Entry, Node
from .noise_handler import noise_probability, noise_to_cluster_test
from .overlap_handler import OverlapHandlerMixin


class LiarTree(ClusTree, OverlapHandlerMixin):
    """ClusTree extended with per-node noise buffers and liar-node novelty growth.

    Parameters
    ----------
    noise_threshold : float in [0, 1]
        A point is classified as noise when its noise probability (Def. 2)
        is >= noise_threshold.  0 = nothing is noise; 1 = everything outside
        every entry radius is noise.
    overlap_reduction : bool
        Enable the local overlap look-ahead and child-entry swap (Sec. 2.2).
        Set to False to fall back to ClusTree's greedy entry selection.
    """

    def __init__(self, noise_threshold: float = 0.5,
                 overlap_reduction: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.noise_threshold = noise_threshold
        self.overlap_reduction = overlap_reduction
        self._enc_split = False   # accumulates to True when irrelevant entries
                                  # are found during descent (set by _calc_closest_entry)

    # =========================================================================
    # Main descent override — wires in liar check, enc_split, and noise check
    # =========================================================================

    def _insert_depth_first(self, x_cf: MicroCluster, t: int):
        """Algorithm 1 depth-first descent with liar and noise handling.

        Compared to ClusTree._insert_depth_first, this adds:
          • liar hook at the top of each iteration (Task 3)
          • the enc_split flag, set by _calc_closest_entry when an irrelevant
            entry is found; once True it stays True for the rest of the descent (Task4)
          • noise check BEFORE leaf check, matching Algorithm 1's order (Task 2)
        """
        node = self.root
        h = None        # hitchhiker CF
        path = []       # entries chosen on the way down (for discard rule)
        events = []
        assigned = -1
        self._enc_split = False   # reset at the start of each insertion

        while True:
            self._on_descend(node, t)

            # ── Task 3 (Anton): liar hook ─────────────────────────────────
            if node.is_liar:
                self._liar_proc(node, x_cf, t)
                break
            # ─────────────────────────────────────────────────────────────

            ex = self._calc_closest_entry(node, x_cf)
            if ex is None:
                break

            # hitchhiker logic — at inner nodes only, BEFORE noise check
            # (Algorithm 1 routes h before the noise test; _leaf_proc handles
            # h at leaf level so we skip the deposit there to avoid buffer=None)
            if not node.is_leaf and h is not None:
                eh = self._calc_closest_entry(node, h)
                if eh is ex:
                    ex.summary.add(h)
                else:
                    self._deposit(eh, h)
                    h = None

            # ── Task 2 (Felix): noise check — BEFORE leaf check (Alg. 1) ─
            np_score = noise_probability(x_cf.mean(), node.entries, self._dist)
            if np_score >= self.noise_threshold:
                self._noise_proc(x_cf, node, t, enc_split=self._enc_split,
                                 events=events)
                events.append("noise")
                break   # point consumed by noise buffer; stop descent
            # ─────────────────────────────────────────────────────────────

            # Leaf insertion — AFTER noise check (Algorithm 1)
            if node.is_leaf:
                assigned = self._leaf_proc(node, x_cf, h, path, events)
                break

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
    # Task 3 — Anton
    # =========================================================================

    def _liar_proc(self, node: Node, x_cf: MicroCluster, t: int) -> None:
        """Grow the liar subtree top-down by one level (Alg. 2, Sec. 2.4).
        """
        # TODO Anton
        pass

    # =========================================================================
    # Task 4 — Paul
    # =========================================================================

    def _calc_closest_entry(self, node: Node, cf: MicroCluster):
        """Closest entry with local overlap look-ahead (Fig. 1, Sec. 2.2).

        Also sets self._enc_split = True if an irrelevant (empty / too-old)
        entry is found, signalling that a split is encouraged downstream.
        Overlap look-ahead is skipped when self.overlap_reduction is False.
        """
        # Check for irrelevant entries → may set self._enc_split = True
        self._mark_enc_split_from_irrelevant(node)

        ranked = self._rank_entries_by_distance(node, cf)
        if not ranked:
            return None

        if not self.overlap_reduction:
            return ranked[0][1]

        return self._select_with_overlap_reduction(ranked, cf)

    def _leaf_proc(self, node: Node, x_cf: MicroCluster, h, path, events,
                   can_split=None):
        """Leaf insertion with encouraged splits (Sec. 2.5).

        When enc_split is True, the hitchhiker insertion can also trigger an
        overflow and split — unlike ClusTree, which merges to delay splits.
        """
        if can_split is None and self._enc_split:
            can_split = True
        return super()._leaf_proc(node, x_cf, h, path, events, can_split)

    # =========================================================================
    # Task2: Noise procedure (Sec. 2.3)
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
            (set by _calc_closest_entry). The paper only checks the Gompertz
            test when this flag is True (Def. 4, Sec. 2.3).
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

        n_avg   = float(np.mean([e.summary.n         for e in valid_entries]))
        rho_avg = float(np.mean([e.summary.density() for e in valid_entries]))
        n_nb    = node.noise_buffer.n
        rho_nb  = node.noise_buffer.density()

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