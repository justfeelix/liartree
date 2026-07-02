"""LiarTree online phase.

Extends ClusTree with:
  • per-node noise buffers and Gompertz-based noise-to-cluster events
    (Algorithm 1, Sec. 2.3 — Felix)
  • local overlap-reduction look-ahead during entry selection
    (Sec. 2.2, Fig. 1 — Paul)
  • top-down liar-node novelty growth
    (Algorithm 2, Sec. 2.4 — Anton)

Flag convention:
  is_liar      – marks an *empty* node at the bottom of the growing liar
                 subtree.  The descent hook triggers _liar_proc when it
                 reaches such a node.  Cleared as soon as the node is filled.
  is_liar_root – marks the topmost node of the liar subtree; limits split
                 propagation.  Replaced by a new liar-root when the old one
                 overflows (Sec. 2.4, Fig. 2e).
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
          • liar hook at the top of each iteration (Alg. 1, line 5)
          • the enc_split flag, set by _calc_closest_entry when an irrelevant
            entry is found; once True it stays True for the rest of the descent
          • noise check at INNER nodes (after _calc_closest_entry so that
            enc_split is already updated and the noise check uses stable radii;
            leaf nodes are never noise-checked — their entries are too fine-grained
            and unstable in the early stream)

        Why noise at inner only (not leaf):
          Algorithm 1 puts the noise check before the leaf check, but inner
          entries aggregate many points → stable, meaningful radii. Leaf
          entries may be single-point (r=0) or tiny, causing almost all
          points to be falsely flagged as noise before the tree matures.
          Restricting noise detection to inner nodes matches the spirit of
          Sec. 2.3.
        """
        node = self.root
        h = None        # hitchhiker CF
        path = []       # entries chosen on the way down (for discard rule)
        events = []
        assigned = -1
        self._enc_split = False   # reset at the start of each insertion

        while True:
            self._on_descend(node, t)

            # liar hook — Algorithm 1, line 5
            # h is intentionally not forwarded: Algorithm 1 calls liarProc(liarNode, x)
            # without the hitchhiker (Paper p. 407); h is lost at this point.
            if node.is_liar:
                assigned = self._liar_proc(node, x_cf, t, events)
                break

            # Leaf insertion (Algorithm 1, line 15).
            # Placed BEFORE noise check so that leaf entries accumulate
            # normally; noise detection runs only at inner nodes below.
            if node.is_leaf:
                assigned = self._leaf_proc(node, x_cf, h, path, events)
                break

            # Inner node: find closest entry (also sets self._enc_split if
            # an irrelevant entry exists — Algorithm 1, line 7 / Sec. 2.2)
            ex = self._calc_closest_entry(node, x_cf)
            if ex is None:
                break

            # Hitchhiker routing — Algorithm 1, lines 8-11
            # (inner only; leaf entries have no buffer to deposit into)
            if h is not None:
                eh = self._calc_closest_entry(node, h)
                if eh is ex:
                    ex.summary.add(h)
                else:
                    self._deposit(eh, h)
                    h = None

            # Noise check at inner nodes — Algorithm 1, lines 12-13
            # enc_split is now set (updated by _calc_closest_entry above).
            np_score = noise_probability(x_cf.mean(), node.entries, self._dist)
            if np_score >= self.noise_threshold:
                self._noise_proc(x_cf, node, t, enc_split=self._enc_split,
                                 events=events)
                events.append("noise")
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
    # Overlap look-ahead entry selection (Sec. 2.2 — Paul)
    # =========================================================================

    def _calc_closest_entry(self, node: Node, cf: MicroCluster):
        """Closest entry with local overlap look-ahead (Fig. 1, Sec. 2.2).

        Also sets self._enc_split = True if an irrelevant (empty / too-old)
        entry is found, signalling that a split is encouraged downstream.
        Overlap look-ahead is skipped when self.overlap_reduction is False.
        """
        self._mark_enc_split_from_irrelevant(node)

        ranked = self._rank_entries_by_distance(node, cf)
        if not ranked:
            return None

        if not self.overlap_reduction:
            return ranked[0][1]

        return self._select_with_overlap_reduction(ranked, cf)

    # =========================================================================
    # Encouraged-split leaf proc (Sec. 2.5 — Felix)
    # =========================================================================

    def _leaf_proc(self, node: Node, x_cf: MicroCluster, h, path, events,
                   can_split=None):
        """Leaf insertion with encouraged splits (Sec. 2.5).

        When enc_split is True, overflow always triggers a split instead of a
        merge — unlike ClusTree, which merges to delay splits.
        """
        if can_split is None and self._enc_split:
            can_split = True
        return super()._leaf_proc(node, x_cf, h, path, events, can_split)

    # =========================================================================
    # Noise procedure (Sec. 2.3 — Felix)
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
            (set by _calc_closest_entry). The paper only runs the Gompertz
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
            return

        n_avg   = float(np.mean([e.summary.n         for e in valid_entries]))
        rho_avg = float(np.mean([e.summary.density() for e in valid_entries]))
        n_nb    = node.noise_buffer.n
        rho_nb  = node.noise_buffer.density()

        # Step 3: graduate if test passes
        if noise_to_cluster_test(n_nb, rho_nb, n_avg, rho_avg):
            self._noise_to_cluster(node, t, events)

    def _noise_to_cluster(self, node: Node, t: int, events: list) -> None:
        """Graduate the noise buffer into a new cluster entry (Sec. 2.3).

        Uses _spawn_liar_root so the new child always has is_leaf=False —
        _liar_proc sets is_leaf=True only once the subtree reaches the regular
        leaf level, keeping get_summary() / _avg_leaf_radius() consistent.
        """
        cf = node.noise_buffer.copy()

        new_entry = Entry(
            summary=cf,
            buffer=MicroCluster.empty(self._dim, t),
            child=None,
            entry_id=self._next_id,
        )
        self._next_id += 1

        liar_child = self._spawn_liar_root(new_entry, t)
        liar_child.parent = node

        node.entries.append(new_entry)
        node.noise_buffer = None
        events.append("noise_to_cluster")

        if len(node.entries) > self.max_fanout:
            self._split(node, events)

    # =========================================================================
    # Liar-node growth — Algorithm 2 (Sec. 2.4 — Anton)
    # =========================================================================

    def _spawn_liar_root(self, entry, t):
        """Create an empty liar-root child node under *entry*.

        Called by _noise_to_cluster after it builds the new entry from the
        noise buffer.  The caller is responsible for setting node.parent to
        the node that contains *entry*.

        Returns the new Node so the caller can set .parent on it.
        """
        node = Node(is_leaf=False, parent_entry=entry, timestamp=t)
        node.is_liar = True
        node.is_liar_root = True
        entry.child = node
        return node

    def _liar_proc(self, liar_node, x_cf, t, events):
        """Grow the liar subtree top-down by one level (Algorithm 2, Sec. 2.4).

        Parameters
        ----------
        liar_node : Node
            The empty liar node at the current bottom of the growing subtree.
            Must have is_liar == True.
        x_cf : MicroCluster
            CF of the incoming object.
        t : float
            Current time stamp.
        events : list[str]
            Mutable event log; appended in-place.

        Returns
        -------
        int
            ID of the assigned leaf entry, or -1 if the object landed in an
            inner liar entry (it will surface once that entry is promoted).
        """
        # ------------------------------------------------------------------
        # Step 1: decay the parent entry and build the 3-way CF partition
        # (Alg. 2, lines 1-9; formula from Sec. 2.4 / paper p. 410)
        # ------------------------------------------------------------------
        eparent = liar_node.parent_entry
        eparent.summary.decay(t, self.lam)

        n = eparent.summary.n
        if n <= 0.0:
            # Degenerate parent (decayed to zero): 3.0/n would raise
            # ZeroDivisionError.  Skip; the object is lost but the tree stays
            # consistent.
            events.append("liar_skip")
            return -1
        L   = eparent.summary.ls.copy()
        var = eparent.summary.variance()   # clamped ≥ 0
        sig = np.sqrt(var)

        offB = np.sqrt((1.0 / 6.0) * (1.0 - (1.0 / 3.0) ** 4) * n * var)
        F    = (n / 3.0) * (sig / 3.0) ** 4   # per-dim

        # Modulo-rotation: dimension d gets offset pattern (A, B, C) shifted
        # by d positions so the three children are not collinear (Alg. 2, l. 3-5)
        offsets  = [np.zeros_like(L), offB, -offB]   # A=0, B=+offB, C=-offB
        ls_parts = [np.zeros_like(L) for _ in range(3)]
        for d in range(L.size):
            ls_parts[d % 3][d]       += L[d] / 3.0 + offsets[0][d]
            ls_parts[(d + 1) % 3][d] += L[d] / 3.0 + offsets[1][d]
            ls_parts[(d + 2) % 3][d] += L[d] / 3.0 + offsets[2][d]

        child_cfs = []
        for k in range(3):
            ss_k = F + (3.0 / n) * ls_parts[k] ** 2
            child_cfs.append(
                MicroCluster(n / 3.0, ls_parts[k].copy(), ss_k.copy(),
                             eparent.summary.timestamp)
            )

        # ------------------------------------------------------------------
        # Step 2: insert x_cf into the closest of the three child CFs
        # (Alg. 2, line 10)
        # ------------------------------------------------------------------
        dists     = [self._dist(x_cf.mean(), cf.mean()) for cf in child_cfs]
        closest_k = int(np.argmin(dists))
        child_cfs[closest_k].add(x_cf)

        # ------------------------------------------------------------------
        # Determine if the new entries land at the regular leaf level.
        #
        # level(node) = len(_node_path_to(node))   [1-indexed, root = 1]
        # leaf level  = self._height
        #
        # is_liar_root case:  entries go INTO liar_node (level liar_depth) →
        #   leaf iff liar_depth == _height
        # non-liar_root case: entries REPLACE eparent in liar_node.parent
        #   (level liar_depth-1) → leaf iff liar_depth - 1 == _height
        #   ↔ liar_depth == _height + 1
        # ------------------------------------------------------------------
        is_liar_root = liar_node.is_liar_root
        liar_depth   = len(self._node_path_to(liar_node))   # 1-indexed level

        if is_liar_root:
            at_leaf_level = (liar_depth == self._height)
        else:
            at_leaf_level = (liar_depth == self._height + 1)

        # Find the liar root (needed for flag cleanup and split boundary)
        cur = liar_node
        while cur is not None and not cur.is_liar_root:
            cur = cur.parent
        liar_root = cur   # may be liar_node itself when is_liar_root

        if liar_root is None:
            # Stale liar flag: the liar_root was cleared (e.g. by a previous
            # _clear_liar_flags call that used a stale reference). Clear the
            # flag on this node and skip — the point is safely discarded.
            liar_node.is_liar = False
            events.append("liar_skip")
            return -1

        # ------------------------------------------------------------------
        # Step 3: build Entry objects
        # 
        # (Alg. 2, lines 11-17 distinguish leaf from inner)
        # ------------------------------------------------------------------
        if at_leaf_level:
            new_entries = [self._new_leaf_entry(cf) for cf in child_cfs]
            assigned_id = new_entries[closest_k].id
        else:
            new_entries = [
                Entry(summary=cf,
                      buffer=MicroCluster.empty(self._dim, t),
                      child=None)
                for cf in child_cfs
            ]
            assigned_id = -1

        # ------------------------------------------------------------------
        # Step 5 (flag block, part 1): spawn liar children BEFORE the split.
        # _split_liar updates e.child.parent for entries that move to the
        # sibling node, so spawning first keeps pointers correct.
        # (Alg. 2, lines 18-22)
        # ------------------------------------------------------------------
        if not at_leaf_level:
            host_node_pre = liar_node if is_liar_root else liar_node.parent
            for e in new_entries:
                liar_child = Node(is_leaf=False, parent=host_node_pre,
                                  parent_entry=e, timestamp=t)
                liar_child.is_liar = True
                e.child = liar_child

        # ------------------------------------------------------------------
        # Step 4: attach new entries to the tree
        # (Alg. 2, lines 11-17)
        # ------------------------------------------------------------------
        if is_liar_root:
            # liar_node was empty; fill it directly (Fig. 2c)
            liar_node.is_liar = False
            for e in new_entries:
                liar_node.entries.append(e)
            host_node = liar_node

            # Set is_leaf BEFORE _split_liar so the sibling (created with
            # Node(is_leaf=node.is_leaf)) inherits the correct flag.
            if at_leaf_level:
                liar_node.is_leaf = True

            if len(liar_node.entries) > self.max_fanout:
                self._split_liar(liar_node, events, liar_root)
        else:
            # Replace eparent in parent with 3 new entries (Fig. 2d)
            parent = liar_node.parent
            parent.entries.remove(eparent)
            for e in new_entries:
                parent.entries.append(e)
            host_node = parent
            liar_node.is_liar = False

            # Same: set is_leaf before _split_liar.
            if at_leaf_level:
                parent.is_leaf = True

            if len(parent.entries) > self.max_fanout:
                self._split_liar(parent, events, liar_root)

        # ------------------------------------------------------------------
        # Step 6 (flag block, part 2): leaf-level promotion
        # ------------------------------------------------------------------
        if at_leaf_level:
            # host_node.is_leaf was already set above (before the split so
            # any sibling inherits the correct flag).
            # Re-find the actual liar_root: _split_liar may have promoted a
            # new_root above the one we found at the top of this call.
            cur = host_node
            while cur is not None and not cur.is_liar_root:
                cur = cur.parent
            if cur is not None:
                self._clear_liar_flags(cur)
            events.append("liar_promote")
        else:
            events.append("liar_grow")

        return assigned_id

    def _split_liar(self, node, events, liar_root):
        """Split an overfull liar-subtree node, stopping at liar_root.

        When liar_root itself overflows a new liar_root is created above it
        (subtree height grows by 1 top-down, Fig. 2e).  Multiple overflows in
        one call are resolved recursively.
        """
        groupA, groupB = self._best_split(node.entries)
        t       = node.timestamp
        sibling = Node(is_leaf=node.is_leaf, parent=node.parent, timestamp=t)
        node.entries    = groupA
        sibling.entries = groupB
        for e in groupB:
            if e.child is not None:
                e.child.parent       = sibling
                e.child.parent_entry = e

        sumA = self._summary_of(groupA, t)
        sumB = self._summary_of(groupB, t)
        pe   = node.parent_entry

        if node is liar_root:
            # Liar-root overflow: grow the liar subtree upward by one level
            new_root = Node(is_leaf=False, parent=node.parent, timestamp=t)
            new_root.is_liar_root = True
            node.is_liar_root     = False

            eA = Entry(sumA, MicroCluster.empty(self._dim, t), node,    None)
            eB = Entry(sumB, MicroCluster.empty(self._dim, t), sibling, None)
            new_root.entries = [eA, eB]

            node.parent          = new_root
            node.parent_entry    = eA
            sibling.parent       = new_root
            sibling.parent_entry = eB

            if pe is not None:
                pe.child              = new_root
                new_root.parent_entry = pe
                new_sum = self._summary_of([eA, eB], t)
                new_sum.add(pe.buffer)
                pe.summary = new_sum
            else:
                self.root      = new_root
                self._height  += 1

            events.append("liar_root_split")
            return

        # Regular liar-subtree split: propagate up to (but not past) liar_root
        if node.parent is None:
            # node has no parent — it escaped the liar subtree boundary (should
            # not happen with correct liar_root tracking, but guard defensively).
            events.append("liar_split_escaped")
            return
        if pe is not None:
            sumA.add(pe.buffer)
            pe.summary = sumA
        eB = Entry(sumB, MicroCluster.empty(self._dim, t), sibling, None)
        sibling.parent_entry = eB
        node.parent.entries.append(eB)
        events.append("split")

        if len(node.parent.entries) > self.max_fanout:
            self._split_liar(node.parent, events, liar_root)

    def _clear_liar_flags(self, node):
        """Recursively clear is_liar / is_liar_root from an entire liar subtree."""
        node.is_liar      = False
        node.is_liar_root = False
        for e in node.entries:
            if e.child is not None:
                self._clear_liar_flags(e.child)
