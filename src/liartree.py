"""LiarTree online phase extensions.

Extends ClusTree with per-node noise buffers, Gompertz-based
noise-to-cluster events, and local overlap-reduction look-ahead during entry
selection.
"""
from __future__ import annotations

import numpy as np

from .clustree import ClusTree
from .microcluster import MicroCluster
from .node import Entry, Node
from .noise_handler import noise_to_cluster_test


class LiarTree(ClusTree):
    """ClusTree extended with noise buffers and overlap-aware routing.

    Parameters
    ----------
    noise_threshold : float in [0, 1] or None
        A point is classified as noise when its noise probability is greater
        than or equal to this threshold. Use None to disable noise gating.
    overlap_reduction : bool
        Enable the local overlap look-ahead and child-entry swap.
    """

    def __init__(self, *args, noise_threshold: float | None = 0.5,
                 overlap_reduction: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_threshold = noise_threshold
        self.overlap_reduction = overlap_reduction
        self._enc_split = False
        self._last_cf_is_noise = False
        self._last_noise_probability = 0.0

    # Main LiarTree addition to ClusTree: descent override with liar and noise.
    def _insert_depth_first(self, x_cf: MicroCluster, t: int):
        node = self.root
        h = None
        path = []
        events = []
        assigned = -1
        self._enc_split = False

        while True:
            self._on_descend(node, t)

            if node.is_liar:
                self._liar_proc(node, x_cf, t)
                break

            if node.is_leaf:
                assigned = self._leaf_proc(node, x_cf, h, path, events)
                break

            ex = self._calc_closest_entry(node, x_cf)
            if ex is None:
                break

            if self._last_cf_is_noise:
                self._noise_proc(
                    x_cf, node, t, enc_split=self._enc_split, events=events
                )
                events.append("noise")
                break

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

    # Task 3: liar procedure placeholder.
    def _liar_proc(self, node: Node, x_cf: MicroCluster, t: int) -> None:
        """Grow the liar subtree top-down by one level."""
        pass

    # Task 4: overlap-aware entry selection.
    def _calc_closest_entry(self, node: Node, cf: MicroCluster):
        """Select the closest entry, routing non-noise CFs through look-ahead."""
        self._last_cf_is_noise = False
        self._last_noise_probability = 0.0
        ranked = self._rank_entries_by_distance(node, cf)
        if not ranked:
            return None

        self._mark_enc_split_from_irrelevant(node)
        closest = ranked[0][1]

        if self.noise_threshold is not None:
            self._last_noise_probability = self._noise_probability_from_ranked(ranked)
            self._last_cf_is_noise = (
                self._last_noise_probability >= self.noise_threshold
            )
            if self._last_cf_is_noise:
                return closest

        if not self.overlap_reduction:
            return closest
        return self._select_with_overlap_reduction_from_ranked(ranked, cf)

    def _leaf_proc(self, node: Node, x_cf: MicroCluster, h, path, events,
                   can_split=None):
        """Leaf insertion with LiarTree's encouraged split flag."""
        if can_split is None and self._enc_split:
            can_split = True
        return super()._leaf_proc(node, x_cf, h, path, events, can_split)

    def _select_with_overlap_reduction_from_ranked(self, ranked, cf):
        """Route to the closest entry, optionally reorganizing overlapping
        sibling subtrees by swapping their closest child entries.

        The method deliberately stays local: only the two overlapping sibling
        subtrees can be mutated, and all object-to-entry/object-to-child
        distances go through ``_dist`` so the anytime budget still accounts for
        the extra look-ahead work.
        """
        closest = ranked[0][1]
        if len(ranked) < 2 or self._time_is_up():
            return closest

        second = ranked[1][1]
        if not self._entries_overlap(closest, second) or self._time_is_up():
            return closest

        if not (self._can_look_ahead(closest) and self._can_look_ahead(second)):
            return closest

        self._on_descend(closest.child, self._time)
        self._on_descend(second.child, self._time)
        if self._time_is_up():
            return closest

        first_child, first_child_dist = self._closest_child_entry(closest, cf)
        second_child, second_child_dist = self._closest_child_entry(second, cf)
        if first_child is None or second_child is None:
            return closest

        swapped = self._maybe_swap_overlapping_children(
            closest, first_child, second, second_child
        )
        if first_child_dist <= second_child_dist:
            return second if swapped else closest
        return closest if swapped else second

    def _rank_entries_by_distance(self, node, cf):
        target = cf.mean()
        ranked = []
        for entry in node.entries:
            if entry.summary.is_empty():
                continue
            ranked.append((self._dist(target, entry.summary.mean()), entry))
        ranked.sort(key=lambda item: item[0])
        return ranked

    def _noise_probability_from_ranked(self, ranked):
        min_ratio = np.inf
        for dist, entry in ranked:
            radius = entry.summary.radius()
            if radius == 0.0:
                continue
            min_ratio = min(min_ratio, dist / radius)
        if not np.isfinite(min_ratio):
            return 1.0
        return float(min(1.0, min_ratio))

    def _mark_enc_split_from_irrelevant(self, node):
        if any(self._is_irrelevant_entry(entry) for entry in node.entries):
            self._enc_split = True

    def _is_irrelevant_entry(self, entry):
        if entry.summary.is_empty():
            return True
        threshold = 2.0 ** (-self.lam * self.horizon)
        return entry.summary.n < threshold

    def _entries_overlap(self, first, second):
        center_distance = self._dist(first.summary.mean(), second.summary.mean())
        return center_distance < first.summary.radius() + second.summary.radius()

    @staticmethod
    def _can_look_ahead(entry):
        child = entry.child
        if child is None or not child.entries:
            return False
        return any(not child_entry.summary.is_empty() for child_entry in child.entries)

    def _closest_child_entry(self, parent_entry, cf):
        target = cf.mean()
        best, best_dist = None, np.inf
        for child_entry in parent_entry.child.entries:
            if child_entry.summary.is_empty():
                continue
            if self._time_is_up():
                break
            dist = self._dist(target, child_entry.summary.mean())
            if dist < best_dist:
                best, best_dist = child_entry, dist
        return best, best_dist

    def _maybe_swap_overlapping_children(self, first_parent, first_child,
                                         second_parent, second_child):
        if first_parent.child is second_parent.child:
            return False

        first_new = self._simulated_parent_summary(
            first_parent, replace_entry=first_child, replacement=second_child
        )
        second_new = self._simulated_parent_summary(
            second_parent, replace_entry=second_child, replacement=first_child
        )

        if not (first_new.radius() < first_parent.summary.radius() and
                second_new.radius() < second_parent.summary.radius()):
            return False

        self._swap_child_entries(first_parent, first_child, second_parent,
                                 second_child)
        return True

    def _simulated_parent_summary(self, parent_entry, replace_entry, replacement):
        entries = [
            replacement if entry is replace_entry else entry
            for entry in parent_entry.child.entries
        ]
        return self._summary_with_buffer(entries, parent_entry.buffer,
                                         parent_entry.child.timestamp)

    def _swap_child_entries(self, first_parent, first_child,
                            second_parent, second_child):
        first_node = first_parent.child
        second_node = second_parent.child
        snapshot = self._snapshot_overlap_swap(
            first_parent, second_parent, first_node, second_node,
            first_child, second_child
        )

        try:
            first_index = first_node.entries.index(first_child)
            second_index = second_node.entries.index(second_child)
            first_node.entries[first_index] = second_child
            second_node.entries[second_index] = first_child
            self._retarget_moved_child(second_child, first_node)
            self._retarget_moved_child(first_child, second_node)

            self._after_overlap_swap_mutation(
                first_parent, first_child, second_parent, second_child
            )

            first_parent.summary = self._summary_with_buffer(
                first_node.entries, first_parent.buffer, first_node.timestamp
            )
            second_parent.summary = self._summary_with_buffer(
                second_node.entries, second_parent.buffer, second_node.timestamp
            )
        except Exception:
            # Deliberate hardening beyond the paper: the paper performs this
            # swap atomically inside calcClosestEntry, while this Python version
            # restores the local tree if an unexpected mid-swap failure occurs.
            self._restore_overlap_swap(snapshot)
            raise

    @staticmethod
    def _retarget_moved_child(entry, new_parent):
        if entry.child is not None:
            entry.child.parent = new_parent
            entry.child.parent_entry = entry

    def _summary_with_buffer(self, entries, buffer, timestamp):
        dim = self._summary_dim(entries, buffer)
        summary = MicroCluster.empty(dim, timestamp)
        for entry in entries:
            summary.add(entry.summary)
        summary.timestamp = timestamp
        if buffer is not None:
            summary.add(buffer)
        return summary

    def _summary_dim(self, entries, buffer):
        if self._dim is not None:
            return self._dim
        for entry in entries:
            if entry.summary.ls.size:
                return entry.summary.ls.size
        if buffer is not None:
            return buffer.ls.size
        return 0

    @staticmethod
    def _snapshot_overlap_swap(first_parent, second_parent, first_node,
                               second_node, first_child, second_child):
        moved_nodes = []
        for entry in (first_child, second_child):
            if entry.child is not None:
                moved_nodes.append(
                    (entry.child, entry.child.parent, entry.child.parent_entry)
                )
        return {
            "first_parent": first_parent,
            "second_parent": second_parent,
            "first_summary": first_parent.summary.copy(),
            "second_summary": second_parent.summary.copy(),
            "first_child_pointer": first_parent.child,
            "second_child_pointer": second_parent.child,
            "first_node": first_node,
            "second_node": second_node,
            "first_entries": list(first_node.entries),
            "second_entries": list(second_node.entries),
            "moved_nodes": moved_nodes,
        }

    @staticmethod
    def _restore_overlap_swap(snapshot):
        snapshot["first_parent"].summary = snapshot["first_summary"]
        snapshot["second_parent"].summary = snapshot["second_summary"]
        snapshot["first_parent"].child = snapshot["first_child_pointer"]
        snapshot["second_parent"].child = snapshot["second_child_pointer"]
        snapshot["first_node"].entries = snapshot["first_entries"]
        snapshot["second_node"].entries = snapshot["second_entries"]
        for node, parent, parent_entry in snapshot["moved_nodes"]:
            node.parent = parent
            node.parent_entry = parent_entry

    def _after_overlap_swap_mutation(self, first_parent, first_child,
                                     second_parent, second_child):
        """Test hook for verifying rollback after a mid-swap failure."""
        return None

    # Task 2: noise procedure.
    def _noise_proc(self, x_cf: MicroCluster, node: Node, t: int,
                    enc_split: bool = True, events: list | None = None) -> None:
        """Handle a noise-classified point."""
        if events is None:
            events = []

        if node.noise_buffer is None:
            node.noise_buffer = MicroCluster.empty(self._dim, t)
        node.noise_buffer.decay(t, self.lam)
        node.noise_buffer.add(x_cf)

        if not enc_split:
            return

        valid_entries = [e for e in node.entries if not e.summary.is_empty()]
        if not valid_entries:
            return

        n_avg = float(np.mean([e.summary.n for e in valid_entries]))
        rho_avg = float(np.mean([e.summary.density() for e in valid_entries]))
        n_nb = node.noise_buffer.n
        rho_nb = node.noise_buffer.density()

        if noise_to_cluster_test(n_nb, rho_nb, n_avg, rho_avg):
            self._noise_to_cluster(node, t, events)

    def _noise_to_cluster(self, node: Node, t: int, events: list) -> None:
        """Graduate the noise buffer into a new cluster entry."""
        cf = node.noise_buffer.copy()

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
        node.noise_buffer = None
        events.append("noise_to_cluster")

        if len(node.entries) > self.max_fanout:
            self._split(node, events)
