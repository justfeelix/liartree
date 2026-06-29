"""LiarTree online phase extensions.

This class builds on :class:`ClusTree` and adds the overlap-reduction look-ahead
used during entry selection. Noise handling and liar-node growth are still
separate extension points; when those steps land, they can compose with
``_select_with_overlap_reduction`` from ``_calc_closest_entry``.
"""
from __future__ import annotations

import numpy as np

from .clustree import ClusTree
from .microcluster import MicroCluster


class LiarTree(ClusTree):
    def __init__(self, *args, noise_threshold=None, overlap_reduction=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_threshold = noise_threshold
        self.overlap_reduction = overlap_reduction

    # TODO: override learn_one to follow Algorithm 1 [Sec. 2.1]
    # TODO: add noise marking around this selection hook [Sec. 2.2, Def. 2, Fig. 1]
    # TODO: noise_proc, noise buffer -> new cluster via the gompertz test [Sec. 2.3, Def. 3-4]
    # TODO: liar_proc, grow the subtree top-down [Alg. 2, Sec. 2.4]
    # TODO: override _leaf_proc to allow encouraged splits [Sec. 2.5]

    def _calc_closest_entry(self, node, cf):
        """Select an entry, using overlap reduction for non-leaf routing."""
        if not self.overlap_reduction:
            return super()._calc_closest_entry(node, cf)
        return self._select_with_overlap_reduction(node, cf)

    def _select_with_overlap_reduction(self, node, cf):
        """Route to the closest entry, optionally reorganizing overlapping
        sibling subtrees by swapping their closest child entries.

        The method deliberately stays local: only the two overlapping sibling
        subtrees can be mutated, and all object-to-entry/object-to-child
        distances go through ``_dist`` so the anytime budget still accounts for
        the extra look-ahead work.
        """
        ranked = self._rank_entries_by_distance(node, cf)
        if not ranked:
            return None

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

        self._swap_child_entries(first_parent, first_child, second_parent, second_child)
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
