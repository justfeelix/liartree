"""Overlap look-ahead logic for the LiarTree (Task 4: Paul).

Mixin class — import into LiarTree via multiple inheritance.
All methods reference self.* attributes that come from ClusTree
(self._dist, self._time_is_up, self.lam, self.horizon, self._dim, etc.).

Implements Section 2.2 / Figure 1: when the nearest and second-nearest entries
overlap, peek one level into both, optionally swap their closest child entries
to reduce radii, then return the best candidate.

Also owns the enc_split flag logic: an irrelevant (empty or too-old) entry in
a node signals that a split is encouraged in the subtree below it, enabling the
noise-to-cluster check further down the descent (Sec. 2.2, Def. 4).
"""
from __future__ import annotations

import numpy as np


class OverlapHandlerMixin:

    # -------------------------------------------------------------------------
    # enc_split flag — set when an irrelevant entry is found during descent
    # -------------------------------------------------------------------------

    def _mark_enc_split_from_irrelevant(self, node) -> None:
        """Set self._enc_split = True if any entry in node is irrelevant.

        An irrelevant entry is one that is empty or whose weight has decayed
        below the one-point-per-snapshot threshold (cf. ClusTree Sec. 3.4).
        A single such entry anywhere on the descent path suffices to encourage
        splits below (Sec. 2.2).  Once True the flag stays True for the rest
        of the insertion.
        """
        if any(self._is_irrelevant_entry(e) for e in node.entries):
            self._enc_split = True

    def _is_irrelevant_entry(self, entry) -> bool:
        if entry.summary.is_empty():
            return True
        threshold = 2.0 ** (-self.lam * self.horizon)
        return entry.summary.n < threshold

    # -------------------------------------------------------------------------
    # Distance ranking — shared by overlap look-ahead and _calc_closest_entry
    # -------------------------------------------------------------------------

    def _rank_entries_by_distance(self, node, cf) -> list:
        """Return [(distance, entry), ...] sorted closest-first, skipping empty.

        Stores the result in self._last_ranked so that callers can reuse the
        computed distances (e.g. for noise probability) without extra _dist calls.
        """
        target = cf.mean()
        ranked = []
        for entry in node.entries:
            if entry.summary.is_empty():
                continue
            ranked.append((self._dist(target, entry.summary.mean()), entry))
        ranked.sort(key=lambda item: item[0])
        self._last_ranked = ranked
        return ranked

    # -------------------------------------------------------------------------
    # Overlap look-ahead (Sec. 2.2 / Fig. 1)
    # -------------------------------------------------------------------------

    def _select_with_overlap_reduction(self, ranked: list, cf) -> object:
        """Return the best entry after optional child-swap overlap reduction.

        If the two closest entries overlap AND both have non-empty children,
        we peek one level deeper, optionally swap the closest child entries
        between the two sibling subtrees to shrink both radii, then return
        the entry whose closest child is nearest to cf.
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

    # ---- overlap helpers ----------------------------------------------------

    def _entries_overlap(self, first, second) -> bool:
        dist = self._dist(first.summary.mean(), second.summary.mean())
        return dist < first.summary.radius() + second.summary.radius()

    @staticmethod
    def _can_look_ahead(entry) -> bool:
        child = entry.child
        if child is None or not child.entries:
            return False
        return any(not e.summary.is_empty() for e in child.entries)

    def _closest_child_entry(self, parent_entry, cf):
        target = cf.mean()
        best, best_dist = None, np.inf
        for child_entry in parent_entry.child.entries:
            if child_entry.summary.is_empty():
                continue
            if self._time_is_up():
                break
            d = self._dist(target, child_entry.summary.mean())
            if d < best_dist:
                best, best_dist = child_entry, d
        return best, best_dist

    def _maybe_swap_overlapping_children(self, first_parent, first_child,
                                          second_parent, second_child) -> bool:
        """Swap first_child and second_child if it reduces both parent radii.

        Returns True if the swap was performed.
        """
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

        self._swap_child_entries(first_parent, first_child,
                                 second_parent, second_child)
        return True

    def _simulated_parent_summary(self, parent_entry, replace_entry, replacement):
        entries = [
            replacement if e is replace_entry else e
            for e in parent_entry.child.entries
        ]
        return self._summary_with_buffer(entries, parent_entry.buffer,
                                         parent_entry.child.timestamp)

    # ---- swap with rollback -------------------------------------------------

    def _swap_child_entries(self, first_parent, first_child,
                            second_parent, second_child) -> None:
        first_node = first_parent.child
        second_node = second_parent.child
        snapshot = self._snapshot_overlap_swap(
            first_parent, second_parent, first_node, second_node,
            first_child, second_child
        )
        try:
            fi = first_node.entries.index(first_child)
            si = second_node.entries.index(second_child)
            first_node.entries[fi] = second_child
            second_node.entries[si] = first_child
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
            self._restore_overlap_swap(snapshot)
            raise

    @staticmethod
    def _retarget_moved_child(entry, new_parent) -> None:
        if entry.child is not None:
            entry.child.parent = new_parent
            entry.child.parent_entry = entry

    # ---- summary helpers ----------------------------------------------------

    def _summary_with_buffer(self, entries, buffer, timestamp):
        from .microcluster import MicroCluster
        dim = self._summary_dim(entries, buffer)
        s = MicroCluster.empty(dim, timestamp)
        for e in entries:
            s.add(e.summary)
        s.timestamp = timestamp
        if buffer is not None:
            s.add(buffer)
        return s

    def _summary_dim(self, entries, buffer) -> int:
        if self._dim is not None:
            return self._dim
        for e in entries:
            if e.summary.ls.size:
                return e.summary.ls.size
        if buffer is not None:
            return buffer.ls.size
        return 0

    # ---- snapshot / restore -------------------------------------------------

    @staticmethod
    def _snapshot_overlap_swap(first_parent, second_parent,
                               first_node, second_node,
                               first_child, second_child) -> dict:
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
    def _restore_overlap_swap(snapshot) -> None:
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
        """Test hook: called after the swap, before summaries are updated."""
        return None
