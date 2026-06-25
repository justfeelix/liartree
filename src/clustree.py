"""ClusTree online phase.

The anytime, self-adapting micro-cluster index from the paper. Insertion and
the four descent strategies (Sec. 3.4) live here; the data structures are in
microcluster.py and node.py. Written so the LiarTree can subclass it and only
override the descent / leaf / split hooks (see liartree.py).

Only depth-first descent is wired up so far.
"""
from __future__ import annotations

import numpy as np

import stream_clustering   # baseframework, made importable in src/__init__
import utils               # baseframework euclidean distance

from .node import Node, Entry
from .microcluster import MicroCluster


class ClusTree(stream_clustering.OnlinePhase_StreamClusterer):
    """Anytime stream clustering over a hierarchy of micro-clusters.

    Parameters
    ----------
    max_fanout, min_fanout : M / m for inner nodes (M=3 is best, Sec. 4.1).
    leaf_capacity_max, leaf_capacity_min : L / l for leaf nodes.
    lam : decay rate of omega(t) = 2^(-lambda*t) (Sec. 3.2).
    horizon : snapshot interval t_snap for the discard rule (Eq. 1).
    descent : "depthfirst" | "breadthfirst" | "bestfirst" | "iterativedf".
    budget : max distance computations per object (low = fast stream). The
        LiarTree uses the same unit, so the two stay comparable.
    """

    def __init__(self, max_fanout=3, min_fanout=1, leaf_capacity_max=3,
                 leaf_capacity_min=1, lam=0.001, horizon=1000,
                 descent="depthfirst", budget=36):
        if descent != "depthfirst":
            raise NotImplementedError("only depth-first descent is wired up so far")
        self.max_fanout = max_fanout
        self.min_fanout = min_fanout
        self.leaf_max = leaf_capacity_max
        self.leaf_min = leaf_capacity_min
        self.lam = lam
        self.horizon = horizon
        self.descent = descent
        self.budget = budget

        self.root = Node(is_leaf=True, timestamp=0)
        self.merge_list = []
        self._time = 0
        self._next_id = 0
        self._dim = None
        self._dist_used = 0

    def learn_one(self, x, t=None):
        """Insert one object (Alg. Fig. 4). Returns (summary_id, cluster_id,
        event_log) as the baseframework expects."""
        if t is None:
            t = self._time + 1
        self._time = t
        self._dist_used = 0

        x = np.asarray(x, dtype=float)
        if self._dim is None:
            self._dim = x.size
        x_cf = MicroCluster.from_point(x, t)

        node = self.root
        h = None            # hitchhiker CF travelling with x
        path = []           # entries chosen on the way down (for the discard rule)
        events = []
        assigned = -1

        while True:
            self._on_descend(node, t)

            if node.is_leaf:
                assigned = self._leaf_proc(node, x_cf, h, path, events)
                break

            ex = self._calc_closest_entry(node, x_cf)
            if ex is None:
                break

            if h is not None:
                eh = self._calc_closest_entry(node, h)
                if eh is ex:
                    ex.summary.add(h)               # hitchhiker keeps riding with x
                else:
                    self._deposit(eh, h)            # drop it in its own entry's buffer
                    h = None

            ex.summary.add(x_cf)
            path.append(ex)

            if self._time_is_up():
                ex.buffer.add(x_cf)
                if h is not None:
                    ex.buffer.add(h)
                events.append("buffered")
                break

            picked = ex.buffer                      # take this entry's buffer along
            if not picked.is_empty():
                if h is None:
                    h = picked.copy()
                else:
                    h.add(picked)
            ex.buffer = MicroCluster.empty(self._dim, t)
            node = ex.child

        return assigned, -1, ";".join(events)

    def get_summary(self):
        """Leaf-level micro-clustering as {id: {position, radius, weight}}."""
        summary = {}
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                for e in node.entries:
                    if not e.summary.is_empty():
                        # weight decayed to the current time so all entries are
                        # reported as one snapshot (mean/radius are decay-invariant)
                        w = e.summary.n * 2.0 ** (-self.lam * (self._time - e.summary.timestamp))
                        summary[e.id] = {
                            "position": e.summary.mean(),
                            "radius": e.summary.radius(),
                            "weight": w,
                        }
            else:
                for e in node.entries:
                    if e.child is not None:
                        stack.append(e.child)
        return summary

    # --- budget and decay ------------------------------------------------
    def _dist(self, a, b):
        """Euclidean distance that counts against the per-object budget. Every
        distance during descent goes through here, so ClusTree and LiarTree
        share the same budget unit."""
        self._dist_used += 1
        return utils.distance_calc(a, b)

    def _time_is_up(self):
        return self._dist_used >= self.budget

    def _on_descend(self, node, t):
        """Decay every entry summary and buffer in node to time t (Sec. 3.2)."""
        for e in node.entries:
            e.summary.decay(t, self.lam)
            if e.buffer is not None:
                e.buffer.decay(t, self.lam)
        node.timestamp = t

    # --- hooks the LiarTree overrides ------------------------------------
    def _calc_closest_entry(self, node, cf):
        """Entry of node whose mean is closest to cf (Sec. 3.1)."""
        target = cf.mean()
        best, best_d = None, np.inf
        for e in node.entries:
            if e.summary.is_empty():
                continue
            d = self._dist(target, e.summary.mean())
            if d < best_d:
                best_d, best = d, e
        return best

    def _leaf_proc(self, node, x_cf, h, path, events):
        """Insert x (and the hitchhiker) as leaf entries, then resolve overflow
        by discarding, splitting, or merging."""
        entry = self._new_leaf_entry(x_cf)
        node.entries.append(entry)
        if h is not None:
            node.entries.append(self._new_leaf_entry(h))

        while len(node.entries) > self.leaf_max:
            if self._discard_insignificant(node, path, events):
                continue
            if not self._time_is_up():
                self._split(node, events)
                break
            self._merge_closest(node, events)
        return entry.id

    def _split(self, node, events):
        """Split an overfull node into the two min-distance groups and propagate
        up; a new root is grown when the root itself splits."""
        groupA, groupB = self._best_split(node.entries)
        t = node.timestamp
        sibling = Node(is_leaf=node.is_leaf, parent=node.parent, timestamp=t)
        node.entries = groupA
        sibling.entries = groupB
        for e in groupB:
            if e.child is not None:
                e.child.parent = sibling

        sumA = self._summary_of(groupA, t)
        sumB = self._summary_of(groupB, t)
        pe = node.parent_entry

        if pe is None:
            new_root = Node(is_leaf=False, timestamp=t)
            eA = Entry(sumA, MicroCluster.empty(self._dim, t), node, None)
            eB = Entry(sumB, MicroCluster.empty(self._dim, t), sibling, None)
            node.parent, node.parent_entry = new_root, eA
            sibling.parent, sibling.parent_entry = new_root, eB
            new_root.entries = [eA, eB]
            self.root = new_root
            events.append("split_root")
            return

        sumA.add(pe.buffer)        # pe.CF must still include its own buffer (invariant)
        pe.summary = sumA
        eB = Entry(sumB, MicroCluster.empty(self._dim, t), sibling, None)
        sibling.parent_entry = eB
        parent = node.parent
        parent.entries.append(eB)
        events.append("split")
        if len(parent.entries) > self.max_fanout:
            self._split(parent, events)

    # --- helpers ---------------------------------------------------------
    def _new_leaf_entry(self, cf):
        e = Entry(summary=cf, entry_id=self._next_id)
        self._next_id += 1
        return e

    def _deposit(self, entry, cf):
        entry.summary.add(cf)
        entry.buffer.add(cf)

    def _summary_of(self, entries, t):
        s = MicroCluster.empty(self._dim, t)
        for e in entries:
            s.add(e.summary)
        s.timestamp = t
        return s

    def _best_split(self, entries):
        means = [e.summary.mean() for e in entries]
        n = len(entries)
        best, best_cost = None, np.inf
        for mask in range(1, 1 << (n - 1)):          # fix entry 0 in group A
            a = [0] + [i for i in range(1, n) if mask & (1 << (i - 1))]
            b = [i for i in range(1, n) if not (mask & (1 << (i - 1)))]
            if not b:
                continue
            cost = self._intra(a, means) + self._intra(b, means)
            if cost < best_cost:
                best_cost, best = cost, (a, b)
        a, b = best
        return [entries[i] for i in a], [entries[i] for i in b]

    def _intra(self, idx, means):
        cost = 0.0
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                cost += utils.distance_calc(means[idx[i]], means[idx[j]])
        return cost

    def _merge_closest(self, node, events):
        entries = node.entries
        bi, bj, bd = 0, 1, np.inf
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                d = utils.distance_calc(entries[i].summary.mean(),
                                        entries[j].summary.mean())
                if d < bd:
                    bd, bi, bj = d, i, j
        keep, drop = entries[bi], entries[bj]
        keep.summary.add(drop.summary)
        if keep.id is not None and drop.id is not None:
            self.merge_list.append((keep.id, drop.id))
        entries.pop(bj)
        events.append("merge")

    def _discard_insignificant(self, node, path, events):
        threshold = 2.0 ** (-self.lam * self.horizon)
        cand = min(node.entries, key=lambda e: e.summary.n)
        if cand.summary.n >= threshold:
            return False
        node.entries.remove(cand)
        for e in path:
            e.summary.subtract(cand.summary)
        events.append("discard")
        return True
