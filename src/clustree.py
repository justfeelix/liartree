"""ClusTree online phase.

The anytime, self-adapting micro-cluster index from the paper. Insertion and
the four descent strategies (Sec. 3.4) live here; the data structures are in
microcluster.py and node.py. Written so the LiarTree can subclass it and only
override the descent / leaf / split hooks (see liartree.py).

Depth-first does the full buffer + hitchhiker descent. The other three
strategies (breadth-first, best-first, iterative depth-first) share a search +
finalize model: explore within the distance budget to pick a leaf, then update
the path and insert there (or buffer if the budget ran out first).
"""
from __future__ import annotations

import heapq
import itertools
import time

import numpy as np

import stream_clustering   # baseframework, made importable in src/__init__
import utils               # baseframework euclidean distance

from .node import Node, Entry
from .microcluster import MicroCluster

_DESCENTS = ("depthfirst", "breadthfirst", "bestfirst", "iterativedf")


class ClusTree(stream_clustering.OnlinePhase_StreamClusterer):
    """Anytime stream clustering over a hierarchy of micro-clusters.

    Parameters
    ----------
    max_fanout, min_fanout : M / m for inner nodes (M=3 is best, Sec. 4.1).
    leaf_capacity_max, leaf_capacity_min : L / l for leaf nodes.
    lam : decay rate of omega(t) = 2^(-lambda*t) (Sec. 3.2).
    horizon : snapshot interval t_snap for the discard rule (Eq. 1).
    descent : one of depthfirst / breadthfirst / bestfirst / iterativedf.
    interrupt : how the anytime insertion is cut off (default "time", as in the
        paper): "time" stops when the per-object time slice elapses or a
        wall-clock deadline (the next object's arrival) is reached; "budget"
        stops after a fixed number of distance computations (deterministic and
        machine-independent, handy for reproducible runs).
    time_slice : per-object time allowance in seconds for interrupt="time"
        (None = no limit unless a deadline is passed to learn_one).
    budget : distance computations per object for interrupt="budget".
    max_height : cap on tree height (0 = unbounded). Once reached, leaf overflow
        merges instead of splitting, so the model size stops growing.

    Fast-stream aggregation (Sec. 3.3) lives in the runner (StreamRunner),
    which drives insert_cf from a separate aggregation thread.
    """

    def __init__(self, max_fanout=3, min_fanout=1, leaf_capacity_max=3,
                 leaf_capacity_min=1, lam=0.001, horizon=1000,
                 descent="depthfirst", interrupt="time", time_slice=None,
                 budget=36, max_height=0):
        if descent not in _DESCENTS:
            raise ValueError("descent must be one of %s" % (_DESCENTS,))
        if interrupt not in ("time", "budget"):
            raise ValueError("interrupt must be 'time' or 'budget'")
        self.max_fanout = max_fanout
        self.min_fanout = min_fanout
        self.leaf_max = leaf_capacity_max
        self.leaf_min = leaf_capacity_min
        self.lam = lam
        self.horizon = horizon
        self.descent = descent
        self.interrupt = interrupt
        self.time_slice = time_slice
        self.budget = budget
        self.max_height = max_height

        self.root = Node(is_leaf=True, timestamp=0, created_by="root")
        self.merge_list = []
        self._time = 0
        self._next_id = 0
        self._dim = None
        self._dist_used = 0
        self._cur_budget = budget
        self._deadline = None
        self._height = 1

    def learn_one(self, x, t=None, deadline=None):
        """Insert one object. Returns (summary_id, cluster_id, event_log) as the
        baseframework expects. deadline is an optional wall-clock time
        (time.perf_counter) by which insertion must stop; on a live stream pass
        the next object's arrival time here (the paper's interruption)."""
        if t is None:
            t = self._time + 1
        self._time = t

        x = np.asarray(x, dtype=float)
        if self._dim is None:
            self._dim = x.size
        x_cf = MicroCluster.from_point(x, t)

        self._begin_insertion(deadline)
        return self._insert(x_cf, t)

    def _begin_insertion(self, deadline=None):
        """Reset the per-object interruption state for the chosen mode."""
        self._dist_used = 0
        self._cur_budget = self.budget
        if self.interrupt == "time" and deadline is not None:
            self._deadline = deadline
        elif self.interrupt == "time" and self.time_slice is not None:
            self._deadline = time.perf_counter() + self.time_slice
        else:
            self._deadline = None

    def _insert(self, cf, t):
        if self.descent == "depthfirst":
            return self._insert_depth_first(cf, t)
        return self._insert_via_search(cf, t)

    def insert_cf(self, cf, t=None):
        """Insert a pre-built CF (a Sec. 3.3 aggregate) full-depth, i.e. with no
        budget limit so it descends all the way to a leaf. The aggregation
        thread in the runner calls this; growth is still bounded by max_height."""
        if t is None:
            t = self._time + 1
        self._time = t
        if self._dim is None:
            self._dim = cf.mean().size
        cf = cf.copy()
        cf.decay(t, self.lam)           # bring the aggregate to the current time
        self._cur_budget = float("inf")
        self._dist_used = 0
        self._deadline = None
        return self._insert(cf, t)

    def avg_leaf_radius(self):
        """Average radius over the current leaf entries (Sec. 3.3 maxradius)."""
        radii = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                radii.extend(e.summary.radius() for e in nd.entries
                             if not e.summary.is_empty())
            else:
                stack.extend(e.child for e in nd.entries if e.child)
        return float(np.mean(radii)) if radii else 0.0

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

    # --- depth-first descent (buffer + hitchhiker) -----------------------
    def _insert_depth_first(self, x_cf, t):
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

    # --- alternative strategies (search + finalize) ----------------------
    def _insert_via_search(self, x_cf, t):
        events = []
        self._on_descend(self.root, t)
        if self.root.is_leaf:
            leaf_node, buf, can_split = self.root, None, None
        elif self.descent == "bestfirst":
            leaf_node, buf, can_split = self._search_best_first(x_cf)
        elif self.descent == "breadthfirst":
            leaf_node, buf, can_split = self._search_breadth_first(x_cf)
        else:                                       # iterativedf
            leaf_node, buf, can_split = self._search_iterative_df(x_cf)
        assigned = self._finalize_search(leaf_node, buf, x_cf, events, can_split)
        return assigned, -1, ";".join(events)

    def _finalize_search(self, leaf_node, buf, x_cf, events, can_split):
        if leaf_node is not None:
            path = self._path_to(leaf_node)
            for e in path:
                e.summary.add(x_cf)
            return self._leaf_proc(leaf_node, x_cf, None, path, events, can_split)
        if buf is not None and buf[0] is not None:
            entry, node = buf
            for e in self._path_to(node):
                e.summary.add(x_cf)
            entry.summary.add(x_cf)
            entry.buffer.add(x_cf)
            events.append("buffered")
        return -1

    def _search_best_first(self, x_cf):
        """Refine the globally closest unrefined entry first; the first leaf
        entry reached is the target."""
        target = x_cf.mean()
        order = itertools.count()
        pq = []
        for e in self.root.entries:
            if not e.summary.is_empty():
                heapq.heappush(pq, (self._dist(target, e.summary.mean()), next(order), e, self.root))

        while pq and not self._time_is_up():
            _, _, e, node = heapq.heappop(pq)
            if e.child is None:
                return node, None, None
            child = e.child
            self._on_descend(child, self._time)
            for ce in child.entries:
                if not ce.summary.is_empty():
                    heapq.heappush(pq, (self._dist(target, ce.summary.mean()),
                                        next(order), ce, child))

        if pq:
            _, _, e, node = pq[0]
            if e.child is None:
                return node, None, None
            return None, (e, node), None
        return None, None, None

    def _search_breadth_first(self, x_cf):
        """Evaluate the tree level by level; the closest leaf entry is the
        target."""
        target = x_cf.mean()
        level = [(self._dist(target, e.summary.mean()), e, self.root)
                 for e in self.root.entries if not e.summary.is_empty()]

        while level and not self._time_is_up():
            level.sort(key=lambda it: it[0])
            if level[0][1].child is None:           # reached the leaf level
                return level[0][2], None, None
            nxt = []
            for _, e, _node in level:
                child = e.child
                self._on_descend(child, self._time)
                for ce in child.entries:
                    if not ce.summary.is_empty():
                        nxt.append((self._dist(target, ce.summary.mean()), ce, child))
            level = nxt

        if level:
            level.sort(key=lambda it: it[0])
            _, e, node = level[0]
            if e.child is None:
                return node, None, None
            return None, (e, node), None
        return None, None, None

    def _search_iterative_df(self, x_cf):
        """Greedy depth-first first, then check siblings level by level while
        the budget lasts and keep the closest leaf."""
        kind, node, entry, dist = self._greedy_descent(self.root, x_cf)
        if kind == "buffer":
            return None, (entry, node), None
        # decide the split now, like depth-first, before the sibling search below
        # spends the rest of the budget
        can_split = not self._time_is_up()
        best_leaf, best_dist = node, dist

        level = 0
        while not self._time_is_up():
            path = self._node_path_to(best_leaf)
            if level >= len(path) - 1:              # only inner levels have siblings to try
                break
            cur, nxt = path[level], path[level + 1]
            for e in cur.entries:
                if self._time_is_up():
                    break
                if e.child is nxt or e.child is None or e.summary.is_empty():
                    continue
                k, nd, _en, d = self._greedy_descent(e.child, x_cf)
                if k == "leaf" and d < best_dist:
                    best_leaf, best_dist = nd, d
            level += 1
        return best_leaf, None, can_split

    def _greedy_descent(self, node, x_cf):
        """Descend by closest entry to a leaf. Returns ("leaf", leaf_node,
        leaf_entry, dist) or ("buffer", node, entry, None) if interrupted."""
        while True:
            self._on_descend(node, self._time)
            e, d = self._closest(node, x_cf)
            if e is None:
                return "buffer", node, None, None
            if node.is_leaf:
                return "leaf", node, e, d
            if self._time_is_up():
                return "buffer", node, e, None
            node = e.child

    # --- budget and decay ------------------------------------------------
    def _dist(self, a, b):
        """Euclidean distance that counts against the per-object budget. Every
        distance during descent goes through here, so ClusTree and LiarTree
        share the same budget unit."""
        self._dist_used += 1
        return utils.distance_calc(a, b)

    def _time_is_up(self):
        if self.interrupt == "time":
            return self._deadline is not None and time.perf_counter() >= self._deadline
        return self._dist_used >= self._cur_budget

    def _on_descend(self, node, t):
        """Decay every entry summary and buffer in node to time t (Sec. 3.2)."""
        for e in node.entries:
            e.summary.decay(t, self.lam)
            if e.buffer is not None:
                e.buffer.decay(t, self.lam)
        node.timestamp = t

    # --- hooks the LiarTree overrides ------------------------------------
    def _closest(self, node, cf):
        """Closest entry of node to cf and its distance (None, inf if empty)."""
        target = cf.mean()
        best, best_d = None, np.inf
        for e in node.entries:
            if e.summary.is_empty():
                continue
            d = self._dist(target, e.summary.mean())
            if d < best_d:
                best_d, best = d, e
        return best, best_d

    def _calc_closest_entry(self, node, cf):
        """Entry of node whose mean is closest to cf (Sec. 3.1)."""
        return self._closest(node, cf)[0]

    def _leaf_proc(self, node, x_cf, h, path, events, can_split=None):
        """Insert x (and the hitchhiker) as leaf entries, then resolve overflow
        by discarding, splitting, or merging. can_split defaults to "budget left"
        (depth-first); the search strategies decide it themselves."""
        if can_split is None:
            can_split = not self._time_is_up()
        entry = self._new_leaf_entry(x_cf)
        node.entries.append(entry)
        if h is not None:
            node.entries.append(self._new_leaf_entry(h))

        while len(node.entries) > self.leaf_max:
            if self._discard_insignificant(node, path, events):
                continue
            capped = self.max_height and self._height >= self.max_height
            if can_split and not capped:
                self._split(node, events)
                break
            self._merge_closest(node, events)
        return entry.id

    def _split(self, node, events):
        """Split an overfull node into the two min-distance groups and propagate
        up; a new root is grown when the root itself splits."""
        groupA, groupB = self._best_split(node.entries)
        t = node.timestamp
        sibling = Node(is_leaf=node.is_leaf, parent=node.parent, timestamp=t,
                       created_by="split")
        node.entries = groupA
        sibling.entries = groupB
        for e in groupB:
            if e.child is not None:
                e.child.parent = sibling

        sumA = self._summary_of(groupA, t)
        sumB = self._summary_of(groupB, t)
        pe = node.parent_entry

        if pe is None:
            new_root = Node(is_leaf=False, timestamp=t, created_by="split")
            eA = Entry(sumA, MicroCluster.empty(self._dim, t), node, None)
            eB = Entry(sumB, MicroCluster.empty(self._dim, t), sibling, None)
            node.parent, node.parent_entry = new_root, eA
            sibling.parent, sibling.parent_entry = new_root, eB
            new_root.entries = [eA, eB]
            self.root = new_root
            self._height += 1
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

    def _path_to(self, node):
        """Entries from the root down to (and pointing at) node."""
        path = []
        nd = node
        while nd.parent_entry is not None:
            path.append(nd.parent_entry)
            nd = nd.parent
        path.reverse()
        return path

    def _node_path_to(self, node):
        """Nodes from the root down to node."""
        nodes = []
        nd = node
        while nd is not None:
            nodes.append(nd)
            nd = nd.parent
        nodes.reverse()
        return nodes

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
