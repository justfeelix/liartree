"""Tree structure for the ClusTree: entries and nodes (Def. 3.1).

An ``Entry`` and the ``Node`` it may point to are tightly coupled (an R-tree
style entry/child relationship), so both live here.
"""
from __future__ import annotations


class Entry:
    """An entry inside a ClusTree node.

    - Inner-node entry: ``{summary CF, buffer CF, pointer to child node}``.
    - Leaf-node entry:  ``{summary CF, unique id}``; ``buffer`` and ``child``
      are ``None`` because inserts at leaf level are final (Sec. 3.1).
    """

    def __init__(self, summary=None, buffer=None, child=None, entry_id=None):
        self.summary = summary    # MicroCluster summarising the subtree / leaf
        self.buffer = buffer      # MicroCluster buffer for hitchhikers (inner only)
        self.child = child        # child Node (inner only)
        self.id = entry_id        # unique id for tracking (leaf entries, Sec. 3.1)

    def is_inner(self):
        return self.child is not None


class Node:
    """A ClusTree node holding between ``m`` and ``M`` entries (``l..L`` at leaf
    level).

    All entries in a node share one timestamp, because decay updates the whole
    node at once when it is descended into (Sec. 3.2).
    """

    def __init__(self, is_leaf=False, parent=None, parent_entry=None, timestamp=0,
                 created_by=None):
        self.entries = []                 # list[Entry]
        self.is_leaf = is_leaf
        self.parent = parent              # parent Node (None at the root)
        self.parent_entry = parent_entry  # entry in the parent node pointing here
        self.timestamp = timestamp

        # Provenance tag for inspection/visualization only -- purely
        # descriptive metadata, never read by the algorithm itself:
        # "root" (the tree's initial node), "split" (born from
        # ClusTree._split / LiarTree._split_liar overflow handling), or
        # "noise_to_cluster" (born from the LiarTree top-down liar-subtree
        # growth in _spawn_liar_root / _liar_proc, Sec. 2.3-2.4).
        self.created_by = created_by

        # Unused by plain ClusTree, here for the LiarTree subclass:
        self.noise_buffer = None          # CF_nb per-node noise buffer (Def. 1)
        self.is_liar = False              # liar node flag (Def. 5)
        self.is_liar_root = False         # liar root flag (Def. 5)

    def is_full(self, max_entries):
        return len(self.entries) >= max_entries
