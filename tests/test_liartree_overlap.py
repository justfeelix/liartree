from __future__ import annotations

import unittest

import numpy as np

from tests._stubs import install_baseframework_stubs

install_baseframework_stubs()

from src.liartree import LiarTree
from src.microcluster import MicroCluster
from src.node import Entry, Node


def point_cf(value, timestamp=1):
    return MicroCluster.from_point(np.array([float(value)]), timestamp)


def assert_summary_equal(testcase, actual, expected):
    testcase.assertAlmostEqual(actual.n, expected.n)
    np.testing.assert_allclose(actual.ls, expected.ls)
    np.testing.assert_allclose(actual.ss, expected.ss)
    testcase.assertEqual(actual.timestamp, expected.timestamp)


class LiarTreeOverlapTests(unittest.TestCase):
    def test_noop_when_fewer_than_two_entries_exist(self):
        tree, first, _second = self._tree_with_entries([0.0, 1.0], None)

        chosen = tree._calc_closest_entry(tree.root, point_cf(0.2))

        self.assertIs(chosen, first)

    def test_noop_when_top_two_entries_do_not_overlap(self):
        tree, first, second = self._tree_with_entries([0.0, 1.0], [10.0, 11.0])
        first_entries = list(first.child.entries)
        second_entries = list(second.child.entries)

        chosen = tree._calc_closest_entry(tree.root, point_cf(0.2))

        self.assertIs(chosen, first)
        self.assertEqual(first.child.entries, first_entries)
        self.assertEqual(second.child.entries, second_entries)

    def test_noop_when_simulated_swap_does_not_shrink_both_radii(self):
        tree, first, second = self._tree_with_entries([7.0, 8.0], [0.0, 10.0])
        first_entries = list(first.child.entries)
        second_entries = list(second.child.entries)

        chosen = tree._calc_closest_entry(tree.root, point_cf(7.1))

        self.assertIs(chosen, first)
        self.assertEqual(first.child.entries, first_entries)
        self.assertEqual(second.child.entries, second_entries)

    def test_swap_updates_summaries_parent_pointers_and_returned_entry(self):
        tree, first, second = self._tree_with_entries(
            [0.0, 10.0], [8.0, 20.0], first_buffer=5.0, second_buffer=13.0
        )
        first_moved = first.child.entries[1]
        second_moved = second.child.entries[0]

        chosen = tree._calc_closest_entry(tree.root, point_cf(9.0))

        self.assertIs(chosen, second)
        self.assertIn(second_moved, first.child.entries)
        self.assertIn(first_moved, second.child.entries)
        self.assertIs(second_moved.child.parent, first.child)
        self.assertIs(first_moved.child.parent, second.child)
        self.assertIs(second_moved.child.parent_entry, second_moved)
        self.assertIs(first_moved.child.parent_entry, first_moved)

        first_expected = tree._summary_with_buffer(
            first.child.entries, first.buffer, first.child.timestamp
        )
        second_expected = tree._summary_with_buffer(
            second.child.entries, second.buffer, second.child.timestamp
        )
        assert_summary_equal(self, first.summary, first_expected)
        assert_summary_equal(self, second.summary, second_expected)

    def test_rollback_restores_tree_after_mid_swap_failure(self):
        class FailingLiarTree(LiarTree):
            def _after_overlap_swap_mutation(self, *args):
                raise RuntimeError("forced")

        tree, first, second = self._tree_with_entries(
            [0.0, 10.0], [8.0, 20.0], first_buffer=5.0, second_buffer=13.0,
            tree_cls=FailingLiarTree,
        )
        first_summary = first.summary.copy()
        second_summary = second.summary.copy()
        first_entries = list(first.child.entries)
        second_entries = list(second.child.entries)
        first_moved = first.child.entries[1]
        second_moved = second.child.entries[0]
        first_moved_parent = first_moved.child.parent
        second_moved_parent = second_moved.child.parent

        with self.assertRaises(RuntimeError):
            tree._calc_closest_entry(tree.root, point_cf(9.0))

        self.assertEqual(first.child.entries, first_entries)
        self.assertEqual(second.child.entries, second_entries)
        self.assertIs(first_moved.child.parent, first_moved_parent)
        self.assertIs(second_moved.child.parent, second_moved_parent)
        assert_summary_equal(self, first.summary, first_summary)
        assert_summary_equal(self, second.summary, second_summary)

    def _tree_with_entries(self, first_values, second_values, first_buffer=None,
                           second_buffer=None, tree_cls=LiarTree):
        tree = tree_cls(lam=0.0, budget=100)
        tree._dim = 1
        tree._time = 1
        tree._cur_budget = 100
        tree._dist_used = 0
        root = Node(is_leaf=False, timestamp=1)
        first = self._parent_entry(tree, root, first_values, first_buffer)
        root.entries = [first]
        second = None
        if second_values is not None:
            second = self._parent_entry(tree, root, second_values, second_buffer)
            root.entries.append(second)
        tree.root = root
        return tree, first, second

    def _parent_entry(self, tree, root, values, buffer_value):
        child = Node(is_leaf=False, parent=root, timestamp=1)
        child.entries = [self._child_entry(child, value) for value in values]
        buffer = (MicroCluster.empty(1, 1) if buffer_value is None
                  else point_cf(buffer_value))
        summary = tree._summary_with_buffer(child.entries, buffer, child.timestamp)
        parent = Entry(summary=summary, buffer=buffer, child=child)
        child.parent_entry = parent
        return parent

    @staticmethod
    def _child_entry(parent, value):
        grandchild = Node(is_leaf=True, parent=parent, timestamp=1)
        entry = Entry(
            summary=point_cf(value),
            buffer=MicroCluster.empty(1, 1),
            child=grandchild,
        )
        grandchild.parent_entry = entry
        return entry


if __name__ == "__main__":
    unittest.main()
