# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone NumPy tests for production policy type 4 and its factory entry.

Run with ``python tests/ut/eplb/core/policy/test_migration_aware.py``.
Only project imports are isolated: the policy and factory implementations are
executed from their actual source files, without vLLM or NPU dependencies.
"""

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
POLICY_DIR = ROOT / "vllm_ascend/eplb/core/policy"


def _load_policy_and_factory():
    path = POLICY_DIR / "policy_migration_aware.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "policy_abstract")
    ]
    namespace = {"EplbPolicy": object}
    exec(compile(tree, str(path), "exec"), namespace)
    policy = namespace["MigrationAwareEplb"]
    factory_path = POLICY_DIR / "policy_factory.py"
    factory_tree = ast.parse(factory_path.read_text(encoding="utf-8"), filename=str(factory_path))
    factory_tree.body = [node for node in factory_tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    existing_names = ("RandomLoadBalance", "DefaultEplb", "SwiftBalanceEplb", "FlashLB")
    existing = {name: type(name, (), {}) for name in existing_names}
    namespace.update(existing, logger=MagicMock(), warm_up=MagicMock())
    exec(compile(factory_tree, str(factory_path), "exec"), namespace)
    return policy, namespace["PolicyFactory"], namespace


def _observations(table, logical_heat):
    result = np.empty(table.shape, dtype=np.float64)
    for layer_index, layer in enumerate(table):
        ids, counts = np.unique(layer, return_counts=True)
        replicas = dict(zip(ids.tolist(), counts.tolist()))
        for rank in range(layer.shape[0]):
            for slot, expert in enumerate(layer[rank]):
                result[layer_index, rank, slot] = logical_heat[layer_index][int(expert)] / replicas[int(expert)]
    return result


def _rank_loads(table, logical_heat):
    loads = []
    for layer_index, layer in enumerate(table):
        ids, counts = np.unique(layer, return_counts=True)
        replicas = dict(zip(ids.tolist(), counts.tolist()))
        loads.append([
            sum(logical_heat[layer_index][int(expert)] / replicas[int(expert)] for expert in rank)
            for rank in layer
        ])
    return np.asarray(loads)


class TestMigrationAwareEplb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_class, cls.factory, cls.factory_globals = _load_policy_and_factory()

    def assert_valid_placement(self, previous, proposed, max_migrations=16):
        self.assertEqual(proposed.shape, previous.shape)
        for old_layer, new_layer in zip(previous, proposed):
            np.testing.assert_array_equal(np.sort(old_layer.ravel()), np.sort(new_layer.ravel()))
            self.assertLessEqual(int(np.count_nonzero(old_layer != new_layer)), max_migrations)
            for old_rank, new_rank in zip(old_layer, new_layer):
                self.assertEqual(len(np.unique(new_rank)), len(new_rank))
                retained = np.isin(old_rank, new_rank)
                np.testing.assert_array_equal(old_rank[retained], new_rank[retained])

    def test_registered_factory_policy_improves_hot_rank_from_190_to_100(self):
        table = np.array([[[0, 1], [2, 3]]], dtype=np.int64)
        heat = [{0: 100.0, 1: 90.0, 2: 10.0, 3: 0.0}]
        workload = _observations(table, heat)
        original_table, original_workload = table.copy(), workload.copy()
        policy = self.factory.generate_policy(4)
        self.assertIsInstance(policy, self.policy_class)
        changed, priority, proposal = policy.rebalance_experts(table, workload)
        proposed = np.asarray(proposal)
        self.assertEqual(changed, 1)
        np.testing.assert_array_equal(priority, [0])
        np.testing.assert_allclose(_rank_loads(table, heat), [[190, 10]])
        np.testing.assert_allclose(_rank_loads(proposed, heat), [[100, 100]])
        self.assert_valid_placement(table, proposed)
        np.testing.assert_array_equal(table, original_table)
        np.testing.assert_array_equal(workload, original_workload)

    def test_redundant_copies_capacity_and_retained_local_slots_are_preserved(self):
        table = np.array([[[0, 1, 2], [3, 4, 0], [5, 6, 1], [7, 8, 2]]])
        heat = [{0: 100, 1: 80, 2: 60, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1}]
        changed, _, proposal = self.policy_class().rebalance_experts(table, _observations(table, heat))
        proposed = np.asarray(proposal)
        self.assertEqual(changed, 1)
        self.assertLess(_rank_loads(proposed, heat).max(), _rank_loads(table, heat).max())
        self.assert_valid_placement(table, proposed)

    def test_migration_budget_and_benefit_threshold_prevent_unprofitable_moves(self):
        table = np.array([[[0, 1], [2, 3]]])
        strong_heat = [{0: 100, 1: 90, 2: 10, 3: 0}]
        weak_heat = [{0: 10, 1: 9, 2: 8, 3: 7}]
        for options, heat in (
            ({"max_migrations_per_layer": 1}, strong_heat),
            ({"max_swaps_per_layer": 0}, strong_heat),
            ({"migration_penalty": 1.0}, strong_heat),
            ({"min_relative_improvement": 0.2}, weak_heat),
        ):
            with self.subTest(options=options):
                changed, priority, proposal = self.policy_class(**options).rebalance_experts(
                    table, _observations(table, heat)
                )
                self.assertEqual(changed, 0)
                self.assertEqual(len(priority), 0)
                np.testing.assert_array_equal(proposal, table)
        changed, _, proposal = self.policy_class(max_migrations_per_layer=2).rebalance_experts(
            table, _observations(table, strong_heat)
        )
        self.assertEqual(changed, 1)
        self.assert_valid_placement(table, np.asarray(proposal), max_migrations=2)

    def test_determinism_and_balanced_followup_do_not_move_experts_again(self):
        table = np.array([[[0, 1], [2, 3]]])
        heat = [{0: 100, 1: 90, 2: 10, 3: 0}]
        policy = self.policy_class()
        first = policy.rebalance_experts(table, _observations(table, heat))
        second = policy.rebalance_experts(table, _observations(table, heat))
        np.testing.assert_array_equal(first[2], second[2])
        proposed = np.asarray(first[2])
        changed, priority, followup = policy.rebalance_experts(proposed, _observations(proposed, heat))
        self.assertEqual(changed, 0)
        self.assertEqual(len(priority), 0)
        np.testing.assert_array_equal(followup, proposed)

    def test_zero_negative_and_nonfinite_statistics_leave_layer_unchanged(self):
        table = np.array([[[0, 1], [2, 3]]])
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(observation=bad):
                workload = np.full(table.shape, bad)
                changed, priority, proposal = self.policy_class().rebalance_experts(table, workload)
                self.assertEqual(changed, 0)
                self.assertEqual(len(priority), 0)
                np.testing.assert_array_equal(proposal, table)
        workload = np.array([[[100.0, -1.0], [1.0, 1.0]]])
        self.assertEqual(self.policy_class().rebalance_experts(table, workload)[0], 0)

    def test_layer_priority_and_incomplete_layer_are_handled_independently(self):
        table = np.array([[[0, 1], [2, 3]]] * 3)
        heat = [{0: 100, 1: 90, 2: 10, 3: 0}, {0: 10, 1: 9, 2: 8, 3: 7}, {0: 100, 1: 90, 2: 10, 3: 0}]
        workload = _observations(table, heat)
        workload[2, 0, 0] = -1
        changed, priority, proposal = self.policy_class().rebalance_experts(table, workload)
        self.assertEqual(changed, 1)
        np.testing.assert_array_equal(priority, [0, 1])
        np.testing.assert_array_equal(proposal[2], table[2])
        self.assert_valid_placement(table, np.asarray(proposal))

    def test_noncontiguous_ids_single_rank_and_extreme_scales(self):
        table = np.array([[[100, 900], [5000, 10]]])
        heat = [{100: 100, 900: 90, 5000: 10, 10: 0}]
        expected = self.policy_class().rebalance_experts(table, _observations(table, heat))[2]
        for scale in (1e-250, 1e250):
            with self.subTest(scale=scale):
                proposal = self.policy_class().rebalance_experts(table, _observations(table, heat) * scale)[2]
                np.testing.assert_array_equal(proposal, expected)
        single = np.array([[[0, 1]]])
        self.assertEqual(self.policy_class().rebalance_experts(single, np.array([[[10, 0]]]))[0], 0)

    def test_invalid_shapes_ids_duplicates_and_parameters_are_rejected(self):
        policy = self.policy_class()
        for table, workload in (
            (np.zeros((2, 2), dtype=int), np.zeros((2, 2))),
            (np.array([[[0, 1], [2, 3]]]), np.zeros((1, 2, 3))),
            (np.empty((0, 2, 2), dtype=int), np.empty((0, 2, 2))),
            (np.array([[[0, -1], [2, 3]]]), np.ones((1, 2, 2))),
            (np.array([[[0.0, 1.0], [2.0, 3.0]]]), np.ones((1, 2, 2))),
            (np.array([[[0, 0], [2, 3]]]), np.ones((1, 2, 2))),
            (np.array([[[0, 1], [2, 2**63]]], dtype=np.uint64), np.ones((1, 2, 2))),
        ):
            with self.subTest(shape=table.shape, table=table.tolist()):
                with self.assertRaises(ValueError):
                    policy.rebalance_experts(table, workload)
        for options in (
            {"min_relative_improvement": -0.1}, {"min_relative_improvement": np.nan},
            {"migration_penalty": -1}, {"migration_penalty": np.inf},
            {"max_migrations_per_layer": -1}, {"max_swaps_per_layer": 1.5}, {"max_swaps_per_layer": True},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.policy_class(**options)

    def test_randomized_valid_inputs_never_increase_peak_or_lose_experts(self):
        rng = np.random.default_rng(17)
        original = np.array([[[0, 1, 2], [3, 4, 0], [5, 6, 1], [7, 8, 2]]])
        for _ in range(40):
            heat = [dict(enumerate(rng.lognormal(1, 2, 9).tolist()))]
            changed, _, proposal = self.policy_class(max_migrations_per_layer=4).rebalance_experts(
                original, _observations(original, heat)
            )
            proposed = np.asarray(proposal)
            self.assert_valid_placement(original, proposed, max_migrations=4)
            self.assertLessEqual(_rank_loads(proposed, heat).max(), _rank_loads(original, heat).max() + 1e-10)
            self.assertEqual(bool(changed), not np.array_equal(original, proposed))

    def test_factory_keeps_existing_policies_and_only_flashlb_warms_up(self):
        warm_up = self.factory_globals["warm_up"]
        warm_up.reset_mock()
        for policy_type, name in enumerate(("RandomLoadBalance", "DefaultEplb", "SwiftBalanceEplb", "FlashLB")):
            self.assertIsInstance(self.factory.generate_policy(policy_type), self.factory_globals[name])
        warm_up.assert_called_once_with()
        warm_up.reset_mock()
        self.factory.generate_policy(4)
        warm_up.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
