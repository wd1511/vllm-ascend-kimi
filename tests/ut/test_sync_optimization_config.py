# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise C's actual config classes with pydantic, without importing vLLM.

The decorator uses the same pydantic dataclass/extra-forbid behavior as the
unchanged config_utils.config helper. Only heavy logging is replaced.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ConfigDict, ValidationError, model_validator
from pydantic.dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]


def _load_configs():
    source = ROOT / "vllm_ascend/ascend_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"SyncOptimizationConfig", "EplbConfig"}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("production synchronization and EPLB config classes must exist")
    namespace = {
        "config": lambda cls: dataclass(cls, config=ConfigDict(extra="forbid")),
        "model_validator": model_validator,
        "logger": SimpleNamespace(info=lambda *args: None),
        "os": os,
    }
    isolated = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    return namespace, tree


class TestSyncOptimizationConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes, cls.tree = _load_configs()

    def test_defaults_enable_bounded_priority_pool_and_event_fast_path(self):
        config = self.classes["SyncOptimizationConfig"]()
        self.assertEqual(config.stream_pool_size, 2)
        self.assertEqual(config.metadata_stream_priority, -1)
        self.assertTrue(config.enable_event_query_fast_path)

    def test_valid_single_stream_neutral_priority_and_disabled_query(self):
        config = self.classes["SyncOptimizationConfig"](
            stream_pool_size=1, metadata_stream_priority=0, enable_event_query_fast_path=False
        )
        self.assertEqual(config.stream_pool_size, 1)
        self.assertEqual(config.metadata_stream_priority, 0)
        self.assertFalse(config.enable_event_query_fast_path)

    def test_out_of_range_pool_and_priority_are_rejected(self):
        for kwargs in (
            {"stream_pool_size": 0},
            {"stream_pool_size": 5},
            {"metadata_stream_priority": -2},
            {"metadata_stream_priority": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.classes["SyncOptimizationConfig"](**kwargs)

    def test_unknown_keys_and_invalid_field_types_are_rejected(self):
        for kwargs in ({"pool_szie": 2}, {"stream_pool_size": "many"}, {"enable_event_query_fast_path": "maybe"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.classes["SyncOptimizationConfig"](**kwargs)

    def test_new_eplb_policy_can_be_selected_with_existing_enable_contract(self):
        with patch.dict(os.environ, {"DYNAMIC_EPLB": "1"}):
            config = self.classes["EplbConfig"](dynamic_eplb=True, eplb_policy_type=4)
        self.assertTrue(config.dynamic_eplb)
        self.assertEqual(config.eplb_policy_type, 4)
        with self.assertRaises(ValidationError):
            self.classes["EplbConfig"](eplb_policy_type=5)

    def test_original_eplb_policies_remain_valid(self):
        for policy in range(4):
            with self.subTest(policy=policy):
                self.assertEqual(self.classes["EplbConfig"](eplb_policy_type=policy).eplb_policy_type, policy)

    def test_sync_subconfig_is_wired_into_ascend_config(self):
        ascend = next(node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        field = next(
            node for node in ascend.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "sync_optimization"
        )
        self.assertEqual(ast.unparse(field.annotation), "SyncOptimizationConfig")
        self.assertEqual(ast.unparse(field.value), "dataclasses.field(default_factory=SyncOptimizationConfig)")


if __name__ == "__main__":
    unittest.main()
