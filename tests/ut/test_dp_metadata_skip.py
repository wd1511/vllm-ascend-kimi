# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/standard-library tests for the dense-context DP skip decision.

The actual utility and runner method are extracted from this source directory.
Only model classification, communication configuration and tensor/transport
boundaries are mocked. No vLLM, PyTorch or NPU installation is required.
"""

import ast
import sys
import types
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
DECODE_TOKEN_LIMIT = 16
PREFILL_TOKEN_LIMIT = 64


class _CommType(IntEnum):
    MC2 = 1
    FUSED_MC2 = 2
    OTHER = 3


class _GraphMode(IntEnum):
    NONE = 0
    FULL = 2


def _compile_function(node, path, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def _load_skip(namespace):
    path = ROOT / "vllm_ascend/utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "should_skip_allreduce_across_dp_group"
    )
    min_tokens = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_MIN_NONEMPTY_MOE_BATCH_TOKENS" for target in node.targets)
    )
    namespace["_MIN_NONEMPTY_MOE_BATCH_TOKENS"] = ast.literal_eval(min_tokens.value)
    return _compile_function(function, path, namespace)


def _load_runner_sync(namespace):
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef)
        and node.name == "_sync_metadata_across_dp"
    )
    return _compile_function(method, path, namespace)


class TestDenseContextDpSkip(unittest.TestCase):
    def setUp(self):
        self.ascend_config = SimpleNamespace(
            get_mc2_comm_alg=MagicMock(return_value="hierarchy"),
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=False),
        )
        self.is_moe = MagicMock(side_effect=lambda config: config.main_is_moe)
        self.is_draft_moe = MagicMock(side_effect=lambda config: config.draft_is_moe)
        self.use_megamoe = MagicMock(side_effect=lambda config: config.megamoe)
        self.select_comm = MagicMock(
            side_effect=lambda tokens, config: (
                config.decode_comm if tokens == DECODE_TOKEN_LIMIT else config.prefill_comm
            )
        )
        self.namespace = {
            "get_ascend_config": lambda: self.ascend_config,
            "is_moe_model": self.is_moe,
            "is_drafter_moe_model": self.is_draft_moe,
            "get_potential_max_tokens": lambda: DECODE_TOKEN_LIMIT,
        }
        self.skip = _load_skip(self.namespace)
        forward_context = types.ModuleType("vllm_ascend.ascend_forward_context")
        forward_context.select_moe_comm_method = self.select_comm
        forward_context.use_cann_megamoe = self.use_megamoe
        comm_method = types.ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
        comm_method.MoECommType = _CommType
        self.modules = patch.dict(sys.modules, {
            "vllm_ascend.ascend_forward_context": forward_context,
            "vllm_ascend.ops.fused_moe.moe_comm_method": comm_method,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    @staticmethod
    def _config(*, main_is_moe=True, draft_is_moe=True, kv_consumer=True,
                megamoe=False, decode_comm=_CommType.MC2,
                prefill_comm=_CommType.MC2, uniform_graph=True):
        return SimpleNamespace(
            main_is_moe=main_is_moe,
            draft_is_moe=draft_is_moe,
            megamoe=megamoe,
            decode_comm=decode_comm,
            prefill_comm=prefill_comm,
            kv_transfer_config=(None if kv_consumer is None else SimpleNamespace(is_kv_consumer=kv_consumer)),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=PREFILL_TOKEN_LIMIT),
            compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(separate_routine=lambda: not uniform_graph)
            ),
        )

    def test_dense_target_skips_even_when_hierarchy_is_configured(self):
        config = self._config(main_is_moe=False)
        self.assertTrue(self.skip(config))
        self.is_moe.assert_called_once_with(config)
        self.is_draft_moe.assert_not_called()
        self.ascend_config.get_mc2_comm_alg.assert_not_called()
        self.select_comm.assert_not_called()

    def test_dense_drafter_of_moe_target_skips_hierarchy(self):
        config = self._config(main_is_moe=True, draft_is_moe=False)
        self.assertTrue(self.skip(config, is_draft_model=True))
        self.is_draft_moe.assert_called_once_with(config)
        self.is_moe.assert_not_called()
        self.ascend_config.get_mc2_comm_alg.assert_not_called()

    def test_hierarchy_moe_target_and_drafter_still_require_collective(self):
        for is_draft_model in (False, True):
            with self.subTest(is_draft_model=is_draft_model):
                self.assertFalse(self.skip(self._config(), is_draft_model=is_draft_model))
        self.use_megamoe.assert_not_called()
        self.select_comm.assert_not_called()

    def test_existing_dense_skip_for_other_algorithms_is_preserved(self):
        for algorithm in ("", "fullmesh", "fullmesh_v2"):
            self.ascend_config.get_mc2_comm_alg.return_value = algorithm
            for is_draft_model in (False, True):
                with self.subTest(algorithm=algorithm, draft=is_draft_model):
                    self.assertTrue(self.skip(
                        self._config(main_is_moe=False, draft_is_moe=False),
                        is_draft_model=is_draft_model,
                    ))

    def test_nonconsumer_moe_still_requires_collective(self):
        self.ascend_config.get_mc2_comm_alg.return_value = "fullmesh"
        for kv_consumer in (None, False):
            with self.subTest(kv_consumer=kv_consumer):
                self.assertFalse(self.skip(self._config(kv_consumer=kv_consumer)))
        self.select_comm.assert_not_called()

    def test_megamoe_uniform_token_guard_is_preserved(self):
        self.ascend_config.get_mc2_comm_alg.return_value = "fullmesh"
        self.assertFalse(self.skip(self._config(megamoe=True)))
        self.select_comm.assert_not_called()

    def test_nonhierarchy_moe_mc2_contracts_are_preserved(self):
        self.ascend_config.get_mc2_comm_alg.return_value = "fullmesh"
        cases = (
            (_CommType.MC2, _CommType.MC2, True, False, True),
            (_CommType.FUSED_MC2, _CommType.FUSED_MC2, True, False, False),
            (_CommType.MC2, _CommType.MC2, False, False, False),
            (_CommType.MC2, _CommType.OTHER, True, False, False),
            (_CommType.MC2, _CommType.OTHER, False, True, True),
            (_CommType.OTHER, _CommType.MC2, True, True, False),
        )
        for decode, prefill, uniform, recompute, expected in cases:
            with self.subTest(decode=decode, prefill=prefill, uniform=uniform, recompute=recompute):
                self.ascend_config.scheduler_config.recompute_scheduler_enable = recompute
                config = self._config(decode_comm=decode, prefill_comm=prefill, uniform_graph=uniform)
                self.assertEqual(self.skip(config), expected)

    def test_decision_does_not_cache_context_or_algorithm(self):
        config = self._config(main_is_moe=False)
        self.assertTrue(self.skip(config))
        config.main_is_moe = True
        self.assertFalse(self.skip(config))
        self.ascend_config.get_mc2_comm_alg.return_value = "fullmesh"
        self.assertTrue(self.skip(config))
        self.ascend_config.get_mc2_comm_alg.return_value = "hierarchy"
        self.assertFalse(self.skip(config))

    def test_dense_hierarchy_runner_keeps_local_metadata_without_collective(self):
        all_reduce = MagicMock(side_effect=AssertionError("dense context must not issue a DP collective"))
        tensor_factory = MagicMock(side_effect=lambda values, **kwargs: tuple(values))
        namespace = {
            "torch": SimpleNamespace(tensor=tensor_factory, int32="int32"),
            "CUDAGraphMode": _GraphMode,
            "should_skip_allreduce_across_dp_group": self.skip,
            "dist": SimpleNamespace(all_reduce=all_reduce),
        }
        synchronize = _load_runner_sync(namespace)
        for rank, tokens, mode in ((0, 3, _GraphMode.NONE), (1, 9, _GraphMode.FULL)):
            with self.subTest(rank=rank):
                runner = SimpleNamespace(dp_size=2, dp_rank=rank, vllm_config=self._config(draft_is_moe=False))
                result = synchronize(runner, tokens, is_draft_model=True, cudagraph_mode=mode)
                self.assertEqual(result, (tokens, (tokens, tokens), mode))
        all_reduce.assert_not_called()
        self.assertEqual(tensor_factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
