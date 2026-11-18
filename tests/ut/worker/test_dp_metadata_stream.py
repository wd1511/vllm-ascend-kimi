# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the delivered DP exchange with real CPU tensors and fake NPU queues.

Run directly: ``python tests/ut/worker/test_dp_metadata_stream.py``.
The actual runner method and graph-mode reduction are extracted from this
checkout, avoiding the full vLLM/torch_npu import graph. These tests establish
the scheduling/data-lifetime contract, not real HCCL or NPU performance.
"""

import ast
import unittest
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _GraphMode(IntEnum):
    NONE = 0
    FULL = 2


def _load_exchange():
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method_name = "_sync_metadata_across_dp"
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    mode_reduction = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_post_process_cudagraph_mode"
    )
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            mode_reduction,
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {
        "torch": torch,
        "CUDAGraphMode": _GraphMode,
        "DP_METADATA_NUM_TOKENS": 0,
        "is_950": lambda: False,
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return type("ExchangeUnderTest", (), {method_name: namespace[method_name]}), namespace


class _NpuQueues:
    def __init__(self):
        self.active_stream = None
        self.streams = []
        self.events = []
        self.operations = []
        self.pending_d2h = []

    def Stream(self, device=None):
        stream = SimpleNamespace(device=device)
        self.streams.append(stream)
        return stream

    def Event(self):
        queues = self

        class Event:
            def __init__(self):
                self.recorded = False

            def record(self, stream=None):
                if queues.active_stream is None:
                    raise AssertionError("record the exchange completion on its NPU stream")
                if stream is not None and stream is not queues.active_stream:
                    raise AssertionError("completion event recorded on a different stream")
                queues.operations.append("record")
                self.recorded = True

            def synchronize(self):
                if not self.recorded:
                    raise AssertionError("cannot wait for an unrecorded exchange")
                queues.operations.append("synchronize")
                pending, queues.pending_d2h = queues.pending_d2h, []
                for finish_copy in pending:
                    finish_copy()

        event = Event()
        self.events.append(event)
        return event

    @contextmanager
    def stream(self, stream):
        previous = self.active_stream
        self.active_stream = stream
        try:
            yield
        finally:
            self.active_stream = previous


class TestDpMetadataStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exchange_type, cls.exchange_globals = _load_exchange()

    @contextmanager
    def _runtime(self, inputs=(), reductions=(), *, dp_size=3, dp_rank=2, skip=False, delayed_d2h=False):
        runner = self.exchange_type()
        runner.dp_size = dp_size
        runner.dp_rank = dp_rank
        # Factories map this generic accelerator descriptor to CPU storage,
        # while preserving the production distinction from CPU staging.
        runner.device = torch.device("privateuseone:0")
        runner.pin_memory = False
        runner.ascend_config = SimpleNamespace(enable_fused_mc2=0)
        runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size))
        runner._dp_metadata_stream = None
        runner._dp_metadata_event = None
        runner._dp_metadata_pinned = None
        runner._dp_metadata_device = None
        queues = _NpuQueues()
        device_allocations = []
        collective_tensors = []
        skip_calls = []
        expected_group = object()
        original_empty = torch.empty
        original_zeros = torch.zeros
        original_copy = torch.Tensor.copy_

        def allocate(original, *args, **kwargs):
            if kwargs.get("device") == runner.device:
                self.assertIsNotNone(queues.active_stream, "device allocation must belong to the side stream")
                self.assertEqual(queues.active_stream.device, runner.device)
                device_allocations.append(queues.active_stream)
                kwargs = {**kwargs, "device": "cpu"}
            return original(*args, **kwargs)

        def should_skip(config, is_draft_model):
            self.assertIs(config, runner.vllm_config)
            skip_calls.append(is_draft_model)
            return skip

        def all_reduce(tensor, group=None):
            self.assertIs(queues.active_stream, runner._dp_metadata_stream)
            self.assertIs(group, expected_group)
            index = len(collective_tensors)
            self.assertLess(index, len(inputs), "unexpected collective on an early-return path")
            torch.testing.assert_close(tensor, torch.tensor(inputs[index], dtype=torch.int32))
            queues.operations.append("all_reduce")
            original_copy(tensor, torch.tensor(reductions[index], dtype=torch.int32))
            collective_tensors.append(tensor)

        def copy_(destination, source, non_blocking=False):
            if destination is runner._dp_metadata_device and source is runner._dp_metadata_pinned:
                self.assertIs(queues.active_stream, runner._dp_metadata_stream)
                self.assertTrue(non_blocking)
                queues.operations.append("h2d")
            elif destination is runner._dp_metadata_pinned and source is runner._dp_metadata_device:
                self.assertIs(queues.active_stream, runner._dp_metadata_stream)
                self.assertTrue(non_blocking)
                queues.operations.append("d2h")
                if delayed_d2h:
                    # Simulate a D2H that becomes visible only when the event
                    # completes. CPU max/min must consume the reduced data.
                    snapshot = source.clone()
                    queues.pending_d2h.append(lambda: original_copy(destination, snapshot))
                    return destination
            return original_copy(destination, source, non_blocking=non_blocking)

        self.exchange_globals["dist"] = SimpleNamespace(all_reduce=all_reduce)
        self.exchange_globals["get_dp_group"] = lambda: SimpleNamespace(device_group=expected_group)
        self.exchange_globals["should_skip_allreduce_across_dp_group"] = should_skip
        with (
            patch.object(torch, "npu", queues, create=True),
            patch.object(torch, "empty", side_effect=lambda *args, **kwargs: allocate(original_empty, *args, **kwargs)),
            patch.object(torch, "zeros", side_effect=lambda *args, **kwargs: allocate(original_zeros, *args, **kwargs)),
            patch.object(torch.Tensor, "copy_", new=copy_),
        ):
            yield runner, queues, device_allocations, collective_tensors, skip_calls
        self.assertEqual(len(collective_tensors), len(reductions))

    def test_dp_size_one_returns_without_resources_or_collective(self):
        with self._runtime(dp_size=1, dp_rank=0) as (runner, queues, allocations, calls, skip_calls):
            result = runner._sync_metadata_across_dp(7, cudagraph_mode=_GraphMode.FULL)
            self.assertEqual(result, (7, None, _GraphMode.FULL))
            self.assertFalse(queues.streams or queues.events or allocations or calls or skip_calls)

    def test_skip_keeps_local_padding_and_graph_mode_without_npu_work(self):
        with self._runtime(skip=True) as (runner, queues, allocations, calls, skip_calls):
            maximum, padding, mode = runner._sync_metadata_across_dp(
                7, is_draft_model=True, cudagraph_mode=_GraphMode.FULL
            )
            self.assertEqual((maximum, mode), (7, _GraphMode.FULL))
            torch.testing.assert_close(padding, torch.tensor([7, 7, 7], dtype=torch.int32))
            self.assertEqual(skip_calls, [True])
            self.assertFalse(queues.streams or queues.events or allocations or calls)

    def test_initial_allocation_and_collective_use_the_selected_device_stream(self):
        packed = [[0, 0, 4], [0, 0, 2]]
        reduced = [[7, 9, 4], [2, 2, 2]]
        with self._runtime([packed], [reduced]) as (runner, queues, allocations, _, _):
            maximum, padding, mode = runner._sync_metadata_across_dp(
                4, cudagraph_mode=_GraphMode.FULL, allow_dp_padding=True
            )
            self.assertEqual((maximum, mode), (9, _GraphMode.FULL))
            torch.testing.assert_close(padding, torch.tensor([9, 9, 9], dtype=torch.int32))
            self.assertEqual(len(allocations), 1)
            self.assertIs(allocations[0], runner._dp_metadata_stream)
            self.assertIsNone(queues.active_stream)

    def test_delayed_d2h_completes_before_cpu_max_min_and_per_rank_padding(self):
        packed = [[0, 0, 4], [0, 0, 2]]
        reduced = [[7, 9, 4], [0, 2, 2]]
        with self._runtime([packed], [reduced], delayed_d2h=True) as (runner, queues, _, _, _):
            maximum, padding, mode = runner._sync_metadata_across_dp(4, cudagraph_mode=_GraphMode.FULL)
            self.assertEqual((maximum, mode), (9, _GraphMode.NONE))
            torch.testing.assert_close(padding, torch.tensor([7, 9, 4], dtype=torch.int32))
            self.assertFalse(queues.pending_d2h)
            self.assertLess(queues.operations.index("d2h"), queues.operations.index("record"))
            self.assertLess(queues.operations.index("record"), queues.operations.index("synchronize"))

    def test_reuse_clears_old_rank_columns_and_preserves_previous_padding(self):
        inputs = [[[0, 0, 4], [0, 0, 2]], [[0, 0, 6], [0, 0, 0]]]
        reductions = [[[7, 9, 4], [2, 2, 2]], [[3, 5, 6], [2, 2, 0]]]
        with self._runtime(inputs, reductions) as (runner, queues, allocations, calls, _):
            _, first_padding, _ = runner._sync_metadata_across_dp(4, cudagraph_mode=_GraphMode.FULL)
            staging = runner._dp_metadata_pinned
            device_buffer = runner._dp_metadata_device
            maximum, second_padding, mode = runner._sync_metadata_across_dp(6, cudagraph_mode=_GraphMode.NONE)
            torch.testing.assert_close(first_padding, torch.tensor([7, 9, 4], dtype=torch.int32))
            torch.testing.assert_close(second_padding, torch.tensor([3, 5, 6], dtype=torch.int32))
            self.assertEqual((maximum, mode), (6, _GraphMode.NONE))
            self.assertIs(runner._dp_metadata_pinned, staging)
            self.assertIs(runner._dp_metadata_device, device_buffer)
            self.assertIs(calls[0], calls[1])
            self.assertEqual(len(queues.streams), 1)
            self.assertEqual(len(queues.events), 1)
            self.assertEqual(len(allocations), 1)

    def test_draft_uses_uniform_max_padding_even_when_padding_flag_is_false(self):
        packed = [[0, 0, 4], [0, 0, 2]]
        reduced = [[7, 9, 4], [2, 0, 2]]
        with self._runtime([packed], [reduced]) as (runner, _, _, _, skip_calls):
            maximum, padding, mode = runner._sync_metadata_across_dp(
                4, is_draft_model=True, cudagraph_mode=_GraphMode.FULL, allow_dp_padding=False
            )
            self.assertEqual((maximum, mode), (9, _GraphMode.NONE))
            torch.testing.assert_close(padding, torch.tensor([9, 9, 9], dtype=torch.int32))
            self.assertEqual(skip_calls, [True])


if __name__ == "__main__":
    unittest.main()
