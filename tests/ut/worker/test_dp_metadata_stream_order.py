# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for the DP metadata stream/lifetime contract.

Run directly with ``python tests/ut/worker/test_dp_metadata_stream_order.py``.
The production method is extracted from this checkout without importing vLLM or
torch_npu. Tensor operations use real CPU PyTorch; only NPU scheduling and the
distributed transport are faked. These tests do not establish NPU performance
or real HCCL ordering, which require the documented multi-rank hardware tests.
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


def _load_runner_type():
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method_name = "_dp_metadata_allreduce_on_side_stream"
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {"torch": torch, "CUDAGraphMode": _GraphMode}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return type("RunnerUnderTest", (), {method_name: namespace[method_name]}), namespace


class _FakeNPU:
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
        runtime = self

        class Event:
            def __init__(self):
                self.recorded = False

            def record(self):
                if runtime.active_stream is None:
                    raise AssertionError("metadata completion must be recorded on its side stream")
                runtime.operations.append("record")
                self.recorded = True

            def synchronize(self):
                if not self.recorded:
                    raise AssertionError("cannot wait for an unrecorded exchange")
                runtime.operations.append("synchronize")
                callbacks, runtime.pending_d2h = runtime.pending_d2h, []
                for callback in callbacks:
                    callback()

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


class TestDpMetadataStreamOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_type, cls.method_globals = _load_runner_type()

    @contextmanager
    def _exchange_runtime(self, reductions, expected_inputs, defer_d2h=False):
        runner = self.runner_type()
        runner.dp_size = 3
        runner.dp_rank = 2
        # A valid device descriptor lets the fake factory distinguish device
        # allocations from CPU staging without requiring the NPU extension.
        runner.device = torch.device("privateuseone:0")
        runner.pin_memory = False
        runner._dp_metadata_stream = None
        runner._dp_metadata_event = None
        runner._dp_metadata_pinned = None
        runner._dp_metadata_device = None
        runtime = _FakeNPU()
        group = object()
        original_empty = torch.empty
        original_zeros = torch.zeros
        original_copy = torch.Tensor.copy_
        device_allocations = []
        calls = []

        def factory(original, *args, **kwargs):
            if kwargs.get("device") == runner.device:
                self.assertIsNotNone(runtime.active_stream, "device allocation escaped the side-stream context")
                self.assertEqual(runtime.active_stream.device, runner.device)
                device_allocations.append(runtime.active_stream)
                kwargs = {**kwargs, "device": "cpu"}
            return original(*args, **kwargs)

        def all_reduce(tensor, group=None):
            self.assertIs(runtime.active_stream, runner._dp_metadata_stream)
            self.assertIs(group, runtime_group)
            torch.testing.assert_close(tensor, torch.tensor(expected_inputs[len(calls)], dtype=torch.int32))
            runtime.operations.append("all_reduce")
            original_copy(tensor, torch.tensor(reductions[len(calls)], dtype=torch.int32))
            calls.append(tensor)

        def copy_(destination, source, non_blocking=False):
            if destination is runner._dp_metadata_device and source is runner._dp_metadata_pinned:
                self.assertIs(runtime.active_stream, runner._dp_metadata_stream)
                runtime.operations.append("h2d")
            elif destination is runner._dp_metadata_pinned and source is runner._dp_metadata_device:
                self.assertIs(runtime.active_stream, runner._dp_metadata_stream)
                runtime.operations.append("d2h")
                if defer_d2h:
                    # Leave staging unchanged until the completion event is
                    # waited. A missing/late wait exposes the local input,
                    # rather than the reduced result, to the CPU caller.
                    snapshot = source.clone()
                    runtime.pending_d2h.append(lambda: original_copy(destination, snapshot))
                    return destination
            return original_copy(destination, source, non_blocking=non_blocking)

        runtime_group = group
        self.method_globals["dist"] = SimpleNamespace(all_reduce=all_reduce)
        self.method_globals["get_dp_group"] = lambda: SimpleNamespace(device_group=group)
        with (
            patch.object(torch, "npu", runtime, create=True),
            patch.object(torch, "empty", side_effect=lambda *args, **kwargs: factory(original_empty, *args, **kwargs)),
            patch.object(torch, "zeros", side_effect=lambda *args, **kwargs: factory(original_zeros, *args, **kwargs)),
            patch.object(torch.Tensor, "copy_", new=copy_),
        ):
            yield runner, runtime, device_allocations, calls
        self.assertEqual(len(calls), len(reductions))

    def test_first_device_allocation_belongs_to_the_side_stream(self):
        reduced = [[4, 8, 12], [2, 2, 2]]
        packed = [[0, 0, 12], [0, 0, 2]]
        with self._exchange_runtime([reduced], [packed]) as (runner, runtime, allocations, _):
            result = runner._dp_metadata_allreduce_on_side_stream(12, _GraphMode.FULL)
            torch.testing.assert_close(result, torch.tensor(reduced, dtype=torch.int32))
            self.assertEqual(len(allocations), 1)
            self.assertIs(allocations[0], runner._dp_metadata_stream)
            self.assertIsNone(runtime.active_stream)

    def test_repacking_clears_other_ranks_and_previous_result_is_a_snapshot(self):
        reductions = [[[4, 8, 12], [2, 2, 2]], [[6, 9, 15], [0, 2, 0]]]
        inputs = [[[0, 0, 12], [0, 0, 2]], [[0, 0, 15], [0, 0, 0]]]
        with self._exchange_runtime(reductions, inputs) as (runner, runtime, allocations, calls):
            first = runner._dp_metadata_allreduce_on_side_stream(12, _GraphMode.FULL)
            # Keep a view, as _sync_metadata_across_dp does when DP padding is
            # disabled, and retain it across a subsequent metadata exchange.
            first_tokens = first[0]
            device_buffer = runner._dp_metadata_device
            staging_buffer = runner._dp_metadata_pinned
            second = runner._dp_metadata_allreduce_on_side_stream(15, _GraphMode.NONE)
            torch.testing.assert_close(first_tokens, torch.tensor([4, 8, 12], dtype=torch.int32))
            torch.testing.assert_close(second, torch.tensor(reductions[1], dtype=torch.int32))
            self.assertIs(runner._dp_metadata_device, device_buffer)
            self.assertIs(runner._dp_metadata_pinned, staging_buffer)
            self.assertEqual(len(runtime.streams), 1)
            self.assertEqual(len(allocations), 1)
            self.assertIs(calls[0], calls[1])

    def test_completion_wait_makes_delayed_d2h_visible_before_return(self):
        reduced = [[4, 8, 12], [0, 2, 2]]
        packed = [[0, 0, 12], [0, 0, 2]]
        with self._exchange_runtime([reduced], [packed], defer_d2h=True) as (runner, runtime, _, _):
            result = runner._dp_metadata_allreduce_on_side_stream(12, _GraphMode.FULL)
            torch.testing.assert_close(result, torch.tensor(reduced, dtype=torch.int32))
            self.assertFalse(runtime.pending_d2h)
            self.assertLess(runtime.operations.index("d2h"), runtime.operations.index("record"))
            self.assertLess(runtime.operations.index("record"), runtime.operations.index("synchronize"))


if __name__ == "__main__":
    unittest.main()
