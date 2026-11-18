# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU tests of the actual pool and runner integration.

Run with ``python tests/ut/worker/test_dp_metadata_stream_order.py``. Real
PyTorch CPU tensors model data; only NPU queues and HCCL are faked. This does
not prove NPU concurrency, backend priority behavior, or serving performance.
"""

import ast
import importlib.util
import sys
import threading
import unittest
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
DEVICE = torch.device("privateuseone:0")


class _GraphMode(IntEnum):
    NONE = 0
    FULL = 2


def _load_sources():
    module_name = "_c_dp_metadata_pool_under_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "vllm_ascend/worker/metadata_stream_pool.py")
    pool_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = pool_module
    spec.loader.exec_module(pool_module)
    source_path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    parsed = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    runner = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    names = ("_sync_metadata_across_dp", "_dp_metadata_allreduce_on_side_stream")
    methods = [node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    mode_reduction = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_post_process_cudagraph_mode"
    )
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), mode_reduction]
        + methods,
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {
        "torch": torch,
        "CUDAGraphMode": _GraphMode,
        "DP_METADATA_NUM_TOKENS": 0,
        "is_950": lambda: False,
        "DpMetadataStreamPool": pool_module.DpMetadataStreamPool,
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_DP_METADATA_ALLREDUCE_ON_COMPUTE_STREAM=False),
    }
    exec(compile(isolated, str(source_path), "exec"), namespace)
    runner_type = type("RunnerUnderTest", (), {name: namespace[name] for name in names})
    return pool_module, runner_type, namespace


class _FakeNPU:
    def __init__(self, complete_on_record):
        self.complete_on_record = complete_on_record
        self.active = None
        self.streams = []
        self.events = []
        self.operations = []

    def Stream(self, device=None, priority=0):
        stream = SimpleNamespace(device=device, priority=priority, pending=[])
        self.streams.append(stream)
        return stream

    def Event(self):
        runtime = self

        class Event:
            def __init__(self):
                self.callbacks = []
                self.completed = False
                self.waits = 0
                self.queries = 0

            def _complete(self):
                for callback in self.callbacks:
                    callback()
                self.callbacks = []
                self.completed = True

            def record(self):
                if runtime.active is None:
                    raise AssertionError("completion must be recorded on its owning NPU lane")
                self.callbacks, runtime.active.pending = runtime.active.pending, []
                self.completed = False
                runtime.operations.append("record")
                if runtime.complete_on_record:
                    self._complete()

            def query(self):
                self.queries += 1
                return self.completed

            def synchronize(self):
                self.waits += 1
                runtime.operations.append("synchronize")
                self._complete()

        event = Event()
        self.events.append(event)
        return event

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous


class TestDpMetadataStreamPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool_module, cls.runner_type, cls.runner_globals = _load_sources()

    @contextmanager
    def _runtime(self, inputs=(), reductions=(), *, complete_on_record=False, fail_collective=False):
        runtime = _FakeNPU(complete_on_record)
        group = object()
        owners = {}
        calls = []
        original_empty = torch.empty
        original_copy = torch.Tensor.copy_

        def empty(*args, **kwargs):
            if kwargs.get("device") == DEVICE:
                self.assertIsNotNone(runtime.active, "first device allocation escaped the NPU lane")
                self.assertEqual(runtime.active.device, DEVICE)
                tensor = original_empty(*args, **{**kwargs, "device": "cpu"})
                owners[id(tensor)] = runtime.active
                return tensor
            return original_empty(*args, **kwargs)

        def all_reduce(tensor, group=None):
            self.assertIs(group, expected_group)
            self.assertIs(owners[id(tensor)], runtime.active)
            index = len(calls)
            self.assertLess(index, len(inputs), "unexpected collective")
            torch.testing.assert_close(tensor, torch.tensor(inputs[index], dtype=torch.int32))
            calls.append((tensor, runtime.active.priority))
            runtime.operations.append("all_reduce")
            if fail_collective:
                raise RuntimeError("injected collective failure")
            original_copy(tensor, torch.tensor(reductions[index], dtype=torch.int32))

        def copy_(destination, source, non_blocking=False):
            if id(destination) in owners:
                self.assertIs(owners[id(destination)], runtime.active)
                self.assertTrue(non_blocking)
                runtime.operations.append("h2d")
            elif id(source) in owners:
                self.assertIs(owners[id(source)], runtime.active)
                self.assertTrue(non_blocking)
                runtime.operations.append("d2h")
                snapshot = source.clone()
                runtime.active.pending.append(lambda: original_copy(destination, snapshot))
                return destination
            return original_copy(destination, source, non_blocking=non_blocking)

        expected_group = group
        self.runner_globals["get_dp_group"] = lambda: SimpleNamespace(device_group=group)
        self.runner_globals["should_skip_allreduce_across_dp_group"] = lambda config, draft: False
        with (
            patch.object(torch, "npu", runtime, create=True),
            patch.object(torch, "empty", side_effect=empty),
            patch.object(torch.Tensor, "copy_", new=copy_),
            patch.object(self.pool_module.dist, "all_reduce", side_effect=all_reduce),
        ):
            yield runtime, group, calls
        self.assertEqual(len(calls), len(inputs))

    def _pool(self, **kwargs):
        return self.pool_module.DpMetadataStreamPool(DEVICE, False, 3, **kwargs)

    def _runner(self):
        runner = self.runner_type()
        runner.device = DEVICE
        runner.pin_memory = False
        runner.dp_size = 3
        runner.dp_rank = 2
        runner._dp_metadata_pool = None
        runner.vllm_config = object()
        runner.ascend_config = SimpleNamespace(
            enable_fused_mc2=0,
            sync_optimization=SimpleNamespace(
                stream_pool_size=2, metadata_stream_priority=-1, enable_event_query_fast_path=True
            )
        )
        return runner

    def test_invalid_pool_bounds_and_priority_are_rejected(self):
        for size in (0, 9, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self._pool(pool_size=size)
        with self.assertRaises(ValueError):
            self._pool(urgent_priority=1)

    def test_target_draft_lanes_repack_and_preserve_snapshots_in_submit_order(self):
        inputs = [[[0, 0, 12], [0, 0, 2]], [[0, 0, 15], [0, 0, 0]], [[0, 0, 7], [0, 0, 2]]]
        reductions = [[[4, 8, 12], [2, 2, 2]], [[6, 9, 15], [0, 2, 0]], [[5, 6, 7], [2, 2, 2]]]
        with self._runtime(inputs, reductions) as (runtime, group, calls):
            pool = self._pool(pool_size=2)
            first = pool.exchange(12, 2, 2, group)
            pool.exchange(15, 0, 2, group, urgent=False)
            last = pool.exchange(7, 2, 2, group)
            torch.testing.assert_close(first, torch.tensor(reductions[0], dtype=torch.int32))
            torch.testing.assert_close(last, torch.tensor(reductions[2], dtype=torch.int32))
            self.assertEqual([stream.priority for stream in runtime.streams], [-1, 0])
            self.assertEqual([priority for _, priority in calls], [-1, 0, -1])
            self.assertIs(calls[0][0], calls[2][0])
            self.assertEqual(pool.stats["streams_created"], 2)
            self.assertEqual(pool.stats["exchanges"], 3)

    def test_single_slot_configuration_keeps_a_hard_allocation_bound(self):
        inputs = [[[0, 0, 12], [0, 0, 2]], [[0, 0, 15], [0, 0, 0]]]
        reductions = [[[4, 8, 12], [2, 2, 2]], [[6, 9, 15], [0, 2, 0]]]
        with self._runtime(inputs, reductions) as (runtime, group, calls):
            pool = self._pool(pool_size=1)
            pool.exchange(12, 2, 2, group)
            pool.exchange(15, 0, 2, group, urgent=False)
            self.assertEqual(len(runtime.streams), 1)
            self.assertIs(calls[0][0], calls[1][0])

    def test_pending_d2h_is_waited_before_cpu_snapshot(self):
        with self._runtime([[[0, 0, 4], [0, 0, 2]]], [[[7, 9, 4], [0, 2, 2]]]) as (runtime, group, _):
            pool = self._pool()
            result = pool.exchange(4, 2, 2, group)
            torch.testing.assert_close(result, torch.tensor([[7, 9, 4], [0, 2, 2]], dtype=torch.int32))
            self.assertEqual(runtime.events[0].waits, 1)
            self.assertLess(runtime.operations.index("d2h"), runtime.operations.index("synchronize"))

    def test_completed_event_query_skips_host_synchronize_without_skipping_collective(self):
        inputs = [[[0, 0, 4], [0, 0, 2]]]
        reductions = [[[7, 9, 4], [0, 2, 2]]]
        with self._runtime(inputs, reductions, complete_on_record=True) as (runtime, group, calls):
            pool = self._pool()
            result = pool.exchange(4, 2, 2, group)
            torch.testing.assert_close(result, torch.tensor(reductions[0], dtype=torch.int32))
            self.assertEqual(runtime.events[0].waits, 0)
            self.assertEqual(pool.stats["event_query_hits"], 1)
            self.assertEqual(len(calls), 1)

    def test_query_fast_path_can_be_disabled(self):
        inputs = [[[0, 0, 4], [0, 0, 2]]]
        reductions = [[[7, 9, 4], [0, 2, 2]]]
        with self._runtime(inputs, reductions, complete_on_record=True) as (runtime, group, _):
            pool = self._pool(enable_event_query_fast_path=False)
            pool.exchange(4, 2, 2, group)
            self.assertEqual(runtime.events[0].queries, 0)
            self.assertEqual(runtime.events[0].waits, 1)

    def test_other_host_thread_cannot_change_collective_submit_order(self):
        with self._runtime([[[0, 0, 4], [0, 0, 2]]], [[[7, 9, 4], [0, 2, 2]]]) as (_, group, calls):
            pool = self._pool()
            pool.exchange(4, 2, 2, group)
            errors = []

            def other_thread():
                try:
                    pool.exchange(4, 2, 2, group)
                except RuntimeError as error:
                    errors.append(str(error))

            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIn("original worker thread", errors[0])
            self.assertEqual(len(calls), 1)

    def test_failed_collective_pool_cannot_silently_retry_a_sequence(self):
        with self._runtime([[[0, 0, 4], [0, 0, 2]]], fail_collective=True) as (_, group, calls):
            pool = self._pool()
            with self.assertRaisesRegex(RuntimeError, "injected collective failure"):
                pool.exchange(4, 2, 2, group)
            with self.assertRaisesRegex(RuntimeError, "failed previously"):
                pool.exchange(4, 2, 2, group)
            self.assertEqual(len(calls), 1)

    def test_runner_unpacks_max_min_and_draft_padding_through_priority_pool(self):
        inputs = [[[0, 0, 4], [0, 0, 2]], [[0, 0, 6], [0, 0, 2]]]
        reductions = [[[7, 9, 4], [0, 2, 2]], [[3, 8, 6], [2, 2, 2]]]
        with self._runtime(inputs, reductions) as (_, _, calls):
            runner = self._runner()
            maximum, first_padding, mode = runner._sync_metadata_across_dp(4, cudagraph_mode=_GraphMode.FULL)
            self.assertEqual((maximum, mode), (9, _GraphMode.NONE))
            torch.testing.assert_close(first_padding, torch.tensor([7, 9, 4], dtype=torch.int32))
            maximum, draft_padding, mode = runner._sync_metadata_across_dp(
                6, is_draft_model=True, cudagraph_mode=_GraphMode.FULL
            )
            self.assertEqual((maximum, mode), (8, _GraphMode.FULL))
            torch.testing.assert_close(draft_padding, torch.tensor([8, 8, 8], dtype=torch.int32))
            torch.testing.assert_close(first_padding, torch.tensor([7, 9, 4], dtype=torch.int32))
            self.assertEqual([priority for _, priority in calls], [-1, 0])

    def test_runner_dp_one_and_skip_return_without_constructing_the_pool(self):
        with self._runtime() as (runtime, _, _):
            runner = self._runner()
            runner.dp_size = 1
            result = runner._sync_metadata_across_dp(4, cudagraph_mode=_GraphMode.FULL)
            self.assertEqual(result, (4, None, _GraphMode.FULL))
            runner.dp_size = 3
            self.runner_globals["should_skip_allreduce_across_dp_group"] = lambda config, draft: True
            maximum, padding, mode = runner._sync_metadata_across_dp(4, cudagraph_mode=_GraphMode.FULL)
            self.assertEqual((maximum, mode), (4, _GraphMode.FULL))
            torch.testing.assert_close(padding, torch.tensor([4, 4, 4], dtype=torch.int32))
            self.assertIsNone(runner._dp_metadata_pool)
            self.assertFalse(runtime.streams)


if __name__ == "__main__":
    unittest.main()
