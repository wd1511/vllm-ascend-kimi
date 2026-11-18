# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regressions for accepted-token synchronization.

Run directly with Python and torch/numpy installed. The tests compile the
actual production methods from their AST, so vLLM, torch-npu, and hardware are
not required. Only the device stream/event boundary is mocked; correction and
request-order publication execute the real source with real CPU tensors.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

EXPECT_DEVICE_WAIT = False
ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "vllm_ascend/worker/model_runner_v1.py"
RUNNER_310_PATH = ROOT / "vllm_ascend/_310p/model_runner_310p.py"
UTILS_PATH = ROOT / "vllm_ascend/spec_decode/utils.py"


def _compile_nodes(nodes, namespace, filename):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


def _load_methods(path, class_name, names, namespace, bases=()):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
    missing = set(names) - methods.keys()
    if missing:
        raise AssertionError(f"{path}: missing production methods: {sorted(missing)}")
    _compile_nodes([methods[name] for name in names], namespace, path)
    return type(class_name, bases, {name: namespace[name] for name in names})


class _Buffer:
    """Separate CPU/device storage, including an optional D2H-read guard."""

    def __init__(self, values, trace=None, ready=None):
        self.cpu = torch.tensor(values, dtype=torch.int32)
        self.gpu = self.cpu.clone()
        self.trace = trace if trace is not None else []
        self.ready = ready if ready is not None else lambda: True
        self.copy_count = 0

    @property
    def np(self):
        if not self.ready():
            raise AssertionError("accepted-token snapshot read before its D2H event")
        return self.cpu.numpy()

    def copy_to_gpu(self, num_items=None):
        if not self.ready():
            raise AssertionError("accepted-token upload before its D2H event")
        self.trace.append("h2d")
        self.copy_count += 1
        self.gpu[:num_items].copy_(self.cpu[:num_items])


class _InputPrepBoundaryReached(Exception):
    pass


class _StopBeforeComputedTokenPreparation:
    def __bool__(self):
        raise _InputPrepBoundaryReached


class TestAcceptedTokenSynchronization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = {"torch": torch, "np": np}
        utils_tree = ast.parse(UTILS_PATH.read_text(encoding="utf-8"))
        correction = next(
            node
            for node in utils_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_num_computed_tokens_for_batch_change"
        )
        _compile_nodes([correction], cls.namespace, UTILS_PATH)
        cls.correction = staticmethod(cls.namespace[correction.name])
        cls.runner_class = _load_methods(
            RUNNER_PATH,
            "NPUModelRunner",
            (
                "_sync_num_accepted_tokens",
                "_publish_num_accepted_tokens_before_mamba",
                "_update_num_computed_tokens_for_async_batch",
            ),
            cls.namespace,
        )
        cls.namespace_310 = {
            "torch": torch,
            "np": np,
            "AscendAttentionState": SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec"),
            "is_rc_device": lambda: False,
        }
        cls.runner_310_class = _load_methods(
            RUNNER_310_PATH,
            "NPUModelRunner310",
            ("_prepare_inputs", "_publish_num_accepted_tokens_before_mamba"),
            cls.namespace_310,
            bases=(cls.runner_class,),
        )

    def setUp(self):
        self.npu = SimpleNamespace(is_current_stream_capturing=MagicMock(return_value=False))
        patcher = patch.object(torch, "npu", self.npu, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_runner(
        self, runner_class=None, counts=(2, 4, 7, 8), previous=(1, 0), asynchronous=True, event=True, completed=False
    ):
        runner = (runner_class or self.runner_class)()
        trace = []
        ready = [not event or completed]
        runner.num_accepted_tokens = _Buffer(counts, trace, lambda: ready[0])
        runner.prev_positions = _Buffer(previous)
        runner.use_async_scheduling = asynchronous
        runner.ascend_config = SimpleNamespace(
            sync_optimization=SimpleNamespace(enable_event_query_fast_path=True)
        )
        runner.input_batch = SimpleNamespace(
            num_accepted_tokens_cpu=np.array([9, 8, 7, 6], dtype=np.int32),
            prev_req_id_to_index={"previous-request": 0},
        )
        runner.num_accepted_tokens_event = MagicMock() if event else None
        if event:
            runner.num_accepted_tokens_event.query.return_value = completed

            def finish_d2h():
                trace.append("host_sync")
                ready[0] = True

            runner.num_accepted_tokens_event.synchronize.side_effect = finish_d2h
        return runner, trace

    def test_async_correction_uses_required_device_dependency_without_host_wait(self):
        for has_event in (True, False):
            with self.subTest(has_event=has_event):
                runner, _ = self._make_runner(previous=(1, -1, 0), event=has_event)
                runner.num_accepted_tokens = _Buffer([71, 72, 73, 74])
                runner.num_computed_tokens = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
                runner.valid_sampled_token_count_gpu = torch.tensor([2, 4, 1, 1], dtype=torch.int32)
                runner.prev_num_draft_tokens = _Buffer([3, 3, 0, 0])
                cpu_computed = torch.tensor([88, 99, 77], dtype=torch.int32)
                trace = []
                stream = MagicMock()
                stream.wait_event.side_effect = lambda event: trace.append("device_wait")
                current_stream = MagicMock(return_value=stream)

                def apply_correction(*args, **kwargs):
                    expected_wait = ["device_wait"] if EXPECT_DEVICE_WAIT and has_event else []
                    self.assertEqual(trace, expected_wait)
                    accepted = args[1] if len(args) > 1 else kwargs["num_accepted_tokens"]
                    if EXPECT_DEVICE_WAIT:
                        self.assertIsNotNone(accepted)
                    else:
                        self.assertIsNone(accepted)
                    trace.append("correction")
                    return self.correction(*args, **kwargs)

                mocked_npu = SimpleNamespace(current_stream=current_stream)
                with (
                    patch.object(torch, "npu", mocked_npu, create=True),
                    patch.dict(self.namespace, {"update_num_computed_tokens_for_batch_change": apply_correction}),
                ):
                    runner._update_num_computed_tokens_for_async_batch(3, cpu_computed)

                torch.testing.assert_close(runner.num_computed_tokens, torch.tensor([24, 99, 12, 40], dtype=torch.int32))
                expected_counts = [4, 1, 2, 74] if EXPECT_DEVICE_WAIT else [71, 72, 73, 74]
                torch.testing.assert_close(runner.num_accepted_tokens.gpu, torch.tensor(expected_counts, dtype=torch.int32))
                if EXPECT_DEVICE_WAIT and has_event:
                    stream.wait_event.assert_called_once_with(runner.num_accepted_tokens_event)
                else:
                    stream.wait_event.assert_not_called()
                if has_event:
                    runner.num_accepted_tokens_event.synchronize.assert_not_called()

    def test_correction_retains_explicit_accepted_output_for_310p_callers(self):
        computed = torch.tensor([10, 20, 30], dtype=torch.int32)
        accepted = torch.tensor([8, 9, 10], dtype=torch.int32)
        self.correction(
            computed,
            accepted,
            torch.tensor([1, 0, -1], dtype=torch.int32),
            torch.tensor([3, 5, 2], dtype=torch.int32),
            torch.tensor([2, 0, 1], dtype=torch.int32),
            torch.tensor([111, 112, 113], dtype=torch.int32),
        )
        torch.testing.assert_close(computed, torch.tensor([111, 13, 113], dtype=torch.int32))
        torch.testing.assert_close(accepted, torch.tensor([1, 3, 1], dtype=torch.int32))

    def test_publish_waits_before_reading_d2h_snapshot(self):
        runner, trace = self._make_runner()
        runner._publish_num_accepted_tokens_before_mamba(2)
        self.assertEqual(trace, ["host_sync", "h2d"])
        runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
        np.testing.assert_array_equal(runner.num_accepted_tokens.np, [4, 2, 1, 1])
        np.testing.assert_array_equal(runner.input_batch.num_accepted_tokens_cpu[:2], [4, 2])
        torch.testing.assert_close(runner.num_accepted_tokens.gpu, torch.tensor([4, 2, 1, 1], dtype=torch.int32))

    def test_completed_event_skips_host_wait_and_still_remaps_and_uploads(self):
        runner, trace = self._make_runner(completed=True)
        runner._sync_num_accepted_tokens = MagicMock(wraps=runner._sync_num_accepted_tokens)
        runner._publish_num_accepted_tokens_before_mamba(2)
        runner.num_accepted_tokens_event.query.assert_called_once_with()
        runner.num_accepted_tokens_event.synchronize.assert_not_called()
        runner._sync_num_accepted_tokens.assert_called_once_with(2, has_prev_mapping=True)
        self.assertEqual(trace, ["h2d"])
        torch.testing.assert_close(runner.num_accepted_tokens.gpu, torch.tensor([4, 2, 1, 1], dtype=torch.int32))

    def test_disabled_fast_path_keeps_original_wait_without_query(self):
        runner, trace = self._make_runner(completed=True)
        runner.ascend_config.sync_optimization.enable_event_query_fast_path = False
        runner._publish_num_accepted_tokens_before_mamba(2)
        runner.num_accepted_tokens_event.query.assert_not_called()
        runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(trace, ["host_sync", "h2d"])

    def test_reused_event_is_queried_again_for_a_new_snapshot(self):
        runner, trace = self._make_runner(completed=True)
        runner._publish_num_accepted_tokens_before_mamba(2)
        ready = [False]
        runner.num_accepted_tokens.ready = lambda: ready[0]
        runner.num_accepted_tokens_event.query.return_value = False

        def complete_second_snapshot():
            trace.append("host_sync")
            runner.num_accepted_tokens.cpu.copy_(torch.tensor([5, 6, 7, 8], dtype=torch.int32))
            ready[0] = True

        runner.num_accepted_tokens_event.synchronize.side_effect = complete_second_snapshot
        runner._publish_num_accepted_tokens_before_mamba(2)
        self.assertEqual(runner.num_accepted_tokens_event.query.call_count, 2)
        runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(trace, ["h2d", "host_sync", "h2d"])
        torch.testing.assert_close(runner.num_accepted_tokens.gpu, torch.tensor([6, 5, 1, 1], dtype=torch.int32))

    def test_capture_path_keeps_original_wait_without_query(self):
        runner, trace = self._make_runner(completed=True)
        self.npu.is_current_stream_capturing.return_value = True
        runner._publish_num_accepted_tokens_before_mamba(2)
        runner.num_accepted_tokens_event.query.assert_not_called()
        runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
        self.assertEqual(trace, ["host_sync", "h2d"])

    def test_publish_handles_synchronous_batch_ownership_and_no_previous_mapping(self):
        for asynchronous, mapping, expected in (
            (False, {"previous-request": 0}, [9, 8]),
            (True, {}, [1, 1]),
        ):
            with self.subTest(asynchronous=asynchronous, mapping=mapping):
                runner, _ = self._make_runner(asynchronous=asynchronous)
                runner.input_batch.prev_req_id_to_index = mapping
                runner._publish_num_accepted_tokens_before_mamba(2)
                np.testing.assert_array_equal(runner.num_accepted_tokens.np[:2], expected)
                np.testing.assert_array_equal(runner.num_accepted_tokens.np[2:], [1, 1])

    def test_publish_without_event_initializes_entire_buffer(self):
        runner, trace = self._make_runner(event=False)
        runner._publish_num_accepted_tokens_before_mamba(2)
        np.testing.assert_array_equal(runner.num_accepted_tokens.np, [1, 1, 1, 1])
        torch.testing.assert_close(runner.num_accepted_tokens.gpu, torch.ones(4, dtype=torch.int32))
        self.assertEqual(trace, [])

    def _run_310_input_prep_through_accepted_counts(self, runner, num_reqs, rc=False):
        # Execute the real 310P preparation prefix, then stop at the next stage.
        # This avoids recreating its synchronization/RC branch in the test.
        runner.input_batch.num_reqs = num_reqs
        runner.input_batch.block_table = MagicMock()
        runner.arange_np = np.arange(4, dtype=np.int32)
        runner.query_pos = _Buffer([0, 0, 0, 0])
        runner._build_attn_state = MagicMock(return_value="decode")
        runner._get_cumsum_and_arange = MagicMock(return_value=np.arange(1, num_reqs + 1, dtype=np.int32))
        runner._compute_prev_positions = MagicMock()
        runner.use_async_spec_decode = _StopBeforeComputedTokenPreparation()
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=num_reqs, scheduled_spec_decode_tokens={})
        with patch.dict(self.namespace_310, {"is_rc_device": lambda: rc}):
            with self.assertRaises(_InputPrepBoundaryReached):
                runner._prepare_inputs(scheduler_output, np.ones(num_reqs, dtype=np.int32))

    def test_310p_early_publication_is_not_remapped_again_by_late_hook(self):
        cases = (
            ("reorder", (1, 0), True, [4, 2]),
            ("new_request", (1, -1, 0), True, [4, 1, 2]),
            ("condense", (2, 0), True, [7, 2]),
            ("synchronous", (1, 0), False, [9, 8]),
        )
        for name, previous, asynchronous, expected in cases:
            with self.subTest(case=name):
                runner, trace = self._make_runner(
                    self.runner_310_class, previous=previous, asynchronous=asynchronous
                )
                runner._sync_num_accepted_tokens = MagicMock(wraps=runner._sync_num_accepted_tokens)
                self._run_310_input_prep_through_accepted_counts(runner, len(previous))
                runner._publish_num_accepted_tokens_before_mamba(len(previous))
                runner._sync_num_accepted_tokens.assert_called_once_with(len(previous), has_prev_mapping=True)
                runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
                self.assertEqual(trace, ["host_sync", "h2d"])
                np.testing.assert_array_equal(runner.num_accepted_tokens.np[: len(previous)], expected)
                np.testing.assert_array_equal(runner.input_batch.num_accepted_tokens_cpu[: len(previous)], expected)
                torch.testing.assert_close(
                    runner.num_accepted_tokens.gpu[: len(previous)], torch.tensor(expected, dtype=torch.int32)
                )

    def test_310p_no_event_rc_prefix_survives_parent_late_hook(self):
        for rc, expected in ((True, [2, 4, 1, 1]), (False, [1, 1, 1, 1])):
            with self.subTest(rc=rc):
                runner, trace = self._make_runner(self.runner_310_class, event=False)
                self._run_310_input_prep_through_accepted_counts(runner, 2, rc=rc)
                before = runner.num_accepted_tokens.gpu.clone()
                before_copy_count = runner.num_accepted_tokens.copy_count
                runner._publish_num_accepted_tokens_before_mamba(2)
                np.testing.assert_array_equal(runner.num_accepted_tokens.np, expected)
                torch.testing.assert_close(runner.num_accepted_tokens.gpu, before)
                self.assertEqual(runner.num_accepted_tokens.copy_count, before_copy_count)
                self.assertEqual(trace, ["h2d"] if rc else [])

    def test_execute_and_prepare_call_the_tested_production_hooks(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        methods = {node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)}

        def self_calls(method_name, hook_name):
            return [
                node
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == hook_name
            ]

        for method_name, hook_name in (
            ("execute_model", "_publish_num_accepted_tokens_before_mamba"),
            ("_prepare_inputs", "_update_num_computed_tokens_for_async_batch"),
        ):
            calls = self_calls(method_name, hook_name)
            self.assertEqual(len(calls), 1, f"{method_name} must call the tested {hook_name} exactly once")

        preparation = self_calls("execute_model", "_prepare_inputs")
        publication = self_calls("execute_model", "_publish_num_accepted_tokens_before_mamba")
        metadata = self_calls("execute_model", "_build_attention_metadata")
        self.assertEqual(len(preparation), 1)
        self.assertEqual(len(metadata), 1)
        self.assertLess(preparation[0].lineno, publication[0].lineno)
        self.assertLess(publication[0].lineno, metadata[0].lineno)
        self.assertEqual(self_calls("execute_model", "_sync_num_accepted_tokens"), [])
        direct_correction = [
            node
            for node in ast.walk(methods["_prepare_inputs"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "update_num_computed_tokens_for_batch_change"
        ]
        self.assertEqual(direct_correction, [], "input preparation must use the tested dependency-aware hook")

        tree_310 = ast.parse(RUNNER_310_PATH.read_text(encoding="utf-8"))
        runner_310 = next(
            node for node in tree_310.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner310"
        )
        self.assertTrue(any(isinstance(base, ast.Name) and base.id == "NPUModelRunner" for base in runner_310.bases))


if __name__ == "__main__":
    unittest.main()
