# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for the input-preparation event-wait relocation.

Run directly with Python, torch, and numpy; vLLM and NPU hardware are not
required. The tested segment and request-order remapping method are extracted
from this checkout's production source. The correction function is a mocked
boundary: these tests check its original tensor inputs and publication order,
without replacing or reproducing its implementation.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

SOURCE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"


def _is_attribute(node, name):
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Attribute):
        parent, separator, attribute = name.rpartition(".")
        return bool(separator) and node.attr == attribute and _is_attribute(node.value, parent)
    return False


def _calls(node, function_name):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _is_attribute(child.func, function_name)
    ]


def _load_production_segment():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    runner_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    methods = {node.name: node for node in runner_node.body if isinstance(node, ast.FunctionDef)}
    prepare = methods["_prepare_inputs"]
    start = next(
        index
        for index, node in enumerate(prepare.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and _is_attribute(node.targets[0].value, "self.req_indices.np")
    )
    end = next(
        index
        for index, node in enumerate(prepare.body)
        if isinstance(node, ast.Assign) and _is_attribute(node.targets[0], "dcp_manager")
    )
    if start >= end:
        raise AssertionError("the independent input copies must precede DCP preparation")
    # Keep the production statements contiguous, including every branch between
    # the independent copies and the first possible computed-token consumer.
    segment = prepare.body[start:end]
    wrapper = ast.parse(
        "def run_segment(self, num_reqs, total_num_scheduled_tokens, req_indices, "
        "num_scheduled_tokens, prev_req_id_to_index, prev_positions_gpu):\n    pass\n"
    ).body[0]
    wrapper.body = segment
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            methods["_sync_num_accepted_tokens"],
            wrapper,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE_PATH), "exec"), namespace)
    runner_type = type(
        "InputPrepRunner",
        (),
        {name: namespace[name] for name in ("_sync_num_accepted_tokens", "run_segment")},
    )
    return runner_type, namespace, methods, segment


class _Buffer:
    def __init__(self, name, values, operations, ready=None):
        self.name = name
        self.cpu = torch.tensor(values, dtype=torch.int32)
        self.gpu = self.cpu.clone()
        self.operations = operations
        self.ready = ready or (lambda: True)
        self.upload_count = 0

    @property
    def np(self):
        if not self.ready():
            raise AssertionError("accepted-token snapshot accessed before the CPU event wait")
        return self.cpu.numpy()

    def copy_to_gpu(self, num_items=None):
        if not self.ready():
            raise AssertionError("accepted-token snapshot uploaded before the CPU event wait")
        self.operations.append(f"{self.name}.upload")
        self.upload_count += 1
        self.gpu[:num_items].copy_(self.cpu[:num_items])


class TestInputPrepSynchronization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner_type, cls.namespace, cls.methods, cls.segment = _load_production_segment()

    def _exercise(
        self,
        previous,
        expected_accepted,
        *,
        asynchronous=True,
        event=True,
        mapping=True,
        correction=True,
        previous_positions_uploaded=False,
    ):
        runner = self.runner_type()
        operations = []
        completed = [not event]
        num_reqs = len(previous)
        num_scheduled_tokens = np.array([2] * num_reqs, dtype=np.int32)
        req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), num_scheduled_tokens)
        total_tokens = len(req_indices)
        runner.device = torch.device("cpu")
        runner.use_async_scheduling = asynchronous
        runner.use_async_spec_decode = correction
        runner.num_accepted_tokens = _Buffer("accepted", [2, 4, 7, 8], operations, lambda: completed[0])
        runner.prev_positions = _Buffer("previous_positions", previous, operations)
        runner.prev_num_draft_tokens = _Buffer("previous_drafts", [3, 3, 3, 0], operations)
        runner.req_indices = _Buffer("req_indices", [0] * total_tokens, operations)
        runner.query_pos = _Buffer("query_pos", [0, 1] * num_reqs, operations)
        runner.num_scheduled_tokens = _Buffer("scheduled_tokens", [0] * 4, operations)
        runner.num_computed_tokens = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        runner.valid_sampled_token_count_gpu = torch.tensor([2, 4, 3, 1], dtype=torch.int32)
        cpu_computed = torch.tensor([101, 202, 303, 404], dtype=torch.int32)
        runner.input_batch = SimpleNamespace(
            num_accepted_tokens_cpu=np.array([9, 8, 7, 6], dtype=np.int32),
            num_computed_tokens_cpu_tensor=cpu_computed,
        )
        previous_mapping = {"previous-request": 0} if mapping else {}
        runner.num_accepted_tokens_event = MagicMock() if event else None
        if event:

            def finish_previous_step():
                operations.append("accepted.wait")
                completed[0] = True

            runner.num_accepted_tokens_event.synchronize.side_effect = finish_previous_step

        original_remap = runner._sync_num_accepted_tokens

        def remap(*args, **kwargs):
            operations.append("accepted.remap")
            return original_remap(*args, **kwargs)

        runner._sync_num_accepted_tokens = MagicMock(side_effect=remap)

        def correction_boundary(*args, **kwargs):
            self.assertEqual(len(args), 6, "the original correction call has six positional tensor inputs")
            self.assertEqual(kwargs, {})
            self.assertTrue(all(isinstance(argument, torch.Tensor) for argument in args))
            self.assertIs(args[0], runner.num_computed_tokens)
            self.assertEqual(args[1].data_ptr(), runner.num_accepted_tokens.gpu.data_ptr())
            self.assertEqual(tuple(args[1].shape), (num_reqs,))
            self.assertEqual(args[2].data_ptr(), runner.prev_positions.gpu.data_ptr())
            self.assertEqual(tuple(args[2].shape), (num_reqs,))
            self.assertIs(args[3], runner.valid_sampled_token_count_gpu)
            self.assertIs(args[4], runner.prev_num_draft_tokens.gpu)
            self.assertEqual(args[5].data_ptr(), cpu_computed.data_ptr())
            torch.testing.assert_close(args[5], cpu_computed[:num_reqs])
            torch.testing.assert_close(args[1], torch.tensor(expected_accepted, dtype=torch.int32))
            if event:
                self.assertIn("accepted.upload", operations)
            else:
                np.testing.assert_array_equal(runner.num_accepted_tokens.np, [1, 1, 1, 1])
            operations.append("correction")

        correction_call = MagicMock(side_effect=correction_boundary)
        self.namespace["update_num_computed_tokens_for_batch_change"] = correction_call
        previous_gpu = runner.prev_positions.gpu[:num_reqs] if previous_positions_uploaded else None
        runner.run_segment(
            num_reqs, total_tokens, req_indices, num_scheduled_tokens, previous_mapping, previous_gpu
        )

        independent = ["req_indices.upload", "query_pos.upload", "scheduled_tokens.upload"]
        self.assertEqual(operations[:3], independent)
        for buffer in (runner.req_indices, runner.query_pos, runner.num_scheduled_tokens):
            self.assertEqual(buffer.upload_count, 1)
        torch.testing.assert_close(runner.req_indices.gpu, torch.from_numpy(req_indices))
        torch.testing.assert_close(runner.num_scheduled_tokens.gpu[:num_reqs], torch.from_numpy(num_scheduled_tokens))
        expected_operations = independent.copy()
        if event:
            expected_operations.extend(["accepted.wait", "accepted.remap", "accepted.upload"])
            runner.num_accepted_tokens_event.synchronize.assert_called_once_with()
            runner._sync_num_accepted_tokens.assert_called_once_with(num_reqs, has_prev_mapping=mapping)
            self.assertEqual(runner.num_accepted_tokens.upload_count, 1)
        else:
            runner._sync_num_accepted_tokens.assert_not_called()
            self.assertEqual(runner.num_accepted_tokens.upload_count, 0)
        should_correct = correction and mapping
        if should_correct:
            if not previous_positions_uploaded:
                expected_operations.append("previous_positions.upload")
            expected_operations.extend(["previous_drafts.upload", "correction"])
            correction_call.assert_called_once()
        else:
            correction_call.assert_not_called()
            torch.testing.assert_close(runner.num_computed_tokens[:num_reqs], cpu_computed[:num_reqs])
        self.assertEqual(operations, expected_operations)
        np.testing.assert_array_equal(runner.num_accepted_tokens.np[:num_reqs], expected_accepted)
        np.testing.assert_array_equal(runner.num_accepted_tokens.np[num_reqs:], np.ones(4 - num_reqs, dtype=np.int32))
        torch.testing.assert_close(
            runner.num_accepted_tokens.gpu[:num_reqs], torch.tensor(expected_accepted, dtype=torch.int32)
        )

    def test_async_reorder_new_and_condense_keep_wait_publish_correction_order(self):
        for name, previous, accepted in (
            ("reorder", [1, 0], [4, 2]),
            ("new_request", [1, -1, 0], [4, 1, 2]),
            ("condense", [2, 0], [7, 2]),
        ):
            for already_uploaded in (False, True):
                with self.subTest(case=name, previous_positions_uploaded=already_uploaded):
                    self._exercise(previous, accepted, previous_positions_uploaded=already_uploaded)

    def test_synchronous_path_preserves_current_batch_counts(self):
        self._exercise([1, 0], [9, 8], asynchronous=False, correction=False)

    def test_missing_previous_mapping_preserves_ones_and_cpu_computed_fallback(self):
        self._exercise([1, 0], [1, 1], mapping=False)

    def test_no_event_preserves_original_ones_fallback(self):
        for correction in (False, True):
            with self.subTest(correction=correction):
                self._exercise([1, 0], [1, 1], event=False, correction=correction)

    def test_real_prepare_and_execute_keep_original_inline_interfaces(self):
        prepare = self.methods["_prepare_inputs"]
        execute = self.methods["execute_model"]
        self.assertEqual(len(_calls(prepare, "self.num_accepted_tokens_event.synchronize")), 1)
        self.assertEqual(len(_calls(prepare, "self._sync_num_accepted_tokens")), 1)
        calls = _calls(prepare, "update_num_computed_tokens_for_batch_change")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 6)
        self.assertFalse(
            any(isinstance(argument, ast.Constant) and argument.value is None for argument in calls[0].args)
        )
        self.assertEqual(_calls(execute, "self.num_accepted_tokens_event.synchronize"), [])
        self.assertEqual(_calls(execute, "self._sync_num_accepted_tokens"), [])
        self.assertNotIn("_publish_num_accepted_tokens_before_mamba", self.methods)
        self.assertNotIn("_update_num_computed_tokens_for_async_batch", self.methods)
        dcp_calls = _calls(prepare, "dcp_manager.rebuild_async_spec_decode_inputs")
        self.assertEqual(len(dcp_calls), 1)
        self.assertLess(calls[0].lineno, dcp_calls[0].lineno)


if __name__ == "__main__":
    unittest.main()
