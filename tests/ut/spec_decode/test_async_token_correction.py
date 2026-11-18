# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for async token correction, without vLLM/NPU imports.

Run directly with ``python tests/ut/spec_decode/test_async_token_correction.py``.
The function under test is loaded from this checkout's production utils.py.
"""

import ast
import unittest
from pathlib import Path

import torch


def _load_token_correction():
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/utils.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    name = "update_num_computed_tokens_for_batch_change"
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {"torch": torch}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace[name]


class TestAsyncTokenCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.correct = staticmethod(_load_token_correction())

    def test_reordered_drafts_use_previous_gpu_counts_and_other_rows_use_cpu(self):
        computed = torch.tensor([10, 20, 30, 40, 999], dtype=torch.int32)
        accepted = torch.tensor([99, 98, 97, 96], dtype=torch.int32)
        self.correct(
            num_computed_tokens=computed,
            num_accepted_tokens=accepted,
            prev_positions=torch.tensor([2, -1, 1, 0], dtype=torch.int64),
            valid_sampled_token_count=torch.tensor([2, 1, 3, 1], dtype=torch.int32),
            prev_num_draft_tokens=torch.tensor([3, 0, 3, 0], dtype=torch.int32),
            cpu_num_computed_tokens=torch.tensor([900, 7, 22, 800], dtype=torch.int32),
        )
        # Old row 2 moves to row 0; new row and non-draft old row accept
        # CPU values. The unused tail of the persistent buffer is unchanged.
        torch.testing.assert_close(computed, torch.tensor([33, 7, 22, 12, 999], dtype=torch.int32))
        torch.testing.assert_close(accepted, torch.tensor([3, 1, 1, 2], dtype=torch.int32))

    def test_nonparticipating_rows_ignore_stale_accepted_counts(self):
        computed = torch.tensor([10, 20, 30], dtype=torch.int32)
        accepted = torch.tensor([8, 9, 10], dtype=torch.int32)
        self.correct(
            num_computed_tokens=computed,
            num_accepted_tokens=accepted,
            prev_positions=torch.tensor([-1, 1, -1], dtype=torch.int64),
            valid_sampled_token_count=torch.tensor([4, 5, 6], dtype=torch.int32),
            prev_num_draft_tokens=torch.zeros(3, dtype=torch.int32),
            cpu_num_computed_tokens=torch.tensor([101, 202, 303], dtype=torch.int32),
        )
        torch.testing.assert_close(computed, torch.tensor([101, 202, 303], dtype=torch.int32))
        torch.testing.assert_close(accepted, torch.ones(3, dtype=torch.int32))

    def test_computed_token_correction_does_not_require_accepted_count_publication(self):
        computed = torch.tensor([10, 20, 30, 40, 999], dtype=torch.int32)
        self.correct(
            num_computed_tokens=computed,
            num_accepted_tokens=None,
            prev_positions=torch.tensor([2, -1, 1, 0], dtype=torch.int64),
            valid_sampled_token_count=torch.tensor([2, 1, 3, 1], dtype=torch.int32),
            prev_num_draft_tokens=torch.tensor([3, 0, 3, 0], dtype=torch.int32),
            cpu_num_computed_tokens=torch.tensor([900, 7, 22, 800], dtype=torch.int32),
        )
        torch.testing.assert_close(computed, torch.tensor([33, 7, 22, 12, 999], dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
