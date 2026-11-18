#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import unittest
from unittest import mock
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear, AscendKimiGateLinear


class TestAscendGateLinear(TestBase):
    def setUp(self):
        super().setUp()

        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 1
        self.mock_group.rank_in_group = 0

        self.patches = [
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

        super().tearDown()

    def test_forward_keeps_router_logits_fp32(self):
        gate = AscendGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            prefix="test.gate",
        )

        self.assertEqual(gate.weight.dtype, torch.float32)

        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        output, output_bias = gate(hidden_states)

        self.assertEqual(output.dtype, torch.float32)
        self.assertIsNone(output_bias)


class TestAscendKimiGateLinear(TestBase):
    def setUp(self):
        super().setUp()

        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 1
        self.mock_group.rank_in_group = 0

        self.patches = [
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

        super().tearDown()

    def test_forward_uses_bf16_router_gemm(self):
        gate = AscendKimiGateLinear(
            input_size=16,
            output_size=4,
            bias=False,
            prefix="test.gate",
        )

        self.assertEqual(gate.weight.dtype, torch.bfloat16)

        hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        output, output_bias = gate(hidden_states)

        self.assertEqual(output.dtype, torch.float32)
        self.assertIsNone(output_bias)
        expected = torch.mm(hidden_states.float(), gate.weight.float().t())
        torch.testing.assert_close(output, expected)

    def test_npu_bf16_inputs_reach_router_gemm_without_cast(self):
        gate = AscendKimiGateLinear(input_size=16, output_size=4, prefix="test.gate")
        hidden_states = mock.MagicMock(spec=torch.Tensor)
        hidden_states.device.type = "npu"
        hidden_states.dtype = torch.bfloat16
        expected = torch.randn(2, 4, dtype=torch.float32)

        with patch("vllm_ascend.ops.fused_moe.gate_linear.torch.mm", return_value=expected) as gemm:
            output, bias = gate(hidden_states)

        self.assertIs(output, expected)
        self.assertIsNone(bias)
        gemm.assert_called_once()
        args, kwargs = gemm.call_args
        self.assertIs(args[0], hidden_states)
        self.assertEqual(args[1].dtype, torch.bfloat16)
        self.assertEqual(kwargs, {"out_dtype": torch.float32})
        hidden_states.float.assert_not_called()
        hidden_states.to.assert_not_called()

    def test_npu_other_input_dtype_delegates_to_upstream(self):
        gate = AscendKimiGateLinear(input_size=16, output_size=4, prefix="test.gate")
        hidden_states = mock.MagicMock(spec=torch.Tensor)
        hidden_states.device.type = "npu"
        hidden_states.dtype = torch.float32
        expected = (torch.randn(2, 4), None)

        with patch(
            "vllm_ascend.ops.fused_moe.gate_linear.GateLinear.forward", return_value=expected
        ) as upstream_forward:
            self.assertIs(gate(hidden_states), expected)

        upstream_forward.assert_called_once_with(hidden_states)


if __name__ == "__main__":
    unittest.main()
