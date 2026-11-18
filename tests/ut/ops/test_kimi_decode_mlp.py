# SPDX-License-Identifier: Apache-2.0
"""The shared-down callback must run between GMM1 and routed activation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe import moe_mlp
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def test_mxfp_situ_releases_shared_down_before_routed_activation():
    order = []
    gate_up = torch.ones(2, 8, dtype=torch.bfloat16)
    activated = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 1, 2)
    expected = torch.ones(2, 4, dtype=torch.bfloat16)

    def gmm1(**kwargs):
        order.append("gmm1")
        return [gate_up]

    def shared_down(event):
        order.append("shared_down")
        assert event is done

    def activation(*args, **kwargs):
        order.append("routed_activation")
        return activated, scale

    def gmm2(**kwargs):
        order.append("gmm2")
        return expected

    done = object()
    with (
        patch.object(moe_mlp, "ASCEND_DEVICE_TYPE", AscendDeviceType.A5),
        patch.object(moe_mlp, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.MC2)),
        patch.object(moe_mlp.torch_npu, "npu_grouped_matmul", side_effect=gmm1),
        patch.object(moe_mlp.torch.npu, "current_stream", return_value=MagicMock(record_event=lambda: done)),
        patch.object(DeviceOperator, "maybe_normalize_mxfp_scale_layout", side_effect=lambda x: x),
        patch.object(DeviceOperator, "npu_grouped_matmul_gmm2", side_effect=gmm2),
        patch.object(moe_mlp, "dispose_tensor"),
        patch.object(moe_mlp.torch.ops, "custom", SimpleNamespace(grouped_situ_mx_quant=activation)),
    ):
        output, _, gmm1_done = moe_mlp.quant_apply_mlp(
            hidden_states=activated,
            w1=torch.ones(2, 4, 4),
            w2=torch.ones(2, 2, 4),
            w1_scale=torch.ones(2, 4),
            w2_scale=torch.ones(2, 2),
            group_list=torch.tensor([1, 1]),
            group_list_type=1,
            dynamic_scale=scale,
            use_mxfp_quant=True,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
            activation=MoEActivation.SITU,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            after_gmm1=shared_down,
        )
    assert output is expected
    assert gmm1_done is done
    assert order == ["gmm1", "shared_down", "routed_activation", "gmm2"]
