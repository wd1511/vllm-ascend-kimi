# SPDX-License-Identifier: Apache-2.0
"""A5 shared SiTU accuracy and latency coverage for the recipes operator."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SituAndMul

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedSparseSituAndMul
from vllm_ascend.ops.kimi_situ import initialize_kimi_situ_ops

BETA = 4.0
LINEAR_BETA = 25.0
GATE_UP_WIDTH = 1536
REPEATS = 20


@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_shared_sparse_situ_accuracy(tokens, record_property):
    if not torch.npu.is_available() or not get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP):
        pytest.skip("requires Ascend A5")
    initialize_kimi_situ_ops()
    if not hasattr(torch.ops.custom, "npu_situ_and_mul_sparse"):
        pytest.skip("requires the recipes sparse SiTU operator")

    torch.manual_seed(7)
    with set_current_vllm_config(VllmConfig()):
        layer = AscendSharedSparseSituAndMul(beta=BETA, linear_beta=LINEAR_BETA).npu()
    x = torch.randn(tokens, GATE_UP_WIDTH, device="npu", dtype=torch.bfloat16)
    assert layer.can_use_sparse(x)
    expected = SituAndMul.forward_native(layer, x)
    for _ in range(3):
        actual = layer.forward_native(x)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        layer.forward_native(x)
    end.record()
    end.synchronize()
    record_property("mean_ms", start.elapsed_time(end) / REPEATS)
