# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 component checks for Kimi combine -> RMSNorm/MX quant -> latent up.

Uses real NPU kernels and the production MXFP8 linear method with Kimi's
3584 -> 7168 dimensions. TP/EP communication and real checkpoints require
additional model tests. Latency measurements report a same-run baseline;
no speedup threshold is assumed before establishing a hardware baseline.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.quantization.method_adapters import AscendLinearMethod

LATENT_WIDTH = 3584
HIDDEN_WIDTH = 7168
MX_GROUP_SIZE = 32
EPSILON = 1e-5


def _require_a5():
    if not kimi_k3.get_current_hardware_profile().supports(kimi_k3.HardwareCapability.DYNAMIC_MX_QUANT_FUSION):
        pytest.skip("requires A5 RMSNormDynamicMxQuant support")
    if not hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant"):
        pytest.skip("requires TorchNPU RMSNormDynamicMxQuant operator")


@pytest.fixture(autouse=True)
def require_a5(monkeypatch):
    _require_a5()
    monkeypatch.setattr(
        kimi_k3,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=SimpleNamespace(fuse_norm_quant=True)),
    )


class _Norm(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.weight = nn.Parameter(gamma, requires_grad=False)
        self.variance_epsilon = EPSILON
        self.variance_size_override = None
        self.bias_loaded = False

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]


class _UpProj(nn.Module):
    def __init__(self):
        super().__init__()
        scheme = kimi_k3.AscendW8A8MXFP8DynamicLinearMethod.__new__(kimi_k3.AscendW8A8MXFP8DynamicLinearMethod)
        scheme.group_size = MX_GROUP_SIZE
        scheme.dynamic_mx_quant_scale_alg = 0
        generator = torch.Generator().manual_seed(17)
        values = torch.randn(HIDDEN_WIDTH, LATENT_WIDTH, generator=generator, dtype=torch.bfloat16).npu()
        quantized, scale = scheme.quantize_input(values)
        self.weight = nn.Parameter(quantized, requires_grad=False)
        self.weight_scale = nn.Parameter(scale.reshape(HIDDEN_WIDTH, -1), requires_grad=False)
        scheme.process_weights_after_loading(self)
        self.quant_method = AscendLinearMethod(scheme)

    def forward(self, x):
        return self.quant_method.apply(self, x), None


@pytest.fixture(scope="module")
def up_proj():
    _require_a5()
    return _UpProj()


def _input(rows, pattern="random", padded=False):
    generator = torch.Generator().manual_seed(31 + rows)
    width = HIDDEN_WIDTH if padded else LATENT_WIDTH
    values = torch.randn(rows, width, generator=generator)
    if pattern == "zero":
        values.zero_()
    elif pattern == "tiny":
        values *= 1e-7
    elif pattern == "large":
        values *= 1e3
    elif pattern == "boundary":
        # Adjacent BF16 values around MX exponent boundaries.
        values = torch.linspace(0.99609375, 1.0078125, rows * width).reshape(rows, width)
    return values.to(torch.bfloat16).npu()[:, :LATENT_WIDTH]


def _gamma():
    return torch.linspace(0.5, 1.5, LATENT_WIDTH, dtype=torch.bfloat16, device="npu")


def _dequantize(values, scale):
    # MX E8M0 scales are stored as exponent bytes, packed in pairs.
    exponents = scale.cpu().reshape(values.shape[0], -1).float() - 127
    scales = torch.exp2(exponents).repeat_interleave(MX_GROUP_SIZE, dim=-1)
    return values.float().cpu() * scales[:, :LATENT_WIDTH]


def _assert_relative_error(actual, expected, tolerance):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all()
    error = torch.linalg.vector_norm(actual - expected)
    reference = torch.linalg.vector_norm(expected)
    if reference == 0:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert error / reference <= tolerance


@pytest.mark.parametrize("rows", [1, 9, 65, 257])
@pytest.mark.parametrize("pattern", ["random", "zero", "tiny", "large", "boundary"])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@torch.inference_mode()
def test_rmsnorm_mx_quant_precision(rows, pattern, epsilon):
    x, gamma = _input(rows, pattern), _gamma()
    normalized = torch_npu.npu_rms_norm(x, gamma, epsilon)[0]
    reference_q, reference_scale = torch_npu.npu_dynamic_mx_quant(normalized, dst_type=torch.float8_e4m3fn, scale_alg=0)
    actual_q, actual_scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
        x, gamma, epsilon=epsilon, dst_type=torch.float8_e4m3fn, scale_alg=0
    )
    assert actual_q.dtype == torch.float8_e4m3fn
    assert actual_q.shape == x.shape
    assert actual_scale.dtype == reference_scale.dtype
    assert actual_scale.shape == reference_scale.shape == (rows, LATENT_WIDTH // (2 * MX_GROUP_SIZE), 2)
    # Fusion can move BF16 rounding across quantization. Compare reconstructed
    # activations, permitting one FP8 bin at a rounding boundary.
    actual = _dequantize(actual_q, actual_scale)
    expected = _dequantize(reference_q, reference_scale)
    torch.testing.assert_close(actual, expected, rtol=0.125, atol=1e-7)
    _assert_relative_error(actual, expected, 0.03)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 65])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_latent_up_precision_and_graph_replay(up_proj, rows, padded):
    norm = _Norm(_gamma())
    fused = kimi_k3.AscendKimiRoutedOutputTransform(norm, up_proj)
    reference = kimi_k3.KimiRoutedOutputTransform(norm, up_proj)
    x = _input(rows, padded=padded)
    assert fused._can_fuse_norm_quant(x), "test must exercise the fused branch"
    for _ in range(3):
        output = fused(x)
    torch.npu.synchronize()
    assert output.shape == (rows, HIDDEN_WIDTH)
    assert output.dtype == torch.bfloat16
    _assert_relative_error(output, reference(x), 0.04)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = fused(x)
    # Replay must read fresh inputs rather than retained quantized buffers.
    for pattern in ("tiny", "zero", "random"):
        x.copy_(_input(rows, pattern))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, fused(x), rtol=0, atol=0)
        _assert_relative_error(captured, reference(x), 0.04)


def _graph_latency_us(run, warmup=10, repeats=100):
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = run()
    for _ in range(warmup):
        graph.replay()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.npu.synchronize()
    assert torch.isfinite(output).all()
    return start.elapsed_time(end) * 1000 / repeats


@pytest.mark.parametrize("rows", [1, 8, 64, 256])
@torch.inference_mode()
def test_latent_up_latency(up_proj, rows, record_property):
    norm = _Norm(_gamma())
    fused = kimi_k3.AscendKimiRoutedOutputTransform(norm, up_proj)
    reference = kimi_k3.KimiRoutedOutputTransform(norm, up_proj)
    x = _input(rows)
    assert fused._can_fuse_norm_quant(x)
    # Alternate order across samples to reduce a fixed ordering bias.
    measurements = {"unfused": [], "fused": []}
    for sample in range(3):
        variants = (("unfused", reference), ("fused", fused))
        for name, transform in variants[:: 1 if sample % 2 == 0 else -1]:
            measurements[name].append(_graph_latency_us(partial(transform, x)))
    unfused_us, fused_us = (sorted(measurements[name])[1] for name in ("unfused", "fused"))
    assert unfused_us > 0 and fused_us > 0
    record_property("unfused_us", unfused_us)
    record_property("fused_us", fused_us)
    record_property("speedup", unfused_us / fused_us)
    print(f"rows={rows}: unfused={unfused_us:.3f}us fused={fused_us:.3f}us speedup={unfused_us / fused_us:.3f}")
