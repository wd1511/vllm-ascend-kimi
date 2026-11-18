# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3


@pytest.fixture
def routed_output_setup(monkeypatch):
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.ones(64, dtype=torch.bfloat16))
    norm.variance_epsilon = 1e-5
    norm.variance_size_override = None
    norm.bias_loaded = False
    normalized = torch.randn(3, 64, dtype=torch.bfloat16)
    norm.forward = MagicMock(return_value=normalized)
    scheme = kimi_k3.AscendW8A8MXFP8DynamicLinearMethod.__new__(kimi_k3.AscendW8A8MXFP8DynamicLinearMethod)
    scheme.group_size = 32
    scheme.dynamic_mx_quant_scale_alg = 0
    up_proj = nn.Module()
    up_proj.weight = nn.Parameter(torch.ones(128, 64, dtype=torch.bfloat16))
    up_proj.quant_method = SimpleNamespace(quant_method=scheme)
    output = torch.randn(3, 128, dtype=torch.bfloat16)
    up_proj.forward = MagicMock(return_value=(output, None))
    quantized = torch.zeros(3, 64, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(3, 1, 2, dtype=torch.uint8)
    fused_op = MagicMock(return_value=(quantized, scale, torch.empty(3, 1)))
    monkeypatch.setattr(torch.ops, "npu", SimpleNamespace(npu_rms_norm_dynamic_mx_quant=fused_op))
    config = SimpleNamespace(ascend_compilation_config=SimpleNamespace(fuse_norm_quant=True))
    profile = SimpleNamespace(supports=MagicMock(return_value=True))
    monkeypatch.setattr(kimi_k3, "get_ascend_config", lambda: config)
    monkeypatch.setattr(kimi_k3, "get_current_hardware_profile", lambda: profile)
    # Device metadata is mocked; the operator itself receives a CPU tensor.
    # Numerical kernel and graph checks live in the NPU test.
    hidden_states = MagicMock(spec=torch.Tensor)
    hidden_states.device = SimpleNamespace(type="npu")
    hidden_states.dtype = torch.bfloat16
    hidden_states.ndim = 2
    hidden_states.shape = normalized.shape
    hidden_states.numel.return_value = normalized.numel()
    hidden_states.contiguous.return_value = normalized
    return SimpleNamespace(
        norm=norm,
        up_proj=up_proj,
        scheme=scheme,
        normalized=normalized,
        output=output,
        quantized=quantized,
        scale=scale,
        fused_op=fused_op,
        config=config,
        profile=profile,
        hidden_states=hidden_states,
    )


@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 2.3e-5])
def test_routed_output_fuses_norm_quant_once(routed_output_setup, epsilon):
    setup = routed_output_setup
    setup.norm.variance_epsilon = epsilon
    transform = kimi_k3.AscendKimiRoutedOutputTransform(setup.norm, setup.up_proj)

    assert transform(setup.hidden_states) is setup.output

    setup.norm.forward.assert_not_called()
    setup.fused_op.assert_called_once_with(
        setup.normalized,
        setup.norm.weight,
        epsilon=epsilon,
        scale_alg=0,
        dst_type=torch.float8_e4m3fn,
    )
    setup.up_proj.forward.assert_called_once()
    values, scale = setup.up_proj.forward.call_args.args[0]
    assert values is setup.quantized
    assert scale is setup.scale
    setup.profile.supports.assert_called_once_with(kimi_k3.HardwareCapability.DYNAMIC_MX_QUANT_FUSION)


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "hardware",
        "unquantized",
        "other_scheme",
        "group_size",
        "scale_alg",
        "missing_operator",
        "no_norm",
    ],
)
def test_routed_output_unsupported_configuration_falls_back(routed_output_setup, monkeypatch, case):
    setup = routed_output_setup
    norm = setup.norm
    if case == "disabled":
        setup.config.ascend_compilation_config.fuse_norm_quant = False
    elif case == "hardware":
        setup.profile.supports.return_value = False
    elif case == "unquantized":
        setup.up_proj.quant_method = None
    elif case == "other_scheme":
        setup.up_proj.quant_method.quant_method = object()
    elif case == "group_size":
        setup.scheme.group_size = 64
    elif case == "scale_alg":
        setup.scheme.dynamic_mx_quant_scale_alg = 1
    elif case == "missing_operator":
        monkeypatch.delattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")
    elif case == "no_norm":
        norm = None
    transform = kimi_k3.AscendKimiRoutedOutputTransform(norm, setup.up_proj)

    assert transform(setup.hidden_states) is setup.output

    setup.fused_op.assert_not_called()
    expected_input = setup.hidden_states if norm is None else setup.normalized
    assert setup.up_proj.forward.call_args.args[0] is expected_input
    if norm is not None:
        setup.norm.forward.assert_called_once_with(setup.hidden_states)


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "fp16",
        "fp32",
        "empty",
        "rank",
        "alignment",
        "weight_shape",
        "weight_layout",
        "loaded_bias",
        "partial_variance",
    ],
)
def test_routed_output_unsupported_input_falls_back(routed_output_setup, case):
    setup = routed_output_setup
    transform = kimi_k3.AscendKimiRoutedOutputTransform(setup.norm, setup.up_proj)
    hidden_states = setup.hidden_states
    if case == "cpu":
        hidden_states = setup.normalized
    elif case in ("fp16", "fp32"):
        hidden_states.dtype = torch.float16 if case == "fp16" else torch.float32
    elif case == "empty":
        hidden_states.shape = torch.Size((0, 64))
        hidden_states.numel.return_value = 0
    elif case == "rank":
        hidden_states.ndim = 3
    elif case == "alignment":
        hidden_states.shape = torch.Size((3, 63))
    elif case == "weight_shape":
        setup.norm.weight = nn.Parameter(torch.ones(128))
    elif case == "weight_layout":
        setup.norm.weight = nn.Parameter(torch.ones(128)[::2])
    elif case == "loaded_bias":
        # Checkpoint loading happens after transform construction.
        setup.norm.bias_loaded = True
    elif case == "partial_variance":
        setup.norm.variance_size_override = 32

    assert transform(hidden_states) is setup.output

    setup.fused_op.assert_not_called()
    setup.norm.forward.assert_called_once_with(hidden_states)
    assert setup.up_proj.forward.call_args.args[0] is setup.normalized


def test_routed_output_truncated_padding_is_contiguous_for_fused_op(routed_output_setup):
    setup = routed_output_setup
    padded = torch.randn(3, 128, dtype=torch.bfloat16)
    truncated = padded[:, :64]
    assert not truncated.is_contiguous()
    setup.hidden_states.contiguous.side_effect = truncated.contiguous
    transform = kimi_k3.AscendKimiRoutedOutputTransform(setup.norm, setup.up_proj)

    transform(setup.hidden_states)

    op_input = setup.fused_op.call_args.args[0]
    assert op_input.is_contiguous()
    torch.testing.assert_close(op_input, truncated)


def test_routed_output_preserves_parameters_and_names(routed_output_setup):
    setup = routed_output_setup
    original = kimi_k3.KimiRoutedOutputTransform(setup.norm, setup.up_proj)
    fused = kimi_k3.AscendKimiRoutedOutputTransform(setup.norm, setup.up_proj)
    expected = dict(original.named_parameters())
    actual = dict(fused.named_parameters())

    assert actual.keys() == expected.keys()
    assert all(actual[name] is parameter for name, parameter in expected.items())
    assert fused.norm is setup.norm
    assert fused.up_proj is setup.up_proj


def test_routed_output_does_not_hide_operator_failures(routed_output_setup):
    setup = routed_output_setup
    transform = kimi_k3.AscendKimiRoutedOutputTransform(setup.norm, setup.up_proj)
    setup.fused_op.side_effect = RuntimeError("fused kernel failed")

    with pytest.raises(RuntimeError, match="fused kernel failed"):
        transform(setup.hidden_states)

    setup.norm.forward.assert_not_called()
    setup.up_proj.forward.assert_not_called()


@pytest.mark.parametrize("use_norm", [False, True])
def test_kimi_moe_installs_routed_output_transform(routed_output_setup, monkeypatch, use_norm):
    setup = routed_output_setup
    setup.config.multistream_overlap_shared_expert = True
    factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(kimi_k3, "AscendKimiGateLinear", MagicMock(return_value=nn.Module()))
    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", MagicMock(side_effect=[nn.Identity(), setup.up_proj]))
    monkeypatch.setattr(kimi_k3, "RMSNorm", MagicMock(return_value=setup.norm))
    monkeypatch.setattr(kimi_k3, "FusedMoEFactory", factory)
    config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=64,
        num_experts=8,
        num_experts_per_token=2,
        routed_expert_hidden_size=64,
        latent_moe_use_norm=use_norm,
        routed_scaling_factor=1.0,
        num_shared_experts=None,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        moe_renormalize=True,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        moe_router_activation_func="sigmoid",
    )

    moe = kimi_k3.AscendKimiMoE(config=config)

    transform = moe.routed_output_transform
    assert type(transform) is kimi_k3.AscendKimiRoutedOutputTransform
    assert transform.norm is (setup.norm if use_norm else None)
    assert transform.up_proj is setup.up_proj
    assert factory.call_args.kwargs["routed_output_transform"] is transform
    assert moe.routed_expert_up_proj is setup.up_proj
