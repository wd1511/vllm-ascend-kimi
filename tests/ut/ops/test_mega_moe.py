from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMegaMoE
from vllm_ascend.quantization.quant_type import QuantType


def _make_moe_config(*, activation=MoEActivation.SITU):
    return SimpleNamespace(
        num_experts=8,
        num_local_experts=2,
        experts_per_token=2,
        activation=activation,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_limit=None,
        moe_parallel_config=SimpleNamespace(sp_size=1),
    )


def _make_fused_input(
    *,
    activation=MoEActivation.SITU,
    group_size=32,
    act_quant_type=torch.float8_e4m3fn,
    weight_quant_type=torch_npu.float4_e2m1fn_x2,
    dynamic_eplb=False,
    global_redundant_expert_num=0,
    lora_context=None,
    shared_weights=None,
    w1=None,
    w2=None,
):
    if w1 is None:
        w1 = torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8)
    if w2 is None:
        w2 = torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8)
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 128),
        topk_weights=torch.randn(4, 2, dtype=torch.bfloat16),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int64),
        w1=w1,
        w2=w2,
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=dynamic_eplb,
        global_redundant_expert_num=global_redundant_expert_num,
        activation=activation,
        mxfp_act_quant_type=act_quant_type,
        mxfp_weight_quant_type=weight_quant_type,
        mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_group_size=group_size,
        w1_scale=torch.ones(2, 128, 2, 2, dtype=torch.uint8),
        w2_scale=torch.ones(2, 128, 1, 2, dtype=torch.uint8),
        shared_weights=shared_weights,
        lora_context=lora_context,
    )


def _run_backend(mega_moe, *, activation=MoEActivation.SITU):
    group = SimpleNamespace(device_group=object())
    get_symm_buffer = MagicMock(return_value=object())
    vllm_config = SimpleNamespace()
    with (
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops",
            return_value=(get_symm_buffer, mega_moe),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_current_vllm_config",
            return_value=vllm_config,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=group,
        ),
    ):
        MegaMoEBackend(_make_moe_config(activation=activation)).fused_experts(_make_fused_input(activation=activation))
    return get_symm_buffer


def test_kimi_situ_uses_cann_activation_contract_and_logical_buffer_width():
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))

    get_symm_buffer = _run_backend(mega_moe)

    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 64
    assert "max_recv_token_num" not in get_symm_buffer.call_args.kwargs
    kwargs = mega_moe.call_args.kwargs
    assert kwargs["activation"] == "situglu"
    assert kwargs["activation_params"] == {"beta": 4.0, "linear_beta": 25.0}
    assert kwargs["activation_clamp"] is None
    assert kwargs["topk_ids"].dtype == torch.int32
    assert kwargs["topk_ids"].is_contiguous()
    assert kwargs["topk_weights"].dtype == torch.float32
    assert kwargs["topk_weights"].is_contiguous()


def test_kimi_k3_latent_layout_derives_logical_and_projected_widths():
    fused_input = build_fused_experts_input(
        hidden_states=torch.empty(1, 3584, device="meta"),
        topk_weights=torch.empty(1, 16, device="meta"),
        topk_ids=torch.empty(1, 16, dtype=torch.int64, device="meta"),
        w1=torch.empty(1, 6144, 1792, dtype=torch.uint8, device="meta"),
        w2=torch.empty(1, 3584, 1536, dtype=torch.uint8, device="meta"),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        activation=MoEActivation.SITU,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
        mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_group_size=32,
        w1_scale=torch.empty(1, 6144, 56, 2, dtype=torch.uint8, device="meta"),
        w2_scale=torch.empty(1, 3584, 48, 2, dtype=torch.uint8, device="meta"),
    )
    backend = MegaMoEBackend(SimpleNamespace(num_experts=896, num_local_experts=1, experts_per_token=16))

    layout = backend._validate_stacked_mxfp_layout(
        fused_input,
        [fused_input.weights.w1],
        [fused_input.weights.w2],
        [fused_input.weights.w1_scale],
        [fused_input.weights.w2_scale],
    )

    assert layout.logical_intermediate_hidden == 3072
    assert layout.projected_hidden == 6144


def test_swiglu_keeps_existing_contract_without_situ_parameters():
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))

    get_symm_buffer = _run_backend(mega_moe, activation=MoEActivation.SILU)

    assert mega_moe.call_args.kwargs["activation"] == "swiglu"
    assert "activation_params" not in mega_moe.call_args.kwargs
    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 64


def test_native_fp4_weights_use_uint8_carrier_at_cann_boundary():
    packed_w1 = torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8)
    native_w1 = packed_w1.view(torch.float4_e2m1fn_x2)
    fused_input = _make_fused_input(w1=native_w1)
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))
    group = SimpleNamespace(device_group=object())

    with (
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops",
            return_value=(MagicMock(return_value=object()), mega_moe),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=group,
        ),
    ):
        MegaMoEBackend(_make_moe_config()).fused_experts(fused_input)

    cann_w1 = mega_moe.call_args.kwargs["l1_weights"][0]
    assert cann_w1.dtype == torch.uint8
    assert cann_w1.data_ptr() == native_w1.data_ptr()
    assert mega_moe.call_args.kwargs["weight1_type"] == torch_npu.float4_e2m1fn_x2


def test_mega_moe_passes_compatible_shared_expert_weights():
    shared_weights = MoEWeights(
        w1=[torch.randint(0, 255, (1, 128, 64), dtype=torch.uint8)],
        w2=[torch.randint(0, 255, (1, 128, 32), dtype=torch.uint8)],
        w1_scale=[torch.ones(1, 128, 2, 2, dtype=torch.uint8)],
        w2_scale=[torch.ones(1, 128, 1, 2, dtype=torch.uint8)],
        quant_type=QuantType.W4A8MXFP,
    )
    fused_input = _make_fused_input(shared_weights=shared_weights)
    expected_output = torch.randn(4, 128)
    mega_moe = MagicMock(return_value=(expected_output, None))
    backend = MegaMoEBackend(_make_moe_config())
    backend._supports_shared_experts = True
    group = SimpleNamespace(device_group=object())

    with (
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops",
            return_value=(MagicMock(return_value=object()), mega_moe),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=group,
        ),
    ):
        output, expert_tokens, shared_fused = backend.fused_experts(fused_input)

    assert output is expected_output
    assert expert_tokens is None
    assert shared_fused is True
    kwargs = mega_moe.call_args.kwargs
    assert kwargs["shared_l1_weights"] is shared_weights.w1
    assert kwargs["shared_l2_weights"] is shared_weights.w2
    assert kwargs["shared_l1_weights_sf"][0].dtype == torch.float8_e8m0fnu
    assert kwargs["shared_l2_weights_sf"][0].dtype == torch.float8_e8m0fnu
    assert kwargs["shared_weight1_type"] == torch_npu.float4_e2m1fn_x2


@pytest.mark.parametrize("missing_attr", ["activation_situ_beta", "activation_situ_linear_beta"])
def test_kimi_situ_requires_both_activation_parameters(missing_attr):
    backend = MegaMoEBackend(_make_moe_config())
    setattr(backend.moe_config, missing_attr, None)

    with pytest.raises(ValueError, match="requires activation_situ_beta"):
        backend._resolve_activation(MoEActivation.SITU)


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"group_size": 64}, "group_size=32"),
        ({"act_quant_type": torch.float16}, "FP8 E4M3"),
        ({"weight_quant_type": torch.float8_e4m3fn}, "FP4 E2M1"),
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"global_redundant_expert_num": 1}, "redundant physical experts"),
        ({"lora_context": object()}, "MoE LoRA"),
    ],
)
def test_mega_moe_rejects_unsupported_runtime_contract(input_kwargs, message):
    backend = MegaMoEBackend(_make_moe_config())

    with pytest.raises(RuntimeError, match=message):
        backend.fused_experts(_make_fused_input(**input_kwargs))


def test_kimi_situ_rejects_old_cann_signature():
    mega_moe = MagicMock(side_effect=TypeError("unexpected keyword argument 'activation_params'"))

    with pytest.raises(RuntimeError, match="Kimi K3 SiTU activation_params"):
        _run_backend(mega_moe)


def test_mega_moe_prepare_preserves_local_batch_when_dp_tokens_differ():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    prepare_finalize.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(sp_size=1),
    )
    hidden_states = torch.randn(4, 128)
    router_logits = torch.randn(4, 8)
    input_ids = torch.tensor([1, 2, 3, 4])

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=6),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
    ):
        prepared = prepare_finalize.prepare(hidden_states, router_logits)

    assert prepared.hidden_states is hidden_states
    assert prepared.router_logits is router_logits
    assert prepare_finalize.pad_and_split_input_ids(input_ids) is input_ids

    output = torch.randn(4, 128)
    torch.testing.assert_close(
        prepare_finalize.finalize(output, reduce_results=False),
        output,
    )


def test_mega_moe_prepare_uses_sequence_parallel_local_token_count():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    prepare_finalize.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(sp_size=8),
    )
    hidden_states = torch.randn(100, 128)
    router_logits = torch.randn(100, 8)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=1024),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=256,
        ) as get_capacity,
    ):
        prepared = prepare_finalize.prepare(hidden_states, router_logits)

    assert prepared.hidden_states is hidden_states
    assert prepared.router_logits is router_logits
    get_capacity.assert_called_once_with(
        ANY,
        sp_size=8,
    )


@pytest.mark.parametrize("num_tokens", [0, 9])
def test_mega_moe_prepare_rejects_input_outside_buffer_capacity(num_tokens):
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    prepare_finalize.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(sp_size=1))
    hidden_states = torch.randn(num_tokens, 128)
    router_logits = torch.randn(num_tokens, 8)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        pytest.raises(ValueError, match="1 to buffer_tokens_per_rank tokens"),
    ):
        prepare_finalize.prepare(hidden_states, router_logits)
