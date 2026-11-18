# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import routed_experts
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("comm_type", list(MoECommType))
@pytest.mark.parametrize("num_prefills", [0, 1], ids=["decode", "prefill"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.uint8, torch.float8_e4m3fn])
def test_selected_weights_follow_communication_backend(monkeypatch, comm_type, num_prefills, dtype):
    context = SimpleNamespace(moe_comm_type=comm_type, attn_metadata=SimpleNamespace(num_prefills=num_prefills))
    monkeypatch.setattr(ascend_forward_context, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_experts, "get_moe_num_logical_experts", lambda *args, **kwargs: 2)
    weights = torch.tensor([[0.123456, 0.876544]], dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    layer = SimpleNamespace(
        router=SimpleNamespace(_select_experts=lambda **kwargs: (weights, ids)),
        log2phy=None,
        n_shared_experts=None,
        moe_config=SimpleNamespace(num_experts=2),
        global_redundant_expert_num=0,
    )
    actual, actual_ids = AscendRoutedExperts._select_experts(
        layer, torch.empty(1, 4, dtype=dtype), torch.zeros(1, 2), False
    )
    preserve_dtype = comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2) or dtype in (
        torch.uint8,
        torch.float8_e4m3fn,
    )
    expected = weights if preserve_dtype else weights.to(dtype)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert actual_ids is ids
    if preserve_dtype:
        assert actual is weights
    elif dtype in (torch.bfloat16, torch.float16):
        assert not torch.equal(actual.float(), weights)


@pytest.mark.parametrize("comm_type", list(MoECommType))
def test_native_router_does_not_round_before_mc2(monkeypatch, comm_type):
    monkeypatch.setattr(ascend_forward_context, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=comm_type))
    layer = SimpleNamespace(
        scoring_func="sigmoid",
        use_grouped_topk=False,
        e_score_correction_bias=None,
        top_k=2,
        routed_scaling_factor=1.0,
        _renormalize_topk_weights=lambda weights: weights,
    )
    logits = torch.tensor([[0.123456, -0.654321]], dtype=torch.float32)
    actual, ids = AscendGroupedTopKRouter._compute_routing(layer, torch.zeros(1, 4, dtype=torch.bfloat16), logits, None)
    expected, expected_ids = logits.sigmoid().topk(2, dim=-1)
    if comm_type not in (MoECommType.MC2, MoECommType.FUSED_MC2):
        expected = expected.bfloat16()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert torch.equal(ids, expected_ids.to(torch.int32))


@pytest.mark.parametrize("comm_type", list(MoECommType))
def test_mxfp4_scheme_preserves_mc2_weights(monkeypatch, comm_type):
    communication = SimpleNamespace(fused_experts=lambda fused_experts_input: fused_experts_input)
    context = SimpleNamespace(moe_comm_type=comm_type, moe_comm_method=communication)
    monkeypatch.setattr(ascend_forward_context, "_EXTRA_CTX", context)
    monkeypatch.setattr(w4a8_mxfp4, "_EXTRA_CTX", context)
    monkeypatch.setattr(w4a8_mxfp4, "build_fused_experts_input", lambda **kwargs: SimpleNamespace(**kwargs))
    weights = torch.tensor([[0.123456, 0.876544]], dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    scheme = SimpleNamespace(quant_type=QuantType.NONE, dynamic_eplb=False)
    result = w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod.apply(
        scheme, MagicMock(), torch.zeros(1, 4, dtype=torch.bfloat16), weights, ids, None, None
    )
    if comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2):
        assert result.topk_weights is weights
    else:
        assert result.topk_weights.dtype == torch.bfloat16
        assert torch.equal(result.topk_weights, weights.bfloat16())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mc2_boundaries_enforce_fp32_without_recopying_fp32_weights(dtype):
    dispatcher = SimpleNamespace(
        a5_need_extra_args=True,
        global_bs=0,
        moe_all_to_all_group_name="test",
        ep_world_size=8,
        ep_rank_id=0,
        mc2_comm_alg="",
        need_extra_args=True,
        need_expert_scale=True,
    )
    weights = torch.tensor([[0.123456, 0.876544]], dtype=dtype)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    hidden_states = torch.zeros(1, 4, dtype=torch.bfloat16)
    quant = SimpleNamespace(
        comm_quant_mode=None,
        dispatch_with_quant=False,
        is_mxfp=False,
        is_fp8=False,
        quant_type=QuantType.NONE,
        use_w4a8_per_channel_gmm_swiglu=False,
    )
    routing = SimpleNamespace(expert_map=torch.arange(2), global_redundant_expert_num=0, mc2_mask=None)
    request = SimpleNamespace(
        hidden_states=hidden_states,
        topk_weights=weights,
        topk_ids=ids,
        quant=quant,
        routing=routing,
        prefer_expert_token_counts=False,
    )
    dispatch = TokenDispatcherWithMC2.get_dispatch_mc2_kwargs(dispatcher, request)
    metadata = SimpleNamespace(
        expert_map=routing.expert_map,
        topk_ids=ids,
        topk_weights=weights,
        ep_recv_counts=torch.zeros(8, dtype=torch.int32),
        tp_recv_counts=None,
        assist_info_for_combine=None,
        expand_scales=None,
        quant=quant,
        mc2_mask=None,
    )
    combine = TokenDispatcherWithMC2.get_combine_mc_kwargs(dispatcher, hidden_states, metadata)
    for kwargs in (dispatch, combine):
        assert kwargs["expert_scales"].dtype == torch.float32
        assert torch.equal(kwargs["expert_scales"], weights.float())
        if dtype == torch.float32:
            assert kwargs["expert_scales"] is weights
