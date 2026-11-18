# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

import vllm_ascend.ops.kimi_kda as kimi_kda_module
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiK3DeltaAttention,
)
from vllm_ascend.ops.kimi_mega_kda import MegaKDAWeights

TEST_BATCH = 16


def _fake_upstream_init(attention, config, vllm_config, prefix=""):
    nn.Module.__init__(attention)
    attention.prefix = prefix
    attention.hidden_size = 7168
    attention.head_dim = 128
    attention.local_num_heads = 6
    attention.local_projection_size = 768
    attention.num_spec = 7
    attention.use_recoverssm = False
    attention.tp_size = 16
    attention.o_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(128, dtype=torch.bfloat16)), eps=1e-6)
    attention.conv_size = 4
    attention.model_config = SimpleNamespace(dtype=torch.bfloat16)
    attention.conv1d = nn.Module()
    attention.conv1d.weight = nn.Parameter(torch.empty(6, 1, 4))
    attention.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda *args, **kwargs: None)
    attention.in_proj_qkvgfab = nn.Module()
    attention.in_proj_qkvgfab.quant_method = None
    attention.f_b_proj = SimpleNamespace(weight=nn.Parameter(torch.empty(768, 128)))
    attention.o_proj = SimpleNamespace(weight=nn.Parameter(torch.empty(7168, 768)), reduce_results=True)


def _make_attention(config_eps=1e-6, upstream_init=_fake_upstream_init):
    config = SimpleNamespace(rms_norm_eps=config_eps)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None, enable_prompt_embeds=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
    )
    with patch(
        "vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__",
        new=upstream_init,
    ):
        return AscendKimiK3DeltaAttention(config, vllm_config)


def _ascend_config(enable_mega_kda):
    return SimpleNamespace(enable_mega_kda=enable_mega_kda)


def test_mega_kda_disabled_by_default():
    with patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(False)):
        attention = _make_attention()
    assert attention.use_mega_kda is False


def test_mega_kda_requires_operator_package():
    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", None),
        pytest.raises(ImportError, match="ops.mega_recurrent_kda"),
    ):
        _make_attention()


def test_mega_kda_allows_tp8():
    # MegaKDA dispatch follows the installed operator's supported shapes;
    # vLLM must not impose the old TP16-only restriction.
    def upstream_init_tp8(attention, config, vllm_config, prefix=""):
        _fake_upstream_init(attention, config, vllm_config, prefix)
        attention.tp_size = 8
        attention.local_num_heads = 12

    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
    ):
        attention = _make_attention(upstream_init=upstream_init_tp8)
    assert attention.use_mega_kda is True


def test_mega_kda_allows_any_checkpoint_epsilon():
    # The newer operator ABI takes the epsilon as its 20th argument, so a
    # non-1e-6 checkpoint no longer blocks the fused path.
    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
    ):
        attention = _make_attention(config_eps=1e-5)
    assert attention.use_mega_kda is True


def test_mega_kda_enabled():
    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
    ):
        attention = _make_attention()
    assert attention.use_mega_kda is True


def test_mega_kda_enabled_under_full_cudagraph():
    # Init-time deployment checks no longer look at the cudagraph mode; per-batch
    # qualification happens in _plan_mega_kda, and _forward_mega_kda restores the
    # layer's token count when the operator emits fewer rows than it received.
    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
    ):
        attention = _make_attention()
    assert attention.use_mega_kda is True


def test_process_weights_prepares_mega_operands_before_forward():
    with (
        patch.object(kimi_kda_module, "get_ascend_config", return_value=_ascend_config(True)),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
    ):
        attention = _make_attention()

    prepared_weights = object()
    attention._pack_conv_weights = MagicMock()
    attention._build_mega_weights = MagicMock(return_value=prepared_weights)
    attention.conv1d.quant_method.process_weights_after_loading()

    attention._pack_conv_weights.assert_called_once_with()
    attention._build_mega_weights.assert_called_once_with()
    assert attention._mega_weights is prepared_weights


def _make_bare_attention():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.prefix = "model.layers.0.self_attn"
    attention.num_spec = 7
    attention.local_num_heads = 2
    attention.local_projection_size = 8
    attention.head_dim = 4
    attention.hidden_size = 32
    attention.tp_size = 16
    attention.uses_mixed_projection = False
    attention.o_proj = SimpleNamespace(
        weight=nn.Parameter(torch.randn(32, 8, dtype=torch.bfloat16)), reduce_results=True
    )
    attention.in_proj_qkvgfab = SimpleNamespace(
        weight=nn.Parameter(torch.randn(4 * 8 + 4 + 2, 32, dtype=torch.bfloat16))
    )
    attention.f_b_proj = SimpleNamespace(weight=nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16)))
    attention.o_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(4, dtype=torch.bfloat16)), eps=1e-6)
    attention.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(torch.randn(4, 3 * 8).to(torch.bfloat16)),
    )
    return attention


def test_build_mega_weights_slices_packed_projection():
    attention = _make_bare_attention()
    # The NZ cast needs NPU; emulate it with the clone that precedes it so
    # the standalone-tensor (offset 0) contract is still exercised.
    with patch.object(kimi_kda_module, "mega_kda_nz_operand", side_effect=lambda t: t.clone()):
        weights = attention._build_mega_weights()
    assert isinstance(weights, MegaKDAWeights)
    # Newer operator ABI: [out, in] slices cast to NZ.
    assert weights.qkv.shape == (3 * 8, 32)
    assert weights.output_gate.shape == (8, 32)
    assert weights.f_a.shape == (4, 32)
    assert weights.beta.shape == (2, 32)
    assert weights.f_b.shape == (8, 4)
    assert weights.o_proj.shape == (32, 8)
    assert weights.o_norm.dtype == torch.float32
    # The conv weight is re-layouted to [3, heads, conv_size, dim].
    assert weights.conv.shape == (3, 2, 4, 4)
    # Slices must tile the packed [QKV | G | f_a | B] layout in order.
    packed = attention.in_proj_qkvgfab.weight
    assert torch.equal(weights.qkv, packed[: 3 * 8])
    assert torch.equal(weights.output_gate, packed[3 * 8 : 4 * 8])
    assert torch.equal(weights.f_a, packed[4 * 8 : 4 * 8 + 4])
    assert torch.equal(weights.beta, packed[4 * 8 + 4 :])
    # Every operand is materialized as a standalone tensor (offset 0)
    # before the NZ cast - npu_format_cast mishandles offset views.
    assert weights.qkv.storage_offset() == 0
    assert weights.output_gate.storage_offset() == 0
    assert weights.f_a.storage_offset() == 0
    assert weights.beta.storage_offset() == 0
    assert weights.f_b.storage_offset() == 0
    assert weights.o_proj.storage_offset() == 0


def test_build_mega_weights_mixed_projection_layout():
    # QuaRot layout: QKV-only in_proj + fused_bfg_proj [B | F | G] rows with
    # raw f_a/f_b parameters (2b28745cd weight sourcing).
    attention = _make_bare_attention()
    attention.uses_mixed_projection = True
    attention.in_proj_qkvgfab = SimpleNamespace(weight=nn.Parameter(torch.randn(3 * 8, 32, dtype=torch.bfloat16)))
    fused_weight = nn.Parameter(torch.randn(2 + 8 + 8, 32, dtype=torch.bfloat16))
    attention.fused_bfg_proj = SimpleNamespace(
        weight=fused_weight,
        f_a_weight=nn.Parameter(torch.randn(4, 32, dtype=torch.bfloat16)),
        f_b_weight=nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16)),
    )
    with patch.object(kimi_kda_module, "mega_kda_nz_operand", side_effect=lambda t: t.clone()):
        weights = attention._build_mega_weights()
    assert weights.qkv.shape == (3 * 8, 32)
    assert weights.beta.shape == (2, 32)
    assert weights.output_gate.shape == (8, 32)
    assert weights.f_a.shape == (4, 32)
    assert weights.f_b.shape == (8, 4)
    # qkv is the whole QKV-only linear; beta/gate narrow the [B | F | G] rows.
    torch.testing.assert_close(weights.qkv, attention.in_proj_qkvgfab.weight)
    torch.testing.assert_close(weights.beta, fused_weight[:2])
    torch.testing.assert_close(weights.output_gate, fused_weight[2 + 8 : 2 + 8 + 8])
    torch.testing.assert_close(weights.f_a, attention.fused_bfg_proj.f_a_weight)
    torch.testing.assert_close(weights.f_b, attention.fused_bfg_proj.f_b_weight)


def test_forward_mega_kda_falls_back_when_weights_not_bf16():
    # Lazy BF16 gate: weights that were never dequantized disable the path
    # at the first weight build instead of crashing.
    attention = _make_bare_attention()
    attention.use_mega_kda = True
    attention.in_proj_qkvgfab.weight.data = attention.in_proj_qkvgfab.weight.data.to(torch.float32)
    attention._forward_default = MagicMock(return_value=torch.zeros(1))
    plan = attention._plan_mega_kda(torch.zeros(16 * 8, 32, dtype=torch.bfloat16))
    assert plan is not None
    output = attention._forward_mega_kda(torch.zeros(16 * 8, 32, dtype=torch.bfloat16), torch.zeros(4), plan)
    attention._forward_default.assert_called_once()
    assert attention.use_mega_kda is False
    assert output.shape == (1,)


def _make_spec_metadata(num_prefills=0, num_decodes=0, batch=TEST_BATCH):
    spec_len = 8
    slots = torch.arange(batch, dtype=torch.int32).view(batch, 1)
    spec_indices = slots * 10 + torch.arange(spec_len, dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=0,
        num_spec_decodes=batch,
        num_spec_decode_tokens=batch * spec_len,
        num_actual_tokens=batch * spec_len,
        spec_state_indices_tensor=spec_indices,
        num_accepted_tokens=torch.full((batch,), 2, dtype=torch.int32),
    )
    metadata.mega_first_verify = torch.zeros(batch, dtype=torch.int32)
    # The gate derives batch/S from the spec-conv query boundaries (2b28745cd).
    metadata.spec_decode_metadata = SimpleNamespace(
        spec_causal_conv1d=SimpleNamespace(
            query_start_loc=_query_start_loc([spec_len] * batch),
        )
    )
    return metadata


def _plan_for(metadata, tp_size=16):
    attention = _make_bare_attention()
    attention.tp_size = tp_size
    hidden = torch.zeros(metadata.num_actual_tokens, 32, dtype=torch.bfloat16)
    forward_context = SimpleNamespace(attn_metadata={attention.prefix: metadata})
    with patch.object(kimi_kda_module, "get_forward_context", return_value=forward_context):
        return attention._plan_mega_kda(hidden)


def _make_non_spec_decode_metadata(batch=TEST_BATCH):
    metadata = _make_spec_metadata(batch=batch)
    metadata.num_spec_decodes = 0
    metadata.num_spec_decode_tokens = 0
    metadata.spec_decode_metadata = None
    metadata.num_decodes = batch
    metadata.num_decode_tokens = batch
    metadata.num_actual_tokens = batch
    metadata.non_spec_query_start_loc = _query_start_loc([1] * batch)
    metadata.non_spec_state_indices_tensor = torch.arange(batch, dtype=torch.int32)
    return metadata


@pytest.mark.parametrize("metadata_kind", ["none", "missing_layer", "non_gdn"])
def test_plan_mega_kda_compiles_without_gdn_metadata(metadata_kind):
    attention = _make_bare_attention()
    metadata = None if metadata_kind == "none" else {}
    if metadata_kind == "non_gdn":
        metadata[attention.prefix] = SimpleNamespace()
    context = SimpleNamespace(attn_metadata=metadata)

    def dispatch(hidden_states):
        plan = attention._plan_mega_kda(hidden_states)
        assert plan is None
        return hidden_states + 1

    hidden_states = torch.zeros(2, attention.hidden_size, dtype=torch.bfloat16)
    with patch.object(kimi_kda_module, "get_forward_context", new=lambda: context):
        compiled = torch.compile(dispatch, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(hidden_states), hidden_states + 1)


def test_plan_mega_kda_matches_recipe_prefill_condition():
    assert _plan_for(_make_spec_metadata()) is not None
    assert _plan_for(_make_spec_metadata(num_prefills=1)) is None
    assert _plan_for(_make_spec_metadata(num_decodes=1)) is not None
    ordinary = _plan_for(_make_non_spec_decode_metadata())
    assert ordinary is not None
    assert ordinary.spec_len == 1


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("multiplier", range(1, 33))
def test_plan_mega_kda_accepts_tp_aligned_batches(tp_size, multiplier):
    batch = tp_size * multiplier
    plan = _plan_for(_make_spec_metadata(batch=batch), tp_size=tp_size)
    assert plan is not None
    assert plan.batch == batch


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_plan_mega_kda_rejects_unaligned_or_oversized_batches(tp_size):
    for batch in (1, tp_size - 1, tp_size + 1, 62, tp_size * 32 + 1, tp_size * 33):
        assert _plan_for(_make_spec_metadata(batch=batch), tp_size=tp_size) is None


def test_plan_mega_kda_accepts_runtime_spec_len():
    # The recipes condition does not gate on speculative width.
    metadata = _make_spec_metadata()
    metadata.spec_decode_metadata.spec_causal_conv1d.query_start_loc = _query_start_loc([4] * TEST_BATCH)
    metadata.num_actual_tokens = TEST_BATCH * 4
    plan = _plan_for(metadata)
    assert plan is not None
    assert plan.spec_len == 4


def test_plan_mega_kda_matches_recipe_for_folded_prefill_metadata():
    metadata = _make_spec_metadata()
    metadata.mega_has_folded_prefills = True
    assert _plan_for(metadata) is not None


def test_plan_mega_kda_count_semantics_follow_reference_pr():
    metadata = _make_spec_metadata()
    metadata.mega_first_verify[:4] = 1  # first four requests are on round one
    plan = _plan_for(metadata)
    assert plan is not None
    assert plan.batch == TEST_BATCH
    assert plan.spec_len == 8
    # First Verify: conv=S; later rounds: previous accepted length.
    assert torch.equal(
        plan.conv_num_accepted_tokens,
        torch.tensor([8] * 4 + [2] * 12, dtype=torch.int32),
    )
    assert torch.equal(plan.num_accepted_tokens, metadata.num_accepted_tokens)
    # Conv slots come from the block-table first column; recurrent indices
    # are flattened to [B*S] per the operator ABI.
    assert torch.equal(plan.conv_state_block_ids, metadata.spec_state_indices_tensor[:, 0])
    assert plan.ssm_state_indices.numel() == TEST_BATCH * 8
    assert torch.equal(plan.ssm_state_indices, metadata.spec_state_indices_tensor.reshape(-1))


@pytest.mark.parametrize("num_prefills,batch", [(1, 16), (0, 62), (0, 528)])
def test_forward_mega_kda_falls_back_when_plan_rejects(num_prefills, batch):
    # forward() routes plan-rejected batches directly to _forward_default,
    # bypassing _forward_mega_kda entirely (no eager_break on the fallback).
    attention = _make_bare_attention()
    attention.use_mega_kda = True
    attention._forward_default = MagicMock(return_value=torch.zeros(1))
    attention._forward_mega_kda = MagicMock(side_effect=AssertionError("MegaKDA must not run"))
    forward_context = SimpleNamespace(
        attn_metadata={attention.prefix: _make_spec_metadata(num_prefills=num_prefills, batch=batch)}
    )
    with patch.object(kimi_kda_module, "get_forward_context", return_value=forward_context):
        output = attention.forward(torch.zeros(batch * 8, 32, dtype=torch.bfloat16), torch.zeros(4))
    attention._forward_mega_kda.assert_not_called()
    attention._forward_default.assert_called_once()
    assert output.shape == (1,)


def test_forward_mega_kda_calls_operator_and_reduces():
    attention = _make_bare_attention()
    attention.use_mega_kda = True
    attention.tp_size = 2
    attention.kv_cache = (
        torch.zeros(4, 10, 3 * 2 * 4, dtype=torch.bfloat16),
        torch.zeros(4, 2, 4, 4, dtype=torch.float32),
    )
    attention.A_log = torch.zeros(2)
    attention.dt_bias = torch.zeros(8)
    attention.gate_lower_bound = -5.0
    recorded = {}

    def fake_run(**kwargs):
        recorded.update(kwargs)
        return torch.zeros(kwargs["hidden_states"].shape)

    with (
        patch.object(
            kimi_kda_module,
            "get_forward_context",
            return_value=SimpleNamespace(attn_metadata={attention.prefix: _make_spec_metadata()}),
        ),
        patch.object(kimi_kda_module, "mega_kda_nz_operand", side_effect=lambda t: t.clone()),
        patch.object(kimi_kda_module, "run_mega_recurrent_kda", side_effect=fake_run),
        patch.object(kimi_kda_module, "tensor_model_parallel_all_reduce", wraps=lambda x: x) as mock_reduce,
    ):
        hidden = torch.zeros(16 * 8, 32, dtype=torch.bfloat16)
        plan = attention._plan_mega_kda(hidden)
        assert plan is not None
        output = attention._forward_mega_kda(hidden, torch.zeros(4), plan)
    assert output.shape == (16 * 8, 32)
    assert recorded["hidden_states"].shape == (16, 8, 32)
    assert recorded["dt_bias"].shape == (2, 4)
    assert recorded["scale"] == 4**-0.5
    assert recorded["gate_lower_bound"] == -5.0
    assert recorded["o_norm_variance_epsilon"] == 1e-6
    mock_reduce.assert_called_once()


def _make_bare_builder(num_blocks=8, capacity=20):
    builder = AscendGDNAttentionMetadataBuilder.__new__(AscendGDNAttentionMetadataBuilder)
    builder._mega_kda_enabled = True
    builder._mega_verified_slots = None
    builder._mega_first_verify = torch.empty((capacity,), dtype=torch.int32)
    builder.kv_cache_spec = SimpleNamespace(num_blocks=num_blocks)
    return builder


def _query_start_loc(lens):
    boundaries = [0]
    for length in lens:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32)


def test_builder_tracks_first_verify_rounds():
    builder = _make_bare_builder()
    block_table = torch.zeros(4, 8, dtype=torch.int32)
    block_table[:, 0] = torch.tensor([2, 3, 5, 7], dtype=torch.int32)
    spec_rows = block_table[[0, 2]]
    masks = torch.tensor([True, False, True, False])

    # Round one for both spec requests.
    first = builder._update_mega_first_verify(block_table, _query_start_loc([8, 1, 8, 1]), masks, spec_rows)
    assert torch.equal(first, torch.tensor([1, 1], dtype=torch.int32))
    # Round two: neither is a first Verify anymore.
    second = builder._update_mega_first_verify(block_table, _query_start_loc([8, 1, 8, 1]), masks, spec_rows)
    assert torch.equal(second, torch.tensor([0, 0], dtype=torch.int32))
    # A prefill on slot 2 resets only that request's round state.
    builder._update_mega_first_verify(
        block_table,
        _query_start_loc([16, 1, 8, 1]),
        torch.tensor([False, False, True, False]),
        block_table[[2]],
    )
    third = builder._update_mega_first_verify(block_table, _query_start_loc([8, 1, 8, 1]), masks, spec_rows)
    assert torch.equal(third, torch.tensor([1, 0], dtype=torch.int32))


def test_builder_skips_tracking_when_disabled():
    builder = _make_bare_builder()
    builder._mega_kda_enabled = False
    assert (
        builder._update_mega_first_verify(
            torch.zeros(1, 2, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            None,
            None,
        )
        is None
    )
