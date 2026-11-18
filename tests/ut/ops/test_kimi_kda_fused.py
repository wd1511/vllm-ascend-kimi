# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused KDA path (the default KDA execution).

Ports the Kimi K3 KDA test deltas from the 27rc_kda_env_ops branch onto the
shared ``AscendKimiK3DeltaAttention`` class: raw beta logits, the packed-QKV
recurrent snapshot ABI, the TND flash prefill, and the Mega-KDA priority.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import vllm_ascend.ops.kimi_kda as kimi_kda_module
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention


class _RecordingLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _input: torch.Tensor):
        return self.output, None


def _fake_upstream_init(attention, _config, _vllm_config, prefix=""):
    nn.Module.__init__(attention)
    attention.prefix = prefix
    attention.hidden_size = 7168
    attention.head_dim = 128
    attention.local_num_heads = 6
    attention.local_projection_size = 768
    attention.gate_lower_bound = -5.0
    attention.use_recoverssm = False
    attention.tp_size = 1
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


def _make_attention(
    enable_mega_kda=False,
    kernels_available=True,
    upstream_init=_fake_upstream_init,
):
    """Build an attention with mocked upstream init and kernel availability."""
    config = SimpleNamespace(rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None, enable_prompt_embeds=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)),
    )
    impl = MagicMock() if kernels_available else None
    with (
        patch.object(
            kimi_kda_module,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_mega_kda=enable_mega_kda),
        ),
        patch.object(kimi_kda_module, "_mega_kda_impl", lambda *args, **kwargs: None),
        patch.object(kimi_kda_module, "_flash_kda_impl", impl),
        patch.object(kimi_kda_module, "_flash_kda_metadata_impl", impl),
        patch.object(kimi_kda_module, "_recurrent_kda_impl", impl),
        patch.object(kimi_kda_module.KimiK3DeltaAttention, "__init__", new=upstream_init),
    ):
        return AscendKimiK3DeltaAttention(config, vllm_config)


def _make_bare_attention():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    return attention


def test_fused_kda_enabled_by_default():
    # No switches configured: the fused path is the default.
    attention = _make_attention()
    assert attention.use_mega_kda is False
    assert attention.use_fused_kda is True


def test_fused_kda_coexists_with_mega_kda():
    # Mega takes qualifying verify batches; fused handles everything else
    # (prefill and non-qualifying batches), AscendC only when fused is
    # unavailable.
    attention = _make_attention(enable_mega_kda=True)
    assert attention.use_mega_kda is True
    assert attention.use_fused_kda is True


def test_fused_kda_falls_back_without_kernels():
    attention = _make_attention(kernels_available=False)
    assert attention.use_fused_kda is False


def test_fused_kda_falls_back_without_gate_lower_bound():
    def upstream_init_without_gate(attention, config, vllm_config, prefix=""):
        _fake_upstream_init(attention, config, vllm_config, prefix)
        attention.gate_lower_bound = None

    attention = _make_attention(upstream_init=upstream_init_without_gate)
    assert attention.use_fused_kda is False


def test_fused_postprocess_bfg_keeps_raw_beta_logits():
    attention = _make_bare_attention()
    attention.use_fused_kda = True
    attention.head_dim = 3
    attention._fused_bfg_output_sizes = (2, 6, 6)
    fused_output = torch.arange(56, dtype=torch.float32).reshape(4, 14).to(torch.bfloat16)
    attention.fused_bfg_proj = _RecordingLinear(fused_output)

    beta, raw_gate, output_gate = attention._postprocess_bfg(attention._project_bfg(torch.randn(4, 5)))

    # The fused KDA kernels receive raw beta logits and apply the sigmoid.
    assert beta.dtype == fused_output.dtype
    torch.testing.assert_close(beta, fused_output[:, :2].unsqueeze(0))
    torch.testing.assert_close(raw_gate, fused_output[:, 2:8].reshape(4, 2, 3).unsqueeze(0))
    torch.testing.assert_close(output_gate, fused_output[:, 8:].reshape(4, 2, 3))


def test_fused_forward_routes_to_fused_dispatch():
    attention = _make_bare_attention()
    attention.use_fused_kda = True
    attention._forward_fused_kda = MagicMock()
    mixed_qkv, g1, g2 = torch.randn(4, 18), torch.randn(1, 4, 2, 3), torch.randn(4, 2, 3)
    beta = torch.randn(1, 4, 2)
    core_attn_out = torch.empty(1, 4, 2, 3)

    attention._kda_forward_impl(
        mixed_qkv=mixed_qkv,
        g1=g1,
        g2=g2,
        beta=beta,
        core_attn_out=core_attn_out,
        beta_is_preprocessed=True,
    )

    attention._forward_fused_kda.assert_called_once_with(mixed_qkv, g1, g2, beta, core_attn_out)


def test_ascendc_forward_skips_fused_dispatch_when_disabled():
    attention = _make_bare_attention()
    attention.use_fused_kda = False
    attention._forward_fused_kda = MagicMock()
    core_attn_out = torch.full((1, 4, 2, 3), torch.nan)

    with patch.object(
        kimi_kda_module,
        "get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._kda_forward_impl(
            mixed_qkv=torch.empty(4, 18),
            g1=torch.empty(1, 4, 2, 3),
            g2=torch.empty(4, 2, 3),
            beta=torch.empty(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    attention._forward_fused_kda.assert_not_called()
    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


@pytest.mark.parametrize("state_indices_rank", [1, 2])
def test_fused_recurrent_consumes_packed_qkv(state_indices_rank):
    attention = _make_bare_attention()
    attention.local_num_heads = 2
    attention.head_dim = 3
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.randn(2))
    attention.dt_bias = nn.Parameter(torch.randn(2, 3))

    batch, sequence_length = 2, 3
    tokens = batch * sequence_length
    mixed_qkv = torch.randn(tokens, 18)
    raw_gate = torch.randn(1, tokens, 2, 3)
    raw_beta = torch.randn(1, tokens, 2)
    recurrent_state = torch.randn(4, 2, 3, 3)
    state_indices = (
        torch.tensor([3, 1], dtype=torch.int64)
        if state_indices_rank == 1
        else torch.tensor([[3, 3, 3], [1, 1, 1]], dtype=torch.int64)
    )
    num_accepted = torch.tensor([2, 3], dtype=torch.int64)
    kernel_output = torch.randn(batch, sequence_length, 2, 3)

    with patch.object(kimi_kda_module, "_recurrent_kda_impl", return_value=kernel_output) as recurrent_impl:
        output = attention._run_recurrent_fused_kda(
            mixed_qkv,
            raw_gate,
            raw_beta,
            recurrent_state,
            torch.tensor([0, 3, 6], dtype=torch.int32),
            state_indices,
            num_accepted_tokens=num_accepted,
        )

    assert output.shape == (1, tokens, 2, 3)
    torch.testing.assert_close(output.reshape(batch, sequence_length, 2, 3), kernel_output)
    args, kwargs = recurrent_impl.call_args
    torch.testing.assert_close(args[0], mixed_qkv.reshape(batch, sequence_length, 18))
    assert kwargs["state"] is recurrent_state
    torch.testing.assert_close(kwargs["beta"], raw_beta.reshape(batch, sequence_length, 2, 1))
    torch.testing.assert_close(kwargs["g"], raw_gate.reshape(batch, sequence_length, 2, 3))
    assert kwargs["scale"] == 3**-0.5
    assert kwargs["lower_bound"] == -5.0
    assert kwargs["layout_qkv"] == "BSND"
    assert kwargs["ssm_state_indices"].dtype == torch.int32
    torch.testing.assert_close(kwargs["ssm_state_indices"], torch.tensor([3, 3, 3, 1, 1, 1], dtype=torch.int32))
    torch.testing.assert_close(kwargs["num_accepted_tokens"], num_accepted.to(torch.int32))


def test_fused_recurrent_rejects_token_count_mismatch():
    attention = _make_bare_attention()
    attention.local_num_heads = 2
    attention.head_dim = 3
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.randn(2))
    attention.dt_bias = nn.Parameter(torch.randn(2, 3))

    with pytest.raises(ValueError, match="different token counts"):
        attention._run_recurrent_fused_kda(
            torch.randn(7, 18),
            torch.randn(1, 7, 2, 3),
            torch.randn(1, 7, 2),
            torch.randn(4, 2, 3, 3),
            torch.tensor([0, 3, 6], dtype=torch.int32),
            torch.tensor([3, 1], dtype=torch.int64),
        )


@pytest.mark.parametrize("compact_metadata", [False, True])
def test_fused_prefill_uses_tnd_flash_kda_and_preserves_cached_state(compact_metadata):
    attention = _make_bare_attention()
    attention.local_num_heads = 2
    attention.head_dim = 3
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.zeros(2))
    attention.dt_bias = nn.Parameter(torch.zeros(2, 3))

    tokens = 68
    q, k, v, raw_gate = (torch.randn(tokens, 2, 3, dtype=torch.bfloat16) for _ in range(4))
    raw_beta = torch.randn(tokens, 2, dtype=torch.bfloat16)
    recurrent_state = torch.arange(6 * 2 * 3 * 3, dtype=torch.float32).reshape(6, 2, 3, 3).to(torch.bfloat16)
    previous_state = recurrent_state.clone()
    boundaries = torch.tensor((0, 3, 68), dtype=torch.int32)
    if compact_metadata:
        # The middle metadata row has no tokens and must not update its cache.
        state_indices = torch.tensor([4, 2, 1], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True, True])
        metadata = SimpleNamespace(
            cu_seqlens_host=(0, 3, 3, 68),
            cu_seqlens_kern=boundaries,
            keep_meta=torch.tensor([True, False, True]),
        )
    else:
        state_indices = torch.tensor([4, 1], dtype=torch.int32)
        has_initial_state = torch.tensor([False, True])
        metadata = SimpleNamespace(cu_seqlens_host=(0, 3, 68), cu_seqlens_kern=None, keep_meta=None)

    expected_initial = torch.stack((torch.zeros(2, 3, 3), previous_state[1].float()))
    chunk_metadata = object()
    final_state = previous_state[:2].float() + 10.0
    output = torch.randn(tokens, 2, 3, dtype=torch.bfloat16)

    def clear_initial_states(states, has_state):
        states[~has_state] = 0

    with (
        patch("vllm_ascend.ops.kimi_kda.clear_ssm_states", side_effect=clear_initial_states),
        patch(
            "vllm_ascend.ops.kimi_kda._flash_kda_metadata_impl",
            return_value=chunk_metadata,
        ) as metadata_impl,
        patch(
            "vllm_ascend.ops.kimi_kda._flash_kda_impl",
            return_value=(output, final_state),
        ) as flash_impl,
    ):
        actual = attention._run_prefill_fused_kda(
            q, k, v, raw_gate, raw_beta, recurrent_state, state_indices, has_initial_state, metadata
        )

    # One TND launch consumes the packed tokens; no per-request padding or loop.
    metadata_impl.assert_called_once()
    metadata_args, metadata_kwargs = metadata_impl.call_args
    assert metadata_args[0] is q
    assert metadata_args[1] is v
    torch.testing.assert_close(metadata_args[2], expected_initial)
    assert metadata_kwargs["layout_qkv"] == "TND"
    torch.testing.assert_close(metadata_kwargs["cu_seqlens"], boundaries)

    flash_impl.assert_called_once()
    flash_args, flash_kwargs = flash_impl.call_args
    assert flash_args == (q, k, v)
    assert flash_kwargs["g"] is raw_gate
    assert flash_kwargs["beta"] is raw_beta
    assert flash_kwargs["layout_qkv"] == "TND"
    assert flash_kwargs["metadata"] is chunk_metadata
    assert flash_kwargs["lower_bound"] == -5.0
    torch.testing.assert_close(flash_kwargs["initial_state"], expected_initial)

    assert torch.equal(actual, output.unsqueeze(0))
    expected_state = previous_state.clone()
    expected_state[4] = final_state[0].to(recurrent_state.dtype)
    expected_state[1] = final_state[1].to(recurrent_state.dtype)
    torch.testing.assert_close(recurrent_state, expected_state)


def test_fused_prefill_rejects_metadata_sequence_count_mismatch():
    attention = _make_bare_attention()
    attention.local_num_heads = 2
    attention.head_dim = 3
    attention.gate_lower_bound = -5.0
    attention.A_log = nn.Parameter(torch.zeros(2))
    attention.dt_bias = nn.Parameter(torch.zeros(2, 3))

    q, k, v, raw_gate = (torch.randn(5, 2, 3, dtype=torch.bfloat16) for _ in range(4))
    raw_beta = torch.randn(5, 2, dtype=torch.bfloat16)
    recurrent_state = torch.zeros(4, 2, 3, 3)
    metadata = SimpleNamespace(cu_seqlens_host=(0, 2, 2, 5), cu_seqlens_kern=None, keep_meta=None)

    with (
        patch("vllm_ascend.ops.kimi_kda.clear_ssm_states"),
        pytest.raises(ValueError, match="same number of sequences"),
    ):
        attention._run_prefill_fused_kda(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            recurrent_state,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([False, True]),
            metadata,
        )


def test_fused_empty_forward_context_clears_preallocated_output():
    attention = _make_bare_attention()
    attention.use_fused_kda = True
    core_attn_out = torch.full((1, 4, 2, 3), torch.nan)

    with patch(
        "vllm_ascend.ops.kimi_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._kda_forward_impl(
            mixed_qkv=torch.empty(4, 18),
            g1=torch.empty(1, 4, 2, 3),
            g2=torch.empty(4, 2, 3),
            beta=torch.empty(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))
