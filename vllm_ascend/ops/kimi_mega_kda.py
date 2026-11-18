# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External Mega-KDA decode operator wrapper for Kimi K3.

``mega_recurrent_kda`` (external ``ops`` package) fuses the KDA input
projections, causal conv1d update, recurrent KDA, gated output norm and
output projection into a single launch.  The 20-argument ABI reads BF16
``[out, in]`` weights pre-cast to NZ format and a
``[3, heads, conv_size, head_dim]`` conv weight; state indices and
accepted-token counts are int32, ``ssm_state_indices`` is flattened to
``[batch * spec_len]``, and the first Verify round after prefill uses
``conv=S, recurrent=1`` while later rounds reuse the previous round's
accepted length.
"""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

_mega_kda_import_error: Exception | None = None
try:
    from ops.mega_recurrent_kda import mega_recurrent_kda as _mega_kda_impl
except ImportError as error:  # pragma: no cover - depends on deployment env
    _mega_kda_impl = None
    _mega_kda_import_error = error

MEGA_KDA_NZ_FORMAT = 29


def _as_int32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return an operator-compatible int32 tensor without redundant copies."""
    if tensor.dtype == torch.int32 and tensor.is_contiguous():
        return tensor
    return tensor.to(torch.int32).contiguous()


@dataclass(frozen=True)
class MegaKDAWeights:
    """NZ-cast BF16 ``[out, in]`` matmul weights, FP32 o_norm, [3,H,W,D] conv."""

    qkv: torch.Tensor
    f_a: torch.Tensor
    f_b: torch.Tensor
    beta: torch.Tensor
    output_gate: torch.Tensor
    o_norm: torch.Tensor
    o_proj: torch.Tensor
    conv: torch.Tensor


def is_mega_kda_available() -> bool:
    """Whether the external ``ops.mega_recurrent_kda`` package is importable."""
    return _mega_kda_impl is not None


def ensure_mega_kda_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Assert one operand is BF16; the operator binary is BF16-only."""
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            f"MegaKDA requires BF16 weights, got {tensor.dtype}; use a BF16 "
            "KDA projection checkpoint for the fused operator"
        )
    return tensor


def mega_kda_nz_operand(tensor: torch.Tensor) -> torch.Tensor:
    ensure_mega_kda_bf16(tensor)
    return torch_npu.npu_format_cast(tensor.clone(), MEGA_KDA_NZ_FORMAT)


def ensure_mega_kda_nd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "npu":
        return tensor
    if int(torch_npu.get_npu_format(tensor)) != int(ACL_FORMAT_FRACTAL_ND):
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def run_mega_recurrent_kda(
    hidden_states: torch.Tensor,
    weights: MegaKDAWeights,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    conv_state_block_ids: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    gate_lower_bound: float,
    o_norm_variance_epsilon: float,
) -> torch.Tensor:
    """Run the fused Mega-KDA decode operator for one KDA layer.

    ``hidden_states`` is ``[batch, spec_len, hidden]``.  The return value is
    the partial (pre-TP-reduction) ``[batch * spec_len, hidden]`` output with
    the output projection already applied.
    """
    assert _mega_kda_impl is not None
    output = _mega_kda_impl(
        hidden_states,
        weights.qkv,
        weights.f_a,
        weights.f_b,
        weights.beta,
        weights.output_gate,
        weights.o_norm,
        weights.o_proj,
        weights.conv,
        conv_state,
        recurrent_state,
        _as_int32_contiguous(conv_state_block_ids),
        _as_int32_contiguous(ssm_state_indices),
        _as_int32_contiguous(conv_num_accepted_tokens),
        _as_int32_contiguous(num_accepted_tokens),
        a_log,
        dt_bias,
        scale,
        gate_lower_bound,
        o_norm_variance_epsilon,
    )
    return output.reshape(-1, hidden_states.shape[-1])
