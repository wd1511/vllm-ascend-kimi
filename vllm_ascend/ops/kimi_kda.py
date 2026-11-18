# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backend for the vLLM 0.27 Kimi K3 delta-attention layer.

The projections, weight loading, and cache specification stay owned by
upstream vLLM.  KDA execution runs on the fused CANNBot DSL kernels for
Atlas 950 by default; ``enable_mega_kda=true`` routes non-prefill decode
batches to the fused Mega operator, with regular kernels handling prefill
and AscendC as the last-resort fallback.
"""

from dataclasses import dataclass
from functools import wraps

import torch
import torch_npu
from einops import rearrange
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.model_executor.utils import replace_parameter
from vllm.models.kimi_k3.nvidia.kda import (
    KimiK3DeltaAttention,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.kimi_mega_kda import (
    MegaKDAWeights,
    _mega_kda_impl,
    _mega_kda_import_error,
    ensure_mega_kda_bf16,
    ensure_mega_kda_nd,
    mega_kda_nz_operand,
    run_mega_recurrent_kda,
)
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import npu_stream_switch

# Fused KDA kernels for Atlas 950, shipped with the CANN operator package
# under the ``ops`` namespace (installed via cannbot-dsl): TND flash
# prefill with prebuilt chunk metadata, and the fused recurrent snapshot
# that consumes packed ShortConv QKV directly. They are the default KDA
# execution path (27rc_kda_env_ops branch); import paths follow
# cann-recipes-infer models/kimi_k3 (modeling_kimi_k3.py). Optional:
# deployments without the package keep the bundled AscendC operators.
_fused_kda_import_error: Exception | None = None
try:
    from ops.flash_kda import flash_kda as _flash_kda_impl
    from ops.flash_kda_metadata import flash_kda_metadata as _flash_kda_metadata_impl
    from ops.fused_recurrent_kda_snapshot import (
        fused_recurrent_kda_op as _recurrent_kda_impl,
    )
except ImportError as error:  # pragma: no cover - depends on deployment env
    _flash_kda_impl = None
    _flash_kda_metadata_impl = None
    _recurrent_kda_impl = None
    _fused_kda_import_error = error

_KDA_CHUNK_SIZE = 64
_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight"
_F_PROJ_SHARD_ID = 1
_MEGA_KDA_MAX_BATCH_MULTIPLIER = 32
_KDA_BFG_STREAM: torch.npu.Stream | None = None

logger = init_logger(__name__)


def _ascend_kda_forward(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
    beta_is_preprocessed: bool,
) -> None:
    # The upstream KDA constructor registers this layer in static_forward_context;
    # each execution context exposes that model's layers through no_compile_layers.
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._kda_forward_impl(mixed_qkv, g1, g2, beta, core_attn_out, beta_is_preprocessed=beta_is_preprocessed)


def _ascend_kda_forward_fake(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
    beta_is_preprocessed: bool,
) -> None:
    # The caller owns the output buffer; tracing must not read runtime metadata.
    return None


# Profile runs can trace with no attention metadata. Keep the fused/AscendC
# dispatch opaque so vLLM's no-guards graph cannot specialize it to zero output.
direct_register_custom_op(
    op_name="ascend_kda_forward",
    op_func=_ascend_kda_forward,
    mutates_args=["core_attn_out"],
    fake_impl=_ascend_kda_forward_fake,
)


def _kda_bfg_stream() -> torch.npu.Stream:
    global _KDA_BFG_STREAM
    if _KDA_BFG_STREAM is None:
        _KDA_BFG_STREAM = torch_npu.npu.Stream()
    return _KDA_BFG_STREAM


class _KDAFusedBFGLinear(MergedColumnParallelLinear):
    """Pack beta, an offline-composed F projection, and the output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        tp_size: int,
        quant_config,
        prefix: str,
    ) -> None:
        projection_size = num_heads * head_dim
        super().__init__(
            input_size=hidden_size,
            output_sizes=[num_heads, projection_size, projection_size],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        if self.tp_size != tp_size:
            raise ValueError(f"KDA fused BFG TP mismatch: layer={self.tp_size}, attention={tp_size}")
        local_projection_size = projection_size // tp_size
        self.f_a_weight = nn.Parameter(
            self.weight.new_empty((head_dim, hidden_size)),
            requires_grad=False,
        )
        self.f_b_weight = nn.Parameter(
            self.weight.new_empty((local_projection_size, head_dim)),
            requires_grad=False,
        )
        self.f_a_weight.weight_loader = self._load_f_a_weight
        self.f_b_weight.weight_loader = self._load_f_b_weight
        self._f_a_loaded = False
        self._f_b_loaded = False

    def _load_f_a_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if param.shape != loaded_weight.shape:
            raise ValueError(
                "KDA f_a_proj checkpoint shape mismatch: "
                f"expected {tuple(param.shape)}, got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)
        self._f_a_loaded = True
        self._maybe_fuse_f_proj()

    def _load_f_b_weight(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        del loaded_shard_id
        if loaded_weight.shape == param.shape:
            local_weight = loaded_weight
        else:
            expected_shape = (param.shape[0] * self.tp_size, param.shape[1])
            if loaded_weight.shape != expected_shape:
                raise ValueError(
                    "KDA f_b_proj checkpoint shape mismatch: "
                    f"expected {expected_shape} or {tuple(param.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            local_weight = loaded_weight.narrow(
                0,
                self.tp_rank * param.shape[0],
                param.shape[0],
            )
        param.data.copy_(local_weight)
        self._f_b_loaded = True
        self._maybe_fuse_f_proj()

    @torch.no_grad()
    def _maybe_fuse_f_proj(self) -> None:
        if not self._f_a_loaded or not self._f_b_loaded:
            return
        output_dim = getattr(self.weight, "output_dim", None)
        if output_dim is None:
            raise ValueError("KDA fused f_proj requires an output-sharded parameter")
        shard_offset = sum(self.output_sizes[:_F_PROJ_SHARD_ID]) // self.tp_size
        shard_size = self.output_sizes[_F_PROJ_SHARD_ID] // self.tp_size
        param_shard = self.weight.narrow(output_dim, shard_offset, shard_size)
        fused_weight = torch.matmul(
            self.f_b_weight.float(),
            self.f_a_weight.float(),
        ).to(dtype=param_shard.dtype)
        if fused_weight.shape != param_shard.shape:
            raise ValueError(
                "KDA composed f_proj shape mismatch: "
                f"expected {tuple(param_shard.shape)}, got {tuple(fused_weight.shape)}"
            )
        param_shard.copy_(fused_weight)


def _zero_padded_output(
    output: torch.Tensor,
    num_live_tokens: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows using a device-side live-token count."""
    token_indices = torch.arange(
        output.shape[1],
        dtype=num_live_tokens.dtype,
        device=output.device,
    )
    valid_tokens = token_indices < num_live_tokens
    return torch.where(valid_tokens.view(1, -1, 1, 1), output, 0.0)


def _zero_padded_recurrent_output(
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Clear graph-padding rows skipped by recurrent KDA."""
    return _zero_padded_output(output, query_start_loc[-1])


def _restore_token_rows(
    output: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Return ``output`` with ``num_tokens`` rows, zero-filling any tail.

    The Mega operator emits one row per live verify token (``batch * spec_len``).
    AttnRes requires an attention layer to return as many rows as it received,
    so the graph-padding tail is materialized as zeros to match the fused and
    AscendC paths.
    """
    if output.shape[0] == num_tokens:
        return output
    restored = output.new_zeros((num_tokens, output.shape[-1]))
    restored[: output.shape[0]].copy_(output)
    return restored


def _prepare_beta(
    beta: torch.Tensor,
    num_actual_tokens: int,
    *,
    is_preprocessed: bool = False,
) -> torch.Tensor:
    """Slice beta and apply sigmoid unless the auxiliary stream already did."""
    beta = beta[:, :num_actual_tokens]
    return beta if is_preprocessed else beta.float().sigmoid()


@dataclass(frozen=True)
class _MegaKDAPlan:
    """Resolved inputs for one fused Mega-KDA decode/verify step."""

    batch: int
    spec_len: int
    conv_state_block_ids: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor


class AscendKimiK3DeltaAttention(KimiK3DeltaAttention):
    """Kimi K3 KDA: fused CANNBot DSL kernels by default, AscendC operators
    as fallback, and the optional fused Mega-KDA decode path."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        quant_config = getattr(vllm_config, "quant_config", None)
        uses_mixed_projection = bool(
            quant_config is not None
            and getattr(
                quant_config,
                "uses_kimi_k3_mixed_kda_projection",
                lambda _prefix: False,
            )(f"{prefix}.in_proj_qkvgfab")
        )
        super().__init__(config, vllm_config, prefix)
        self.uses_mixed_projection = uses_mixed_projection
        if uses_mixed_projection:
            # vLLM 0.27 packs all KDA input projections into one linear.  A
            # QuaRot checkpoint instead stores q/k/v as W8A8 and keeps B/F/G
            # in floating point. Split those precision groups so DynamicQuant
            # can overlap the composed BFG projection.
            self.in_proj_qkvgfab = MergedColumnParallelLinear(
                self.hidden_size,
                [self.projection_size] * 3,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkv",
            )
            del self.f_b_proj
            self.fused_bfg_proj = _KDAFusedBFGLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                tp_size=self.tp_size,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_gfab",
            )
            self._fused_bfg_output_sizes = (
                self.local_num_heads,
                self.local_projection_size,
                self.local_projection_size,
            )
        # Preserve the checkpoint RMS epsilon (upstream defaults to 1e-5).
        self.o_norm.eps = config.rms_norm_eps
        self.use_mega_kda = self._resolve_use_mega_kda()
        self.use_fused_kda = self._resolve_use_fused_kda()
        # Mega-KDA NZ operand cache; invalidated on weight reload.
        self._mega_weights: MegaKDAWeights | None = None
        # npu_causal_conv1d_custom consumes an activation-dtype [W, 3C]
        # tensor; materialize that layout once after weight loading.
        self.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            nn.Parameter(
                torch.empty(
                    self.conv_size,
                    3 * self.local_projection_size,
                    dtype=self.model_config.dtype,
                    device=self.conv1d.weight.device,
                ),
                requires_grad=False,
            ),
        )
        original_process_weights = self.conv1d.quant_method.process_weights_after_loading

        @wraps(original_process_weights)
        def process_weights_and_pack(*args, **kwargs):
            result = original_process_weights(*args, **kwargs)
            self._pack_conv_weights()
            # Weight reload invalidates the NZ operand cache.
            self._mega_weights = None
            # Build the operator operands while weight processing is still
            # eager.  Otherwise ACL graph capture records the ND/NZ casts,
            # clones, and Conv layout conversion and replays them around
            # every MegaKDA invocation.
            if self.use_mega_kda:
                self._prepare_mega_weights()
            return result

        self.conv1d.quant_method.process_weights_after_loading = process_weights_and_pack

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "use_mega_kda", False):
            plan = self._plan_mega_kda(hidden_states)
            if plan is not None:
                return self._forward_mega_kda(hidden_states, positions, plan)
        return self._forward_default(hidden_states, positions)

    def _forward_default(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.uses_mixed_projection:
            num_tokens = hidden_states.size(0)
            mixed_qkv, beta, g1, g2 = self._run_overlapped_qkv_bfg(hidden_states)
            core_attn_out = torch.empty(
                (1, num_tokens, self.local_num_heads, self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._forward(
                mixed_qkv=mixed_qkv,
                g1=g1,
                g2=g2,
                beta=beta,
                core_attn_out=core_attn_out,
                beta_is_preprocessed=True,
            )
            core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
            return self.o_proj(core_attn_out)[0]
        return super().forward(hidden_states, positions)

    def _resolve_use_mega_kda(self) -> bool:
        """Resolve the Mega-KDA switch: config on and operator importable.

        BF16 is checked lazily at the first weight build so runtime
        dequantized checkpoints are not blocked here.
        """
        try:
            enabled = bool(getattr(get_ascend_config(), "enable_mega_kda", False))
        except Exception as error:
            logger.warning_once("mega_kda off: reading ascend config failed (%s)", error)
            enabled = False
        if not enabled:
            logger.warning_once("mega_kda off: enable_mega_kda is not set")
            return False
        if _mega_kda_impl is None:
            raise ImportError(
                "enable_mega_kda=True but ops.mega_recurrent_kda is not available"
            ) from _mega_kda_import_error
        logger.info_once(
            "MegaKDA enabled on %s (runtime B and S).",
            self.prefix,
        )
        return True

    def _resolve_use_fused_kda(self) -> bool:
        """Fused CANNBot DSL kernels handle every batch Mega-KDA does not.

        Mega-KDA only takes qualifying verify batches; prefill and
        non-qualifying batches fall back here (recipe architecture).
        AscendC stays as the last resort when the kernels are unavailable.
        """
        if _recurrent_kda_impl is None or _flash_kda_impl is None or _flash_kda_metadata_impl is None:
            logger.warning_once(
                "Fused KDA kernels (ops.flash_kda, ops.flash_kda_metadata, "
                "ops.fused_recurrent_kda_snapshot) are not available; using "
                "the AscendC KDA path. Install cannbot-dsl to enable the "
                "fused path."
            )
            return False
        if self.gate_lower_bound is None:
            logger.warning_once("Fused KDA kernels require gate_lower_bound; using the AscendC KDA path.")
            return False
        return True

    def _plan_mega_kda(self, hidden_states: torch.Tensor) -> _MegaKDAPlan | None:
        # This Python dispatch remains visible to Dynamo. A trace without
        # metadata selects the default path, whose core dispatch is opaque.
        # Compiled Mega-KDA selection needs a separate runtime dispatch boundary.
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if not isinstance(attn_metadata_raw, dict) or self.prefix not in attn_metadata_raw:
            return None
        attn_metadata = attn_metadata_raw[self.prefix]
        if not isinstance(attn_metadata, GDNAttentionMetadata):
            return None
        # Match the recipes implementation: Mega-KDA is selected for every
        # non-prefill forward.  The recipe does not distinguish speculative
        # and ordinary decode here; the metadata below supplies the correct
        # request/state layout for either form.
        if attn_metadata.num_prefills:
            logger.info_once("mega_kda skip: prefill step")
            return None

        is_spec_decode = attn_metadata.num_spec_decodes > 0
        if is_spec_decode:
            spec_meta = attn_metadata.spec_decode_metadata
            if spec_meta is None or attn_metadata.spec_state_indices_tensor is None:
                logger.info_once("mega_kda skip: spec decode metadata missing")
                return None
            qloc = spec_meta.spec_causal_conv1d.query_start_loc
            state_indices = attn_metadata.spec_state_indices_tensor
            accepted = attn_metadata.num_accepted_tokens
            if accepted is None:
                raise RuntimeError("MegaKDA metadata is missing accepted-token counts")
        else:
            qloc = attn_metadata.non_spec_query_start_loc
            state_indices = attn_metadata.non_spec_state_indices_tensor
            if qloc is None or state_indices is None:
                logger.info_once("mega_kda skip: decode metadata missing")
                return None
            # Ordinary decode advances one token per request.  Keep the same
            # accepted-count ABI as the recipes implementation.
            accepted = torch.ones(state_indices.shape[0], dtype=torch.int32, device=state_indices.device)

        batch = int(qloc.numel() - 1)
        if batch <= 0 or hidden_states.ndim != 2 or hidden_states.shape[0] % batch:
            logger.info_once("mega_kda skip: hidden rows do not form a runtime batch")
            return None
        if batch % self.tp_size != 0 or batch > self.tp_size * _MEGA_KDA_MAX_BATCH_MULTIPLIER:
            logger.info_once(
                "mega_kda skip: batch=%d must be divisible by TP=%d and <= TP*%d",
                batch,
                self.tp_size,
                _MEGA_KDA_MAX_BATCH_MULTIPLIER,
            )
            return None
        spec_len = hidden_states.shape[0] // batch
        expected_spec_len = self.num_spec + 1
        if spec_len != expected_spec_len:
            logger.info_once(
                "mega_kda skip: spec_len=%d, expected DSpark verify length=%d",
                spec_len,
                expected_spec_len,
            )
            return None
        accepted = accepted[:batch]
        first_verify = getattr(attn_metadata, "mega_first_verify", None)
        if is_spec_decode and first_verify is not None:
            conv_accepted = torch.where(
                first_verify[:batch] > 0,
                torch.full_like(accepted, spec_len),
                accepted,
            )
        else:
            conv_accepted = accepted
        if state_indices.ndim == 1:
            conv_state_block_ids = state_indices[:batch]
            ssm_state_indices = state_indices[:batch].reshape(-1)
        else:
            conv_state_block_ids = state_indices[:batch, 0]
            ssm_state_indices = state_indices[:batch, :spec_len].reshape(-1)
        return _MegaKDAPlan(
            batch=batch,
            spec_len=spec_len,
            conv_state_block_ids=conv_state_block_ids,
            ssm_state_indices=ssm_state_indices,
            conv_num_accepted_tokens=conv_accepted,
            num_accepted_tokens=accepted,
        )

    def _build_mega_weights(self) -> MegaKDAWeights:
        """Assemble the cached operator operand set.

        Supports the standard packed layout ([QKV | G | f_a | B] rows plus
        f_b_proj) and the QuaRot mixed layout (QKV-only in_proj plus
        fused_bfg_proj [B | F | G] rows).  Operands are cloned to standalone
        offset-0 tensors before the NZ cast (npu_format_cast mishandles
        offset views); conv is re-layouted to [3, H, W, D].
        """
        # Take NZ-cast weights back to ND before slicing.
        o_proj_weight = ensure_mega_kda_nd(self.o_proj.weight.detach())
        o_norm_weight = self.o_norm.weight.detach()
        if o_norm_weight.dtype != torch.float32:
            o_norm_weight = o_norm_weight.to(torch.float32)
        conv = ensure_mega_kda_nd(self.get_parameter(_PACKED_CONV_WEIGHT_NAME).detach())
        conv = ensure_mega_kda_bf16(conv)
        if self.uses_mixed_projection:
            # QuaRot layout: QKV-only in_proj, B/F/G in fused_bfg_proj.
            weight = ensure_mega_kda_nd(self.in_proj_qkvgfab.weight.detach())
            fused = self.fused_bfg_proj
            fused_weight = ensure_mega_kda_nd(fused.weight.detach())
            out_dim = getattr(fused.weight, "output_dim", 0)
            qkv_src = weight[: 3 * self.local_projection_size]
            f_a_src = ensure_mega_kda_nd(fused.f_a_weight.detach())
            f_b_src = ensure_mega_kda_nd(fused.f_b_weight.detach())
            beta_src = fused_weight.narrow(out_dim, 0, self.local_num_heads)
            gate_src = fused_weight.narrow(
                out_dim,
                self.local_num_heads + self.local_projection_size,
                self.local_projection_size,
            )
        else:
            weight = ensure_mega_kda_nd(self.in_proj_qkvgfab.weight.detach())
            local_projection = self.local_projection_size
            qkv_end = 3 * local_projection
            gate_end = qkv_end + local_projection
            f_a_end = gate_end + self.head_dim
            qkv_src = weight[:qkv_end]
            gate_src = weight[qkv_end:gate_end]
            f_a_src = weight[gate_end:f_a_end]
            beta_src = weight[f_a_end : f_a_end + self.local_num_heads]
            f_b_src = ensure_mega_kda_nd(self.f_b_proj.weight.detach())
        return MegaKDAWeights(
            qkv=mega_kda_nz_operand(qkv_src),
            f_a=mega_kda_nz_operand(f_a_src),
            f_b=mega_kda_nz_operand(f_b_src),
            beta=mega_kda_nz_operand(beta_src),
            output_gate=mega_kda_nz_operand(gate_src),
            o_norm=o_norm_weight,
            o_proj=mega_kda_nz_operand(o_proj_weight),
            conv=conv.view(self.conv_size, 3, self.local_num_heads, self.head_dim).permute(1, 2, 0, 3).contiguous(),
        )

    def _prepare_mega_weights(self) -> bool:
        """Build and cache MegaKDA operands outside the captured forward."""
        if getattr(self, "_mega_weights", None) is not None:
            return True
        if self.in_proj_qkvgfab.weight.dtype != torch.bfloat16:
            return False
        self._mega_weights = self._build_mega_weights()
        return True

    @eager_break_during_capture
    def _forward_mega_kda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        plan: _MegaKDAPlan,
    ) -> torch.Tensor:
        """Run the fused Mega operator for a qualifying batch.

        Only called from ``forward`` when the plan already qualified; the
        ``@eager_break_during_capture`` decorator must NOT cover the
        fallback path (it would force fused/AscendC execution out of the
        graph on every step, corrupting AttnRes partial_block shapes).
        """
        if not self._prepare_mega_weights():
            # Lazy BF16 gate, checked after loading/dequantization.
            logger.warning_once(
                "mega_kda off: projection weights are %s, want bf16; "
                "dequantize the KDA projections or use a BF16 checkpoint",
                self.in_proj_qkvgfab.weight.dtype,
            )
            self.use_mega_kda = False
            return self._forward_default(hidden_states, positions)
        logger.info_once("mega_kda on: batch=%d", plan.batch)
        conv_state, recurrent_state = self.kv_cache
        if (
            conv_state.ndim != 3
            or conv_state.shape[1] < 10
            or conv_state.shape[2] != 3 * self.local_num_heads * self.head_dim
            or recurrent_state.ndim != 4
            or recurrent_state.shape[1:] != (self.local_num_heads, self.head_dim, self.head_dim)
            or conv_state.dtype != torch.bfloat16
            or recurrent_state.dtype != torch.float32
        ):
            raise RuntimeError(
                "MegaKDA requires KDA cache shapes "
                f"conv=[C,>=10,{3 * self.local_num_heads * self.head_dim}] BF16 and "
                f"recurrent=[R,{self.local_num_heads},{self.head_dim},{self.head_dim}] FP32; "
                f"got conv={tuple(conv_state.shape)}/{conv_state.dtype}, "
                f"recurrent={tuple(recurrent_state.shape)}/{recurrent_state.dtype}"
            )
        output = run_mega_recurrent_kda(
            hidden_states=hidden_states.reshape(plan.batch, plan.spec_len, -1),
            weights=self._mega_weights,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_state_block_ids=plan.conv_state_block_ids,
            ssm_state_indices=plan.ssm_state_indices,
            conv_num_accepted_tokens=plan.conv_num_accepted_tokens,
            num_accepted_tokens=plan.num_accepted_tokens,
            a_log=self.A_log.detach().reshape(-1).contiguous(),
            dt_bias=self.dt_bias.view(self.local_num_heads, self.head_dim),
            scale=self.head_dim**-0.5,
            gate_lower_bound=float(self.gate_lower_bound if self.gate_lower_bound is not None else -5.0),
            o_norm_variance_epsilon=float(self.o_norm.eps),
        )
        # The operator applies the row-parallel o_proj internally; mirror its
        # reduction semantics (reduce_results is False on the SP path).
        if self.tp_size > 1 and self.o_proj.reduce_results:
            tensor_model_parallel_all_reduce(output)
        # The operator only covers the live verify tokens; restore the layer's
        # input token count so AttnRes partial_block and delta stay aligned.
        return _restore_token_rows(output, hidden_states.size(0))

    def _run_overlapped_qkv_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fork BFG work to an auxiliary stream and join it back to main."""
        if torch.compiler.is_compiling() and not get_ascend_config().ascend_compilation_config.enable_npugraph_ex:
            # The fusion-pass backend uses AOTAutograd, which cannot execute
            # torch_npu's fake_record_stream nodes from the multi-stream path.
            fused_bfg = self._project_bfg(hidden_states)
            quantized_qkv = self._quantize_fused_qkv(hidden_states)
            mixed_qkv = self._matmul_fused_qkv(quantized_qkv)
            beta, g1, g2 = self._postprocess_bfg(fused_bfg)
            return mixed_qkv, beta, g1, g2

        main_stream = torch.npu.current_stream()
        bfg_stream = _kda_bfg_stream()

        hidden_states_ready = main_stream.record_event()
        hidden_states.record_stream(bfg_stream)
        with npu_stream_switch(bfg_stream):
            bfg_stream.wait_event(hidden_states_ready)
            fused_bfg = self._project_bfg(hidden_states)
            bfg_projection_ready = bfg_stream.record_event()

        quantized_qkv = self._quantize_fused_qkv(hidden_states)
        quant_ready = main_stream.record_event()

        # Stage 1 join: DynamicQuant on main overlaps the BFG GEMM on the
        # auxiliary stream, but the two Cube matmuls remain serialized.
        main_stream.wait_event(bfg_projection_ready)
        mixed_qkv = self._matmul_fused_qkv(quantized_qkv)

        with npu_stream_switch(bfg_stream):
            # Stage 2: after both first-stage branches complete, overlap the
            # QKV Cube matmul with beta's FP32 conversion and sigmoid vector
            # work. Split and reshape the F/output gates here as well so all
            # BFG output handling occurs after the QKV matmul is enqueued.
            bfg_stream.wait_event(quant_ready)
            beta, g1, g2 = self._postprocess_bfg(fused_bfg)
            bfg_ready = bfg_stream.record_event()

        for tensor in (beta, g1, g2):
            tensor.record_stream(main_stream)
        # bfg_ready is the auxiliary stream tail. Joining that exact event is
        # required for multi-stream ACL graph capture as well as eager reuse.
        main_stream.wait_event(bfg_ready)
        return mixed_qkv, beta, g1, g2

    def _project_bfg(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.fused_bfg_proj(hidden_states)[0]

    def _postprocess_bfg(
        self,
        fused_bfg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta, raw_gate, output_gate = fused_bfg.split(
            self._fused_bfg_output_sizes,
            dim=-1,
        )
        if getattr(self, "use_fused_kda", False):
            # Keep raw beta logits; the fused KDA kernels apply the sigmoid.
            beta = beta.unsqueeze(0)
        else:
            beta = beta.float().sigmoid().unsqueeze(0)
        raw_gate = rearrange(raw_gate, "n (h d) -> 1 n h d", d=self.head_dim)
        output_gate = rearrange(output_gate, "n (h d) -> n h d", d=self.head_dim)
        return beta, raw_gate, output_gate

    def _quantize_fused_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        quant_method = self.in_proj_qkvgfab.quant_method
        inner_quant_method = getattr(quant_method, "quant_method", quant_method)
        if (
            isinstance(
                inner_quant_method,
                (
                    AscendW4A8MXFPDynamicLinearMethod,
                    AscendW8A8MXFP8DynamicLinearMethod,
                ),
            )
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
        ):
            if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
                return torch_npu.npu_dynamic_mx_quant(
                    hidden_states,
                    dst_type=torch.float8_e4m3fn,
                    scale_alg=inner_quant_method.dynamic_mx_quant_scale_alg,
                )
            return torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
            )
        return hidden_states

    def _matmul_fused_qkv(
        self,
        qkv_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        quant_method = self.in_proj_qkvgfab.quant_method
        if quant_method is None:
            raise RuntimeError("KDA fused QKV quantization method is not initialized")
        return quant_method.apply(
            self.in_proj_qkvgfab,
            qkv_input,
            bias=None,
        )

    @staticmethod
    def _run_causal_conv1d(
        mixed_qkv: torch.Tensor,
        conv_weights_t: torch.Tensor,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        *,
        run_mode: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(mixed_qkv)
        # Consume the operator's declared output alias. Returning ``output``
        # independently would let graph functionalization treat the custom-op
        # result as dead and expose the uninitialized allocation instead.
        return torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            mixed_qkv,
            conv_weights_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )

    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        if self.conv1d.weight.is_meta:
            return
        packed_param = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        packed_weight = (
            self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            .transpose(0, 1)
            .to(device=packed_param.device, dtype=packed_param.dtype)
            .contiguous()
        )
        replace_parameter(
            self,
            _PACKED_CONV_WEIGHT_NAME,
            packed_weight,
            prefer_copy=True,
        )

    def _run_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=(self.gate_lower_bound if self.gate_lower_bound is not None else -5.0),
        )

    def _run_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # The recurrent cache uses [H,V,K]. The fused prefill operator accepts
        # that state layout directly through state_v_first.
        initial_state_vk = recurrent_state[state_indices].contiguous()
        clear_ssm_states(initial_state_vk, has_initial_state)

        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            self.head_dim**-0.5,
            _KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_vk,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=prebuilt_metadata.chunk_indices_chunk64_host,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
            use_gate_in_kernel=True,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            disable_recompute=False,
            return_intermediate_states=False,
            state_v_first=True,
        )
        recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)
        return result[0]

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        *,
        beta_is_preprocessed: bool = False,
    ) -> None:
        """Keep metadata-dependent KDA execution outside Dynamo tracing."""
        torch.ops.vllm.ascend_kda_forward(mixed_qkv, g1, g2, beta, core_attn_out, self.prefix, beta_is_preprocessed)

    @eager_break_during_capture
    def _kda_forward_impl(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        *,
        beta_is_preprocessed: bool = False,
    ) -> None:
        """Dispatch speculative, prefill, and decode tokens through KDA kernels."""
        if getattr(self, "use_fused_kda", False):
            # Fused KDA path: raw beta logits, packed QKV, TND prefill.
            self._forward_fused_kda(mixed_qkv, g1, g2, beta, core_attn_out)
            return
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        g2 = g2[:num_actual_tokens]
        beta = _prepare_beta(
            beta,
            num_actual_tokens,
            is_preprocessed=beta_is_preprocessed,
        )

        conv_state, recurrent_state = self.kv_cache
        conv_weights_t = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None
                assert non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                None,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_spec.chunk(3, dim=-1)
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent(
                q_spec,
                k_spec,
                v_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_recurrent_output(
                core_spec,
                attn_metadata.spec_query_start_loc,
            )

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    prefill_meta.causal_conv1d.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    None,
                    run_mode=1,
                )

            q_non_spec, k_non_spec, v_non_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim) for x in mixed_non_spec.chunk(3, dim=-1)
            )
            assert raw_gate_non_spec is not None
            assert beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent(
                    q_non_spec[:, :num_decode_tokens],
                    k_non_spec[:, :num_decode_tokens],
                    v_non_spec[:, :num_decode_tokens],
                    raw_gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                if split_non_spec:
                    q_non_spec = q_non_spec[:, num_decode_tokens:]
                    k_non_spec = k_non_spec[:, num_decode_tokens:]
                    v_non_spec = v_non_spec[:, num_decode_tokens:]
                    raw_gate_non_spec = raw_gate_non_spec[:, num_decode_tokens:]
                    beta_non_spec = beta_non_spec[:, num_decode_tokens:]

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            core_non_spec = _zero_padded_recurrent_output(
                core_non_spec,
                attn_metadata.non_spec_query_start_loc,
            )

        if core_spec is None and core_non_spec is None:
            # Idle DP dummy runs carry graph-shaped metadata with no live work.
            # Do not feed a previous replay's output through the norm gate.
            core_attn_out.zero_()
            return

        num_live_tokens = None
        if core_spec is not None:
            assert attn_metadata.spec_query_start_loc is not None
            num_live_tokens = attn_metadata.spec_query_start_loc[-1]
        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            num_non_spec_tokens = attn_metadata.non_spec_query_start_loc[-1]
            num_live_tokens = num_non_spec_tokens if num_live_tokens is None else num_live_tokens + num_non_spec_tokens
        assert num_live_tokens is not None

        # Reuse the caller-owned result buffer. FULL graphs can leave rows
        # outside the live spec/non-spec index sets, so define them before the
        # two index copies rather than allocating a temporary merged tensor.
        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None
            assert non_spec_token_indices is not None
            assert spec_token_indices.numel() + non_spec_token_indices.numel() <= num_actual_tokens
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec

        # The registered Ascend FusedRMSNormGated uses the fused norm-gate
        # kernel while preserving the upstream parameter/loading contract.
        normalized = self.o_norm(core_attn_out[:, :num_actual_tokens], g2)
        # Mask again after the norm gate: zero * sigmoid(NaN) is still NaN in
        # static padding rows whose captured gate values are not live.
        core_attn_out[:, :num_actual_tokens].copy_(_zero_padded_output(normalized, num_live_tokens))
        core_attn_out[:, num_actual_tokens:].zero_()

    # ------------------------------------------------------------------
    # Fused KDA path (default unless Mega-KDA is on): CANNBot DSL kernels
    # with raw beta logits, packed QKV and TND prefill.
    # ------------------------------------------------------------------

    def _run_recurrent_fused_kda(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cu_seqlens
        if state_indices.ndim == 1:
            batch_size = state_indices.shape[0]
            sequence_length = mixed_qkv.shape[0] // batch_size
        elif state_indices.ndim == 2:
            batch_size, sequence_length = state_indices.shape
        else:
            raise ValueError(f"KDA state indices must be rank 1 or 2, got rank {state_indices.ndim}")

        expected_tokens = batch_size * sequence_length
        if mixed_qkv.shape[0] != expected_tokens:
            raise ValueError(
                "KDA recurrent input and state indices describe different token counts: "
                f"{mixed_qkv.shape[0]} != {batch_size} * {sequence_length}"
            )

        # The packaged snapshot ABI consumes the packed ShortConv QKV directly
        # and fuses the Q/K split, L2 norm, and gate/beta activation inside the
        # kernel. It requires int32 state indices and accepted lengths, updates
        # the state pool in place, and returns out [B, S, H, D] bf16.
        output = _recurrent_kda_impl(
            mixed_qkv.reshape(batch_size, sequence_length, mixed_qkv.shape[-1]).contiguous(),
            state=recurrent_state,
            beta=raw_beta.reshape(batch_size, sequence_length, self.local_num_heads, 1).contiguous(),
            g=raw_gate.reshape(batch_size, sequence_length, self.local_num_heads, self.head_dim).contiguous(),
            scale=self.head_dim**-0.5,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.reshape(self.local_num_heads, self.head_dim).contiguous(),
            lower_bound=self.gate_lower_bound,
            layout_qkv="BSND",
            ssm_state_indices=state_indices.reshape(-1).clamp_min(0).to(torch.int32).contiguous(),
            num_accepted_tokens=None if num_accepted_tokens is None else num_accepted_tokens.to(torch.int32),
        )
        return output.reshape(1, expected_tokens, self.local_num_heads, self.head_dim)

    def _run_prefill_fused_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        prebuilt_metadata,
    ) -> torch.Tensor:
        cu_seqlens = (
            prebuilt_metadata.cu_seqlens_host
            if prebuilt_metadata.cu_seqlens_kern is None
            else prebuilt_metadata.cu_seqlens_kern
        )
        keep = prebuilt_metadata.keep_meta
        if keep is not None:
            state_indices = state_indices[keep]
            has_initial_state = has_initial_state[keep]

        # Preserve vLLM's cache-derived initial state for continued prefills.
        initial_state = recurrent_state[state_indices].float().contiguous()
        clear_ssm_states(initial_state, has_initial_state)

        num_sequences = state_indices.shape[0]
        if has_initial_state.numel() != num_sequences:
            raise ValueError(
                "Kimi KDA prefill metadata is inconsistent: state_indices and "
                "has_initial_state must describe the same number of sequences."
            )
        # TND consumes the real packed tokens and int32 request offsets; the
        # kernel handles tail chunks, so no per-request padding, host sync, or
        # launch loop is needed (recipes KimiDeltaAttention._prefill_flash_kda).
        if isinstance(cu_seqlens, torch.Tensor):
            query_start_loc = cu_seqlens.to(torch.int32).contiguous()
        else:
            query_start_loc = torch.tensor(cu_seqlens, dtype=torch.int32, device=q.device)
        if query_start_loc.numel() != num_sequences + 1:
            raise ValueError(
                "Kimi KDA prefill metadata is inconsistent: compact cu_seqlens, "
                "state_indices, and has_initial_state must describe the same number of sequences."
            )

        q, k, v, raw_gate, raw_beta = (tensor.contiguous() for tensor in (q, k, v, raw_gate, raw_beta))
        metadata = _flash_kda_metadata_impl(
            q,
            v,
            initial_state,
            layout_qkv="TND",
            cu_seqlens=query_start_loc,
        )
        output, final_state = _flash_kda_impl(
            q,
            k,
            v,
            g=raw_gate,
            beta=raw_beta,
            scale=self.head_dim**-0.5,
            initial_state=initial_state,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.reshape(self.local_num_heads, self.head_dim).contiguous(),
            lower_bound=self.gate_lower_bound,
            layout_qkv="TND",
            metadata=metadata,
        )
        recurrent_state[state_indices] = final_state.to(recurrent_state.dtype)
        return output.unsqueeze(0)

    def _forward_fused_kda(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """Dispatch speculative, prefill, and decode tokens through the CANNBot DSL kernels."""
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        g2 = g2[:num_actual_tokens]
        # Preserve vLLM's beta keyword while passing raw logits to CANNBot.
        raw_beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        conv_weights_t = self.get_parameter(_PACKED_CONV_WEIGHT_NAME)
        spec_masks = attn_metadata.spec_sequence_masks
        spec_token_indices = attn_metadata.spec_token_indx
        non_spec_token_indices = attn_metadata.non_spec_token_indx

        if spec_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_spec = mixed_qkv
                raw_gate_spec = g1
                beta_spec = raw_beta
                mixed_non_spec = raw_gate_non_spec = beta_non_spec = None
            else:
                assert spec_token_indices is not None
                assert non_spec_token_indices is not None
                mixed_spec = mixed_qkv.index_select(0, spec_token_indices)
                raw_gate_spec = g1.index_select(1, spec_token_indices)
                beta_spec = raw_beta.index_select(1, spec_token_indices)
                mixed_non_spec = mixed_qkv.index_select(0, non_spec_token_indices)
                raw_gate_non_spec = g1.index_select(1, non_spec_token_indices)
                beta_non_spec = raw_beta.index_select(1, non_spec_token_indices)
        else:
            mixed_spec = raw_gate_spec = beta_spec = None
            mixed_non_spec = mixed_qkv
            raw_gate_non_spec = g1
            beta_non_spec = raw_beta

        core_spec = None
        if mixed_spec is not None:
            spec_meta = attn_metadata.spec_decode_metadata
            assert spec_meta is not None
            spec_conv_meta = spec_meta.spec_causal_conv1d
            mixed_spec = self._run_causal_conv1d(
                mixed_spec,
                conv_weights_t,
                conv_state,
                spec_conv_meta.query_start_loc,
                spec_conv_meta.cache_indices,
                None,
                run_mode=1,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            assert raw_gate_spec is not None and beta_spec is not None
            assert attn_metadata.spec_query_start_loc is not None
            assert attn_metadata.spec_state_indices_tensor is not None
            core_spec = self._run_recurrent_fused_kda(
                mixed_spec,
                raw_gate_spec,
                beta_spec,
                recurrent_state,
                attn_metadata.spec_query_start_loc,
                attn_metadata.spec_state_indices_tensor,
                num_accepted_tokens=spec_conv_meta.num_accepted_tokens,
            )
            core_spec = _zero_padded_recurrent_output(
                core_spec,
                attn_metadata.spec_query_start_loc,
            )

        core_non_spec = None
        if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
            if attn_metadata.num_prefills > 0:
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    prefill_meta.causal_conv1d.query_start_loc,
                    prefill_meta.causal_conv1d.cache_indices,
                    prefill_meta.causal_conv1d.initial_state_mode,
                    run_mode=0,
                )
            elif attn_metadata.num_decodes > 0:
                decode_meta = attn_metadata.non_spec_decode_metadata
                assert decode_meta is not None
                mixed_non_spec = self._run_causal_conv1d(
                    mixed_non_spec,
                    conv_weights_t,
                    conv_state,
                    decode_meta.causal_conv1d.query_start_loc,
                    decode_meta.causal_conv1d.cache_indices,
                    None,
                    run_mode=1,
                )

            assert raw_gate_non_spec is not None
            assert beta_non_spec is not None

            split_non_spec = spec_masks is None and attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
            num_decode_tokens = attn_metadata.num_decode_tokens
            core_decode = None
            if split_non_spec:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_decode = self._run_recurrent_fused_kda(
                    mixed_non_spec[:num_decode_tokens],
                    raw_gate_non_spec[:, :num_decode_tokens],
                    beta_non_spec[:, :num_decode_tokens],
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                )

            if attn_metadata.num_prefills > 0:
                prefill_mixed = mixed_non_spec[num_decode_tokens:] if split_non_spec else mixed_non_spec
                prefill_gate = raw_gate_non_spec[:, num_decode_tokens:] if split_non_spec else raw_gate_non_spec
                prefill_beta = beta_non_spec[:, num_decode_tokens:] if split_non_spec else beta_non_spec

                # TND layout: q/k/v/g as [tokens, H, D] and beta as [tokens, H].
                q_non_spec, k_non_spec, v_non_spec = (
                    rearrange(x, "n (h d) -> n h d", d=self.head_dim) for x in prefill_mixed.chunk(3, dim=-1)
                )

                assert attn_metadata.prefill_state_indices is not None
                assert attn_metadata.prefill_has_initial_state is not None
                prefill_meta = attn_metadata.non_spec_prefill_metadata
                assert prefill_meta is not None
                core_prefill = self._run_prefill_fused_kda(
                    q_non_spec,
                    k_non_spec,
                    v_non_spec,
                    prefill_gate.squeeze(0),
                    prefill_beta.squeeze(0),
                    recurrent_state,
                    attn_metadata.prefill_state_indices,
                    attn_metadata.prefill_has_initial_state,
                    prefill_meta.chunk,
                )
                core_non_spec = (
                    torch.cat((core_decode, core_prefill), dim=1) if core_decode is not None else core_prefill
                )
            elif attn_metadata.num_decodes > 0:
                assert attn_metadata.non_spec_query_start_loc is not None
                assert attn_metadata.non_spec_state_indices_tensor is not None
                core_non_spec = self._run_recurrent_fused_kda(
                    mixed_non_spec,
                    raw_gate_non_spec,
                    beta_non_spec,
                    recurrent_state,
                    attn_metadata.non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    attn_metadata.non_spec_state_indices_tensor,
                )

        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            core_non_spec = _zero_padded_recurrent_output(
                core_non_spec,
                attn_metadata.non_spec_query_start_loc,
            )

        if core_spec is None and core_non_spec is None:
            # Idle DP dummy runs carry graph-shaped metadata with no live work.
            # Do not feed a previous replay's output through the norm gate.
            core_attn_out.zero_()
            return

        num_live_tokens = None
        if core_spec is not None:
            assert attn_metadata.spec_query_start_loc is not None
            num_live_tokens = attn_metadata.spec_query_start_loc[-1]
        if core_non_spec is not None:
            assert attn_metadata.non_spec_query_start_loc is not None
            num_non_spec_tokens = attn_metadata.non_spec_query_start_loc[-1]
            num_live_tokens = num_non_spec_tokens if num_live_tokens is None else num_live_tokens + num_non_spec_tokens
        assert num_live_tokens is not None

        # Reuse the caller-owned result buffer. FULL graphs can leave rows
        # outside the live spec/non-spec index sets, so define them before the
        # two index copies rather than allocating a temporary merged tensor.
        core_attn_out[:, :num_actual_tokens].zero_()
        if core_spec is not None and core_non_spec is not None:
            assert spec_token_indices is not None
            assert non_spec_token_indices is not None
            assert spec_token_indices.numel() + non_spec_token_indices.numel() <= num_actual_tokens
            core_attn_out[:, :num_actual_tokens].index_copy_(1, spec_token_indices, core_spec)
            core_attn_out[:, :num_actual_tokens].index_copy_(1, non_spec_token_indices, core_non_spec)
        elif core_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_spec
        elif core_non_spec is not None:
            core_attn_out[:, :num_actual_tokens] = core_non_spec

        # The registered Ascend FusedRMSNormGated uses the fused norm-gate
        # kernel while preserving the upstream parameter/loading contract.
        normalized = self.o_norm(core_attn_out[:, :num_actual_tokens], g2)
        # Mask again after the norm gate: zero * sigmoid(NaN) is still NaN in
        # static padding rows whose captured gate values are not live.
        core_attn_out[:, :num_actual_tokens].copy_(_zero_padded_output(normalized, num_live_tokens))
        core_attn_out[:, num_actual_tokens:].zero_()
