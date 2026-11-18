#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps

import custom_ops  # noqa: F401  # Register recipes custom ops and graph converters.
import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import tensor_model_parallel_all_gather, tensor_model_parallel_reduce_scatter
from vllm.logger import logger
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoEMethodBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.lora.fused_moe import has_lora
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    npu_stream_switch,
    shared_experts_calculation_stream,
    shared_experts_comm_stream,
)


def _record_stream(tensor: torch.Tensor, stream: torch.npu.Stream | None) -> None:
    """Register a cross-stream use so the caching allocator keeps the storage.

    CPU/meta tensors (unit tests, tracing) have no device stream to track.
    """
    if stream is not None and tensor.device.type == "npu":
        tensor.record_stream(stream)


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    after_dispatch: torch.npu.Event | None = field(default=None)
    after_gmm1: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    after_routed_finalize: torch.npu.Event | None = field(default=None)
    shared_experts_fused: bool = False


class SharedExpertParallelMode(Enum):
    """Effective activation and weight layout for a shared-expert forward."""

    TENSOR_PARALLEL = auto()  # Full activations, TP-sharded weights.
    SHARED_EXPERT_DATA_PARALLEL_ONLY = auto()  # Full activations, replicated weights (DP only).
    SEQUENCE_PARALLEL_ONLY = auto()  # Sharded activations, TP-sharded weights (SP only).
    SEQUENCE_PARALLEL_SEDP = auto()  # Sharded activations, replicated weights (SP + DP).


def _run_shared_sparse_situ(
    gate_up: torch.Tensor,
    expert_tokens: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    return torch.ops.custom.npu_situ_and_mul_sparse(
        gate_up,
        expert_tokens,
        beta=beta,
        alpha=linear_beta,
        high_precision=False,
    )


class AscendSharedSparseSituAndMul(SituAndMul):
    """Use the recipes sparse SiTU activation for NPU BF16 shared experts."""

    def __init__(self, beta: float, linear_beta: float | None):
        super().__init__(beta=beta, linear_beta=linear_beta, compile_native=False)
        self.register_buffer(
            "_all_rows_expert_tokens",
            torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64),
            persistent=False,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu" or x.dtype != torch.bfloat16 or self.linear_beta is None:
            return super().forward_native(x)
        return _run_shared_sparse_situ(x, self._all_rows_expert_tokens, self.beta, self.linear_beta)


class AscendSharedExperts:
    """Ascend-owned shared expert executor.

    Keep the original shared expert module registered on ``AscendMoERunner``
    for checkpoint compatibility while moving split/overlap execution details
    out of the runner.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        quant_type: QuantType,
        quant_method: FusedMoEMethodBase,
        gate_up_core_limit: tuple[int, int] | None = None,
        use_sparse_situ: bool = False,
    ):
        self.layer = layer
        self.moe_config = moe_config
        self.hidden_size = moe_config.hidden_dim
        self.shared_expert_input_size = getattr(
            layer.gate_up_proj,
            "input_size",
            self.hidden_size,
        )
        self.in_dtype = moe_config.in_dtype
        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
        self.is_sequence_parallel = moe_config.is_sequence_parallel
        if use_sparse_situ and isinstance(layer.act_fn, SituAndMul) and layer.act_fn.linear_beta is not None:
            layer.act_fn = AscendSharedSparseSituAndMul(layer.act_fn.beta, layer.act_fn.linear_beta)
        self.situ_activation = layer.act_fn if isinstance(layer.act_fn, SituAndMul) else None
        self.quant_type = quant_type
        self.lora_context = None
        # Event of a pre-forked shared-input all-gather joined on the compute stream.
        self._pending_input_gather_evt = None
        # Stream that produces the shared-expert result consumed by the runner.
        self._output_stream = None
        if (
            self.quant_type == QuantType.W4A8MXFP
            and self.situ_activation is not None
            and not isinstance(self.situ_activation, AscendSharedSparseSituAndMul)
        ):
            layer.register_buffer(
                "_all_rows_expert_tokens",
                torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64),
                persistent=False,
            )
        ascend_config = get_ascend_config()
        self.multistream_overlap = ascend_config.multistream_overlap_shared_expert
        self.weights_replicated = ascend_config.enable_shared_expert_dp
        self._mega_moe_weights: MoEWeights | None = None
        self.gate_up_core_limit = (
            gate_up_core_limit if self.multistream_overlap and get_ascend_device_type() == AscendDeviceType.A5 else None
        )

        if self.multistream_overlap:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self.validate_consistency()
                return result

            quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    def get_mega_moe_weights(self) -> MoEWeights | None:
        """Return one replicated shared expert in MegaMoE's MXFP layout."""
        if self.quant_type not in (QuantType.W4A8MXFP, QuantType.W8A8MXFP):
            return None
        if has_lora(self.lora_context):
            return None

        cached_weights = getattr(self, "_mega_moe_weights", None)
        if cached_weights is not None:
            return cached_weights

        tp_size = self.moe_config.tp_group.world_size
        if tp_size > 1 and not self.weights_replicated:
            return None
        if getattr(self.layer, "expert_gate", None) is not None:
            return None
        if not type(self.layer.act_fn).__name__.startswith("SiluAndMul"):
            return None

        gate_up_proj = self.layer.gate_up_proj
        down_proj = self.layer.down_proj
        if getattr(gate_up_proj, "bias", None) is not None or getattr(down_proj, "bias", None) is not None:
            return None
        required_attrs = (
            (gate_up_proj, "weight"),
            (gate_up_proj, "weight_scale"),
            (down_proj, "weight"),
            (down_proj, "weight_scale"),
        )
        if any(not hasattr(module, name) for module, name in required_attrs):
            return None

        hidden_size = getattr(self, "shared_expert_input_size", self.moe_config.hidden_dim)
        intermediate_hidden = self.moe_config.intermediate_size_per_partition
        weight_dtypes = {gate_up_proj.weight.dtype, down_proj.weight.dtype}
        if weight_dtypes == {torch.float8_e4m3fn}:
            shared_quant_type = QuantType.W8A8MXFP
            weight_pack_factor = 1
        elif weight_dtypes == {torch.uint8}:
            shared_quant_type = QuantType.W4A8MXFP
            weight_pack_factor = 2
        else:
            return None

        expected_shapes = {
            "w1": (1, 2 * intermediate_hidden, hidden_size // weight_pack_factor),
            "w2": (1, hidden_size, intermediate_hidden // weight_pack_factor),
            "w1_scale": (1, 2 * intermediate_hidden, (hidden_size + 63) // 64, 2),
            "w2_scale": (1, hidden_size, (intermediate_hidden + 63) // 64, 2),
        }
        shared_weights = MoEWeights(
            w1=[gate_up_proj.weight.transpose(-1, -2).unsqueeze(0).contiguous()],
            w2=[down_proj.weight.transpose(-1, -2).unsqueeze(0).contiguous()],
            w1_scale=[gate_up_proj.weight_scale.transpose(-3, -2).unsqueeze(0).contiguous()],
            w2_scale=[down_proj.weight_scale.transpose(-3, -2).unsqueeze(0).contiguous()],
            quant_type=shared_quant_type,
        )
        actual_shapes = {
            "w1": tuple(shared_weights.w1[0].shape),
            "w2": tuple(shared_weights.w2[0].shape),
            "w1_scale": tuple(shared_weights.w1_scale[0].shape),
            "w2_scale": tuple(shared_weights.w2_scale[0].shape),
        }
        if actual_shapes != expected_shapes:
            return None

        self._mega_moe_weights = shared_weights
        return shared_weights

    def _gate_up_core_scope(self):
        """Reserve vector cores from the overlapping routed-expert path."""
        if self.gate_up_core_limit is None:
            return nullcontext()
        return torch_npu.npu.npugraph_ex.scope.limit_core_num(*self.gate_up_core_limit)

    def validate_consistency(self):
        """Validate that split shared expert computation matches integrated computation."""
        test_input = (
            torch.rand(
                10,
                self.shared_expert_input_size,
                device="npu",
                dtype=self.in_dtype,
            )
            * 2
            - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self.layer(test_input)
        part1_out = self.part1(test_input)
        split_out = self.part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error(
                "[fused_moe/layer] Shared expert split computation validation failed."
                " The split-path computation does not match the integrated-path result."
                " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                diff.max().item(),
                integrated_out.sum().item(),
                integrated_out.norm().item(),
                split_out.sum().item(),
                split_out.norm().item(),
                self.shared_expert_input_size,
                self.in_dtype,
            )
            raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
        logger.info(
            "[fused_moe/layer] Shared expert split computation validation passed."
            " Integrated and split-path results are consistent."
        )

    def part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.layer.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._activate_situ(shared_gate_up)
        shared_out, _ = self.layer.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self.layer, "expert_gate") and self.layer.expert_gate is not None:
            gate_out, _ = self.layer.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _activate_situ(self, gate_up: torch.Tensor, *, force_sparse: bool = False) -> torch.Tensor:
        if force_sparse:
            assert self.situ_activation is not None
            expert_tokens = (
                self.situ_activation._all_rows_expert_tokens
                if isinstance(self.situ_activation, AscendSharedSparseSituAndMul)
                else self.layer._all_rows_expert_tokens
            )
            return _run_shared_sparse_situ(
                gate_up, expert_tokens, self.situ_activation.beta, self.situ_activation.linear_beta or 0.0
            )
        return self.layer.act_fn(gate_up)

    def parallel_mode(self) -> SharedExpertParallelMode:
        """Resolve the effective activation/weight layout for this forward."""
        # EP rewrites FusedMoEParallelConfig.tp_size to 1 because each routed
        # expert is local. Shared-expert linears still span the physical TP
        # group, so their layout must be derived from that group instead.
        tp_size = self.moe_config.tp_group.world_size
        if tp_size <= 1:
            return SharedExpertParallelMode.TENSOR_PARALLEL

        if self.moe_config.is_sequence_parallel:
            # SP has already sharded the token dimension before entering the
            # runner. Replicated weights (SP+DP) compute directly on the
            # shard; TP-sharded weights (SP-only) gather the shard first and
            # reduce-scatter the output back.
            if self.weights_replicated:
                return SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP
            return SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        if self.weights_replicated:
            return SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
        return SharedExpertParallelMode.TENSOR_PARALLEL

    def _prepare_local_dp_input(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        original_num_tokens = hidden_states.shape[0]
        # See parallel_mode(): moe_config.tp_size describes routed experts in
        # EP, while this token split follows the shared-expert TP group.
        tp_group = self.moe_config.tp_group
        tp_size = tp_group.world_size
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
        hidden_states = torch.tensor_split(
            hidden_states,
            tp_size,
            dim=0,
        )[tp_group.rank_in_group]
        return hidden_states, (original_num_tokens, pad_size)

    def _finalize_local_dp_output(
        self,
        shared_out: torch.Tensor,
        metadata: tuple[int, int],
    ) -> torch.Tensor:
        original_num_tokens, pad_size = metadata
        shared_out = self.moe_config.tp_group.all_gather(shared_out, dim=0)
        if pad_size > 0:
            shared_out = shared_out[:original_num_tokens]
        return shared_out

    def _gather_sp_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gather SP-sharded activations to the full sequence via TP all-gather + unpad.

        SP shards the token dimension across the TP group as [ceil(T/TP), H]
        per rank (``sequence_parallel_chunk`` pads T to a multiple of TP).
        TP-sharded shared-expert weights need the full activations, so
        all-gather the shards back to [T_padded, H] and drop the padding rows.
        """
        if not getattr(self, "multistream_overlap", False):
            gathered = tensor_model_parallel_all_gather(hidden_states, dim=0)
            return gathered[: _EXTRA_CTX.num_tokens]

        # Fallback path (no pre-forked gather): run the collective on the comm
        # stream and join it before the compute continues on this stream.
        input_ready = torch.npu.current_stream().record_event()
        _record_stream(hidden_states, shared_experts_comm_stream())
        with npu_stream_switch(shared_experts_comm_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            gathered = tensor_model_parallel_all_gather(hidden_states, dim=0)
            _record_stream(gathered, shared_experts_calculation_stream())
            gather_done = torch.npu.current_stream().record_event()
        torch.npu.current_stream().wait_event(gather_done)
        return gathered[: _EXTRA_CTX.num_tokens]

    def _pad_and_reduce_scatter(self, shared_out: torch.Tensor) -> torch.Tensor:
        """Pad the full output to a TP multiple, then reduce-scatter to the SP shard.

        Called on exit of the SP-only path (TP-sharded weights) to convert the
        full [T, H] output back to this rank's SP shard [ceil(T/TP), H]. The
        reduce-scatter also sums the TP-partial down-projection results
        (down_proj is built with reduce_results=False), replacing the usual
        TP all-reduce.
        """
        tp_size = self.moe_config.tp_group.world_size
        original_num_tokens = shared_out.shape[0]
        pad_size = (tp_size - original_num_tokens % tp_size) % tp_size
        if pad_size > 0:
            shared_out = F.pad(shared_out, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(shared_out, dim=0)

    def start_input_all_gather(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.npu.Event | None]:
        """Start an SP gather that can overlap the routed input transform."""
        if not (self.multistream_overlap and self.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY):
            return hidden_states, None

        input_ready = torch.npu.current_stream().record_event()
        _record_stream(hidden_states, shared_experts_comm_stream())
        with npu_stream_switch(shared_experts_comm_stream(), enabled=True):
            torch.npu.current_stream().wait_event(input_ready)
            # Keep TP padding across the custom-op boundary; the shared MLP
            # trims it before computation.
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            _record_stream(hidden_states, shared_experts_calculation_stream())
            all_gather_done = torch.npu.current_stream().record_event()
        self._pending_input_gather_evt = all_gather_done
        return hidden_states, all_gather_done

    def wait_for_output(self) -> None:
        """Join the deferred SP shared collectives on the current stream."""
        if not self.multistream_overlap:
            return
        stream = getattr(self, "_output_stream", None) or shared_experts_calculation_stream()
        torch.npu.current_stream().wait_stream(stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fused_moe_evts: FusedMoEEvents,
        input_is_gathered: bool = False,
        defer_output_wait: bool = False,
    ):
        mode = self.parallel_mode()
        local_dp_metadata = None
        # MC2 (decode) choreography, mirroring the recipes event ladder while
        # accounting for vllm-ascend's fused routed GMM1+activation+quant op:
        # the shared activation is released right after dispatch (recipes'
        # event-6 equivalent) so it overlaps the routed GMM1 window, and the
        # down projection is released as soon as the routed GMM1 completes
        # (recipes' event-8 equivalent) so the reduce-scatter overlaps the
        # routed GMM2/combine window. When the activation is fused into GMM1
        # there is no post-GMM1 boundary and before_gmm2 is the earliest point.
        # Other comm types keep the original positioning (activation with
        # GMM2, down projection with the routed combine).
        mc2_comm = getattr(_EXTRA_CTX, "moe_comm_type", None) is MoECommType.MC2
        activation_ready = fused_moe_evts.after_dispatch if mc2_comm else fused_moe_evts.before_gmm2
        if mc2_comm:
            down_projection_ready = (
                fused_moe_evts.after_gmm1 if fused_moe_evts.after_gmm1 is not None else fused_moe_evts.before_gmm2
            )
        else:
            down_projection_ready = fused_moe_evts.before_combine

        def maybe_wait_event(evt: torch.npu.Event | None):
            # Without overlap, routed and shared kernels are enqueued on the
            # same stream. Their ordering is already guaranteed; the event
            # ladder belongs only to the auxiliary shared-compute stream.
            if self.multistream_overlap and evt is not None:
                torch.npu.current_stream().wait_event(evt)

        main_stream = torch.npu.current_stream()
        compute_done = None
        # Do not even acquire the auxiliary stream in serial mode: function
        # arguments are evaluated before npu_stream_switch checks enabled.
        compute_scope = (
            npu_stream_switch(shared_experts_calculation_stream(), enabled=True)
            if self.multistream_overlap
            else nullcontext()
        )
        with compute_scope:
            if self.multistream_overlap:
                # The input is allocated on the main stream and consumed here.
                _record_stream(hidden_states, shared_experts_calculation_stream())
            if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
                # Full activations + replicated weights: shard tokens locally,
                # run the MLP, then gather its complete output.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                hidden_states, local_dp_metadata = self._prepare_local_dp_input(hidden_states)
            elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
                # Sharded activations + TP-sharded weights: gather the SP
                # shard to full activations before the MLP; the output is
                # padded and reduce-scattered back below.
                pending_gather = getattr(self, "_pending_input_gather_evt", None)
                self._pending_input_gather_evt = None
                if not input_is_gathered:
                    maybe_wait_event(fused_moe_evts.before_routed_experts)
                    hidden_states = self._gather_sp_input(hidden_states)
                elif pending_gather is not None:
                    # The gather was forked on the comm stream before the routed
                    # input transform; join it here, on the consumer stream.
                    torch.npu.current_stream().wait_event(pending_gather)
            # Only used for int quantization
            has_quantized_shared_without_lora = (
                not has_lora(self.lora_context)
                and hasattr(self.layer.gate_up_proj, "weight_scale")
                and hasattr(self.layer.down_proj, "weight_scale")
            )
            if has_quantized_shared_without_lora and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.after_routed_experts)
                with self._gate_up_core_scope():
                    hidden_states = torch_npu.npu_quant_matmul(
                        quantized_x,
                        self.layer.gate_up_proj.weight,
                        self.layer.gate_up_proj.weight_scale,
                        pertoken_scale=None,
                        bias=None,
                        output_dtype=torch.int32,
                    )
                # Execute activation in the routed GMM1 window (MC2) or
                # concurrently with gmm2.

                maybe_wait_event(activation_ready)
                if self.situ_activation is not None:
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.dequant_situ_quant(
                        x=hidden_states,
                        weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                        activation_scale=pertoken_scale,
                        bias=None,
                        quant_scale=None,
                        quant_offset=None,
                        group_index=None,
                        beta=self.situ_activation.beta,
                        linear_beta=self.situ_activation.linear_beta or 0.0,
                        activate_left=True,
                        quant_mode="dynamic",
                    )
                else:
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                        x=hidden_states,
                        weight_scale=self.layer.gate_up_proj.weight_scale_fp32,
                        activation_scale=pertoken_scale,
                        bias=None,
                        quant_scale=None,
                        quant_offset=None,
                        group_index=None,
                        activate_left=True,
                        quant_mode=1,
                        swiglu_mode=1,
                        clamp_limit=self.swiglu_limit,
                        **(
                            {}
                            if get_ascend_device_type() == AscendDeviceType.A5
                            else {"glu_alpha": self.swiglu_alpha, "glu_bias": self.swiglu_beta}
                        ),
                    )
                maybe_wait_event(down_projection_ready)
                shared_out = torch_npu.npu_quant_matmul(
                    quantized_x,
                    self.layer.down_proj.weight,
                    self.layer.down_proj.weight_scale,
                    pertoken_scale=swiglu_out_scale,
                    bias=None,
                    output_dtype=original_dtype,
                )
            elif has_quantized_shared_without_lora and self.quant_type == QuantType.W4A8MXFP:
                original_dtype = hidden_states.dtype
                # Execute dynamic quant concurrently with MoE gate.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                with self._gate_up_core_scope():
                    hidden_states = self.layer.gate_up_proj((quantized_x, pertoken_scale))[0]
                # Execute activation in the routed GMM1 window (MC2) or
                # concurrently with gmm2.
                maybe_wait_event(activation_ready)
                if self.situ_activation is not None:
                    activated = self._activate_situ(hidden_states, force_sparse=True)
                    maybe_wait_event(down_projection_ready)
                    shared_out = self.layer.down_proj(activated)[0]
                else:
                    quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                        hidden_states,
                        topk_weight=None,
                        group_index=None,
                        dst_type=torch.float8_e4m3fn,
                        quant_mode=2,
                        clamp_value=self.swiglu_limit,
                    )
                    maybe_wait_event(down_projection_ready)
                    shared_out = self.layer.down_proj((quantized_x, swiglu_out_scale))[0]
            else:
                # Ensure the shared experts wait for hidden_states to be ready.
                maybe_wait_event(fused_moe_evts.before_routed_experts)
                # Execute the gate projection and activation concurrently with the
                # dispatch communication.
                maybe_wait_event(fused_moe_evts.before_dispatch)
                with self._gate_up_core_scope():
                    part1_out = self.part1(hidden_states)
                maybe_wait_event(down_projection_ready)
                shared_out = self.part2(hidden_states, part1_out)

            if self.multistream_overlap and mode in (
                SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY,
                SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY,
            ):
                compute_done = torch.npu.current_stream().record_event()

        # Shared-expert output collectives run on the comm stream so the compute
        # stream is never queued behind an HCCL task. The comm stream joins the
        # compute stream first, then the routed finalize when the reduced output
        # may share its communicator (non-MC2).
        tail_stream = None
        if self.multistream_overlap and mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY:
            comm_stream = shared_experts_comm_stream()
            with npu_stream_switch(comm_stream, enabled=True):
                # shared_out was allocated on the compute stream and is read by
                # the collective below; keep its storage alive for this stream.
                _record_stream(shared_out, comm_stream)
                if compute_done is not None:
                    torch.npu.current_stream().wait_event(compute_done)
                if not mc2_comm:
                    maybe_wait_event(fused_moe_evts.after_routed_finalize)
                shared_out = self._pad_and_reduce_scatter(shared_out)
                _record_stream(shared_out, main_stream)
            tail_stream = shared_experts_comm_stream()
        elif self.multistream_overlap:
            # No comm-stream hop here: the MLP output stays on the compute
            # stream and is consumed by the main stream after wait_stream.
            _record_stream(shared_out, main_stream)
            tail_stream = shared_experts_calculation_stream()

        if mode is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY:
            assert local_dp_metadata is not None
            if self.multistream_overlap:
                comm_stream = shared_experts_comm_stream()
                with npu_stream_switch(comm_stream, enabled=True):
                    _record_stream(shared_out, comm_stream)
                    if compute_done is not None:
                        torch.npu.current_stream().wait_event(compute_done)
                    shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
                    _record_stream(shared_out, main_stream)
                tail_stream = shared_experts_comm_stream()
            else:
                shared_out = self._finalize_local_dp_output(shared_out, local_dp_metadata)
        elif mode is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY and not self.multistream_overlap:
            shared_out = self._pad_and_reduce_scatter(shared_out)

        self._output_stream = tail_stream

        # Defer only when a routed output transform can overlap this collective;
        # otherwise preserve the original join point for non-latent models.
        if self.multistream_overlap and (
            mode is not SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY or not defer_output_wait
        ):
            torch.npu.current_stream().wait_stream(tail_stream)
        return shared_out
