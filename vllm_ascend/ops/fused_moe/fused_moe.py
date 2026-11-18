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
import torch
import torch.nn.functional as F
import torch_npu
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CompilationMode, get_current_vllm_config_or_none
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import MoERunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _layer_name_type,
    _moe_forward_shared,
    _resolve_layer_name,
    get_layer_from_name,
)
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.decode_overlap import ROUTER_CORE_LIMIT, DecodeSharedExpertSchedule
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    SharedExpertParallelMode,
)
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicFusedMoEMethod,
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
    npu_stream_switch,
    router_stream,
    vllm_version_is,
)


def _ascend_kimi_moe_fake(
    hidden_states: torch.Tensor, input_ids: torch.Tensor | None, layer_name: _layer_name_type
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _ascend_kimi_moe(
    hidden_states: torch.Tensor, input_ids: torch.Tensor | None, layer_name: _layer_name_type
) -> torch.Tensor:
    # Resolve once, like the upstream MoE op. Keep runtime phase selection and
    # all stream orchestration opaque to Dynamo, but capturable by ACLGraph.
    runner = get_layer_from_name(_resolve_layer_name(layer_name))
    return runner._forward_kimi_moe(hidden_states, input_ids)


direct_register_custom_op(
    op_name="ascend_kimi_moe",
    op_func=_ascend_kimi_moe,
    fake_impl=_ascend_kimi_moe_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _ascend_moe_forward_shared_sp_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: object,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer the local SP outputs of the shared-expert custom op.

    The explicit shared input contains all gathered tokens, while the real
    shared output has already been reduce-scattered back to the routed input's
    local token count.  The upstream fake uses the shared input's token count,
    which is only correct before the early all-gather optimization.
    """
    del router_logits, input_ids, layer_name
    assert shared_experts_input is not None
    shared_out = shared_experts_input.new_empty(
        (*hidden_states.shape[:-1], shared_experts_input.shape[-1]),
    )
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded),
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


@eager_break_during_capture
def _ascend_moe_forward_shared_sp(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: str,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the shared-expert MoE path eagerly during graph capture.

    The shared experts overlap their input gather, compute and output
    collective on different NPU streams.  This orchestration is plain Python,
    so it must not end up inside the captured graph, same as ``dsa_forward``.
    """
    return _moe_forward_shared(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        layer_name,
        hidden_dim_unpadded,
    )


# Keep the gathered shared input as an explicit graph dependency, but expose
# the SP-local shared output shape to torch.compile/ACL graph tracing.
direct_register_custom_op(
    op_name="ascend_moe_forward_shared_sp",
    op_func=_ascend_moe_forward_shared_sp,
    fake_impl=_ascend_moe_forward_shared_sp_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
    def __init__(
        self,
        layer_name,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: AscendRoutedExperts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1,
        shared_expert_gate_up_core_limit: tuple[int, int] | None = None,
        shared_expert_sparse_situ: bool = False,
    ):
        super().__init__(
            layer_name,
            moe_config,
            router,
            routed_experts,
            enable_dbo,
            gate,
            shared_experts,
            shared_expert_gate,
            routed_input_transform,
            routed_output_transform,
            routed_scaling_factor,
        )
        self._gate = gate
        self.hidden_size = moe_config.hidden_dim

        self.quant_type = routed_experts.quant_type
        self.routed_experts.router = router

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        if self.moe_config.ep_size > 1:
            self.moe_config.ep_group = get_ep_group()
            self.moe_config.mc2_group = get_mc2_group()

        self.ascend_shared_experts = None
        # The Kimi entry keeps its full-width input until latent projection.
        self._kimi_gate = None
        self._kimi_gate_multistream = False
        self._decode_shared_gate_up_core_limit = getattr(
            get_ascend_config(), "kimi_k3_decode_shared_gate_up_core_limit", ROUTER_CORE_LIMIT
        )
        if shared_experts is not None:
            vllm_config = get_current_vllm_config_or_none()
            # A compiled output transform cannot safely own a Python stream
            # wait for work launched inside the opaque MoE op. Join before
            # returning from that op; only eager execution may defer the wait.
            self._defer_sp_shared_output_wait = (
                vllm_config is not None and vllm_config.compilation_config.mode == CompilationMode.NONE
            )
            routed_experts.return_with_event = True
            self.ascend_shared_experts = AscendSharedExperts(
                shared_experts,
                self.moe_config,
                self.quant_type,
                self._quant_method,
                gate_up_core_limit=shared_expert_gate_up_core_limit,
                use_sparse_situ=shared_expert_sparse_situ,
            )
            if self._can_overlap_sp_shared_with(self.routed_input_transform):
                self._forward_entry = torch.ops.vllm.ascend_moe_forward_shared_sp

        setup_moe_comm_method(self.moe_config)
        alltoall_comm = get_moe_comm_method(MoECommType.ALLTOALL)
        if alltoall_comm is not None:
            expert_ids_per_ep_rank = getattr(alltoall_comm.token_dispatcher, "expert_ids_per_ep_rank", None)
            if expert_ids_per_ep_rank is not None:
                self.routed_experts.register_buffer(
                    "expert_ids_per_ep_rank",
                    expert_ids_per_ep_rank,
                    persistent=False,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_kimi_gate_multistream", False) and shared_experts_input is None:
            return torch.ops.vllm.ascend_kimi_moe(hidden_states, input_ids, self._encode_layer_name())
        return super().forward(hidden_states, router_logits, input_ids, shared_experts_input)

    def _can_run_decode_ladder(self, hidden_states: torch.Tensor) -> bool:
        shared = self.ascend_shared_experts
        if shared is None:
            return False
        shared_linears = (shared.layer.gate_up_proj, shared.layer.down_proj)
        bf16_shared = all(
            linear.weight.dtype == torch.bfloat16 and not hasattr(linear, "weight_scale") for linear in shared_linears
        )
        mxfp4_shared = all(
            type(getattr(getattr(linear, "quant_method", None), "quant_method", None))
            is AscendW4A8MXFPDynamicLinearMethod
            and hasattr(linear, "weight_scale")
            for linear in shared_linears
        )
        metadata = get_forward_context().attn_metadata
        metadata = list(metadata.values()) if isinstance(metadata, dict) else [metadata]
        # Missing or mixed/prefill metadata must not select the decode policy,
        # even if enable_prefill_mc2 makes MC2 available for that batch.
        decode_only = bool(metadata) and all(getattr(m, "num_prefills", None) == 0 for m in metadata)
        down_scheme = getattr(getattr(self.routed_input_transform, "quant_method", None), "quant_method", None)
        routed_scheme = getattr(self.routed_experts.quant_method, "quant_method", None)
        return (
            decode_only
            and getattr(self, "_kimi_gate_multistream", False)
            and _EXTRA_CTX.moe_comm_type is MoECommType.MC2
            and get_ascend_device_type() is AscendDeviceType.A5
            and self._can_overlap_sp_shared_with(self.routed_input_transform)
            and self.moe_config.is_sequence_parallel
            and hidden_states.dtype == torch.bfloat16
            and type(down_scheme) is AscendW8A8MXFP8DynamicLinearMethod
            and type(routed_scheme) is AscendW4A8MXFPDynamicFusedMoEMethod
            and not self.routed_experts.dynamic_eplb
            and getattr(self.routed_experts, "_ascend_moe_lora_context", None) is None
            and type(self.routed_experts.router) is AscendFusedTopKRouter
            and self.routed_experts.router.custom_routing_function is None
            and self.routed_experts.router.tid2eid is None
            and (bf16_shared or mxfp4_shared)
            and shared.situ_activation is not None
            and getattr(shared.layer, "expert_gate", None) is None
            and str(getattr(self.moe_config.activation, "value", self.moe_config.activation)) == "situ"
        )

    def _forward_kimi_moe(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Match MoERunner.forward's projection/padding/reduction contract inside
        # one opaque op. Its public output is always the original hidden width.
        schedule = None
        if self._can_run_decode_ladder(hidden_states):
            routed_input, logits, routing, shared_input, gather_done = self._decode_kimi_front(hidden_states, input_ids)
            routed_input, pre_trunc, post_trunc = self._maybe_pad_hidden_states(hidden_states, routed_input)
            schedule = DecodeSharedExpertSchedule(
                self.ascend_shared_experts,
                shared_input[: _EXTRA_CTX.num_tokens],
                gather_done,
                self._decode_shared_gate_up_core_limit,
                quantized_shared=hasattr(self.ascend_shared_experts.layer.gate_up_proj, "weight_scale"),
            )
            with self._sequence_parallel_context():
                fused_output, _ = self.routed_experts.forward_impl(
                    hidden_states=routed_input,
                    router_logits=logits,
                    input_ids=input_ids,
                    router_state=(logits, routing[2]),
                    precomputed_topk=routing,
                    shared_expert_schedule=schedule,
                )
            shared_output = None
        else:
            routed_input, shared_input = self.apply_routed_input_transform(hidden_states)
            routed_input, pre_trunc, post_trunc = self._maybe_pad_hidden_states(shared_input, routed_input)
            result = self._forward_impl(routed_input, hidden_states, shared_input, input_ids)
            shared_output, fused_output = result if isinstance(result, tuple) else (None, result)

        if pre_trunc is not None:
            fused_output = fused_output[..., :pre_trunc]
        fused_output, reduced = self._maybe_reduce_routed_output_before_transform(
            fused_output, self._fused_output_is_reduced
        )
        shared_output = self._maybe_reduce_shared_expert_output(shared_output, reduced)
        shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
        # The staged branch owns its output event; do not join a previous
        # invocation's legacy shared-output stream in apply_routed_output_transform.
        if schedule is not None:
            fused_output = super().apply_routed_output_transform(fused_output)
            shared_output = schedule.join()
        else:
            fused_output = self.apply_routed_output_transform(fused_output)
        result = fused_output if shared_output is None else shared_output + fused_output
        result = self._maybe_reduce_final_output(result, post_trunc, reduced)
        return self._maybe_add_zero_expert_output(result)

    def _decode_kimi_front(self, hidden_states, input_ids):
        main_stream = torch.npu.current_stream()
        shared_input, gather_done = self.ascend_shared_experts.start_input_all_gather(hidden_states)
        self.ascend_shared_experts._pending_input_gather_evt = None
        input_ready = main_stream.record_event()
        hidden_states.record_stream(router_stream())
        prepare = _EXTRA_CTX.moe_comm_method.prepare_finalize
        pad_size = max(0, _EXTRA_CTX.padded_num_tokens // prepare.tp_size - hidden_states.shape[0])
        with npu_stream_switch(router_stream()):
            torch.npu.current_stream().wait_event(input_ready)
            with torch_npu.npu.npugraph_ex.scope.limit_core_num(*ROUTER_CORE_LIMIT):
                logits, _ = self._kimi_gate(hidden_states)
            gate_done = torch.npu.current_stream().record_event()
            # This exact router reads only shape/dtype from hidden_states.
            # Keep the *projected and padded* metadata shape to preserve its
            # fused-router eligibility without waiting for the projection.
            routing_metadata = hidden_states.new_empty((hidden_states.shape[0] + pad_size, self.moe_config.hidden_dim))
            routing = self.routed_experts._select_decode_experts(
                routing_metadata, logits, pad_size, input_ids, _EXTRA_CTX.in_profile_run
            )
        scheme = self.routed_input_transform.quant_method.quant_method
        quantized = scheme.quantize_input(hidden_states)
        main_stream.wait_event(gate_done)
        routed_input, _ = self.routed_input_transform(quantized)
        # The routed entry joins TopK before dispatch. Down and TopK can overlap.
        return routed_input, logits, routing, shared_input, gather_done

    @property
    def is_internal_router(self) -> bool:
        if vllm_version_is("0.27.1"):
            gate = self.gate
            return gate is not None and hasattr(gate, "weight_fp32")
        else:
            # main (cdc4824a21): vllm#51838 removed the gate branch in
            # DeepseekV2MoE.forward, always passing router_logits=hidden_states.
            # The runner must recompute router_logits via the gate.
            return self.gate is not None

    @property
    def use_dp_chunking(self) -> bool:
        """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
        chunked path. Always return False to stay on forward_impl."""
        return False

    def _can_overlap_sp_shared_with(self, routed_transform: object | None) -> bool:
        """Limit SP shared-expert overlap changes to routed transforms.

        Kimi-K3 is currently the only Ascend model that supplies these latent
        transforms.  Keep the guard capability-based so every model without
        them stays on the original single-stream synchronization path.
        """
        shared_experts = getattr(self, "ascend_shared_experts", None)
        return (
            routed_transform is not None
            and shared_experts is not None
            and shared_experts.multistream_overlap
            and shared_experts.parallel_mode() is SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
        )

    def _should_defer_shared_output_wait(self) -> bool:
        return self._can_overlap_sp_shared_with(self.routed_output_transform) and self._defer_sp_shared_output_wait

    @property
    def _fused_output_is_reduced(self) -> bool:
        # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
        # TP all-reduce for the routed output, and AscendSharedExperts.forward
        # handles it for the shared output. Signal this to the upstream
        # MoERunner.forward() so _maybe_reduce_final_output does not apply a
        # second TP all-reduce (which would double-count the contributions).
        moe_comm_type = _EXTRA_CTX.moe_comm_type
        return moe_comm_type in {
            MoECommType.ALLTOALL,
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        } or (moe_comm_type == MoECommType.ALLGATHER and self.moe_config.is_sequence_parallel)

    def _get_shared_expert_parallel_mode(self) -> SharedExpertParallelMode:
        shared_experts = getattr(self, "ascend_shared_experts", None)
        if shared_experts is None or not hasattr(shared_experts, "parallel_mode"):
            return SharedExpertParallelMode.TENSOR_PARALLEL
        return shared_experts.parallel_mode()

    def _reduce_shared_output_if_needed(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool,
    ) -> torch.Tensor | None:
        if (
            shared_output is not None
            and fused_output_is_reduced
            and self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.TENSOR_PARALLEL
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    @property
    def local_num_experts(self) -> int:
        """Number of physical experts managed by this EPLB layer."""
        return self.moe_config.num_local_experts

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

    # Shared-expert layout-specific communication is handled by
    # AscendSharedExperts, so only standard TP weights need a separate
    # all-reduce when routed output has already been reduced.
    def _maybe_reduce_shared_expert_output(  # type: ignore[misc]
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced
        return self._reduce_shared_output_if_needed(
            shared_output,
            fused_output_is_reduced,
        )

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        fused_output, fused_output_is_reduced = super()._maybe_reduce_routed_output_before_transform(
            fused_output,
            fused_output_is_reduced,
        )

        if (
            self._get_shared_expert_parallel_mode() is SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY
            and not fused_output_is_reduced
        ):
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    # Ascend already handles reduction in its own dispatch path, so
    # the upstream kwarg is accepted for interface alignment only.
    def _maybe_reduce_final_output(  # type: ignore[misc]
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced
        if not output_is_reduced and not self.moe_config.is_sequence_parallel:
            # Use the normal TP collective when the upstream reduction
            # contract requires it. Sequence-parallel outputs are token
            # shards, so reducing them position-wise would corrupt the result.
            states = tensor_model_parallel_all_reduce(states)
        if trunc_size is not None and trunc_size > 0:
            return states[..., :trunc_size]
        return states

    def set_lora_context(self, lora_context):
        self.routed_experts._ascend_moe_lora_context = lora_context
        if self.ascend_shared_experts is not None:
            self.ascend_shared_experts.set_lora_context(lora_context)

    def _forward_with_fused_shared_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Fuse a compatible shared expert without changing latent-MoE semantics."""
        if self.ascend_shared_experts is None:
            return None
        # MegaMoE exposes one x for both routed and shared experts. Kimi K3's
        # latent routed input differs from its full-width shared input, so it
        # must retain the standalone shared path.
        if shared_hidden_states is not hidden_states:
            return None
        get_shared_weights = getattr(self.ascend_shared_experts, "get_mega_moe_weights", None)
        if get_shared_weights is None:
            return None
        if _EXTRA_CTX.moe_comm_type != MoECommType.FUSED_MC2:
            return None
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        supports_fused_shared = getattr(moe_comm_method, "supports_fused_shared_experts", None)
        if supports_fused_shared is None or not supports_fused_shared():
            return None
        shared_weights = get_shared_weights()
        if shared_weights is None:
            return None

        if self.is_internal_router:
            gate = self.gate
            assert gate is not None
            gate_weight = gate.weight_fp32 if hasattr(gate, "weight_fp32") else gate.weight.to(torch.float32)
            router_logits = F.linear(shared_hidden_states.float(), gate_weight)

        combined_out, fused_moe_events = self.routed_experts.forward_impl(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
            shared_experts=shared_weights,
            shared_experts_input=shared_hidden_states,
        )
        if not fused_moe_events.shared_experts_fused:
            raise RuntimeError("MegaMoE returned without fusing the provided shared expert weights.")
        # vLLM expects (shared, routed). MegaMoE has already summed both, so
        # return the combined value in the shared slot and a zero routed slot.
        return combined_out, torch.zeros_like(combined_out)

    def _apply_kimi_front(self, hidden_states: torch.Tensor, routed_input: torch.Tensor | None = None):
        """Gate GEMM + shared-input gather, inside the MoE op.

        Keep stream orchestration inside the opaque MoE op. The normal runner
        path projects routed inputs before upstream padding; direct callers
        without a projected input still run the projection here.

        ``hidden_states`` must be the pre-transform activations the gate was
        built for. When upstream already ran the latent down projection before
        the entry call, the transformed result is passed as ``routed_input``
        so the projection is not applied twice.
        """
        shared_experts = self.ascend_shared_experts
        gather_input = shared_experts is not None and self._can_overlap_sp_shared_with(self.routed_input_transform)
        gathered = hidden_states
        if gather_input:
            gathered, _ = shared_experts.start_input_all_gather(hidden_states)

        gate = getattr(self, "_kimi_gate", None)
        fork_gate = (
            getattr(self, "_kimi_gate_multistream", False)
            and getattr(_EXTRA_CTX, "moe_comm_type", None) == MoECommType.MC2
            and self.moe_config.is_sequence_parallel
        )
        gate_done = None
        if fork_gate:
            gate_ready = torch.npu.current_stream().record_event()
            hidden_states.record_stream(router_stream())
            with npu_stream_switch(router_stream()):
                torch.npu.current_stream().wait_event(gate_ready)
                router_logits, _ = gate(hidden_states)
                gate_done = torch.npu.current_stream().record_event()
            # Join the gate before handing its output to expert routing (or
            # running the fallback projection for a direct caller).
            torch.npu.current_stream().wait_event(gate_done)
        else:
            router_logits, _ = gate(hidden_states)

        if routed_input is None:
            routed_input, _ = super().apply_routed_input_transform(hidden_states)
        return routed_input, router_logits, gate_done, gathered

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project before upstream padding, preserving full-width shared input."""
        if getattr(self, "_kimi_gate", None) is not None:
            # Upstream pads to the expert width before entering the MoE op.
            # Project first: negative padding would otherwise crop the input.
            routed_input, _ = super().apply_routed_input_transform(hidden_states)
            return routed_input, hidden_states
        gathered_input = hidden_states
        all_gather_done = None
        shared_experts = self.ascend_shared_experts
        if self._can_overlap_sp_shared_with(self.routed_input_transform):
            assert shared_experts is not None
            gathered_input, all_gather_done = shared_experts.start_input_all_gather(hidden_states)
        routed_input, shared_input = super().apply_routed_input_transform(hidden_states)
        if all_gather_done is not None:
            # The shared compute stream joins this gather before consuming the
            # result; waiting here would stall the routed input transform.
            shared_input = gathered_input
        return routed_input, shared_input

    def apply_routed_output_transform(self, fused_output: torch.Tensor) -> torch.Tensor:
        """Run the latent up projection before joining the SP shared output."""
        fused_output = super().apply_routed_output_transform(fused_output)
        shared_experts = self.ascend_shared_experts
        if self._should_defer_shared_output_wait():
            assert shared_experts is not None
            shared_experts.wait_for_output()
        return fused_output

    if vllm_version_is("0.27.1"):

        def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # The Kimi K3 front stays outside the SP context, matching the
            # previous placement in apply_routed_input_transform.
            kimi_gate = getattr(self, "_kimi_gate", None)
            router_state = None
            if kimi_gate is not None:
                # Upstream may apply the latent down projection before the entry
                # call; the gate always consumes the pre-transform activations,
                # which upstream then passes as shared_experts_input.
                # Explicit original input marks the upstream transform contract;
                # equal widths (including alignment padding) do not imply that
                # the projection still needs to run.
                front_input = shared_experts_input if shared_experts_input is not None else hidden_states
                routed_input = hidden_states if shared_experts_input is not None else None
                hidden_states, router_logits, gate_done, shared_experts_input = self._apply_kimi_front(
                    front_input, routed_input
                )
                if gate_done is not None:
                    router_state = (router_logits, gate_done)
            with self._sequence_parallel_context():
                shared_hidden_states = shared_experts_input if shared_experts_input is not None else hidden_states
                if self.ascend_shared_experts is None:
                    return self.routed_experts.forward_impl(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        input_ids=input_ids,
                        router_state=router_state,
                    )
                fused_shared_out = self._forward_with_fused_shared_experts(
                    hidden_states,
                    router_logits,
                    shared_hidden_states,
                    input_ids,
                )
                if fused_shared_out is not None:
                    return fused_shared_out
                # The runner input transform provides a padded gathered tensor
                # in SP multistream mode. Trim the shared-MLP view before use.
                shared_input_is_gathered = self._can_overlap_sp_shared_with(self.routed_input_transform)
                defer_shared_output_wait = self._should_defer_shared_output_wait()
                shared_expert_input = (
                    shared_hidden_states[: _EXTRA_CTX.num_tokens] if shared_input_is_gathered else shared_hidden_states
                )
                if self.is_internal_router:
                    gate = self.gate
                    assert gate is not None
                    # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
                    # increase with extra hidden states. We also assume that all gate
                    # linear is unquantized so that we the weight is pre-casted in
                    # process_weights_after_loading of AscendUnquantizedLinearMethod.
                    hidden_states_fp32 = shared_hidden_states.float()
                    before_routed_experts = torch.npu.current_stream().record_event()
                    # v0.27.1: weight_fp32 is guaranteed by is_internal_router.
                    router_logits = F.linear(hidden_states_fp32, gate.weight_fp32)
                    after_routed_experts = torch.npu.current_stream().record_event()
                else:
                    before_routed_experts = torch.npu.current_stream().record_event()
                    after_routed_experts = None

                routed_out, fused_moe_events = self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                    router_state=router_state,
                )
                fused_moe_events.before_routed_experts = before_routed_experts
                fused_moe_events.after_routed_experts = after_routed_experts
                if shared_input_is_gathered:
                    fused_moe_events.after_routed_finalize = torch.npu.current_stream().record_event()

                shared_out = self.ascend_shared_experts.forward(
                    shared_expert_input,
                    fused_moe_events,
                    input_is_gathered=shared_input_is_gathered,
                    defer_output_wait=defer_shared_output_wait,
                )
                return shared_out, routed_out

    else:

        def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # The Kimi K3 front stays outside the SP context, matching the
            # previous placement in apply_routed_input_transform.
            kimi_gate = getattr(self, "_kimi_gate", None)
            router_state = None
            if kimi_gate is not None:
                # Upstream may apply the latent down projection before the entry
                # call; the gate always consumes the pre-transform activations,
                # which upstream then passes as shared_experts_input.
                # Explicit original input marks the upstream transform contract;
                # equal widths (including alignment padding) do not imply that
                # the projection still needs to run.
                front_input = shared_experts_input if shared_experts_input is not None else hidden_states
                routed_input = hidden_states if shared_experts_input is not None else None
                hidden_states, router_logits, gate_done, shared_experts_input = self._apply_kimi_front(
                    front_input, routed_input
                )
                if gate_done is not None:
                    router_state = (router_logits, gate_done)
            with self._sequence_parallel_context():
                shared_hidden_states = shared_experts_input if shared_experts_input is not None else hidden_states
                if self.ascend_shared_experts is None:
                    if self.is_internal_router:
                        gate = self.gate
                        assert gate is not None
                        hidden_states_fp32 = hidden_states.float()
                        router_logits = F.linear(
                            hidden_states_fp32,
                            gate.weight_fp32 if hasattr(gate, "weight_fp32") else gate.weight.to(torch.float32),
                        )
                    return self.routed_experts.forward_impl(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        input_ids=input_ids,
                        router_state=router_state,
                    )
                fused_shared_out = self._forward_with_fused_shared_experts(
                    hidden_states,
                    router_logits,
                    shared_hidden_states,
                    input_ids,
                )
                if fused_shared_out is not None:
                    return fused_shared_out
                # See the v0.27.1 branch above for the input-shape contract.
                shared_input_is_gathered = self._can_overlap_sp_shared_with(self.routed_input_transform)
                defer_shared_output_wait = self._should_defer_shared_output_wait()
                shared_expert_input = (
                    shared_hidden_states[: _EXTRA_CTX.num_tokens] if shared_input_is_gathered else shared_hidden_states
                )
                if self.is_internal_router:
                    gate = self.gate
                    assert gate is not None
                    # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
                    # increase with extra hidden states. We also assume that all gate
                    # linear is unquantized so that we the weight is pre-casted in
                    # process_weights_after_loading of AscendUnquantizedLinearMethod.
                    hidden_states_fp32 = shared_hidden_states.float()
                    before_routed_experts = torch.npu.current_stream().record_event()
                    # main (cdc4824a21): is_internal_router only checks self.gate,
                    # weight_fp32 may be absent, fall back to gate.weight.
                    router_logits = F.linear(
                        hidden_states_fp32,
                        gate.weight_fp32 if hasattr(gate, "weight_fp32") else gate.weight.to(torch.float32),
                    )
                    after_routed_experts = torch.npu.current_stream().record_event()
                else:
                    before_routed_experts = torch.npu.current_stream().record_event()
                    after_routed_experts = None

                routed_out, fused_moe_events = self.routed_experts.forward_impl(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                    router_state=router_state,
                )
                fused_moe_events.before_routed_experts = before_routed_experts
                fused_moe_events.after_routed_experts = after_routed_experts
                if shared_input_is_gathered:
                    fused_moe_events.after_routed_finalize = torch.npu.current_stream().record_event()

                shared_out = self.ascend_shared_experts.forward(
                    shared_expert_input,
                    fused_moe_events,
                    input_is_gathered=shared_input_is_gathered,
                    defer_output_wait=defer_shared_output_wait,
                )
                return shared_out, routed_out
