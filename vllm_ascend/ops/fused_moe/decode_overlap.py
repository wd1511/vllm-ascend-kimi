# SPDX-License-Identifier: Apache-2.0
"""Per-call shared-expert stages for split MC2 decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu

from vllm_ascend.utils import npu_stream_switch, shared_experts_calculation_stream, shared_experts_comm_stream

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts

ROUTER_CORE_LIMIT = (32, 1)


@dataclass
class DecodeSharedExpertSchedule:
    # Owned by one MoE invocation; no callbacks or activations survive replay.
    shared_experts: AscendSharedExperts
    hidden_states: torch.Tensor
    gather_done: torch.npu.Event
    gate_up_core_limit: tuple[int, int]
    quantized_shared: bool = False
    activated: torch.Tensor | None = None
    output: torch.Tensor | None = None
    output_done: torch.npu.Event | None = None

    def after_dispatch(self, before_dispatch: torch.npu.Event, after_dispatch: torch.npu.Event) -> None:
        main_stream = torch.npu.current_stream()
        shared_stream = shared_experts_calculation_stream()
        self.hidden_states.record_stream(shared_stream)
        with npu_stream_switch(shared_stream):
            shared_stream.wait_event(self.gather_done)
            gate_up_input = self.hidden_states
            if self.quantized_shared:
                # Match the existing MXFP4 path: quantization is outside the
                # gate/up core limit, and the linear consumes the quantized pair.
                gate_up_input = torch_npu.npu_dynamic_mx_quant(gate_up_input, dst_type=torch.float8_e4m3fn)
            shared_stream.wait_event(before_dispatch)
            core_scope = (
                self.shared_experts._gate_up_core_scope()
                if self.quantized_shared
                else torch_npu.npu.npugraph_ex.scope.limit_core_num(*self.gate_up_core_limit)
            )
            with core_scope:
                gate_up = self.shared_experts.part1(gate_up_input)
            shared_stream.wait_event(after_dispatch)
            # Do not extend the gate/up core scope to SiTU or routed GMM1.
            if self.quantized_shared:
                self.activated = self.shared_experts._activate_situ(gate_up, force_sparse=True)
            else:
                self.activated = self.shared_experts._activate_situ(gate_up)
            shared_situ_done = shared_stream.record_event()
        # Record the producer event before queueing the consumer wait.
        main_stream.wait_event(shared_situ_done)

    def after_gmm1(self, after_gmm1: torch.npu.Event) -> None:
        if self.activated is None:
            raise RuntimeError("Shared SiTU must be submitted before routed GMM1.")
        shared_stream = shared_experts_calculation_stream()
        comm_stream = shared_experts_comm_stream()
        with npu_stream_switch(shared_stream):
            shared_stream.wait_event(after_gmm1)
            shared_out = self.shared_experts.layer.down_proj(self.activated)[0]
            compute_done = shared_stream.record_event()
            shared_out.record_stream(comm_stream)
        with npu_stream_switch(comm_stream):
            comm_stream.wait_event(compute_done)
            self.output = self.shared_experts._pad_and_reduce_scatter(shared_out)
            self.output_done = comm_stream.record_event()
        self.activated = None

    def join(self) -> torch.Tensor:
        if self.output is None or self.output_done is None:
            raise RuntimeError("The routed MLP did not release the shared down projection.")
        main_stream = torch.npu.current_stream()
        main_stream.wait_event(self.output_done)
        self.output.record_stream(main_stream)
        return self.output
