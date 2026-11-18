# SPDX-License-Identifier: Apache-2.0
"""Validate cross-stream buffers/events during eager and ACLGraph replay.

Collectives are represented by copies here. Real TP8/EP HCCL and checkpoint
accuracy must additionally be checked with the Kimi precision workload.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.decode_overlap import DecodeSharedExpertSchedule
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, shared_experts_comm_stream


@pytest.mark.parametrize("num_tokens", [1, 8, 17])
@pytest.mark.parametrize("core_limit", [(32, 1), (16, 32)])
@torch.inference_mode()
def test_shared_schedule_replay_refreshes_buffers(num_tokens, core_limit):
    if get_ascend_device_type() is not AscendDeviceType.A5:
        pytest.skip("Kimi split MC2 decode scheduling targets A5")
    hidden = torch.ones((num_tokens, 128), dtype=torch.bfloat16, device="npu")
    shared = SimpleNamespace(
        part1=lambda x: x * 2,
        _activate_situ=lambda x: x + 3,
        layer=SimpleNamespace(down_proj=lambda x: (x * 5, None)),
        _pad_and_reduce_scatter=lambda x: x.clone(),
    )

    def run():
        main = torch.npu.current_stream()
        ready = main.record_event()
        comm = shared_experts_comm_stream()
        hidden.record_stream(comm)
        with torch.npu.stream(comm):
            comm.wait_event(ready)
            gathered = hidden.clone()
            gather_done = comm.record_event()
        before = main.record_event()
        routed = hidden * 7
        after = main.record_event()
        schedule = DecodeSharedExpertSchedule(shared, gathered, gather_done, core_limit)
        schedule.after_dispatch(before, after)
        routed = routed + 11
        schedule.after_gmm1(main.record_event())
        routed = routed * 13
        return routed + schedule.join()

    for _ in range(3):
        output = run()
    torch.npu.synchronize()
    torch.testing.assert_close(output, (hidden * 7 + 11) * 13 + (hidden * 2 + 3) * 5, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = run()
    for value in (2, 3, 1):
        hidden.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, (hidden * 7 + 11) * 13 + (hidden * 2 + 3) * 5, rtol=0, atol=0)
