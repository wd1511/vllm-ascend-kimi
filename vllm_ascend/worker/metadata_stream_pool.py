# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded NPU lanes for ordered, CPU-visible DP metadata exchanges.

This pool owns its streams and staging buffers. It does not schedule model
kernels, create CPU threads, cache collective results, or omit collectives.
Each exchange completes before returning; priority lanes can isolate target
and draft submissions, but do not make consecutive exchanges overlap.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


MAX_METADATA_STREAMS = 8


@dataclass
class _MetadataSlot:
    stream: Any
    event: Any
    priority: int
    host: torch.Tensor
    device: torch.Tensor
    pending: bool = False
    sequence: int = 0


class DpMetadataStreamPool:
    """Submit collectives serially from one worker on ready NPU lanes.

    Target metadata requests ``urgent_priority``; draft metadata requests the
    normal priority (0). Slots are created lazily, up to ``pool_size``. With a
    single slot or a full pool, an existing ready slot can serve either role;
    its NPU priority is immutable. Event.query is an optional non-spinning
    completion fast path, never permission to consume incomplete D2H data.
    """

    def __init__(
        self,
        device: torch.device,
        pin_memory: bool,
        dp_size: int,
        pool_size: int = 2,
        urgent_priority: int = -1,
        enable_event_query_fast_path: bool = True,
    ) -> None:
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or not 1 <= pool_size <= MAX_METADATA_STREAMS:
            raise ValueError(f"pool_size must be an integer in [1, {MAX_METADATA_STREAMS}]")
        if isinstance(dp_size, bool) or not isinstance(dp_size, int) or dp_size <= 0:
            raise ValueError("dp_size must be a positive integer")
        if isinstance(urgent_priority, bool) or not isinstance(urgent_priority, int) or urgent_priority > 0:
            raise ValueError("urgent_priority must be an integer <= 0")
        self.device = device
        self.pin_memory = pin_memory
        self.dp_size = dp_size
        self.pool_size = pool_size
        self.urgent_priority = urgent_priority
        self.enable_event_query_fast_path = enable_event_query_fast_path
        self._slots: list[_MetadataSlot] = []
        self._owner_thread: int | None = None
        self._call_lock = threading.Lock()
        self._failed = False
        self._sequence = 0
        self._stats = {"exchanges": 0, "streams_created": 0, "event_query_hits": 0, "event_waits": 0}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _new_slot(self, priority: int) -> _MetadataSlot:
        stream = torch.npu.Stream(device=self.device, priority=priority)
        event = torch.npu.Event()
        host = torch.empty(2, self.dp_size, dtype=torch.int32, pin_memory=self.pin_memory)
        with torch.npu.stream(stream):
            # Allocate where the storage will be used. H2D overwrites every
            # element; no initialization on the compute stream is required.
            device = torch.empty(2, self.dp_size, dtype=torch.int32, device=self.device)
        slot = _MetadataSlot(stream, event, priority, host, device)
        self._slots.append(slot)
        self._stats["streams_created"] += 1
        return slot

    def _ready(self, slot: _MetadataSlot) -> bool:
        if not slot.pending:
            return True
        if self.enable_event_query_fast_path and slot.event.query():
            slot.pending = False
            self._stats["event_query_hits"] += 1
            return True
        return False

    def _wait(self, slot: _MetadataSlot) -> None:
        if not self._ready(slot):
            slot.event.synchronize()
            slot.pending = False
            self._stats["event_waits"] += 1

    def _acquire(self, priority: int) -> _MetadataSlot:
        preferred = [slot for slot in self._slots if slot.priority == priority]
        for slot in preferred:
            if self._ready(slot):
                return slot
        if len(self._slots) < self.pool_size:
            return self._new_slot(priority)
        for slot in self._slots:
            if self._ready(slot):
                return slot
        # The bound is hard: do not allocate an unbounded number of streams
        # or overwrite an in-flight staging buffer when all slots are busy.
        oldest = min(self._slots, key=lambda slot: slot.sequence)
        self._wait(oldest)
        return oldest

    def exchange(
        self, num_tokens: int, mode_value: int, dp_rank: int, group: Any, urgent: bool = True
    ) -> torch.Tensor:
        """Return a private CPU snapshot after one ordered SUM all-reduce."""
        if not 0 <= dp_rank < self.dp_size:
            raise ValueError("dp_rank is outside the configured DP group")
        if not 0 <= num_tokens <= torch.iinfo(torch.int32).max:
            raise ValueError("num_tokens must fit a non-negative int32")
        if not 0 <= mode_value <= torch.iinfo(torch.int32).max:
            raise ValueError("mode_value must fit a non-negative int32")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("DP metadata exchanges cannot overlap or re-enter on the host")
        try:
            caller = threading.get_ident()
            if self._owner_thread is None:
                self._owner_thread = caller
            elif caller != self._owner_thread:
                raise RuntimeError("DP metadata collectives must be submitted by their original worker thread")
            if self._failed:
                raise RuntimeError("This metadata pool failed previously; recreate the worker before reuse")
            try:
                slot = self._acquire(self.urgent_priority if urgent else 0)
                slot.host.zero_()
                slot.host[0, dp_rank] = num_tokens
                slot.host[1, dp_rank] = mode_value
                with torch.npu.stream(slot.stream):
                    slot.device.copy_(slot.host, non_blocking=True)
                    # Every exchange submits exactly once. Priority selection
                    # changes the issuing stream, never collective call order.
                    dist.all_reduce(slot.device, group=group)
                    slot.host.copy_(slot.device, non_blocking=True)
                    slot.event.record()
                    slot.pending = True
                self._sequence += 1
                slot.sequence = self._sequence
                self._wait(slot)
                result = slot.host.clone()
                self._stats["exchanges"] += 1
                return result
            except Exception:
                # A failed enqueue can leave H2D/collective work outstanding.
                # Do not reuse its staging memory or retry a collective with
                # an ambiguous sequence number in this worker.
                self._failed = True
                raise
        finally:
            self._call_lock.release()
