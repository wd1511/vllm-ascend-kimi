# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, deterministic expert placement for the existing Ascend EPLB worker.

This policy changes placement, not router scores, top-k selection, or token
admission. Logical-expert replica counts and each rank's allocated capacity
remain fixed. Existing collection, map publication, and weight transfer code
consume the returned deployment through PolicyFactory policy type 4.
"""

import math
from numbers import Integral

import numpy as np

from .policy_abstract import EplbPolicy


class MigrationAwareEplb(EplbPolicy):
    """Balance predicted rank load while charging for incoming expert weights.

    Logical heat is summed across physical replicas and divided by the fixed
    replica count, matching EplbWorker._compute_imbalance's load model. A bounded
    cross-rank swap search minimizes squared rank load plus a migration penalty.
    It never increases the predicted peak during search and publishes a layer
    only if peak savings exceed both the relative threshold and migration cost.

    ``migration_penalty * mean_rank_load`` is a *load proxy per migrated slot*,
    not measured transfer latency. Defaults are deliberately conservative;
    hardware throughput/transfer benefits require profiling. Negative or
    nonfinite observations indicate incomplete statistics and leave that layer
    unchanged. Replica allocation itself is intentionally unchanged.
    """

    def __init__(
        self,
        min_relative_improvement: float = 0.05,
        migration_penalty: float = 0.01,
        max_swaps_per_layer: int = 8,
        max_migrations_per_layer: int = 16,
    ):
        if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement <= 1:
            raise ValueError("min_relative_improvement must be finite and in [0, 1]")
        if not math.isfinite(migration_penalty) or migration_penalty < 0:
            raise ValueError("migration_penalty must be finite and nonnegative")
        for name, value in (
            ("max_swaps_per_layer", max_swaps_per_layer),
            ("max_migrations_per_layer", max_migrations_per_layer),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.min_relative_improvement = float(min_relative_improvement)
        self.migration_penalty = float(migration_penalty)
        self.max_swaps_per_layer = int(max_swaps_per_layer)
        self.max_migrations_per_layer = int(max_migrations_per_layer)

    @staticmethod
    def _retain_local_slots(original: np.ndarray, proposed: np.ndarray) -> np.ndarray:
        """Keep every expert that remains on a rank in its original slot."""
        aligned = original.copy()
        for rank in range(original.shape[0]):
            retained = np.isin(original[rank], proposed[rank])
            newcomers = proposed[rank][~np.isin(proposed[rank], original[rank])]
            aligned[rank, ~retained] = newcomers
        return aligned

    def _balance_layer(self, original: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float]:
        ranks, slots = original.shape
        if (
            ranks < 2
            or self.max_swaps_per_layer == 0
            or self.max_migrations_per_layer < 2
            or not np.isfinite(observed).all()
            or (observed < 0).any()
        ):
            return original, 0.0
        scale = float(observed.max())
        if scale == 0:
            return original, 0.0

        # Normalize before accumulating to avoid overflow on large windows.
        # Relative gains and the migration proxy are invariant to this scale.
        expert_ids, inverse, replica_counts = np.unique(original, return_inverse=True, return_counts=True)
        compact = inverse.reshape(original.shape).copy()
        logical_heat = np.bincount(
            compact.ravel(), weights=(observed / scale).ravel(), minlength=len(expert_ids)
        )
        unit_heat = logical_heat / replica_counts
        rank_loads = unit_heat[compact].sum(axis=1)
        before_peak = float(rank_loads.max())
        mean_load = float(rank_loads.mean())
        if before_peak == 0:
            return original, 0.0
        tolerance = np.finfo(np.float64).eps * max(1.0, float(np.square(rank_loads).sum())) * 32
        original_membership = np.zeros((ranks, len(expert_ids)), dtype=np.int16)
        original_membership[np.arange(ranks)[:, None], compact] = 1
        membership = original_membership.astype(bool)
        moved = 0

        for _ in range(self.max_swaps_per_layer):
            best_delta = -tolerance
            best_swap = None
            peak = float(rank_loads.max())
            for left in range(ranks - 1):
                left_ids = compact[left]
                left_heat = unit_heat[left_ids]
                for right in range(left + 1, ranks):
                    right_ids = compact[right]
                    # Duplicate replicas on a rank are unsupported by the
                    # existing transfer/map format, so reject them in search.
                    valid = (~membership[right, left_ids])[:, None] & (~membership[left, right_ids])[None, :]
                    delta_load = left_heat[:, None] - unit_heat[right_ids][None, :]
                    left_after = rank_loads[left] - delta_load
                    right_after = rank_loads[right] + delta_load
                    valid &= (left_after <= peak + tolerance) & (right_after <= peak + tolerance)

                    # Count incoming experts relative to the original rank
                    # sets, rather than charging again for every search step.
                    migration_delta = (
                        original_membership[left, left_ids][:, None]
                        - original_membership[left, right_ids][None, :]
                        + original_membership[right, right_ids][None, :]
                        - original_membership[right, left_ids][:, None]
                    )
                    candidate_moved = moved + migration_delta
                    valid &= candidate_moved <= self.max_migrations_per_layer
                    squared_load_delta = (
                        2 * delta_load * (rank_loads[right] - rank_loads[left]) + 2 * np.square(delta_load)
                    )
                    score_delta = (
                        squared_load_delta + self.migration_penalty * mean_load**2 * migration_delta
                    )
                    score_delta = np.where(valid, score_delta, np.inf)
                    flat_index = int(np.argmin(score_delta))
                    candidate_delta = float(score_delta.flat[flat_index])
                    if candidate_delta < best_delta:
                        left_slot, right_slot = divmod(flat_index, slots)
                        best_delta = candidate_delta
                        best_swap = (left, right, left_slot, right_slot, int(candidate_moved.flat[flat_index]))
            if best_swap is None:
                break
            left, right, left_slot, right_slot, moved = best_swap
            left_id = compact[left, left_slot]
            right_id = compact[right, right_slot]
            compact[left, left_slot], compact[right, right_slot] = right_id, left_id
            membership[left, left_id] = membership[right, right_id] = False
            membership[left, right_id] = membership[right, left_id] = True
            rank_loads = unit_heat[compact].sum(axis=1)

        saving = before_peak - float(rank_loads.max())
        migration_cost = self.migration_penalty * mean_load * moved
        net_gain = saving - migration_cost
        if saving < self.min_relative_improvement * before_peak or net_gain <= tolerance:
            return original, 0.0
        proposed = self._retain_local_slots(original, expert_ids[compact])
        return proposed, net_gain / before_peak

    def rebalance_experts(
        self, current_expert_table, expert_workload
    ) -> tuple[int, np.ndarray, list[list[list[int]]]]:
        """Return the existing worker's (changed, layer priority, deployment).

        Both inputs have shape ``[layers, ranks, allocated_slots_per_rank]``.
        CPU torch tensors and NumPy arrays are supported through NumPy's array
        protocol. No input is mutated. Tables must contain nonnegative logical
        expert IDs, with no duplicate ID on an individual rank.
        """
        table = np.asarray(current_expert_table)
        workload = np.asarray(expert_workload, dtype=np.float64)
        if table.ndim != 3 or any(size == 0 for size in table.shape) or workload.shape != table.shape:
            raise ValueError("expert table and workload must have the same nonempty [layers, ranks, slots] shape")
        if (
            not np.issubdtype(table.dtype, np.integer)
            or (table < 0).any()
            or (np.issubdtype(table.dtype, np.unsignedinteger) and table.max() > np.iinfo(np.int64).max)
        ):
            raise ValueError("expert table must contain nonnegative integer logical expert IDs")
        placement = table.astype(np.int64, copy=True)
        for layer in placement:
            for rank in layer:
                if len(np.unique(rank)) != len(rank):
                    raise ValueError("a logical expert may appear at most once on each rank")
        proposed = placement.copy()
        layer_gains = []
        for layer in range(placement.shape[0]):
            proposed[layer], gain = self._balance_layer(placement[layer], workload[layer])
            if gain > 0:
                layer_gains.append((layer, gain))
        priority = np.array(
            [layer for layer, _ in sorted(layer_gains, key=lambda item: (-item[1], item[0]))], dtype=np.int64
        )
        return int(len(priority) > 0), priority, proposed.tolist()
