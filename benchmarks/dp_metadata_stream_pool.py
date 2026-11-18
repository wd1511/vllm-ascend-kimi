# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU DP exchange correctness and timing, run with torchrun.

This diagnostic uses its own HCCL group with expansion mode 5. It does not
replace measurement in the model's actual DP group or full-model validation.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument(
        "--backlog", type=int, default=4, help="independent compute-stream matmuls before each exchange"
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.matrix_size < 1 or args.backlog < 0:
        parser.error("iterations/matrix-size must be positive; warmup/backlog must be nonnegative")
    return args


def main():
    args = _arguments()
    import torch
    import torch.distributed as dist
    import torch_npu

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"group_name": "dp_metadata_c_benchmark", "hccl_op_expansion_mode": 5}
    dist.init_process_group("hccl", pg_options=options)
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size < 2:
            raise ValueError("Use torchrun with at least two ranks")
        path = Path(__file__).resolve().parents[1] / "vllm_ascend/worker/metadata_stream_pool.py"
        spec = importlib.util.spec_from_file_location("_c_metadata_stream_pool", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        pool = module.DpMetadataStreamPool(device, True, size, pool_size=2, urgent_priority=-1)
        left = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
        right = torch.randn_like(left)
        output = torch.empty_like(left)
        torch.npu.synchronize()

        def legacy_exchange(tokens):
            packed = torch.zeros(2, size, device=device, dtype=torch.int32)
            packed[0, rank] = tokens
            packed[1, rank] = 2
            dist.all_reduce(packed)
            return packed.cpu()

        def side_exchange(tokens):
            return pool.exchange(tokens, 2, rank, dist.group.WORLD, urgent=True)

        for label, exchange in (("compute", legacy_exchange), ("C_pool", side_exchange)):
            host_ms, step_ms = [], []
            for index in range(args.warmup + args.iterations):
                # Align this microbenchmark's ranks outside the timed region.
                dist.barrier()
                torch.npu.synchronize()
                step_start = time.perf_counter()
                for _ in range(args.backlog):
                    torch.mm(left, right, out=output)
                host_start = time.perf_counter()
                packed = exchange(index + rank + 1)
                host_end = time.perf_counter()
                # Account for remaining independent compute separately from
                # the metadata Host wait; a shorter wait alone is not speedup.
                torch.npu.synchronize()
                step_end = time.perf_counter()
                expected = torch.tensor(
                    [[index + peer + 1 for peer in range(size)], [2] * size], dtype=torch.int32
                )
                torch.testing.assert_close(packed, expected)
                if index >= args.warmup:
                    host_ms.append((host_end - host_start) * 1000)
                    step_ms.append((step_end - step_start) * 1000)
            print(json.dumps({
                "rank": rank,
                "path": label,
                "iterations": args.iterations,
                "backlog": args.backlog,
                "metadata_host_ms_median": statistics.median(host_ms),
                "metadata_host_ms_max": max(host_ms),
                "whole_step_ms_median": statistics.median(step_ms),
                "correctness": "passed",
            }), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
