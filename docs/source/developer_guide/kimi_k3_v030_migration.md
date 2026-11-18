# PR12 core limits multistream and SiTU extraction v2

## Source and scope

This overlay is extracted only from
[cann-framework/vllm-ascend PR12](https://github.com/cann-framework/vllm-ascend/pull/12).
The fixed merge base is `0807cef7025695bfdec9267377b4449d470ba1e6` and the
reviewed PR head is `147d7d105b777e2cff5c79bd48fabe748e703701`.
Apply it to that base. Matching only the `v0.30.0` version string does not
establish source compatibility.

The overlay retains Kimi Router/shared communication streams, staged MC2
Decode dependencies, projection core limits, V1/V2 eligibility checks and
the PR12 sparse/grouped SiTU substitutions.
It also retains Ascend sequence-parallel imports in the Kimi model and
O-projection operator so unaligned runtime token counts preserve padding.
Text compilation, named SuperKernel scopes, latent KN/NZ weight-layout
changes and MegaMoE phase switching remain outside this overlay. SiTU
operator selection follows the reviewed PR; the weight-loading, packing and
quantization input contracts retain the fixed base layouts.

## Configuration and eligibility

Use ordinary split MC2. Merge these fields into the existing additional
configuration and restart all workers:

```json
{
  "multistream_overlap_shared_expert": true,
  "enable_fused_mc2": 0,
  "enable_shared_expert_dp": false,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_super_kernel": false
  }
}
```

`enable_fused_mc2=0` disables the monolithic fused backend; it does not force
MC2 selection for every batch. Confirm the actual backend is MC2.
The base's fused/shared-overlap mutual exclusion remains intact. This overlay
does not add PR12's Prefill MegaMoE/Decode MC2 phase policy.

The strict Decode schedule requires A5 capability, explicit Decode-only
attention metadata, MC2, SP, BF16 activations, MXFP8 latent projection,
W4A8MXFP routed experts and compatible BF16 or W4A8MXFP shared projections.
It excludes draft execution, dynamic EPLB, LoRA, custom router variants and a
shared output gate. V1 and V2 use the same checks. Prefill and unsupported
batches retain the ordinary path and do not acquire Decode core limits.
For SP, preserve the deployment's EP/TP settings; TP must exceed one.

## Core policy

| Projection | AIC | AIV | Scope |
| --- | --- | --- | --- |
| Router | 32 | 1 | Eligible Decode Router projection only |
| BF16 shared Gate/Up | 32 | 1 | Eligible Decode shared Gate/Up |
| MXFP shared Gate/Up | 16 | 32 | Eligible Decode shared Gate/Up |

The producer compute stream is selected before entering the Gate/Up core
scope. Activation, shared Down and communication remain outside that scope.
These limits are constants from PR12; no new configuration key is added.

## Dependency sequence

1. Start shared input AllGather on the dedicated shared communication stream.
2. Queue Router Gate and TopK on the Router stream. Main-stream latent input
   quantization can overlap Gate; latent projection waits for Gate, not TopK.
3. MC2 prepares its native masks, scales and padding. Dispatch waits for TopK.
   At the Dispatch boundary, submit shared Gate/Up on its compute stream.
4. The shared consumer waits for AllGather readiness before input preparation
   and quantization. Decode Dispatch can overlap the AllGather tail; the main
   stream does not join AllGather before Dispatch on this dedicated path.
5. Shared SiTU waits for Dispatch. Routed GMM1 waits for shared SiTU.
6. GMM1 completion releases shared Down, then its ReduceScatter on the
   communication stream. This branch does not wait for routed activation,
   GMM2 or Combine. The scheduled MXFP path splits native GMM1 from activation
   so this completion callback is available.
7. The routed branch completes activation, GMM2, Combine and latent up.
   Join this invocation's shared output event before final Add.

Non-Decode paths retain the main-stream shared-gather join before routed
collectives. Per-call schedule objects own their tensors/events; cleanup
releases Router references. Record cross-stream tensor lifetimes. Explicit
preprojected routed input is preserved without a second latent projection.

## SiTU operators and availability

The original PR12 guide attributes routed sparse/grouped SiTU and MC2 counts
to GitCode PR35, shared sparse SiTU to PR42, and native TopK alias protection
to PR65. This extraction takes their implementation from the fixed PR12 head;
it does not apply patches from those older branches directly.

| Path | Operator | high_precision | Expert-token input |
| --- | --- | --- | --- |
| NPU BF16 shared experts | `custom.npu_situ_and_mul_sparse` | `False` | Nonpersistent int64-max all-rows sentinel |
| Eligible A5 BF16/FP16 unquantized routed experts | `custom.npu_situ_and_mul_sparse` | `True` | Per-expert counts |
| Eligible A5 W4A8MXFP routed experts | `custom.grouped_situ_mx_quant` | `False` | Per-expert counts |

The activation parameters retain `beta` and `alpha=linear_beta`. Shared
activation keeps the native fallback for non-NPU/non-BF16 inputs, missing
linear beta or an unavailable sparse operator. Routed selection retains PR12's
hardware, dtype, positive-parameter, group-list and quantization guards.
The grouped path also checks the group size, expert-weight-list layout and
LoRA constraints. Missing grouped SiTU uses the native activation/quantization
fallback. On the selected A5 path, separating GMM1 from activation remains
necessary even when a custom operator is unavailable; the scheduled callback
must run before routed activation/GMM2.

Eligible MC2 dispatch requests per-expert counts directly, preserving
`group_list_type=1` through GMM1, SiTU and GMM2. Other dispatches keep their
existing layout. This avoids an extra prefix-sum-to-count `diff`/`cat` on the
requested MC2 path. The A5 TopK call uses `torch.ops.npu` to avoid the
`torch_npu` Python alias overwritten by recipes registration.

Each A5 Kimi SiTU model worker calls `initialize_kimi_situ_ops` before tracing.
It tries to load the installed `custom_ops` extension and checks both operators;
missing packages or registrations produce warnings and native fallbacks.
Install the matching recipes `ops/ascendc/torch_ops_extension` package and CANN
operator package in every worker environment. Graph execution also requires
the matching graph converters. This overlay does not bundle these binaries.
Verify registration and actual operator selection; successful inference alone
can still be the native fallback. Custom and native paths may differ numerically.

## Apply and verify

Review `core_multistream_situ.patch`, then either apply that patch to a clean base
or copy the overlay's `vllm_ascend`, `tests` and `docs` files by relative path.
Use only one method. `file_manifest.json` lists every PR12 file and the
selection decision, source hashes and exported hashes.

Run these commands in the complete source tree after applying the overlay.
The focused CPU suites require PyTorch and can run without importing the
complete vLLM/NPU runtime:

```bash
python -m unittest discover -s tests/ut/ops -p 'test_kimi_moe_front_overlap.py'
python -m unittest discover -s tests/ut/ops -p 'test_kimi_v30_moe.py'
python -m unittest discover -s tests/ut/ops -p 'test_kimi_v30_review.py'
python -m unittest discover -s tests/ut/ops -p 'test_kimi_situ_loading.py'
python -m unittest discover -s tests/ut/models -p 'test_kimi_v2_sequence_parallel.py'
```

`test_kimi_situ_loading.py` covers extension registration, missing-package
warnings, Prefill core-limit exclusion and TopK/Dispatch dependencies.
The existing adapter/EPLB/fused-MoE and device test modules require the complete
project runtime for pytest collection. The added NPU sparse-SiTU regression
checks registration before operator selection and records accuracy and latency;
it skips when hardware or operators are unavailable. A skip is not a hardware
validation pass. In the target environment also run:

```bash
pytest -q tests/ut/device/test_device_op.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_shared_sparse_situ.py
```

CPU event/stream mocks and SP tracing do not establish NPU numerical accuracy,
ACLGraph replay, HCCL safety or a speedup. On A5, compare identical weights and
requests with overlap disabled/enabled for both runners, eager and graph
execution, multiple TP/EP sizes, unaligned token counts, Prefill/mixed fallback
and repeated replay. Record full-step latency, throughput, TPOT and P99.
Profiler traces must verify the event order and actual core limits.
