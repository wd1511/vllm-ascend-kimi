# Kimi K3 A5 MegaMoE

## Scope

This design routes eligible Kimi K3 routed experts through CANN MegaMoE on
Ascend A5. It applies to the vLLM 0.27-compatible release line and W4A8 MXFP
checkpoints. The existing MoE implementation remains the fallback for other
hardware, quantization formats, and unsupported runtime configurations.

MegaMoE replaces only the routed-expert core:

```text
full hidden states
  |-- gate -> top-k ids and weights
  |-- latent down projection -> [tokens, 3584]
       |-- MegaMoE: dispatch -> GMM1 -> SiTU -> GMM2 -> combine
       `-- routed RMSNorm -> latent up projection
  `-- shared experts -----------------------------------------> sum
```

The Kimi-specific gate, latent projections, routed output transform, and
shared experts stay in the existing `AscendKimiMoE` implementation.

## Selection

The `FUSED_MC2`/MegaMoE path is selected only when all of these conditions
hold:

- the device is A5, the process is not a PD decode node, and expert
  parallelism has more than one rank;
- `additional_config.enable_fused_mc2` enables the A5 fused path (internally `1`;
  use `2` in user configuration to enable the operator availability check and shared overlap);
- the instantiated routed-expert quantization is `W4A8MXFP`;
- the activation is SiLU/SwiGLU or SiTU;
- the MXFP group size is 32 when the metadata exposes it;
- dynamic EPLB, redundant experts, mixed placement, and MoE LoRA are disabled;
- the batch fits the process-wide symmetric-buffer capacity;
- every active DP rank is running a pure prefill batch. A batch containing any
  decode request keeps the existing MC2, AllGather, or AllToAll path.

If a condition is not met, communication selection uses the existing MC2,
AllGather, or AllToAll path. Selection is made from the instantiated layer's
quantization type where possible, rather than from a model-name check.

## Prefill / Decode 与 DP 一致性

本次恢复基于包含 PR54、PR59 的 release，保留它们的 KDA 编译、Router 和 norm/quant 修改。
A5 MegaMoE 仅用于 v1 runner 的纯 prefill 批次，按以下规则执行：

| 本轮所有 DP rank 的请求 | MoE 路径 | 图模式 |
| --- | --- | --- |
| 所有活跃 rank 都是纯 prefill，其他 rank 空闲 | 满足权重/容量条件时使用 MegaMoE | NONE |
| 任一活跃 rank 含 decode（包括同 rank 的混合批次） | 全部回退 MC2/AllGather/AllToAll | 含 prefill 时全部 NONE |
| 全部为 decode | 普通 MoE | 取所有 rank 支持模式的最小值 |
| decode rank 有图，另一个参与 rank 无图 | 普通 MoE | 全部 NONE |
| ACLGraph capture/replay | 普通 MoE，禁止 MegaMoE | 按捕获配置 |

请求阶段按 CPU 上 `num_computed_tokens >= num_prompt_tokens` 判断，不能按 token 数判断：
短 prefill 和 chunked prefill 尾块也可能只有一个 token。只要本地包含 prefill，先通报
NONE，再通过主线已有的 CPU DP group 同步，避免某些 rank 回放 decode 图而其他 rank
执行 prefill。活跃 rank 的 decode 标志做 OR；空闲 dummy 的合成 decode 标志不参与
这个 OR，但其图模式仍必须参与最小值归约。所有 rank 都是 dummy 时保留 dummy 标志，
使 decode 图捕获走普通 MoE。

DP 通信沿用最新主线的 CPU group，没有恢复旧的 NPU all-reduce。仅 A5 fused 开启时
扩展为四个字段（token 数、图模式、真实工作标志、decode 标志）；关闭时保留两个字段。
普通内存 profile 明确按 prefill 预热 MegaMoE 缓冲区，图捕获 dummy 则明确禁止 MegaMoE。
未传入同步阶段的 forward 调用保守回退；MRv2 尚未接入此 DP 阶段协议，暂不启用 A5 MegaMoE。

可在只有 CPU PyTorch 的环境执行调度回归：

```bash
python tests/ut/worker/test_megamoe_prefill_policy.py -v
```

该测试模拟 DP collective 并检查每个 rank 实际提交的元数据、最终图模式和通信后端，
不能替代多卡运行。A5 验收需覆盖上表、连续 prefill→decode→prefill 切换及 SP 开关；
确认 decode profiling 没有 MegaMoE、prefill rank 与 dummy rank 均能完成，并比较真实权重精度。

## Kimi K3 Operator Contract

Kimi K3 routes in latent space. For the released shape, the MegaMoE arguments
are:

| Argument | Value |
| --- | --- |
| `num_experts` | 896 |
| `num_topk` | 16 |
| `hidden` | `routed_expert_hidden_size`, 3584 |
| `intermediate_hidden` | logical SiTU intermediate width, 3072 |
| dispatch mode | MXFP, 4 |
| activation dtype | FP8 E4M3 |
| weight dtype | packed FP4 E2M1 |

The weight tensors passed by the W4A8 MXFP method use the checkpoint-oriented
stacked layout expected by MegaMoE:

```text
w1       [local_experts, 2 * intermediate, hidden / 2]
w2       [local_experts, hidden, intermediate / 2]
w1_scale [local_experts, 2 * intermediate, hidden / 64, 2]
w2_scale [local_experts, hidden, intermediate / 64, 2]
```

For Kimi K3, W1's concatenated gate/up projection width is 6144, while the
`intermediate_hidden` argument passed to the CANN symmetric-buffer API is the
logical width 3072. Passing 6144 violates the Ascend 950 API range.

The existing grouped-matmul representation is restored with transposed views;
no expert-weight copy is introduced in the forward path. Unsigned-byte MXFP
scale storage is reinterpreted as E8M0 without a copy.

CANN 9.3 requires packed FP4 weights to cross the MegaMoE Python boundary as
`uint8`, with `weight1_type` and `weight2_type` carrying the logical FP4 type.
Kimi's native FP4 W1 metadata is therefore exposed through a zero-copy
`uint8` dtype view only for the MegaMoE call.

Kimi K3 uses SiTU, which is not equivalent to SwiGLU. The backend maps the
vLLM activation configuration to this CANN call contract:

```python
mega_moe(
    ...,
    activation="situglu",
    activation_params={
        "beta": activation_situ_beta,
        "linear_beta": activation_situ_linear_beta,
    },
)
```

For the released Kimi K3 configuration, `linear_beta` is 25 and `beta` is 4.
`activation_clamp` is not used for SiTU. If the installed ops-transformer
package does not expose `activation_params`, execution fails explicitly
instead of silently substituting SwiGLU.

## Buffer And Collective Rules

MegaMoE is a collective operation. Every rank must choose the same backend,
use the same expert configuration, and acquire a compatible symmetric buffer.
The buffer is cached on the dedicated MegaMoE process group and shared by all
compatible MoE layers in the process.

The per-rank capacity is:

```text
min(mega_moe_max_tokens / expert_parallel_size,
    ceil(execution_tokens_per_dp_rank / sequence_parallel_size))
```

The expert-parallel size comes from the actual EP process group and therefore
does not include pipeline-parallel stages. The execution limit comes from the
scheduler budget in standalone and prefill deployments. For sequence-parallel
models, including Kimi K3 with TP greater than one, both backend selection and
DP padding use the rank-local token count. Non-sequence-parallel models such as
a DeepSeek V4 configuration with SP disabled retain the full per-DP-rank token
count. Decode-only nodes never select MegaMoE and retain the existing
MC2/AllToAll path.

Compatible shared experts can be passed to the same MegaMoE invocation when
the installed operator exposes the shared-weight arguments, routed and shared
experts consume the same activation tensor, and the shared weights are
replicated (`enable_shared_expert_dp=true` when TP is greater than one).
DeepSeek V4 can satisfy this contract. Kimi K3 latent MoE cannot:
its routed experts consume the 3584-wide latent projection while its shared
expert consumes the full-width model hidden state. Since MegaMoE exposes only
one input tensor, Kimi keeps the standalone shared-expert path to preserve
shape and numerical semantics.

The persistent symmetric buffer is materialized during memory profiling, before
KV cache allocation. A request larger than the buffer capacity falls back
during communication selection; a direct contract violation raises an error
before entering CANN.

For example, with TP 8, DP 4, EP 32, `mega_moe_max_tokens=65536`, and
`max_num_batched_tokens=8192`, Kimi K3 uses an SP size of 8 and allocates for
`min(65536 / 32, 8192 / 8) = 1024` tokens per rank. That buffer supports an
8192-token scheduler batch. A non-SP DeepSeek V4 configuration keeps an SP size
of 1 and is limited to 2048 scheduler tokens on the same topology; larger
batches use the existing fallback communication path.

## Configuration

Enable the path with the regular additional configuration interface:

```shell
vllm serve <KIMI_K3_W4A8_MXFP_PATH> \
  --enable-expert-parallel \
  --additional-config '{"enable_fused_mc2":2,"mega_moe_max_tokens":65536}'
```

`enable_fused_mc2=2` selects MegaMoE when the operator is available; `1`
continues to select the A3 `dispatch_ffn_combine` path. MegaMoE no longer
automatically disables `multistream_overlap_shared_expert`, while the A3 path
still does. Validate multistream overlap on the target multi-rank topology
before enabling both switches in production.

All ranks must use the same configuration and a CANN/ops-transformer build that
exports `mega_moe`, `get_symm_buffer_for_mega_moe`, and the SiTU
`activation_params` argument.

## Validation

CPU unit tests cover communication selection, the SiTU argument mapping,
SwiGLU compatibility, the packed weight/scale layout, and the K3 buffer width.
Release qualification on A5 must additionally cover:

1. a two-rank operator smoke test comparing MegaMoE with the decomposed path;
2. eager prefill and decode with uneven per-rank token counts, confirming that
   decode contains no MegaMoE invocation;
3. a full Kimi K3 service request on the intended multi-node EP topology;
4. accuracy comparison with `enable_fused_mc2=0`;
5. profiler confirmation that routed layers execute MegaMoE and do not execute
   separate dispatch, SiTU, GMM2, and combine operators;
6. peak HBM and throughput measurements for the selected
   `mega_moe_max_tokens` value.
