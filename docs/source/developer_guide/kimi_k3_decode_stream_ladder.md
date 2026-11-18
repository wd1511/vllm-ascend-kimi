# Kimi K3 Decode 多流依赖对齐

## 范围

基于已经合入 PR48 的 `releases/v0.27.1rc`，参考 recipes `4447f2f` 的 split Decode 实现。
本改动针对 A5、MC2、SP、BF16 或 MXFP4 shared experts、MXFP4 routed experts、MXFP8 latent down。
只在明确的纯 Decode metadata 下启用，混合 Prefill、缺失 metadata、LoRA、动态 EPLB、
自定义路由、其他量化组合走原有计算路径。关闭 `multistream_overlap_shared_expert` 后使用原入口。

不修改 Router 的 dtype、TopK 公式及权重 Cast，不将性能优化解释为精度修复。
不改 Prefill ALLTOALLV 的通信算法和事件策略。

## 执行顺序

1. 提前启动 Shared AllGather。
2. Router 流运行 Gate 和 TopK；主流同时提交 latent 输入量化。
3. 主流等待 Gate 完成后执行真实 latent down，与 TopK 重叠。
4. 保留原始输入给 shared；真实投影后才进行专家维度 padding。
5. Dispatch 前后记录事件；shared gate/up 等待 AG 和 Dispatch 前事件，SiTU 等待 Dispatch 完成。
6. 主流等待 Shared SiTU，再执行 Routed GMM1。
7. GMM1 完成后提交 Shared down 和通信流 ReduceScatter，主流继续激活/量化、GMM2、Combine。
8. 主流完成 latent up，再等待 Shared ReduceScatter，最后相加。

每次调用创建独立的 shared 调度状态，通过参数传递至 Dispatch/GMM1 回调。
不使用模块全局状态保存激活或回调，不调用设备 `synchronize()`、`item()`。

SiTU 完成后再启动 Routed GMM1 的依赖不按 shared 权重类型区分。原 PR51 仅允许 BF16
shared 进入上述流程，MXFP4 shared 会回退到 PR48 的 SiTU/GMM1 并行路径；现在两者
使用同一条事件链。MXFP4 shared 输入先进行一次 DynamicMxQuant，再把量化值和 scale
传入 gate/up；SiTU 继续调用原 sparse 算子，down 内部的量化仍在 GMM1 完成后执行。
目前仅放行这两种已对照计算接口的 shared 实现，不代表所有量化格式或 BF16 routed 已获支持。

## 图与输入契约

Kimi 多流使用独立的 `ascend_kimi_moe` 算子入口，完整输入和完整输出均为主干宽度。
运行时在算子内部选择 Decode 编排或原有计算路径，避免 Dynamo 固化 Prefill/Decode 分支。
该边界同时涵盖 padding、输出截断、归约和 latent up；不使用返回临时输出的 eager break。
因此多流开启时编译边界扩大，须验证 eager、ACLGraph、npugraph_ex 和静态 kernel。
回退路径保留原有数学和通信顺序，但不能据此假定编译性能完全不变。

新前端使用与原路径相同的 MXFP8 scale 算法。TopK 提前执行时只传入投影后形状的元数据张量，
仅允许已核对只读取其 shape/dtype 的内置路由，实际专家输入始终来自真实投影。
fake 输出保持输入 shape/dtype；真实上游 padding 和输出截断契约由回归测试覆盖。

## 控核对比

Router 使用 `(32, 1)`。新 Decode 路径的 BF16 shared gate/up 默认也使用 `(32, 1)`。
MXFP4 shared gate/up 复用原有 `_gate_up_core_scope()`（Kimi A5 通常为 `(16, 32)`），
不套用 BF16 参数。输入量化、SiTU 和 down 均不被包进该 gate/up 控核范围。
本次不修改 SiTU、Routed GMM1 的算子参数或核数选择；控核决定资源使用，事件决定执行顺序，
即使已有控核也仍须保留 SiTU 到 GMM1 的等待。

在原启动脚本的 `--additional-config` JSON 中合并以下字段，保留其他配置：

```json
{
  "multistream_overlap_shared_expert": true,
  "kimi_k3_decode_shared_gate_up_core_limit": [32, 1]
}
```

只将该字段改为 `[16, 32]`，重启独立进程后即可对比 PR48 的 shared gate/up 核数配置。
该参数只作用于新 Decode BF16 分支；AIC 范围 1～32，AIV 范围 1～64。
不要同时改变权重、batch、输入、量化或图选项。

## 验证

无需 NPU 的事件依赖检查：

```bash
python3 tests/ut/ops/test_kimi_decode_schedule.py -v
```

完整依赖环境下运行：

```bash
pytest -q tests/ut/ops/test_kimi_decode_mlp.py
pytest -q tests/ut/ops/test_fused_moe.py -k 'kimi'
pytest -q tests/ut/test_ascend_config.py -k 'kimi_decode_shared_core_limit'
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_decode_schedule_npu.py
```

单卡 NPU 测试检查跨流缓冲区和反复 graph replay，通信以拷贝代替，不能替代 TP/EP 实测。
真实环境继续使用现有 `kimi_k3_graph_precision.py`，保留 0.27 的已通过基线，
分别采集新分支的 eager/aclgraph/npugraph_ex/static_kernel，检查 greedy 和固定输入 decode。
同时覆盖多流关闭、非 TP 整倍数 token 数、不同 capture size 和重复 replay。

Profiling 应看到 Shared SiTU 结束后 GMM1 才开始，AG 与前端重叠，Shared down 与 routed
激活/量化重叠，Shared RS 与 GMM2/Combine 重叠。事件只允许重叠，不保证设备实际重叠比例。
BF16、MXFP4 shared 分别检查；同时核对改动前后各算子的 AIC/AIV 核数和量化调用次数。
每组配置预热后固定请求重复采集，分别统计同一层从 latent 输入量化到最终 Add 的耗时、
Dispatch 耗时、Shared SiTU/GMM1 重叠和 Shared RS/GMM2 重叠；比较中位数及波动。

本地没有 PyTorch、pytest、NPU 和模型权重，不能给出真实精度或加速比结论。
原 PR48 的精度通过记录不是本分支的精度验证结果。
