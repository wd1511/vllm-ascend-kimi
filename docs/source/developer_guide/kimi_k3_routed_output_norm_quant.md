# Kimi K3 routed output 的 RMSNorm/MX 量化融合

## Router 计算

本分支同时使用 `AscendKimiGateLinear`，router 权重为 BF16，NPU 上的 BF16 输入
直接参与矩阵乘，输出保持 FP32，供原有 TopK 使用。该路径省去 router 激活的
BF16 → FP32 显式转换；其他使用 `AscendGateLinear` 的模型保留原有 FP32 路径。
非 BF16 的 NPU 输入交给上游 GateLinear，CPU 路径以 FP32 计算模拟接口。

router 权重 dtype 的变化可能影响专家选择，需要用真实权重检查输出精度。
`fuse_norm_quant` 只控制下文的 norm/quant 融合，不撤销 router 的 BF16 优化。

## 接入位置与范围

MoE combine、输出截断及原有归约/缩放完成后，latent up 的输入依次经过
`routed_expert_norm` 和 MXFP8 动态量化。本改动用
`torch.ops.npu.npu_rms_norm_dynamic_mx_quant` 合并这两步，将返回的
`(quantized, scale)` 传给原 `routed_expert_up_proj`，复用量化矩阵乘。

`AscendKimiRoutedOutputTransform` 继承上游 transform，保留相同的 `norm`、
`up_proj` 模块和参数名。显式调用融合算子，使其也能在不透明的
`ascend_kimi_moe` 内执行；该边界中的计算不依赖外层图匹配才能融合。
原 Decode ladder 和原调度路径共用此 transform，latent up 仍在 shared 输出等待之前完成。

融合条件：

- 设备具有 `DYNAMIC_MX_QUANT_FUSION` 能力（当前为 A5），且 TorchNPU 已注册该算子。
- latent up 使用 `AscendW8A8MXFP8DynamicLinearMethod`，group size 为 32，scale algorithm 为 0。
- 输入为非空二维 NPU BF16 张量，宽度是 64 的整数倍；Kimi K3 的 latent 宽度为 3584。
- norm weight 形状与输入宽度一致且连续，没有已加载的 norm bias，也没有局部方差归一化。

combine 输出截去专家 padding 后可能具有不连续的行布局，融合前会调用
`contiguous()`。FP16 等其他 dtype、其他量化方案、空输入及上述条件不成立时，
继续调用上游 transform。算子执行错误会直接暴露，不会捕获后静默回退。

## 开关与对照

沿用已有 `ascend_compilation_config.fuse_norm_quant`，默认开启；不增加环境变量。
在启动脚本原有的 `--additional-config` JSON 中合并：

```json
{
  "ascend_compilation_config": {
    "fuse_norm_quant": false
  }
}
```

修改后重新启动进程。该开关也控制仓库原有的 norm/quant 图融合，服务级 A/B 结果
可能包含其他位置的变化。下述组件基准直接对比两个 transform，仅测量本次接入位置。

## 验证命令

在匹配本分支依赖的环境运行 CPU 单元测试：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/models/test_kimi_routed_output_transform.py \
  tests/ut/models/test_kimi_k3_adapter.py \
  tests/ut/ops/test_gate_linear.py \
  tests/ut/quantization/methods/test_w8a8_mxfp8.py \
  tests/ut/ops/test_fused_moe.py \
  -k 'routed_output or opaque_entry or prequantized or shared_expert_gate_up_core_limit or GateLinear'
```

测试覆盖启用与回退、epsilon、加载 bias 后回退、参数身份、模型接线、跳过重复量化，
以及两条 MoE 调度路径的截断和执行顺序。Router 测试检查 BF16 权重、FP32 logits、
NPU 矩阵乘收到原输入而不额外 Cast，以及不同输入 dtype 的上游回退。
CPU 中的 NPU 算子使用 mock。

在 Atlas 950DT 的完整 vLLM Ascend 环境运行实际算子及组件测试：

```bash
pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_routed_output_norm_quant_npu.py \
  -k 'not latency'
```

算子测试覆盖 1/9/65/257 token、两种 epsilon，以及随机、零、极小、极大和量化边界输入。
比较量化值和 scale 的 shape/dtype，以及反量化后的数值；融合前后的 BF16 舍入位置
可能不同，不要求 FP8 字节完全一致。组件测试使用实际 MXFP8 linear method，
验证 3584 → 7168 的 latent up、padding 截断和反复 ACLGraph replay 时读取新输入。

采集组件耗时：

```bash
pytest -q -s \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_routed_output_norm_quant_npu.py \
  -k latency -o junit_family=legacy \
  --junitxml=/tmp/kimi-routed-output-latency.xml
```

基准覆盖 1/8/64/256 token，使用相同权重和输入、预热及图回放，交替执行融合与未融合
版本，输出三次测量的中位数和比值。测量包含 norm/quant 与 latent up；先建立实际硬件
基线，再设置性能回归阈值。

## 整模验收与当前证据

组件测试不执行 TP/EP 通信。真实权重仍需按原 Kimi K3 部署配置验收：

1. 固定权重、请求和并行配置，对比融合开关两侧的 eager、ACLGraph、npugraph_ex 和静态 kernel。
2. 覆盖 prefill/decode、多流开关、DSpark、不同 capture size，以及非 TP 整倍数 token 数。
3. 检查固定输入 logits、greedy 输出和长序列稳定性。
4. profiling 定位 combine 后的 latent up，确认独立 RMSNorm/DynamicMxQuant 被一个
   RmsNormDynamicMxQuant 替换，后接一次原量化矩阵乘，且没有重复量化。
5. 比较该片段、MoE 层及服务 TPOT；记录输入连续化产生的拷贝开销。

开发机为 CPU PyTorch 2.7.1 / TorchNPU 2.7.1.post4，缺少 vLLM 和可见 NPU，
不能执行完整 UT 初始化、设备精度、图回放或整模验收。隔离 CPU 检查使用本提交的类和
测试函数、上游 vLLM v0.27.1 的 transform/runner 方法，验证控制流与接口；
本机还验证了真实 TorchNPU schema/meta 输出为 FP8 E4M3 和 `(tokens, 56, 2)` scale。
这些结果不能替代目标镜像及真实权重的验证，当前没有设备加速比结论。
