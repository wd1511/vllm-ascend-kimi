# PR48 latent down 独立修复

## 修复范围

原路径在输入变换处直接返回 7168 维原始输入，上游按照专家维度
3584 做负 padding，将其截断。MoE 内部又根据宽度误判投影已经完成，
最终跳过真实的 `routed_expert_down_proj`。

现在恢复输入变换阶段的真实投影，再进入上游 padding 和 MoE：

```text
原始输入 -> latent down -> 专家维度 padding -> routed experts
原始输入 -> gate / shared experts
```

MoE 内部通过显式的原始输入参数复用已投影结果，不再根据宽度猜测是否做过投影。
正常路径只执行一次 latent down。已删除 PR48 的临时调测日志及其专用变量，
保留正常运行日志和错误诊断。

这是正确性修复，不是 recipes 多流调度对齐：正常路径的 latent down
现在先于 MoE 内部 gate 执行。未调整 shared SiTU/GMM1、通信事件或路由精度。

## 验证

```bash
pytest -q tests/ut/ops/test_fused_moe.py -k 'kimi_front or kimi_runner_forward'
```

回归测试经过真实的上游 `forward` 和 padding 路径，检查投影值不等于输入截断值、
投影只执行一次、对齐 padding、等宽投影和显式传入已投影输入的路径。

在相同真实权重、输入和运行配置下，分别重新采集 eager 和 aclgraph，
用已有精度脚本对比。不要覆盖已确认正确的基线；`selfcheck` 只能检查重复性。
最终仍需 NPU 实测确认精度恢复，CPU 控制流测试不能替代真实权重验证。
