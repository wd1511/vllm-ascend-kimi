# PR !54 共享专家多流的 ACLGraph 捕获排查

## 当前结论（2026-11-01）

以用户最新澄清为准：开启 `multistream_overlap_shared_expert` 报错，关闭正常。
先前按“关闭时报错”处理的串行清理补丁 `448b8967f` 未解决现场故障。
它验证了串行分支不访问 overlap 辅助流，不代表捕获问题已修复。

现场第一条错误发生在 `torch.npu.graph` 退出时的 `capture_end`：

```text
error code is 107025
Invalid_Argument(EE1017): Capture model validity check failed.
In the cross-stream capture scenario, the last task (task_id=27)
on the sub stream (stream_id=16) is not an event record task.
rtStreamEndCapture execution failed:
capture model contains a stream that was not joined to the origin stream
```

设备为 7，worker 为 TP7/EP7。后续 model runner 异常是同一错误的重复包装。
该日志不能单独证明 stream 16 是哪条 Python 流，也不是 `record_stream` 不支持的错误。
精度问题暂缓，本轮不修改 fusion pass、AttnRes 或编译配置。

## 与 PR !34 的差异及本次修复

已核对远端 PR !34 的 head 为 `1322e2f86`，PR !54 的修改前 head 为 `448b8967f`。
!34 中的 `c86b6025e` 引入了编译模式下的共享输出等待策略，!54 缺少这段策略。

!54 原有 `shared_experts.wait_for_output()`，但放在 MoE 自定义算子外的
`apply_routed_output_transform()` 中。SP 多流路径无条件传入 `defer_output_wait=True`，
因而 MoE 内共享通信流执行 reduce-scatter 后不立即 join。
编译时，不能依赖算子外的 Python 流操作完整保留这种跨流依赖。

本次恢复 !34 的策略：

- 构造 runner 时读取 compilation mode，仅 `CompilationMode.NONE` 允许延迟 join。
- 编译模式在两套版本的 `_forward_impl()` 中传入 `defer_output_wait=False`。
- `AscendSharedExperts.forward()` 在通信流尾部工作之后调用主流的 `wait_stream(tail_stream)`。
- eager 模式保留 reduce-scatter 与 latent up projection 的重叠，之后调用 `wait_for_output()`。
- 共享专家多流关闭时保持串行分支；不额外创建或等待辅助流。

`wait_stream` 的作用包括在被等待流上记录尾部事件，再让当前流等待该事件。
只在主流末尾添加 `record_event()`，或在编译图外重复添加 `wait_for_output()`，不能代替这个位置的 join。
当前改动与现场错误相符，但仍是待 NPU 验证的修复，不能据本地 mock 宣称 stream 16 已被定位。
代价是编译模式下不再让共享 reduce-scatter 与算子外的 latent up projection 重叠；需测量性能影响。

## 本地验证及限制

本地没有 NPU、PyTorch、pytest、真实权重或远端环境连接。
新增/更新 `tests/ut/ops/test_fused_moe.py`，覆盖编译模式、延迟等待策略和通信尾流等待顺序。
使用标准库 AST 提取实际生产方法及对应测试，再通过 mock 执行控制流：

- 30 组 runner 构造检查：缺省配置/eager/编译、共享布局、overlap、SiTU 组合。
- 2 组 forward 等待参数传递、3 组 output transform 等待检查。
- 模拟通信返回时的完成事件等待，检查其后才执行共享尾流 join。
- 前一版 64 组串行分支及 overlap 事件阶梯检查继续通过。

以上不覆盖真实 tensor 数值、torch.compile 追踪、NPU 捕获或图重放。
原 !54 代码运行新增等待策略检查失败，本次代码通过。正式 pytest 因缺少依赖未运行。
Ruff check/format、Python 语法检查及 git diff --check 通过。
`bash format.sh ci` 未完成：markdownlint 安装使用 Node 18 与 npm 12，报 `ERR_REQUIRE_ESM`。

## 原 NPU 环境复测

在运行环境更新 PR !54 分支，先保存本地改动，避免覆盖现场配置。
在工作区干净且已位于该 PR 分支时执行：

```bash
cd /vllm-workspace/vllm-ascend
git fetch origin feat/kimi-k3-support-torch-compile
git merge --ff-only origin/feat/kimi-k3-support-torch-compile
git log -1 --oneline
python -c "import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)"
rg -n '_should_defer_shared_output_wait|_defer_sp_shared_output_wait' \
  vllm_ascend/ops/fused_moe/fused_moe.py
```

执行针对性 UT：

```bash
pytest -q tests/ut/ops/test_fused_moe.py \
  -k 'sp_multistream or forward_impl_keeps_full_width or serial_shared_experts'
```

首先保留现场所有参数，仅确认 `additional_config` 下共享专家多流为 True。
使用原权重、TP8/EP、原捕获档位及原失败命令，从全新进程复测。
可使用原 precision 脚本采集，关闭编译缓存避免复用旧图：

```bash
mkdir -p /tmp/k3-pr54-capture-check
VLLM_DISABLE_COMPILE_CACHE=1 VLLM_LOGGING_LEVEL=DEBUG \
python /home/w00855685/precision/kimi_k3_graph_precision.py capture \
  --mode aclgraph --max-tokens 8 \
  --output /tmp/k3-pr54-capture-check/shared-on.json \
  > /tmp/k3-pr54-capture-check/shared-on.log 2>&1
```

随后仅把共享专家多流改为 False，另起进程保存 `shared-off.json` 与 `shared-off.log`。
不要同时改 fusion pass、AttnRes、捕获档位或强制 eager。
验收要求两种配置都完成捕获、真实请求及实际 decode replay；启动完成或单 token 返回不足以验收。
确认单请求实际命中捕获档位后，再进行性能及精度比较。

如果仍失败，保留第一条 runtime 错误前后日志、两个仓库 SHA、实际配置、软件版本和捕获档位。
下一步需要 CANN runtime 流映射及任务记录，区分共享计算/通信流、KDA、HCCL 或算子内部流。
