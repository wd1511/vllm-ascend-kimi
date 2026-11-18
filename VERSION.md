# 0.27.1rc-v1.5

新版 0.27.1rc + 261027/v1.1C

## 源码基线

GitCode 的 `wangtao43/vllm-ascend` 发布分支 `releases/v0.27.1rc`，提交
`75efd5e19a35c6d33212b897ea7e96ccfea0fe7a`。
将 `261027/vllm-ascend-kimi-v1.1C` 与其原始基线 `0838a2fcce8616bb07499a89427e92244f4acfce` 的差异
移植到该提交。

## 功能

- 在 v1.1B 基础上加入有界 DP 元数据流池，按 target/draft 选择流优先级，并对已完成的事件使用查询快路径。
- dense 路径按条件跳过 DP 同步。Model Runner V1 增加可选的迁移感知 EPLB 策略 4。
- A5 MegaMoE 沿用当前发布分支的四字段 CPU 元数据同步，保留跨 DP 的有效 rank 和 decode 标志协调。

## 使用与验证

按 [安装说明](README.zh.md) 配置配套的 vLLM、CANN、PyTorch 与 torch-npu。本分支的仓库依赖见 [requirements.txt](requirements.txt)。

普通 DP 元数据默认走独立流，可设置 `VLLM_ASCEND_DP_METADATA_ALLREDUCE_ON_COMPUTE_STREAM=1` 切换到计算流。
`--additional-config` 的 `sync_optimization` 可配置 `stream_pool_size`（默认 2）、
`metadata_stream_priority`（默认 -1）和 `enable_event_query_fast_path`（默认 true）。
使用 EPLB 策略 4 时，设置 `DYNAMIC_EPLB=1`，并在 `eplb_config` 中设置 `dynamic_eplb=true` 和 `eplb_policy_type=4`。

安装 CPU 版 PyTorch、NumPy 和 pydantic 后可从仓库根目录运行隔离回归。

```bash
python3 -B tests/ut/eplb/core/policy/test_migration_aware.py
python3 -B tests/ut/spec_decode/test_async_token_correction.py
python3 -B tests/ut/test_dp_metadata_skip.py
python3 -B tests/ut/test_sync_optimization_config.py
python3 -B tests/ut/worker/test_accepted_token_sync.py
python3 -B tests/ut/worker/test_dp_metadata_stream_order.py
```

目标 NPU 环境可运行双进程 DP 元数据微基准。

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 benchmarks/dp_metadata_stream_pool.py
```

完整 vLLM 测试、Ascend NPU 实机正确性和性能验证待执行。

[EventSync 源码与 Profiling 报告](https://github.com/wd1511/vllm-ascend-kimi/blob/e32a1b02e16b26ab1b723a92ee762624ef1330dd/01EventSyn/README.md)
[GitCode 版本提交](https://gitcode.com/wangtao43/vllm-ascend/commit/75efd5e19a35c6d33212b897ea7e96ccfea0fe7a)
