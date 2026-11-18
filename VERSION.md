# 0.27.1rc-v1.4

新版 0.27.1rc + 261027/v1.1B

## 源码基线

GitCode 的 `wangtao43/vllm-ascend` 发布分支 `releases/v0.27.1rc`，提交
`75efd5e19a35c6d33212b897ea7e96ccfea0fe7a`。
将 `261027/vllm-ascend-kimi-v1.1B` 与其原始基线 `0838a2fcce8616bb07499a89427e92244f4acfce` 的差异
移植到该提交。

## 功能

- 通过可选 accepted 输出拆开输入校正和 accepted-token 发布。公共异步校正只更新 computed-token，Host 在 Mamba 前发布已完成的 accepted-token 快照。
- 310P 专用输入准备沿用其早期同步和 accepted 输出。
- A5 MegaMoE 沿用当前发布分支的四字段 CPU 元数据同步，保留跨 DP 的有效 rank 和 decode 标志协调。

## 使用与验证

按 [安装说明](README.zh.md) 配置配套的 vLLM、CANN、PyTorch 与 torch-npu。本分支的仓库依赖见 [requirements.txt](requirements.txt)。

普通 DP 元数据默认走独立流。设置 `VLLM_ASCEND_DP_METADATA_ALLREDUCE_ON_COMPUTE_STREAM=1` 可切换到计算流。

安装 CPU 版 PyTorch、NumPy 和 pydantic 后可从仓库根目录运行隔离回归。

```bash
python3 -B tests/ut/worker/test_dp_metadata_stream_order.py
python3 -B tests/ut/worker/test_accepted_token_sync.py
python3 -B tests/ut/spec_decode/test_async_token_correction.py
```

完整 vLLM 测试、Ascend NPU 实机正确性和性能验证待执行。

[EventSync 源码与 Profiling 报告](https://github.com/wd1511/vllm-ascend-kimi/blob/e32a1b02e16b26ab1b723a92ee762624ef1330dd/01EventSyn/README.md)
[GitCode 版本提交](https://gitcode.com/wangtao43/vllm-ascend/commit/75efd5e19a35c6d33212b897ea7e96ccfea0fe7a)
