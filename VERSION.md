# 0.27.1rc-v1.1

新版 0.27.1rc + 261027/v1.0

## 源码基线

GitCode 的 `wangtao43/vllm-ascend` 发布分支 `releases/v0.27.1rc`，提交
`75efd5e19a35c6d33212b897ea7e96ccfea0fe7a`。
将 `261027/vllm-ascend-kimi-v1.0` 与其原始基线 `0838a2fcce8616bb07499a89427e92244f4acfce` 的差异
移植到该提交。

## 功能

- 移植原始同步优化提交 `8c3723643`。满足 DP 同步条件时，专用 NPU 流负责元数据 all-reduce，Host 在 Mamba 前同步 accepted-token 状态。
- 该快照保留当时的首次 buffer 分配和 310P 调用顺序，供复现原始方案。
- A5 MegaMoE 沿用当前发布分支的四字段 CPU 元数据同步，保留跨 DP 的有效 rank 和 decode 标志协调。

## 使用与验证

按 [安装说明](README.zh.md) 配置配套的 vLLM、CANN、PyTorch 与 torch-npu。本分支的仓库依赖见 [requirements.txt](requirements.txt)。

其余 DP 场景的元数据默认走独立流。设置 `VLLM_ASCEND_DP_METADATA_ALLREDUCE_ON_COMPUTE_STREAM=1` 可切换到计算流。

完整 vLLM 测试、Ascend NPU 实机正确性和性能验证待执行。

[EventSync 源码与 Profiling 报告](https://github.com/wd1511/vllm-ascend-kimi/blob/e32a1b02e16b26ab1b723a92ee762624ef1330dd/01EventSyn/README.md)
[GitCode 版本提交](https://gitcode.com/wangtao43/vllm-ascend/commit/75efd5e19a35c6d33212b897ea7e96ccfea0fe7a)
