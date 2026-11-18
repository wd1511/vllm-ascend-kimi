# 0.27.1rc-v1.0

新版 0.27.1rc，供本仓的 EventSync 系列对照。

## 源码基线

GitCode 的 `wangtao43/vllm-ascend` 发布分支 `releases/v0.27.1rc`，提交
`75efd5e19a35c6d33212b897ea7e96ccfea0fe7a`。

## 功能

该分支包含发布分支已有的 Kimi-K3 适配、共享专家分流、编译支持以及 A5 MegaMoE
在纯预填充批次上的 DP 协调。源码与上面的发布提交一致。

## 使用与验证

按 [安装说明](README.zh.md) 配置配套的 vLLM、CANN、PyTorch 与 torch-npu。
本分支的仓库依赖见 [requirements.txt](requirements.txt)。

完整 vLLM 测试和 Ascend NPU 实机验证待执行。

[GitCode 版本提交](https://gitcode.com/wangtao43/vllm-ascend/commit/75efd5e19a35c6d33212b897ea7e96ccfea0fe7a)
