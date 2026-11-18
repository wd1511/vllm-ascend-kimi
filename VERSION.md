# 0.27.1rc-v1.2

新版 0.27.1rc + 261027/v1.1

## 源码基线

GitCode 的 `wangtao43/vllm-ascend` 发布分支 `releases/v0.27.1rc`，提交
`75efd5e19a35c6d33212b897ea7e96ccfea0fe7a`。
将 `261027/vllm-ascend-kimi-v1.1` 与其原始基线 `0838a2fcce8616bb07499a89427e92244f4acfce` 的差异
移植到该提交。

## 功能

- 将 DP 元数据交换提交到独立流，在该流内首次分配设备 buffer。
- 将 accepted-token 等待放到独立输入准备之后，保留原有请求映射、correction helper 和 310P 调用顺序。
- A5 MegaMoE 沿用当前发布分支的四字段 CPU 元数据同步，保留跨 DP 的有效 rank 和 decode 标志协调。

## 使用与验证

按 [安装说明](README.zh.md) 配置配套的 vLLM、CANN、PyTorch 与 torch-npu。本分支的仓库依赖见 [requirements.txt](requirements.txt)。

普通 DP 满足同步条件时使用独立流。沿用原启动配置。

安装 CPU 版 PyTorch、NumPy 和 pydantic 后可从仓库根目录运行隔离回归。

```bash
python3 -B tests/ut/worker/test_dp_metadata_stream.py
python3 -B tests/ut/worker/test_input_prep_sync.py
```

完整 vLLM 测试、Ascend NPU 实机正确性和性能验证待执行。

[EventSync 源码与 Profiling 报告](https://github.com/wd1511/vllm-ascend-kimi/blob/e32a1b02e16b26ab1b723a92ee762624ef1330dd/01EventSyn/README.md)
[GitCode 版本提交](https://gitcode.com/wangtao43/vllm-ascend/commit/75efd5e19a35c6d33212b897ea7e96ccfea0fe7a)
