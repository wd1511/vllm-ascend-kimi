# 0.30.0-v1.5

本分支对应新版 0.30.0。源码直接取自 `cann-framework/vllm-ascend` 的 `releases/v0.30.0`，固定于 `5c3d71048fa55669b408bc1c8eeabc2d6183bc82`。 [源发布分支](https://github.com/cann-framework/vllm-ascend/tree/releases/v0.30.0) · [源码提交](https://github.com/cann-framework/vllm-ascend/commit/5c3d71048fa55669b408bc1c8eeabc2d6183bc82)

## 本版内容

源码包含 PR #5 的 AttnRes 和 PR #12 的 Kimi 文本编译、Decode 多流和 MoE SuperKernel。此后源发布分支又为 V2 加入协调后的 Prefill MegaMoE 和模型内 Router stream 复用，并通过 PR #17 接入 FlashMLA 及相关元数据。

## 使用

按 [仓库 README](README.zh.md) 安装与该发布分支匹配的 vLLM、CANN 和 torch-npu。Git 检出后运行 `git submodule update --init --recursive` 获取固定版本的构建依赖。

Kimi 的编译、Decode 多流、SiTU 依赖和 Prefill MegaMoE 分别有硬件、量化及运行时约束。V2 由 `VLLM_USE_V2_MODEL_RUNNER=1` 启用。具体配置和期望日志见 [Kimi K3 0.30 移植说明](docs/source/developer_guide/kimi_k3_v030_migration.md)。

## 验证

本分支相对源提交只增加本说明，源码和子模块指针与源仓一致。NPU 端到端推理、图捕获、多卡通信和性能需在目标硬件验证。
