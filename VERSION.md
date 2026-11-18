# 0.30.0-v1.4

本分支对应 `GitHub_PR5_PR12_完整版`。源码来自 `cann-framework/vllm-ascend` 的 `cdab93b6d2fd0f8855494641392893372fe38da9`，其父提交 `0807cef7025695bfdec9267377b4449d470ba1e6` 已包含 PR #5。分支在该合并提交上增加本说明，源码和子模块指针与源仓一致。 [PR #5](https://github.com/cann-framework/vllm-ascend/pull/5) · [PR #12](https://github.com/cann-framework/vllm-ascend/pull/12)

## 功能

- PR #5 为 Kimi K3 的 AttnRes 增加 `fused`、`two_phase`、`original` 和 `native` 四种后端。默认请求 `fused`，必要算子不可用时回退到 `two_phase`。
- PR #12 增加 Kimi 文本编译和 Decode MoE 多流。符合条件的 A5、MC2、SP、纯 Decode 批次使用分阶段的共享专家计算和通信，并限制 Router Gate 和共享 Gate-Up 的计算核。
- 图捕获可将符合条件的 Decode MoE 调度范围交给 SuperKernel 优化。符合硬件、量化和批次条件时，V1 纯 Prefill 可用 `enable_fused_mc2=2` 选择 MegaMoE。

## 使用

在与该提交匹配的 vLLM、CANN 和 torch-npu 环境中构建。Git 检出后运行 `git submodule update --init --recursive`，按固定提交获取 Catlass 及其 googletest 子模块。

需要验证 V2 Decode SuperKernel 时，设置 `VLLM_USE_V2_MODEL_RUNNER=1`，并将以下字段合入 `--additional-config`。

```json
{
  "multistream_overlap_shared_expert": true,
  "enable_fused_mc2": 0,
  "enable_shared_expert_dp": false,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": true,
    "enable_super_kernel": true
  }
}
```

稀疏或分组 SiTU 使用可选 recipes `custom_ops`，需在各 worker 安装匹配扩展。接口约束、回退条件和运行时调度见 [Kimi K3 0.30 移植说明](docs/source/developer_guide/kimi_k3_v030_migration.md)。

## 验证

归档中主仓的 4,478 个文件已逐一核对 Git blob 和文件模式。分支保留 Catlass 指针 `41bf90da655bba3c66d0acd7e00abe33960ecfd6`。完整构建、NPU 图捕获、多卡通信和性能仍需在目标环境验证。
