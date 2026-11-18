# 0.30.0-v1.2 PR12_控核与多流_SiTU

按 PR12 v2 快照保留 Kimi Router 与共享专家多流、投影控核、V1/V2 的 Decode 条件检查，以及共享和路由专家 SiTU 替换。符合条件的 Router 和 BF16 共享 Gate/Up 使用 32 个 AIC、1 个 AIV，MXFP 共享 Gate/Up 使用 16 个 AIC、32 个 AIV。

源码基于 `0807cef7025695bfdec9267377b4449d470ba1e6`，按工作区 `PR12_控核与多流_SiTU` 交付补丁重建。SiTU 的实现来源见 [cann-framework/vllm-ascend PR12](https://github.com/cann-framework/vllm-ascend/pull/12)。

## 配置

按下方配置启用普通 MC2 和共享专家并行。仅符合 A5、Decode、MC2、SP、BF16 和量化要求的批次进入专用调度。基线上含 PR5 的注意力残差实现，详细事件依赖和外部算子要求见本分支 [PR12 v2 技术说明](docs/source/developer_guide/kimi_k3_v030_migration.md)。

将下列字段并入现有的 `--additional-config`。

```json
{
  "multistream_overlap_shared_expert": true,
  "enable_fused_mc2": 0,
  "enable_shared_expert_dp": false,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_super_kernel": false
  }
}
```

在各模型 worker 环境中安装匹配的 recipes `custom_ops` 扩展和 CANN 算子包。图模式还需要配套的图转换器。算子缺失时按原生路径执行。运行后检查注册和实际选用的算子。

## 验证

已对照原交付清单核验 26 个文件，运行了 26 项隔离 CPU 回归。完整仓库测试、Ascend NPU 精度和性能验证尚未运行。

在源码根目录运行本分支的隔离 CPU 回归。

```bash
python tests/ut/ops/test_kimi_moe_front_overlap.py
python tests/ut/ops/test_kimi_v30_moe.py
python tests/ut/ops/test_kimi_v30_review.py
python tests/ut/ops/test_kimi_situ_loading.py
python tests/ut/models/test_kimi_v2_sequence_parallel.py
```
