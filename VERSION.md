# 0.30.0-v1.0 PR12_仅SiTU

保留共享专家 BF16 sparse SiTU、符合条件的 A5 未量化路由专家 sparse SiTU，以及 W4A8MXFP 路由专家 grouped SiTU。A5 TopK 直接使用注册的 NPU 算子，避免 recipes 导入后覆盖 Python 别名。

源码基于 `f89e5592d6fbcb36ee8ce0e538d23f0dd8f951b8`，按工作区 `PR12_仅SiTU` 交付补丁重建。SiTU 的实现来源见 [cann-framework/vllm-ascend PR12](https://github.com/cann-framework/vllm-ascend/pull/12)。

## 配置

建议在同一组 MoE 运行参数上比较原版和本分支。比较 SiTU 效果时，保持共享专家多流和 Fused MC2 关闭。

将下列字段并入现有的 `--additional-config`。

```json
{
  "multistream_overlap_shared_expert": false,
  "enable_fused_mc2": 0
}
```

在各模型 worker 环境中安装匹配的 recipes `custom_ops` 扩展和 CANN 算子包。图模式还需要配套的图转换器。算子缺失时按原生路径执行。运行后检查注册和实际选用的算子。

## 验证

已对照原交付清单核验 9 个文件，运行了 7 项隔离 CPU 回归。完整仓库测试、Ascend NPU 精度和性能验证尚未运行。

在源码根目录运行本分支的隔离 CPU 回归。

```bash
python tests/ut/ops/test_kimi_situ.py
```
