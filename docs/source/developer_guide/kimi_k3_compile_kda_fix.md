# Kimi K3 编译时 KDA metadata 分支修复

## 故障与证据

Issue #5 的用户交接记录确认：profile run 首次追踪时 `attn_metadata=None`，
Python KDA 分派进入 `core_attn_out.zero_()`；vLLM 的 VLLM_COMPILE 路径丢弃
相应 Dynamo guards，后续请求继续使用缺失 KDA 核心计算的图。
原始 FX 图四个 KDA 层只剩清零及投影，修复后的图包含四个 `ascend_kda_forward` 节点。

用户提供的硬件验证使用 8×Ascend950DT、CANN 9.3.0、torch/torch_npu 2.10、
Kimi-K3-trim5 五层模型、TP8+EP、四个输入 token、`max_tokens=1`。
原补丁使首 token 从 84176 恢复为 eager 的 31390，报告分数为 -5.621110439300537，
与该参照逐位一致。这是指定案例的结果，不代表完整模型、多步 decode 或静态 kernel 已通过。
原始日志位于用户测试机 `/tmp/k3-compile-bisect-run1/`，本次本地未重新执行硬件实验。

## 正式实现与审查

- 将 fused、AscendC 共用的核心分派放到不透明 `vllm::ascend_kda_forward` 内。
  metadata 在图执行时读取；fake 实现不访问 metadata，也不执行 KDA。
- 输出由调用方分配，schema 声明 `mutates_args=["core_attn_out"]`，确保 AOT
  functionalization 保留输出写入。空 metadata 的清零分支保留在真实实现中，
  供 profile/dummy 执行使用；它不会成为唯一的编译计算路径。
- 复用上游 KDA 构造器注册的 `static_forward_context`，通过当前
  `get_forward_context().no_compile_layers[layer_name]` 查找层。
  不新增交接补丁中的 `_KDA_LAYERS` 全局字典，避免多模型同名前缀覆盖及强引用残留。
- `eager_break_during_capture` 放在 op 内部的真实实现 `_kda_forward_impl` 上，
  分开 Dynamo 的不透明边界和 breakable graph 的运行时捕获边界。
- 不改变 beta 预处理、prefill/decode 内核、缓存索引和 KDA 数学逻辑。
  卷积/SSM 缓存仍由层对象持有并在不透明执行中更新；本次没有重构其隐式副作用契约。
  因此必须补做连续 decode、重复 replay、跨请求状态隔离验证，首 token 一致不足以覆盖这些行为。
- Mega-KDA 的 `_plan_mega_kda` 仍在 Python 分派中。空 metadata 的首次追踪会选择
  现在已经修复的默认 KDA 路径；本 PR 不宣称编译模式的 Mega-KDA 选择已修复。
  本次验收固定 `enable_mega_kda=false`，后续需将整个 Mega/默认分派放入运行时边界。

修改后的层查找和捕获装饰器位置不同于用户已验证补丁，需对最终提交重新做 NPU 验证。
后续按用户要求撤销 PR54 的 AttnRes Phase-2 返回值传递及注册算子选择改动，
恢复 PR54 引入前的 Python wrapper、单输出和原地 partial 更新约定；保留模型编译入口。
以上硬件结果对应回退之前的版本，不能作为回退后 eager/aclgraph/npugraph_ex 的验收结果。
回退后须在相同权重、请求和算子包下重新与指定 0.27 eager 基线对比，
并确认 Python wrapper 的编译兼容性及底层算子的原地更新 schema。
本地 CPU 测试使用生产函数的源码和真实 PyTorch 2.10 Dynamo/AOT，模拟数值内核，
确认空 metadata 编译后仍能读取新 metadata、按当前 context 找到同名新层、输出修改被保留，
并且不重新编译。既有算子测试改为直接测试 `_kda_forward_impl`；新测试覆盖外层 op 边界。

```bash
python tests/ut/ops/test_kimi_kda_compile.py -v
pytest -q tests/ut/ops/test_kimi_kda.py tests/ut/ops/test_kimi_kda_fused.py
```

第二条需要完整 vLLM/torch_npu/算子依赖。本地未运行完整插件测试和 NPU 测试。

## 静态 kernel 的可执行回退方案

先保留此正确性修复，合并以下字段到原服务的 `additional_config`，其他配置保持一致：

```json
{
  "enable_mega_kda": false,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": false
  }
}
```

这关闭静态算子特化，保留 npugraph_ex 路径；是否成功仍以当前安装版本实际运行结果为准。
在新的服务进程中先验证 eager 与此模式的输出、真实 KDA 执行和状态更新。
不要通过恢复 KDA 清零来绕开静态 kernel 错误，也不要在报设备错误后继续复用该进程。

若必须启用静态 kernel，可针对已报告失败的三个 CANN op type 做选择性排除，
其他算子继续静态编译。已有实验分支采用的私有接口是
`npugraph_ex._acl_concrete_graph.static_kernel._set_static_kernel_blacklist`。
必须先核对测试机安装版本确实提供该接口及列表覆盖语义，保留已有排除项。
在 compiler backend 构建之前、仅在 static kernel 开启时，诊断补丁的调用形式为：

```python
from npugraph_ex._acl_concrete_graph.static_kernel import _set_static_kernel_blacklist

# Merge these CANN op types into the deployment's existing exclusions.
_set_static_kernel_blacklist([
    "FusedInferAttentionScore",  # Preserve PR54's existing exclusion.
    "CausalConv1d",
    "SituAndMulSparse",
    "GroupedSituMxQuant",
])
```

这不是 `ascend_kda_forward` 的 blacklist：排除的是底层静态算子编译，不是删除 KDA 节点。
预期由既有动态算子实现执行；必须在日志和 profiling 中验证三个算子实际回退且仍执行，
不得把“编译不报错”当作回退成功。该私有接口不适用于所有 npugraph_ex/torchair 版本；
若不存在或不支持可靠回退，保持 `enable_static_kernel=false`。
本 PR 参考 PR34，在 npugraph_ex 开启 static kernel 时自动排除
`FusedInferAttentionScore`，在 backend 创建前设置并记录日志；其他三个算子仍需按实际错误诊断。
该设置不删除 FIA 计算，预期使用其动态 kernel；NPU 上的捕获、回放和精度验证仍待执行。

## 三个失败算子的长期处理

| 算子 | 用户报告 | 修复与验证方向 |
| --- | --- | --- |
| CausalConv1d | E90003，输入 `[16,4608]`、权重 `[4,4608]`、状态 `[84287,3,4608]` | 用原始 opcompile JSON 在相同 CANN 上单算子复现；核对 shape、属性、可选输入、dtype/layout，适配静态 tiling 或更新匹配的算子包。不能仅凭 E90003 断言“大状态一定不支持”。 |
| SituAndMulSparse | `op_func_attr is None` | 查自定义算子注册/生成元信息及静态编译工具兼容性，修复后重新构建算子包并做单算子验证；不能伪造属性值绕过检查。 |
| GroupedSituMxQuant | opc 失败 | 先取得 opc 最早的具体错误、输入签名及完整编译日志，再定位实现/注册/工具链问题；“opc 失败”本身不足以判定原因。 |

`SituAndMulSparse` 是 BF16 激活算子，也用于未量化的 shared experts，
不能推断“取消权重量化后必然消失”。`GroupedSituMxQuant` 的确执行激活量化，
但它报静态编译错误不等于 checkpoint 数值错误；具体使用何种专家量化方案以运行代码和配置为准。

## 507015 的归因步骤

同步点报告 507015 可能是在收集更早的异步设备错误。
“85/88 静态编译成功”与该设备错误的因果关系尚未证实，不能据此宣布安装包不完整就是根因。

1. 固定最终 KDA 修复、权重、输入和 capture sizes，分别运行 static=false 与 static=true。
2. static=true 下排除上述三个算子，确认动态回退、捕获、真实推理和连续 replay 正常。
3. 若成功，再逐个取消排除，每次独立进程，记录首个失败算子与最早的设备/编译错误。
4. 保留完整 CANN/device 日志、opcompile JSON、算子包版本和安装结果。
   静态包若为共享安装位置，应由环境维护者按对应版本流程清理/隔离，避免干扰其他服务。
5. 修复前运行相同 static 配置仅作为补充对照：若同样失败说明问题不依赖本补丁；
   若修复前通过，也不能归咎 custom op，因为修复前根本没有执行真实 KDA/卷积。

最终验收要求：图内有 KDA 调用、prefill/decode 都实际执行、状态更新正确，
eager/aclgraph/npugraph_ex 对齐；静态 kernel 单独给出通过或未通过结论。
