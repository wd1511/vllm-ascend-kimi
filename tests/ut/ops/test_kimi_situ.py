# SPDX-License-Identifier: Apache-2.0
"""Exercise the Kimi SiTU integration on CPU with device operators stubbed."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SHARED = "vllm_ascend/ops/fused_moe/shared_experts.py"
MOE_MLP = "vllm_ascend/ops/fused_moe/moe_mlp.py"
MOE_UTILS = "vllm_ascend/ops/fused_moe/moe_utils.py"
QUANT = "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py"
SITU_OPS = "vllm_ascend/ops/kimi_situ.py"
ACTIVATION = NS(SITU="situ")
QUANT_TYPE = NS(NONE=0, W4A8MXFP=1)
CAPABILITY = NS(CANN_MEGAMOE_MXFP=object())
BETA = 4.0
ALPHA = 25.0


def load(path, names, ns, class_name=None):
    """Load real implementations without the full vLLM or torch_npu import graph."""
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
        cls.decorator_list = []
        nodes = [cls]
    else:
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", None) in names
            or (
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
            )
        ]
    nodes.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec"), ns)


class TorchOpsStub:
    def __init__(self, custom=None, native=None):
        self.ops = NS(custom=NS() if custom is None else custom, _C_ascend=NS() if native is None else native)

    def __getattr__(self, name):
        return getattr(torch, name)


class ReferenceSitu(nn.Module):
    def __init__(self, beta, linear_beta, compile_native=False):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward_native(self, x):
        gate, up = x.float().chunk(2, dim=-1)
        return (
            self.beta
            * (gate / self.beta).tanh()
            * gate.sigmoid()
            * self.linear_beta
            * (up / self.linear_beta).tanh()
        ).to(x.dtype)

    def forward(self, x):
        return self.forward_native(x)


def load_shared_activation(ns):
    ns["SituAndMul"] = ReferenceSitu
    load(SHARED, {"__init__", "can_use_sparse", "forward_native"}, ns, "AscendSharedSparseSituAndMul")
    return ns["AscendSharedSparseSituAndMul"](BETA, ALPHA)


class TestSituRegistration(unittest.TestCase):
    def test_existing_sparse_registration_loads_grouped_once(self):
        custom = NS(npu_situ_and_mul_sparse=Mock())

        def register(_):
            custom.grouped_situ_mx_quant = Mock()

        ns = dict(torch=TorchOpsStub(custom), import_module=Mock(side_effect=register), logger=Mock())
        load(SITU_OPS, {"SITU_OP_NAMES", "initialize_kimi_situ_ops"}, ns)
        ns["initialize_kimi_situ_ops"]()
        ns["initialize_kimi_situ_ops"]()
        ns["import_module"].assert_called_once_with("custom_ops")
        ns["logger"].warning_once.assert_not_called()

    def test_unavailable_extension_leaves_native_fallback(self):
        for error in (ImportError("custom_ops"), OSError("libcust_opapi.so"), None):
            with self.subTest(error=error):
                custom = NS(npu_situ_and_mul_sparse=Mock())
                ns = dict(torch=TorchOpsStub(custom), import_module=Mock(side_effect=error), logger=Mock())
                load(SITU_OPS, {"SITU_OP_NAMES", "initialize_kimi_situ_ops"}, ns)
                ns["initialize_kimi_situ_ops"]()
                self.assertFalse(hasattr(custom, "grouped_situ_mx_quant"))
                self.assertIn("grouped_situ_mx_quant", ns["logger"].warning_once.call_args.args[1])

    def test_model_preserves_native_activation_when_sparse_is_ineligible(self):
        cases = [
            (True, "situ", ALPHA, True, True),
            (True, "situ", ALPHA, False, False),
            (False, "situ", ALPHA, True, False),
            (True, "silu", ALPHA, True, False),
            (True, "situ", None, True, False),
        ]
        for a5, activation, alpha, op_installed, expect_sparse in cases:
            with self.subTest(a5=a5, activation=activation, alpha=alpha, op_installed=op_installed):
                custom = NS()
                native_activation = NS(beta=BETA, linear_beta=alpha)
                sparse_activation = object()

                def register():
                    if op_installed:
                        custom.npu_situ_and_mul_sparse = Mock()

                init_ops = Mock(side_effect=register)
                sparse_factory = Mock(return_value=sparse_activation)
                moe_factory = Mock(
                    return_value=NS(moe_config=NS(num_logical_experts=4, num_experts=4, num_local_experts=4))
                )
                ns = dict(
                    torch=TorchOpsStub(custom),
                    nn=nn,
                    get_current_hardware_profile=lambda: NS(supports=lambda capability: a5),
                    HardwareCapability=CAPABILITY,
                    initialize_kimi_situ_ops=init_ops,
                    GateLinear=Mock(return_value=NS()),
                    KimiMLP=Mock(return_value=NS(act_fn=native_activation)),
                    AscendSharedSparseSituAndMul=sparse_factory,
                    FusedMoEFactory=moe_factory,
                )
                load("vllm_ascend/models/kimi_k3.py", {"__init__"}, ns, "AscendKimiMoE")
                config = NS(
                    hidden_size=8,
                    moe_intermediate_size=4,
                    num_experts=4,
                    num_experts_per_token=2,
                    num_shared_experts=1,
                    routed_expert_hidden_size=None,
                    latent_moe_use_norm=False,
                    routed_scaling_factor=1.0,
                    hidden_act=activation,
                    activation_situ_beta=BETA,
                    activation_situ_linear_beta=alpha,
                    moe_renormalize=True,
                    use_grouped_topk=False,
                    num_expert_group=None,
                    topk_group=None,
                    moe_router_activation_func="softmax",
                )
                layer = ns["AscendKimiMoE"](
                    config=config, vllm_config=NS(parallel_config=NS(enable_eplb=False))
                )
                self.assertEqual(init_ops.call_count, int(a5 and activation == "situ"))
                self.assertIs(layer.shared_experts, moe_factory.call_args.kwargs["shared_experts"])
                self.assertIs(layer.shared_experts.act_fn, sparse_activation if expect_sparse else native_activation)
                if expect_sparse:
                    sparse_factory.assert_called_once_with(BETA, ALPHA)
                else:
                    sparse_factory.assert_not_called()


class TestSharedSitu(unittest.TestCase):
    def test_cpu_fallback_and_nonpersistent_expert_counts(self):
        sparse_op = Mock()
        activation = load_shared_activation(dict(torch=TorchOpsStub(NS(npu_situ_and_mul_sparse=sparse_op))))
        x = torch.linspace(-7, 7, 24, dtype=torch.bfloat16).reshape(3, 8)
        torch.testing.assert_close(activation(x), ReferenceSitu.forward_native(activation, x))
        sparse_op.assert_not_called()
        self.assertEqual(activation._all_rows_expert_tokens.dtype, torch.int64)
        self.assertEqual(activation._all_rows_expert_tokens.tolist(), [torch.iinfo(torch.int64).max])
        self.assertNotIn("_all_rows_expert_tokens", activation.state_dict())

    def test_mxfp4_shared_uses_sparse_activation_or_native_quantized_pair(self):
        sparse_output, down_output, prepared_input = object(), object(), object()
        sparse_op = Mock(return_value=sparse_output)
        quantized_pair = (object(), object())
        native_op = Mock(return_value=quantized_pair)
        custom = NS(npu_situ_and_mul_sparse=sparse_op)
        ns = dict(torch=TorchOpsStub(custom, NS(situ_mx_quant=native_op)))
        activation = load_shared_activation(ns)
        load(SHARED, {"SITU_MX_DST_TYPE_E4M3FN"}, ns)
        load(SHARED, {"apply_activation", "_run_a8_mxfp_mlp", "part2", "_apply_output_gate"}, ns, "AscendSharedExperts")
        shared = ns["AscendSharedExperts"]()
        down_proj = Mock(return_value=(down_output, None))
        shared.layer = NS(act_fn=activation, down_proj=down_proj)
        shared.situ_activation = activation
        shared._prepare_linear_input_for_overlap = Mock(return_value=prepared_input)
        shared._wait_for_routed_stage = Mock()

        device = NS(type="npu", index=0)
        gate_up = NS(device=device, dtype=torch.bfloat16, ndim=2, shape=(3, 8))
        counts = NS(device=device)
        activation._buffers["_all_rows_expert_tokens"] = counts
        submitted = NS(gate_up=gate_up, hidden_states=object())
        milestones = NS(shared_activation_overlap_start=None, routed_combine_start=None)
        self.assertIs(shared._run_a8_mxfp_mlp(submitted, milestones), down_output)
        sparse_op.assert_called_once_with(gate_up, counts, beta=BETA, alpha=ALPHA, high_precision=False)
        native_op.assert_not_called()
        shared._prepare_linear_input_for_overlap.assert_called_once_with(down_proj, sparse_output)
        down_proj.assert_called_once_with(prepared_input)

        del custom.npu_situ_and_mul_sparse
        down_proj.reset_mock()
        shared._prepare_linear_input_for_overlap.reset_mock()
        self.assertIs(shared._run_a8_mxfp_mlp(submitted, milestones), down_output)
        native_op.assert_called_once_with(
            x=gate_up, beta=BETA, linear_beta=ALPHA, activate_left=True, dst_type=ns["SITU_MX_DST_TYPE_E4M3FN"]
        )
        down_proj.assert_called_once_with(quantized_pair)
        shared._prepare_linear_input_for_overlap.assert_not_called()


class TestRoutedSitu(unittest.TestCase):
    def test_unquantized_sparse_converts_group_counts_and_falls_back(self):
        x = torch.linspace(-7, 7, 24, dtype=torch.bfloat16).reshape(3, 8)
        sparse_output = object()
        sparse = Mock(return_value=sparse_output)
        custom = NS(npu_situ_and_mul_sparse=sparse)
        ns = dict(
            torch=TorchOpsStub(custom),
            MoEActivation=ACTIVATION,
            QuantType=QUANT_TYPE,
            HardwareCapability=CAPABILITY,
            get_current_hardware_profile=lambda: NS(supports=lambda capability: True),
        )
        load(MOE_UTILS, {"cumsum_group_list"}, ns)
        load(MOE_MLP, {"_unified_apply_activation", "_apply_situ"}, ns)
        apply = ns["_unified_apply_activation"]
        for group_type, group_list in ((0, [0, 2, 3]), (1, [0, 2, 1])):
            with self.subTest(group_type=group_type):
                inp = NS(
                    activation=ACTIVATION.SITU,
                    activation_situ_beta=BETA,
                    activation_situ_linear_beta=ALPHA,
                    quant=NS(quant_type=QUANT_TYPE.NONE),
                    group_list_type=group_type,
                    group_list=torch.tensor(group_list, dtype=torch.int32),
                )
                self.assertIs(apply(inp, x, None), sparse_output)
                gate_up, counts = sparse.call_args.args
                self.assertIs(gate_up, x)
                torch.testing.assert_close(counts, torch.tensor([0, 2, 1], dtype=torch.int64))
                self.assertEqual(sparse.call_args.kwargs, dict(beta=BETA, alpha=ALPHA, high_precision=True))

        del custom.npu_situ_and_mul_sparse
        torch.testing.assert_close(apply(inp, x, None), ReferenceSitu(BETA, ALPHA)(x))

    def test_mxfp4_grouped_situ_preserves_scale_layout_and_native_fused_fallback(self):
        gate_up = torch.zeros((3, 8), dtype=torch.bfloat16)
        scale = torch.arange(6, dtype=torch.uint8).reshape(3, 2)
        sparse_out, native_out = object(), object()
        grouped = Mock(return_value=(sparse_out, scale))
        native = Mock(return_value=(native_out, scale, None))
        gmm1 = Mock(return_value=[gate_up])
        custom = NS(grouped_situ_mx_quant=grouped)
        ns = dict(
            torch=TorchOpsStub(custom),
            torch_npu=NS(float8_e8m0fnu="e8m0", float4_e2m1fn_x2="fp4", npu_grouped_matmul=gmm1),
            DeviceOperator=NS(npu_grouped_matmul_situ_quant=native),
            MoEActivation=ACTIVATION,
            AscendMoEScheme=object,
            HardwareCapability=CAPABILITY,
            get_current_hardware_profile=lambda: NS(supports=lambda capability: True),
            dispose_tensor=Mock(),
        )
        load(MOE_UTILS, {"cumsum_group_list", "maybe_normalize_mxfp_scale_layout"}, ns)
        load(
            QUANT,
            {"_use_grouped_situ", "apply_gmm1_act_quant"},
            ns,
            "AscendW4A8MXFPDynamicFusedMoEMethod",
        )
        method = ns["AscendW4A8MXFPDynamicFusedMoEMethod"]()
        method.group_size = 32
        method.use_expert_weight_list = False
        method.quant_type = QUANT_TYPE.W4A8MXFP
        method._quant_hidden_states = lambda x, s: (x, s)
        method._get_mlp_weights = lambda layer: ([layer.w13_weight], [layer.w13_weight_scale], [], [])
        inp = NS(
            hidden_states=torch.zeros(3, 4),
            dynamic_scale=scale,
            layer=NS(w13_weight=object(), w13_weight_scale=object()),
            activation=ACTIVATION.SITU,
            group_list_type=0,
            group_list=torch.tensor([0, 2, 3], dtype=torch.int32),
            activation_situ_beta=BETA,
            activation_situ_linear_beta=ALPHA,
            lora_context=None,
        )
        out, out_scale = method.apply_gmm1_act_quant(inp)
        self.assertIs(out, sparse_out)
        self.assertIs(grouped.call_args.args[0], gate_up)
        torch.testing.assert_close(grouped.call_args.args[1], torch.tensor([0, 2, 1], dtype=torch.int64))
        torch.testing.assert_close(out_scale, scale.reshape(3, 1, 2))
        self.assertEqual(grouped.call_args.kwargs, dict(beta=BETA, alpha=ALPHA, high_precision=False))
        native.assert_not_called()
        gmm1.assert_called_once()

        del custom.grouped_situ_mx_quant
        gmm1.reset_mock()
        out, out_scale = method.apply_gmm1_act_quant(inp)
        self.assertIs(out, native_out)
        torch.testing.assert_close(out_scale, scale.reshape(3, 1, 2))
        self.assertIs(native.call_args.kwargs["group_list"], inp.group_list)
        self.assertEqual(native.call_args.kwargs["group_list_type"], inp.group_list_type)
        gmm1.assert_not_called()


if __name__ == "__main__":
    unittest.main()
