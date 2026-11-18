# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for storage, serial execution and caller input contracts."""

import ast
import hashlib
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, methods, ns, base=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [] if base is None else [ast.Name(id=base, ctx=ast.Load())]
    node.decorator_list = []
    node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


class TestKimiMigrationReview(unittest.TestCase):
    def test_model_uses_normalized_shared_situ_beta(self):
        cfg = NS(
            hidden_size=4,
            moe_intermediate_size=4,
            num_experts=2,
            num_experts_per_token=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            routed_scaling_factor=1.0,
            num_shared_experts=1,
            activation_situ_beta=None,
            activation_situ_linear_beta=25.0,
            hidden_act="situ",
            moe_renormalize=True,
            use_grouped_topk=False,
            num_expert_group=None,
            topk_group=None,
            moe_router_activation_func="softmax",
        )
        shared = torch.nn.Module()
        shared.act_fn = NS(beta=1.0, linear_beta=25.0)
        runner = torch.nn.Module()
        runner.moe_config = NS(num_logical_experts=2, num_experts=2, num_local_experts=2)
        sparse = Mock(side_effect=lambda beta, alpha: NS(beta=float(beta), linear_beta=alpha))
        ns = dict(
            torch=torch,
            nn=torch.nn,
            Module=torch.nn.Module,
            GateLinear=lambda **kw: torch.nn.Module(),
            KimiMLP=lambda **kw: shared,
            FusedMoEFactory=lambda **kw: runner,
            AscendSharedSparseSituAndMul=sparse,
            initialize_kimi_situ_ops=Mock(),
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            get_ascend_config=lambda: NS(multistream_overlap_shared_expert=False),
        )
        cls = load_class("vllm_ascend/models/kimi_k3.py", "AscendKimiMoE", {"__init__"}, ns, "Module")
        cls(config=cfg, vllm_config=NS(parallel_config=NS(enable_eplb=False)))
        sparse.assert_called_once_with(1.0, 25.0)
        ns["initialize_kimi_situ_ops"].assert_called_once_with()

    def test_mxfp_shared_keeps_sparse_activation_in_all_execution_modes(self):
        class Sparse:
            beta = 4.0
            linear_beta = 25.0

        fused = Mock(side_effect=AssertionError("Kimi sparse SiTU was bypassed by fused quantization"))
        ns = dict(
            AscendSharedSparseSituAndMul=Sparse,
            torch=NS(ops=NS(_C_ascend=NS(situ_mx_quant=fused))),
            SITU_MX_DST_TYPE_E4M3FN=36,
        )
        cls = load_class("vllm_ascend/ops/fused_moe/shared_experts.py", "AscendSharedExperts", {"_run_a8_mxfp_mlp"}, ns)
        shared = cls()
        shared.situ_activation = Sparse()
        activated = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        shared.apply_activation = Mock(return_value=activated)
        shared._wait_for_routed_stage = Mock()
        shared.part2 = Mock(return_value=activated + 1)
        submitted = NS(gate_up=torch.ones(1, 4, dtype=torch.bfloat16), hidden_states=torch.ones(1, 2))
        for mode in ("prefill", "serial", "ordinary_decode"):
            with self.subTest(mode=mode):
                milestones = NS(shared_activation_overlap_start=object(), routed_combine_start=object())
                output = shared._run_a8_mxfp_mlp(submitted, milestones)
                torch.testing.assert_close(output, activated + 1)
                self.assertIs(shared.part2.call_args.args[1], activated)
        self.assertEqual(shared.apply_activation.call_count, 3)
        fused.assert_not_called()

    def test_preprojected_front_does_not_project_again_or_record_serial_events(self):
        projection = Mock(side_effect=lambda x: (x[:, :2] * 2, x))

        class Base:
            def apply_routed_input_transform(self, x):
                return projection(x)

        event_access = Mock(side_effect=AssertionError("serial router recorded an event"))
        ns = dict(
            torch=NS(npu=NS(current_stream=event_access)),
            Base=Base,
            nullcontext=nullcontext,
            npu_stream_switch=lambda *a, **kw: nullcontext(),
            PreparedSharedExpertInput=lambda x: NS(hidden_states=x, ready_event=None),
        )
        cls = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"apply_routed_input_transform"}, ns, "Base"
        )
        runner = cls()
        runner.ascend_shared_experts = None
        runner._kimi_gate = object()
        runner._kimi_router_stream = None
        runner.routed_input_transform = object()
        runner._can_overlap_sp_shared_with = lambda t: False
        runner._can_run_kimi_decode = lambda x: False
        runner._compute_router_logits = Mock(return_value=torch.ones(2, 8))
        original = torch.ones(2, 4)
        projected = torch.full((2, 2), 3.0)
        actual, shared = runner.apply_routed_input_transform(original, routed_input=projected)
        self.assertIs(actual, projected)
        self.assertIs(shared, original)
        projection.assert_not_called()
        runner._compute_router_logits.assert_called_once_with(original, original)
        event_access.assert_not_called()

    def test_serial_shared_forward_never_acquires_auxiliary_stream(self):
        stream = Mock(side_effect=AssertionError("serial execution acquired an auxiliary stream"))
        modes = NS(TENSOR_PARALLEL=0, SHARED_EXPERT_DATA_PARALLEL_ONLY=1, SEQUENCE_PARALLEL_ONLY=2)
        paths = NS(A8_INT_FUSED=0, A8_MXFP_FUSED=1, LINEAR_WRAPPER=2)
        ns = dict(
            nullcontext=nullcontext,
            shared_experts_calculation_stream=stream,
            npu_stream_switch=lambda *a, **kw: nullcontext(),
            SharedExpertParallelMode=modes,
            SharedExpertMLPPath=paths,
            PreparedSharedExpertExecution=NS,
            SubmittedSharedExpertGateUp=NS,
        )
        cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "AscendSharedExperts",
            {
                "forward",
                "enqueue_gate_up_after_router",
                "_wait_for_milestone",
                "_compute_stream_context",
                "wait_for_output",
            },
            ns,
        )
        shared = cls()
        shared.multistream_overlap = False
        shared.parallel_mode = lambda: modes.TENSOR_PARALLEL
        shared._select_mlp_path = lambda: paths.LINEAR_WRAPPER
        x = torch.tensor([[1.0, 2.0]])
        shared._prepare_execution_input = lambda *a: (x, None)
        shared._prepare_gate_up_input = lambda x: x
        shared.part1 = lambda x: x * 2
        shared._finish_shared_mlp = lambda submitted, milestones: submitted.gate_up + 1
        output = shared.forward(NS(hidden_states=x), NS(router_output_ready=None))
        torch.testing.assert_close(output, x * 2 + 1)
        shared.wait_for_output()
        stream.assert_not_called()

    def test_public_moe_op_keeps_preprojected_input(self):
        op = Mock(return_value=torch.zeros(2, 4))
        ns = dict(torch=NS(ops=NS(vllm=NS(ascend_kimi_moe=op))))
        cls = load_class("vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"forward"}, ns)
        runner = cls()
        runner._kimi_gate = object()
        runner._encode_layer_name = lambda: "layer"
        projected = torch.ones(2, 2)
        original = torch.ones(2, 4)
        scores = torch.ones(2, 8)
        runner.forward(projected, scores, shared_experts_input=original)
        self.assertTrue(
            any(arg is original for arg in op.call_args.args), "the opaque operation dropped the original shared input"
        )
        self.assertTrue(any(arg is scores for arg in op.call_args.args))

    def test_both_runners_admit_decode_and_reject_unsupported_schemes(self):
        from unittest.mock import patch

        class Routed:
            pass

        class Linear:
            pass

        class Latent:
            pass

        class Router:
            pass

        modules = {
            "vllm_ascend.ops.fused_moe.router.fused_topk_router": NS(AscendFusedTopKRouter=Router),
            "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4": NS(
                AscendW4A8MXFPDynamicFusedMoEMethod=Routed, AscendW4A8MXFPDynamicLinearMethod=Linear
            ),
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8": NS(AscendW8A8MXFP8DynamicLinearMethod=Latent),
        }
        context = NS(attn_metadata={"layer": NS(num_prefills=0)})
        ns = dict(
            torch=torch,
            get_forward_context=lambda: context,
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            _EXTRA_CTX=NS(moe_comm_type="mc2", is_draft_model=False),
            MoECommType=NS(MC2="mc2"),
            QuantType=NS(W4A8MXFP="fp4"),
        )
        cls = load_class("vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"_can_run_kimi_decode"}, ns)
        runner = cls()
        runner.quant_type = "fp4"
        runner.moe_config = NS(is_sequence_parallel=True)
        runner._can_overlap_sp_shared_with = lambda transform: True
        runner.routed_input_transform = NS(quant_method=NS(quant_method=Latent()))
        runner.routed_output_transform = object()
        router = Router()
        router.custom_routing_function = None
        router.tid2eid = None
        router.bias_vl = None
        scheme = Routed()
        scheme.use_expert_weight_list = False
        runner.routed_experts = NS(
            quant_method=NS(quant_method=scheme), router=router, dynamic_eplb=False, _use_v2_model_runner=False
        )
        layer = NS(
            gate_up_proj=NS(weight=torch.ones(2, 2, dtype=torch.bfloat16)),
            down_proj=NS(weight=torch.ones(2, 2, dtype=torch.bfloat16)),
        )
        runner.ascend_shared_experts = NS(layer=layer, situ_activation=object())
        x = torch.ones(2, 2, dtype=torch.bfloat16)
        with patch.dict("sys.modules", modules):
            for runner_v2 in (False, True):
                runner.routed_experts._use_v2_model_runner = runner_v2
                self.assertTrue(runner._can_run_kimi_decode(x))
                # Hybrid KDA/MLA metadata and MTP verification are Decode
                # when every attention group reports zero prefill requests.
                context.attn_metadata = {"kda": NS(num_prefills=0), "mla": NS(num_prefills=0)}
                self.assertTrue(runner._can_run_kimi_decode(x))
                for metadata in (
                    None,
                    {},
                    {"layer": NS()},
                    {"layer": NS(num_prefills=1)},
                    {"kda": NS(num_prefills=0), "mla": NS(num_prefills=1)},
                ):
                    context.attn_metadata = metadata
                    self.assertFalse(runner._can_run_kimi_decode(x), (runner_v2, metadata))
                context.attn_metadata = {"layer": NS(num_prefills=0)}
            for obj, attribute, value in [
                (router, "bias_vl", object()),
                (ns["_EXTRA_CTX"], "is_draft_model", True),
                (runner.moe_config, "is_sequence_parallel", False),
                (runner.routed_experts, "dynamic_eplb", True),
                (ns["_EXTRA_CTX"], "moe_comm_type", "alltoall"),
                (scheme, "use_expert_weight_list", True),
                (router, "custom_routing_function", lambda: None),
                (router, "tid2eid", object()),
                (layer, "expert_gate", object()),
                (runner.routed_experts, "_ascend_moe_lora_context", object()),
            ]:
                old = getattr(obj, attribute, None)
                setattr(obj, attribute, value)
                self.assertFalse(runner._can_run_kimi_decode(x), attribute)
                setattr(obj, attribute, old)
            runner.routed_input_transform.quant_method.quant_method = object()
            self.assertFalse(runner._can_run_kimi_decode(x))


NATIVE_F89_METHOD_HASHES = (
    ("AscendKimiDecoderLayer", "forward", "8f6e678b1ac8bbd0dd4b6e988890bcec41bf7004db173475ffde5ac94a8fce82"),
    (
        "AscendKimiDecoderLayer",
        "prepare_attn_residual",
        "ac3d38df6a9f1c0e681e13f2369d82060a8a4accdb1f0fea52e627d1286fc239",
    ),
    (
        "AscendKimiDecoderLayer",
        "forward_attn_residual",
        "10e88407699f1fe63b142f79a75ee3b414165933c7db3eeeb8a6e79137f63eb4",
    ),
    ("AscendKimiLinearModel", "__init__", "403bff0349be27c8a19ec723625019770e810830c7d43db1744f733554b43d63"),
    ("AscendKimiLinearModel", "load_weights", "ab16d00c4233c4ff5ef007d3b5e5f862d088de14e1fd735ef9446a186c04e5e5"),
    ("AscendKimiLinearModel", "forward", "2d0856dc1f8eb03076da728f22611c05ea1c45078840eeb4444a31f901a96d8d"),
)


class TestKimiNativeResidual(unittest.TestCase):
    def test_native_methods_match_frozen_f89_ast(self):
        # Frozen from f89e5592d6fbcb36ee8ce0e538d23f0dd8f951b8. Empty
        # type_params fields are ignored for Python 3.11/3.13 AST parity.
        expected = NATIVE_F89_METHOD_HASHES
        tree = ast.parse((ROOT / "vllm_ascend/models/kimi_k3.py").read_text())
        for owner, name, digest in expected:
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for node in ast.walk(fn):
                node._fields = tuple(field for field in node._fields if field != "type_params")
            actual = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual, digest, f"{owner}.{name}")

    def test_removed_backend_and_compile_state_are_absent(self):
        model = (ROOT / "vllm_ascend/models/kimi_k3.py").read_text()
        for name in (
            "support_torch_compile",
            "AttnResBackend",
            "AttnResQueryPlan",
            "load_attn_res_fused_ops",
            "resolve_attn_res_mode",
            "attn_res_mode",
            "attn_res_query_plan",
            "attn_res_fused_ops",
            "attn_res_op",
            "VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED",
        ):
            self.assertNotIn(name, model)
        tree = ast.parse(model)
        self.assertFalse(
            any(
                (isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.ops.kimi_attn_res")
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name == "vllm_ascend.ops.kimi_attn_res" for alias in node.names)
                )
                for node in ast.walk(tree)
            )
        )
        self.assertIn("torch.ops._C_ascend.attn_res_fwd", model)

    def test_native_prepare_preserves_operator_arguments_and_write_flags(self):
        for write in (False, True):
            op = Mock(return_value=(object(), object(), object()))
            ns = dict(torch=NS(ops=NS(_C_ascend=NS(attn_res_fwd=op))))
            cls = load_class("vllm_ascend/models/kimi_k3.py", "AscendKimiDecoderLayer", {"prepare_attn_residual"}, ns)
            layer = cls()
            layer.self_attention_res_proj = NS(weight=object())
            layer.self_attention_res_norm = NS(weight=object(), variance_epsilon=1e-6)
            layer.input_layernorm = NS(weight=object(), variance_epsilon=1e-5)
            layer.prev_valid_blocks, layer.block_write_idx, layer.is_block_write_layer = 3, 3, write
            prefix, addend, bank = object(), object(), object()
            result = layer.prepare_attn_residual(prefix, bank, addend, return_materialized=True, optimize_prefill=True)
            self.assertIs(result, op.return_value)
            args = op.call_args.args
            for actual, expected in zip(
                args[:5],
                (prefix, addend, bank, layer.self_attention_res_proj.weight, layer.self_attention_res_norm.weight),
            ):
                self.assertIs(actual, expected)
            self.assertEqual(args[5:7], (1e-6, 3))
            self.assertIs(args[7], layer.input_layernorm.weight)
            self.assertEqual(args[8:], (1e-5, 3 if write else -1, True))
            self.assertEqual(op.call_args.kwargs, {"optimize_prefill": True})

    def test_native_decoder_preserves_deferred_and_completed_mlp_add(self):
        for write in (False, True):
            for deferred in (False, True):
                with self.subTest(write=write, deferred=deferred):

                    def native(prefix, addend, *args, **kwargs):
                        if kwargs.get("mix") is False:
                            return prefix + addend, prefix + addend, None
                        return torch.full((2, 1), 4.0), torch.full((2, 1), 9.0), None

                    op = Mock(side_effect=native)
                    cls = load_class(
                        "vllm_ascend/models/kimi_k3.py",
                        "AscendKimiDecoderLayer",
                        {"forward_attn_residual"},
                        dict(torch=NS(ops=NS(_C_ascend=NS(attn_res_fwd=op)))),
                    )
                    layer = cls()
                    layer.is_block_write_layer, layer.use_sequence_parallel, layer.prev_valid_blocks = write, False, 3
                    layer.self_attn = lambda **kwargs: torch.full((2, 1), 2.0)
                    layer.mlp = lambda x: x + 3
                    layer.mlp_res_proj = NS(weight=object())
                    layer.mlp_res_norm = NS(weight=object(), variance_epsilon=1e-6)
                    layer.post_attention_layernorm = NS(weight=object(), variance_epsilon=1e-5)
                    prefix, bank = torch.full((2, 1), 7.0), torch.zeros(2, 4, 1)
                    result = layer.forward_attn_residual(
                        torch.arange(2),
                        prefix,
                        bank,
                        prepared_attn_input=(prefix, prefix, None),
                        defer_mlp_add=deferred,
                        optimize_prefill=True,
                    )
                    self.assertIs(result[1], bank)
                    self.assertEqual(op.call_count, 1 if deferred else 2)
                    first = op.call_args_list[0]
                    self.assertEqual(first.args[6], 3 + int(write))
                    self.assertEqual(first.kwargs, {"optimize_prefill": True})
                    if write:
                        self.assertIsNone(first.args[1])
                    else:
                        self.assertIs(first.args[0], prefix)
                    torch.testing.assert_close(result[0], torch.full((2, 1), 9.0 if deferred else 16.0))
                    if deferred:
                        torch.testing.assert_close(result[2], torch.full((2, 1), 7.0))
                    else:
                        self.assertEqual(op.call_args.kwargs, {"mix": False, "optimize_prefill": True})

    def test_native_model_preserves_pp_and_dspark_state_boundaries(self):
        for first in (False, True):
            for last in (False, True):
                for materialized in (False, True):
                    with self.subTest(first=first, last=last, materialized=materialized):
                        state = NS(is_first_rank=first, is_last_rank=last)
                        base_value = 1.0 if first else 7.0
                        incoming_bank = torch.full((2, 1, 1), 5.0)
                        prepared = []

                        class Layer:
                            def prepare_attn_residual(self, prefix, bank, addend, _prepared=prepared, **kwargs):
                                _prepared.append((bank, addend, kwargs))
                                return prefix + 100, prefix, prefix + 100

                            def __call__(self, **kwargs):
                                self_input = kwargs["prepared_attn_input"][1]
                                return self_input + 10, kwargs["residual"], torch.full_like(self_input, 3)

                        def native(prefix, addend, bank, *args, **kwargs):
                            raw = prefix + addend
                            return raw + 1000, raw, raw + 1000

                        op = Mock(side_effect=native)
                        ns = dict(
                            torch=NS(ops=NS(_C_ascend=NS(attn_res_fwd=op))),
                            get_pp_group=lambda _state=state: _state,
                            PPTransportDataType=NS(AUX_HIDDEN_STATES="aux"),
                            get_pp_transport_tensors=lambda tensors, key: (
                                [] if tensors is None else list(tensors["aux"])
                            ),
                            add_pp_transport_tensors=lambda tensors, key, aux: dict(tensors, aux=aux),
                            IntermediateTensors=dict,
                            cdiv=lambda a, b: (a + b - 1) // b,
                            _use_attn_res_prefill_cache=lambda: True,
                        )
                        cls = load_class("vllm_ascend/models/kimi_k3.py", "AscendKimiLinearModel", {"forward"}, ns)
                        model = cls()
                        model.config = NS(attn_res_block_size=1)
                        model.start_layer, model.end_layer = (0, 1) if first else (1, 2)
                        model.layers = [Layer()] if first else [None, Layer()]
                        model.use_sequence_parallel = False
                        model.dspark_aux_capture_materialized = materialized
                        model.aux_hidden_state_layers = {model.start_layer, model.end_layer}
                        model.output_attn_res_proj = NS(weight=object())
                        model.output_attn_res_norm = NS(weight=object(), variance_epsilon=1e-6)
                        model._maybe_add_hidden_state = lambda aux, index, x, residual: aux.append(x.clone()) or aux
                        incoming = (
                            None
                            if first
                            else {
                                "hidden_states": torch.full((2, 1), base_value),
                                "residual": incoming_bank,
                                "aux": [torch.full((2, 1), 9.0)],
                            }
                        )
                        result = model.forward(None, torch.arange(2), incoming, torch.full((2, 1), base_value))
                        bank, addend, flags = prepared[0]
                        self.assertIsNone(addend)
                        self.assertEqual(flags, {"return_materialized": materialized, "optimize_prefill": True})
                        self.assertEqual(bank.shape, (2, model.end_layer, 1))
                        if not first:
                            torch.testing.assert_close(bank[:, :1], incoming_bank)
                        if last:
                            output, aux = result
                            torch.testing.assert_close(output, torch.full((2, 1), base_value + 1013))
                            self.assertEqual(op.call_args.args[6], model.end_layer)
                            self.assertEqual(op.call_args.kwargs, {"optimize_prefill": True})
                        else:
                            aux = result["aux"]
                            torch.testing.assert_close(result["hidden_states"], torch.full((2, 1), base_value + 13))
                            self.assertIs(result["residual"], bank)
                            op.assert_not_called()
                        expected_aux = [] if first else [9.0]
                        if materialized:
                            expected_aux.append(base_value + 100)
                            if last:
                                expected_aux.append(base_value + 1013)
                        else:
                            if first:
                                expected_aux.append(base_value)
                            expected_aux.append(base_value + 13)
                        self.assertEqual(len(aux), len(expected_aux))
                        for actual, expected in zip(aux, expected_aux):
                            torch.testing.assert_close(actual, torch.full((2, 1), expected))

    def test_native_reload_keeps_gate_mapping_without_derived_state(self):
        class Base:
            def load_weights(self, weights):
                self.loaded = list(weights)
                return {args[0] for args in self.loaded}

        eplb = Mock(side_effect=lambda model, weights, redundant: list(weights))
        cls = load_class(
            "vllm_ascend/models/kimi_k3.py",
            "AscendKimiLinearModel",
            {"load_weights"},
            dict(Base=Base, load_eplb_expert_weights=eplb),
            "Base",
        )
        model = cls()
        prefix = "layers.0.attn."
        targets = [
            prefix + "fused_bfg_proj.weight",
            prefix + "fused_bfg_proj.f_a_weight",
            prefix + "fused_bfg_proj.f_b_weight",
        ]
        model.named_parameters = lambda: [(name, object()) for name in targets]
        model.n_redundant_experts = 2
        for value in (1.0, 2.0):
            sources = [prefix + name + ".weight" for name in ("b_proj", "f_a_proj", "f_b_proj", "g_proj")]
            tensors = [torch.tensor([value]) for name in sources]
            weights = [(name, tensor, {"tag": "checkpoint"}) for name, tensor in zip(sources, tensors)]
            result = model.load_weights(weights)
            self.assertEqual(result, set(targets))
            for index, args in enumerate(model.loaded):
                self.assertIs(args[1], tensors[index])
                self.assertEqual(args[2], {"tag": "checkpoint", "loaded_shard_id": (0, None, None, 2)[index]})
            self.assertTrue(all(args[2] == {"tag": "checkpoint"} for args in weights))
            self.assertEqual(eplb.call_args.args[2], 2)
            self.assertEqual(set(model.__dict__), {"named_parameters", "n_redundant_experts", "loaded"})

    def test_native_prefill_cache_keeps_metadata_and_device_guards(self):
        path = ROOT / "vllm_ascend/models/kimi_k3.py"
        fn = next(
            node
            for node in ast.parse(path.read_text()).body
            if getattr(node, "name", None) == "_use_attn_res_prefill_cache"
        )
        context = NS(attn_metadata={})
        ns = dict(is_forward_context_available=lambda: True, get_forward_context=lambda: context, is_950=lambda: True)
        module = ast.Module(body=[fn], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
        select = ns["_use_attn_res_prefill_cache"]
        for metadata in (
            None,
            {},
            {"layer": NS()},
            {"layer": NS(num_prefills=0, num_decodes=1)},
            {"layer": NS(num_prefills=True, num_decodes=0)},
            {"layer": NS(num_prefills=1, num_decodes=0, spec_sequence_masks=object())},
        ):
            context.attn_metadata = metadata
            self.assertFalse(select())
        context.attn_metadata = {"kda": NS(num_prefills=1, num_decodes=0), "mla": NS(num_prefills=2, num_decodes=0)}
        self.assertTrue(select())
        ns["is_950"] = lambda: False
        self.assertFalse(select())
        ns["is_950"] = lambda: True
        ns["is_forward_context_available"] = lambda: False
        self.assertFalse(select())


if __name__ == "__main__":
    unittest.main()
