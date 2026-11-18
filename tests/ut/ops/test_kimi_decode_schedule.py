# SPDX-License-Identifier: Apache-2.0
"""CPU-only dependency tests; runnable directly without torch or NPU packages.

The device simulator checks event edges, not NPU latency or numerical accuracy.
Runner methods are compiled from their source to isolate them from plugin loading.
"""

import ast
import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class Stream:
    def __init__(self, device, name):
        self.device = device
        self.name = name
        self.tail = None

    def record_event(self):
        return self.device.enqueue(self.name, "record")

    def wait_event(self, event):
        if event not in self.device.edges:
            raise AssertionError("wait before event record")
        return self.device.enqueue(self.name, "wait", event)


class Device:
    def __init__(self):
        self.edges = {}
        self.labels = {}
        self.streams = {name: Stream(self, name) for name in ("main", "router", "shared", "comm")}
        self.current = self.streams["main"]
        self.limits = []
        self.active_limits = {}
        self.op_limits = {}

    def enqueue(self, stream, label, dependency=None):
        index = len(self.edges)
        tail = self.streams[stream].tail
        self.edges[index] = {d for d in (tail, dependency) if d is not None}
        self.labels[index] = (stream, label)
        self.streams[stream].tail = index
        return index

    def op(self, name):
        self.op_limits[name] = self.active_limits.get(self.current.name)
        return self.enqueue(self.current.name, name)

    def precedes(self, first, second):
        return first in self.edges[second] or any(self.precedes(first, dep) for dep in self.edges[second])

    def named(self, name):
        return next(index for index, (_, label) in self.labels.items() if label == name)

    @contextmanager
    def switch(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    @contextmanager
    def core_limit(self, *limit):
        stream = self.current.name
        self.limits.append((stream, limit))
        previous = self.active_limits.get(stream)
        self.active_limits[stream] = limit
        try:
            yield
        finally:
            self.active_limits[stream] = previous


class Tensor:
    def __init__(self, value=1, shape=(2, 4)):
        self.value = value
        self.shape = shape
        self.dtype = "bf16"
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream.name)

    def new_empty(self, shape):
        return Tensor(shape=shape)

    def __getitem__(self, index):
        return self


def runner_methods(names, namespace, filename="fused_moe.py", class_name="AscendMoERunner"):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe" / filename
    tree = ast.parse(source.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    runner.body = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    runner.bases = []
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), runner]
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    return namespace[class_name]


class TestKimiDecodeSchedule(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            float8_e4m3fn="fp8",
            npu=SimpleNamespace(current_stream=lambda: self.device.current),
        )
        self.torch_npu = SimpleNamespace(
            npu=SimpleNamespace(
                npugraph_ex=SimpleNamespace(
                    scope=SimpleNamespace(limit_core_num=lambda *limit: self.device.core_limit(*limit))
                )
            )
        )
        utils = SimpleNamespace(
            npu_stream_switch=lambda stream: self.device.switch(stream),
            shared_experts_calculation_stream=lambda: self.device.streams["shared"],
            shared_experts_comm_stream=lambda: self.device.streams["comm"],
        )
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/decode_overlap.py"
        spec = importlib.util.spec_from_file_location("kimi_schedule_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "torch_npu": self.torch_npu,
                "vllm_ascend.utils": utils,
                spec.name: module,
            },
        ):
            spec.loader.exec_module(module)
        self.schedule_class = module.DecodeSharedExpertSchedule

    def test_ladder_dependencies_and_values_across_calls(self):
        for limit in ((32, 1), (16, 32)):
            for _ in range(2):
                self.device = Device()
                self.run_ladder(limit)

    def test_mxfp4_shared_waits_before_gmm1_and_preserves_core_scopes(self):
        for limit in ((32, 1), (16, 32)):
            for _ in range(2):
                self.device = Device()
                self.run_ladder(limit, quantized_shared=True)

    def run_ladder(self, limit, quantized_shared=False):
        input_scale = object()

        def quantize(x, *, dst_type):
            self.assertEqual(dst_type, self.torch.float8_e4m3fn)
            self.device.op("shared input quant")
            return x, input_scale

        def gate_up(x):
            if quantized_shared:
                x, scale = x
                self.assertIs(scale, input_scale)
            self.device.op("shared gate/up")
            return Tensor(x.value * 2)

        def activate(x, *, force_sparse=False):
            self.assertEqual(force_sparse, quantized_shared)
            self.device.op("shared SiTU")
            return Tensor(x.value + 3)

        def down(x):
            if quantized_shared:
                self.device.op("shared down quant")
            self.device.op("shared down")
            return Tensor(x.value * 5), None

        def scatter(x):
            self.device.op("shared RS")
            return Tensor(x.value / 2)

        shared = SimpleNamespace(
            part1=gate_up,
            _activate_situ=activate,
            layer=SimpleNamespace(down_proj=down),
            _pad_and_reduce_scatter=scatter,
            _gate_up_core_scope=lambda: self.device.core_limit(16, 32),
        )
        self.torch_npu.npu_dynamic_mx_quant = MagicMock(side_effect=quantize)
        main = self.device.current
        ready = main.record_event()
        with self.device.switch(self.device.streams["comm"]):
            self.device.current.wait_event(ready)
            self.device.op("AG")
            gathered = self.device.current.record_event()
        schedule = self.schedule_class(shared, Tensor(7), gathered, limit, quantized_shared=quantized_shared)
        before = main.record_event()
        self.device.op("dispatch")
        after = main.record_event()
        schedule.after_dispatch(before, after)
        # An existing routed core scope must survive the shared-stage switch.
        with self.device.core_limit(32, 64):
            self.device.op("GMM1")
        schedule.after_gmm1(main.record_event())
        self.device.op("routed activation")
        self.device.op("GMM2")
        self.device.op("combine/up")
        output = schedule.join()
        self.device.op("add")
        self.assertEqual(output.value, (7 * 2 + 3) * 5 / 2)
        self.assertIn("main", output.streams)
        expected_limit = (16, 32) if quantized_shared else limit
        self.assertEqual(self.device.op_limits["shared gate/up"], expected_limit)
        self.assertIsNone(self.device.op_limits["shared SiTU"])
        self.assertEqual(self.device.op_limits["GMM1"], (32, 64))
        self.assertIsNone(self.device.op_limits["shared down"])
        if quantized_shared:
            self.torch_npu.npu_dynamic_mx_quant.assert_called_once()
            self.assertIsNone(self.device.op_limits["shared input quant"])
            self.assertIsNone(self.device.op_limits["shared down quant"])
            self.assertTrue(self.device.precedes(self.device.named("AG"), self.device.named("shared input quant")))
            self.assertTrue(self.device.precedes(self.device.named("GMM1"), self.device.named("shared down quant")))
        else:
            self.torch_npu.npu_dynamic_mx_quant.assert_not_called()
        for first, second in (
            ("AG", "shared gate/up"),
            ("dispatch", "shared SiTU"),
            ("shared SiTU", "GMM1"),
            ("GMM1", "shared down"),
            ("shared down", "shared RS"),
            ("shared RS", "add"),
            ("combine/up", "add"),
        ):
            self.assertTrue(self.device.precedes(self.device.named(first), self.device.named(second)), (first, second))
        for first, second in (
            ("shared gate/up", "dispatch"),
            ("shared down", "routed activation"),
            ("shared RS", "GMM2"),
        ):
            self.assertFalse(self.device.precedes(self.device.named(first), self.device.named(second)))
            self.assertFalse(self.device.precedes(self.device.named(second), self.device.named(first)))

    def test_missing_stages_fail_before_waiting(self):
        schedule = self.schedule_class(None, Tensor(), self.device.current.record_event(), (32, 1))
        with self.assertRaisesRegex(RuntimeError, "Shared SiTU"):
            schedule.after_gmm1(self.device.current.record_event())
        with self.assertRaisesRegex(RuntimeError, "did not release"):
            schedule.join()

    def test_comm_pipeline_submits_shared_stage_before_mlp(self):
        for name in ("float32", "float16", "int8", "float8_e4m3fn", "uint8"):
            setattr(self.torch, name, name)
        order = []
        schedule = SimpleNamespace(
            after_dispatch=lambda *args: order.append("shared_situ"),
            after_gmm1=lambda event: order.append("shared_down"),
        )

        def dispatch(**kwargs):
            order.append("dispatch")
            return SimpleNamespace(combine_metadata=None, group_list_type=1, group_list=None)

        def mlp(*, mlp_compute_input, after_gmm1):
            order.append("gmm1")
            after_gmm1(self.device.current.record_event())
            order.extend(("routed_activation", "gmm2"))
            return Tensor(), None, None

        def combine(**kwargs):
            order.append("combine")
            return kwargs["hidden_states"]

        namespace = dict(
            torch=self.torch,
            _EXTRA_CTX=SimpleNamespace(moe_comm_method=object()),
            build_token_dispatch_input=lambda **kw: kw,
            TokenDispatcherWithMC2=type("MC2", (), {}),
            build_mlp_compute_input=lambda **kw: kw,
            unified_apply_mlp=mlp,
            FusedExpertsResult=SimpleNamespace,
        )
        comm = runner_methods({"fused_experts"}, namespace, "moe_comm_method.py", "MoECommMethod")()
        comm.moe_config = object()
        comm.token_dispatcher = SimpleNamespace(token_dispatch=dispatch, token_combine=combine)
        comm.fused_experts(SimpleNamespace(hidden_states=Tensor()), shared_expert_schedule=schedule)
        self.assertEqual(
            order, ["dispatch", "shared_situ", "gmm1", "shared_down", "routed_activation", "gmm2", "combine"]
        )

    def test_front_launches_gate_topk_and_quant_before_down(self):
        device = self.device
        namespace = dict(
            torch=self.torch,
            torch_npu=self.torch_npu,
            ROUTER_CORE_LIMIT=(32, 1),
            npu_stream_switch=device.switch,
            router_stream=lambda: device.streams["router"],
            _EXTRA_CTX=SimpleNamespace(
                padded_num_tokens=8,
                in_profile_run=False,
                moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(tp_size=2)),
            ),
        )
        runner = runner_methods({"_decode_kimi_front"}, namespace)()
        original = Tensor(11, shape=(3, 4))

        def gather(x):
            ready = device.current.record_event()
            with device.switch(device.streams["comm"]):
                device.current.wait_event(ready)
                device.op("AG")
                return Tensor(x.value, (6, 4)), device.current.record_event()

        def gate(x):
            self.assertIs(x, original)
            device.op("gate")
            return Tensor(2, (3, 9)), None

        def topk(metadata, logits, pad, ids, force):
            self.assertEqual(metadata.shape, (4, 2))
            self.assertEqual(pad, 1)
            device.op("topk")
            return Tensor(), Tensor(), device.current.record_event()

        def quant(x):
            self.assertIs(x, original)
            device.op("quant")
            return x, "scale"

        def project(x):
            self.assertEqual(x, (original, "scale"))
            device.op("down")
            return Tensor(original.value * 7, (3, 2)), None

        runner.ascend_shared_experts = SimpleNamespace(start_input_all_gather=gather)
        runner._kimi_gate = gate
        runner.moe_config = SimpleNamespace(hidden_dim=2)
        runner.routed_experts = SimpleNamespace(_select_decode_experts=topk)
        runner.routed_input_transform = MagicMock(side_effect=project)
        runner.routed_input_transform.quant_method.quant_method = SimpleNamespace(quantize_input=quant)
        projected, _, _, _, _ = runner._decode_kimi_front(original, None)
        self.assertEqual(projected.value, 77)
        runner.routed_input_transform.assert_called_once()
        self.assertTrue(device.precedes(device.named("gate"), device.named("down")))
        self.assertTrue(device.precedes(device.named("quant"), device.named("down")))
        self.assertFalse(device.precedes(device.named("gate"), device.named("quant")))
        self.assertFalse(device.precedes(device.named("topk"), device.named("down")))
        self.assertFalse(device.precedes(device.named("down"), device.named("topk")))
        self.assertFalse(device.precedes(device.named("AG"), device.named("down")))

    def test_decode_guard_rejects_prefill_and_unsupported_paths(self):
        down_type = type("DownScheme", (), {})
        routed_type = type("RoutedScheme", (), {})
        shared_type = type("SharedScheme", (), {})
        router_type = type("Router", (), {})
        metadata = SimpleNamespace(num_prefills=0)
        ctx = SimpleNamespace(attn_metadata={"layer": metadata})
        comm = SimpleNamespace(MC2="mc2")
        extra = SimpleNamespace(moe_comm_type="mc2")
        namespace = dict(
            get_forward_context=lambda: ctx,
            _EXTRA_CTX=extra,
            MoECommType=comm,
            torch=self.torch,
            get_ascend_device_type=lambda: "a5",
            AscendDeviceType=SimpleNamespace(A5="a5"),
            AscendW8A8MXFP8DynamicLinearMethod=down_type,
            AscendW4A8MXFPDynamicFusedMoEMethod=routed_type,
            AscendW4A8MXFPDynamicLinearMethod=shared_type,
            AscendFusedTopKRouter=router_type,
        )
        runner = runner_methods({"_can_run_decode_ladder"}, namespace)()
        runner._kimi_gate_multistream = True
        runner.moe_config = SimpleNamespace(is_sequence_parallel=True, activation="situ")
        runner.routed_input_transform = SimpleNamespace(quant_method=SimpleNamespace(quant_method=down_type()))
        router = router_type()
        router.custom_routing_function = None
        router.tid2eid = None
        runner.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=routed_type()), dynamic_eplb=False, router=router
        )
        up = SimpleNamespace(weight=Tensor())
        runner.ascend_shared_experts = SimpleNamespace(
            layer=SimpleNamespace(gate_up_proj=up, down_proj=SimpleNamespace(weight=Tensor())), situ_activation=object()
        )
        runner._can_overlap_sp_shared_with = lambda _: True
        self.assertTrue(runner._can_run_decode_ladder(Tensor()))
        for obj, attribute, invalid in (
            (metadata, "num_prefills", 1),
            (extra, "moe_comm_type", "alltoall"),
            (runner, "_kimi_gate_multistream", False),
            (runner.routed_experts, "dynamic_eplb", True),
            (router, "custom_routing_function", object()),
        ):
            old = getattr(obj, attribute)
            setattr(obj, attribute, invalid)
            self.assertFalse(runner._can_run_decode_ladder(Tensor()), attribute)
            setattr(obj, attribute, old)
        up.weight_scale = Tensor()
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))
        del up.weight_scale
        # The profiling's FP4 shared weights must reach the staged path too.
        down = runner.ascend_shared_experts.layer.down_proj
        for linear in (up, down):
            linear.weight.dtype = "uint8"
            linear.weight_scale = Tensor()
            linear.quant_method = SimpleNamespace(quant_method=shared_type())
        self.assertTrue(runner._can_run_decode_ladder(Tensor()))
        metadata.num_prefills = 1
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))
        metadata.num_prefills = 0
        down.quant_method.quant_method = object()
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))
        down.quant_method.quant_method = shared_type()
        del down.weight_scale
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))
        down.weight_scale = Tensor()
        ctx.attn_metadata = None
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))
        runner.ascend_shared_experts = None
        self.assertFalse(runner._can_run_decode_ladder(Tensor()))


if __name__ == "__main__":
    unittest.main()
