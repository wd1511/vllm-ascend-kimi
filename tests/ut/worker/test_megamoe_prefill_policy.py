# SPDX-License-Identifier: Apache-2.0
"""CPU regression for MegaMoE DP phase/graph agreement.

Run directly with CPU torch. Load production policy methods without importing
NPU plugins; the collective simulator checks every rank's outgoing metadata.
This verifies scheduling decisions, not HCCL or ACLGraph execution.
"""

import ast
import logging
import unittest
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]


class GraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def load_policy():
    runner_path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    context_path = ROOT / "vllm_ascend/ascend_forward_context.py"
    runner_tree = ast.parse(runner_path.read_text())
    context_tree = ast.parse(context_path.read_text())
    quant_tree = ast.parse((ROOT / "vllm_ascend/quantization/quant_type.py").read_text())
    runner = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    runner.bases = []
    runner.decorator_list = []
    runner.body = [
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_sync_metadata_across_dp", "_determine_batch_execution_and_padding"}
    ]
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in quant_tree.body if isinstance(n, ast.ClassDef)]
    nodes += [
        n
        for n in context_tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {"MoECommType", "_select_a5_moe_comm_method"}
    ]
    nodes += [
        n
        for n in runner_tree.body
        if (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id.startswith("DP_METADATA_") for t in n.targets)
        )
        or (
            isinstance(n, ast.FunctionDef)
            and n.name in {"_post_process_cudagraph_mode", "_post_process_batch_contains_decode"}
        )
    ]
    nodes.append(runner)
    namespace = {
        "torch": torch,
        "Enum": Enum,
        "CUDAGraphMode": GraphMode,
        "logging": logging,
        "logger": Mock(isEnabledFor=Mock(return_value=False)),
        "np": SimpleNamespace(all=lambda x: bool(x.all())),
        "is_950": lambda: True,
        "should_skip_allreduce_across_dp_group": lambda *a: False,
        "enable_sp": lambda *a: False,
        "oproj_tp_enable": lambda: False,
        "embedding_tp_enable": lambda: False,
        "BatchDescriptor": lambda n: SimpleNamespace(num_tokens=n, num_reqs=None),
        "get_a5_mega_moe_buffer_tokens_per_rank": lambda *a, **kw: 128,
        "get_ep_group": lambda: SimpleNamespace(world_size=32),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(runner_path), "exec"), namespace
    )
    namespace["_A5_MEGA_MOE_QUANT_TYPES"] = {namespace["QuantType"].W4A8MXFP}
    namespace["_A5_MEGA_MOE_GROUP_SIZE"] = 32
    return namespace


class TestMegaMoEPrefillPolicy(unittest.TestCase):
    def setUp(self):
        self.ns = load_policy()
        self.config = SimpleNamespace(
            enable_fused_mc2=1,
            mix_placement=False,
            eplb_config=SimpleNamespace(dynamic_eplb=False, num_redundant_experts=0),
        )
        self.ns["get_ascend_config"] = lambda: self.config
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_experts_per_tok=16)),
            kv_transfer_config=None,
            use_v2_model_runner=False,
        )

    def select(self, tokens, has_decode, mode):
        return self.ns["_select_a5_moe_comm_method"](
            tokens,
            self.vllm_config,
            64,
            self.ns["QuantType"].W4A8MXFP,
            "situ",
            32,
            1,
            has_decode,
            mode,
        ).name

    def run_batch(self, phases, expected_mode, expected_comm, tokens=None, unavailable=()):
        # P=short/chunked prefill; D=decode; M=mixed; I=idle decode dummy.
        tokens = tokens or [1] * len(phases)
        modes = [
            GraphMode.NONE if p in ("P", "M") or i in unavailable else GraphMode.FULL for i, p in enumerate(phases)
        ]
        packed = torch.tensor(
            [
                tokens,
                modes,
                [int(p != "I") for p in phases],
                [int(p in ("D", "M", "I")) for p in phases],
            ],
            dtype=torch.int32,
        )
        cpu_group = object()
        self.ns["get_dp_group"] = lambda: SimpleNamespace(cpu_group=cpu_group)
        for rank, phase in enumerate(phases):
            with self.subTest(phases=phases, rank=rank):
                runner = self.ns["NPUModelRunner"]()
                runner.dp_size = len(phases)
                runner.dp_rank = rank
                runner.ascend_config = self.config
                runner.parallel_config = SimpleNamespace(data_parallel_rank=rank)
                runner.vllm_config = SimpleNamespace(
                    parallel_config=SimpleNamespace(data_parallel_size=len(phases)),
                    observability_config=SimpleNamespace(cudagraph_metrics=False),
                )
                runner.model_config = SimpleNamespace(is_encoder_decoder=False)
                runner._pad_for_sequence_parallelism = lambda n: n
                runner.use_dcp = False
                runner.uniform_decode_query_len = 1
                runner.input_batch = SimpleNamespace(
                    # A one-token prefill after a chunk looks like uniform decode
                    # to the old graph dispatcher; the explicit phase must win.
                    num_computed_tokens_cpu=torch.tensor([5]),
                    num_prompt_tokens=torch.tensor([8]),
                    lora_id_to_lora_request={},
                )

                def dispatch(num_tokens, valid_modes=None, rank=rank, **kwargs):
                    mode = (
                        next(iter(valid_modes))
                        if valid_modes
                        else (GraphMode.NONE if rank in unavailable else GraphMode.FULL)
                    )
                    return mode, SimpleNamespace(num_tokens=num_tokens, num_reqs=1)

                runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)

                def reduce(tensor, group, rank=rank):
                    self.assertIs(group, cpu_group)
                    self.assertEqual(tensor.device.type, "cpu")
                    torch.testing.assert_close(tensor[:, rank], packed[:, rank])
                    self.assertEqual(int(tensor.sum()), int(packed[:, rank].sum()))
                    tensor.copy_(packed)

                self.ns["dist"] = SimpleNamespace(all_reduce=Mock(side_effect=reduce))
                mode, _, _, dp_tokens, _ = runner._determine_batch_execution_and_padding(
                    tokens[rank],
                    1,
                    None,
                    tokens[rank],
                    False,
                    is_dummy_run=phase == "I",
                    batch_contains_decode=phase in ("D", "M", "I"),
                    batch_contains_prefill=phase in ("P", "M"),
                )
                self.ns["dist"].all_reduce.assert_called_once()
                self.assertEqual(mode, expected_mode)
                self.assertEqual(
                    self.select(int(dp_tokens.max()), runner._synced_batch_contains_decode, mode), expected_comm
                )

    def test_all_prefill_and_idle_ranks_agree_on_megamoe(self):
        self.run_batch(["P", "P"], GraphMode.NONE, "FUSED_MC2", tokens=[3, 7])
        self.run_batch(["P", "I"], GraphMode.NONE, "FUSED_MC2", tokens=[4, 1])

    def test_any_real_decode_disables_megamoe_on_every_rank(self):
        self.run_batch(["P", "D"], GraphMode.NONE, "MC2")
        self.run_batch(["M", "I"], GraphMode.NONE, "MC2")
        self.run_batch(["P", "D"], GraphMode.NONE, "ALLTOALL", tokens=[129, 1])

    def test_decode_capture_and_missing_peer_graph(self):
        self.run_batch(["D", "D"], GraphMode.FULL, "MC2")
        self.run_batch(["D", "I"], GraphMode.FULL, "MC2")
        self.run_batch(["D", "I"], GraphMode.NONE, "MC2", unavailable=(1,))
        self.run_batch(["I", "I"], GraphMode.FULL, "MC2")

    def test_graph_and_unknown_v2_phase_cannot_select_megamoe(self):
        for mode in (GraphMode.FULL, GraphMode.PIECEWISE):
            self.assertEqual(self.select(1, False, mode), "MC2")
        self.vllm_config.use_v2_model_runner = True
        self.assertEqual(self.select(1, False, GraphMode.NONE), "MC2")

    def test_profiles_and_capture_dummy_flags(self):
        tree = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text())
        dummy = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run")
        call = next(
            n
            for n in ast.walk(dummy)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_determine_batch_execution_and_padding"
        )
        value = next(k.value for k in call.keywords if k.arg == "batch_contains_decode")
        expression = compile(ast.Expression(body=value), "dummy_phase", "eval")
        for profile, prefill, capturing, expected in (
            (True, False, False, False),
            (True, True, False, False),
            (False, False, False, True),
            (False, True, False, False),
            (False, True, True, True),
            (False, False, True, True),
        ):
            self.assertEqual(
                eval(expression, dict(is_profile=profile, with_prefill=prefill, is_graph_capturing=capturing)), expected
            )

    def test_disabled_keeps_two_field_cpu_collective(self):
        self.config.enable_fused_mc2 = 0
        runner = self.ns["NPUModelRunner"]()
        runner.dp_size, runner.dp_rank = 2, 0
        runner.ascend_config = self.config
        runner.vllm_config = self.vllm_config
        group = object()
        self.ns["get_dp_group"] = lambda: SimpleNamespace(cpu_group=group)

        def reduce(tensor, group):
            self.assertEqual(tuple(tensor.shape), (2, 2))
            self.assertEqual(tensor.device.type, "cpu")
            tensor[:, 1] = torch.tensor([9, 0])

        self.ns["dist"] = SimpleNamespace(all_reduce=Mock(side_effect=reduce))
        maximum, counts, mode = runner._sync_metadata_across_dp(4, cudagraph_mode=GraphMode.FULL)
        self.assertEqual((maximum, counts.tolist(), mode), (9, [4, 9], GraphMode.NONE))
        self.assertEqual(self.select(4, False, GraphMode.NONE), "MC2")


if __name__ == "__main__":
    unittest.main()
