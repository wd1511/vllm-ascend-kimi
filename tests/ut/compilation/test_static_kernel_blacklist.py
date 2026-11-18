# SPDX-License-Identifier: Apache-2.0
"""CPU-only backend setup regression; run directly without NPU dependencies."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def load_compile_function(namespace):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/compiler_interface.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "npugraph_ex_compile")
    # Postpone production type annotations without loading torch/vllm/torch_npu.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["npugraph_ex_compile"]


class TestStaticKernelBlacklist(unittest.TestCase):
    def test_blacklist_precedes_backend_and_only_applies_to_static_mode(self):
        for enabled in (False, True):
            with self.subTest(enable_static_kernel=enabled):
                order = []
                blacklist = Mock(side_effect=lambda ops, order=order: order.append("blacklist"))
                compiled = object()
                graph = object()
                inputs = []
                backend = Mock(return_value=compiled)
                get_backend = Mock(
                    side_effect=lambda order=order, backend=backend, **kwargs: (order.append("backend"), backend)[1]
                )
                modules = {
                    name: ModuleType(name)
                    for name in (
                        "npugraph_ex",
                        "npugraph_ex.configs",
                        "npugraph_ex.configs.compiler_config",
                        "npugraph_ex.npu_fx_compiler",
                    )
                }
                # The static-only dependency is absent when the flag is off.
                if enabled:
                    modules["npugraph_ex._acl_concrete_graph"] = ModuleType("npugraph_ex._acl_concrete_graph")
                    static_module = ModuleType("npugraph_ex._acl_concrete_graph.static_kernel")
                    static_module._set_static_kernel_blacklist = blacklist
                    modules[static_module.__name__] = static_module
                nge = modules["npugraph_ex"]
                nge.CompilerConfig = SimpleNamespace
                nge.get_npu_backend = get_backend
                modules["npugraph_ex.configs.compiler_config"]._process_kwargs_options = Mock()
                compiler = type("Compiler", (), {"_get_compiled_gm": Mock()})
                modules["npugraph_ex.npu_fx_compiler"]._NpuFxCompiler = compiler
                namespace = {
                    "os": os,
                    "torch": SimpleNamespace(npu=Mock()),
                    "logger": Mock(),
                    "_configure_backend": Mock(side_effect=lambda *a, order=order, **kw: order.append("configure")),
                    "graph_returns_tuple": lambda graph: True,
                }
                compile_fn = load_compile_function(namespace)
                with patch.dict(sys.modules, modules):
                    result, _ = compile_fn(
                        graph, inputs, {}, SimpleNamespace(), SimpleNamespace(enable_static_kernel=enabled), None
                    )
                self.assertIs(result, compiled)
                backend.assert_called_once_with(graph, inputs)
                if enabled:
                    blacklist.assert_called_once_with(["FusedInferAttentionScore"])
                    self.assertEqual(order, ["blacklist", "configure", "backend"])
                else:
                    blacklist.assert_not_called()
                    self.assertEqual(order, ["configure", "backend"])


if __name__ == "__main__":
    unittest.main()
