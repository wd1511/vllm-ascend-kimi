#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import suppress
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.logger import logger
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        trace_name: str,
        decode_only: bool = False,
    ) -> None:
        super().__init__(profiler_config)
        self.profiler_config = profiler_config
        self.trace_name = trace_name
        self.decode_only = decode_only
        self.profiler: Any | None = None
        self._decode_steps = 0
        self._skipped_steps = 0
        self._phase_index = 0
        self._phase_decode_steps = 0
        if not decode_only:
            self.profiler = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str, analyse: bool = True) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = False
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError(
                "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time."
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        trace_handler_kwargs: dict[str, Any] = {"worker_name": trace_name}
        if not analyse:
            trace_handler_kwargs["analyse_flag"] = False

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                **trace_handler_kwargs,
            ),
        )

    def _start(self) -> None:
        self._decode_steps = 0
        self._skipped_steps = 0
        self._phase_index = 0
        self._phase_decode_steps = 0

        if not self.decode_only:
            if self.profiler is None:
                self.profiler = self._create_profiler(self.profiler_config, self.trace_name)
            self.profiler.start()
            return

        # Decode-only mode starts a separate profiling session for each
        # contiguous decode phase. The underlying torch_npu profiler is not
        # started here, so no empty CANN PROF_* directory is produced.
        self.profiler = None
        logger.info("Decode-only profiling armed for trace %s", self.trace_name)

    def _stop(self) -> None:
        if not self.decode_only:
            if self.profiler is not None:
                self.profiler.stop()
            return

        self._stop_phase()

        if self._decode_steps == 0:
            logger.warning(
                "Decode-only profiling recorded no decode steps. Check whether the workload contains pure decode "
                "steps; if every step is a PD mixed batch, consider PD disaggregation or a decode-only window."
            )
        logger.info(
            "Decode-only profiling finished: sessions=%d, decode_steps=%d, skipped_steps=%d",
            self._phase_index,
            self._decode_steps,
            self._skipped_steps,
        )

    def _start_phase(self) -> bool:
        phase_trace_name = f"{self.trace_name}_phase{self._phase_index + 1}"
        try:
            self.profiler = self._create_profiler(
                self.profiler_config,
                phase_trace_name,
                analyse=False,
            )
            self.profiler.start()
        except Exception:
            logger.exception("Failed to start decode-only profiling session.")
            self.profiler = None
            return False

        self._phase_index += 1
        self._phase_decode_steps = 0
        logger.info("Decode-only profiling session %d started.", self._phase_index)
        return True

    def _stop_phase(self) -> None:
        if self.profiler is None:
            return

        try:
            try:
                self._add_metadata()
            except Exception:
                logger.exception("Failed to add decode-only profiling metadata.")
            self.profiler.stop()
            logger.info(
                "Decode-only profiling session %d stopped: decode_steps=%d.",
                self._phase_index,
                self._phase_decode_steps,
            )
        except Exception:
            logger.exception("Failed to stop decode-only profiling session.")
        finally:
            self.profiler = None
            self._phase_decode_steps = 0

    def pause(self) -> None:
        if not self.decode_only:
            return
        self._skipped_steps += 1
        self._stop_phase()

    def _add_metadata(self) -> None:
        if self.profiler is None:
            return
        add_metadata = getattr(self.profiler, "add_metadata", None)
        if add_metadata is None:
            return
        add_metadata("decode_only", "true")
        add_metadata("decode_phase", str(self._phase_index))
        add_metadata("decode_steps", str(self._phase_decode_steps))
        add_metadata("skipped_steps", str(self._skipped_steps))

    def _profiler_step(self) -> bool:
        if not self.decode_only:
            return True
        if self.profiler is None and not self._start_phase():
            return False
        self._decode_steps += 1
        self._phase_decode_steps += 1
        return True
