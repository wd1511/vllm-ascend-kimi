#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


def _new_request_prompt_len(req: "NewRequestData") -> int | None:
    prompt_token_ids = getattr(req, "prompt_token_ids", None)
    if prompt_token_ids:
        return len(prompt_token_ids)

    prompt_embeds = getattr(req, "prompt_embeds", None)
    if prompt_embeds is not None:
        shape = getattr(prompt_embeds, "shape", None)
        return int(shape[0]) if shape is not None else len(prompt_embeds)

    prefill_token_ids = getattr(req, "prefill_token_ids", None)
    if prefill_token_ids:
        return len(prefill_token_ids)

    return None


def _new_request_is_past_prefill(req: "NewRequestData") -> bool:
    prompt_len = _new_request_prompt_len(req)
    if prompt_len is None or prompt_len <= 0:
        return False
    return int(getattr(req, "num_computed_tokens", 0)) >= prompt_len


def _is_cached_decode_request(cached, req_id: str) -> bool:
    resumed_req_ids = getattr(cached, "resumed_req_ids", None)
    if resumed_req_ids and req_id in resumed_req_ids:
        return False

    is_context_phase = getattr(cached, "is_context_phase", None)
    if callable(is_context_phase):
        return not is_context_phase(req_id)

    num_output_tokens = getattr(cached, "num_output_tokens", None)
    req_ids = getattr(cached, "req_ids", None)
    if num_output_tokens is None or req_ids is None:
        return False

    try:
        index = req_ids.index(req_id)
    except ValueError:
        return False
    return num_output_tokens[index] > 0


def is_decode_only_batch(scheduler_output: "SchedulerOutput") -> bool:
    """Return whether the scheduler output contains only decode requests."""
    if scheduler_output is None:
        return False

    total_num_scheduled_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
    if total_num_scheduled_tokens is None or total_num_scheduled_tokens <= 0:
        return False

    if getattr(scheduler_output, "scheduled_encoder_inputs", None):
        return False

    num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not num_scheduled_tokens:
        return False

    preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_req_ids:
        if any(req_id in preempted_req_ids for req_id in num_scheduled_tokens):
            return False

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached_reqs is None:
        return False
    cached_req_ids = set(getattr(cached_reqs, "req_ids", None) or ())
    new_reqs = {
        req.req_id: req for req in (getattr(scheduler_output, "scheduled_new_reqs", None) or ())
    }

    spec_decode_tokens = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    for req_id, num_tokens in num_scheduled_tokens.items():
        new_req = new_reqs.get(req_id)
        if new_req is not None:
            if not _new_request_is_past_prefill(new_req):
                return False
        elif req_id not in cached_req_ids or not _is_cached_decode_request(cached_reqs, req_id):
            return False

        valid_tokens = num_tokens - len(spec_decode_tokens.get(req_id) or ())
        if valid_tokens != 1:
            return False

    return True
