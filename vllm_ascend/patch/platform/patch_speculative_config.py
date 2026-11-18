from contextlib import contextmanager
from copy import copy

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


def _is_redhat_window_gqa(hf_config: PretrainedConfig) -> bool:
    architectures = getattr(hf_config, "architectures", ()) or ()
    return (
        getattr(hf_config, "model_type", None) in ("qwen3", "speculators")
        and (
            "Qwen3DSparkModel" in architectures
            or "DSparkDraftModel" in architectures
        )
        and bool(getattr(hf_config, "use_sliding_window", False))
        and bool(getattr(hf_config, "sample_from_anchor", False))
        and getattr(hf_config, "block_size", None) == 8
        and getattr(hf_config, "sliding_window", None) is not None
    )


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    is_qwen3_dspark = (
        hf_config.model_type == "qwen3"
        and (
            "DSparkDraftModel" in architectures
            or "Qwen3DSparkModel" in architectures
        )
        or hf_config.model_type == "speculators"
        and (
            "DSparkDraftModel" in architectures
            or "Qwen3DSparkModel" in architectures
        )
    )
    if is_qwen3_dspark:
        # RedHat's Kimi/DSpark GQA checkpoint keeps the Qwen3 decoder fields
        # under ``transformer_layer_config``.  vLLM constructs the draft
        # model from the top-level HF config, so expose those fields before the
        # normal Qwen3 DSpark loader sees the object.  The legacy flat format
        # continues through the original path below.
        # Keep the outer value: RedHat stores this flag outside
        # ``transformer_layer_config`` while some older exports put it inside.
        outer_non_causal = getattr(hf_config, "sliding_window_non_causal", None)
        transformer_config = getattr(hf_config, "transformer_layer_config", None)
        if isinstance(transformer_config, dict):
            hf_config.update(transformer_config)
        # SpeculatorsConfig keeps ``model_type=speculators`` after translating
        # the nested Qwen3 decoder fields.  The Ascend Kimi runner uses the
        # model type to select the materialized GQA auxiliary stream, so expose
        # the actual draft architecture after conversion.
        if getattr(hf_config, "model_type", None) == "speculators":
            hf_config.model_type = "qwen3"

        dflash_config = dict(getattr(hf_config, "dflash_config", None) or {})
        aux_layer_ids = getattr(hf_config, "aux_hidden_state_layer_ids", None)
        if aux_layer_ids is None:
            aux_layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        target_layer_ids = dflash_config.get("target_layer_ids")
        if target_layer_ids is None:
            target_layer_ids = getattr(hf_config, "target_layer_ids", None)
        if target_layer_ids is None and aux_layer_ids:
            # The RedHat field is one-based; vLLM's target_layer_ids are
            # zero-based and eagle3_utils adds one when capturing states.
            target_layer_ids = [int(layer_id) - 1 for layer_id in aux_layer_ids]

        normalized = {
            "architectures": ["Qwen3DSparkModel"],
            "mask_token_id": dflash_config.get(
                "mask_token_id", getattr(hf_config, "mask_token_id", None)
            ),
        }
        if target_layer_ids is not None:
            normalized["target_layer_ids"] = list(target_layer_ids)
            # Keep an explicit alias for vLLM revisions that look up the
            # DSpark spelling before falling back to target_layer_ids.
            normalized["dspark_target_layer_ids"] = list(target_layer_ids)
        if aux_layer_ids:
            normalized["eagle_aux_hidden_state_layer_ids"] = [int(i) for i in aux_layer_ids]
        non_causal = outer_non_causal
        if non_causal is None and isinstance(transformer_config, dict):
            non_causal = transformer_config.get("sliding_window_non_causal")
        if non_causal is not None:
            # DFlash's layer resolver uses ``is_causal`` as the explicit
            # override.  RedHat names the same contract from the opposite
            # direction.
            causal = not bool(non_causal)
            normalized["is_causal"] = causal
            dflash_config["causal"] = causal
        elif (
            bool(getattr(hf_config, "use_sliding_window", False))
            and getattr(hf_config, "layer_types", None)
            and all(layer_type == "sliding_attention" for layer_type in hf_config.layer_types)
        ):
            # SpeculatorsConfig drops outer-only fields while converting the
            # checkpoint.  RedHat's all-SWA layer list still unambiguously
            # identifies its causal draft mask.
            dflash_config.setdefault("causal", True)
        # The checkpoint was trained with an anchor-aligned history window:
        # every one of the eight query rows sees the same W history tokens.
        # FIA expresses a causal window relative to each query row, so extend
        # its pre-token span by the seven row offset.  The proposer still
        # crops the block table/sequence lengths to the real W-token cache
        # window; this value only corrects the kernel's query-relative mask.
        sliding_window = getattr(hf_config, "sliding_window", None)
        native_width = getattr(hf_config, "block_size", None)
        if (
            sliding_window is not None
            and bool(getattr(hf_config, "use_sliding_window", False))
            and bool(getattr(hf_config, "sample_from_anchor", False))
            and aux_layer_ids is not None
            and native_width == 8
        ):
            # DFlash resolves the attention implementation from this nested
            # flag.  Keep the normalization valid even when an exporter omits
            # ``layer_types`` after converting the speculators config.
            dflash_config["use_swa"] = True
            dflash_config["swa_window_size"] = int(sliding_window) + int(native_width) - 1
        if target_layer_ids is not None:
            dflash_config["target_layer_ids"] = list(target_layer_ids)
        if normalized["mask_token_id"] is not None:
            dflash_config.setdefault("mask_token_id", normalized["mask_token_id"])
        if dflash_config:
            normalized["dflash_config"] = dflash_config
        hf_config.update(normalized)
    return hf_config


@contextmanager
def _temporarily_disable_dspark_dcp(self: SpeculativeConfig):
    target_parallel_config = self.target_parallel_config
    if getattr(self, "method", None) != "dspark" or target_parallel_config.decode_context_parallel_size <= 1:
        yield
        return

    guard_parallel_config = copy(target_parallel_config)
    guard_parallel_config.decode_context_parallel_size = 1
    self.target_parallel_config = guard_parallel_config
    try:
        yield
    finally:
        self.target_parallel_config = target_parallel_config


def _dspark_post_init(self):
    # TODO: This block can be deleted after the upstream supports the overlay of mla dcp and dspark
    draft_model_config = getattr(self, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    with _temporarily_disable_dspark_dcp(self):
        _orig_post_init(self)
    # ``draft_model_config`` is normally created by the upstream post-init
    # above.  Refresh it before applying the RedHat Window-GQA ABI fix; on
    # entry it is usually still None for a model-path based DSpark config.
    draft_model_config = getattr(self, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    # RedHat's native block has eight anchor-sampled rows, while the Kimi K3
    # Verify path in Ascend uses the seven-token public ABI.  Normalize the
    # auto-generated speculators value (8) after upstream has loaded the draft
    # config, but before the runner constructs the proposer and sampler.
    if draft_hf_config is not None and _is_redhat_window_gqa(draft_hf_config):
        if self.num_speculative_tokens == 8:
            self.num_speculative_tokens = 7
            if hasattr(draft_hf_config, "num_lookahead_tokens"):
                draft_hf_config.num_lookahead_tokens = 7
    if self.use_dspark():
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
