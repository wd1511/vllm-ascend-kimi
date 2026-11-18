# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Kimi K3 MLA decode operators from the optional cann_ops_transformer wheel.

The no-RoPE MLA decode path for Kimi K3 fuses the Q/KV down projections,
both RMSNorms, the W_UK absorption, and the merged latent-cache update into
the BF16 ``ops.mla_prolog_bf16`` operator, then runs ``flash_mla`` over the
merged ``ckv+kr`` PA_NZ cache.  The operators ship in matching CANN 9.2 and
CANNBot DSL installs and are therefore loaded lazily with actionable failure
messages.

The W8A8C8 variant uses ``quant_mla_prolog`` from the ``cannbot-dsl``
package for the quantized prolog and ``quant_flash_mla_with_kvcache`` from
``cann_ops_transformer`` for the quantized FlashMLA attention, with an FP8
PA_NZ cache using 32-byte tiles.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

import torch

PA_NZ_TILE = 16
PA_NZ_TILE_FP8 = 32
_MASK_MODE_NONE = 0
_MASK_MODE_CAUSAL = 3


def _get_cann_op(namespace, *names: str) -> Callable:
    """Return the first available operator name from a CANN namespace."""
    for name in names:
        operator = getattr(namespace, name, None)
        if operator is not None:
            return operator
    raise AttributeError(f"none of the operators are available: {', '.join(names)}")


@lru_cache
def _get_k3_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load the BF16 FlashMLA decode and metadata operators."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return (
            _get_cann_op(namespace, "flash_mla", "flash_mla_with_kvcache"),
            _get_cann_op(
                namespace,
                "flash_mla_metadata",
                "flash_mla_with_kvcache_metadata",
            ),
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Kimi K3 BF16 FlashMLA decode requires flash_mla and "
            "flash_mla_metadata from a CANN 9.2 matching cann_ops_transformer install."
        ) from exc


@lru_cache
def _get_bf16_mla_prolog_op() -> Callable:
    """Load the Kimi K3 BF16 fused MLA prolog from cannbot-dsl."""
    try:
        module = import_module("ops.mla_prolog_bf16")
        return module.mla_prolog_bf16
    except ImportError as exc:
        raise RuntimeError(
            "Kimi K3 BF16 FlashMLA requires mla_prolog_bf16 from the "
            "cannbot-dsl package."
        ) from exc


@lru_cache
def _get_quant_mla_prolog_op() -> Callable:
    """Load the W8A8C8 quantized MLA prolog from the cannbot-dsl package."""
    try:
        module = import_module("ops.quant_mla_prolog")
        return module.quant_mla_prolog_op
    except ImportError as exc:
        raise RuntimeError(
            "Kimi K3 W8A8C8 FlashMLA requires quant_mla_prolog_op from the "
            "cannbot-dsl package (pip install cannbot-dsl)."
        ) from exc


@lru_cache
def _get_quant_flash_mla_op() -> tuple[Callable, Callable]:
    """Load the quantized FlashMLA attention and metadata operators."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return (
            namespace.quant_flash_mla_with_kvcache,
            namespace.quant_flash_mla_with_kvcache_metadata,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Kimi K3 W8A8C8 FlashMLA requires quant_flash_mla_with_kvcache and "
            "quant_flash_mla_with_kvcache_metadata from a CANN 9.2 matching "
            "cann_ops_transformer install."
        ) from exc


def pa_nz_cache_view(kv_cache: torch.Tensor, tile: int = PA_NZ_TILE) -> torch.Tensor:
    """View prolog's logical PA storage as FlashMLA's physical PA_NZ layout.

    Prolog writes ``[block_num, block_size, num_kv, cache_dim]`` while
    FlashMLA reads the same storage as
    ``[block_num, num_kv, cache_dim // tile, block_size, tile]``.
    BF16 caches use 16-element tiles; FP8 caches use 32-element tiles.
    """
    block_num, block_size, num_kv, cache_dim = kv_cache.shape
    if cache_dim % tile:
        raise ValueError(f"PA_NZ cache width {cache_dim} is not aligned to {tile}.")
    return kv_cache.view(
        block_num,
        num_kv,
        cache_dim // tile,
        block_size,
        tile,
    )


def pa_nz_tile_indices(slots: torch.Tensor, block_size: int, tile_count: int) -> torch.Tensor:
    """Flat NZ-tile row indices for each (slot, tile) pair of a merged cache.

    The merged cache is written through its flattened
    ``[block_num * tile_count * block_size, tile]`` NZ view.  For a token at
    ``slot`` in block ``b`` with in-block offset ``s``, tile ``t`` lives at
    flat row ``((b * tile_count + t) * block_size + s)``.
    """
    flat_slots = slots.reshape(-1).to(torch.int64)
    block_ids = torch.div(flat_slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(flat_slots, block_size)
    tile_ids = torch.arange(tile_count, dtype=torch.int64, device=flat_slots.device)
    return (block_ids[:, None] * tile_count * block_size + tile_ids[None, :] * block_size + offsets[:, None]).reshape(
        -1
    )


def prepare_partial_block_tail_metadata(
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_size: int,
    num_requests: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare fixed-shape metadata for clearing partial cache blocks."""
    if num_requests is None:
        num_requests = min(block_table.shape[0], cache_seqlens.numel())
    if num_requests <= 0:
        return block_table.new_empty((0,), dtype=torch.long), torch.empty(
            (0, block_size), dtype=torch.bool, device=cache_seqlens.device
        )

    block_table = block_table[:num_requests]
    cache_seqlens = cache_seqlens[:num_requests].reshape(-1).to(torch.long)
    request_ids = torch.arange(num_requests, device=block_table.device, dtype=torch.long)
    last_columns = torch.div(
        torch.clamp_min(cache_seqlens, 1) - 1,
        block_size,
        rounding_mode="floor",
    )
    last_block_ids = block_table[request_ids, last_columns].to(torch.long)
    valid_tail = torch.remainder(cache_seqlens, block_size)
    valid_tail.masked_fill_(valid_tail == 0, block_size)
    offsets = torch.arange(block_size, device=cache_seqlens.device, dtype=torch.long)
    invalid = offsets[None, :] >= valid_tail[:, None]
    return last_block_ids, invalid


def clear_partial_block_tail(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    num_requests: int | None = None,
    tail_metadata: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Clear unused slots after the last valid token of each request.

    The request and slot dimensions stay fixed for a captured decode graph.
    Do not compact the mask into a dynamic index list here: dynamic-shape
    ``index_put_`` would make graph replay depend on the current sequence
    lengths. Only the last block of each live request is touched.
    """
    if num_requests is None:
        num_requests = min(block_table.shape[0], cache_seqlens.numel())
    if num_requests <= 0:
        return
    block_size = kv_cache.shape[1]
    if tail_metadata is None:
        tail_metadata = prepare_partial_block_tail_metadata(
            block_table,
            cache_seqlens,
            block_size,
            num_requests=num_requests,
        )
    last_block_ids, invalid = tail_metadata

    # aclnnIndexSelect does not accept FP8 directly. Byte-viewing it preserves
    # the cache layout and keeps the same fixed-shape operation for C8.
    cache_view = kv_cache.view(torch.int8) if kv_cache.dtype == torch.float8_e4m3fn else kv_cache
    last_blocks = cache_view.index_select(0, last_block_ids)
    tile = PA_NZ_TILE_FP8 if kv_cache.dtype == torch.float8_e4m3fn else PA_NZ_TILE
    last_blocks_nz = pa_nz_cache_view(last_blocks, tile=tile)
    mask_shape = (num_requests, 1, 1, block_size, 1)
    # Ascend masked_fill does not reliably apply implicit broadcasting to the
    # trailing cache dimensions. expand_as keeps the mask fixed-shape and
    # avoids materializing a cache-sized boolean tensor.
    invalid_mask = invalid.view(mask_shape).expand_as(last_blocks_nz)
    last_blocks_nz.masked_fill_(invalid_mask, 0)
    # The cache allocator does not share blocks when prefix caching is
    # disabled. A fixed-shape index_copy therefore updates exactly the live
    # request tails while remaining capture/replay safe.
    cache_view.index_copy_(0, last_block_ids, last_blocks)



def merged_pa_nz_gather(
    merged_cache: torch.Tensor,
    kv_lora_rank: int,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the history blocks of a merged PA_NZ cache for paged reads.

    BF16 caches are written with 16-element NZ tiles and FP8 caches with
    32-element tiles (``pa_nz_tile_indices`` / ``write_fp8_cache``): the tile
    of block ``b`` at in-block offset ``s`` lives at flat tile-row
    ``((b * tile_count + t) * block_size + s)``.  The returned views always
    use 16-element tiles because FIA's PA_NZ layout hard-requires a 16-wide
    last axis (EZ0010), so the FP8 storage is re-tiled during the gather
    (each 32-byte tile splits into two 16-element tiles).  Only the blocks
    referenced by ``block_table`` are gathered into a compact buffer
    (materializing the whole pool per layer crashed the MTE copy engine and
    exhausted device memory on large caches), and the returned remapped
    block table indexes that compact buffer.

    ``kv_lens`` bounds each request's valid columns: every table column at
    or beyond ``ceil(kv_len / block_size)`` is a stale scheduler entry that
    may point at freed blocks holding another request's KV, and the paged
    read can touch that content past the request's kv length.  Those
    columns are redirected to a dedicated all-zero block appended after the
    gathered blocks, so any read past a request's context sees zeros --
    which are benign in the softmax -- instead of stale KV.
    """
    is_fp8 = merged_cache.dtype == torch.float8_e4m3fn
    write_tile = PA_NZ_TILE_FP8 if is_fp8 else PA_NZ_TILE
    _, block_size, num_kv, cache_dim = merged_cache.shape
    if cache_dim % write_tile:
        raise ValueError(f"PA_NZ cache width {cache_dim} is not aligned to {write_tile}.")
    write_tile_count = cache_dim // write_tile
    unique_blocks, inverse = torch.unique(block_table.reshape(-1), return_inverse=True)
    # aclnnIndexSelect rejects the FP8 dtype; gather through a lossless uint8
    # view (same 1-byte width, the same bridge write_fp8_cache relies on) and
    # restore the cache dtype on the returned views.
    gather_src = merged_cache.view(torch.uint8) if is_fp8 else merged_cache
    compact = gather_src.index_select(0, unique_blocks.to(torch.int64))
    num_unique = unique_blocks.numel()
    device = merged_cache.device
    nope_tiles = kv_lora_rank // PA_NZ_TILE
    if is_fp8:
        # Re-tile 32-element write tiles into FIA's 16-element read tiles.
        # Output tile t16 covers token dims [t16*16, t16*16+16), i.e. half
        # (t16 % 2) of write tile t16 // 2, which sits at 16-byte row
        # 2*((b*write_tile_count + t16//2)*block_size + p) + (t16 % 2).
        rows16 = compact.view(-1, PA_NZ_TILE)
        read_tile_count = cache_dim // PA_NZ_TILE
        t16 = torch.arange(read_tile_count, device=device)
        block_base = 2 * torch.arange(num_unique, device=device)[:, None, None] * write_tile_count * block_size
        tile_src = 2 * torch.div(t16[None, :, None], 2, rounding_mode="floor") * block_size + torch.remainder(
            t16[None, :, None], 2
        )
        pos_src = 2 * torch.arange(block_size, device=device)[None, None, :]
        src = block_base + tile_src + pos_src
        nope_view = rows16.index_select(0, src[:, :nope_tiles, :].reshape(-1)).view(
            num_unique, num_kv, nope_tiles, block_size, PA_NZ_TILE
        )
        rope_view = rows16.index_select(0, src[:, nope_tiles:, :].reshape(-1)).view(
            num_unique, num_kv, read_tile_count - nope_tiles, block_size, PA_NZ_TILE
        )
        nope_view = nope_view.view(merged_cache.dtype)
        rope_view = rope_view.view(merged_cache.dtype)
    else:
        rope_tiles = write_tile_count - nope_tiles
        block_offsets = torch.arange(num_unique, device=device) * write_tile_count * block_size
        nope_rows = (block_offsets[:, None] + torch.arange(nope_tiles * block_size, device=device)).reshape(-1)
        rope_rows = (
            block_offsets[:, None] + nope_tiles * block_size + torch.arange(rope_tiles * block_size, device=device)
        ).reshape(-1)
        flat_rows = compact.view(-1, write_tile)
        nope_view = flat_rows.index_select(0, nope_rows).view(num_unique, num_kv, nope_tiles, block_size, write_tile)
        rope_view = flat_rows.index_select(0, rope_rows).view(num_unique, num_kv, rope_tiles, block_size, write_tile)
    remapped_table = inverse.view_as(block_table).to(block_table.dtype)
    # Redirect stale columns (at or beyond ceil(kv_len / block_size)) to the
    # appended zero block; inverse is not used afterwards, so the in-place
    # fill is safe even when .to() kept the view aliasing.
    valid_cols = (kv_lens.reshape(-1, 1) + block_size - 1) // block_size
    stale_columns = torch.arange(block_table.shape[1], device=device)[None, :] >= valid_cols
    remapped_table.masked_fill_(stale_columns, num_unique)
    nope_view = torch.cat([nope_view, nope_view.new_zeros((1,) + nope_view.shape[1:])], dim=0)
    rope_view = torch.cat([rope_view, rope_view.new_zeros((1,) + rope_view.shape[1:])], dim=0)
    return nope_view, rope_view, remapped_table


def mla_prolog_no_rope(
    hidden_states: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    cache_index: torch.Tensor,
    q_norm_eps: float,
    kv_norm_eps: float,
):
    """Run the fused no-RoPE prolog with a merged PA_NZ cache update.

    Omitting ``rope_sin``/``rope_cos`` selects Kimi K3's no-RoPE path.
    ``ckvkr_repo_mode=1`` updates the merged ``ckv+kr`` cache in place and
    returns the query with its rope segment already merged, which is the
    input FlashMLA consumes.
    """
    prolog_op = _get_bf16_mla_prolog_op()
    return prolog_op(
        hidden_states.contiguous(),
        weight_dq,
        weight_uq_qr,
        weight_dkv_kr,
        weight_uk,
        # The operator validates both RMSNorm weights as fp32.
        q_norm_weight.float().contiguous(),
        kv_norm_weight.float().contiguous(),
        kv_cache,
        cache_index.reshape(-1).to(torch.int64),
        q_norm_eps,
    )


def flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    cu_seqlens_q: torch.Tensor,
    q_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    mask_mode: int,
    seqused_q: torch.Tensor | None = None,
):
    """Build the FlashMLA metadata tensor for one decode step."""
    _, metadata_op = _get_k3_flash_mla_ops()
    return metadata_op(
        cache_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=q_len,
        max_seqlen_kv=-1,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        layout_q="TND",
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q_len: int,
    softmax_scale: float,
    head_dim_v: int,
    attn_mask: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlashMLA against the merged PA_NZ latent cache.

    ``q`` is prolog's merged query whose last dimension is the merged QK
    width (``kv_lora_rank + qk_rope_head_dim``); ``head_dim_v`` is the
    latent width (``kv_lora_rank``).
    """
    mask_mode = _MASK_MODE_NONE if q_len == 1 else _MASK_MODE_CAUSAL
    if metadata is None:
        metadata = flash_mla_metadata(
            cache_seqlens,
            num_heads_q=q.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            q_len=q_len,
            head_dim_qk=q.shape[-1],
            head_dim_v=head_dim_v,
            mask_mode=mask_mode,
            seqused_q=seqused_q,
        )
    attention_op, _ = _get_k3_flash_mla_ops()
    output, _ = attention_op(
        q.contiguous(),
        pa_nz_cache_view(kv_cache),
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=head_dim_v,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_NZ",
        layout_out="TND",
        return_softmax_lse=False,
    )
    return output


# ---------------------------------------------------------------------------
# W8A8C8 quantized path
# ---------------------------------------------------------------------------


def pack_mxfp8_nz(weight: torch.Tensor) -> torch.Tensor:
    """Pack a checkpoint-oriented [O, K] E4M3 weight into [O/32, K, 32]."""
    output, k = weight.shape
    return weight.view(torch.uint8).reshape(output // 32, 32, k).transpose(1, 2).contiguous().view(weight.dtype)


def logical_mx_scale(scale: torch.Tensor, output: int, k: int) -> torch.Tensor:
    """Reshape a [O, K//32] uint8 scale to [O, K//64, 2] e8m0fnu."""
    return scale.view(torch.uint8).reshape(output, k // 64, 2).contiguous().view(torch.float8_e8m0fnu)


def reverse_mxfp8_weight(linear) -> torch.Tensor:
    """Get the W8A8 weight in checkpoint [O, K] orientation from a vllm-ascend linear.

    After AscendW8A8MXFP8DynamicLinearMethod.process_weights_after_loading,
    the weight is transposed to [K, O].  Detect and reverse that here.
    """
    w = linear.weight.data
    if w.shape[0] != w.shape[1] and w.shape[1] < w.shape[0]:
        pass
    return w


def quant_mla_prolog(
    hidden_states: torch.Tensor,
    hidden_scale: torch.Tensor,
    weight_qa: torch.Tensor,
    weight_qb: torch.Tensor,
    weight_kva: torch.Tensor,
    weight_uk: torch.Tensor,
    scale_qa: torch.Tensor,
    scale_qb: torch.Tensor,
    scale_kva: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    kv_scale: torch.Tensor,
    norm_eps: float = 1e-6,
):
    """Run the W8A8C8 quantized no-RoPE prolog.

    ``hidden_states`` must already be dynamically MX-quantized to FP8 with
    its scale reshaped to e8m0fnu.  Returns ``(q, q_descale)`` where ``q``
    is the quantized absorbed query and ``q_descale`` dequantizes it for
    the quant FlashMLA call.
    """
    prolog_op = _get_quant_mla_prolog_op()
    outs = prolog_op(
        hidden_states,
        weight_qa,
        weight_qb,
        weight_kva,
        weight_uk,
        hidden_scale,
        scale_qa,
        scale_qb,
        scale_kva,
        q_norm_weight,
        kv_norm_weight,
        kv_cache,
        cache_index.reshape(-1).to(torch.int64),
        kv_scale,
        norm_eps=norm_eps,
        quant_mode_aw=1,
        quant_mode_c=1,
    )
    return outs[0], outs[1]


def quant_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    cu_seqlens_q: torch.Tensor,
    seqused_q: torch.Tensor,
    q_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    mask_mode: int,
):
    """Build quantized FlashMLA metadata for one decode step."""
    _, metadata_op = _get_quant_flash_mla_op()
    return metadata_op(
        cache_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=q_len,
        max_seqlen_kv=-1,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        layout_q="TND",
        quant_mode=1,
    )


def quant_flash_mla_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    q_descale: torch.Tensor,
    kv_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q_len: int,
    softmax_scale: float,
    head_dim_v: int,
    attn_mask: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the quantized FlashMLA against the FP8 PA_NZ latent cache.

    The FP8 cache uses 32-element tiles (vs 16 for BF16).  ``q`` is the
    quantized prolog output and ``q_descale``/``kv_scale`` dequantize the
    attention scores.
    """
    mask_mode = _MASK_MODE_NONE if q_len == 1 else _MASK_MODE_CAUSAL
    if metadata is None:
        metadata = quant_flash_mla_metadata(
            cache_seqlens,
            num_heads_q=q.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            q_len=q_len,
            head_dim_qk=q.shape[-1],
            head_dim_v=head_dim_v,
            mask_mode=mask_mode,
        )
    attention_op, _ = _get_quant_flash_mla_op()
    output, _ = attention_op(
        q,
        pa_nz_cache_view(kv_cache, tile=PA_NZ_TILE_FP8),
        q_descale,
        kv_scale,
        block_table,
        cache_seqlens,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        metadata=metadata,
        attn_mask=attn_mask,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        max_seqlen_q=q_len,
        head_dim_v=head_dim_v,
        layout_q="TND",
        layout_kv="PA_NZ",
        layout_out="TND",
    )
    return output


def write_fp8_cache(
    merged: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    descale: torch.Tensor,
) -> None:
    """Write prefill KV into the FP8 PA_NZ cache.

    ``descale`` is a divisor (despite the name): fp8 = value / scale.
    Negative padded slots are skipped.  The cache uses 32-element tiles.
    """
    import torch_npu  # noqa: PLC0415

    slots = slots.reshape(-1).to(torch.int64)
    valid = slots >= 0
    slots = slots[valid]
    rows = (merged[valid].float() / descale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    tile_count = cache.shape[-1] // PA_NZ_TILE_FP8
    indices = pa_nz_tile_indices(slots, cache.shape[1], tile_count)
    torch_npu.npu_scatter_nd_update_(
        cache.view(torch.uint8).view(-1, PA_NZ_TILE_FP8), indices.view(-1, 1), rows.reshape(-1, PA_NZ_TILE_FP8)
    )
