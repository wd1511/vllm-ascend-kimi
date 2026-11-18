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

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.flash_mla import (
    PA_NZ_TILE,
    PA_NZ_TILE_FP8,
    merged_pa_nz_gather,
    pa_nz_tile_indices,
)


def _as_comparable(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


class TestMergedPaNzLayout(TestBase):
    """Write/read contract of the merged PA_NZ cache.

    BF16 caches are written with 16-element NZ tiles and FP8 caches with
    32-element tiles; both writers (``pa_nz_tile_indices`` scatter,
    ``write_fp8_cache``) place the tile of block ``b`` at in-block offset
    ``s`` at flat tile-row ``((b * tile_count + t) * block_size + s)``.
    ``merged_pa_nz_gather`` must read the same token bytes back through
    16-element tile views -- FIA's PA_NZ layout hard-requires a 16-wide last
    axis (EZ0010) -- re-tiling the FP8 storage during the gather, and must
    redirect every table column beyond ``ceil(kv_len / block_size)`` to an
    appended all-zero block.
    """

    def _run_roundtrip(self, dtype: torch.dtype, write_tile: int):
        torch.manual_seed(0)
        num_blocks, block_size, kv_lora_rank, rope_dim = 6, 8, 32, 32
        cache_dim = kv_lora_rank + rope_dim
        nope_tiles = kv_lora_rank // PA_NZ_TILE
        rope_tiles = rope_dim // PA_NZ_TILE

        merged_cache = torch.zeros(num_blocks, block_size, 1, cache_dim, dtype=dtype)
        slots = torch.tensor([0, 3, block_size, block_size + 5, 2 * block_size + 7, 5 * block_size + block_size - 1])
        if dtype == torch.bfloat16:
            tokens = torch.randn(slots.numel(), cache_dim, dtype=dtype)
        else:
            tokens = (
                torch.arange(slots.numel() * cache_dim)
                .remainder(251)
                .to(torch.uint8)
                .view(torch.float8_e4m3fn)
                .reshape(slots.numel(), cache_dim)
            )
        indices = pa_nz_tile_indices(slots, block_size, cache_dim // write_tile)
        merged_cache.view(-1, write_tile)[indices] = tokens.reshape(-1, write_tile)

        # Unsorted table with duplicate and unreferenced blocks; kv_lens make
        # every column valid so the roundtrip covers the full mapping.
        block_table = torch.tensor([[2, 0, 2, 5], [0, 0, 5, 2]], dtype=torch.int32)
        kv_lens = torch.tensor([4 * block_size, 4 * block_size], dtype=torch.int32)
        nope_view, rope_view, remapped_table = merged_pa_nz_gather(merged_cache, kv_lora_rank, block_table, kv_lens)

        self.assertEqual(nope_view.dtype, dtype)
        # 3 unique blocks plus the appended zero block.
        self.assertEqual(nope_view.shape, (4, 1, nope_tiles, block_size, PA_NZ_TILE))
        self.assertEqual(rope_view.shape, (4, 1, rope_tiles, block_size, PA_NZ_TILE))
        self.assertEqual(remapped_table.dtype, torch.int32)
        # unique -> [0, 2, 5]; every entry maps to its position in that order.
        expected_pos = {0: 0, 2: 1, 5: 2}
        for r in range(2):
            for c in range(4):
                self.assertEqual(remapped_table[r, c].item(), expected_pos[block_table[r, c].item()])

        # Every written token must read back through the 16-element tile
        # views: tile t of a view covers the token dims [base + t*16, +16).
        for i, slot in enumerate(slots.tolist()):
            b, p = slot // block_size, slot % block_size
            if b not in expected_pos:
                continue
            ci = expected_pos[b]
            for t in range(nope_tiles + rope_tiles):
                if t < nope_tiles:
                    view, base, tt = nope_view, 0, t
                else:
                    view, base, tt = rope_view, kv_lora_rank, t - nope_tiles
                got = _as_comparable(view[ci, 0, tt, p])
                want = _as_comparable(tokens[i, base + tt * PA_NZ_TILE : base + (tt + 1) * PA_NZ_TILE])
                self.assertTrue(torch.equal(got, want), f"token {i} tile {t} mismatch")

    def test_roundtrip_bf16_native_16_tile_write(self):
        self._run_roundtrip(torch.bfloat16, PA_NZ_TILE)

    def test_roundtrip_fp8_retile_from_32_tile_write(self):
        self._run_roundtrip(torch.float8_e4m3fn, PA_NZ_TILE_FP8)

    def test_stale_columns_redirect_to_zero_block(self):
        """Regression guard for the cross-request contamination: columns at
        or beyond ceil(kv_len / block_size) are stale scheduler entries that
        may point at freed blocks holding another request's KV, and the
        paged read can touch them past the request's kv length."""
        num_blocks, block_size, kv_lora_rank, rope_dim = 6, 8, 32, 32
        cache_dim = kv_lora_rank + rope_dim
        merged_cache = torch.zeros(num_blocks, block_size, 1, cache_dim, dtype=torch.bfloat16)
        slots = torch.tensor([2 * block_size + 3])
        tokens = torch.randn(1, cache_dim, dtype=torch.bfloat16)
        merged_cache.view(-1, PA_NZ_TILE)[pa_nz_tile_indices(slots, block_size, cache_dim // PA_NZ_TILE)] = (
            tokens.reshape(-1, PA_NZ_TILE)
        )

        # Request 0 may read 2 blocks (columns 0-1); columns 2-3 are stale.
        block_table = torch.tensor([[2, 0, 5, 2]], dtype=torch.int32)
        kv_lens = torch.tensor([2 * block_size], dtype=torch.int32)
        nope_view, rope_view, remapped_table = merged_pa_nz_gather(merged_cache, kv_lora_rank, block_table, kv_lens)

        # unique -> [0, 2, 5]; the zero block is appended at index 3 and the
        # stale columns (2, 3) remap to it instead of their scheduler ids.
        self.assertEqual(remapped_table.tolist(), [[1, 0, 3, 3]])
        self.assertEqual(nope_view.shape[0], 4)
        self.assertTrue(torch.all(nope_view[3] == 0))
        self.assertTrue(torch.all(rope_view[3] == 0))
        # The valid columns still roundtrip the written token (tile t covers
        # dims [t*16, t*16+16)).
        self.assertTrue(torch.equal(nope_view[1, 0, 0, 3], tokens[0, :PA_NZ_TILE]))
        self.assertTrue(torch.equal(nope_view[1, 0, 1, 3], tokens[0, PA_NZ_TILE:kv_lora_rank]))
        self.assertTrue(torch.equal(rope_view[1, 0, 0, 3], tokens[0, kv_lora_rank : kv_lora_rank + PA_NZ_TILE]))
        self.assertTrue(torch.equal(rope_view[1, 0, 1, 3], tokens[0, kv_lora_rank + PA_NZ_TILE :]))

    def test_fia_read_layout_is_always_16_tile(self):
        """Regression guard for the EZ0010 rejection: FIA's PA_NZ layout
        requires the last axis of key/value to be 16, so the gathered views
        must never expose the FP8 storage's 32-element write tiles."""
        num_blocks, block_size, kv_lora_rank = 2, 4, 32
        cache_dim = 64
        cache = torch.zeros(num_blocks, block_size, 1, cache_dim, dtype=torch.float8_e4m3fn)
        nope_view, rope_view, _ = merged_pa_nz_gather(
            cache, kv_lora_rank, torch.tensor([[0]], dtype=torch.int32), torch.tensor([block_size], dtype=torch.int32)
        )
        self.assertEqual(nope_view.shape[-1], PA_NZ_TILE)
        self.assertEqual(rope_view.shape[-1], PA_NZ_TILE)
        self.assertEqual(nope_view.shape[2], kv_lora_rank // PA_NZ_TILE)

    def test_unaligned_cache_width_raises(self):
        cache = torch.zeros(2, 4, 1, 60, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            merged_pa_nz_gather(cache, 32, torch.tensor([[0]], dtype=torch.int32), torch.tensor([4], dtype=torch.int32))
