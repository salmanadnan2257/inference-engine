"""Physical paged KV storage: one (K, V) tensor pair per transformer layer."""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer key/value tensors laid out as fixed-size blocks.

    Shape per layer: ``[num_blocks * block_size, num_heads, head_dim]`` so a
    flat slot index (``block_id * block_size + offset``) addresses one token.
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 num_heads: int, head_dim: int, dtype: torch.dtype = torch.float32) -> None:
        self.block_size = block_size
        shape = (num_blocks * block_size, num_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` of shape [T, H, D] into the given flat slots."""
        self.k[layer].index_copy_(0, slots, k)
        self.v[layer].index_copy_(0, slots, v)

    def copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        """Apply COW copies (src_block, dst_block) across all layers."""
        bs = self.block_size
        for src, dst in copies:
            s, d = slice(src * bs, (src + 1) * bs), slice(dst * bs, (dst + 1) * bs)
            for k, v in zip(self.k, self.v):
                k[d] = k[s]
                v[d] = v[s]

    def gather(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read K/V rows for the given flat slots, shape [len(slots), H, D]."""
        return self.k[layer][slots], self.v[layer][slots]
