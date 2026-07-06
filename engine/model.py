"""GPT-2 model executor with paged attention.

This is an independent forward implementation. Hugging Face is used only to
fetch pretrained weights, which are transposed out of Conv1D layout and loaded
into plain ``nn.Linear`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .block_manager import BlockManager
from .cache import KVCache
from .sequence import Sequence

NEG_INF = float("-inf")


@dataclass
class ModelConfig:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, layer_idx: int,
                ctx: "ForwardContext | None") -> torch.Tensor:
        B, T, E = x.shape
        q, k, v = self.c_attn(x).split(E, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        if ctx is None:
            # Cache-free causal attention (parity tests, toy models).
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True)
        else:
            ctx.cache.write(layer_idx, ctx.slot_mapping,
                            k.reshape(B * T, self.n_head, self.head_dim),
                            v.reshape(B * T, self.n_head, self.head_dim))
            keys, values = ctx.cache.gather(layer_idx, ctx.gather_slots.reshape(-1))
            Lmax = ctx.gather_slots.shape[1]
            keys = keys.view(B, Lmax, self.n_head, self.head_dim).transpose(1, 2)
            values = values.view(B, Lmax, self.n_head, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), keys, values, attn_mask=ctx.attn_mask)
        out = out.transpose(1, 2).reshape(B, T, E)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, layer_idx: int,
                ctx: "ForwardContext | None") -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), layer_idx, ctx)
        return x + self.mlp(self.ln_2(x))


@dataclass
class ForwardContext:
    """Per-forward paged attention metadata, shared by all layers."""

    cache: KVCache
    slot_mapping: torch.Tensor   # [B*T] flat write slots for the new tokens
    gather_slots: torch.Tensor   # [B, Lmax] flat read slots (padded with 0)
    attn_mask: torch.Tensor      # [B, 1, T, Lmax] additive float mask


class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                ctx: ForwardContext | None = None) -> torch.Tensor:
        """input_ids, positions: [B, T]. Returns logits [B, T, vocab]."""
        x = self.wte(input_ids) + self.wpe(positions)
        for i, block in enumerate(self.h):
            x = block(x, i, ctx)
        x = self.ln_f(x)
        return x @ self.wte.weight.t()

    @classmethod
    def from_hf(cls, model_name: str) -> "GPT2":
        """Load pretrained GPT-2 family weights from Hugging Face."""
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        hc = hf.config
        cfg = ModelConfig(vocab_size=hc.vocab_size, n_positions=hc.n_positions,
                          n_embd=hc.n_embd, n_layer=hc.n_layer, n_head=hc.n_head,
                          layer_norm_eps=hc.layer_norm_epsilon)
        model = cls(cfg)
        sd = hf.state_dict()
        new_sd: dict[str, torch.Tensor] = {}
        new_sd["wte.weight"] = sd["transformer.wte.weight"]
        new_sd["wpe.weight"] = sd["transformer.wpe.weight"]
        new_sd["ln_f.weight"] = sd["transformer.ln_f.weight"]
        new_sd["ln_f.bias"] = sd["transformer.ln_f.bias"]
        for i in range(cfg.n_layer):
            src, dst = f"transformer.h.{i}", f"h.{i}"
            for ln in ("ln_1", "ln_2"):
                new_sd[f"{dst}.{ln}.weight"] = sd[f"{src}.{ln}.weight"]
                new_sd[f"{dst}.{ln}.bias"] = sd[f"{src}.{ln}.bias"]
            # HF Conv1D stores weights as [in, out]; nn.Linear wants [out, in].
            for name in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                new_sd[f"{dst}.{name}.weight"] = sd[f"{src}.{name}.weight"].t().contiguous()
                new_sd[f"{dst}.{name}.bias"] = sd[f"{src}.{name}.bias"]
        model.load_state_dict(new_sd)
        model.eval()
        return model


class ModelRunner:
    """Executes batched forwards for sequences against a paged KV cache."""

    def __init__(self, model: GPT2, num_blocks: int, block_size: int) -> None:
        self.model = model
        self.block_size = block_size
        cfg = model.cfg
        self.cache = KVCache(cfg.n_layer, num_blocks, block_size,
                             cfg.n_head, cfg.head_dim)

    def apply_copies(self, copies: list[tuple[int, int]]) -> None:
        self.cache.copy_blocks(copies)

    @torch.inference_mode()
    def execute(self, seqs: list[Sequence], block_manager: BlockManager) -> torch.Tensor:
        """Run one forward over each sequence's uncomputed tail tokens.

        Every sequence must contribute the same number of new tokens
        ``T = len(seq) - seq.num_computed`` (T=1 for decode; arbitrary equal T
        for prefill batches of one or speculative verification). Marks the
        tokens as computed and returns logits ``[B, T, vocab]``.
        """
        T = len(seqs[0]) - seqs[0].num_computed
        assert T >= 1 and all(len(s) - s.num_computed == T for s in seqs)
        B = len(seqs)
        input_ids = torch.tensor([s.token_ids[s.num_computed:] for s in seqs])
        positions = torch.tensor([list(range(s.num_computed, len(s))) for s in seqs])
        slot_mapping = torch.tensor(
            [slot for s in seqs
             for slot in block_manager.slot_mapping(s, s.num_computed, len(s))])

        lens = [len(s) for s in seqs]
        Lmax = max(lens)
        gather = torch.zeros((B, Lmax), dtype=torch.long)
        for b, s in enumerate(seqs):
            gather[b, :lens[b]] = torch.tensor(
                block_manager.slot_mapping(s, 0, lens[b]))
        # Position t of the query block attends to key positions <= ctx + t.
        key_pos = torch.arange(Lmax).view(1, 1, Lmax)
        limit = torch.tensor([s.num_computed for s in seqs]).view(B, 1, 1) \
            + torch.arange(T).view(1, T, 1)
        mask = torch.where(key_pos <= limit, 0.0, NEG_INF).view(B, 1, T, Lmax)

        ctx = ForwardContext(self.cache, slot_mapping, gather, mask)
        logits = self.model(input_ids, positions, ctx)
        for s in seqs:
            s.num_computed = len(s)
        return logits
