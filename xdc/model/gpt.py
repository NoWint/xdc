"""Tiny GPT model used by XDC's training and pipeline-inference engines.

The model is deliberately implemented in plain PyTorch so that every layer can be
instantiated independently on a different device, and so the pipeline result can be
checked bit-for-bit (within float tolerance) against a monolithic forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0
    bias: bool = True

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in allowed})
        cfg.validate()
        return cfg


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # Prompt prefill (no cache) needs a causal mask. Single-token decode with a
        # populated cache does not. A cached multi-token suffix (rare) also does not,
        # because it can only attend to cached prefix + itself in the correct order;
        # XDC only issues prefill once followed by single-token steps.
        is_causal = past_kv is None and t > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.proj(y))
        present = (k, v) if use_cache else None
        return y, present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        a, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    """Monolithic model used for local training and correctness benchmarks."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv=None,
        pos_offset: int = 0,
        use_cache: bool = False,
        hidden: Optional[torch.Tensor] = None,
    ):
        if hidden is None:
            b, t = idx.shape
            if pos_offset + t > self.cfg.block_size:
                raise ValueError("sequence exceeds model block_size")
            pos = torch.arange(pos_offset, pos_offset + t, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        else:
            x = hidden
        presents = []
        for i, block in enumerate(self.blocks):
            pk = past_kv[i] if past_kv is not None else None
            x, present = block(x, past_kv=pk, use_cache=use_cache)
            presents.append(present)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.eval()
        past = None
        pos = 0
        for _ in range(max_new_tokens):
            if past is None:
                idx_cond = idx
                pos = 0
            else:
                idx_cond = idx[:, -1:]
                pos = idx.shape[1] - 1
            if pos + idx_cond.shape[1] > self.cfg.block_size:
                idx_cond = idx[:, -(self.cfg.block_size - max(pos, 0)) :]
                pos = max(pos, 0)
            logits, _, present = self.forward(
                idx_cond, past_kv=past, pos_offset=pos, use_cache=True
            )
            past = present
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    vals, _ = torch.topk(logits, k)
                    logits[logits < vals[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


class PipelineStage(nn.Module):
    """A contiguous slice of GPT.

    Stage 0 owns token+position embeddings; the final stage owns the final norm and
    LM head. Intermediate stages exchange only the hidden activation.
    """

    def __init__(self, cfg: ModelConfig, layer_start: int, layer_end: int):
        super().__init__()
        cfg.validate()
        if not (0 <= layer_start < layer_end <= cfg.n_layer):
            raise ValueError("invalid layer range")
        self.cfg = cfg
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = layer_start == 0
        self.is_last = layer_end == cfg.n_layer
        if self.is_first:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(layer_end - layer_start)]
        )
        if self.is_last:
            self.norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def layer_keys(self, global_state: dict) -> dict:
        """Extract this stage's parameters from a monolithic state dict."""
        out = {}
        if self.is_first:
            for key in ("tok_emb.weight", "pos_emb.weight"):
                if key in global_state:
                    out[key] = global_state[key]
        for local_i in range(self.layer_end - self.layer_start):
            global_i = self.layer_start + local_i
            prefix = f"blocks.{global_i}."
            suffix = f"blocks.{local_i}."
            for key, value in global_state.items():
                if key.startswith(prefix):
                    out[suffix + key[len(prefix) :]] = value
        if self.is_last:
            for key in ("norm.weight", "norm.bias", "head.weight"):
                if key in global_state:
                    out[key] = global_state[key]
        return out

    def load_global_state(self, global_state: dict) -> None:
        own = self.layer_keys(global_state)
        missing, unexpected = self.load_state_dict(own, strict=False)
        if missing or unexpected:
            raise ValueError(f"bad pipeline state slice: missing={missing}, unexpected={unexpected}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        past_kv=None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ):
        if self.is_first:
            if input_ids is None:
                raise ValueError("first stage requires input_ids")
            b, t = input_ids.shape
            if pos_offset + t > self.cfg.block_size:
                raise ValueError("sequence exceeds model block_size")
            pos = torch.arange(pos_offset, pos_offset + t, device=input_ids.device)
            x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :])
        else:
            if hidden is None:
                raise ValueError("non-first stage requires hidden states")
            x = hidden
        presents = []
        for i, block in enumerate(self.blocks):
            pk = past_kv[i] if past_kv is not None else None
            x, present = block(x, past_kv=pk, use_cache=use_cache)
            presents.append(present)
        logits = None
        if self.is_last:
            x = self.norm(x)
            logits = self.head(x)
        if use_cache:
            return x, logits, presents
        return x, logits
