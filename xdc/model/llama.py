"""Self-contained Llama/Qwen-family model with true stage slicing.

XDC needs to construct *only one stage* of a transformer on a worker: loading a
full Hugging Face checkpoint first and then deleting layers would defeat the
memory-pooling goal.  This module therefore implements the architecture directly
from ``config.json`` and maps Hugging Face safetensors names into a stage's local
state dict.  It is not a general replacement for ``transformers``; it implements
the decoder-only Llama family that covers Llama 2/3, Qwen2/2.5, Qwen3 and
SmolLM2 checkpoints.

Supported weight layout (the standard HF layout):

    model.embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
    model.layers.{i}.mlp.{gate,up,down}_proj.{weight,bias}
    model.layers.{i}.{input,post_attention}_layernorm.weight
    model.layers.{i}.self_attn.{q,k}_norm.weight       (Qwen3)
    model.norm.weight
    lm_head.weight

The implementation is deliberately small and dependency-free beyond PyTorch so
that training and inference on Windows, macOS and Linux share one code path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 0
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    model_type: str = "llama"
    architectures: list[str] = field(default_factory=list)
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    qk_norm: bool = False
    sliding_window: int | None = None
    rope_scaling: dict | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.hidden_size < 1 or self.num_attention_heads < 1:
            raise ValueError("hidden_size and num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if self.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {self.hidden_act!r}; XDC implements silu/SwiGLU")
        if self.rope_scaling:
            # Llama 3.1-style frequency scaling changes the rotation.  Refuse to
            # silently produce wrong logits.
            raise ValueError("rope_scaling is not implemented; use a checkpoint with plain RoPE")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "LlamaConfig":
        get = data.get
        n_heads = int(get("num_attention_heads", 32))
        model_type = str(get("model_type", "llama"))
        # HF Qwen2 hard-codes q/k/v bias and a bias-free output projection.
        attention_bias = bool(get("attention_bias", model_type == "qwen2"))
        if model_type == "qwen2":
            attention_bias = True
        return cls(
            vocab_size=int(get("vocab_size", 32000)),
            hidden_size=int(get("hidden_size", 4096)),
            intermediate_size=int(get("intermediate_size", 11008)),
            num_hidden_layers=int(get("num_hidden_layers", 32)),
            num_attention_heads=n_heads,
            num_key_value_heads=int(get("num_key_value_heads", n_heads)),
            head_dim=int(get("head_dim", 0) or 0),
            max_position_embeddings=int(get("max_position_embeddings", 4096)),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(get("rope_theta", 10000.0)),
            attention_bias=attention_bias,
            mlp_bias=bool(get("mlp_bias", False)),
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            hidden_act=str(get("hidden_act", "silu")),
            model_type=model_type,
            architectures=list(get("architectures", []) or []),
            bos_token_id=get("bos_token_id"),
            eos_token_id=get("eos_token_id"),
            pad_token_id=get("pad_token_id"),
            qk_norm=bool(get("qk_norm", model_type == "qwen3")),
            sliding_window=get("sliding_window"),
            rope_scaling=get("rope_scaling"),
        )

    def num_parameters(self, include_embedding: bool = True) -> int:
        h, i = self.hidden_size, self.intermediate_size
        n = self.num_hidden_layers
        attn = h * (self.n_heads + 2 * self.n_kv_heads) * self.dim + (self.n_heads * self.dim) * h
        mlp = h * i * 3
        norms = 2 * h + (2 * self.dim if self.qk_norm else 0)
        emb = self.vocab_size * h if include_embedding else 0
        head = 0 if self.tie_word_embeddings else self.vocab_size * h
        return n * (attn + mlp + norms) + emb + h + head


# ---------------------------------------------------------------------- modules
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to ``[B, H, T, D]`` query/key tensors with ``cos`` [T, D]."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class LoRALinear(nn.Linear):
    """A frozen linear layer plus a trainable low-rank update.

    Subclassing ``nn.Linear`` keeps the base weight name (``weight``), so a
    standard Hugging Face safetensors checkpoint loads unchanged.  With
    ``lora_B`` initialised to zero the initial function is exactly the base
    model.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + update * self.scaling


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim
        q_bias = cfg.attention_bias
        o_bias = cfg.attention_bias and cfg.model_type != "qwen2"
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=q_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=q_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=o_bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # The cache keeps the un-expanded (GQA) keys and values, including the
        # prefix, so the next decode step can append to the full history.
        k_cache, v_cache = k, v
        if self.n_kv_heads != self.n_heads:
            # Repeat KV heads instead of torch>=2.5's enable_gqa flag so the
            # same path runs on the older PyTorch used by Intel macOS CI.
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # With no padding and a full prefix, PyTorch's causal kernel is exact.
        # A decode step has one query and a populated cache, so no mask is needed.
        causal = past is None and t > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        out = self.o_proj(y)
        present = (k_cache, v_cache) if use_cache else None
        return out, present


class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        h, present = self.self_attn(self.input_layernorm(x), cos, sin, past=past, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


# ---------------------------------------------------------------------- stages
class LlamaStage(nn.Module):
    """A contiguous slice of a Llama-family decoder.

    Stage 0 owns the token embedding, the final stage owns the norm and LM head.
    Intermediate stages hold exactly their own decoder layers; there is no hidden
    full-model copy.
    """

    def __init__(self, cfg: LlamaConfig, layer_start: int, layer_end: int):
        super().__init__()
        cfg.validate()
        if not (0 <= layer_start < layer_end <= cfg.num_hidden_layers):
            raise ValueError("invalid layer range")
        self.cfg = cfg
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_first = layer_start == 0
        self.is_last = layer_end == cfg.num_hidden_layers
        if self.is_first:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(cfg) for _ in range(layer_end - layer_start)]
        )
        if self.is_last:
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope_cache: dict[tuple[str, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    # ---------------------------------------------------------------- state
    def global_key(self, local_key: str) -> str | None:
        if local_key == "embed_tokens.weight":
            return "model.embed_tokens.weight"
        if local_key in {"norm.weight", "lm_head.weight"}:
            if local_key == "norm.weight":
                return "model.norm.weight"
            return "lm_head.weight"
        if local_key.startswith("layers."):
            _, idx, rest = local_key.split(".", 2)
            return f"model.layers.{self.layer_start + int(idx)}.{rest}"
        return None

    def local_key(self, global_key: str) -> str | None:
        if global_key == "model.embed_tokens.weight":
            return "embed_tokens.weight" if self.is_first else None
        if global_key == "model.norm.weight":
            return "norm.weight" if self.is_last else None
        if global_key == "lm_head.weight":
            return "lm_head.weight" if self.is_last else None
        prefix = "model.layers."
        if not global_key.startswith(prefix):
            return None
        parts = global_key[len(prefix) :].split(".", 1)
        if len(parts) != 2:
            return None
        idx = int(parts[0])
        if not (self.layer_start <= idx < self.layer_end):
            return None
        return f"layers.{idx - self.layer_start}.{parts[1]}"

    def layer_state(self, global_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in global_state.items():
            local = self.local_key(key)
            if local is not None:
                out[local] = value
        if self.is_last and "lm_head.weight" not in out and "model.embed_tokens.weight" in global_state:
            # The stage file must own its bytes; sharing the embedding tensor
            # would make safetensors reject the save.
            out["lm_head.weight"] = global_state["model.embed_tokens.weight"].clone()
        return out

    def load_global_state(self, global_state: Mapping[str, torch.Tensor]) -> None:
        local = self.layer_state(global_state)
        missing, unexpected = self.load_state_dict(local, strict=False)
        # rotary inv_freq keys, when present in an HF checkpoint, are not
        # parameters.  LoRA adapters may be absent from a base checkpoint; they
        # are initialised to a zero update, so missing adapter keys are harmless.
        missing = [k for k in missing if "rotary" not in k and "lora_" not in k]
        unexpected = [k for k in unexpected if "rotary" not in k]
        if missing or unexpected:
            raise ValueError(f"bad Llama stage slice: missing={missing[:8]} unexpected={unexpected[:8]}")

    # ---------------------------------------------------------------- rope
    def rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        key = (str(device), seq_len, dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        dim = self.cfg.dim
        inv_freq = 1.0 / (
            self.cfg.rope_theta
            ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        out = (emb.cos().to(dtype), emb.sin().to(dtype))
        if len(self._rope_cache) > 16:
            self._rope_cache.clear()
        self._rope_cache[key] = out
        return out

    # ---------------------------------------------------------------- forward
    def forward_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ):
        if self.is_first:
            if input_ids is None:
                raise ValueError("first stage requires input_ids")
            b, t = input_ids.shape
            x = self.embed_tokens(input_ids)
        else:
            if hidden is None:
                raise ValueError("non-first stage requires hidden states")
            x = hidden
            t = x.shape[1]
        total = pos_offset + t
        if total > self.cfg.max_position_embeddings:
            raise ValueError("sequence exceeds model max_position_embeddings")
        cos, sin = self.rope_cos_sin(total, x.device, x.dtype)
        cos = cos[pos_offset:total]
        sin = sin[pos_offset:total]
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = past_kv[i] if past_kv is not None else None
            x, present = layer(x, cos, sin, past=past, use_cache=use_cache)
            presents.append(present)
        return x, presents

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.is_last:
            raise ValueError("only the final stage has an LM head")
        return self.lm_head(self.norm(hidden))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        pos_offset: int = 0,
        use_cache: bool = False,
        targets: Optional[torch.Tensor] = None,
        loss_scale: float = 1.0,
    ):
        x, presents = self.forward_features(
            input_ids=input_ids,
            hidden=hidden,
            past_kv=past_kv,
            pos_offset=pos_offset,
            use_cache=use_cache,
        )
        logits = self.head_logits(x) if self.is_last else None
        loss = None
        if targets is not None:
            if logits is None:
                raise ValueError("targets are only accepted by the final stage")
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="mean"
            ) * loss_scale
        if use_cache:
            return x, logits, presents, loss
        return x, logits, loss


class LlamaModel(nn.Module):
    """Monolithic reference implementation used for evaluation and parity tests."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LlamaStage(cfg, 0, cfg.num_hidden_layers)

    def load_hf_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.model.load_global_state(state)

    def state_dict_global(self) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in self.model.state_dict().items():
            gkey = self.model.global_key(key)
            if gkey is None:
                continue
            if gkey == "lm_head.weight" and self.cfg.tie_word_embeddings:
                continue
            out[gkey] = value.detach().clone()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv=None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ):
        if use_cache:
            x, logits, presents, loss = self.model(
                input_ids=input_ids,
                past_kv=past_kv,
                pos_offset=pos_offset,
                use_cache=True,
                targets=targets,
            )
            return logits, loss, presents
        _, logits, loss = self.model(
            input_ids=input_ids,
            past_kv=past_kv,
            pos_offset=pos_offset,
            use_cache=False,
            targets=targets,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.eval()
        past = None
        out = input_ids
        for _ in range(max_new_tokens):
            if past is None:
                pos = 0
                feed = out
            else:
                pos = out.shape[1] - 1
                feed = out[:, -1:]
            if pos + feed.shape[1] > self.cfg.max_position_embeddings:
                break
            logits, _, present = self.forward(
                feed, past_kv=past, pos_offset=pos, use_cache=True
            )
            past = present
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < vals[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(
                    F.softmax(logits, -1), 1, generator=generator
                )
            out = torch.cat((out, next_id), dim=1)
        return out


# ---------------------------------------------------------------------- LoRA
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def apply_lora(
    module: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    targets: Iterable[str] = LORA_TARGETS,
    _freeze: bool = True,
) -> int:
    """Replace matching ``nn.Linear`` children with :class:`LoRALinear`.

    Returns the number of adapted projections.  LoRA freezes the whole base
    module (including embeddings, norms and the LM head); only ``lora_A`` and
    ``lora_B`` remain trainable.  Base weights keep their original parameter
    names, so HF safetensors mapping is unchanged.
    """
    targets = tuple(targets)
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and name in targets:
            new = LoRALinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                r=r,
                alpha=alpha,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None and new.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += apply_lora(child, r=r, alpha=alpha, targets=targets, _freeze=False)
    if _freeze and replaced:
        for name, param in module.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad_(False)
    return replaced


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach()
        for key, value in module.state_dict().items()
        if ".lora_A" in key or ".lora_B" in key or key.endswith("lora_A") or key.endswith("lora_B")
    }


def trainable_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    seen: set[int] = set()
    for param in module.parameters():
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            yield param
