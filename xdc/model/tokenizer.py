"""Tokenizer adapters and on-disk token datasets.

XDC never ships model-specific tokenisation in the training code path.  A model
directory declares either the built-in byte tokenizer (the TinyGPT test model) or
a Hugging Face ``tokenizer.json``.  Datasets are converted once by the
coordinator into a flat ``uint32`` token file; workers only ever receive token
shards, never tokenizer code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

TOKEN_DTYPE = np.dtype("<u4")
BYTE_VOCAB = 256


@dataclass
class TokenizerInfo:
    kind: str = "bytes"
    vocab_size: int = BYTE_VOCAB
    path: str = ""
    sha256: str = ""
    eos_token_id: int | None = None
    bos_token_id: int | None = None
    add_bos: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "path": self.path,
            "sha256": self.sha256,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "add_bos": self.add_bos,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TokenizerInfo":
        data = data or {}
        return cls(
            kind=str(data.get("kind", "bytes")),
            vocab_size=int(data.get("vocab_size", BYTE_VOCAB)),
            path=str(data.get("path", "")),
            sha256=str(data.get("sha256", "")),
            eos_token_id=data.get("eos_token_id"),
            bos_token_id=data.get("bos_token_id"),
            add_bos=bool(data.get("add_bos", False)),
        )


class ByteTokenizer:
    kind = "bytes"
    vocab_size = BYTE_VOCAB

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")

    def encode_bytes(self, data: bytes) -> list[int]:
        return list(data)


class HFTokenizer:
    """Thin wrapper around the Rust ``tokenizers`` library.

    Only ``tokenizer.json`` is loaded.  No Python tokenizer module from a model
    repository is imported, which keeps ``trust_remote_code`` out of the design.
    """

    kind = "hf"

    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Hugging Face tokenizers require the optional dependency: pip install 'xdc[hf]'"
            ) from exc
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {path}")
        self.path = path
        self._tok = Tokenizer.from_file(str(path))
        self.vocab_size = int(self._tok.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def encode_bytes(self, data: bytes) -> list[int]:
        return self.encode(data.decode("utf-8", errors="replace"))


def load_tokenizer(model_dir: str | Path, info: TokenizerInfo | None = None) -> ByteTokenizer | HFTokenizer:
    model_dir = Path(model_dir)
    if info is None:
        info = tokenizer_info_from_dir(model_dir)
    if info.kind == "bytes":
        return ByteTokenizer()
    if info.kind == "hf":
        path = Path(info.path)
        if not path.is_absolute():
            path = model_dir / path
        return HFTokenizer(path)
    raise ValueError(f"unknown tokenizer kind {info.kind!r}")


def tokenizer_info_from_dir(model_dir: str | Path) -> TokenizerInfo:
    model_dir = Path(model_dir)
    tok_json = model_dir / "tokenizer.json"
    if not tok_json.exists():
        return TokenizerInfo(kind="bytes")
    from .checkpoint import sha256_file

    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return TokenizerInfo(
        kind="hf",
        vocab_size=int(cfg.get("vocab_size", 0)) or 0,
        path=tok_json.name,
        sha256=sha256_file(tok_json),
        eos_token_id=cfg.get("eos_token_id"),
        bos_token_id=cfg.get("bos_token_id"),
        add_bos=False,
    )


def write_tokens(path: str | Path, ids: Iterable[int]) -> tuple[int, str]:
    """Write a flat little-endian uint32 token file atomically."""
    from .checkpoint import sha256_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(ids), dtype=TOKEN_DTYPE)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        arr.tofile(fh)
        fh.flush()
        import os

        os.fsync(fh.fileno())
    tmp.replace(path)
    return int(arr.size), sha256_file(path)


def read_tokens(path: str | Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=TOKEN_DTYPE)


def tokenize_text(text: str, tokenizer: ByteTokenizer | HFTokenizer) -> list[int]:
    return tokenizer.encode(text)


def tokenize_bytes(data: bytes, tokenizer: ByteTokenizer | HFTokenizer) -> list[int]:
    return tokenizer.encode_bytes(data)


def sample_token_batch(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    span = int(tokens.size) - (seq_len + 1)
    if span <= 0:
        raise ValueError(
            f"token shard too small ({int(tokens.size)} tokens) for seq_len={seq_len}"
        )
    starts = torch.randint(0, span + 1, (batch_size,), generator=generator)
    x = torch.empty((batch_size, seq_len), dtype=torch.long)
    y = torch.empty((batch_size, seq_len), dtype=torch.long)
    for i, start in enumerate(starts.tolist()):
        chunk = torch.from_numpy(tokens[start : start + seq_len + 1].astype(np.int64))
        x[i] = chunk[:-1]
        y[i] = chunk[1:]
    return x, y


def fixed_token_batches(
    tokens: np.ndarray,
    seq_len: int,
    max_batches: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic validation batches for comparable loss values."""
    if int(tokens.size) < seq_len + 1:
        raise ValueError("validation token split too small")
    gen = torch.Generator().manual_seed(seed)
    span = int(tokens.size) - (seq_len + 1)
    out = []
    for _ in range(max_batches):
        start = int(torch.randint(0, span + 1, (1,), generator=gen).item())
        chunk = torch.from_numpy(tokens[start : start + seq_len + 1].astype(np.int64))
        out.append((chunk[:-1][None, :], chunk[1:][None, :]))
    return out


def tokens_from_safetensors_meta(meta: dict) -> TokenizerInfo | None:
    raw = meta.get("xdc.tokenizer")
    if not raw:
        return None
    try:
        return TokenizerInfo.from_dict(json.loads(raw))
    except Exception:
        return None
