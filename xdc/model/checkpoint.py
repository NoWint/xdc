"""Safe tensor checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load, load_file, save, save_file


def tensor_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        out[key] = value.detach().to("cpu", dtype=torch.float32).contiguous()
    return out


def save_state(path: str | Path, state: Mapping[str, torch.Tensor], meta: dict | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = tensor_state(state)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    save_file(clean, str(tmp), metadata={k: str(v) for k, v in (meta or {}).items()})
    os.replace(tmp, path)
    return sha256_file(path)


def load_state(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return load_file(str(path), device=str(device))


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def serialize_state_bytes(state: Mapping[str, torch.Tensor], dtype=None) -> bytes:
    """Serialize a tensor dict to safetensors bytes in-memory.

    ``dtype`` enables network compression (e.g. ``torch.float16``) without a
    lossy custom format.
    """
    clean = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        t = value.detach().to("cpu").contiguous()
        if dtype is not None:
            t = t.to(dtype)
        clean[key] = t
    return save(clean)


def deserialize_state_bytes(data: bytes, device: str = "cpu") -> dict[str, torch.Tensor]:
    values = load(data)
    return {k: v.to(device) for k, v in values.items()}
