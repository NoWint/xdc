"""Optional C++ acceleration facade with explicit Python fallbacks."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from . import _native  # type: ignore

    AVAILABLE = True
    VERSION = _native.version()
except Exception as _exc:  # pragma: no cover - depends on local compiler
    _native = None
    AVAILABLE = False
    VERSION = f"python-fallback ({_exc.__class__.__name__})"


def aggregate(
    base: np.ndarray,
    updates: list[np.ndarray],
    weights: list[float] | np.ndarray,
    momentum_prev: Optional[np.ndarray],
    outer_lr: float,
    momentum: float,
    nesterov: bool,
    max_norm_ratio: float,
):
    if not AVAILABLE:
        raise RuntimeError("native extension unavailable")
    if not isinstance(weights, np.ndarray):
        weights = np.asarray(weights, dtype=np.float64)
    return _native.aggregate(
        np.ascontiguousarray(base, dtype=np.float32),
        [np.ascontiguousarray(u, dtype=np.float32) for u in updates],
        weights,
        None if momentum_prev is None else np.ascontiguousarray(momentum_prev, dtype=np.float32),
        float(outer_lr),
        float(momentum),
        bool(nesterov),
        float(max_norm_ratio),
    )


def aggregate_tensors(
    base_arrays: list[np.ndarray],
    update_arrays: list[list[np.ndarray]],
    weights: list[float] | np.ndarray,
    momentum_prev: list[np.ndarray] | None,
    outer_lr: float,
    momentum: float,
    nesterov: bool,
    max_norm_ratio: float,
):
    if not AVAILABLE:
        raise RuntimeError("native extension unavailable")
    if not isinstance(weights, np.ndarray):
        weights = np.asarray(weights, dtype=np.float64)
    return _native.aggregate_tensors(
        [np.ascontiguousarray(a, dtype=np.float32) for a in base_arrays],
        [[np.ascontiguousarray(a, dtype=np.float32) for a in row] for row in update_arrays],
        weights,
        None if momentum_prev is None else [np.ascontiguousarray(a, dtype=np.float32) for a in momentum_prev],
        float(outer_lr),
        float(momentum),
        bool(nesterov),
        float(max_norm_ratio),
    )


def topk_sample(logits: np.ndarray, top_k: int, temperature: float, seed: int) -> Optional[int]:
    if not AVAILABLE:
        return None
    return int(
        _native.topk_sample(
            np.ascontiguousarray(logits, dtype=np.float32).reshape(-1),
            int(top_k),
            float(temperature),
            int(seed) & 0xFFFFFFFFFFFFFFFF,
        )
    )


def byte_batch(data: bytes, batch_size: int, seq_len: int, seed: int) -> Optional[np.ndarray]:
    if not AVAILABLE:
        return None
    return _native.byte_batch(
        np.frombuffer(data, dtype=np.uint8), int(batch_size), int(seq_len), int(seed) & 0xFFFFFFFFFFFFFFFF
    )


def quantize_int8(flat: np.ndarray, block_size: int = 4096):
    if not AVAILABLE:
        raise RuntimeError("native extension unavailable")
    return _native.quantize_int8(np.ascontiguousarray(flat, dtype=np.float32), int(block_size))


def dequantize_int8(codes: np.ndarray, scales: np.ndarray, block_size: int = 4096):
    if not AVAILABLE:
        raise RuntimeError("native extension unavailable")
    return _native.dequantize_int8(codes, scales, int(block_size))
