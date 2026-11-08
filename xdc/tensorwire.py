"""Versioned, pickle-free tensor wire encoding for tunnel RPC."""

from __future__ import annotations

import base64

import numpy as np


def encode_array(arr: np.ndarray, dtype: str = "float32") -> dict:
    dtype = {"fp32": "float32", "fp16": "float16"}.get(dtype, dtype)
    if dtype not in {"float32", "float16"}:
        raise ValueError(f"unsupported tensor wire dtype {dtype}")
    out = np.ascontiguousarray(arr, dtype=np.float32).astype(dtype)
    return {
        "shape": list(out.shape),
        "dtype": dtype,
        "data_b64": base64.b64encode(out.tobytes()).decode("ascii"),
    }


def decode_array(obj: dict) -> np.ndarray:
    dtype = obj.get("dtype", "float32")
    if dtype not in {"float32", "float16"}:
        raise ValueError("unsupported tensor wire dtype")
    raw = base64.b64decode(obj["data_b64"])
    shape = tuple(int(x) for x in obj["shape"])
    expected = 1
    for dim in shape:
        expected *= dim
    itemsize = 4 if dtype == "float32" else 2
    if len(raw) != expected * itemsize:
        raise ValueError("tensor wire length mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.float32)
