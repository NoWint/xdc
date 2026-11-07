"""Versioned, pickle-free tensor wire encoding for tunnel RPC."""

from __future__ import annotations

import base64

import numpy as np


def encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": "float32",
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(obj: dict) -> np.ndarray:
    if obj.get("dtype", "float32") != "float32":
        raise ValueError("unsupported tensor wire dtype")
    raw = base64.b64decode(obj["data_b64"])
    shape = tuple(int(x) for x in obj["shape"])
    expected = 1
    for dim in shape:
        expected *= dim
    if len(raw) != expected * 4:
        raise ValueError("tensor wire length mismatch")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape)
