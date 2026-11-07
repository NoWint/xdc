"""Byte-level tokenisation and deterministic local data batching."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def text_to_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def ids_to_text(ids: list[int]) -> str:
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


def load_datasets(path: str | Path) -> bytes:
    """Load a UTF-8 text file or a JSONL instruction dataset into raw bytes.

    JSONL records may contain any of: ``text``, ``prompt`` + ``response``,
    ``instruction`` + ``input`` + ``output``.  Records are separated by two
    newlines so the model learns document boundaries.
    """
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        chunks: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if isinstance(rec, str):
                    chunks.append(rec)
                    continue
                if "text" in rec:
                    chunks.append(str(rec["text"]))
                elif "prompt" in rec or "response" in rec:
                    chunks.append(f"{rec.get('prompt', '')}{rec.get('response', '')}")
                elif "instruction" in rec or "output" in rec:
                    user = str(rec.get("instruction", ""))
                    inp = str(rec.get("input", ""))
                    out = str(rec.get("output", ""))
                    chunks.append(f"### Instruction\n{user}\n### Input\n{inp}\n### Response\n{out}")
                else:
                    chunks.append(json.dumps(rec, ensure_ascii=False))
        return "\n\n".join(chunks).encode("utf-8")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"dataset {path} is empty")
    return data


def split_train_val(data: bytes, val_fraction: float = 0.1, min_val_bytes: int = 512) -> tuple[bytes, bytes]:
    if not data:
        raise ValueError("empty dataset")
    val_size = max(min_val_bytes, int(len(data) * val_fraction))
    val_size = min(val_size, max(1, len(data) // 2))
    return data[:-val_size], data[-val_size:]


def shard_bounds(total: int, rank: int, world: int) -> tuple[int, int]:
    if world < 1 or not (0 <= rank < world):
        raise ValueError("invalid shard rank/world")
    start = total * rank // world
    end = total * (rank + 1) // world
    return start, end


def sample_batch(
    data: bytes,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a deterministic next-token batch from a byte shard."""
    span = len(data) - (seq_len + 1)
    if span <= 0:
        raise ValueError(f"data shard too small ({len(data)} bytes) for seq_len={seq_len}")
    ids = torch.empty((batch_size, seq_len + 1), dtype=torch.long)
    for b in range(batch_size):
        start = int(torch.randint(0, span + 1, (1,), generator=generator).item())
        chunk = data[start : start + seq_len + 1]
        ids[b] = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()
    ids = ids.to(device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def val_batches(
    data: bytes, seq_len: int, max_batches: int = 8, seed: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed validation batches, so loss numbers are comparable between rounds."""
    if len(data) < seq_len + 1:
        raise ValueError("validation split too small")
    gen = torch.Generator().manual_seed(seed)
    batches = []
    span = len(data) - (seq_len + 1)
    for _ in range(max_batches):
        start = int(torch.randint(0, span + 1, (1,), generator=gen).item())
        chunk = data[start : start + seq_len + 1]
        ids = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()[None, :]
        batches.append((ids[:, :-1].contiguous(), ids[:, 1:].contiguous()))
    return batches
