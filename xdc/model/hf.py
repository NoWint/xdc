"""Safetensors-only Hugging Face model storage and true stage extraction.

Workers never download a full checkpoint for pipeline training.  The coordinator
holds the full model (it is the parameter server) and writes one contiguous
safetensors file per stage; a worker reads only that file.  This module also
contains the small amount of Hugging Face Hub plumbing needed to fetch a model
without importing remote Python code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from .checkpoint import save_state
from .llama import LlamaConfig, LlamaStage, apply_lora, trainable_parameters
from .tokenizer import TokenizerInfo, tokenizer_info_from_dir

ALLOWED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
)


class HFModelError(RuntimeError):
    pass


def _is_safetensor(name: str | Path) -> bool:
    return str(name).endswith(".safetensors")


def discover_safetensors(model_dir: str | Path) -> dict[str, str]:
    """Return ``{tensor_key: filename}`` for a local model directory."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise HFModelError("model.safetensors.index.json has no weight_map")
        files = {}
        for key, filename in weight_map.items():
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise HFModelError("unsafe shard filename in safetensors index")
            if not (model_dir / filename).exists():
                raise HFModelError(f"missing safetensors shard {filename}")
            files[key] = filename
        return files
    candidates = [p for p in model_dir.iterdir() if p.is_file() and _is_safetensor(p)]
    if not candidates:
        raise HFModelError(
            f"{model_dir} contains no .safetensors weights. XDC refuses pickle-based "
            "pytorch_model.bin checkpoints because loading them executes arbitrary code."
        )
    # Prefer the conventional single-file name, otherwise use the first shard.
    candidates.sort(key=lambda p: (p.name != "model.safetensors", p.name))
    out = {}
    for path in candidates:
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                out[key] = path.name
    return out


class SafetensorStore:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.files = discover_safetensors(self.model_dir)
        self._handles: dict[str, object] = {}

    def keys(self) -> Iterable[str]:
        return self.files.keys()

    def has(self, key: str) -> bool:
        return key in self.files

    def get(self, key: str) -> torch.Tensor:
        filename = self.files.get(key)
        if filename is None:
            raise KeyError(key)
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(str(self.model_dir / filename), framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_model_config(model_dir: str | Path) -> LlamaConfig:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise HFModelError(f"config.json not found in {model_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return LlamaConfig.from_dict(data)


def validate_model_dir(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    cfg = load_model_config(model_dir)
    store = SafetensorStore(model_dir)
    try:
        required = expected_global_keys(cfg)
        missing = [k for k in required if k not in store.files]
        # A tied LM head may be absent from the checkpoint; it is derived from
        # the embedding matrix on the final stage.
        if "lm_head.weight" in missing and cfg.tie_word_embeddings:
            missing.remove("lm_head.weight")
        if missing:
            raise HFModelError(f"checkpoint is missing {len(missing)} required tensors, e.g. {missing[:5]}")
        tok = tokenizer_info_from_dir(model_dir)
        if tok.kind == "hf" and not (model_dir / tok.path).exists():
            raise HFModelError("tokenizer.json referenced by the model is missing")
        return {
            "config": cfg.to_dict(),
            "parameters": cfg.num_parameters(),
            "tensors": len(store.files),
            "safetensors_bytes": sum(
                (model_dir / name).stat().st_size for name in set(store.files.values())
            ),
            "tokenizer": tok.to_dict(),
        }
    finally:
        store.close()


def expected_global_keys(cfg: LlamaConfig) -> list[str]:
    """Exact parameter keys for this architecture, without allocating tensors."""
    with torch.device("meta"):
        stage = LlamaStage(cfg, 0, cfg.num_hidden_layers)
    keys = []
    for local in stage.state_dict().keys():
        global_key = stage.global_key(local)
        if global_key:
            keys.append(global_key)
    return keys


def full_state(store: SafetensorStore, cfg: LlamaConfig) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key in expected_global_keys(cfg):
        if key == "lm_head.weight" and not store.has(key):
            if not cfg.tie_word_embeddings:
                raise HFModelError("lm_head.weight is missing but embeddings are not tied")
            # Tied models derive the head from the embedding at load time; do
            # not duplicate the tensor in the global checkpoint.
            continue
        state[key] = store.get(key).to(torch.float32).contiguous()
    return state


def stage_state(store: SafetensorStore, cfg: LlamaConfig, start: int, end: int) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        stage = LlamaStage(cfg, start, end)
    out: dict[str, torch.Tensor] = {}
    for local in stage.state_dict().keys():
        global_key = stage.global_key(local)
        if global_key is None:
            continue
        if global_key == "lm_head.weight" and not store.has(global_key):
            if not cfg.tie_word_embeddings:
                raise HFModelError("lm_head.weight is missing but embeddings are not tied")
            tensor = store.get("model.embed_tokens.weight").clone()
        else:
            try:
                tensor = store.get(global_key)
            except KeyError as exc:
                raise HFModelError(f"checkpoint is missing {global_key}") from exc
        out[local] = tensor.to(torch.float32).contiguous()
    return out


def write_stage_file(
    store: SafetensorStore,
    cfg: LlamaConfig,
    start: int,
    end: int,
    path: str | Path,
    tokenizer: TokenizerInfo | None = None,
) -> str:
    state = stage_state(store, cfg, start, end)
    meta = {
        "kind": "xdc-pipeline-stage",
        "format": "hf-llama",
        "layer_start": start,
        "layer_end": end,
        "config": json.dumps(cfg.to_dict(), separators=(",", ":")),
        "tokenizer": json.dumps((tokenizer or TokenizerInfo()).to_dict(), separators=(",", ":")),
        "parameters": sum(int(t.numel()) for t in state.values()),
    }
    return save_state(path, state, meta)


def write_full_file(store: SafetensorStore, cfg: LlamaConfig, path: str | Path) -> str:
    state = full_state(store, cfg)
    return save_state(
        path,
        state,
        {
            "kind": "xdc-hf-global",
            "format": "hf-llama",
            "config": json.dumps(cfg.to_dict(), separators=(",", ":")),
            "parameters": sum(int(t.numel()) for t in state.values()),
        },
    )


def estimate_stage_bytes(
    cfg: LlamaConfig,
    start: int,
    end: int,
    *,
    lora_rank: int = 0,
    batch_size: int = 1,
    seq_len: int = 128,
    bytes_per_param: int = 4,
) -> dict:
    """Static memory estimate used by the scheduler before assigning a stage.

    Includes parameters, gradients, Adam moments and a conservative activation
    allowance.  It is intentionally an estimate: workers report measured bytes
    after loading, and the acceptance report uses those measured values.
    """
    with torch.device("meta"):
        stage = LlamaStage(cfg, start, end)
        if lora_rank > 0:
            apply_lora(stage, r=lora_rank, alpha=float(lora_rank * 2))
    params = sum(int(p.numel()) for p in stage.parameters())
    if lora_rank > 0:
        lora_params = sum(int(p.numel()) for p in trainable_parameters(stage))
    else:
        lora_params = params
    activation = batch_size * seq_len * cfg.hidden_size * 4 * 8
    param_bytes = params * bytes_per_param
    grad_bytes = lora_params * 4
    optimizer_bytes = lora_params * 8
    return {
        "parameters": params,
        "trainable_parameters": lora_params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": activation,
        "total_bytes": param_bytes + grad_bytes + optimizer_bytes + activation,
    }


# ---------------------------------------------------------------------- hub
def pull_model(repo_id: str, dest: str | Path, revision: str | None = None) -> dict:
    """Download config/tokenizer/safetensors into ``dest``.

    Only an allow-list is fetched and remote Python code is never executed.
    Returns the validation summary.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise HFModelError(
            "model download requires the optional dependency: pip install 'xdc[hf]'"
        ) from exc
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt", "*.msgpack"],
    )
    return validate_model_dir(dest)


def copy_model_files(src: str | Path, dest: str | Path) -> list[str]:
    """Copy only allow-listed files from a local model directory."""
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        if path.name in ALLOWED_MODEL_FILES or (path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json"):
            shutil.copy2(path, dest / path.name)
            copied.append(path.name)
    return copied
