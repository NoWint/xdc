"""Model registry: one place that knows how a job's model is sliced.

A job model dictionary is either the built-in TinyGPT test model or an HF Llama
family checkpoint stored under the coordinator's model directory.  Both expose
the same stage surface used by pipeline training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .gpt import ModelConfig, PipelineStage
from .hf import (
    HFModelError,
    SafetensorStore,
    estimate_stage_bytes,
    load_model_config,
    stage_state,
)
from .llama import LlamaConfig, LlamaStage, apply_lora, trainable_parameters


def model_kind(spec: Mapping[str, Any]) -> str:
    kind = spec.get("kind")
    if kind in {"tinygpt", "hf"}:
        return str(kind)
    if "hidden_size" in spec and "num_hidden_layers" in spec:
        return "hf"
    return "tinygpt"


def model_config(spec: Mapping[str, Any]) -> ModelConfig | LlamaConfig:
    if model_kind(spec) == "hf":
        return LlamaConfig.from_dict(spec)
    return ModelConfig.from_dict(dict(spec))


def num_parameters(spec: Mapping[str, Any]) -> int:
    cfg = model_config(spec)
    return int(cfg.num_parameters()) if isinstance(cfg, LlamaConfig) else _tiny_parameters(cfg)


def _tiny_parameters(cfg: ModelConfig) -> int:
    with torch.device("meta"):
        model = PipelineStage(cfg, 0, cfg.n_layer)
    return sum(int(p.numel()) for p in model.parameters())


def build_stage(kind: str, cfg: ModelConfig | LlamaConfig, start: int, end: int):
    if kind == "hf":
        return LlamaStage(cfg, start, end)  # type: ignore[arg-type]
    return PipelineStage(cfg, start, end)  # type: ignore[arg-type]


def stage_state_from_global(
    kind: str,
    cfg: ModelConfig | LlamaConfig,
    global_state: Mapping[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    if kind == "hf":
        stage = LlamaStage(cfg, start, end)  # type: ignore[arg-type]
        return stage.layer_state(global_state)
    stage = PipelineStage(cfg, start, end)  # type: ignore[arg-type]
    return stage.layer_keys(dict(global_state))


def stage_layer_count(cfg: ModelConfig | LlamaConfig) -> int:
    return int(cfg.n_layer) if isinstance(cfg, ModelConfig) else int(cfg.num_hidden_layers)


def apply_lora_if_requested(module, rank: int, alpha: float) -> int:
    if rank <= 0:
        return 0
    if isinstance(module, LlamaStage):
        return apply_lora(module, r=rank, alpha=alpha)
    # TinyGPT's projection names differ; adapt the attention/MLP linear layers.
    return apply_lora(
        module,
        r=rank,
        alpha=alpha,
        targets=("qkv", "proj", "fc1", "fc2"),
    )


def trainable_state(module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def count_trainable(module) -> int:
    return sum(int(p.numel()) for p in trainable_parameters(module))


def estimate_stage(
    kind: str,
    cfg: ModelConfig | LlamaConfig,
    start: int,
    end: int,
    *,
    lora_rank: int,
    batch_size: int,
    seq_len: int,
) -> dict:
    if kind == "hf":
        return estimate_stage_bytes(  # type: ignore[arg-type]
            cfg,
            start,
            end,
            lora_rank=lora_rank,
            batch_size=batch_size,
            seq_len=seq_len,
        )
    with torch.device("meta"):
        stage = PipelineStage(cfg, 0, cfg.n_layer)  # type: ignore[arg-type]
        total = sum(int(p.numel()) for p in stage.parameters())
        trainable = total
        if lora_rank > 0:
            adapted = PipelineStage(cfg, 0, cfg.n_layer)  # type: ignore[arg-type]
            apply_lora_if_requested(adapted, lora_rank, float(lora_rank * 2))
            trainable = count_trainable(adapted)
    hidden = int(cfg.n_embd)  # type: ignore[attr-defined]
    activation = batch_size * seq_len * hidden * 4 * 8
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "param_bytes": total * 4,
        "grad_bytes": trainable * 4,
        "optimizer_bytes": trainable * 8,
        "activation_bytes": activation,
        "total_bytes": total * 4 + trainable * 12 + activation,
    }


def resolve_model_dir(spec: Mapping[str, Any], data_dir: Path) -> Path:
    """Resolve a model directory declared in a job spec.

    ``model_name`` is a directory under the coordinator model root.  Absolute
    paths are accepted only when they stay inside that root.
    """
    root = (data_dir / "models").resolve()
    raw = spec.get("model_dir") or spec.get("model_name")
    if not raw:
        raise HFModelError("HF model spec requires model_name")
    path = Path(str(raw))
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root not in path.parents and path != root:
        raise HFModelError("model directory escapes the coordinator model root")
    if not path.is_dir():
        raise HFModelError(f"model directory not found: {path}")
    return path


def load_hf_model_bundle(spec: Mapping[str, Any], data_dir: Path) -> tuple[Path, LlamaConfig]:
    path = resolve_model_dir(spec, data_dir)
    cfg = load_model_config(path)
    return path, cfg


def full_state_from_dir(path: Path, cfg: LlamaConfig) -> dict[str, torch.Tensor]:
    with SafetensorStore(path) as store:
        from .hf import full_state

        return full_state(store, cfg)


def stage_state_from_dir(path: Path, cfg: LlamaConfig, start: int, end: int) -> dict[str, torch.Tensor]:
    with SafetensorStore(path) as store:
        return stage_state(store, cfg, start, end)
