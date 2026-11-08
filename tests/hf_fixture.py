"""Tiny, dependency-light Hugging Face model fixtures for pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from xdc.model.llama import LlamaConfig, LlamaModel

WORDS = [
    "the", "a", "and", "to", "of", "it", "was", "he", "she", "they", "in", "that",
    "his", "her", "you", "I", "we", "on", "for", "with", "as", "had", "but", "not",
    "at", "said", "be", "my", "me", "what", "all", "so", "up", "out", "one", "about",
    "there", "if", "them", "some", "could", "time", "like", "into", "just", "him",
    "know", "take", "people", "good", "see", "no", "its", "who", "now", "go", "get",
    "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work",
    "seem", "feel", "try", "leave", "call", "cat", "dog", "day", "night", "little",
    "big", "said", "then", "when", "where", "why", "how",
]


def _tokenizer_json(vocab_size: int = 128) -> dict:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for index, word in enumerate(WORDS[: vocab_size - 3], start=3):
        vocab.setdefault(word, index)
    added = [
        {
            "id": i,
            "content": word,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": word in {"<unk>", "<s>", "</s>"},
        }
        for word, i in vocab.items()
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def build_tiny_hf(
    dest: str | Path,
    *,
    layers: int = 4,
    hidden: int = 64,
    heads: int = 4,
    kv_heads: int = 2,
    intermediate: int = 128,
    vocab: int = 128,
    max_positions: int = 128,
    tied: bool = False,
    seed: int = 0,
) -> LlamaConfig:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=hidden // heads,
        max_position_embeddings=max_positions,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=tied,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    model = LlamaModel(cfg)
    state = model.state_dict_global()
    if tied:
        # Match real tied checkpoints: the LM head is absent and derived from
        # the embedding at load time.
        state.pop("lm_head.weight", None)
    (dest / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (dest / "tokenizer.json").write_text(
        json.dumps(_tokenizer_json(vocab)), encoding="utf-8"
    )
    save_file({k: v.contiguous() for k, v in state.items()}, str(dest / "model.safetensors"))
    return cfg
