"""Estimate structure-aware Transformer parameter counts.

The StructTransformerModel ties token_embedding and lm_head weights, so the
lm_head contributes no extra trainable matrix beyond the token embedding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer_factory import build_tokenizer


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_vocab_size(config: dict[str, Any], local_files_only: bool = False) -> tuple[int, str]:
    cfg = dict(config)
    if local_files_only:
        cfg["hf_local_files_only"] = True
    try:
        tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts=None)
        return int(tokenizer.vocab_size), "tokenizer"
    except Exception as exc:
        if "vocab_size" not in cfg:
            raise RuntimeError("Could not load tokenizer and config has no vocab_size fallback.") from exc
        return int(cfg["vocab_size"]), f"config fallback ({type(exc).__name__}: {exc})"


def estimate_struct_params(config: dict[str, Any], vocab_size: int) -> dict[str, int]:
    n_layer = int(config["n_layer"])
    n_embd = int(config["n_embd"])
    block_size = int(config["block_size"])
    max_depth = int(config.get("max_depth", 32))
    num_states = int(config.get("num_states", 9))

    token_embedding = vocab_size * n_embd
    position_embedding = block_size * n_embd
    depth_embedding = (max_depth + 1) * n_embd
    state_embedding = num_states * n_embd

    # qkv/proj attention + 4x MLP + two LayerNorm modules.
    per_block = 12 * n_embd * n_embd + 13 * n_embd
    transformer_blocks = n_layer * per_block

    final_layer_norm = 2 * n_embd
    lm_head_extra = 0
    depth_head = n_embd * (max_depth + 1) + (max_depth + 1)
    state_head = n_embd * num_states + num_states

    total = (
        token_embedding
        + position_embedding
        + depth_embedding
        + state_embedding
        + transformer_blocks
        + final_layer_norm
        + lm_head_extra
        + depth_head
        + state_head
    )

    return {
        "total_parameters": total,
        "trainable_parameters": total,
        "token_embedding_parameters": token_embedding,
        "position_embedding_parameters": position_embedding,
        "depth_embedding_parameters": depth_embedding,
        "state_embedding_parameters": state_embedding,
        "position_depth_state_embedding_parameters": position_embedding + depth_embedding + state_embedding,
        "transformer_block_parameters": transformer_blocks,
        "lm_head_parameters": lm_head_extra,
        "depth_head_parameters": depth_head,
        "state_head_parameters": state_head,
        "final_layer_norm_parameters": final_layer_norm,
    }


def format_count(value: int) -> str:
    return f"{value:,} ({value / 1_000_000:.2f}M)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/struct_pretrain_200m_zh.yaml")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    vocab_size, source = resolve_vocab_size(config, local_files_only=args.local_files_only)
    counts = estimate_struct_params(config, vocab_size)

    print(f"config: {args.config}")
    print(f"vocab_size: {vocab_size:,} ({source})")
    print(f"total parameters: {format_count(counts['total_parameters'])}")
    print(f"trainable parameters: {format_count(counts['trainable_parameters'])}")
    print(f"token embedding parameters: {format_count(counts['token_embedding_parameters'])}")
    print(
        "position/depth/state embedding parameters: "
        f"{format_count(counts['position_depth_state_embedding_parameters'])}"
    )
    print(f"  position embedding: {format_count(counts['position_embedding_parameters'])}")
    print(f"  depth embedding: {format_count(counts['depth_embedding_parameters'])}")
    print(f"  state embedding: {format_count(counts['state_embedding_parameters'])}")
    print(f"transformer block parameters: {format_count(counts['transformer_block_parameters'])}")
    print(f"lm_head parameters: {format_count(counts['lm_head_parameters'])} (tied to token embedding)")
    print(f"depth_head parameters: {format_count(counts['depth_head_parameters'])}")
    print(f"state_head parameters: {format_count(counts['state_head_parameters'])}")


if __name__ == "__main__":
    main()
