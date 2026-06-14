"""Plain continuation generation for pretrained structure-aware checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerConfig, StructTransformerModel
from parser.structure_annotator import StructureAnnotator
from parser.structure_states import PLAIN
from tokenizer.tokenizer_factory import load_tokenizer

DEFAULT_PROMPTS = [
    "人工智能是一门研究如何让计算机模拟人类智能的技术，",
    "在现代计算机系统中，操作系统的主要作用是",
    "Transformer 模型的核心思想是通过自注意力机制",
    "大语言模型的预训练阶段通常使用大规模文本语料，",
    "机器学习通常可以分为监督学习、无监督学习和强化学习，",
    "当训练数据质量较差时，模型可能会学习到噪声和重复模式，",
    '下面是一个 JSON 示例：\n{\n  "name": "张三",\n  "age":',
    "下面是一个 Markdown 代码块示例：\n\n```python\ndef hello():",
    "下面是一个 YAML 配置文件：\n\nmodel:\n  name: struct-transformer\n  layers:",
    "语言模型在生成长文本时，可能出现重复、主题漂移和逻辑断裂，",
]

SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint_config(model_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(resolve_path(model_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config is missing or invalid.")
    return config


def load_model(model_path: str | Path, tokenizer, device: torch.device) -> StructTransformerModel:
    checkpoint = torch.load(resolve_path(model_path), map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model_config = StructTransformerConfig(
        vocab_size=int(tokenizer.vocab_size),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.03)),
        lambda_state=float(cfg.get("lambda_state", 0.05)),
    )
    model = StructTransformerModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    return logits.masked_fill(logits < values[:, [-1]], -float("inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return logits

    adjusted = logits.clone()
    for token_id in set(int(idx) for idx in generated_ids):
        score = adjusted[:, token_id]
        adjusted[:, token_id] = torch.where(score < 0, score * penalty, score / penalty)
    return adjusted


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: list[int],
    ngram_size: int,
) -> torch.Tensor:
    if ngram_size <= 0 or len(generated_ids) + 1 < ngram_size:
        return logits

    prefix_size = ngram_size - 1
    current_prefix = tuple(generated_ids[-prefix_size:])
    banned_tokens: set[int] = set()
    for index in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[index : index + ngram_size])
        if ngram[:-1] == current_prefix:
            banned_tokens.add(int(ngram[-1]))

    if not banned_tokens:
        return logits

    adjusted = logits.clone()
    adjusted[:, list(banned_tokens)] = -float("inf")
    return adjusted


def annotate_token_ids(ids: list[int], tokenizer, annotator: StructureAnnotator) -> tuple[list[int], list[int]]:
    tokens: list[str] = []
    positions: list[int] = []
    for pos, token_id in enumerate(ids):
        token = tokenizer.id_to_token.get(int(token_id), tokenizer.unk_token)
        if token in SPECIAL_TOKENS:
            continue
        tokens.append(token)
        positions.append(pos)

    depth_ids = [0] * len(ids)
    state_ids = [PLAIN] * len(ids)
    if tokens:
        depths, states = annotator.annotate_tokens(tokens)
        for pos, depth, state in zip(positions, depths, states):
            depth_ids[pos] = depth
            state_ids[pos] = state
    return depth_ids, state_ids


@torch.no_grad()
def generate_response(
    model: StructTransformerModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.6,
    top_k: int | None = 40,
    repetition_penalty: float = 1.12,
    no_repeat_ngram_size: int = 4,
) -> str:
    annotator = StructureAnnotator(max_depth=model.config.max_depth)
    ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        full_ids = input_ids[0].tolist()
        depth_ids, state_ids = annotate_token_ids(full_ids, tokenizer, annotator)
        context_ids = full_ids[-model.config.block_size :]
        context_depths = depth_ids[-model.config.block_size :]
        context_states = state_ids[-model.config.block_size :]

        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        d = torch.tensor([context_depths], dtype=torch.long, device=device)
        s = torch.tensor([context_states], dtype=torch.long, device=device)
        logits = model(x, d, s)["lm_logits"][:, -1, :]
        logits = apply_repetition_penalty(logits, full_ids, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, full_ids, no_repeat_ngram_size)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = apply_top_k(logits / max(temperature, 1e-6), top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if int(next_id.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def build_tokenizer_from_checkpoint(cfg: dict[str, Any], args: argparse.Namespace):
    tokenizer_ref = args.tokenizer or cfg.get("tokenizer_name") or cfg.get("tokenizer_path")
    if tokenizer_ref is None:
        raise ValueError("Tokenizer was not provided and checkpoint config has no tokenizer_name/tokenizer_path.")
    hf_cache_dir = args.hf_cache_dir or cfg.get("hf_cache_dir")
    local_files_only = args.local_files_only or bool(cfg.get("hf_local_files_only", False))
    tokenizer_type = "hf" if "/" in str(tokenizer_ref) and not resolve_path(tokenizer_ref).exists() else "auto"
    return load_tokenizer(
        str(tokenizer_ref),
        PROJECT_ROOT,
        tokenizer_type=tokenizer_type,
        hf_cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_checkpoint_config(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = build_tokenizer_from_checkpoint(cfg, args)
    model = load_model(args.model, tokenizer, device)

    prompts = DEFAULT_PROMPTS if args.all_prompts else [args.prompt]
    if prompts == [None]:
        raise ValueError("Provide --prompt or use --all-prompts.")
    for prompt in prompts:
        assert prompt is not None
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print(
            generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=None if args.top_k <= 0 else args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        )


if __name__ == "__main__":
    main()
