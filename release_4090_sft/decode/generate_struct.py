
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


SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>"}


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


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered = logits
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(filtered, min(top_k, filtered.size(-1)))
        filtered = filtered.masked_fill(filtered < values[:, [-1]], -float("inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        filtered = filtered.masked_fill(indices_to_remove, -float("inf"))

    return filtered


def load_checkpoint_config(model_path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(PROJECT_ROOT / model_path, map_location="cpu", weights_only=False)
    return ckpt.get("config", {})


def annotate_token_ids(
    ids: list[int],
    tokenizer,
    annotator: StructureAnnotator,
) -> tuple[list[int], list[int]]:
    """
    根据当前完整 token id 序列重新计算 depth/state。
    注意：这里用完整上下文标注，然后再截取 block_size，
    避免 block 截断后丢失前文结构状态。
    """
    tokens = []
    positions = []

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


def load_model(
    model_path: str | Path,
    tokenizer,
    device: torch.device,
) -> StructTransformerModel:
    ckpt = torch.load(PROJECT_ROOT / model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model_config = StructTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.1)),
        lambda_state=float(cfg.get("lambda_state", 0.2)),
    )

    model = StructTransformerModel(model_config)
    model.load_state_dict(ckpt["model"])
    del ckpt
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_response(
    model: StructTransformerModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 160,
    temperature: float = 0.85,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    do_sample: bool = True,
) -> str:
    annotator = StructureAnnotator(max_depth=model.config.max_depth)

    ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        full_ids = input_ids[0].tolist()
        full_depths, full_states = annotate_token_ids(full_ids, tokenizer, annotator)

        context_ids = full_ids[-model.config.block_size:]
        context_depths = full_depths[-model.config.block_size:]
        context_states = full_states[-model.config.block_size:]

        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        d = torch.tensor([context_depths], dtype=torch.long, device=device)
        s = torch.tensor([context_states], dtype=torch.long, device=device)

        out = model(x, d, s)
        logits = out["lm_logits"][:, -1, :]
        logits = apply_repetition_penalty(logits, full_ids, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, full_ids, no_repeat_ngram_size)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            logits = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if int(next_id.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_cfg = load_checkpoint_config(args.model)
    tokenizer_ref = args.tokenizer or ckpt_cfg.get("tokenizer_name") or ckpt_cfg.get("tokenizer_path")
    if tokenizer_ref is None:
        raise ValueError("Tokenizer was not provided and checkpoint config has no tokenizer_name/tokenizer_path.")
    hf_cache_dir = args.hf_cache_dir or ckpt_cfg.get("hf_cache_dir")
    local_files_only = args.local_files_only or bool(ckpt_cfg.get("hf_local_files_only", False))
    tokenizer_type = "hf" if "/" in tokenizer_ref and not (PROJECT_ROOT / tokenizer_ref).exists() else "auto"
    tokenizer = load_tokenizer(
        tokenizer_ref,
        PROJECT_ROOT,
        tokenizer_type=tokenizer_type,
        hf_cache_dir=hf_cache_dir,
        local_files_only=local_files_only,
    )
    model = load_model(args.model, tokenizer, device)

    output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=not args.greedy,
    )
    print(output)


if __name__ == "__main__":
    main()
