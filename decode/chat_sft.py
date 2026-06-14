"""Chat-style inference for SFT checkpoints.

This script uses the same prompt/answer wrapper as the SFT dataset builder, so
the inference prompt matches the format seen during supervised fine-tuning.
"""

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
from train.structure_dataset import build_instruction_parts


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


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    return logits.masked_fill(logits < values[:, [-1]], -float("inf"))


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(int(idx) for idx in generated_ids):
        score = adjusted[:, token_id]
        adjusted[:, token_id] = torch.where(score < 0, score * penalty, score / penalty)
    return adjusted


def apply_no_repeat_ngram(logits: torch.Tensor, generated_ids: list[int], ngram_size: int) -> torch.Tensor:
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


def format_sft_prompt(instruction: str, input_text: str = "") -> str:
    prompt = instruction.strip()
    if input_text.strip():
        prompt = prompt + "\n" + input_text.strip()
    prompt_part, _ = build_instruction_parts(prompt, "")
    return prompt_part


@torch.no_grad()
def generate_answer(
    model: StructTransformerModel,
    tokenizer,
    instruction: str,
    input_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    annotator = StructureAnnotator(max_depth=model.config.max_depth)
    prompt = format_sft_prompt(instruction, input_text)
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids: list[int] = []
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
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = apply_top_k(logits / max(temperature, 1e-6), top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id:
            break
        generated_ids.append(token_id)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(generated_ids).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--input", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.interactive and args.prompt is None:
        raise ValueError("Provide --prompt or use --interactive.")

    cfg = load_checkpoint_config(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = build_tokenizer_from_checkpoint(cfg, args)
    model = load_model(args.model, tokenizer, device)

    def answer_once(prompt: str, input_text: str = "") -> str:
        return generate_answer(
            model=model,
            tokenizer=tokenizer,
            instruction=prompt,
            input_text=input_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=None if args.top_k <= 0 else args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    if args.interactive:
        print("SFT chat ready. Type /exit to quit.")
        while True:
            try:
                prompt = input("\nUser> ").strip()
            except EOFError:
                break
            if prompt in {"/exit", "exit", "quit", "q"}:
                break
            if not prompt:
                continue
            print("Assistant> " + answer_once(prompt))
        return

    print(answer_once(args.prompt or "", args.input))


if __name__ == "__main__":
    main()
