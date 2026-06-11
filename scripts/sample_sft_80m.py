"""Sample the SFT checkpoint using the same instruction template as training.

Examples:
    python -B scripts/sample_sft_80m.py --prompt "水的沸点是多少？"
    python -B scripts/sample_sft_80m.py --model checkpoints/struct_sft_80m_qa.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode.generate_struct import generate_response, load_checkpoint_config, load_model
from tokenizer.tokenizer_factory import load_tokenizer


DEFAULT_PROMPTS = [
    "水的沸点是多少？",
    "太阳主要由什么组成？",
    "光合作用是什么？",
    "牛顿第一定律说明什么？",
    "请用 JSON 表示一个用户信息。",
    "请写一个读取 JSONL 的 Python 函数。",
    "如果你不知道答案，应该怎么回答？",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_instruction_prompt(prompt: str) -> str:
    return "问题：" + prompt.strip() + "\n回答："


def extract_answer(text: str) -> str:
    marker = "回答："
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/struct_sft_80m_qa.pt")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty > 1.0 reduces repeated token reuse; 1.0 disables it.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Forbid repeated n-grams of this size; 0 disables it.",
    )
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--show-full", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0. Use 1.0 to disable it.")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size must be >= 0. Use 0 to disable it.")

    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_cfg = load_checkpoint_config(model_path)
    tokenizer_ref = ckpt_cfg.get("tokenizer_name") or ckpt_cfg.get("tokenizer_path")
    if tokenizer_ref is None:
        raise ValueError("Checkpoint config has no tokenizer_name/tokenizer_path.")
    tokenizer_type = "hf" if "/" in tokenizer_ref and not (PROJECT_ROOT / tokenizer_ref).exists() else "auto"
    tokenizer = load_tokenizer(
        tokenizer_ref,
        PROJECT_ROOT,
        tokenizer_type=tokenizer_type,
        hf_cache_dir=ckpt_cfg.get("hf_cache_dir"),
        local_files_only=args.local_files_only or bool(ckpt_cfg.get("hf_local_files_only", False)),
    )
    model = load_model(model_path, tokenizer, device)

    prompts = args.prompt or DEFAULT_PROMPTS
    for index, prompt in enumerate(prompts, start=1):
        formatted_prompt = build_instruction_prompt(prompt)
        output = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=None if args.top_k <= 0 else args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            do_sample=not args.greedy,
        )
        print("=" * 80)
        print(f"CASE {index}: {prompt}")
        print("-" * 80)
        print(output if args.show_full else extract_answer(output))


if __name__ == "__main__":
    main()
