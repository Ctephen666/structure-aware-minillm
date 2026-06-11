"""Sample completions from the 80M structure-aware pretrained checkpoint.

Examples:
    python -B scripts/sample_pretrain_80m.py
    python -B scripts/sample_pretrain_80m.py --prompt "人工智能的基本目标是"
    python -B scripts/sample_pretrain_80m.py --greedy --max-new-tokens 120
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
    "人工智能的基本目标是",
    """请继续补全文档：

# 训练总结

## 数据来源
本次中文结构感知模型预训练主要使用""",
    """请继续补全下面的 JSON：
{
  "name": "结构感知模型",
  "features": [""",
    """请继续编写 Python 代码：
```python
def read_jsonl(path):
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
""",
    """下面是一段 YAML 配置，请继续补全：
model:
  name: struct-mini-llm
  layers: 16
training:
""",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/struct_pretrain_80m.pt")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
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
        output = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
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
        print(f"CASE {index}")
        print("-" * 80)
        print(output)


if __name__ == "__main__":
    main()
