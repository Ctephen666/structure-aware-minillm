"""Count tokenizer tokens in JSONL files with {"text": "..."} rows.

Examples:
    python scripts/count_tokens.py --input data/pretrain/pretrain_80m_train.jsonl --input data/pretrain/pretrain_80m_valid.jsonl
    python scripts/count_tokens.py --config configs/struct_pretrain_80m.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizer.tokenizer_factory import build_tokenizer
from train.structure_dataset import row_to_training_text


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config if isinstance(config, dict) else {}


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            yield row


def count_file(path: Path, tokenizer, add_special_tokens: bool, log_every: int) -> dict[str, int]:
    rows = 0
    tokens = 0
    bytes_read = 0
    for row in iter_rows(path):
        text = row_to_training_text(row)
        if not text.strip():
            continue
        tokens += len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        rows += 1
        bytes_read += len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1
        if rows % log_every == 0:
            print(f"{path.name}: rows={rows}, tokens={tokens:,}, bytes={bytes_read / 1024 / 1024:.1f} MB")
    return {"rows": rows, "tokens": tokens, "bytes": bytes_read}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/struct_pretrain_80m.yaml")
    parser.add_argument("--input", action="append", default=None)
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--log-every", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs = args.input or [cfg["train_path"], cfg["valid_path"]]
    tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts=None)

    total_rows = 0
    total_tokens = 0
    total_bytes = 0
    for value in inputs:
        path = resolve_path(value)
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        stats = count_file(path, tokenizer, add_special_tokens=not args.no_special_tokens, log_every=args.log_every)
        total_rows += stats["rows"]
        total_tokens += stats["tokens"]
        total_bytes += stats["bytes"]
        print(f"{path}: rows={stats['rows']:,}, tokens={stats['tokens']:,}, bytes={stats['bytes'] / 1024 / 1024:.1f} MB")

    print(f"TOTAL rows={total_rows:,} tokens={total_tokens:,} bytes={total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
