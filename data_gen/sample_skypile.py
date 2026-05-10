"""Stream a byte-limited SkyPile-150B sample into local JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)?\s*", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "K": 1024,
        "MB": 1024**2,
        "M": 1024**2,
        "GB": 1024**3,
        "G": 1024**3,
        "TB": 1024**4,
        "T": 1024**4,
    }
    return int(number * multipliers[unit])


def get_text(row: dict[str, Any]) -> str:
    text = row.get("text", "")
    if not isinstance(text, str):
        return ""
    return text.strip()


def load_stream(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets to sample SkyPile: pip install datasets") from exc

    return load_dataset(dataset_name, split=split, streaming=True)


def write_jsonl(path: Path, rows: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with path.open("w", encoding="utf-8") as file:
        for text in rows:
            line = json.dumps({"text": text}, ensure_ascii=False)
            total_bytes += len((line + "\n").encode("utf-8"))
            file.write(line + "\n")
    return total_bytes


def sample_skypile_to_files(
    dataset: str,
    split: str,
    target_bytes: str,
    out_prefix: str,
    valid_ratio: float = 0.01,
    min_chars: int = 200,
    seed: int = 42,
    max_rows: int | None = None,
) -> tuple[Path, Path, int, int]:
    random.seed(seed)
    target_size = parse_size(target_bytes)
    rows: list[str] = []
    seen_bytes = 0

    stream = load_stream(dataset, split)
    for idx, row in enumerate(stream):
        if max_rows is not None and idx >= max_rows:
            break
        text = get_text(row)
        if len(text) < min_chars:
            continue
        encoded_size = len((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
        if seen_bytes + encoded_size > target_size and rows:
            break
        rows.append(text)
        seen_bytes += encoded_size
        if seen_bytes >= target_size:
            break

    if not rows:
        raise RuntimeError("No rows were sampled. Check network access or dataset permissions.")

    random.shuffle(rows)
    valid_count = max(1, int(len(rows) * valid_ratio))
    valid_rows = rows[:valid_count]
    train_rows = rows[valid_count:]

    out_path_prefix = PROJECT_ROOT / out_prefix
    train_path = out_path_prefix.with_name(out_path_prefix.name + "_train.jsonl")
    valid_path = out_path_prefix.with_name(out_path_prefix.name + "_valid.jsonl")
    train_bytes = write_jsonl(train_path, train_rows)
    valid_bytes = write_jsonl(valid_path, valid_rows)
    return train_path, valid_path, train_bytes, valid_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skywork/SkyPile-150B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-bytes", default="100MB")
    parser.add_argument("--out-prefix", default="data/pretrain/skypile_100mb")
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    train_path, valid_path, train_bytes, valid_bytes = sample_skypile_to_files(
        dataset=args.dataset,
        split=args.split,
        target_bytes=args.target_bytes,
        out_prefix=args.out_prefix,
        valid_ratio=args.valid_ratio,
        min_chars=args.min_chars,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Bytes: train={train_bytes}, valid={valid_bytes}, total={train_bytes + valid_bytes}")
    print(f"Train path: {train_path}")
    print(f"Valid path: {valid_path}")


if __name__ == "__main__":
    main()
