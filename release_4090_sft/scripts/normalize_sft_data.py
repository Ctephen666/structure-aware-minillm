"""Normalize SFT JSONL files before instruction tuning.

This keeps the original files untouched and writes fixed copies by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REPLACEMENTS = [
    ("100°C", "100 摄氏度"),
    ("100 °C", "100 摄氏度"),
    ("100℃", "100 摄氏度"),
    ("100 ℃", "100 摄氏度"),
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def normalize_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        normalized = normalize_text(value)
        return normalized, int(normalized != value)
    if isinstance(value, list):
        changed = 0
        normalized_items = []
        for item in value:
            normalized_item, item_changed = normalize_value(item)
            normalized_items.append(normalized_item)
            changed += item_changed
        return normalized_items, changed
    if isinstance(value, dict):
        changed = 0
        normalized_row = {}
        for key, item in value.items():
            normalized_item, item_changed = normalize_value(item)
            normalized_row[key] = normalized_item
            changed += item_changed
        return normalized_row, changed
    return value, 0


def normalize_jsonl(input_path: Path, output_path: Path, force: bool) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    if input_path == output_path:
        raise ValueError("Refusing to overwrite the input file; choose a different output path.")
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        raise FileExistsError(f"Output exists, pass --force to overwrite: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed_rows = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line_no, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_no}: {exc}") from exc
            normalized, changed = normalize_value(row)
            rows += 1
            changed_rows += int(changed > 0)
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return rows, changed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="data/sft/sft_train.jsonl")
    parser.add_argument("--valid-input", default="data/sft/sft_valid.jsonl")
    parser.add_argument("--train-output", default="data/sft/sft_train_fixed.jsonl")
    parser.add_argument("--valid-output", default="data/sft/sft_valid_fixed.jsonl")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [
        (resolve_path(args.train_input), resolve_path(args.train_output), "train"),
        (resolve_path(args.valid_input), resolve_path(args.valid_output), "valid"),
    ]
    total_rows = 0
    total_changed = 0
    for input_path, output_path, name in jobs:
        rows, changed = normalize_jsonl(input_path, output_path, force=args.force)
        total_rows += rows
        total_changed += changed
        print(f"{name}: wrote {output_path} rows={rows} changed={changed}")
    print(f"done: rows={total_rows} changed={total_changed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - command-line script should print a compact failure.
        print(f"normalize_sft_data failed: {exc}", file=sys.stderr)
        raise
