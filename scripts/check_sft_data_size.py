"""Check that SFT JSONL files are large enough for a planned run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def count_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/sft/sft_train_8h_fixed.jsonl")
    parser.add_argument("--valid", default="data/sft/sft_valid_8h_fixed.jsonl")
    parser.add_argument("--min-total-rows", type=int, default=100000)
    parser.add_argument("--min-valid-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = resolve_path(args.train)
    valid_path = resolve_path(args.valid)
    train_rows = count_rows(train_path)
    valid_rows = count_rows(valid_path)
    total_rows = train_rows + valid_rows
    print(f"SFT rows: train={train_rows}, valid={valid_rows}, total={total_rows}")

    if total_rows < args.min_total_rows:
        raise RuntimeError(
            f"SFT data is too small for the 8h plan: total={total_rows}, "
            f"required>={args.min_total_rows}. Re-run data preparation with reachable public sources, "
            "or lower MIN_ROWS only if you intentionally accept more data repetition."
        )
    if valid_rows < args.min_valid_rows:
        raise RuntimeError(
            f"Validation data is too small: valid={valid_rows}, required>={args.min_valid_rows}."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - command-line script should print a compact failure.
        print(f"check_sft_data_size failed: {exc}", file=sys.stderr)
        raise
