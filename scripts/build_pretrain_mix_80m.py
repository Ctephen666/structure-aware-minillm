"""Build the 80M pretraining mix and split train/valid.

Examples:
    python scripts/build_pretrain_mix_80m.py
    python scripts/build_pretrain_mix_80m.py --max-output-mb 1536 --valid-ratio 0.01 --force
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SourceState:
    name: str
    path: Path
    weight: float
    quota_bytes: int
    file: TextIO
    written_bytes: int = 0
    read_rows: int = 0
    done: bool = False

    @property
    def progress(self) -> float:
        return self.written_bytes / max(self.quota_bytes, 1)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_text_line(file: TextIO) -> str | None:
    for line in file:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"].strip():
            return json.dumps({"text": row["text"].strip()}, ensure_ascii=False) + "\n"
    return None


def compute_quotas(paths: list[Path], weights: list[float], max_output_mb: int | None) -> list[int]:
    sizes = [path.stat().st_size for path in paths]
    weight_sum = sum(weights)
    total_by_availability = min(size / (weight / weight_sum) for size, weight in zip(sizes, weights))
    target_total = int(total_by_availability)
    if max_output_mb is not None:
        target_total = min(target_total, max_output_mb * 1024 * 1024)
    return [int(target_total * weight / weight_sum) for weight in weights]


def pick_source(sources: list[SourceState]) -> SourceState | None:
    active = [source for source in sources if not source.done and source.written_bytes < source.quota_bytes]
    if not active:
        return None
    return min(active, key=lambda source: source.progress)


def write_line(line: str, train_file: TextIO, valid_file: TextIO, rng: random.Random, valid_ratio: float) -> str:
    if rng.random() < valid_ratio:
        valid_file.write(line)
        return "valid"
    train_file.write(line)
    return "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb", default="data/pretrain/chinese_fineweb_edu_sample.jsonl")
    parser.add_argument("--cci3", default="data/pretrain/cci3_hq_sample.jsonl")
    parser.add_argument("--structure", default="data/pretrain/structure_synthetic.jsonl")
    parser.add_argument("--train-output", default="data/pretrain/pretrain_80m_train.jsonl")
    parser.add_argument("--valid-output", default="data/pretrain/pretrain_80m_valid.jsonl")
    parser.add_argument("--fineweb-weight", type=float, default=70.0)
    parser.add_argument("--cci3-weight", type=float, default=15.0)
    parser.add_argument("--structure-weight", type=float, default=15.0)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument(
        "--max-output-mb",
        type=int,
        default=1536,
        help="Approximate final mixed JSONL size. Tune after count_tokens.py; use 0 to consume all available quota.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--drop-empty-sources",
        action="store_true",
        help="Drop empty source files and renormalize weights. This changes the requested 70/15/15 mix.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_ratio < 0.5:
        raise ValueError(f"valid-ratio must be between 0 and 0.5, got {args.valid_ratio}.")

    source_paths = [resolve_path(args.fineweb), resolve_path(args.cci3), resolve_path(args.structure)]
    names = ["chinese_fineweb_edu", "cci3_hq", "structure_synthetic"]
    weights = [args.fineweb_weight, args.cci3_weight, args.structure_weight]

    filtered_names = []
    filtered_paths = []
    filtered_weights = []
    for name, path, weight in zip(names, source_paths, weights):
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        size = path.stat().st_size
        if size <= 0:
            message = (
                f"Source file is empty: {path}\n"
                "Regenerate it before building the mix. For CCI3-HQ this often means the dataset is gated, "
                "unavailable, or produced no rows after filtering."
            )
            if not args.drop_empty_sources:
                raise ValueError(message + "\nPass --drop-empty-sources only if you intentionally want to renormalize.")
            print(f"Warning: {message}")
            continue
        filtered_names.append(name)
        filtered_paths.append(path)
        filtered_weights.append(weight)

    if not filtered_paths:
        raise ValueError("No non-empty source files are available.")

    source_paths = filtered_paths
    weights = filtered_weights
    names = filtered_names
    max_output_mb = None if args.max_output_mb == 0 else args.max_output_mb
    quotas = compute_quotas(source_paths, weights, max_output_mb)
    for name, path, quota in zip(names, source_paths, quotas):
        if quota <= 0:
            raise ValueError(f"Computed zero quota for {name}; source file may be too small: {path}")
        print(f"{name}: source={path.stat().st_size / 1024 / 1024:.1f} MB quota={quota / 1024 / 1024:.1f} MB")
    if args.dry_run:
        return

    train_output = resolve_path(args.train_output)
    valid_output = resolve_path(args.valid_output)
    for output in (train_output, valid_output):
        if output.exists() and output.stat().st_size > 0 and not args.force:
            raise FileExistsError(f"Output exists, pass --force to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    sources: list[SourceState] = []
    try:
        for name, path, weight, quota in zip(names, source_paths, weights, quotas):
            sources.append(SourceState(name=name, path=path, weight=weight, quota_bytes=quota, file=path.open("r", encoding="utf-8")))

        total_rows = 0
        train_rows = 0
        valid_rows = 0
        with train_output.open("w", encoding="utf-8", newline="\n") as train_file, valid_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as valid_file:
            while True:
                source = pick_source(sources)
                if source is None:
                    break
                line = read_text_line(source.file)
                if line is None:
                    source.done = True
                    continue
                source.read_rows += 1
                line_bytes = len(line.encode("utf-8"))
                if source.written_bytes + line_bytes > source.quota_bytes and source.written_bytes > 0:
                    source.done = True
                    continue
                source.written_bytes += line_bytes
                split = write_line(line, train_file, valid_file, rng, args.valid_ratio)
                train_rows += split == "train"
                valid_rows += split == "valid"
                total_rows += 1
                if total_rows % args.log_every == 0:
                    stats = ", ".join(f"{item.name}={item.written_bytes / 1024 / 1024:.1f}MB" for item in sources)
                    print(f"mixed rows={total_rows}, train={train_rows}, valid={valid_rows}, {stats}")
    finally:
        for source in sources:
            source.file.close()

    print(f"Finished mix: train={train_output} valid={valid_output}")
    for source in sources:
        print(f"  {source.name}: {source.written_bytes / 1024 / 1024:.1f} MB rows={source.read_rows}")


if __name__ == "__main__":
    main()
