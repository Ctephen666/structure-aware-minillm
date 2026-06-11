"""Build the knowledge-focused continued-pretraining mix.

Default ratio:
    edu_knowledge: 55%
    wiki_knowledge: 25%
    cci3_knowledge: 15%
    structure_light: 5%

Examples:
    python -B scripts/build_continue_knowledge_mix.py
    python -B scripts/build_continue_knowledge_mix.py --max-output-mb 1024 --force
    python -B scripts/build_continue_knowledge_mix.py --drop-empty-sources --force
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
    parser.add_argument("--input-dir", default="data/pretrain/continue_knowledge")
    parser.add_argument("--edu", default=None)
    parser.add_argument("--wiki", default=None)
    parser.add_argument("--cci3", default=None)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--train-output", default="data/pretrain/continue_knowledge_train.jsonl")
    parser.add_argument("--valid-output", default="data/pretrain/continue_knowledge_valid.jsonl")
    parser.add_argument("--edu-weight", type=float, default=55.0)
    parser.add_argument("--wiki-weight", type=float, default=25.0)
    parser.add_argument("--cci3-weight", type=float, default=15.0)
    parser.add_argument("--structure-weight", type=float, default=5.0)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--max-output-mb", type=int, default=1024, help="Use 0 to consume all available quota.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--drop-empty-sources",
        action="store_true",
        help="Drop missing/empty sources and renormalize weights. Use when CCI3 is unavailable.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_ratio < 0.5:
        raise ValueError(f"valid-ratio must be between 0 and 0.5, got {args.valid_ratio}.")

    input_dir = Path(args.input_dir)
    source_specs = [
        ("edu_knowledge", resolve_path(args.edu or input_dir / "edu_knowledge.jsonl"), args.edu_weight),
        ("wiki_knowledge", resolve_path(args.wiki or input_dir / "wiki_knowledge.jsonl"), args.wiki_weight),
        ("cci3_knowledge", resolve_path(args.cci3 or input_dir / "cci3_knowledge.jsonl"), args.cci3_weight),
        ("structure_light", resolve_path(args.structure or input_dir / "structure_light.jsonl"), args.structure_weight),
    ]

    names: list[str] = []
    source_paths: list[Path] = []
    weights: list[float] = []
    for name, path, weight in source_specs:
        if not path.exists() or path.stat().st_size <= 0:
            message = f"Source missing or empty: {path}"
            if not args.drop_empty_sources:
                raise FileNotFoundError(message + "\nPass --drop-empty-sources only if you want to renormalize.")
            print(f"Warning: {message}; dropped.")
            continue
        names.append(name)
        source_paths.append(path)
        weights.append(weight)

    if not source_paths:
        raise ValueError("No usable source files are available.")

    max_output_mb = None if args.max_output_mb == 0 else args.max_output_mb
    quotas = compute_quotas(source_paths, weights, max_output_mb)
    for name, path, quota in zip(names, source_paths, quotas):
        if quota <= 0:
            raise ValueError(f"Computed zero quota for {name}: {path}")
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
            sources.append(SourceState(name, path, weight, quota, path.open("r", encoding="utf-8")))

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
