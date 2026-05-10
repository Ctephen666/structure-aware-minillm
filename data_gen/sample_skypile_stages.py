"""Run staged SkyPile-150B sampling: 100MB, 1GB, then 5GB.

Examples:
    python data_gen/sample_skypile_stages.py
    python data_gen/sample_skypile_stages.py --only 100mb
    python data_gen/sample_skypile_stages.py --only 1gb 5gb --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_gen.sample_skypile import sample_skypile_to_files


@dataclass(frozen=True)
class Stage:
    name: str
    size: str
    out_prefix: str

    @property
    def train_path(self) -> Path:
        prefix = PROJECT_ROOT / self.out_prefix
        return prefix.with_name(prefix.name + "_train.jsonl")

    @property
    def valid_path(self) -> Path:
        prefix = PROJECT_ROOT / self.out_prefix
        return prefix.with_name(prefix.name + "_valid.jsonl")


STAGES = {
    "100mb": Stage("100mb", "100MB", "data/pretrain/skypile_100mb"),
    "1gb": Stage("1gb", "1GB", "data/pretrain/skypile_1gb"),
    "5gb": Stage("5gb", "5GB", "data/pretrain/skypile_5gb"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skywork/SkyPile-150B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--only", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing sampled files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages without downloading.")
    args = parser.parse_args()

    for stage_name in args.only:
        stage = STAGES[stage_name]
        print(f"\n== SkyPile stage: {stage.name} ({stage.size}) ==")
        print(f"train: {stage.train_path}")
        print(f"valid: {stage.valid_path}")

        exists = stage.train_path.exists() and stage.valid_path.exists()
        if exists and not args.force:
            print("skip: files already exist; use --force to resample")
            continue
        if args.dry_run:
            print("dry-run: not sampling")
            continue

        train_path, valid_path, train_bytes, valid_bytes = sample_skypile_to_files(
            dataset=args.dataset,
            split=args.split,
            target_bytes=stage.size,
            out_prefix=stage.out_prefix,
            valid_ratio=args.valid_ratio,
            min_chars=args.min_chars,
            seed=args.seed,
        )
        print(f"done: train_bytes={train_bytes}, valid_bytes={valid_bytes}, total={train_bytes + valid_bytes}")
        print(f"train_path={train_path}")
        print(f"valid_path={valid_path}")


if __name__ == "__main__":
    main()
