"""Prepare data, check quality, and launch formal 200M Chinese pretraining."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/struct_pretrain_200m_zh.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--target-gb", type=float, default=None)
    parser.add_argument("--shard-mb", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = resolve_path("data/pretrain/zh_overtrain")
    manifest_path = resolve_path(cfg["train_manifest"])
    valid_path = resolve_path(cfg["valid_path"])

    if not args.skip_prepare and (
        args.force_prepare or not data_dir.exists() or not manifest_path.exists() or not valid_path.exists()
    ):
        prepare_command = [
            sys.executable,
            "tools/prepare_zh_overtrain_data.py",
            "--output-dir",
            "data/pretrain/zh_overtrain",
        ]
        if args.force_prepare:
            prepare_command.append("--clean-output")
        if args.target_gb is not None:
            prepare_command.extend(["--target-gb", str(args.target_gb)])
        if args.shard_mb is not None:
            prepare_command.extend(["--shard-mb", str(args.shard_mb)])
        run_command(prepare_command)

    if args.prepare_only:
        return

    run_command(
        [
            sys.executable,
            "tools/check_zh_overtrain_quality.py",
            "--data-dir",
            "data/pretrain/zh_overtrain",
            "--strict",
        ]
    )

    if args.quality_only:
        return

    train_command = [
        sys.executable,
        "train/train_pretrain_200m.py",
        "--config",
        args.config,
    ]
    if args.resume:
        train_command.extend(["--resume", args.resume])
    if args.max_steps is not None:
        train_command.extend(["--max-steps", str(args.max_steps)])
    if args.checkpoint_path is not None:
        train_command.extend(["--checkpoint-path", args.checkpoint_path])
    if args.device is not None:
        train_command.extend(["--device", args.device])
    if args.dry_run:
        train_command.append("--dry-run")

    run_command(train_command)


if __name__ == "__main__":
    main()
