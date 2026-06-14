"""Quality checks for the cleaned Chinese overtrain dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMMAND_KEYWORDS = (
    "--checkpoint",
    "--config",
    "--device",
    "init-from-checkpoint",
    "Traceback",
    "python train",
    "pip install",
    "conda install",
    "usage:",
    "README",
)

ZH_RE = re.compile(r"[\u4e00-\u9fff]")
SPACE_RE = re.compile(r"\s+")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def iter_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    yield value
            elif isinstance(item, str):
                yield item


def chinese_count(text: str) -> int:
    return len(ZH_RE.findall(text))


def command_contaminated(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in COMMAND_KEYWORDS)


def normalized_hash(text: str) -> str:
    normalized = SPACE_RE.sub("", text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def update_samples(samples: list[str], text: str, seen_count: int, rng: random.Random, max_samples: int) -> None:
    preview = text[:300]
    if len(samples) < max_samples:
        samples.append(preview)
        return
    index = rng.randint(0, seen_count - 1)
    if index < max_samples:
        samples[index] = preview


def collect_stats(paths: list[Path], min_chinese_chars: int, sample_count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    seen: set[str] = set()
    duplicate_rows = 0
    rows = 0
    total_chars = 0
    total_chinese = 0
    command_rows = 0
    short_rows = 0
    samples: list[str] = []

    for path in paths:
        for text in iter_jsonl(path):
            rows += 1
            text_len = len(text)
            zh = chinese_count(text)
            total_chars += text_len
            total_chinese += zh
            if command_contaminated(text):
                command_rows += 1
            if zh < min_chinese_chars:
                short_rows += 1
            digest = normalized_hash(text)
            if digest in seen:
                duplicate_rows += 1
            else:
                seen.add(digest)
            update_samples(samples, text, rows, rng, sample_count)

    return {
        "rows": rows,
        "avg_chars": total_chars / max(rows, 1),
        "chinese_ratio": total_chinese / max(total_chars, 1),
        "command_contamination_ratio": command_rows / max(rows, 1),
        "duplicate_ratio": duplicate_rows / max(rows, 1),
        "too_short_ratio": short_rows / max(rows, 1),
        "samples": samples,
    }


def format_bytes(num_bytes: int) -> str:
    gb = num_bytes / 1024**3
    mb = num_bytes / 1024**2
    return f"{gb:.2f} GB ({mb:.1f} MB)"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    sample_lines = []
    for index, sample in enumerate(summary["samples"], start=1):
        escaped = sample.replace("\n", "\\n")
        sample_lines.append(f"{index}. {escaped}")

    report = f"""# zh_overtrain Data Quality Report

- Total file size: {format_bytes(summary['total_bytes'])}
- Shard count: {summary['shard_count']}
- Train samples: {summary['train_rows']:,}
- Valid samples: {summary['valid_rows']:,}
- Average character length: {summary['avg_chars']:.1f}
- Chinese character ratio: {summary['chinese_ratio']:.4f}
- Command contamination ratio: {summary['command_contamination_ratio']:.6f}
- Duplicate ratio: {summary['duplicate_ratio']:.6f}
- Too-short text ratio: {summary['too_short_ratio']:.6f}
- Minimum required size: {format_bytes(summary['min_bytes'])}
- Size check: {"PASS" if summary['size_ok'] else "FAIL"}

## Random Samples

{chr(10).join(sample_lines)}
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/pretrain/zh_overtrain")
    parser.add_argument("--min-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--min-chinese-chars", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when the 2GB size floor is not met.")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    manifest_path = data_dir / "train_manifest.json"
    valid_path = data_dir / "valid.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing train manifest: {manifest_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing valid file: {valid_path}")

    manifest = load_json(manifest_path)
    shard_paths = [resolve_path(shard["path"]) for shard in manifest.get("shards", [])]
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing[:3]}")

    total_bytes = sum(path.stat().st_size for path in shard_paths) + valid_path.stat().st_size
    train_stats = collect_stats(shard_paths, args.min_chinese_chars, args.sample_count, args.seed)
    valid_stats = collect_stats([valid_path], args.min_chinese_chars, 0, args.seed + 1)

    summary = {
        "total_bytes": total_bytes,
        "shard_count": len(shard_paths),
        "train_rows": train_stats["rows"],
        "valid_rows": valid_stats["rows"],
        "avg_chars": train_stats["avg_chars"],
        "chinese_ratio": train_stats["chinese_ratio"],
        "command_contamination_ratio": train_stats["command_contamination_ratio"],
        "duplicate_ratio": train_stats["duplicate_ratio"],
        "too_short_ratio": train_stats["too_short_ratio"],
        "samples": train_stats["samples"],
        "min_bytes": args.min_bytes,
        "size_ok": total_bytes >= args.min_bytes,
    }

    report_path = data_dir / "data_quality_report.md"
    write_report(report_path, summary)
    print(f"report: {report_path}")
    print(f"total size: {format_bytes(total_bytes)}")
    print(f"shards: {len(shard_paths)}")
    print(f"train samples: {summary['train_rows']:,}")
    print(f"valid samples: {summary['valid_rows']:,}")
    print(f"chinese ratio: {summary['chinese_ratio']:.4f}")
    print(f"command contamination ratio: {summary['command_contamination_ratio']:.6f}")
    print(f"duplicate ratio: {summary['duplicate_ratio']:.6f}")
    print(f"too-short ratio: {summary['too_short_ratio']:.6f}")
    if not summary["size_ok"]:
        print(
            "WARNING: cleaned dataset is below 2GB. Do not switch to a smaller model; "
            "add more high-quality Chinese data or rerun preparation with larger sources."
        )
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
