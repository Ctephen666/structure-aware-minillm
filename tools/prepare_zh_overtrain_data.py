"""Prepare a large cleaned Chinese pretraining dataset.

This tool streams existing local JSONL sources, cleans text, repairs common
UTF-8-as-GBK mojibake, deduplicates rows, and writes shard JSONL files plus a
train manifest. It does not download data by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from dataclasses import dataclass
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

DEFAULT_SOURCES = [
    ("general", "data/pretrain/chinese_fineweb_edu_sample.jsonl"),
    ("general", "data/pretrain/pretrain_80m_train.jsonl"),
    ("general", "data/pretrain/cci3_hq_sample.jsonl"),
    ("education", "data/pretrain/continue_knowledge/wiki_knowledge.jsonl"),
    ("education", "data/pretrain/continue_knowledge/edu_knowledge.jsonl"),
    ("technical", "data/pretrain/continue_knowledge_train.jsonl"),
    ("structured", "data/pretrain/continue_knowledge/structure_light.jsonl"),
    ("structured", "data/pretrain/structure_synthetic.jsonl"),
]

CATEGORY_TARGETS = {
    "general": 0.60,
    "education": 0.25,
    "technical": 0.10,
    "structured": 0.05,
}

ZH_RE = re.compile(r"[\u4e00-\u9fff]")
SPACE_RE = re.compile(r"\s+")


@dataclass
class Source:
    category: str
    path: Path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def chinese_count(text: str) -> int:
    return len(ZH_RE.findall(text))


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return chinese_count(text) / max(len(text), 1)


def symbol_density(text: str) -> float:
    if not text:
        return 1.0
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace() and not ("\u4e00" <= ch <= "\u9fff"))
    return symbol_count / len(text)


def command_contaminated(text: str) -> bool:
    lowered = text.lower()
    for keyword in COMMAND_KEYWORDS:
        if keyword.lower() in lowered:
            return True
    return False


def repeated_ngram_ratio(text: str, n: int = 4, max_chars: int = 4000) -> float:
    chars = [ch for ch in text[:max_chars] if not ch.isspace()]
    if len(chars) < n * 4:
        return 0.0
    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def repair_mojibake(text: str) -> str:
    """Repair common Chinese mojibake when it clearly improves text quality."""

    original = text
    candidates = [original]
    for encoding in ("gbk", "cp936", "latin1"):
        try:
            candidates.append(original.encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue

    def score(candidate: str) -> tuple[float, int]:
        bad_markers = candidate.count("�") + candidate.count("?")
        return chinese_ratio(candidate) - bad_markers * 0.01, chinese_count(candidate)

    best = max(candidates, key=score)
    if score(best) > score(original):
        return best
    return original


def normalize_text(text: str, fix_mojibake: bool = True) -> str:
    if fix_mojibake:
        text = repair_mojibake(text)
    text = text.replace("\u3000", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    normalized = SPACE_RE.sub("", text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def keep_text(text: str, min_chinese_chars: int, max_symbol_density: float, max_ngram_repeat: float) -> tuple[bool, str]:
    if chinese_count(text) < min_chinese_chars:
        return False, "too_short"
    if symbol_density(text) > max_symbol_density:
        return False, "symbol_dense"
    if command_contaminated(text):
        return False, "command_contaminated"
    if max_ngram_repeat < 1.0 and repeated_ngram_ratio(text) > max_ngram_repeat:
        return False, "repeated_ngram"
    return True, "kept"


def extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return ""
    for key in ("text", "content", "passage", "document"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    prompt = item.get("prompt")
    answer = item.get("answer")
    if isinstance(prompt, str) and isinstance(answer, str):
        return f"{prompt}\n{answer}"
    return ""


def iter_jsonl_texts(path: Path) -> Iterable[str]:
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
            text = extract_text(item)
            if text:
                yield text


class ShardWriter:
    def __init__(self, output_dir: Path, shard_bytes: int) -> None:
        self.output_dir = output_dir
        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.index = 0
        self.current_file = None
        self.current_path: Path | None = None
        self.current_bytes = 0
        self.current_rows = 0
        self.manifest: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        if self.current_file is not None:
            self._close_current()
        self.current_path = self.shards_dir / f"shard_{self.index:04d}.jsonl"
        self.current_file = self.current_path.open("w", encoding="utf-8", newline="\n")
        self.current_bytes = 0
        self.current_rows = 0
        self.index += 1

    def _close_current(self) -> None:
        if self.current_file is None or self.current_path is None:
            return
        self.current_file.close()
        if self.current_rows > 0:
            self.manifest.append(
                {
                    "path": str(self.current_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "bytes": self.current_bytes,
                    "samples": self.current_rows,
                }
            )
        elif self.current_path.exists():
            self.current_path.unlink()
        self.current_file = None
        self.current_path = None

    def write(self, text: str) -> int:
        encoded = (json_dumps({"text": text}) + "\n").encode("utf-8")
        if self.current_file is None:
            self._open_next()
        if self.current_rows > 0 and self.current_bytes + len(encoded) > self.shard_bytes:
            self._open_next()
        assert self.current_file is not None
        self.current_file.write(encoded.decode("utf-8"))
        self.current_bytes += len(encoded)
        self.current_rows += 1
        return len(encoded)

    def close(self) -> None:
        self._close_current()


def parse_sources(values: list[str] | None) -> list[Source]:
    if not values:
        values = [f"{category}:{path}" for category, path in DEFAULT_SOURCES]
    sources: list[Source] = []
    for value in values:
        if ":" in value and not Path(value).drive:
            category, path_value = value.split(":", 1)
        else:
            category, path_value = "general", value
        path = resolve_path(path_value)
        if path.exists():
            sources.append(Source(category=category, path=path))
    if not sources:
        raise FileNotFoundError("No source JSONL files found. Pass --source category:path.")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/pretrain/zh_overtrain")
    parser.add_argument("--source", action="append", default=None, help="category:path JSONL source.")
    parser.add_argument("--target-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--target-gb", type=float, default=None)
    parser.add_argument("--min-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--shard-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--shard-mb", type=float, default=None)
    parser.add_argument("--valid-samples", type=int, default=10000)
    parser.add_argument("--valid-rate", type=float, default=0.002)
    parser.add_argument("--min-chinese-chars", type=int, default=50)
    parser.add_argument("--max-symbol-density", type=float, default=0.35)
    parser.add_argument("--max-ngram-repeat", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fix-mojibake", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()
    if args.target_gb is not None:
        args.target_bytes = int(args.target_gb * 1024**3)
    if args.shard_mb is not None:
        args.shard_bytes = int(args.shard_mb * 1024**2)

    rng = random.Random(args.seed)
    output_dir = resolve_path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = parse_sources(args.source)
    quota_by_category = {
        category: int(args.target_bytes * fraction) for category, fraction in CATEGORY_TARGETS.items()
    }
    written_by_category = {category: 0 for category in CATEGORY_TARGETS}
    writer = ShardWriter(output_dir, args.shard_bytes)
    valid_path = output_dir / "valid.jsonl"

    seen: set[str] = set()
    stats = {
        "source_files": len(sources),
        "seen_rows": 0,
        "kept_train_rows": 0,
        "kept_valid_rows": 0,
        "written_train_bytes": 0,
        "written_valid_bytes": 0,
        "too_short": 0,
        "symbol_dense": 0,
        "command_contaminated": 0,
        "repeated_ngram": 0,
        "duplicate": 0,
    }

    with valid_path.open("w", encoding="utf-8", newline="\n") as valid_file:
        for source in sources:
            category_quota = quota_by_category.get(source.category, args.target_bytes)
            if written_by_category.get(source.category, 0) >= category_quota:
                continue
            print(f"reading {source.category}: {source.path}")
            for raw_text in iter_jsonl_texts(source.path):
                stats["seen_rows"] += 1
                text = normalize_text(raw_text, fix_mojibake=not args.no_fix_mojibake)
                keep, reason = keep_text(
                    text,
                    min_chinese_chars=args.min_chinese_chars,
                    max_symbol_density=args.max_symbol_density,
                    max_ngram_repeat=args.max_ngram_repeat,
                )
                if not keep:
                    stats[reason] += 1
                    continue
                digest = text_hash(text)
                if digest in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(digest)

                row_bytes = len((json_dumps({"text": text}) + "\n").encode("utf-8"))
                send_to_valid = (
                    stats["kept_valid_rows"] < args.valid_samples
                    and rng.random() < args.valid_rate
                )
                if send_to_valid:
                    valid_file.write(json_dumps({"text": text}) + "\n")
                    stats["kept_valid_rows"] += 1
                    stats["written_valid_bytes"] += row_bytes
                else:
                    actual_bytes = writer.write(text)
                    stats["kept_train_rows"] += 1
                    stats["written_train_bytes"] += actual_bytes
                    written_by_category[source.category] = written_by_category.get(source.category, 0) + actual_bytes

                if stats["written_train_bytes"] >= args.target_bytes:
                    break
                if written_by_category.get(source.category, 0) >= category_quota:
                    break
            if stats["written_train_bytes"] >= args.target_bytes:
                break

    writer.close()

    manifest = {
        "version": 1,
        "description": "Cleaned overtrain Chinese pretraining shards for 200M structure-aware LLM.",
        "format": {"type": "jsonl", "field": "text"},
        "target_bytes": args.target_bytes,
        "min_required_bytes": args.min_bytes,
        "shard_bytes": args.shard_bytes,
        "shards": writer.manifest,
        "valid_path": str(valid_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "stats": stats,
        "category_bytes": written_by_category,
        "sources": [
            {"category": source.category, "path": str(source.path.relative_to(PROJECT_ROOT)).replace("\\", "/")}
            for source in sources
        ],
    }
    manifest_path = output_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote manifest: {manifest_path}")
    print(f"train bytes: {stats['written_train_bytes'] / 1024**3:.2f} GB")
    print(f"valid rows: {stats['kept_valid_rows']:,}")
    print(f"shards: {len(writer.manifest)}")
    if stats["written_train_bytes"] < args.min_bytes:
        print(
            "WARNING: cleaned train data is below 2GB minimum; do not downsize the model. "
            "Add more high-quality Chinese sources or relax filters deliberately."
        )


if __name__ == "__main__":
    main()
