"""Stream and sample high-quality Chinese pretraining data.

Examples:
    python scripts/download_pretrain_data_80m.py
    python scripts/download_pretrain_data_80m.py --fineweb-target-mb 4608 --cci3-target-mb 900
    python scripts/download_pretrain_data_80m.py --include-fineweb2 --force
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]{1,160}>")
REPEATED_CHAR_RE = re.compile(r"(.)\1{11,}")
MOJIBAKE_RE = re.compile(r"(\ufffd|锟|閻|鐗|涓|绋|鏉|娑|搢)")
BAD_CONTENT_RE = re.compile(r"(彩票|博彩|成人|裸聊|代孕|色情|六合彩|网赚|贷款)")
BOILERPLATE_RE = re.compile(r"(免责声明|版权所有|联系我们|扫码|点击查看|上一篇|下一篇|广告合作|未经许可)")


@dataclass(frozen=True)
class DatasetCandidate:
    name: str
    config: str | None = None
    data_files: str | list[str] | None = None
    split: str = "train"


@dataclass(frozen=True)
class OutputSpec:
    label: str
    path: str
    target_mb: int
    candidates: tuple[DatasetCandidate, ...]


FINEWEB_CANDIDATES = (
    DatasetCandidate("opencsg/chinese-fineweb-edu-v2"),
    DatasetCandidate("opencsg/Fineweb-Edu-Chinese-V2.2", data_files="4_5/*.parquet"),
    DatasetCandidate("opencsg/Fineweb-Edu-Chinese-V2.1", data_files="4_5/*.parquet"),
)

FINEWEB2_CANDIDATES = (
    DatasetCandidate("HuggingFaceFW/fineweb-2", config="zho_Hans"),
    DatasetCandidate("HuggingFaceFW/fineweb-2", config="zho_Hant"),
    DatasetCandidate("epfml/FineWeb2-HQ", config="zho_Hans"),
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = HTML_TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def chinese_ratio(text: str) -> float:
    return len(CHINESE_RE.findall(text)) / max(len(text), 1)


def ascii_alpha_ratio(text: str) -> float:
    return len(ASCII_ALPHA_RE.findall(text)) / max(len(text), 1)


def is_good_text(text: str, min_chars: int, max_chars: int, min_chinese_ratio: float) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if chinese_ratio(text) < min_chinese_ratio:
        return False
    if ascii_alpha_ratio(text) > 0.45:
        return False
    if MOJIBAKE_RE.search(text) or BAD_CONTENT_RE.search(text):
        return False
    if BOILERPLATE_RE.search(text) and len(BOILERPLATE_RE.findall(text)) >= 2:
        return False
    if REPEATED_CHAR_RE.search(text):
        return False
    if text.count("http") > 2 or text.count("@") > 4:
        return False
    if len(set(text)) / max(len(text), 1) < 0.06:
        return False
    return True


def row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""
    for key in ("text", "content", "article", "body", "raw_content", "正文", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def row_allowed(row: Any, min_score: float | None) -> bool:
    if min_score is None or not isinstance(row, dict):
        return True
    for key in ("score", "quality_score", "int_score"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value) >= min_score
        except (TypeError, ValueError):
            continue
    return True


def load_stream(candidate: DatasetCandidate, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets: pip install datasets") from exc

    kwargs: dict[str, Any] = {"streaming": True, "split": candidate.split}
    if cache_dir:
        kwargs["cache_dir"] = str(resolve_path(cache_dir))
    if candidate.data_files is not None:
        kwargs["data_files"] = candidate.data_files
    if candidate.config is not None:
        return load_dataset(candidate.name, candidate.config, **kwargs)
    return load_dataset(candidate.name, **kwargs)


def iter_candidates(candidates: Iterable[DatasetCandidate], cache_dir: str | None) -> Iterator[Any]:
    errors = []
    for candidate in candidates:
        try:
            print(f"Streaming dataset: {candidate.name} config={candidate.config} data_files={candidate.data_files}")
            yield from load_stream(candidate, cache_dir)
            return
        except Exception as exc:  # noqa: BLE001 - keep fallback diagnostics.
            errors.append(f"{candidate.name}: {exc}")
            print(f"  failed, trying next candidate: {exc}", file=sys.stderr)
    raise RuntimeError("Could not load any candidate:\n" + "\n".join(errors))


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def write_sample(spec: OutputSpec, args: argparse.Namespace) -> None:
    output_path = resolve_path(spec.path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    target_bytes = int(spec.target_mb * 1024 * 1024)
    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        print(f"Skip existing file: {output_path}")
        return
    if args.dry_run:
        print(f"Would write {spec.label}: {output_path} target={spec.target_mb} MB")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_path.exists():
        temp_path.unlink()
    candidates = list(spec.candidates)
    if spec.label == "Chinese FineWeb-Edu" and args.include_fineweb2:
        candidates.extend(FINEWEB2_CANDIDATES)

    rng = random.Random(args.seed)
    seen: set[bytes] = set()
    written_bytes = 0
    accepted = 0
    rejected = 0

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in iter_candidates(candidates, args.cache_dir):
                if not row_allowed(row, args.min_score):
                    rejected += 1
                    continue
                text = normalize_text(row_text(row))
                if not is_good_text(text, args.min_chars, args.max_chars, args.min_chinese_ratio):
                    rejected += 1
                    continue
                digest = text_digest(text)
                if digest in seen:
                    rejected += 1
                    continue
                if len(seen) < args.dedup_max_items:
                    seen.add(digest)
                elif rng.random() < 0.001:
                    seen.clear()
                    seen.add(digest)

                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                file.write(line)
                written_bytes += len(line.encode("utf-8"))
                accepted += 1
                if accepted % args.log_every == 0:
                    print(
                        f"{spec.label}: {written_bytes / 1024 / 1024:.1f} MB, "
                        f"accepted={accepted}, rejected={rejected}"
                    )
                if written_bytes >= target_bytes:
                    break
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        raise

    if written_bytes <= 0:
        temp_path.unlink(missing_ok=True)
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        raise RuntimeError(
            f"{spec.label} produced 0 bytes. Check dataset access/name and filters, then rerun with --force."
        )

    temp_path.replace(output_path)

    print(
        f"Finished {spec.label}: {output_path} "
        f"{written_bytes / 1024 / 1024:.1f} MB, accepted={accepted}, rejected={rejected}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-output", default="data/pretrain/chinese_fineweb_edu_sample.jsonl")
    parser.add_argument("--cci3-output", default="data/pretrain/cci3_hq_sample.jsonl")
    parser.add_argument("--fineweb-target-mb", type=int, default=4608)
    parser.add_argument("--cci3-target-mb", type=int, default=900)
    parser.add_argument("--fineweb-dataset", default=None, help="Override the primary FineWeb-Edu dataset id.")
    parser.add_argument("--fineweb-config", default=None)
    parser.add_argument("--fineweb-data-files", default=None)
    parser.add_argument("--cci3-dataset", default="BAAI/CCI3-HQ")
    parser.add_argument("--cci3-config", default=None)
    parser.add_argument("--cci3-data-files", default=None)
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--min-chinese-ratio", type=float, default=0.35)
    parser.add_argument("--dedup-max-items", type=int, default=3_000_000)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--include-fineweb2", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fineweb_candidates = FINEWEB_CANDIDATES
    if args.fineweb_dataset is not None:
        fineweb_candidates = (
            DatasetCandidate(args.fineweb_dataset, config=args.fineweb_config, data_files=args.fineweb_data_files),
            *FINEWEB_CANDIDATES,
        )
    cci3_candidates = (
        DatasetCandidate(args.cci3_dataset, config=args.cci3_config, data_files=args.cci3_data_files),
    )
    specs = [
        OutputSpec(
            label="Chinese FineWeb-Edu",
            path=args.fineweb_output,
            target_mb=args.fineweb_target_mb,
            candidates=fineweb_candidates,
        ),
        OutputSpec(
            label="BAAI CCI3-HQ",
            path=args.cci3_output,
            target_mb=args.cci3_target_mb,
            candidates=cci3_candidates,
        ),
    ]
    for spec in specs:
        write_sample(spec, args)


if __name__ == "__main__":
    main()
