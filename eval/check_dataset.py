"""Dataset validator for structure-aware LLM trap datasets.

Usage:
    python eval/check_dataset.py --input data/train.jsonl
    python eval/check_dataset.py --input data/valid.jsonl
    python eval/check_dataset.py --input data/test.jsonl

This script validates:
1. Required fields exist.
2. JSON answers are valid JSON.
3. Markdown fenced answers do not close the outer fence early.
4. Mixed samples are both Markdown-safe and contain valid inner JSON blocks.
5. Optional unsafe_answer is actually unsafe when possible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BT = chr(96)
TD = "~"

REQUIRED_FIELDS = ["id", "task_type", "prompt", "answer", "format"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
                if not isinstance(item, dict):
                    rows.append({
                        "_line_no": line_no,
                        "_raw": raw,
                        "_load_error": "line is not a JSON object",
                    })
                else:
                    item["_line_no"] = line_no
                    rows.append(item)
            except json.JSONDecodeError as e:
                rows.append({
                    "_line_no": line_no,
                    "_raw": raw,
                    "_load_error": str(e),
                })
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_required_fields(sample: dict[str, Any]) -> tuple[bool, str]:
    if "_load_error" in sample:
        return False, f"jsonl_load_error: {sample['_load_error']}"

    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        return False, "missing_fields: " + ",".join(missing)

    if not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip():
        return False, "empty_prompt"

    if not isinstance(sample.get("answer"), str) or not sample["answer"].strip():
        return False, "empty_answer"

    return True, "ok"


def check_json_answer(answer: str) -> tuple[bool, str]:
    try:
        json.loads(answer)
        return True, "ok"
    except json.JSONDecodeError as e:
        return False, f"invalid_json: {e}"


def detect_fence(line: str) -> tuple[str, int, str] | None:
    """Detect Markdown fence.

    Return:
        (fence_char, fence_len, suffix)

    fence_char is either backtick or tilde.
    suffix is language tag or any non-fence text after the fence.
    """
    stripped = line.strip()
    if not stripped:
        return None

    first = stripped[0]
    if first not in (BT, TD):
        return None

    count = 0
    for ch in stripped:
        if ch == first:
            count += 1
        else:
            break

    if count < 3:
        return None

    suffix = stripped[count:].strip()
    return first, count, suffix


def is_closing_fence_for(line: str, fence_char: str, fence_len: int) -> bool:
    stripped = line.strip()
    prefix = fence_char * fence_len
    if not stripped.startswith(prefix):
        return False

    rest = stripped[fence_len:].strip()
    return rest == ""


def find_outer_fence(lines: list[str]) -> tuple[int, int, str, int, str] | None:
    """Find first and last non-empty lines and parse the outer opening fence."""
    first_idx = None
    last_idx = None

    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            break

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break

    if first_idx is None or last_idx is None:
        return None

    opening = detect_fence(lines[first_idx])
    if opening is None:
        return None

    char, length, suffix = opening
    return first_idx, last_idx, char, length, suffix


def check_markdown_fence_answer(answer: str) -> tuple[bool, str]:
    lines = answer.splitlines()
    if len(lines) < 2:
        return False, "markdown_too_short"

    outer = find_outer_fence(lines)
    if outer is None:
        return False, "missing_outer_opening_fence"

    first_idx, last_idx, outer_char, outer_len, outer_suffix = outer

    allowed_outer_suffix = {"", "markdown", "md", "text", "txt"}
    if outer_suffix.lower() not in allowed_outer_suffix:
        return False, f"unexpected_outer_fence_suffix: {outer_suffix}"

    if not is_closing_fence_for(lines[last_idx], outer_char, outer_len):
        return False, "missing_or_wrong_outer_closing_fence"

    # Check body lines. A body line must not be able to close the outer fence.
    body_lines = lines[first_idx + 1:last_idx]
    for local_i, line in enumerate(body_lines, start=first_idx + 2):
        if is_closing_fence_for(line, outer_char, outer_len):
            return False, f"early_outer_fence_close_at_line_{local_i}"

    # Check all inner fences for possible conflict with outer fence.
    for local_i, line in enumerate(body_lines, start=first_idx + 2):
        fence = detect_fence(line)
        if fence is None:
            continue

        char, length, suffix = fence

        # Same fence char with length >= outer_len can conflict.
        # Inner opening with language suffix is still risky if length >= outer_len.
        if char == outer_char and length >= outer_len:
            return False, f"inner_fence_can_conflict_with_outer_at_line_{local_i}"

    return True, "ok"


def extract_json_blocks_from_markdown(answer: str) -> list[str]:
    lines = answer.splitlines()
    blocks: list[str] = []

    in_json_block = False
    fence_char = ""
    fence_len = 0
    buf: list[str] = []

    for line in lines:
        fence = detect_fence(line)

        if not in_json_block:
            if fence is not None:
                char, length, suffix = fence
                if suffix.lower() == "json":
                    in_json_block = True
                    fence_char = char
                    fence_len = length
                    buf = []
            continue

        if is_closing_fence_for(line, fence_char, fence_len):
            blocks.append("\n".join(buf).strip())
            in_json_block = False
            fence_char = ""
            fence_len = 0
            buf = []
        else:
            buf.append(line)

    return blocks


def check_mixed_answer(answer: str) -> tuple[bool, str]:
    ok, msg = check_markdown_fence_answer(answer)
    if not ok:
        return False, "mixed_markdown_failed: " + msg

    json_blocks = extract_json_blocks_from_markdown(answer)
    if not json_blocks:
        return False, "mixed_no_json_block_found"

    for idx, block in enumerate(json_blocks):
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            return False, f"mixed_inner_json_invalid_block_{idx}: {e}"

    return True, "ok"


def check_safe_answer(sample: dict[str, Any]) -> tuple[bool, str]:
    task_type = str(sample.get("task_type", "")).lower()
    fmt = str(sample.get("format", "")).lower()
    answer = sample.get("answer", "")

    if fmt == "json" or "json_escape" in task_type:
        return check_json_answer(answer)

    if task_type == "mixed" or fmt == "mixed":
        return check_mixed_answer(answer)

    if fmt == "markdown" or "markdown" in task_type or "fence" in task_type:
        return check_markdown_fence_answer(answer)

    return False, f"unknown_task_or_format: task_type={task_type}, format={fmt}"


def check_unsafe_answer(sample: dict[str, Any]) -> tuple[bool, str]:
    """Optional check for unsafe_answer.

    Return True when either:
    - no unsafe_answer exists, or
    - unsafe_answer is actually unsafe.
    """
    if "unsafe_answer" not in sample:
        return True, "no_unsafe_answer"

    unsafe_answer = sample.get("unsafe_answer")
    if not isinstance(unsafe_answer, str) or not unsafe_answer.strip():
        return True, "empty_unsafe_answer"

    clone = dict(sample)
    clone["answer"] = unsafe_answer

    ok, msg = check_safe_answer(clone)
    if ok:
        return False, "unsafe_answer_unexpectedly_passed"
    return True, "ok"


def check_one_sample(sample: dict[str, Any]) -> dict[str, Any]:
    ok, msg = check_required_fields(sample)
    if not ok:
        return {
            "ok": False,
            "reason": msg,
            "id": sample.get("id"),
            "line_no": sample.get("_line_no"),
            "task_type": sample.get("task_type"),
            "format": sample.get("format"),
        }

    safe_ok, safe_msg = check_safe_answer(sample)
    unsafe_ok, unsafe_msg = check_unsafe_answer(sample)

    final_ok = safe_ok and unsafe_ok
    reasons = []
    if not safe_ok:
        reasons.append(safe_msg)
    if not unsafe_ok:
        reasons.append(unsafe_msg)

    return {
        "ok": final_ok,
        "reason": "ok" if final_ok else " | ".join(reasons),
        "id": sample.get("id"),
        "line_no": sample.get("_line_no"),
        "task_type": sample.get("task_type"),
        "format": sample.get("format"),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    failed = total - passed

    failure_reasons = Counter()
    by_task = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})

    for r in results:
        task_type = str(r.get("task_type", "unknown"))
        by_task[task_type]["total"] += 1
        if r["ok"]:
            by_task[task_type]["passed"] += 1
        else:
            by_task[task_type]["failed"] += 1
            reason_head = str(r["reason"]).split(":")[0]
            failure_reasons[reason_head] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else 0.0,
        "failure_reasons": dict(failure_reasons),
        "by_task": dict(by_task),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--failed_output", default="results/dataset_check_failed.jsonl")
    parser.add_argument("--summary_output", default="results/dataset_check_summary.json")
    args = parser.parse_args()

    samples = load_jsonl(args.input)
    results = [check_one_sample(sample) for sample in samples]

    failed_rows = [
        {"check_result": result, "sample": sample}
        for sample, result in zip(samples, results)
        if not result["ok"]
    ]

    summary = summarize(results)

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_output).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_jsonl(args.failed_output, failed_rows)

    print("Dataset check finished.")
    print(f"Input: {args.input}")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.4f}")

    if summary["failure_reasons"]:
        print("Failure reasons:")
        for key, value in summary["failure_reasons"].items():
            print(f"  {key}: {value}")

    if summary["failed"] > 0:
        print(f"Failed samples saved to: {args.failed_output}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
