"""Evaluate a baseline MiniLLM on structure-trap data.

Example:
    python eval/run_all.py \
        --model checkpoints/baseline.pt \
        --tokenizer checkpoints/baseline_tokenizer.json \
        --input data/test.jsonl \
        --output results/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerConfig, TransformerModel
from tokenizer.regex_tokenizer import RegexTokenizer


BT = chr(96)
TD = "~"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (PROJECT_ROOT / path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(model_path: str | Path, tokenizer: RegexTokenizer, device: torch.device) -> TransformerModel:
    checkpoint = torch.load(PROJECT_ROOT / model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    state_dict = checkpoint["model"]

    checkpoint_vocab_size = state_dict["token_embedding.weight"].shape[0]
    if checkpoint_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size}, but checkpoint vocab_size={checkpoint_vocab_size}. "
            "Please use the tokenizer saved with this checkpoint."
        )

    model_cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(cfg.get("block_size", 512)),
        n_layer=int(cfg.get("n_layer", 4)),
        n_head=int(cfg.get("n_head", 4)),
        n_embd=int(cfg.get("n_embd", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model = TransformerModel(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_instruction_prompt(prompt: str) -> str:
    return "### Instruction:\n" + prompt.strip() + "\n\n### Response:\n"


@torch.no_grad()
def generate_response(
    model: TransformerModel,
    tokenizer: RegexTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    do_sample: bool,
) -> str:
    formatted_prompt = build_instruction_prompt(prompt)
    input_ids = [tokenizer.bos_id] + tokenizer.encode(formatted_prompt, add_special_tokens=False)
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = generated.size(1)

    for _ in range(max_new_tokens):
        context = generated[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :]

        if not do_sample:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(float(temperature), 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)

        if int(next_id.item()) == tokenizer.eos_id:
            break

    new_ids = generated[0, prompt_len:].tolist()
    if tokenizer.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]

    return tokenizer.decode(new_ids).strip()


def detect_fence(line: str) -> tuple[str, int, str] | None:
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
    return stripped[fence_len:].strip() == ""


def find_outer_fence(lines: list[str]) -> tuple[int, int, str, int, str] | None:
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
    if outer_suffix.lower() not in {"", "markdown", "md", "text", "txt"}:
        return False, f"unexpected_outer_fence_suffix:{outer_suffix}"

    if not is_closing_fence_for(lines[last_idx], outer_char, outer_len):
        return False, "missing_or_wrong_outer_closing_fence"

    body_lines = lines[first_idx + 1:last_idx]
    for local_i, line in enumerate(body_lines, start=first_idx + 2):
        if is_closing_fence_for(line, outer_char, outer_len):
            return False, f"early_outer_fence_close_at_line_{local_i}"

    for local_i, line in enumerate(body_lines, start=first_idx + 2):
        fence = detect_fence(line)
        if fence is None:
            continue

        char, length, suffix = fence
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


def check_json_answer(answer: str) -> tuple[bool, str]:
    try:
        json.loads(answer.strip())
        return True, "ok"
    except json.JSONDecodeError as e:
        return False, f"invalid_json:{e}"


def check_mixed_answer(answer: str) -> tuple[bool, str]:
    ok, msg = check_markdown_fence_answer(answer)
    if not ok:
        return False, "mixed_markdown_failed:" + msg

    json_blocks = extract_json_blocks_from_markdown(answer)
    if not json_blocks:
        return False, "mixed_no_json_block_found"

    for idx, block in enumerate(json_blocks):
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            return False, f"mixed_inner_json_invalid_block_{idx}:{e}"

    return True, "ok"


def evaluate_one(sample: dict[str, Any], output: str) -> tuple[bool, str, str]:
    task_type = str(sample.get("task_type", "")).lower()
    fmt = str(sample.get("format", "")).lower()

    if fmt == "json" or "json_escape" in task_type:
        ok, reason = check_json_answer(output)
        return ok, reason, "json"

    if task_type == "mixed" or fmt == "mixed":
        ok, reason = check_mixed_answer(output)
        return ok, reason, "mixed"

    if fmt == "markdown" or "markdown" in task_type or "fence" in task_type:
        ok, reason = check_markdown_fence_answer(output)
        return ok, reason, "markdown"

    return False, f"unknown_task_or_format:{task_type}/{fmt}", "unknown"


def compute_metrics(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    failure_reasons = defaultdict(int)

    total = 0
    passed = 0

    for row in case_rows:
        total += 1
        group = row["group"]
        by_group[group]["total"] += 1

        if row["ok"]:
            passed += 1
            by_group[group]["passed"] += 1
        else:
            by_group[group]["failed"] += 1
            failure_reasons[row["reason"].split(":")[0]] += 1

    failed = total - passed

    metrics = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "overall_pass_rate": passed / total if total else 0.0,
        "fer": failed / total if total else 0.0,
        "failure_reasons": dict(failure_reasons),
        "by_group": {},
    }

    for group, stat in by_group.items():
        group_total = stat["total"]
        pass_rate = stat["passed"] / group_total if group_total else 0.0
        metrics["by_group"][group] = {
            **stat,
            "pass_rate": pass_rate,
        }

    metrics["json_ppr"] = metrics["by_group"].get("json", {}).get("pass_rate", 0.0)
    metrics["markdown_tdr"] = metrics["by_group"].get("markdown", {}).get("pass_rate", 0.0)
    metrics["mixed_ppr"] = metrics["by_group"].get("mixed", {}).get("pass_rate", 0.0)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/baseline_metrics.json")
    parser.add_argument("--cases-output", default="results/baseline_eval_cases.jsonl")
    parser.add_argument("--failures-output", default="results/baseline_failure_cases.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample", action="store_true", help="Use sampling. Default is greedy decoding.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = RegexTokenizer.load(PROJECT_ROOT / args.tokenizer)
    model = load_model(args.model, tokenizer, device)

    samples = load_jsonl(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]

    case_rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples, start=1):
        output = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=str(sample["prompt"]),
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=args.sample,
        )

        ok, reason, group = evaluate_one(sample, output)

        case_rows.append({
            "id": sample.get("id"),
            "line_no": sample.get("_line_no"),
            "task_type": sample.get("task_type"),
            "format": sample.get("format"),
            "group": group,
            "ok": ok,
            "reason": reason,
            "prompt": sample.get("prompt"),
            "target_answer": sample.get("answer"),
            "model_output": output,
        })

        print(f"[{idx}/{len(samples)}] {sample.get('id')} {group} ok={ok} reason={reason}")

    metrics = compute_metrics(case_rows)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(args.cases_output, case_rows)
    failures = [row for row in case_rows if not row["ok"]]
    write_jsonl(args.failures_output, failures)

    print("\nEvaluation finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {args.output}")
    print(f"Cases saved to: {args.cases_output}")
    print(f"Failures saved to: {args.failures_output}")


if __name__ == "__main__":
    main()
