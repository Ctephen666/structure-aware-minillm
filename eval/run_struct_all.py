"""Evaluate Struct-LM on structure-trap data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode.generate_struct import generate_response, load_model
from eval.run_all import compute_metrics, evaluate_one, load_jsonl, write_jsonl
from tokenizer.regex_tokenizer import RegexTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/struct_metrics.json")
    parser.add_argument("--cases-output", default="results/struct_eval_cases.jsonl")
    parser.add_argument("--failures-output", default="results/struct_failure_cases.jsonl")
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

    case_rows = []
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

    print("\nStruct evaluation finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to: {args.output}")
    print(f"Cases saved to: {args.cases_output}")
    print(f"Failures saved to: {args.failures_output}")


if __name__ == "__main__":
    main()
