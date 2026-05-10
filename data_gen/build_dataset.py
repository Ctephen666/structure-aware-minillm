"""Build structure-trap data for baseline and Struct-LM experiments.

Usage:
    python data_gen/build_dataset.py --out_dir data --train 8000 --valid 800 --test 800
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


BT = chr(96)
BT3 = BT * 3
BT4 = BT * 4

API_NAMES = ["login", "register", "uploadImage", "getProfile", "createOrder", "searchUser"]
DOC_TITLES = ["API Guide", "Agent Config", "Backend Contract", "Module Notes", "Integration Manual"]


def dump_json(obj: Any, indent: int | None = None) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def make_markdown_sample(idx: int) -> dict[str, Any]:
    api = random.choice(API_NAMES)
    title = random.choice(DOC_TITLES)
    payload = {
        "code": 200,
        "message": "success",
        "data": {"api": api, "token": "abc123"},
    }
    payload_text = dump_json(payload, indent=2)

    prompt = (
        f"Write Markdown source for the {api} API. The whole answer must be wrapped in a "
        "markdown code fence and must include an inner JSON code block."
    )
    answer = (
        f"{BT4}markdown\n"
        f"# {title}\n\n"
        f"The `{api}` endpoint returns this payload:\n\n"
        f"{BT3}json\n"
        f"{payload_text}\n"
        f"{BT3}\n\n"
        "Store this content as literal Markdown source.\n"
        f"{BT4}"
    )
    unsafe_answer = (
        f"{BT3}markdown\n"
        f"# {title}\n\n"
        f"{BT3}json\n"
        f"{payload_text}\n"
        f"{BT3}\n\n"
        "This can close the outer fence too early.\n"
        f"{BT3}"
    )
    return {
        "id": f"md_trap_{idx:06d}",
        "task_type": "markdown_fence",
        "format": "markdown",
        "prompt": prompt,
        "answer": answer,
        "unsafe_answer": unsafe_answer,
        "unsafe_pattern": "outer_BT3_inner_BT3_collision",
        "expected_strategy": "outer_BT4",
    }


def make_json_sample(idx: int) -> dict[str, Any]:
    api = random.choice(API_NAMES)
    inner = {
        "status": "success",
        "api": api,
        "data": {"id": random.randint(1, 999), "enabled": True},
    }
    inner_text = dump_json(inner)
    outer = {
        "agent_name": "StructureFormatter",
        "version": "1.0",
        "configuration": {
            "model_type": "mini-llm",
            "system_prompt_template": "Return JSON like this: " + inner_text,
        },
    }
    answer = dump_json(outer)
    unsafe_answer = (
        '{"agent_name":"StructureFormatter","version":"1.0","configuration":'
        '{"model_type":"mini-llm","system_prompt_template":"Return JSON like this: '
        + inner_text
        + '"}}'
    )
    prompt = (
        f"Generate one valid JSON config for an agent. The system_prompt_template field must contain "
        f"a JSON example for the {api} API inside the string."
    )
    return {
        "id": f"json_trap_{idx:06d}",
        "task_type": "json_escape",
        "format": "json",
        "prompt": prompt,
        "answer": answer,
        "unsafe_answer": unsafe_answer,
        "unsafe_pattern": "unescaped_inner_json_quotes",
        "expected_strategy": "escape_inner_quotes",
    }


def make_mixed_sample(idx: int) -> dict[str, Any]:
    api = random.choice(API_NAMES)
    inner = {"status": "success", "api": api, "data": {"id": random.randint(1, 999)}}
    outer_json = {
        "agent_name": "MarkdownJsonAgent",
        "configuration": {
            "system_prompt_template": "Return JSON like this: " + dump_json(inner),
        },
    }
    json_block = dump_json(outer_json, indent=2)
    prompt = (
        "Generate Markdown source wrapped in a markdown code fence. Inside it include a JSON code block, "
        "and inside that JSON include another JSON example inside a string field."
    )
    answer = (
        f"{BT4}markdown\n"
        "# Mixed Structure Config\n\n"
        "Below is the agent configuration:\n\n"
        f"{BT3}json\n"
        f"{json_block}\n"
        f"{BT3}\n\n"
        "This document tests Markdown fences and JSON string escaping together.\n"
        f"{BT4}"
    )
    unsafe_answer = (
        f"{BT3}markdown\n"
        "# Mixed Structure Config\n\n"
        f"{BT3}json\n"
        f"{json_block}\n"
        f"{BT3}\n\n"
        "The outer fence can close too early here.\n"
        f"{BT3}"
    )
    return {
        "id": f"mixed_trap_{idx:06d}",
        "task_type": "mixed",
        "format": "markdown",
        "prompt": prompt,
        "answer": answer,
        "unsafe_answer": unsafe_answer,
        "unsafe_pattern": "markdown_fence_plus_json_escape",
        "expected_strategy": "outer_BT4_and_escape_inner_quotes",
    }


def build_split(count: int, start_idx: int = 0) -> list[dict[str, Any]]:
    rows = []
    makers = [make_markdown_sample, make_json_sample, make_mixed_sample]
    for i in range(count):
        rows.append(makers[i % len(makers)](start_idx + i))
    random.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--valid", type=int, default=800)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    train = build_split(args.train, 0)
    valid = build_split(args.valid, args.train)
    test = build_split(args.test, args.train + args.valid)

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "test.jsonl", test)
    write_jsonl(out_dir / "preview_100.jsonl", train[:100])

    print(f"Wrote {len(train)} train samples to {out_dir / 'train.jsonl'}")
    print(f"Wrote {len(valid)} valid samples to {out_dir / 'valid.jsonl'}")
    print(f"Wrote {len(test)} test samples to {out_dir / 'test.jsonl'}")
    print(f"Wrote preview to {out_dir / 'preview_100.jsonl'}")


if __name__ == "__main__":
    main()
