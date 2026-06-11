"""Generate a small structure-focused corpus for continued pretraining.

This is intentionally much smaller than the first-round structure corpus. The
knowledge continuation stage should mostly improve facts and explanations.

Examples:
    python -B scripts/generate_continue_structure_data.py
    python -B scripts/generate_continue_structure_data.py --target-mb 80 --force
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


TOPICS = [
    "\u77e5\u8bc6\u68c0\u7d22",
    "\u6570\u636e\u6e05\u6d17",
    "\u8bad\u7ec3\u914d\u7f6e",
    "\u7ed3\u6784\u6821\u9a8c",
    "\u65e5\u5fd7\u89e3\u6790",
    "\u5b9e\u9a8c\u8bb0\u5f55",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ident(rng: random.Random, prefix: str = "item") -> str:
    tail = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
    return f"{prefix}_{tail}"


def zh_sentence(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    return f"{topic}\u9700\u8981\u4fdd\u7559\u5b57\u6bb5\u987a\u5e8f\u3001\u8fb9\u754c\u7b26\u53f7\u548c\u8f6c\u4e49\u5b57\u7b26\u3002"


def make_json_sample(rng: random.Random) -> str:
    rows = []
    for index in range(rng.randint(2, 5)):
        rows.append(
            {
                "id": ident(rng, "case"),
                "step": index,
                "title": rng.choice(TOPICS),
                "ok": rng.choice([True, False]),
                "note": zh_sentence(rng),
                "values": [rng.randint(1, 99), round(rng.random(), 4), None],
            }
        )
    return json.dumps({"task": rng.choice(TOPICS), "items": rows}, ensure_ascii=False, indent=2)


def make_yaml_sample(rng: random.Random) -> str:
    name = ident(rng, "service")
    return f"""version: 1
service:
  name: {name}
  owner: research-team
  description: "{zh_sentence(rng)}"
training:
  batch_size: {rng.choice([1, 2, 4])}
  learning_rate: {rng.choice(["3.0e-5", "5.0e-5", "1.0e-4"])}
  checkpoint: checkpoints/{name}.pt
"""


def make_python_sample(rng: random.Random) -> str:
    fn = ident(rng, "load")
    return f'''import json
from pathlib import Path


def {fn}(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)
    return rows
'''


def make_markdown_sample(rng: random.Random) -> str:
    outer = "~" * rng.randint(4, 6)
    return f"""{outer}markdown
# {rng.choice(TOPICS)}

{zh_sentence(rng)}

```json
{make_json_sample(rng)}
```

```yaml
{make_yaml_sample(rng)}
```
{outer}
"""


def make_sample(rng: random.Random) -> str:
    maker = rng.choices(
        [make_json_sample, make_yaml_sample, make_python_sample, make_markdown_sample],
        weights=[35, 20, 20, 25],
        k=1,
    )[0]
    parts = [maker(rng)]
    while sum(len(part) for part in parts) < rng.randint(900, 1800):
        parts.append(maker(rng))
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pretrain/continue_knowledge/structure_light.jsonl")
    parser.add_argument("--target-mb", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)
    target_bytes = args.target_mb * 1024 * 1024
    if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
        print(f"Skip existing file: {output_path}")
        return
    if args.dry_run:
        print(f"Would write {output_path} target={args.target_mb} MB")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    written_bytes = 0
    rows = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        while written_bytes < target_bytes:
            line = json.dumps({"text": make_sample(rng)}, ensure_ascii=False) + "\n"
            file.write(line)
            written_bytes += len(line.encode("utf-8"))
            rows += 1
            if rows % args.log_every == 0:
                print(f"structure_light: {written_bytes / 1024 / 1024:.1f} MB, rows={rows}")
    print(f"Finished structure_light: {output_path} {written_bytes / 1024 / 1024:.1f} MB rows={rows}")


if __name__ == "__main__":
    main()
