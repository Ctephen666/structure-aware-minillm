"""Generate synthetic structure-heavy pretraining JSONL.

Examples:
    python scripts/generate_structure_data.py
    python scripts/generate_structure_data.py --target-mb 900 --force
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


TOPICS = [
    "检索系统",
    "训练任务",
    "配置中心",
    "日志解析",
    "数据清洗",
    "权限审计",
    "缓存策略",
    "异步队列",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ident(rng: random.Random, prefix: str = "item") -> str:
    tail = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
    return f"{prefix}_{tail}"


def sentence(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    return f"{topic}需要保持字段顺序、边界符号和转义字符稳定，避免在长上下文中破坏结构。"


def make_json_sample(rng: random.Random) -> str:
    rows = []
    for idx in range(rng.randint(3, 8)):
        rows.append(
            {
                "id": ident(rng, "case"),
                "step": idx,
                "enabled": rng.choice([True, False]),
                "notes": sentence(rng),
                "payload": {
                    "path": f"C:\\\\workspace\\\\{ident(rng)}\\\\config.json",
                    "quote": "他说：\"请保留 JSON 字符串里的转义。\"",
                    "values": [rng.randint(1, 99), rng.random(), None],
                },
            }
        )
    return json.dumps({"task": rng.choice(TOPICS), "items": rows}, ensure_ascii=False, indent=2)


def make_yaml_sample(rng: random.Random) -> str:
    name = ident(rng, "service")
    return f"""version: 1
service:
  name: {name}
  owner: platform-team
  description: "{sentence(rng)}"
  retry:
    max_attempts: {rng.randint(2, 8)}
    backoff: exponential
  routes:
    - path: /api/{name}/status
      method: GET
      auth: false
    - path: /api/{name}/items
      method: POST
      auth: true
features:
  json_mode: true
  markdown_report: true
  escape_check: "\\\\n \\\\t \\\\\""
"""


def make_python_sample(rng: random.Random) -> str:
    fn = ident(rng, "parse")
    return f'''from __future__ import annotations

import json
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


def main() -> None:
    print({fn}(r"C:\\data\\sample.jsonl")[:2])


if __name__ == "__main__":
    main()
'''


def make_js_sample(rng: random.Random) -> str:
    fn = ident(rng, "render")
    return f"""export function {fn}(items) {{
  const escaped = items.map((item, index) => ({{
    id: `${{index}}-${{item.id ?? "unknown"}}`,
    text: String(item.text ?? "").replace(/\\\\n/g, "\\\\n"),
    meta: {{
      active: Boolean(item.active),
      tags: Array.isArray(item.tags) ? item.tags : []
    }}
  }}));
  return JSON.stringify({{ count: escaped.length, items: escaped }}, null, 2);
}}

console.log({fn}([{{ id: "a", text: "第一行\\n第二行", active: true }}]));
"""


def make_markdown_sample(rng: random.Random) -> str:
    fence = "~" * rng.randint(4, 6)
    inner = "`" * 3
    return f"""{fence}markdown
# {rng.choice(TOPICS)}结构报告

{sentence(rng)}

{inner}json
{make_json_sample(rng)}
{inner}

{inner}yaml
{make_yaml_sample(rng)}
{inner}

请注意内部代码块不能提前关闭外层 fence。
{fence}
"""


def make_nested_code_sample(rng: random.Random) -> str:
    outer = "~" * 5
    return f"""{outer}markdown
下面是一个包含多层代码块的说明：

```python
def build_prompt():
    return \"\"\"请输出 JSON：
    ```json
    {{"name": "demo", "escaped": "\\\\n\\\\t\\\\""}}
    ```
    \"\"\"
```

```javascript
{make_js_sample(rng)}
```
{outer}
"""


def make_escape_sample(rng: random.Random) -> str:
    text = {
        "windows_path": f"C:\\\\Users\\\\runner\\\\{ident(rng)}\\\\output.txt",
        "regex": r"^(?P<name>[\w\-]+)\s*:\s*(?P<value>.+)$",
        "dialog": "用户说：\"保留换行\\n和制表\\t，不要吞掉反斜杠\\\\。\"",
        "markdown": "```json\n{\"ok\": true, \"msg\": \"hello\"}\n```",
    }
    return json.dumps(text, ensure_ascii=False, indent=2)


def make_sample(rng: random.Random) -> str:
    maker = rng.choices(
        [
            make_json_sample,
            make_markdown_sample,
            make_yaml_sample,
            make_python_sample,
            make_js_sample,
            make_nested_code_sample,
            make_escape_sample,
        ],
        weights=[20, 20, 12, 16, 16, 10, 6],
        k=1,
    )[0]
    parts = [maker(rng)]
    while sum(len(part) for part in parts) < rng.randint(1800, 4200):
        parts.append(maker(rng))
    return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pretrain/structure_synthetic.jsonl")
    parser.add_argument("--target-mb", type=int, default=900)
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
                print(f"structure: {written_bytes / 1024 / 1024:.1f} MB, rows={rows}")
    print(f"Finished structure data: {output_path} {written_bytes / 1024 / 1024:.1f} MB rows={rows}")


if __name__ == "__main__":
    main()
