"""Prepare instruction SFT data for the 80M structure-aware model.

The built-in seed set is intentionally factual and compact. You can mix extra
JSONL files with fields like prompt/answer, instruction/output, or question/answer.

Examples:
    python -B scripts/prepare_sft_qa_data.py
    python -B scripts/prepare_sft_qa_data.py --target-size 5000 --force
    python -B scripts/prepare_sft_qa_data.py --extra-jsonl data/my_sft.jsonl --target-size 30000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


FACTS = [
    ("水的沸点是多少？", "在标准大气压下，水的沸点约为 100 摄氏度。"),
    ("太阳主要由什么组成？", "太阳主要由氢和氦组成，其中氢占主要部分。"),
    ("太阳是什么？", "太阳是太阳系中心的恒星，主要由氢和氦组成。"),
    ("地球围绕什么运动？", "地球围绕太阳公转，同时也绕自身轴自转。"),
    ("月球是什么？", "月球是地球的天然卫星。"),
    ("光合作用是什么？", "光合作用是绿色植物利用光能，将二氧化碳和水合成有机物，并释放氧气的过程。"),
    ("牛顿第一定律说明什么？", "牛顿第一定律说明，物体在不受外力或合外力为零时，会保持静止或匀速直线运动状态。"),
    ("电流的单位是什么？", "电流的国际单位是安培，符号是 A。"),
    ("电压的单位是什么？", "电压的国际单位是伏特，符号是 V。"),
    ("能量的国际单位是什么？", "能量的国际单位是焦耳，符号是 J。"),
    ("人体呼吸系统的主要功能是什么？", "人体呼吸系统的主要功能是吸入氧气、排出二氧化碳，并完成气体交换。"),
    ("血液循环的主要作用是什么？", "血液循环负责运输氧气、营养物质、代谢废物和激素，维持身体正常运行。"),
    ("DNA 的主要作用是什么？", "DNA 是遗传信息的主要载体，指导生物体的生长、发育和遗传。"),
    ("细胞是什么？", "细胞是生物体结构和功能的基本单位。"),
    ("蒸发和沸腾有什么区别？", "蒸发可以在任何温度下发生，只发生在液体表面；沸腾在达到沸点时发生，液体内部和表面都会产生汽化。"),
    ("摄氏度和开尔文有什么关系？", "开尔文温度约等于摄氏温度加 273.15。"),
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("长江是什么？", "长江是中国最长的河流，也是世界上较长的河流之一。"),
    ("JSON 是什么？", "JSON 是一种轻量级数据交换格式，常用于表示对象、数组、字符串、数字、布尔值和空值。"),
    ("YAML 是什么？", "YAML 是一种可读性较强的数据序列化格式，常用于配置文件。"),
    ("Python 中 list 和 tuple 的区别是什么？", "list 是可变序列，可以修改元素；tuple 是不可变序列，创建后通常不能修改。"),
    ("机器学习中的过拟合是什么？", "过拟合是指模型过度记住训练数据细节，导致在新数据上的泛化能力变差。"),
    ("什么是训练集和验证集？", "训练集用于更新模型参数，验证集用于评估模型在未参与训练数据上的表现。"),
    ("什么是学习率？", "学习率控制每次参数更新的步长，过大可能不稳定，过小会训练缓慢。"),
    ("Transformer 的注意力机制有什么作用？", "注意力机制让模型根据上下文为不同 token 分配不同权重，从而捕捉长距离依赖。"),
    ("语言模型的预训练目标是什么？", "语言模型预训练通常通过预测下一个 token，让模型学习语言规律和知识分布。"),
    ("SFT 是什么？", "SFT 是监督微调，通过 prompt 和 answer 样本训练模型按照指令生成答案。"),
    ("什么是梯度累积？", "梯度累积是在多次小 batch 前向和反向传播后再更新参数，用来模拟更大的 batch size。"),
    ("什么是 checkpoint？", "checkpoint 是训练过程中保存的模型参数和训练状态，可用于恢复训练或推理。"),
]


EXPLANATIONS = [
    (
        "请解释为什么标准大气压下水的沸点约为 100 摄氏度。",
        "水的沸点取决于外界压强。在标准大气压下，水蒸气压达到外界压强时开始大量汽化，因此沸点约为 100 摄氏度。压强降低时沸点会下降，压强升高时沸点会上升。",
    ),
    (
        "请简要解释太阳为什么会发光发热。",
        "太阳内部发生核聚变反应，氢原子核聚变成氦原子核，并释放大量能量。这些能量最终以光和热的形式向外辐射。",
    ),
    (
        "请解释光合作用对生态系统的重要性。",
        "光合作用把太阳能转化为有机物中的化学能，同时释放氧气。它为生态系统提供基础能量来源，也是大气中氧气的重要来源。",
    ),
    (
        "请解释 JSON 为什么适合做数据交换格式。",
        "JSON 语法简单，容易被人阅读，也容易被程序解析。它能表示对象、数组、字符串、数字、布尔值和空值，因此适合在不同系统之间传递结构化数据。",
    ),
    (
        "请解释过拟合为什么会降低模型泛化能力。",
        "过拟合时模型学习了训练集中的偶然噪声和细节，而不是稳定规律。因此它在训练集上表现很好，但遇到新数据时容易出错。",
    ),
]


UNKNOWN_QA = [
    (
        "如果你不知道某个问题的答案，应该怎么回答？",
        "如果我不知道答案，应该明确说明不确定，并给出可以核查的方向，而不是编造信息。",
    ),
    (
        "请回答：某个不存在的化学元素 Xy-999 的熔点是多少？",
        "我无法确认 Xy-999 是真实存在的化学元素，因此不能给出可靠熔点。建议先核查元素名称或权威资料。",
    ),
]


PROMPT_VARIANTS = [
    "{question}",
    "请回答：{question}",
    "用一句话回答：{question}",
    "请给出简洁准确的回答：{question}",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def digest_row(prompt: str, answer: str) -> str:
    return hashlib.blake2b(f"{prompt}\n{answer}".encode("utf-8"), digest_size=16).hexdigest()


def normalize_row(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    answer = row.get("answer") or row.get("output") or row.get("response")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    prompt = prompt.strip()
    answer = answer.strip()
    if not prompt or not answer:
        return None
    return {"prompt": prompt, "answer": answer}


def iter_extra_rows(paths: Iterable[str]) -> Iterable[dict[str, str]]:
    for value in paths:
        path = resolve_path(value)
        if not path.exists():
            raise FileNotFoundError(f"Missing extra SFT JSONL: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    continue
                normalized = normalize_row(row)
                if normalized is not None:
                    yield normalized


def structure_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt": "请用 JSON 表示一个用户信息，包含 name、age 和 skills。",
            "answer": '{\n  "name": "张三",\n  "age": 28,\n  "skills": ["Python", "数据分析"]\n}',
        },
        {
            "prompt": "请写一个读取 JSONL 文件的 Python 函数。",
            "answer": '```python\nimport json\nfrom pathlib import Path\n\n\ndef read_jsonl(path):\n    rows = []\n    with Path(path).open("r", encoding="utf-8") as file:\n        for line in file:\n            if line.strip():\n                rows.append(json.loads(line))\n    return rows\n```',
        },
        {
            "prompt": "请给出一个简单的 YAML 训练配置示例。",
            "answer": "```yaml\nmodel:\n  name: struct-mini-llm\n  layers: 16\ntraining:\n  batch_size: 2\n  learning_rate: 1.0e-5\n```",
        },
        {
            "prompt": "请用 Markdown 写一个包含标题、列表和代码块的示例。",
            "answer": "# 示例\n\n- 第一项\n- 第二项\n\n```json\n{\n  \"ok\": true\n}\n```",
        },
    ]


def built_in_rows(rng: random.Random, target_size: int) -> list[dict[str, str]]:
    base_rows: list[dict[str, str]] = []
    for question, answer in FACTS:
        for template in PROMPT_VARIANTS:
            base_rows.append({"prompt": template.format(question=question), "answer": answer})
    for prompt, answer in EXPLANATIONS:
        base_rows.append({"prompt": prompt, "answer": answer})
    for prompt, answer in UNKNOWN_QA:
        base_rows.append({"prompt": prompt, "answer": answer})
    base_rows.extend(structure_rows())

    rows: list[dict[str, str]] = []
    while len(rows) < target_size:
        row = dict(rng.choice(base_rows))
        if rng.random() < 0.25 and not row["prompt"].startswith("请"):
            row["prompt"] = "请准确回答：" + row["prompt"]
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", default="data/sft/sft_train.jsonl")
    parser.add_argument("--valid-output", default="data/sft/sft_valid.jsonl")
    parser.add_argument("--extra-jsonl", action="append", default=[])
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate prompt/answer pairs after mixing extras.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.valid_ratio < 0.5:
        raise ValueError(f"valid-ratio must be between 0 and 0.5, got {args.valid_ratio}.")

    train_output = resolve_path(args.train_output)
    valid_output = resolve_path(args.valid_output)
    for path in (train_output, valid_output):
        if path.exists() and path.stat().st_size > 0 and not args.force:
            raise FileExistsError(f"Output exists, pass --force to overwrite: {path}")

    rng = random.Random(args.seed)
    rows = built_in_rows(rng, args.target_size)
    rows.extend(iter_extra_rows(args.extra_jsonl))

    if args.dedupe:
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            key = digest_row(row["prompt"], row["answer"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped

    rng.shuffle(rows)
    valid_size = max(1, int(len(rows) * args.valid_ratio))
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    print(f"SFT rows: train={len(train_rows)}, valid={len(valid_rows)}, total={len(rows)}")
    if args.dry_run:
        print(f"Would write train={train_output}")
        print(f"Would write valid={valid_output}")
        return

    write_jsonl(train_output, train_rows)
    write_jsonl(valid_output, valid_rows)
    print(f"Wrote train: {train_output}")
    print(f"Wrote valid: {valid_output}")


if __name__ == "__main__":
    main()
