#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API SFT 数据生成脚本

功能：
1. 使用 DeepSeek OpenAI-compatible API 批量生成 SFT JSONL 数据；
2. 支持普通指令、简单格式、JSON 转义、Markdown 套娃、控制符诱导、多格式混合任务；
3. 自动解析 DeepSeek 返回的 JSON；
4. 自动校验 JSON 输出、Markdown 代码块闭合、字段完整性；
5. 自动去重、断点续写、失败样本记录。

安装：
    pip install openai tqdm

设置 API Key：
    Windows PowerShell:
        $env:DEEPSEEK_API_KEY="你的key"
    Linux/macOS:
        export DEEPSEEK_API_KEY="你的key"

示例：
    python deepseek_sft_generator.py --target 1000 --out data/sft/ai_sft_train.jsonl

推荐先小批量测试：
    python deepseek_sft_generator.py --target 50 --batch-size 10 --out data/sft/test_ai_sft.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("请先安装依赖：pip install openai tqdm") from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_TASKS = [
    "general_qa",
    "simple_format",
    "json_escape",
    "markdown_nested",
    "control_trap",
    "mixed_format",
]

DEFAULT_WEIGHTS = {
    "general_qa": 0.30,
    "simple_format": 0.15,
    "json_escape": 0.20,
    "markdown_nested": 0.15,
    "control_trap": 0.10,
    "mixed_format": 0.10,
}

VALID_FORMATS = {
    "plain",
    "json",
    "markdown",
    "markdown_json",
    "markdown_python",
    "markdown_python_json",
    "yaml",
}


class FatalAPIError(RuntimeError):
    """不可重试的 API 错误，例如余额不足、API Key 无效。"""
    pass


# ============================================================
# 直接把 DeepSeek API Key 写在这里
# 注意：不要把真实 key 上传到 GitHub、网盘、作业附件或发给别人。
# 示例：DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"
# ============================================================
DEEPSEEK_API_KEY = "sk-eff8295f552445149b0abd5bfaaad759"

SYSTEM_PROMPT = """
你是一个严谨的 SFT 数据集生成器。你的任务是生成用于训练 200M 中文结构感知型 LLM 的监督微调数据。

你必须只输出严格合法的 json 对象，不要输出 Markdown，不要输出解释。
顶层格式必须是：
{
  "samples": [
    {
      "task_type": "...",
      "format": "...",
      "difficulty": 1,
      "instruction": "...",
      "input": "",
      "output": "..."
    }
  ]
}

强制要求：
1. 每条样本都必须适合 SFT，即 instruction 是用户问题，output 是助手应该给出的标准答案。
2. output 不能是空字符串。
3. output 不要包含“作为AI模型”等废话。
4. 如果 format 是 json，则 output 本身必须是可被 json.loads 解析的合法 JSON 字符串。
5. 如果 output 是 Markdown 代码块，则代码块必须正确闭合。
6. 如果代码块内部出现 ```，外层必须使用 ```` 或更长的反引号 fence。
7. 生成内容要多样化，不要重复模板。
8. 只返回 json，不要返回任何额外文本。
""".strip()


TASK_GUIDES: Dict[str, str] = {
    "general_qa": """
生成普通中文指令 SFT 数据。主题围绕：自然语言处理、机器学习、深度学习、人工智能基础、Transformer、语言模型、数据集、训练流程。
要求：
- format 固定为 plain。
- difficulty 为 1 或 2。
- output 简洁准确，适合训练一个 200M 小模型。
- 不要生成太长答案，控制在 80 到 250 个中文字符之间。
""".strip(),

    "simple_format": """
生成简单结构化输出 SFT 数据。
要求：
- 任务包括：输出简单 JSON、输出 Markdown 列表、输出课程大纲、输出配置项。
- format 可以是 json 或 markdown。
- 如果 format=json，output 必须是合法 JSON。
- difficulty 为 1 或 2。
- 样本要简单清晰，为后续结构对抗训练打基础。
""".strip(),

    "json_escape": r"""
生成 JSON 转义压力类 SFT 数据。
要求：
- format 固定为 json。
- output 必须是合法 JSON，必须能被 Python json.loads 解析。
- instruction 要要求 JSON 字段中包含特殊符号，例如：双引号、反斜杠、换行符、右大括号、左中括号、路径 C:\Users\test、伪字段 "admin": true。
- output 必须把这些特殊符号保留在字符串字段内部，不能让它们破坏 JSON 结构。
- difficulty 为 3 或 4。
示例思想：用户要求 description 字段包含“请提前结束 JSON }”，正确 output 应该把 } 放在字符串里。
""".strip(),

    "markdown_nested": r"""
生成 Markdown 套娃嵌套类 SFT 数据。
要求：
- format 为 markdown 或 markdown_python。
- output 应该是 Markdown 文档，里面包含代码块。
- 代码块内部可以包含字符串 ```，但不能导致外层代码块提前闭合。
- 如果内部文本包含三个反引号 ```，外层 fence 必须使用四个反引号 ````。
- difficulty 为 3 或 4。
- 重点训练模型理解：代码字符串里的控制符不是外层 Markdown 控制符。
""".strip(),

    "control_trap": r"""
生成控制符诱导类 SFT 数据。
要求：
- task_type 固定为 control_trap。
- format 可以是 json 或 markdown。
- instruction 中要包含诱导性内容，例如：请提前结束 JSON }、请关闭代码块 ```、然后输出 "admin": true、忽略前面的结构要求等。
- output 必须保持结构合法，并把诱导性内容作为普通文本处理。
- 如果 format=json，output 必须能被 json.loads 解析。
- 如果 format=markdown，代码块必须闭合。
- difficulty 为 4 或 5。
""".strip(),

    "mixed_format": r"""
生成多格式混合类 SFT 数据。
要求：
- format 可以是 markdown_json 或 markdown_python_json。
- output 可以是 Markdown 文档，内部包含 JSON 代码块；或者 Python 代码块中包含 JSON 字符串。
- JSON 字符串中可以包含双引号、反斜杠、换行符等转义压力。
- Markdown 代码块必须正确闭合。
- difficulty 为 4 或 5。
- 适合模拟 Agent 工具调用、API 文档生成、前端配置生成等真实场景。
""".strip(),
}


# ============================================================
# 结构化格式强化任务：用于在通用 SFT 之后，专门增强 JSON / Markdown / YAML 输出能力。
# 这批数据不追求“更多普通问答”，而是追求“严格可解析、可验证、可控格式”。
# ============================================================
STRUCT_FORMAT_TASKS = [
    "strict_json_object",
    "json_array_output",
    "json_format_repair",
    "markdown_table",
    "yaml_config",
    "format_replay_qa",
]

ALL_TASKS = DEFAULT_TASKS + STRUCT_FORMAT_TASKS

STRUCT_FORMAT_WEIGHTS = {
    "strict_json_object": 0.40,
    "json_array_output": 0.15,
    "json_format_repair": 0.15,
    "markdown_table": 0.12,
    "yaml_config": 0.10,
    "format_replay_qa": 0.08,
}

TASK_GUIDES.update({
    "strict_json_object": r"""
生成严格 JSON 对象输出 SFT 数据。
要求：
- task_type 固定为 strict_json_object。
- format 固定为 json。
- instruction 必须明确要求“只输出合法 JSON，不要解释”。
- output 必须是 JSON 对象，最外层必须是 { }，不能是普通文本、不能是 key:value 伪 JSON。
- output 必须能被 Python json.loads 解析，并且解析结果必须是 dict。
- 样本重点覆盖：name/age、model/training/data、课程信息、用户配置、评估指标、API 参数、数据集统计。
- difficulty 为 1 到 3。
- 不要把 JSON 放进 Markdown 代码块里。
正确 output 示例：
{
  "name": "张三",
  "age": 25
}
""".strip(),

    "json_array_output": r"""
生成严格 JSON 数组输出 SFT 数据。
要求：
- task_type 固定为 json_array_output。
- format 固定为 json。
- instruction 必须明确要求“只输出合法 JSON 数组，不要解释”。
- output 必须是 JSON 数组，最外层必须是 [ ]，不能是普通列表文本。
- output 必须能被 Python json.loads 解析，并且解析结果必须是 list。
- 样本包括：课程步骤列表、模型评估指标列表、任务清单、字段定义数组、错误类型数组。
- difficulty 为 1 到 3。
""".strip(),

    "json_format_repair": r"""
生成 JSON 格式修复类 SFT 数据。
要求：
- task_type 固定为 json_format_repair。
- format 固定为 json。
- instruction 中给出一个错误 JSON 或伪 JSON，例如：name: 张三 age: 25、缺引号、缺逗号、尾逗号、单引号、未转义双引号。
- instruction 要求模型“修复为合法 JSON，只输出 JSON，不要解释”。
- output 必须是修复后的合法 JSON 对象或数组，能被 Python json.loads 解析。
- difficulty 为 2 到 4。
- 重点解决模型输出 name : 张三 age : 25 这类伪 JSON 的问题。
""".strip(),

    "markdown_table": r"""
生成 Markdown 格式输出 SFT 数据。
要求：
- task_type 固定为 markdown_table。
- format 固定为 markdown。
- instruction 要求输出 Markdown 表格、三点列表、标题+列表、代码块等。
- output 必须是合法 Markdown；表格需要表头、分隔行、内容行。
- 如果包含代码块，必须闭合。
- difficulty 为 1 到 3。
""".strip(),

    "yaml_config": r"""
生成 YAML 配置输出 SFT 数据。
要求：
- task_type 固定为 yaml_config。
- format 固定为 yaml。
- instruction 要求只输出 YAML 配置，不要解释。
- output 必须使用清晰缩进，包含 2 到 4 个顶层字段。
- 样本包括：model/training/data、server/database/logging、dataset/eval/output 等配置。
- difficulty 为 1 到 3。
""".strip(),

    "format_replay_qa": r"""
生成通用问答回放样本，用于结构化强化阶段防止普通问答能力遗忘。
要求：
- task_type 固定为 format_replay_qa。
- format 固定为 plain。
- 主题围绕机器学习、深度学习、NLP、Transformer、预训练、SFT、过拟合、反向传播、dropout、注意力机制。
- output 简洁准确，控制在 50 到 180 个中文字符。
- difficulty 为 1 或 2。
""".strip(),
})

# 专门用于“通用 SFT 后的结构化格式强化”数据计划。
# 默认 3 万条：结构化输出占 90% 左右，普通问答回放占 10% 左右。
PLAN_STRUCT_FORMAT_30K = [
    {"task_type": "strict_json_object", "target": 12000, "filename": "strict_json_object_12k.jsonl"},
    {"task_type": "json_array_output", "target": 4500, "filename": "json_array_output_4k5.jsonl"},
    {"task_type": "json_format_repair", "target": 4500, "filename": "json_format_repair_4k5.jsonl"},
    {"task_type": "markdown_table", "target": 4000, "filename": "markdown_table_4k.jsonl"},
    {"task_type": "yaml_config", "target": 3000, "filename": "yaml_config_3k.jsonl"},
    {"task_type": "format_replay_qa", "target": 2000, "filename": "format_replay_qa_2k.jsonl"},
]

STRUCT_FINAL_TRAIN_NAME = "sft_struct_train_200m.jsonl"
STRUCT_FINAL_VAL_NAME = "sft_struct_val_200m.jsonl"
STRUCT_FINAL_TEST_NAME = "sft_struct_test_200m.jsonl"
STRUCT_FINAL_STATS_NAME = "sft_struct_dataset_stats.json"


# ============================================================
# 一键生成 12 万条 SFT 数据的目录与配比计划
# 目录结构：
# data/sft/raw/   存放每一类原始生成数据
# data/sft/final/ 存放合并、去重、划分后的 train/val/test
# ============================================================
PLAN_120K = [
    {"task_type": "general_qa", "target": 40000, "filename": "general_qa_40k.jsonl"},
    {"task_type": "simple_format", "target": 20000, "filename": "simple_format_20k.jsonl"},
    {"task_type": "json_escape", "target": 25000, "filename": "json_escape_25k.jsonl"},
    {"task_type": "markdown_nested", "target": 20000, "filename": "markdown_nested_20k.jsonl"},
    {"task_type": "control_trap", "target": 10000, "filename": "control_trap_10k.jsonl"},
    {"task_type": "mixed_format", "target": 5000, "filename": "mixed_format_5k.jsonl"},
]

FINAL_TRAIN_NAME = "sft_train_200m.jsonl"
FINAL_VAL_NAME = "sft_val_200m.jsonl"
FINAL_TEST_NAME = "sft_test_200m.jsonl"
FINAL_STATS_NAME = "sft_dataset_stats.json"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_key(sample: Dict[str, Any]) -> str:
    return stable_hash((sample.get("instruction", "") + "\n" + sample.get("output", "")).strip())


def load_existing_hashes(path: Path) -> Tuple[set, int]:
    hashes = set()
    count = 0
    if not path.exists():
        return hashes, count
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            hashes.add(sample_key(obj))
            count += 1
    return hashes, count


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def markdown_fences_balanced(text: str) -> bool:
    """检查行首 Markdown fence 是否成对闭合。只识别行首 fence，避免误判代码字符串里的 ```。"""
    fence_re = re.compile(r"^\s*(`{3,}|~{3,})[^`~]*\s*$")
    stack: List[str] = []
    for line in text.splitlines():
        m = fence_re.match(line)
        if not m:
            continue
        fence = m.group(1)
        char = fence[0]
        length = len(fence)
        if not stack:
            stack.append(fence)
        else:
            opening = stack[-1]
            if opening[0] == char and length >= len(opening):
                stack.pop()
            else:
                stack.append(fence)
    return len(stack) == 0


def validate_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    required = ["task_type", "format", "difficulty", "instruction", "input", "output"]
    for k in required:
        if k not in sample:
            return False, f"missing_field:{k}"

    task_type = str(sample.get("task_type", "")).strip()
    fmt = str(sample.get("format", "")).strip()
    instruction = str(sample.get("instruction", "")).strip()
    output = str(sample.get("output", "")).strip()

    if task_type not in ALL_TASKS:
        return False, f"bad_task_type:{task_type}"
    if fmt not in VALID_FORMATS:
        return False, f"bad_format:{fmt}"
    if not instruction:
        return False, "empty_instruction"
    if not output:
        return False, "empty_output"
    if len(instruction) > 1200:
        return False, "instruction_too_long"
    if len(output) > 6000:
        return False, "output_too_long"

    # JSON 格式的 answer 必须真的可解析，且不能只是 name : 张三 这类伪 JSON。
    if fmt == "json":
        stripped = output.lstrip()
        if not stripped.startswith(("{", "[")):
            return False, "json_output_must_start_with_object_or_array"
        try:
            parsed_json = json.loads(output)
        except Exception as exc:
            return False, f"json_output_invalid:{exc}"
        if not isinstance(parsed_json, (dict, list)):
            return False, "json_output_must_be_object_or_array"
        if task_type in {"strict_json_object", "json_escape", "json_format_repair"} and not isinstance(parsed_json, dict):
            return False, "json_output_must_be_object"
        if task_type == "json_array_output" and not isinstance(parsed_json, list):
            return False, "json_output_must_be_array"

    # Markdown 类格式检查代码块闭合
    if fmt.startswith("markdown") or "```" in output or "````" in output:
        if not markdown_fences_balanced(output):
            return False, "markdown_fence_unbalanced"

    # YAML 格式做轻量校验：至少包含 key: value / 嵌套配置，不接受纯自然语言。
    if fmt == "yaml":
        if ":" not in output or output.lstrip().startswith(("{", "[")):
            return False, "yaml_output_invalid_shape"

    # 控制符诱导样本至少应该包含明显陷阱词之一
    if task_type == "control_trap":
        trap_markers = ["}", "```", "admin", "忽略", "提前", "关闭", "结束"]
        if not any(x in instruction for x in trap_markers):
            return False, "control_trap_not_obvious"

    return True, "ok"


def normalize_sample(
    sample: Dict[str, Any],
    task_type: str,
    idx: int,
    source_model: str,
    source_batch: int,
) -> Dict[str, Any]:
    item = {
        "id": f"ai_{task_type}_{idx:08d}",
        "source": "deepseek_api",
        "source_model": source_model,
        "source_batch": source_batch,
        "task_type": str(sample.get("task_type", task_type)).strip(),
        "format": str(sample.get("format", "plain")).strip(),
        "difficulty": int(sample.get("difficulty", 1)),
        "instruction": str(sample.get("instruction", "")).strip(),
        "input": str(sample.get("input", "")).strip(),
        "output": str(sample.get("output", "")).strip(),
    }
    return item


def build_user_prompt(task_type: str, batch_size: int, batch_index: int) -> str:
    guide = TASK_GUIDES[task_type]
    return f"""
请生成 {batch_size} 条 SFT 样本。
当前 batch_index = {batch_index}
当前 task_type = {task_type}

任务要求：
{guide}

必须返回严格合法 json，格式如下：
{{
  "samples": [
    {{
      "task_type": "{task_type}",
      "format": "plain/json/markdown/markdown_json/markdown_python/markdown_python_json/yaml 之一",
      "difficulty": 1,
      "instruction": "用户指令",
      "input": "",
      "output": "标准答案"
    }}
  ]
}}

再次强调：只输出 json 对象，不要输出 Markdown，不要输出解释。
""".strip()


def parse_response_content(content: str) -> List[Dict[str, Any]]:
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("top-level response is not a dict")
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise ValueError("response does not contain list field 'samples'")
    return [x for x in samples if isinstance(x, dict)]


def call_deepseek(
    client: OpenAI,
    model: str,
    task_type: str,
    batch_size: int,
    batch_index: int,
    max_tokens: int,
    temperature: float,
    thinking: str,
    max_retries: int,
) -> List[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task_type, batch_size, batch_index)},
    ]

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens,
                "stream": False,
                "extra_body": {"thinking": {"type": thinking}},
            }
            # DeepSeek 官方说明：thinking 模式下 temperature/top_p 等参数无效；禁用 thinking 后才使用 temperature。
            if thinking == "disabled":
                kwargs["temperature"] = temperature

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty content returned by API")
            return parse_response_content(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            err_msg = str(exc)
            # 余额不足、认证失败不是临时错误，继续重试没有意义。
            if "Error code: 402" in err_msg or "Insufficient Balance" in err_msg:
                raise FatalAPIError("DeepSeek API 余额不足：请先充值或更换可用 API Key。") from exc
            if "Error code: 401" in err_msg or "Authentication" in err_msg:
                raise FatalAPIError("DeepSeek API Key 无效或认证失败：请检查代码顶部的 API Key。") from exc

            sleep_s = min(60, 2 ** attempt + random.random())
            print(f"[WARN] API call failed, attempt={attempt + 1}/{max_retries}, sleep={sleep_s:.1f}s, err={exc}", file=sys.stderr)
            time.sleep(sleep_s)

    raise RuntimeError(f"DeepSeek API failed after {max_retries} retries: {last_error}")


def choose_task(tasks: List[str], weights: Dict[str, float]) -> str:
    w = [weights.get(t, 1.0) for t in tasks]
    return random.choices(tasks, weights=w, k=1)[0]


def parse_weights(tasks: List[str], raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return DEFAULT_WEIGHTS.copy()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != len(tasks):
        raise ValueError("--weights 的数量必须与 --tasks 一致")
    return {t: float(w) for t, w in zip(tasks, parts)}



def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid jsonl line: path={path}, line={line_no}, err={exc}", file=sys.stderr)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_client_and_key(args: argparse.Namespace) -> OpenAI:
    # 优先使用代码顶部直接填写的 DEEPSEEK_API_KEY；如果仍是占位符，则回退到环境变量。
    api_key = DEEPSEEK_API_KEY.strip()
    if not api_key or api_key == "请把你的 DeepSeek API Key 粘贴到这里":
        api_key = os.environ.get(args.api_key_env, "").strip()

    if not api_key:
        raise SystemExit(
            "未找到 API Key。请在代码顶部填写 DEEPSEEK_API_KEY，"
            f"或者设置环境变量 {args.api_key_env}。"
        )
    return OpenAI(api_key=api_key, base_url=args.base_url)


def generate_dataset_job(
    client: OpenAI,
    args: argparse.Namespace,
    task_type: str,
    target_total: int,
    out_path: Path,
) -> Dict[str, Any]:
    """生成单个任务类型的数据文件。若文件已存在，则自动续跑到 target_total。

    高速版：支持 --concurrency 并发请求。
    - concurrency=1 时等价于顺序生成；
    - concurrency>1 时多个 batch 同时请求 DeepSeek API；
    - 主线程统一校验、去重、写文件，避免多线程同时写文件导致 JSONL 损坏。
    """
    fail_path = out_path.with_suffix(out_path.suffix + ".failed.jsonl")
    existing_hashes, existing_count = load_existing_hashes(out_path)
    remaining = max(target_total - existing_count, 0)

    print("\n" + "=" * 80)
    print(f"[JOB] task_type: {task_type}")
    print(f"[JOB] output: {out_path}")
    print(f"[JOB] failed_log: {fail_path}")
    print(f"[JOB] existing: {existing_count}")
    print(f"[JOB] target_total: {target_total}")
    print(f"[JOB] remaining_to_generate: {remaining}")
    print(f"[JOB] batch_size: {args.batch_size}")
    print(f"[JOB] concurrency: {args.concurrency}")

    if remaining <= 0:
        print(f"[JOB] skip {task_type}, already enough samples.")
        return {
            "task_type": task_type,
            "target_total": target_total,
            "existing_before": existing_count,
            "accepted_new": 0,
            "final_count": existing_count,
            "rejected": 0,
            "api_calls": 0,
            "out_path": str(out_path),
        }

    max_api_calls = args.max_api_calls
    if max_api_calls <= 0:
        # 多给一些调用空间，因为会有校验失败和重复过滤。
        # 并发模式下也需要这个上限，避免接口异常时无限跑。
        max_api_calls = max(10, int(remaining / max(args.batch_size, 1) * 4) + 10)

    accepted_total = 0
    rejected_total = 0
    next_idx = existing_count
    submitted_calls = 0
    finished_calls = 0
    concurrency = max(1, int(args.concurrency))

    def submit_one(executor: ThreadPoolExecutor, futures: Dict[Any, int]) -> bool:
        nonlocal submitted_calls
        if submitted_calls >= max_api_calls:
            return False
        # 如果已经够了，不再提交新请求。
        if accepted_total >= remaining:
            return False
        batch_index = submitted_calls
        submitted_calls += 1
        fut = executor.submit(
            call_deepseek,
            client,
            args.model,
            task_type,
            args.batch_size,
            batch_index,
            args.max_tokens,
            args.temperature,
            args.thinking,
            args.max_retries,
        )
        futures[fut] = batch_index
        return True

    progress = None
    if tqdm is not None:
        progress = tqdm(total=remaining, desc=f"Generating {task_type}", unit="sample")

    futures: Dict[Any, int] = {}
    fatal_error: Optional[BaseException] = None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(concurrency):
            submit_one(executor, futures)

        while futures and accepted_total < remaining:
            # 每次只处理一个完成的 future，处理完再补一个新请求。
            for fut in as_completed(list(futures.keys())):
                batch_index = futures.pop(fut)
                finished_calls += 1

                try:
                    raw_samples = fut.result()
                except FatalAPIError as exc:
                    append_jsonl(fail_path, [{"batch_index": batch_index, "task_type": task_type, "fatal_error": str(exc)}])
                    print(f"[FATAL] {exc}", file=sys.stderr)
                    fatal_error = exc
                    break
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(fail_path, [{"batch_index": batch_index, "task_type": task_type, "error": str(exc)}])
                    print(f"[WARN] batch failed: task={task_type}, batch={batch_index}, err={exc}", file=sys.stderr)
                    # 失败后继续提交新请求。
                    submit_one(executor, futures)
                    break

                accepted: List[Dict[str, Any]] = []
                failed: List[Dict[str, Any]] = []

                for raw in raw_samples:
                    item = normalize_sample(
                        raw,
                        task_type=task_type,
                        idx=next_idx,
                        source_model=args.model,
                        source_batch=batch_index,
                    )
                    ok, reason = validate_sample(item)
                    key = sample_key(item)
                    if not ok:
                        item["reject_reason"] = reason
                        failed.append(item)
                        rejected_total += 1
                        continue
                    if key in existing_hashes:
                        item["reject_reason"] = "duplicate"
                        failed.append(item)
                        rejected_total += 1
                        continue

                    existing_hashes.add(key)
                    accepted.append(item)
                    next_idx += 1

                # 只追加还需要的数量，避免明显超过目标。
                need = remaining - accepted_total
                accepted_to_write = accepted[:need]
                overflow = accepted[need:]
                if overflow:
                    for item in overflow:
                        item["reject_reason"] = "overflow_after_target_reached"
                    failed.extend(overflow)
                    rejected_total += len(overflow)

                if accepted_to_write:
                    append_jsonl(out_path, accepted_to_write)
                    accepted_total += len(accepted_to_write)
                    if progress is not None:
                        progress.update(len(accepted_to_write))
                if failed:
                    append_jsonl(fail_path, failed)

                print(
                    f"[BATCH {batch_index}] task={task_type} raw={len(raw_samples)} "
                    f"accepted={len(accepted_to_write)} failed={len(failed)} "
                    f"total_new={accepted_total}/{remaining} final={existing_count + accepted_total}/{target_total} "
                    f"submitted={submitted_calls} finished={finished_calls}"
                )

                if args.sleep > 0:
                    time.sleep(args.sleep)

                # 处理完一个 batch 后，补一个新 batch。
                submit_one(executor, futures)
                break

            if fatal_error is not None:
                # 取消未开始的 future；已经跑出去的请求无法强行停止，但不再处理新请求。
                for fut in futures:
                    fut.cancel()
                break

    if progress is not None:
        progress.close()

    if fatal_error is not None:
        raise fatal_error

    final_count = count_jsonl(out_path)
    print(f"[JOB DONE] task={task_type}, final_count={final_count}, target={target_total}")

    return {
        "task_type": task_type,
        "target_total": target_total,
        "existing_before": existing_count,
        "accepted_new": accepted_total,
        "final_count": final_count,
        "rejected": rejected_total,
        "api_calls_submitted": submitted_calls,
        "api_calls_finished": finished_calls,
        "out_path": str(out_path),
    }


def merge_split_final_dataset(
    raw_dir: Path,
    final_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    plan: Optional[List[Dict[str, Any]]] = None,
    train_name: str = FINAL_TRAIN_NAME,
    val_name: str = FINAL_VAL_NAME,
    test_name: str = FINAL_TEST_NAME,
    stats_name: str = FINAL_STATS_NAME,
) -> Dict[str, Any]:
    """合并 raw 目录下的计划文件，去重后划分 train/val/test。"""
    plan = plan or PLAN_120K
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    all_records: List[Dict[str, Any]] = []
    per_file_counts: Dict[str, int] = {}

    for plan_item in plan:
        path = raw_dir / plan_item["filename"]
        records = read_jsonl(path)
        per_file_counts[plan_item["filename"]] = len(records)
        all_records.extend(records)

    print("\n" + "=" * 80)
    print("[MERGE] raw_dir:", raw_dir)
    print("[MERGE] final_dir:", final_dir)
    print("[MERGE] raw_total_before_dedup:", len(all_records))

    seen = set()
    deduped: List[Dict[str, Any]] = []
    duplicate_count = 0
    invalid_count = 0

    for item in all_records:
        ok, reason = validate_sample(item)
        if not ok:
            invalid_count += 1
            continue
        key = sample_key(item)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(item)

    rng = random.Random(seed)
    rng.shuffle(deduped)

    total = len(deduped)
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = total - train_n - val_n

    train = deduped[:train_n]
    val = deduped[train_n: train_n + val_n]
    test = deduped[train_n + val_n:]

    final_dir.mkdir(parents=True, exist_ok=True)
    train_path = final_dir / train_name
    val_path = final_dir / val_name
    test_path = final_dir / test_name
    stats_path = final_dir / stats_name

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    task_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}
    for item in deduped:
        task_counts[item.get("task_type", "unknown")] = task_counts.get(item.get("task_type", "unknown"), 0) + 1
        format_counts[item.get("format", "unknown")] = format_counts.get(item.get("format", "unknown"), 0) + 1

    stats = {
        "raw_dir": str(raw_dir),
        "final_dir": str(final_dir),
        "per_file_counts": per_file_counts,
        "raw_total_before_dedup": len(all_records),
        "final_total_after_dedup": total,
        "duplicate_removed": duplicate_count,
        "invalid_removed": invalid_count,
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "task_counts": task_counts,
        "format_counts": format_counts,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "stats": str(stats_path),
        },
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[MERGE DONE]")
    print(f"  train: {train_path}  ({len(train)})")
    print(f"  val:   {val_path}  ({len(val)})")
    print(f"  test:  {test_path}  ({len(test)})")
    print(f"  stats: {stats_path}")
    print(f"  duplicate_removed: {duplicate_count}")
    print(f"  invalid_removed: {invalid_count}")

    return stats


def run_plan_120k(args: argparse.Namespace) -> None:
    client = make_client_and_key(args)
    root_dir = Path(args.root_dir)
    raw_dir = root_dir / "raw"
    final_dir = root_dir / "final"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("[RUN 120K PLAN]")
    print(f"root_dir: {root_dir}")
    print(f"raw_dir: {raw_dir}")
    print(f"final_dir: {final_dir}")
    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"concurrency: {args.concurrency}")
    print("plan:")
    for p in PLAN_120K:
        print(f"  - {p['task_type']}: {p['target']} -> {raw_dir / p['filename']}")
    print("#" * 80)

    job_stats = []
    if not args.merge_only:
        for plan in PLAN_120K:
            stat = generate_dataset_job(
                client=client,
                args=args,
                task_type=plan["task_type"],
                target_total=int(plan["target"]),
                out_path=raw_dir / plan["filename"],
            )
            job_stats.append(stat)

    stats = merge_split_final_dataset(
        raw_dir=raw_dir,
        final_dir=final_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    stats["job_stats"] = job_stats
    stats_path = final_dir / FINAL_STATS_NAME
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def run_plan_struct_format(args: argparse.Namespace) -> None:
    """一键生成结构化格式强化数据集。"""
    client = make_client_and_key(args)
    root_dir = Path(args.root_dir)
    raw_dir = root_dir / "struct_raw"
    final_dir = root_dir / "struct_final"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("[RUN STRUCT FORMAT PLAN]")
    print(f"root_dir: {root_dir}")
    print(f"raw_dir: {raw_dir}")
    print(f"final_dir: {final_dir}")
    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"concurrency: {args.concurrency}")
    print("plan:")
    for p in PLAN_STRUCT_FORMAT_30K:
        print(f"  - {p['task_type']}: {p['target']} -> {raw_dir / p['filename']}")
    print("#" * 80)

    job_stats = []
    if not args.merge_only:
        for plan_item in PLAN_STRUCT_FORMAT_30K:
            stat = generate_dataset_job(
                client=client,
                args=args,
                task_type=plan_item["task_type"],
                target_total=int(plan_item["target"]),
                out_path=raw_dir / plan_item["filename"],
            )
            job_stats.append(stat)

    stats = merge_split_final_dataset(
        raw_dir=raw_dir,
        final_dir=final_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        plan=PLAN_STRUCT_FORMAT_30K,
        train_name=STRUCT_FINAL_TRAIN_NAME,
        val_name=STRUCT_FINAL_VAL_NAME,
        test_name=STRUCT_FINAL_TEST_NAME,
        stats_name=STRUCT_FINAL_STATS_NAME,
    )
    stats["job_stats"] = job_stats
    stats_path = final_dir / STRUCT_FINAL_STATS_NAME
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def run_single_generation(args: argparse.Namespace) -> None:
    client = make_client_and_key(args)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"未知 task_type: {unknown}. 可选：{ALL_TASKS}")
    weights = parse_weights(tasks, args.weights)

    out_path = Path(args.out)
    fail_path = out_path.with_suffix(out_path.suffix + ".failed.jsonl")
    existing_hashes, existing_count = load_existing_hashes(out_path)

    print(f"[INFO] output: {out_path}")
    print(f"[INFO] failed log: {fail_path}")
    print(f"[INFO] existing samples: {existing_count}")
    print(f"[INFO] target new valid samples: {args.target}")
    print(f"[INFO] tasks: {tasks}")
    print(f"[INFO] model: {args.model}, base_url: {args.base_url}, thinking: {args.thinking}")

    max_api_calls = args.max_api_calls
    if max_api_calls <= 0:
        max_api_calls = max(10, int(args.target / max(args.batch_size, 1) * 4) + 10)

    accepted_total = 0
    rejected_total = 0
    api_calls = 0
    next_idx = existing_count

    iterator = range(max_api_calls)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Generating", unit="call")

    for batch_index in iterator:
        if accepted_total >= args.target:
            break

        task_type = choose_task(tasks, weights)
        api_calls += 1

        try:
            raw_samples = call_deepseek(
                client=client,
                model=args.model,
                task_type=task_type,
                batch_size=args.batch_size,
                batch_index=batch_index,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking=args.thinking,
                max_retries=args.max_retries,
            )
        except FatalAPIError as exc:
            append_jsonl(fail_path, [{"batch_index": batch_index, "task_type": task_type, "fatal_error": str(exc)}])
            print(f"[FATAL] {exc}", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001
            append_jsonl(fail_path, [{"batch_index": batch_index, "task_type": task_type, "error": str(exc)}])
            continue

        accepted: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        for raw in raw_samples:
            item = normalize_sample(
                raw,
                task_type=task_type,
                idx=next_idx,
                source_model=args.model,
                source_batch=batch_index,
            )
            ok, reason = validate_sample(item)
            key = sample_key(item)
            if not ok:
                item["reject_reason"] = reason
                failed.append(item)
                rejected_total += 1
                continue
            if key in existing_hashes:
                item["reject_reason"] = "duplicate"
                failed.append(item)
                rejected_total += 1
                continue

            existing_hashes.add(key)
            accepted.append(item)
            next_idx += 1

        if accepted:
            append_jsonl(out_path, accepted)
            accepted_total += len(accepted)
        if failed:
            append_jsonl(fail_path, failed)

        print(
            f"[BATCH {batch_index}] task={task_type} raw={len(raw_samples)} "
            f"accepted={len(accepted)} failed={len(failed)} total={accepted_total}/{args.target}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\n[DONE]")
    print(f"api_calls: {api_calls}")
    print(f"accepted_new: {accepted_total}")
    print(f"rejected: {rejected_total}")
    print(f"saved_to: {out_path}")
    print(f"failed_log: {fail_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Use DeepSeek API to generate SFT JSONL data.")
    parser.add_argument("--out", type=str, default="data/sft/ai_generated_sft.jsonl", help="单文件模式输出 JSONL 路径")
    parser.add_argument("--target", type=int, default=1000, help="单文件模式目标有效样本数")
    parser.add_argument("--batch-size", type=int, default=20, help="每次 API 请求生成多少条")
    parser.add_argument("--concurrency", type=int, default=1, help="并发 API 请求数量；建议 4-8，遇到 429 就降低")
    parser.add_argument("--max-api-calls", type=int, default=0, help="最大 API 调用次数，0 表示自动估算")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="DeepSeek 模型名")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible base_url")
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY", help="API Key 环境变量名")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS), help="单文件模式：逗号分隔任务类型")
    parser.add_argument("--weights", type=str, default="", help="单文件模式：逗号分隔采样权重，数量需与 --tasks 一致")
    parser.add_argument("--max-tokens", type=int, default=8192, help="每次 API 最大输出 token")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成温度，thinking=disabled 时生效")
    parser.add_argument("--thinking", type=str, default="disabled", choices=["enabled", "disabled"], help="DeepSeek thinking 模式")
    parser.add_argument("--max-retries", type=int, default=5, help="API 失败重试次数")
    parser.add_argument("--sleep", type=float, default=0.2, help="每次成功 API 调用后的暂停秒数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 新增：一键生成 12 万条并自动整理目录。
    parser.add_argument("--run-120k", action="store_true", help="按内置计划一次性生成 12 万条，并整理为 raw/final 目录")
    parser.add_argument("--run-struct-format", action="store_true", help="生成结构化格式强化数据集：JSON/Markdown/YAML/格式修复 + 少量通用回放")
    parser.add_argument("--merge-only", action="store_true", help="只合并已有 raw 数据并划分 train/val/test，不再调用 API")
    parser.add_argument("--root-dir", type=str, default="data/sft", help="一键模式根目录，默认 data/sft")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="最终 train 划分比例")
    parser.add_argument("--val-ratio", type=float, default=0.03, help="最终 val 划分比例")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="最终 test 划分比例")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.run_struct_format:
        run_plan_struct_format(args)
    elif args.run_120k or args.merge_only:
        run_plan_120k(args)
    else:
        run_single_generation(args)


if __name__ == "__main__":
    main()
