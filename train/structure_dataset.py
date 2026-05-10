"""Structure-aware dataset for masked SFT and pretraining samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parser.structure_annotator import StructureAnnotator
from parser.structure_states import PLAIN


IGNORE_INDEX = -100


def tokens_to_ids(tokens: list[str], tokenizer) -> list[int]:
    if hasattr(tokenizer, "token_to_id_value"):
        return [int(tokenizer.token_to_id_value(token)) for token in tokens]
    return [int(tokenizer.token_to_id.get(token, tokenizer.unk_id)) for token in tokens]


def build_instruction_parts(prompt: str, answer: str) -> tuple[str, str]:
    prompt_part = "### Instruction:\n" + prompt.strip() + "\n\n### Response:\n"
    answer_part = answer.strip() + "\n"
    return prompt_part, answer_part


def load_structure_rows(path: str | Path) -> list[dict[str, Any] | str]:
    rows: list[dict[str, Any] | str] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rows.append(line)
                continue
            rows.append(item if isinstance(item, dict) else str(item))
    return rows


def row_to_training_text(row: dict[str, Any] | str) -> str:
    if isinstance(row, str):
        return row
    prompt = row.get("prompt")
    answer = row.get("answer")
    if isinstance(prompt, str) and isinstance(answer, str):
        prompt_part, answer_part = build_instruction_parts(prompt, answer)
        return prompt_part + answer_part
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("content"), str):
        return row["content"]
    return ""


def _annotate_with_special_tokens(text: str, tokenizer, annotator: StructureAnnotator) -> tuple[list[int], list[int], list[int]]:
    tokens = tokenizer.tokenize(text)
    token_ids = [tokenizer.bos_id] + tokens_to_ids(tokens, tokenizer) + [tokenizer.eos_id]
    depth_ids, state_ids = annotator.annotate_tokens(tokens)
    depth_ids = [0] + depth_ids + [0]
    state_ids = [PLAIN] + state_ids + [PLAIN]
    return token_ids, depth_ids, state_ids


class StructureLanguageModelingDataset(Dataset):
    """Fixed-length chunks with token, depth, and state targets."""

    def __init__(self, source: str | Path | list[dict[str, Any] | str], tokenizer, block_size: int) -> None:
        self.block_size = block_size
        rows = load_structure_rows(source) if isinstance(source, (str, Path)) else source

        annotator = StructureAnnotator()
        input_ids: list[int] = []
        labels: list[int] = []
        depth_ids: list[int] = []
        state_ids: list[int] = []
        depth_targets: list[int] = []
        state_targets: list[int] = []

        for row in rows:
            encoded = self._encode_row(row, tokenizer, annotator)
            if encoded is None:
                continue
            row_input_ids, row_labels, row_depth_ids, row_state_ids, row_depth_targets, row_state_targets = encoded
            input_ids.extend(row_input_ids)
            labels.extend(row_labels)
            depth_ids.extend(row_depth_ids)
            state_ids.extend(row_state_ids)
            depth_targets.extend(row_depth_targets)
            state_targets.extend(row_state_targets)

        if len(input_ids) < block_size:
            raise ValueError(
                f"Dataset has {len(input_ids)} shifted tokens, but block_size={block_size} needs at least {block_size}."
            )

        lengths = {len(input_ids), len(labels), len(depth_ids), len(state_ids), len(depth_targets), len(state_targets)}
        if len(lengths) != 1:
            raise ValueError(f"Structure dataset alignment error: lengths={sorted(lengths)}")

        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.depth_ids = torch.tensor(depth_ids, dtype=torch.long)
        self.state_ids = torch.tensor(state_ids, dtype=torch.long)
        self.depth_targets = torch.tensor(depth_targets, dtype=torch.long)
        self.state_targets = torch.tensor(state_targets, dtype=torch.long)

    def _encode_row(
        self,
        row: dict[str, Any] | str,
        tokenizer,
        annotator: StructureAnnotator,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]] | None:
        if isinstance(row, dict) and isinstance(row.get("prompt"), str) and isinstance(row.get("answer"), str):
            prompt_part, answer_part = build_instruction_parts(row["prompt"], row["answer"])
            full_text = prompt_part + answer_part
            prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_part, add_special_tokens=False)
            token_ids, depth_full, state_full = _annotate_with_special_tokens(full_text, tokenizer, annotator)

            expected_len = 1 + len(prompt_ids) + len(answer_ids) + 1
            if len(token_ids) != expected_len:
                raise ValueError(
                    f"SFT tokenization alignment failed: full={len(token_ids)}, parts={expected_len}."
                )

            label_full = [IGNORE_INDEX] * (1 + len(prompt_ids)) + answer_ids + [tokenizer.eos_id]
            labels = label_full[1:]
        else:
            text = row_to_training_text(row)
            if not text.strip():
                return None
            token_ids, depth_full, state_full = _annotate_with_special_tokens(text, tokenizer, annotator)
            labels = token_ids[1:]

        input_ids = token_ids[:-1]
        depth_ids = depth_full[:-1]
        state_ids = state_full[:-1]
        depth_targets = depth_full[1:]
        state_targets = state_full[1:]

        for i, label in enumerate(labels):
            if label == IGNORE_INDEX:
                depth_targets[i] = IGNORE_INDEX
                state_targets[i] = IGNORE_INDEX

        lengths = {len(input_ids), len(labels), len(depth_ids), len(state_ids), len(depth_targets), len(state_targets)}
        if len(lengths) != 1:
            raise ValueError(f"Encoded sample alignment error: lengths={sorted(lengths)}")

        return input_ids, labels, depth_ids, state_ids, depth_targets, state_targets

    def __len__(self) -> int:
        return len(self.input_ids) - self.block_size + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = slice(index, index + self.block_size)
        return (
            self.input_ids[item],
            self.labels[item],
            self.depth_ids[item],
            self.state_ids[item],
            self.depth_targets[item],
            self.state_targets[item],
        )
