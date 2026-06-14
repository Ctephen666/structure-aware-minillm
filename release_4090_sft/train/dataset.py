"""Dataset loading utilities for autoregressive language modeling.

Root fix:
- Do NOT write literal "<bos>" or "<eos>" into training text.
- Let tokenizer.encode(..., add_special_tokens=True) add real BOS/EOS token ids.
- Support both:
  1. SFT-style JSONL: {"prompt": "...", "answer": "..."}
  2. Pretrain-style JSONL: {"text": "..."} or {"content": "..."}
  3. Plain text lines
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def format_prompt_answer(prompt: str, answer: str) -> str:
    """Format one instruction sample without literal special-token strings."""
    return (
        "### Instruction:\n"
        f"{prompt.strip()}\n\n"
        "### Response:\n"
        f"{answer.strip()}\n"
    )


def read_text_samples(path: str | Path) -> list[str]:
    """Read JSONL samples and convert them into training strings.

    Supported JSONL formats:
    - {"prompt": "...", "answer": "..."}  -> instruction tuning text
    - {"text": "..."}                     -> pretraining text
    - {"content": "..."}                  -> pretraining text
    - plain text lines                    -> pretraining text

    Important:
    This function intentionally does NOT insert literal "<bos>" or "<eos>".
    Real BOS/EOS ids are added later by tokenizer.encode(add_special_tokens=True).
    """
    path = Path(path)
    samples: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                samples.append(line)
                continue

            if isinstance(item, dict):
                prompt = item.get("prompt")
                answer = item.get("answer")

                if isinstance(prompt, str) and isinstance(answer, str):
                    samples.append(format_prompt_answer(prompt, answer))
                elif isinstance(item.get("text"), str):
                    samples.append(item["text"])
                elif isinstance(item.get("content"), str):
                    samples.append(item["content"])
            else:
                samples.append(str(item))

    return [sample for sample in samples if sample and sample.strip()]


class LanguageModelingDataset(Dataset):
    """Fixed-length next-token prediction dataset.

    This is a simple concatenated LM dataset:
    - Every sample is encoded with real BOS/EOS token ids.
    - No literal "<bos>" or "<eos>" text is inserted.
    - Loss is computed on all tokens.

    For the later true answer-mask SFT stage, you can add a separate
    MaskedInstructionDataset. Keep this class simple for baseline/pretrain.
    """

    def __init__(self, texts: list[str], tokenizer, block_size: int) -> None:
        self.block_size = block_size

        token_ids: list[int] = []
        for text in texts:
            token_ids.extend(tokenizer.encode(text, add_special_tokens=True))

        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Dataset has {len(token_ids)} tokens, but block_size={block_size} needs at least {block_size + 1}."
            )

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


StructureDataset = LanguageModelingDataset
