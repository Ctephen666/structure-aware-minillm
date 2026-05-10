"""HuggingFace tokenizer adapter used by pretraining scripts."""

from __future__ import annotations

from pathlib import Path


class HFTokenizer:
    """Small adapter exposing the tokenizer interface used in this project."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.token_to_id = tokenizer.get_vocab()
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, **kwargs) -> "HFTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Please install transformers to use HFTokenizer: pip install transformers") from exc

        tokenizer = AutoTokenizer.from_pretrained(str(name_or_path), **kwargs)
        tokenizer.model_max_length = 10**12
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id or 0)

    @property
    def unk_id(self) -> int:
        if self.tokenizer.unk_token_id is not None:
            return int(self.tokenizer.unk_token_id)
        return self.pad_id

    @property
    def bos_id(self) -> int:
        for value in (self.tokenizer.bos_token_id, self.tokenizer.cls_token_id):
            if value is not None:
                return int(value)
        return self.eos_id

    @property
    def eos_id(self) -> int:
        for value in (self.tokenizer.eos_token_id, self.tokenizer.sep_token_id):
            if value is not None:
                return int(value)
        return self.pad_id

    @property
    def unk_token(self) -> str:
        return self.tokenizer.unk_token or "<unk>"

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def token_to_id_value(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return self.unk_id
        return int(token_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [int(idx) for idx in self.tokenizer.encode(text, add_special_tokens=False)]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
