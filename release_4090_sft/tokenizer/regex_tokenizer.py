"""A structure-friendly regex tokenizer for the mini language model.

This tokenizer preserves important Markdown and JSON control symbols as atomic
units where possible, such as code fences and escaped quotes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


BT = chr(96)
BT3 = BT * 3
BT4 = BT * 4
TD3 = "~" * 3

# Order matters: longer / structural tokens must appear before generic fallback.
TOKEN_PATTERN = re.compile(
    r"`{3,}[A-Za-z0-9_-]*"          # Markdown backtick fences, e.g. BT3json, BT4markdown
    r"|~{3,}[A-Za-z0-9_-]*"         # Markdown tilde fences
    r"|\\\""                       # escaped double quote
    r"|\\\\"                       # escaped backslash
    r"|\\n"                         # literal backslash+n sequence
    r"|\r\n|\n|\t"                  # real newlines / tabs
    r"|[ \f\v]+"                    # spaces, excluding newline and tab
    r"|[A-Za-z_][A-Za-z_0-9]*"      # identifiers / English words
    r"|\d+(?:\.\d+)?"              # numbers
    r"|[\u4e00-\u9fff]"             # CJK chars, one char per token for MVP
    r"|[^\sA-Za-z_0-9]",            # punctuation fallback
    re.UNICODE,
)


DEFAULT_EXTRA_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    BT3, BT4, BT3 + "markdown", BT4 + "markdown", BT3 + "json", BT3 + "kotlin", BT3 + "python",
    TD3, TD3 + "markdown", TD3 + "json",
    "\\\"", "\\\\", "\\n",
    "{", "}", "[", "]", ":", ",", '"', "`", "~",
]


class RegexTokenizer:
    """Regex tokenizer with a learned token-to-id vocabulary."""

    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            token_to_id = {}
            for token in DEFAULT_EXTRA_TOKENS[:4]:
                token_to_id[token] = len(token_to_id)
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        special = {self.pad_token, self.bos_token, self.eos_token, self.pad_token}
        tokens = [self.id_to_token.get(int(idx), self.unk_token) for idx in ids]
        return "".join(token for token in tokens if token not in special)

    @classmethod
    def train_from_texts(
        cls,
        texts: list[str],
        vocab_size: int,
        min_freq: int = 1,
    ) -> "RegexTokenizer":
        tokenizer = cls()

        # Insert structural tokens first so they always exist in the vocabulary.
        for token in DEFAULT_EXTRA_TOKENS:
            if token not in tokenizer.token_to_id:
                tokenizer.token_to_id[token] = len(tokenizer.token_to_id)

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenizer.tokenize(text))

        for token, freq in counter.most_common():
            if len(tokenizer.token_to_id) >= vocab_size:
                break
            if freq < min_freq:
                continue
            if token not in tokenizer.token_to_id:
                tokenizer.token_to_id[token] = len(tokenizer.token_to_id)

        tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegexTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=data["token_to_id"])
