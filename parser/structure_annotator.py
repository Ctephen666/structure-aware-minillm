"""Token-level structure annotation for JSON and Markdown fence states."""

from __future__ import annotations

from dataclasses import dataclass

from parser.structure_states import (
    JSON_ARRAY,
    JSON_ESCAPE,
    JSON_OBJECT,
    JSON_STRING,
    MAX_DEPTH,
    MD_INNER_FENCE,
    MD_OUTER_FENCE,
    MD_TEXT,
    PLAIN,
    clamp_depth,
)


BT = chr(96)
TD = "~"


@dataclass
class Fence:
    char: str
    length: int


class StructureAnnotator:
    """Annotate tokenizer tokens with depth/state before each token is emitted."""

    def __init__(self, max_depth: int = MAX_DEPTH) -> None:
        self.max_depth = max_depth
        self.reset()

    def reset(self) -> None:
        self.json_stack: list[str] = []
        self.json_depth = 0
        self.inside_json_string = False
        self.escape_pending = False

        self.markdown_fence_stack: list[Fence] = []
        self.line_start = True

    def annotate_tokens(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        self.reset()
        depth_ids: list[int] = []
        state_ids: list[int] = []

        for token in tokens:
            depth_ids.append(self._current_depth())
            state_ids.append(self._current_state(token))
            self._update_after_token(token)

        return depth_ids, state_ids

    def annotate_text(self, text: str, tokenizer) -> tuple[list[int], list[int]]:
        return self.annotate_tokens(tokenizer.tokenize(text))

    def _current_depth(self) -> int:
        return min(clamp_depth(self.json_depth + len(self.markdown_fence_stack)), self.max_depth)

    def _current_state(self, token: str = "") -> int:
        if self.escape_pending:
            return JSON_ESCAPE
        if self.inside_json_string and token in {'\\"', "\\\\", "\\n"}:
            return JSON_ESCAPE
        if self.inside_json_string:
            return JSON_STRING
        if self.json_stack:
            if self.json_stack[-1] == "object":
                return JSON_OBJECT
            return JSON_ARRAY
        if len(self.markdown_fence_stack) >= 2:
            return MD_INNER_FENCE
        if len(self.markdown_fence_stack) == 1:
            return MD_OUTER_FENCE
        return MD_TEXT if not self.line_start else PLAIN

    def _update_after_token(self, token: str) -> None:
        self._update_markdown_state(token)
        self._update_json_state(token)
        self._update_line_position(token)

    def _update_json_state(self, token: str) -> None:
        if self.inside_json_string:
            if self.escape_pending:
                self.escape_pending = False
                return
            if token == "\\":
                self.escape_pending = True
                return
            if token in {'\\"', "\\n"}:
                return
            if token == "\\\\":
                return
            if token == '"':
                self.inside_json_string = False
            return

        if token == '"':
            self.inside_json_string = True
        elif token == "{":
            self.json_stack.append("object")
            self.json_depth += 1
        elif token == "[":
            self.json_stack.append("array")
            self.json_depth += 1
        elif token in {"}", "]"}:
            if self.json_stack:
                self.json_stack.pop()
            self.json_depth = max(0, self.json_depth - 1)

    def _update_markdown_state(self, token: str) -> None:
        if not self.line_start:
            return

        fence = self._detect_fence_token(token)
        if fence is None:
            return

        char, length, suffix = fence
        if self.markdown_fence_stack:
            top = self.markdown_fence_stack[-1]
            if char == top.char and length >= top.length and suffix == "":
                self.markdown_fence_stack.pop()
                return

        self.markdown_fence_stack.append(Fence(char=char, length=length))

    def _detect_fence_token(self, token: str) -> tuple[str, int, str] | None:
        stripped = token.strip()
        if not stripped:
            return None

        char = stripped[0]
        if char not in (BT, TD):
            return None

        length = 0
        for ch in stripped:
            if ch == char:
                length += 1
            else:
                break

        if length < 3:
            return None

        suffix = stripped[length:].strip()
        return char, length, suffix

    def _update_line_position(self, token: str) -> None:
        if "\n" in token:
            tail = token.rsplit("\n", 1)[1]
            self.line_start = tail.strip() == ""
            return
        if token.strip():
            self.line_start = False
