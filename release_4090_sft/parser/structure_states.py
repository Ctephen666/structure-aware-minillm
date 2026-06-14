"""Structure state ids used by Struct-LM."""

PLAIN = 0
JSON_OBJECT = 1
JSON_ARRAY = 2
JSON_STRING = 3
JSON_ESCAPE = 4
MD_TEXT = 5
MD_OUTER_FENCE = 6
MD_INNER_FENCE = 7
MD_INLINE_CODE = 8

NUM_STATES = 9
MAX_DEPTH = 32


def clamp_depth(depth: int) -> int:
    return max(0, min(int(depth), MAX_DEPTH))

