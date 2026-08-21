from __future__ import annotations

import re


def find_matching_brace(text: str, open_pos: int) -> int:
    """Return the matching unescaped closing brace, or -1 if malformed."""
    if not (0 <= open_pos < len(text)) or text[open_pos] != "{":
        return -1
    depth = 0
    for index in range(open_pos, len(text)):
        if text[index] not in "{}":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        if text[index] == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return -1
    return -1


def normalize_latex(text: str) -> str:
    """Apply only safe display normalization; this is not a full TeX parser."""
    value = text.strip()
    # A JSONL producer may accidentally encode ``\bar`` as the JSON escape
    # ``\b`` followed by ``ar``. Restore this unambiguous TeX command before
    # structural validation; other control characters remain invalid.
    value = value.replace("\x08ar", r"\bar")
    # Strip math delimiters only when they are the sole unescaped delimiters
    # and genuinely wrap the entire value. Removing the last ``$`` from prose
    # ending in inline math would corrupt an otherwise complete answer.
    dollars = [
        index
        for index, char in enumerate(value)
        if char == "$" and (index == 0 or value[index - 1] != "\\")
    ]
    if dollars == [0, len(value) - 1]:
        value = value[1:-1].strip()
    elif dollars == [0, 1, len(value) - 2, len(value) - 1]:
        value = value[2:-2].strip()
    value = re.sub(r"\\left(?=\s*[^A-Za-z])", "", value)
    value = re.sub(r"\\right(?=\s*[^A-Za-z])", "", value)
    return value
