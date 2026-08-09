from __future__ import annotations

import re


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
