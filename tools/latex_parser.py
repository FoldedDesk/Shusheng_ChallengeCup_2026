from __future__ import annotations

import re


def normalize_latex(text: str) -> str:
    """Apply only safe display normalization; this is not a full TeX parser."""
    value = text.strip()
    # A JSONL producer may accidentally encode ``\bar`` as the JSON escape
    # ``\b`` followed by ``ar``. Restore this unambiguous TeX command before
    # structural validation; other control characters remain invalid.
    value = value.replace("\x08ar", r"\bar")
    value = re.sub(r"^\$\$?\s*|\s*\$?\$$", "", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    return value
