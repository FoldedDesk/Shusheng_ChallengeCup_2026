from __future__ import annotations

import re


def normalize_latex(text: str) -> str:
    """Apply only safe display normalization; this is not a full TeX parser."""
    value = text.strip()
    value = re.sub(r"^\$\$?\s*|\s*\$?\$$", "", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    return value
