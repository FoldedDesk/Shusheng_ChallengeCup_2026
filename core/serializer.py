from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any


def safe_json(value: Any) -> Any:
    """Convert common scientific and diagnostic values to JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, Path, Exception)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(item) for item in value]
    if hasattr(value, "tolist"):
        return safe_json(value.tolist())
    return str(value)
