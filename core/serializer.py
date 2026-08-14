from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from core.runtime_failure import is_recoverable_runtime_failure


_RECURSIVE_REFERENCE = "<recursive reference>"


def _safe_text(value: Any) -> str:
    try:
        return str(value)
    except BaseException as error:
        if not is_recoverable_runtime_failure(error):
            raise
        return f"<unserializable {type(value).__name__}>"


def safe_json(value: Any, _active: set[int] | None = None) -> Any:
    """Convert common scientific and diagnostic values to JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, Path, Exception)):
        return _safe_text(value)

    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        return _RECURSIVE_REFERENCE

    active.add(identity)
    try:
        if isinstance(value, dict):
            try:
                items = value.items()
            except BaseException as error:
                if not is_recoverable_runtime_failure(error):
                    raise
                return _safe_text(value)
            return {
                _safe_text(key): safe_json(item, active)
                for key, item in items
            }
        if isinstance(value, (list, tuple, set)):
            return [safe_json(item, active) for item in value]

        try:
            tolist = getattr(value, "tolist", None)
        except BaseException as error:
            if not is_recoverable_runtime_failure(error):
                raise
            return _safe_text(value)
        if callable(tolist):
            try:
                converted = tolist()
            except BaseException as error:
                if not is_recoverable_runtime_failure(error):
                    raise
                return _safe_text(value)
            return safe_json(converted, active)
        return _safe_text(value)
    finally:
        active.discard(identity)
