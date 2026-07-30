from __future__ import annotations

from typing import Callable, Optional, TypeVar


T = TypeVar("T")


def retry_once(operation: Callable[[], T]) -> Optional[T]:
    """One retry keeps transient model failures local to the current problem."""
    for _ in range(2):
        try:
            return operation()
        except Exception:
            continue
    return None
