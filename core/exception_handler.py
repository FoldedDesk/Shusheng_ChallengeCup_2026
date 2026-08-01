from __future__ import annotations

from typing import Callable, Optional, TypeVar


T = TypeVar("T")


class IncompleteModelResponseError(RuntimeError):
    """The model returned text, but not the response contract it was given."""


def retry_once(
    operation: Callable[[], T],
    on_error: Optional[Callable[[Exception], None]] = None,
    attempts: int = 2,
) -> Optional[T]:
    """One retry keeps transient model failures local to the current problem."""
    for _ in range(max(1, attempts)):
        try:
            return operation()
        except Exception as exc:
            if on_error:
                on_error(exc)
            continue
    return None
