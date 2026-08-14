"""Classify failures that the public entry point may safely degrade."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FuturesCancelledError


_FATAL_FAILURES = (KeyboardInterrupt, SystemExit, GeneratorExit)
_CANCELLATION_NAMES = frozenset({"CancelledError", "CanceledError"})
try:
    _BASE_EXCEPTION_GROUPS = (BaseExceptionGroup,)
except NameError:  # Python < 3.11 has no exception groups.
    _BASE_EXCEPTION_GROUPS = ()


def is_recoverable_runtime_failure(error: BaseException) -> bool:
    """Return whether one item may degrade instead of aborting evaluation.

    ``func_timeout.FunctionTimedOut`` deliberately derives from
    ``BaseException``.  Detect it by MRO name so the entry point remains
    compatible with both the real package and platform-provided equivalents,
    without adding a runtime dependency on ``func_timeout``.
    """

    if isinstance(error, _FATAL_FAILURES):
        return False
    if isinstance(error, (asyncio.CancelledError, FuturesCancelledError)):
        return False
    if isinstance(error, MemoryError):
        return False

    if _BASE_EXCEPTION_GROUPS and isinstance(error, _BASE_EXCEPTION_GROUPS):
        return all(
            is_recoverable_runtime_failure(child)
            for child in error.exceptions
        )

    hierarchy_names = {base.__name__ for base in type(error).__mro__}
    if hierarchy_names & _CANCELLATION_NAMES:
        return False
    if "FunctionTimedOut" in hierarchy_names:
        return True
    return isinstance(error, Exception)
