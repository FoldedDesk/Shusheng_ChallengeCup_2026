"""Shared execution limits for the local evaluation runner.

A hard item has six remote-model stages: decomposition, three solver
candidates, one critic, and one verifier. Only solver candidates have a
single recovery retry, so its real worst case is nine request windows.
"""

from __future__ import annotations

from math import ceil


MAX_CONCURRENCY = 3
REQUEST_TIMEOUT_SECONDS = 60
# Decomposer, critic and verifier make one request. Each of the three solver
# candidates may make one recovery request after malformed model output.
HARD_REQUEST_WINDOWS = 9
HARD_MODEL_STAGES = 6
PER_ITEM_TIMEOUT_SECONDS = 20 * 60
OVERALL_TIMEOUT_SECONDS = 360 * 60


def hard_item_worst_case_seconds() -> int:
    return HARD_REQUEST_WINDOWS * REQUEST_TIMEOUT_SECONDS


def dataset_worst_case_seconds(item_count: int) -> int:
    return ceil(item_count / MAX_CONCURRENCY) * hard_item_worst_case_seconds()


assert hard_item_worst_case_seconds() <= PER_ITEM_TIMEOUT_SECONDS
assert dataset_worst_case_seconds(112) <= OVERALL_TIMEOUT_SECONDS
