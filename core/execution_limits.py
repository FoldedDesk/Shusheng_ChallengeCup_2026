"""Shared execution limits for the bounded score-first evaluation runner."""

from __future__ import annotations

from math import ceil


MAX_CONCURRENCY = 3
REQUEST_TIMEOUT_SECONDS = 120
# Two independent deep solves may each need one bounded answer completion.
# Provider retries are owned by the client.
HARD_REQUEST_WINDOWS = 4
HARD_MODEL_STAGES = 4
PER_ITEM_TIMEOUT_SECONDS = 20 * 60
OVERALL_TIMEOUT_SECONDS = 360 * 60


def hard_item_worst_case_seconds() -> int:
    return HARD_REQUEST_WINDOWS * REQUEST_TIMEOUT_SECONDS


def dataset_worst_case_seconds(item_count: int) -> int:
    return ceil(item_count / MAX_CONCURRENCY) * hard_item_worst_case_seconds()


assert hard_item_worst_case_seconds() <= PER_ITEM_TIMEOUT_SECONDS
assert dataset_worst_case_seconds(112) <= OVERALL_TIMEOUT_SECONDS
