"""Deterministic per-problem call budgets for the public submission path."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StageBudget:
    solve_tokens: int
    review_tokens: int
    repair_tokens: int
    review_min_remaining_seconds: int
    repair_min_remaining_seconds: int
    allow_review: bool
    allow_repair: bool
    require_independent_review: bool = False
    emergency_tokens: int = 2048
    max_calls: int = 3

    def trace_content(self) -> dict:
        return asdict(self)


def plan_stage_budget(
    spec,
    has_whole_tool_answer: bool,
    *,
    deep_reasoning: bool = False,
) -> StageBudget:
    """Plan at most two full solves and one bounded repair.

    Review admission is a correctness decision.  The caller may skip optional
    arbitration near the wall-clock budget, but it must never skip an
    emergency answer recovery merely because the first solve was slow.
    """
    if has_whole_tool_answer:
        return StageBudget(0, 0, 0, 0, 0, False, False, max_calls=0)

    high_risk = (
        spec.profile.difficulty == "hard"
        or deep_reasoning
        or spec.profile.problem_type in {"proof", "derivation", "explanation"}
        or spec.verification_required
        or spec.risk_score >= 2
    )
    if high_risk:
        # Recovery only has to turn an existing derivation into a concise,
        # gradable submission. Output wrappers never determine reasoning cost.
        return StageBudget(
            8192,
            8192,
            4096,
            0,
            45,
            True,
            True,
            True,
            emergency_tokens=2048,
            max_calls=4,
        )
    return StageBudget(
        4096,
        4096,
        2048,
        0,
        30,
        True,
        True,
        False,
        emergency_tokens=1024,
        max_calls=4,
    )
