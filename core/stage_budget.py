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

    def trace_content(self) -> dict:
        return asdict(self)


def plan_stage_budget(spec, has_whole_tool_answer: bool) -> StageBudget:
    """Reserve enough time for repair instead of spending it on every stage."""
    if has_whole_tool_answer:
        return StageBudget(0, 0, 0, 0, 0, False, False)
    if spec.profile.difficulty == "hard":
        return StageBudget(4096, 3072, 1024, 55, 22, True, True)
    if spec.profile.problem_type in {"proof", "derivation", "explanation"}:
        return StageBudget(3584, 2560, 896, 45, 20, True, True)
    return StageBudget(3072, 2048, 768, 45, 18, True, True)
