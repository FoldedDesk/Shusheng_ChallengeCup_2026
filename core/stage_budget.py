"""Adaptive, bounded model-call budgets for one independent problem."""

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
    """Allocate one solve, an accuracy review, and at most one repair.

    The third call is capacity, not an automatic stage.  The orchestrator may
    spend it only on a real disagreement or an unusable/truncated response.
    """
    if has_whole_tool_answer:
        return StageBudget(0, 0, 0, 0, 0, False, False, max_calls=0)

    task = getattr(spec.profile, "task_kind", spec.profile.problem_type)
    shape = spec.profile.answer_shape
    high_risk = bool(
        spec.profile.difficulty == "hard"
        or deep_reasoning
        or task in {"proof", "derivation", "explanation", "construction"}
        or shape in {"roots", "choice"}
        or spec.verification_required
        or spec.risk_score >= 3
    )
    medium_risk = bool(
        spec.profile.difficulty == "medium"
        and (
            spec.answer_contract.mode != "answer_only"
            or spec.profile.subject_confidence == "low"
            or spec.risk_score >= 2
            or shape in {"truth", "probability", "count", "interval"}
        )
    )
    if high_risk:
        return StageBudget(
            solve_tokens=8192,
            review_tokens=8192,
            repair_tokens=4096,
            review_min_remaining_seconds=75,
            repair_min_remaining_seconds=30,
            allow_review=True,
            allow_repair=True,
            require_independent_review=True,
            emergency_tokens=3072,
        )
    if medium_risk:
        return StageBudget(
            solve_tokens=6144,
            review_tokens=6144,
            repair_tokens=3072,
            review_min_remaining_seconds=75,
            repair_min_remaining_seconds=30,
            allow_review=True,
            allow_repair=True,
            require_independent_review=True,
            emergency_tokens=3072,
        )
    return StageBudget(
        solve_tokens=4096,
        review_tokens=4096,
        repair_tokens=2048,
        review_min_remaining_seconds=75,
        repair_min_remaining_seconds=30,
        allow_review=True,
        allow_repair=True,
        require_independent_review=False,
        emergency_tokens=1536,
    )
