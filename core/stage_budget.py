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
    plan_tokens: int = 1536
    allow_plan: bool = False

    def trace_content(self) -> dict:
        return asdict(self)


def plan_stage_budget(
    spec,
    has_whole_tool_answer: bool,
    *,
    deep_reasoning: bool = False,
) -> StageBudget:
    """Allocate a deep solve, an independent check, and bounded recovery.

    The second and third calls are capacity, not automatic stages.  Deep
    problems spend their largest budget on one coherent derivation; remaining
    calls recover its conclusion or perform one focused mathematical check.
    """
    if has_whole_tool_answer:
        return StageBudget(
            0, 0, 0, 0, 0, False, False,
            max_calls=0, plan_tokens=0, allow_plan=False,
        )

    task = getattr(spec.profile, "task_kind", spec.profile.problem_type)
    shape = spec.profile.answer_shape
    quick_choice = bool(
        (shape in {"choice", "truth"} and not deep_reasoning)
        or (
            (shape == "text" or task == "fill_blank")
            and spec.profile.difficulty != "hard"
            and spec.risk_score <= 2
        )
    )
    if quick_choice:
        # Definition and concept questions benefit from two fresh, explicit
        # option audits.  A single 16K thinking call commonly overthinks them
        # and then exhausts the response before emitting the labels.
        return StageBudget(
            solve_tokens=4096,
            review_tokens=4096,
            repair_tokens=2048,
            review_min_remaining_seconds=90,
            repair_min_remaining_seconds=30,
            allow_review=True,
            allow_repair=True,
            require_independent_review=True,
            emergency_tokens=1536,
            max_calls=3,
            plan_tokens=0,
            allow_plan=False,
        )

    high_risk = bool(
        spec.profile.difficulty == "hard"
        or deep_reasoning
        or task in {"proof", "derivation", "explanation", "construction"}
        or shape == "roots"
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
        strict_requirements = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
            if requirement.strict
        }
        proof_bearing = bool(
            spec.answer_contract.mode != "answer_only"
            or strict_requirements.intersection({
                "all_solutions", "construction_object", "construction_check"
            })
        )
        solve_tokens = (
            8192
            if proof_bearing
            or spec.profile.difficulty == "hard"
            or shape == "count"
            else 6144
        )
        return StageBudget(
            # Keep ordinary complex work within one focused solver window.
            # Longer proof recovery is incremental; blindly raising every
            # hard item to 16K increased both truncation and answer regressions
            # in the controlled replay.
            solve_tokens=solve_tokens,
            review_tokens=8192,
            repair_tokens=4096 if proof_bearing else 3072,
            review_min_remaining_seconds=120,
            repair_min_remaining_seconds=45,
            allow_review=True,
            allow_repair=True,
            require_independent_review=True,
            emergency_tokens=3072 if proof_bearing else 2048,
            # Four 120-second windows across three platform slots keep the
            # 112-item worst case below the six-hour global limit. The fourth
            # call is recovery capacity, not an automatic stage.
            max_calls=4,
            plan_tokens=2048,
            allow_plan=True,
        )
    if medium_risk:
        return StageBudget(
            solve_tokens=6144,
            review_tokens=4096,
            repair_tokens=3072,
            review_min_remaining_seconds=120,
            repair_min_remaining_seconds=45,
            allow_review=True,
            allow_repair=True,
            require_independent_review=True,
            emergency_tokens=3072,
            # A model-only Critic correction is diagnostic, not a certificate;
            # keep medium-risk work to solve, recovery, and one audit.
            max_calls=3,
            plan_tokens=1536,
        )
    return StageBudget(
        solve_tokens=6144,
        review_tokens=4096,
        repair_tokens=3072,
        review_min_remaining_seconds=120,
        repair_min_remaining_seconds=45,
        allow_review=True,
        allow_repair=True,
        require_independent_review=False,
        emergency_tokens=1536,
        max_calls=3,
        plan_tokens=1024,
    )
