"""Deterministic candidate validation and selection for submitted answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from reasoning.finalizer import Finalizer

if TYPE_CHECKING:
    from classifier.problem_spec import ProblemSpec


@dataclass(frozen=True)
class ToolEvidence:
    result: str
    scope: str
    operation: str
    verified: bool


@dataclass(frozen=True)
class CandidateAssessment:
    answer: str
    source: str
    extraction_method: str
    score: int
    complete_goals: bool
    shape_valid: bool
    formatting_valid: bool
    tool_status: str
    goal_coverage: tuple[bool, ...]
    coverage_uncertain: bool
    rejected_reasons: tuple[str, ...]
    raw_has_meta: bool = False
    explicit_answer: bool = False

    @property
    def accepted(self) -> bool:
        return (
            self.formatting_valid
            and self.shape_valid
            and self.tool_status != "conflict"
            and "meta_without_explicit_answer" not in self.rejected_reasons
        )


def assess_candidate(
    answer: str,
    source: str,
    spec: "ProblemSpec",
    evidence: tuple[ToolEvidence, ...],
    extraction_method: str = "direct",
    extraction_reasons: tuple[str, ...] = (),
    raw_has_meta: bool = False,
    explicit_answer: bool = False,
) -> CandidateAssessment:
    value = str(answer or "").strip()
    formatting_reasons = Finalizer.validate_structure(value)
    coverage = tuple(_goal_covered(value, goal) for goal in spec.goals)
    complete = bool(coverage) and all(coverage)
    shape_valid = _valid_shape(value, spec.profile.answer_shape) and _frame_valid(value, spec)
    tool_status = _tool_status(value, evidence)
    reasons = list(extraction_reasons) + list(formatting_reasons)
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    score = (8 if complete else -8) + (4 if shape_valid else -4) + (4 if not formatting_reasons else -12)
    score += 4 if tool_status == "pass" else (-8 if tool_status == "conflict" else 0)
    return CandidateAssessment(
        value, source, extraction_method, score, complete, shape_valid,
        not formatting_reasons, tool_status, coverage, not complete, tuple(reasons), raw_has_meta, explicit_answer,
    )


def choose_candidate(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
    usable = [candidate for candidate in candidates if candidate.accepted]
    if not usable:
        return None
    # Completeness and validity dominate source preference; length is only a stable final tie-breaker.
    return max(
        usable,
        key=lambda item: (
            item.complete_goals,
            item.formatting_valid,
            item.tool_status == "pass",
            item.explicit_answer,
            item.source == "review",
            item.score,
            len(item.answer),
        ),
    )


def _goal_covered(answer: str, goal) -> bool:
    if not answer:
        return False
    compact = _compact(answer)
    if goal.requirements:
        return all(requirement.matches(compact) for requirement in goal.requirements)
    return all(_compact(term) in compact for term in goal.required_terms)


def _valid_shape(answer: str, shape: str) -> bool:
    if not answer:
        return False
    if shape == "roots" and re.fullmatch(r"\[\s*[^\[\]]+[,，][^\[\]]+\s*\]", answer):
        return False
    if shape == "proof":
        # A derivation can legitimately terminate in a one-symbol conclusion
        # after finalization has removed an unsafe scratchpad.
        return len(answer) >= 4 or bool(re.fullmatch(r"[A-Za-z0-9_+\-*/=^\\]+", answer))
    return not _is_scratchpad(answer)


def _tool_status(answer: str, evidence: tuple[ToolEvidence, ...]) -> str:
    whole = [item for item in evidence if item.scope == "whole_goal" and item.verified]
    if not whole:
        return "unknown"
    normalized = _compact(answer)
    for item in whole:
        expected = _compact(item.result)
        if expected and (expected in normalized or normalized in expected):
            return "pass"
    return "conflict"


def _is_scratchpad(answer: str) -> bool:
    return Finalizer.contains_meta(answer)


def _frame_valid(answer: str, spec: "ProblemSpec") -> bool:
    frame = spec.answer_frame
    if frame.style != "sentence":
        return True
    if frame.question_kind == "age":
        return bool(frame.subject and frame.subject in answer and frame.unit in answer)
    if frame.question_kind == "count":
        return frame.unit in answer
    if frame.question_kind == "probability":
        return "概率" in answer
    if frame.question_kind == "truth":
        judgement = re.search(r"(?:是|否|可以|不可以|正确|错误|成立|不成立)", answer)
        return bool(judgement and (not frame.subject or _compact(frame.subject) in _compact(answer)))
    return True


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")
