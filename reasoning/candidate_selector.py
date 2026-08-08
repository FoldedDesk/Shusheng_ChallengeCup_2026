"""Deterministic candidate validation and selection for submitted answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers

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
    verification_verdict: str = ""

    @property
    def accepted(self) -> bool:
        return (
            self.formatting_valid
            and self.shape_valid
            and self.tool_status != "conflict"
            and "meta_without_explicit_answer" not in self.rejected_reasons
            and "missing_required_goal" not in self.rejected_reasons
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
    verification_verdict: str = "",
) -> CandidateAssessment:
    value = str(answer or "").strip()
    formatting_reasons = Finalizer.validate_structure(value)
    coverage = tuple(_goal_covered(value, goal) for goal in spec.goals)
    complete = bool(coverage) and all(coverage)
    missing_strict = any(
        requirement.strict and not requirement.matches(value)
        for goal in spec.goals
        for requirement in goal.requirements
    )
    shape_valid = _valid_shape(value, spec.profile.answer_shape) and _frame_valid(value, spec)
    tool_status = _tool_status(value, evidence)
    reasons = list(extraction_reasons) + list(formatting_reasons)
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    if missing_strict:
        reasons.append("missing_required_goal")
    score = (8 if complete else -8) + (4 if shape_valid else -4) + (4 if not formatting_reasons else -12)
    score += 4 if tool_status == "pass" else (2 if tool_status == "partial_pass" else (-8 if tool_status == "conflict" else 0))
    return CandidateAssessment(
        value, source, extraction_method, score, complete, shape_valid,
        not formatting_reasons, tool_status, coverage, not complete, tuple(reasons), raw_has_meta,
        explicit_answer, verification_verdict,
    )


def choose_candidate(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
    usable = [candidate for candidate in candidates if candidate.accepted]
    if not usable:
        return None
    agreement = {
        id(candidate): sum(equivalent_answers(candidate.answer, other.answer) for other in usable)
        for candidate in usable
    }
    return max(
        usable,
        key=lambda item: (
            item.complete_goals,
            item.tool_status == "pass",
            item.tool_status == "partial_pass",
            agreement[id(item)],
            item.verification_verdict == "corrected",
            item.formatting_valid,
            item.explicit_answer,
            item.score,
            item.source == "solve",
            item.source == "arbitration",
            len(item.answer),
        ),
    )


def _goal_covered(answer: str, goal) -> bool:
    if not answer:
        return False
    compact = _compact(answer)
    if goal.requirements:
        return all(requirement.matches(answer) for requirement in goal.requirements)
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
    normalized = _compact(answer)
    for item in whole:
        expected = _compact(item.result)
        if expected and (expected in normalized or normalized in expected):
            return "pass"
    if whole:
        return "conflict"
    partial = [item for item in evidence if item.scope == "subexpression" and item.verified]
    if any(
        equivalent_answers(answer, item.result)
        or (_compact(item.result) and _compact(item.result) in normalized)
        for item in partial
    ):
        return "partial_pass"
    return "unknown"


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
        if not judgement:
            return False
        bare = re.fullmatch(r"(?:是|否|可以|不可以|正确|错误|成立|不成立)", _compact(answer))
        return not bare or not frame.subject or _compact(frame.subject) in _compact(answer)
    return True


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")
