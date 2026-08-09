"""Deterministic candidate validation and selection for submitted answers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING

from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
from classifier.choice import answer_choice_labels

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
    validation_tier: str = "rejected"

    @property
    def accepted(self) -> bool:
        return self.validation_tier == "complete"

    @property
    def degraded(self) -> bool:
        return self.validation_tier == "degraded"


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
    extracted_value = Finalizer.extract_result(value)
    semantic_value = (
        extracted_value.answer
        if (
            extracted_value.valid
            and extracted_value.explicit_answer
            and (
                "\n" not in value
                or (
                    len(Finalizer._boxed_values(value)) == 1
                    and bool(re.fullmatch(
                        r"(?:\\\[|\$\$?)?\s*\\boxed\{.*\}\s*(?:\\\]|\$\$?)?",
                        value,
                        re.DOTALL,
                    ))
                )
            )
        )
        else value
    )
    formatting_reasons = Finalizer.validate_structure(value)
    coverage = tuple(_goal_covered(semantic_value, goal) for goal in spec.goals)
    complete = bool(coverage) and all(coverage)
    contract = getattr(spec, "answer_contract", None)
    proof_contract = getattr(contract, "mode", "") == "proof"
    proof_reasoning_missing = proof_contract and not bool(re.search(
        r"(?:因为|由于|由|依据|根据|所以|故|因此|推出|可得|"
        r"\b(?:because|since|therefore|hence|thus|by)\b)",
        semantic_value,
        re.IGNORECASE,
    ))
    if proof_reasoning_missing:
        complete = False
    hard_requirement_names = {
        "judgement", "integral_value", "pointwise_limit", "intersection",
        "exact_comparison", "determinant", "trace", "generator_enumeration",
        "domain", "solution_formula", "first_iteration", "iteration_formula",
        "field_value", "degree_value", "galois_verdict", "two_items",
        "phrase_decomposition", "encoded_string", "alternative_result",
    }
    missing_strict = any(
        requirement.strict
        and requirement.name in hard_requirement_names
        and not requirement.matches(semantic_value)
        for goal in spec.goals
        for requirement in goal.requirements
    )
    shape_valid = (
        _valid_shape(semantic_value, spec.profile.answer_shape)
        and _frame_valid(semantic_value, spec)
    )
    tool_status = _tool_status(value, evidence)
    reasons = list(extraction_reasons) + list(formatting_reasons)
    if _has_false_binomial_identity(value):
        reasons.append("numeric_identity_conflict")
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    if proof_reasoning_missing:
        reasons.append("missing_proof_reasoning")
    missing_required = missing_strict or (len(spec.goals) > 1 and not complete)
    if not complete:
        reasons.append("missing_required_goal")
    hard_format_reasons = {
        "empty", "placeholder", "meaningless_fragment", "markup_fragment", "meta_text",
        "meta_without_explicit_answer", "control_character", "uncertain_fragment",
        "provider_truncated_without_explicit_answer", "provider_truncated_ambiguous_box",
        "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
        "unclosed_display_latex", "unclosed_latex_environment", "unclosed_latex_brace",
        "unclosed_group_delimiter",
        "trailing_fragment", "truncated_sentence",
        "numeric_identity_conflict",
    }
    hard_rejected = bool(set(reasons) & hard_format_reasons) or tool_status == "conflict"
    tier = "rejected" if hard_rejected else (
        "complete" if complete and shape_valid and not missing_required and not formatting_reasons else "degraded"
    )
    score = (12 if tier == "complete" else (2 if tier == "degraded" else -20))
    score += (4 if complete else -2) + (3 if shape_valid else -2) + (3 if not formatting_reasons else -3)
    score += 4 if tool_status == "pass" else (2 if tool_status == "partial_pass" else (-8 if tool_status == "conflict" else 0))
    if spec.answer_frame.style == "proof":
        has_derivation = bool(re.search(r"(?:因为|由于|由|故|因此|推出|可得).*[=≤≥<>]|[=≤≥<>].*(?:故|因此|推出|可得)", value, re.DOTALL))
        if len(value) >= 80 and has_derivation:
            score += 4
        elif len(value) < 45:
            score -= 4
    return CandidateAssessment(
        answer=value,
        source=source,
        extraction_method=extraction_method,
        score=score,
        complete_goals=complete,
        shape_valid=shape_valid,
        formatting_valid=not formatting_reasons,
        tool_status=tool_status,
        goal_coverage=coverage,
        coverage_uncertain=not complete,
        rejected_reasons=tuple(dict.fromkeys(reasons)),
        raw_has_meta=raw_has_meta,
        explicit_answer=explicit_answer,
        verification_verdict=verification_verdict,
        validation_tier=tier,
    )


def choose_candidate(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
    usable = [candidate for candidate in candidates if candidate.validation_tier in {"complete", "degraded"}]
    if not usable:
        return None
    agreement = {}
    for candidate in usable:
        candidate_stage = _source_stage(candidate.source)
        agreeing_stages = {
            _source_stage(other.source)
            for other in usable
            if other is not candidate
            and _source_stage(other.source) != candidate_stage
            and equivalent_answers(candidate.answer, other.answer)
        }
        agreement[id(candidate)] = len(agreeing_stages)
    return max(
        usable,
        key=lambda item: (
            item.validation_tier == "complete",
            item.tool_status == "pass",
            sum(item.goal_coverage),
            item.complete_goals,
            item.formatting_valid,
            _source_stage(item.source) == "arbitration",
            item.verification_verdict == "corrected",
            _source_stage(item.source) == "verify",
            agreement[id(item)],
            item.tool_status == "partial_pass",
            not item.raw_has_meta,
            item.score,
            item.explicit_answer,
        ),
    )


def _goal_covered(answer: str, goal) -> bool:
    if not answer:
        return False
    compact = _compact(answer)
    if goal.requirements:
        return all(requirement.matches(answer) for requirement in goal.requirements)
    if goal.required_terms:
        return all(_compact(term) in compact for term in goal.required_terms)
    if goal.kind == "proof":
        return bool(re.search(r"(?:因为|由于|由|依据|故|因此|推出|可得)", answer))
    if goal.kind == "truth_judgement":
        return bool(re.search(r"(?:是|否|正确|错误|成立|不成立|收敛|发散|可约|不可约)", answer))
    if goal.kind == "domain_or_interval":
        return bool(re.search(r"(?:区间|定义域|[\[(][^\n,，]+[,，][^\n)\]]+[)\]])", answer))
    if goal.kind == "formula":
        return bool(re.search(r"[=+\-*/^]|\\(?:frac|sum|int|sqrt|begin)", answer))
    if goal.kind == "comparison":
        return len(re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", answer)) >= 2 or bool(
            re.search(r"(?:大于|小于|相等|误差|放大|不变|趋于)", answer)
        )
    if goal.kind == "construction":
        return bool(re.search(r"(?:取|令|构造|例如|=|\{|\[)", answer))
    return bool(re.search(r"[\w\u4e00-\u9fff=+\-*/^\\]", answer))


def _valid_shape(answer: str, shape: str) -> bool:
    if not answer:
        return False
    if shape == "choice":
        return bool(answer_choice_labels(answer))
    if shape == "roots" and re.fullmatch(r"\[\s*[^\[\]]+[,，][^\[\]]+\s*\]", answer):
        return False
    if shape == "proof":
        # A derivation can legitimately terminate in a one-symbol conclusion
        # after finalization has removed an unsafe scratchpad.
        return len(answer) >= 4 or bool(re.fullmatch(r"[A-Za-z0-9_+\-*/=^\\]+", answer))
    if shape in {"number", "expression", "matrix", "interval"}:
        if re.search(
            r"(?im)^\s*(?:we\s+(?:can|need|should|will|must|might|now|try)|"
            r"suppose\b|the\s+(?:problem|task)\b|so\s+(?:the|this)\b|"
            r"in\s+column\b|end\s+at\b|case\s+\d+\b|"
            r"我们(?:可以|需要|应该|将要)|(?:这个|该)(?:问题|任务))",
            answer,
            re.IGNORECASE,
        ):
            return False
        if len(answer) > 1200 and not re.search(r"(?:结论|final|answer|=)", answer, re.IGNORECASE):
            return False
    return not _is_scratchpad(answer)


def _source_stage(source: str) -> str:
    raw = str(source or "").split("#", 1)[0]
    return {
        "continue": "solve",
        "continue_last": "solve",
        "verify_recovered": "verify",
        "continue_verify": "verify",
        "retry_verify": "verify",
        "last_chance": "rescue",
    }.get(raw, raw)


def _tool_status(answer: str, evidence: tuple[ToolEvidence, ...]) -> str:
    whole = [item for item in evidence if item.scope == "whole_goal" and item.verified]
    normalized = _compact(answer)
    for item in whole:
        expected = _compact(item.result)
        if expected and (expected == normalized or equivalent_answers(answer, item.result)):
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
    for item in partial:
        if item.operation != "recurrence_solution" or "=" not in item.result:
            continue
        lhs = _compact(item.result.split("=", 1)[0])
        if lhs and lhs in normalized:
            return "conflict"
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
        verdict_pattern = (
            r"(?:是|否|可以|不可以|正确|错误|成立|不成立|"
            r"属于|不属于|可导|不可导|不可复可导|调和|不调和|为解|不是解)"
        )
        judgement = re.search(verdict_pattern, answer)
        if not judgement:
            return False
        bare = re.fullmatch(verdict_pattern, _compact(answer))
        return not bare or not frame.subject or _compact(frame.subject) in _compact(answer)
    return True


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")


def _has_false_binomial_identity(answer: str) -> bool:
    """Reject closed, directly checkable binomial equalities that are false."""
    value = str(answer or "").replace("−", "-")
    patterns = (
        re.compile(
            r"\\binom\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}"
            r"\s*=\s*(-?\d+)(?![\d.])",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z])C\s*\(\s*(\d+)\s*[,，]\s*(\d+)\s*\)"
            r"\s*=\s*(-?\d+)(?![\d.])",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<![A-Za-z])C\s*_\s*\{?\s*(\d+)\s*\}?\s*\^\s*\{?\s*(\d+)\s*\}?"
            r"\s*=\s*(-?\d+)(?![\d.])",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?<!\d)(\d+)\s+choose\s+(\d+)\s*=\s*(-?\d+)(?![\d.])",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(value):
            n, k, stated = map(int, match.groups())
            expected = math.comb(n, k) if 0 <= k <= n else 0
            if stated != expected:
                return True
    return False
