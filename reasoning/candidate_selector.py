"""Structured completeness, mathematical checks, and independent selection."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import TYPE_CHECKING, Iterable

from classifier.choice import answer_choice_labels, option_labels
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
    certificate_method: str = ""
    certificate_checks: tuple[str, ...] = ()
    certificate_issues: tuple[str, ...] = ()
    support: str = ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    source: str = "local"
    detail: str = ""
    decisive: bool = False

    def trace_content(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "source": self.source,
            "detail": self.detail,
            "decisive": self.decisive,
        }


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
    result_coverage: tuple[bool, ...] = ()
    support_coverage: tuple[bool, ...] = ()
    format_coverage: tuple[bool, ...] = ()
    method_id: str = ""
    independence_group: str = ""
    math_checks: tuple[CheckResult, ...] = ()
    correctness_tier: str = "unverified"

    @property
    def accepted(self) -> bool:
        return self.validation_tier == "complete"

    @property
    def degraded(self) -> bool:
        return self.validation_tier == "degraded"

    @property
    def failed_check(self) -> bool:
        return any(item.status == "fail" for item in self.math_checks)

    @property
    def passed_check_count(self) -> int:
        return sum(item.status == "pass" and item.decisive for item in self.math_checks)

    @property
    def sanity_check_count(self) -> int:
        return sum(item.status == "pass" and not item.decisive for item in self.math_checks)


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
    *,
    method_id: str = "",
    independence_group: str = "",
    math_checks: Iterable[CheckResult] = (),
) -> CandidateAssessment:
    value = str(answer or "").strip()
    checks = tuple(math_checks)
    structure_reasons = Finalizer.validate_structure(value)
    consistency = candidate_consistency_reasons(value, spec)
    task_kind = getattr(spec.profile, "task_kind", spec.profile.problem_type)
    proof_like = task_kind in {"proof", "derivation", "explanation"}

    result_coverage: list[bool] = []
    support_coverage: list[bool] = []
    format_coverage: list[bool] = []
    for goal in spec.goals:
        result_coverage.append(all(
            requirement.matches(value)
            for requirement in goal.result_requirements
            if requirement.strict
        ))
        support_coverage.append(all(
            requirement.matches(value)
            for requirement in goal.support_requirements
            if requirement.strict
        ))
        format_coverage.append(all(
            requirement.matches(value)
            for requirement in goal.format_requirements
            if requirement.strict
        ))

    complete = bool(result_coverage) and all(result_coverage) and all(support_coverage) and all(format_coverage)
    shape_valid = _valid_shape(value, spec)
    tool_status = _tool_status(value, evidence)
    reasons = [*extraction_reasons, *structure_reasons, *consistency]
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if not complete:
        reasons.append("missing_answer_obligation")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    if any(item.status == "fail" for item in checks):
        reasons.append("mathematical_check_failed")
    if proof_like and not _has_proof_support(value):
        reasons.append("missing_proof_reasoning")
    if raw_has_meta and not explicit_answer:
        reasons.append("meta_without_explicit_answer")

    hard_reasons = {
        "empty", "placeholder", "meaningless_fragment", "markup_fragment",
        "meta_text", "meta_without_explicit_answer", "control_character",
        "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
        "unclosed_display_latex", "unclosed_latex_environment", "unclosed_latex_brace",
        "unclosed_group_delimiter", "unclosed_quote", "trailing_fragment",
        "truncated_sentence", "numeric_identity_conflict", "final_conclusion_conflict",
        "named_scalar_conflict", "tool_conflict", "mathematical_check_failed",
    }
    hard_rejected = bool(set(reasons).intersection(hard_reasons))
    if hard_rejected or not value:
        tier = "rejected"
    elif complete and shape_valid and not structure_reasons:
        tier = "complete"
    else:
        tier = "degraded"

    pass_count = sum(item.status == "pass" and item.decisive for item in checks)
    correctness_tier = "certified" if tool_status == "pass" else (
        "checked" if pass_count else "unverified"
    )
    score = {"complete": 30, "degraded": 5, "rejected": -50}[tier]
    score += {"pass": 20, "partial_pass": 8, "unknown": 0, "conflict": -30}[tool_status]
    score += pass_count * 6
    score += min(2, sum(item.status == "pass" and not item.decisive for item in checks))
    score -= sum(item.status == "unknown" for item in checks)
    score += 4 if explicit_answer else 0
    if proof_like and len(value) >= 80 and _has_proof_support(value):
        score += 6

    coverage = tuple(
        result and support and formatting
        for result, support, formatting in zip(result_coverage, support_coverage, format_coverage)
    )
    return CandidateAssessment(
        answer=value,
        source=source,
        extraction_method=extraction_method,
        score=score,
        complete_goals=complete,
        shape_valid=shape_valid,
        formatting_valid=not structure_reasons,
        tool_status=tool_status,
        goal_coverage=coverage,
        coverage_uncertain=not complete or correctness_tier == "unverified",
        rejected_reasons=tuple(dict.fromkeys(reasons)),
        raw_has_meta=raw_has_meta,
        explicit_answer=explicit_answer,
        verification_verdict=verification_verdict,
        validation_tier=tier,
        result_coverage=tuple(result_coverage),
        support_coverage=tuple(support_coverage),
        format_coverage=tuple(format_coverage),
        method_id=method_id,
        independence_group=independence_group or source,
        math_checks=checks,
        correctness_tier=correctness_tier,
    )


def choose_candidate(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
    usable = [
        item for item in candidates
        if item.validation_tier in {"complete", "degraded"}
        and not item.failed_check
    ]
    if not usable:
        return None

    agreement: dict[int, int] = {}
    for candidate in usable:
        groups = {
            other.independence_group
            for other in usable
            if other is not candidate
            and other.independence_group != candidate.independence_group
            and _candidate_equivalent(candidate, other)
        }
        agreement[id(candidate)] = len(groups)

    return max(usable, key=lambda item: (
        item.validation_tier == "complete",
        item.tool_status == "pass",
        agreement[id(item)] > 0,
        agreement[id(item)],
        item.passed_check_count,
        item.correctness_tier == "checked",
        item.complete_goals,
        item.formatting_valid,
        item.source in {"independent", "arbitration", "recovery"},
        item.score,
        item.explicit_answer,
        -len(item.answer),
    ))


def candidate_consistency_reasons(answer: str, spec=None) -> tuple[str, ...]:
    del spec
    value = str(answer or "")
    reasons: list[str] = []
    if _numeric_identity_conflict(value):
        reasons.append("numeric_identity_conflict")
    explicit = [item.answer for item in Finalizer.extract_explicit_results(value) if item.valid and item.answer]
    if len(explicit) >= 2:
        first = explicit[0]
        if any(not equivalent_answers(first, item) for item in explicit[1:]):
            reasons.append("final_conclusion_conflict")
    if _named_scalar_revision_conflict(value):
        reasons.append("named_scalar_conflict")
    return tuple(dict.fromkeys(reasons))


def _numeric_identity_conflict(value: str) -> bool:
    text = str(value or "")
    delimiters = "\n,，;；:：$="
    depths: list[int] = []
    depth = 0
    for character in text:
        if character in ")]}" and depth > 0:
            depth -= 1
        depths.append(depth)
        if character in "([{":
            depth += 1
    for equality in re.finditer(r"(?<![<>=])=(?!=)", text):
        equality_depth = depths[equality.start()]
        left_positions = [
            position for position, character in enumerate(text[:equality.start()])
            if character in delimiters and depths[position] == equality_depth
        ]
        left_boundary = max(left_positions) if left_positions else -1
        right_positions = [
            position for position in range(equality.end(), len(text))
            if text[position] in delimiters and depths[position] == equality_depth
        ]
        right_boundary = min(right_positions) if right_positions else len(text)
        left = _normalize_numeric_syntax(
            text[left_boundary + 1:equality.start()].strip(" \\()[]")
        )
        right = _normalize_numeric_syntax(
            text[equality.end():right_boundary].strip(" \\()[]。.!?？")
        )
        numeric_syntax = r"[-+]?\d[\d\s()+\-*/^.]*"
        if not re.fullmatch(numeric_syntax, left) or not re.fullmatch(numeric_syntax, right):
            continue
        left_value = _safe_numeric(left)
        right_value = _safe_numeric(right)
        if (
            left_value is not None
            and right_value is not None
            and not math.isclose(left_value, right_value, rel_tol=1e-10, abs_tol=1e-10)
        ):
            return True
    return False


def _normalize_numeric_syntax(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(
        r"\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}",
        r"(\1)/(\2)",
        text,
    )
    text = re.sub(r"\\(?:times|cdot)(?![A-Za-z])|[×·]", "*", text)
    text = re.sub(r"\\div(?![A-Za-z])|÷", "/", text)
    return text.replace("{", "(").replace("}", ")")


def _named_scalar_revision_conflict(value: str) -> bool:
    text = str(value or "")
    if not re.search(
        r"修正|更正|重新|不符|矛盾|错误|有误|反复验证|"
        r"\b(?:however|correction|corrected|recheck|re-evaluate|incorrect|wrong|conflict)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    scalar = (
        r"(?:\\boxed\s*\{\s*)?"
        r"(?:\\(?:d?frac|tfrac)\s*\{[-+]?\d+\}\s*\{[-+]?\d+\}|"
        r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?)"
        r"(?:\s*\})?"
    )
    assignment = re.compile(
        r"(?P<lhs>[A-Za-z](?:\s*_\s*\{[^{}]{1,30}\})?"
        r"(?:\s*\([^=\n]{0,60}\))?)\s*=\s*(?P<rest>.+)",
        re.IGNORECASE,
    )
    terminal = re.compile(
        rf"(?:^|=)\s*(?P<rhs>{scalar})\s*(?:\$|\\\]|[.。；;])*\s*$",
        re.IGNORECASE,
    )
    assertions: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = assignment.search(line)
        if not match:
            continue
        terminal_match = terminal.search(match.group("rest"))
        if not terminal_match:
            continue
        lhs = re.sub(r"[\s{}\\]", "", match.group("lhs")).casefold()
        rhs = re.sub(
            r"^\\boxed\s*\{(.*)\}$",
            r"\1",
            terminal_match.group("rhs").strip(),
        )
        assertions.setdefault(lhs, []).append(rhs)
    for values in assertions.values():
        if len(values) < 2:
            continue
        first = values[0]
        if any(not equivalent_answers(first, other) for other in values[1:]):
            return True
    return False


def _candidate_equivalent(left: CandidateAssessment, right: CandidateAssessment) -> bool:
    return equivalent_answers(_comparison_value(left.answer), _comparison_value(right.answer))


def _comparison_value(value: str) -> str:
    extracted = Finalizer.extract_result(value)
    return extracted.answer if extracted.valid and extracted.answer else value


def _tool_status(answer: str, evidence: tuple[ToolEvidence, ...]) -> str:
    whole = [item for item in evidence if item.scope == "whole_goal" and item.verified]
    if whole:
        candidate = _comparison_value(answer)
        if any(equivalent_answers(candidate, item.result) for item in whole):
            return "pass"
        return "conflict"
    goal_results = [
        item for item in evidence if item.scope == "goal_result" and item.verified
    ]
    if goal_results:
        if any(
            not equivalent_answers(goal_results[0].result, item.result)
            for item in goal_results[1:]
        ):
            return "unknown"
        candidate = _comparison_value(answer)
        if any(equivalent_answers(candidate, item.result) for item in goal_results):
            return "pass"
        return "conflict"
    partial = [item for item in evidence if item.scope == "subexpression" and item.verified]
    candidate = _comparison_value(answer)
    if any(equivalent_answers(candidate, item.result) for item in partial):
        return "partial_pass"
    return "unknown"


def _valid_shape(answer: str, spec: "ProblemSpec") -> bool:
    value = str(answer or "").strip()
    if not value:
        return False
    shape = spec.profile.answer_shape
    if shape == "choice":
        labels = answer_choice_labels(_comparison_value(value))
        available = set(option_labels(spec.problem_text))
        return bool(labels) and set(labels) <= available
    if shape == "truth":
        if not re.search(r"是|否|正确|错误|成立|不成立|可|不可|true|false|yes|no|holds?|does not", value, re.IGNORECASE):
            return False
        subject = spec.answer_frame.subject
        bare = re.fullmatch(r"\s*(?:是|否|正确|错误|成立|不成立|true|false|yes|no)[。.!]?\s*", value, re.IGNORECASE)
        return not (bare and subject)
    if shape == "roots":
        return bool(re.search(r"[A-Za-z]\s*=|解集|无解|不存在|\\varnothing|no solutions?", value, re.IGNORECASE))
    if shape == "interval":
        return bool(re.search(r"[\[(].+[,，].+[\])]|区间|定义域|范围|\\cup", value, re.DOTALL))
    if shape == "matrix":
        return bool(re.search(r"\\begin\{[pbvBV]?matrix\}|\[\s*\[", value))
    if shape in {"number", "count", "probability"}:
        return bool(re.search(r"[-+]?\d|\\frac|\\sqrt|\\pi|π|∞|\\infty", value))
    if shape == "proof":
        return _has_proof_support(value) and len(value) >= 24
    return bool(re.search(r"[\w\u4e00-\u9fff\\=+\-*/^]", value))


def _has_proof_support(value: str) -> bool:
    if re.search(
        r"因为|由于|根据|由.*得|所以|故|因此|从而|推出|假设|反设|若.*则|矛盾|"
        r"\b(?:because|since|therefore|hence|thus|by|assume|suppose|contradiction|"
        r"implies?|follows from|if\b.*\bthen)\b",
        value,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    return len(re.findall(r"(?<![<>!])=(?!=)|≤|≥|<|>|\\(?:leq|geq|implies)", value)) >= 2


def _safe_numeric(expression: str) -> float | None:
    try:
        tree = ast.parse(str(expression).replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError):
        return None
    allowed = (
        ast.Expression, ast.Constant, ast.UnaryOp, ast.BinOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.UAdd, ast.USub,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return None
    try:
        result = eval(compile(tree, "<numeric-check>", "eval"), {"__builtins__": {}}, {})
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None
    return float(result) if isinstance(result, (int, float)) and math.isfinite(float(result)) else None
