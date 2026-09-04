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
    assurance: str = "schema"


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
    transport_truncated: bool = False

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
    if not explicit_answer and extraction_method in {
        "direct", "whole_response", "whole_solution"
    }:
        # A phrase such as "next we prove..." is ordinary exposition when a
        # conclusion actually follows it.  It signals an unfinished draft only
        # when at least one announced action remains unresolved at the end.
        future_pattern = re.compile(
            r"接下来|下一步|随后(?:还|将|需要)|仍需|还需|继续(?:分析|计算|推导|证明)|"
            r"\b(?:next|then)\s+(?:i|we)\s+(?:need|will|should|must)\b|"
            r"\b(?:i|we)\s+(?:will|need\s+to|should|must)\s+(?:continue|analy[sz]e|"
            r"compute|derive|prove|check)\b",
            re.IGNORECASE,
        )
        completion_pattern = re.compile(
            r"故|因此|所以|从而|可得|得证|结论|成立|"
            r"\b(?:therefore|hence|thus|consequently|proved|follows?|holds?)\b|"
            r"^\s*(?:FINAL|最终答案|最终结论|结论)\s*[:：]",
            re.IGNORECASE | re.MULTILINE,
        )
        future_action = any(
            not completion_pattern.search(value[match.end():])
            for match in future_pattern.finditer(value)
        )
        intermediate_only = bool(
            re.search(
                r"计算过程(?:如下)?|推导过程(?:如下)?|中间(?:值|结果|步骤)|"
                r"intermediate\s+(?:value|result|step)|work(?:ing)?\s+in\s+progress",
                value,
                re.IGNORECASE,
            )
            and not re.search(
                r"最终答案|答案|结论|结果\s*(?:为|是|=)|"
                r"(?:故|因此|所以|从而|可得)\s*(?:得|有|为)?|"
                r"therefore|hence|thus|final\s+answer|answer\s*(?:is|=)|"
                r"result\s*(?:is|=)",
                value,
                re.IGNORECASE,
            )
        )
        if future_action:
            reasons.append("unlabelled_future_action")
        if intermediate_only or (future_action and re.search(r"中间|intermediate", value, re.IGNORECASE)):
            reasons.append("unlabelled_intermediate_result")
        process_body = bool(
            "\n" in value
            and re.search(
                r"^\s*(?:假设|设|先|由|计算|推导|证明|assume|suppose|let|first|compute|derive|proof)\b",
                value,
                re.IGNORECASE,
            )
            and not re.search(
                r"FINAL\s*:|最终答案|最终结论|答案\s*(?:为|是|:|：)|"
                r"结论\s*(?:为|是|:|：)|因此|所以|故|从而|"
                r"final\s+answer|conclusion|therefore|hence|thus",
                value,
                re.IGNORECASE,
            )
        )
        if process_body:
            reasons.append("unlabelled_process_body")
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if not complete:
        reasons.append("missing_answer_obligation")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    if any(item.status == "fail" for item in checks):
        reasons.append("mathematical_check_failed")
    if proof_like and not _has_proof_support(value):
        reasons.append("missing_required_support")
        reasons.append("missing_proof_reasoning")
    if task_kind == "construction":
        requirements = tuple(
            requirement
            for goal in spec.goals
            for requirement in goal.requirements
        )
        object_requirements = [
            item for item in requirements if item.name == "construction_object"
        ]
        check_requirements = [
            item for item in requirements if item.name == "construction_check"
        ]
        if object_requirements and not all(item.matches(value) for item in object_requirements):
            reasons.append("missing_construction_object")
        if check_requirements and not all(item.matches(value) for item in check_requirements):
            reasons.append("missing_construction_verification")
    if raw_has_meta and not explicit_answer:
        reasons.append("meta_without_explicit_answer")

    hard_reasons = {
        "empty", "placeholder", "meaningless_fragment", "markup_fragment",
        "meta_text", "meta_without_explicit_answer", "control_character",
        "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
        "unclosed_display_latex", "unclosed_latex_environment", "unclosed_latex_brace",
        "unclosed_group_delimiter", "unclosed_quote", "trailing_fragment",
        "truncated_sentence", "omitted_fragment", "numeric_identity_conflict", "final_conclusion_conflict",
        "named_scalar_conflict", "tool_conflict", "mathematical_check_failed",
        "unresolved_self_retraction", "unlabelled_future_action",
        "unlabelled_intermediate_result", "unlabelled_process_body",
        "uncertain_fragment", "referential_fragment",
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

    source_priority = {
        "primary": 4,
        "independent": 3,
        "independent_solve": 3,
        "primary_recovery": 3,
        "blind_tiebreak": 2,
        "deep_arbitration": 2,
        "last_chance": 2,
        "recovery": 1,
    }
    # The public protocol makes a FINAL/答案-labelled conclusion authoritative.
    # Models often box individual options or intermediate values while checking
    # them, so an unlabelled box must not beat a labelled composite answer merely
    # because the fragment is shorter.
    method_priority = {
        "tagged_solution": 5,
        "terminal_supported_solution": 5,
        "label_boxed": 4,
        "label_next_line_boxed": 4,
        "bracket_label": 4,
        "label": 3,
        "label_next_line": 3,
        "whole_solution": 2,
        "boxed": 1,
    }

    group_position: dict[int, int] = {}
    next_position: dict[str, int] = {}
    for item in usable:
        group = item.independence_group
        group_position[id(item)] = next_position.get(group, 0)
        next_position[group] = group_position[id(item)] + 1

    # When every evidential field ties within one response group, its later
    # explicit conclusion is the final correction. Cross-group order is not
    # evidence, and answer length is only presentation.
    return max(usable, key=lambda item: (
        item.tool_status == "pass",
        item.passed_check_count,
        agreement[id(item)] > 0,
        agreement[id(item)],
        item.validation_tier == "complete",
        item.correctness_tier == "checked",
        item.complete_goals,
        item.formatting_valid,
        method_priority.get(item.extraction_method, 0),
        not item.transport_truncated,
        source_priority.get(item.source, 0),
        item.score,
        item.explicit_answer,
        group_position[id(item)],
    ))


def candidate_consistency_reasons(answer: str, spec=None) -> tuple[str, ...]:
    value = str(answer or "")
    reasons: list[str] = []
    if _numeric_identity_conflict(value):
        reasons.append("numeric_identity_conflict")
    explicit_results = [
        item for item in Finalizer.extract_explicit_results(value)
        if item.valid and item.answer
    ]
    # A proof may box a lemma value or an intermediate matrix.  Only repeated
    # answer-labelled conclusions are authoritative enough to contradict one
    # another; unlabelled boxes remain mathematical body content.
    labelled_methods = {
        "label", "label_boxed", "bracket_label",
        "label_next_line", "label_next_line_boxed",
    }
    labelled = [
        item.answer for item in explicit_results
        if item.method in labelled_methods
    ]
    if len(labelled) >= 2:
        first = labelled[0]
        if any(not equivalent_answers(first, item) for item in labelled[1:]):
            reasons.append("final_conclusion_conflict")
    if _authoritative_result_conflict(value):
        reasons.append("final_conclusion_conflict")
    if _authoritative_binary_property_conflict(value, spec):
        reasons.append("final_conclusion_conflict")
    if _named_scalar_revision_conflict(value):
        reasons.append("named_scalar_conflict")
    return tuple(dict.fromkeys(reasons))


def _authoritative_result_conflict(value: str) -> bool:
    """Compare an explicit answer with a conflicting terminal conclusion."""
    text = str(value or "").strip()
    if not text:
        return False
    text = re.sub(
        r"((?:因此|所以|故|从而|由此可得)?\s*(?:正确)?(?:答案|结论|结果)\s*(?:为|是|=|:|：))"
        r"\s*\n\s*(?=\\(?:\[|boxed))",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    cue = re.compile(
        r"^\s*(?P<cue>FINAL|最终答案|最终结论|答案|结论|CONCLUSION|"
        r"更正后答案|修正后答案|corrected\s+answer|correct\s+answer|"
        r"因此|所以|故|从而|由此可得|therefore|hence|thus)"
        r"\s*[:：,，]?",
        re.IGNORECASE,
    )
    explicit_cue = re.compile(
        r"FINAL|最终答案|最终结论|答案|结论|CONCLUSION|更正|修正|correct",
        re.IGNORECASE,
    )
    explicit_results = [
        item
        for item in Finalizer.extract_explicit_results(text)
        if item.valid and item.answer and item.method in {
            "label", "label_boxed", "bracket_label",
            "label_next_line", "label_next_line_boxed",
        }
    ]
    answer_noun = re.compile(
        r"^(?:正确)?(?:答案|结论|结果|所求(?:值|数|概率|总数)?)|"
        r"^(?:the\s+)?(?:answer|conclusion|result)\b",
        re.IGNORECASE,
    )
    named_result = re.compile(
        r"(?:^|[\s$\\(])(?:[A-Za-z][A-Za-z0-9_]*(?:\s*[\[(][^\]\)]{0,60}[\])])?"
        r"|\\[A-Za-z]+(?:\s*_[A-Za-z0-9{}]+)?)\s*=|"
        r"(?:值|数|概率|总数|个数|置换数|维数|阶数|次数|系数|根)"
        r"\s*(?:为|是|等于|=)",
        re.IGNORECASE,
    )
    clauses = [
        clause.strip()
        for clause in re.split(r"[。!?？；;\n]+|(?<!\d)\.(?!\d)", text)
        if clause.strip()
    ]
    conclusions: list[str] = []
    for index, clause in enumerate(clauses):
        match = cue.search(clause)
        if match is None:
            continue
        if index != len(clauses) - 1 and not explicit_cue.search(match.group("cue")):
            continue
        body = cue.sub("", clause, count=1).strip()
        # Once an explicit FINAL exists, a terminal "therefore ..." clause
        # in its proof can be an intermediate cardinality or lemma.  Compare
        # it only when it is answer-labelled or states a named mathematical
        # result.  Thus "x=2", "P(A)=1/4", and "置换数为105" are checked,
        # while a narrative fact such as "共有42个元素形成轨道" is not
        # mistaken for a second final answer.
        if (
            explicit_results
            and not explicit_cue.search(match.group("cue"))
            and not answer_noun.search(body)
            and not named_result.search(body)
        ):
            continue
        body = re.sub(
            r"^(?:正确)?(?:答案|结论|结果|所求(?:值|数|概率|总数)?)\s*(?:为|是|=|is|are)?\s*",
            "",
            body,
            flags=re.IGNORECASE,
        )
        explicit = [
            item.answer
            for item in Finalizer.extract_explicit_results(clause)
            if item.valid and item.answer
        ]
        if explicit:
            conclusions.append(explicit[-1])
        elif body and not re.fullmatch(
            r"(?:the|this|that)?\s*(?:claim|assertion|statement|result|"
            r"conclusion)\s+(?:follows|holds|is\s+(?:proved|true|established))"
            r"\s*[.。]?|"
            r"(?:原)?(?:命题|断言|结论)(?:得证|成立|为真)|证毕",
            body,
            re.IGNORECASE,
        ):
            conclusions.append(body)
    if not conclusions and re.match(r"^\s*\\boxed\s*\{", text):
        extracted = Finalizer.extract_result(text)
        if extracted.valid and extracted.answer:
            conclusions.append(extracted.answer)
    elif re.match(r"^\s*\\boxed\s*\{", text):
        extracted = Finalizer.extract_result(text)
        if extracted.valid and extracted.answer:
            conclusions.insert(0, extracted.answer)
    if len(conclusions) < 2:
        return False

    signatures = [_conclusion_signature(item) for item in conclusions]
    signatures = [item for item in signatures if item is not None]
    for index, ((left_name, left_value), left_raw) in enumerate(
        zip(signatures, conclusions)
    ):
        for (right_name, right_value), right_raw in zip(
            signatures[index + 1:], conclusions[index + 1:]
        ):
            if left_name == "__multi__" or right_name == "__multi__":
                left_fields = _conclusion_assignments(left_raw)
                right_fields = _conclusion_assignments(right_raw)
                common_fields = left_fields.keys() & right_fields.keys()
                if common_fields and any(
                    not equivalent_answers(left_fields[field], right_fields[field])
                    for field in common_fields
                ):
                    return True
                continue
            if left_name and right_name and left_name != right_name:
                continue
            if bool(left_name) != bool(right_name) and not (
                _atomic_conclusion(left_value) and _atomic_conclusion(right_value)
            ):
                continue
            if not equivalent_answers(left_value, right_value):
                return True
    return False


def _conclusion_assignments(value: str) -> dict[str, str]:
    text = str(value or "").strip().strip("$ ")
    text = re.sub(r"^\\boxed\s*\{(.*)\}$", r"\1", text, flags=re.DOTALL)
    fields: dict[str, str] = {}
    lhs = (
        r"(?:[A-Za-z](?:\s*_\s*\{?[^=,，]{1,30}\}?|\s*\([^=,，]{1,40}\)|"
        r"\s*\[[^=,，]{1,40}\])?|\\?[A-Za-z]+\s*[\[(][^=,，]{1,40}[\])])"
    )
    for part in re.split(r"[,，]", text):
        match = re.search(
            rf"(?P<lhs>{lhs})\s*=\s*(?P<rhs>.+)$",
            part.strip(),
        )
        if not match:
            continue
        name = re.sub(r"[\s{}\\]", "", match.group("lhs")).casefold()
        result = match.group("rhs").strip().strip("$ ")
        result = re.sub(
            r"\s*(?:对所有[^。；;\n]{0,80}成立|for\s+all[^.;\n]{0,80}|holds?\s+for[^.;\n]{0,80})$",
            "",
            result,
            flags=re.IGNORECASE,
        ).strip()
        fields[name] = result
    return fields


def _conclusion_signature(value: str) -> tuple[str, str] | None:
    text = str(value or "").strip().strip("$ ")
    text = re.sub(r"^\\boxed\s*\{(.*)\}$", r"\1", text, flags=re.DOTALL)
    text = text.strip(" 。.!?？")
    text = re.sub(
        r"^(?:显式解|所求解|解|explicit\s+solution|solution)\s*(?:为|是|=|is)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*[（(]\s*(?:定义域|domain)[^）)\n]{0,120}[）)]\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip("$ ")
    approximate = re.search(
        r"^(?P<lhs>.+?)\s*(?:\\approx|≈)\s*(?P<rhs>.+)$",
        text,
    )
    if approximate:
        name = re.sub(r"[\s{}\\]", "", approximate.group("lhs")).casefold()
        return name, approximate.group("rhs").strip()
    named_approximation = re.search(
        r"^(?P<lhs>[A-Za-z][^。；;\n]{0,80}?)\s*(?:的近似值|"
        r"approximate\s+value)\s*(?:为|是|=|is)\s*(?P<rhs>.+)$",
        text,
        re.IGNORECASE,
    )
    if named_approximation:
        name = re.sub(
            r"[\s{}\\]", "", named_approximation.group("lhs")
        ).casefold()
        result = named_approximation.group("rhs").strip()
        result = re.sub(r"^\\\((.*)\\\)$", r"\1", result, flags=re.DOTALL)
        return name, result.strip("$ ")
    equality_count = len(re.findall(r"(?<![<>=])=(?!=)", text))
    if equality_count:
        if equality_count > 1 and re.search(r"[,，]", text):
            return "__multi__", text
        parts = re.split(r"(?<![<>=])=(?!=)", text)
        name = re.sub(r"[\s{}\\]", "", parts[0]).casefold()
        return name, parts[-1].strip()
    named = re.search(
        r"(?:为|是|等于|is|are|equals?)\s*\$?\s*(.+?)\s*\$?$",
        text,
        re.IGNORECASE,
    )
    if named:
        return "", named.group(1).strip()
    return "", text


def _atomic_conclusion(value: str) -> bool:
    normalized = _normalize_numeric_syntax(str(value or "").strip().strip("$"))
    if re.search(r"\d", normalized) and _safe_numeric(normalized) is not None:
        return True
    return bool(re.fullmatch(
        r"[A-H]|是|否|正确|错误|成立|不成立|不可约|可约|"
        r"true|false|yes|no|irreducible|reducible",
        str(value or "").strip(" $。.!"),
        re.IGNORECASE,
    ))


def _authoritative_binary_property_conflict(value: str, spec=None) -> bool:
    """Reject opposite verdicts stated as conclusions about the asked property."""
    problem = str(getattr(spec, "problem_text", "") or "")
    if not re.search(r"不可约|irreducible", problem, re.IGNORECASE):
        return False
    cue = re.compile(
        r"^\s*(?:最终(?:答案|结论)|答案|结论|综上(?:所述)?|因此|所以|故|"
        r"从而|由此可得|final\s+(?:answer|conclusion)|conclusion|therefore|hence|thus)"
        r"\s*[:：,，]?",
        re.IGNORECASE,
    )
    positive = re.compile(r"不可约|\birreducible\b", re.IGNORECASE)
    negative = re.compile(r"(?<!不)可约|\breducible\b", re.IGNORECASE)
    polarities: set[str] = set()
    for clause in re.split(r"[。.!?？；;\n]+", str(value or "")):
        if not cue.search(clause):
            continue
        body = cue.sub("", clause, count=1)
        conditional = bool(re.search(
            r"^(?:若|如果|假设|倘若|if|suppose|assuming)\b",
            body.strip(),
            re.IGNORECASE,
        ))
        if conditional:
            continue
        if positive.search(body):
            polarities.add("irreducible")
        if negative.search(body):
            polarities.add("reducible")
    return len(polarities) > 1


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
            text[left_boundary + 1:equality.start()].strip(" ()[]")
        ).strip()
        right = _normalize_numeric_syntax(
            text[equality.end():right_boundary].strip(" ()[]。.!?？")
        ).strip()
        numeric_syntax = r"[-+\d\s()+*/^.]+"
        if (
            not re.fullmatch(numeric_syntax, left)
            or not re.fullmatch(numeric_syntax, right)
            or not re.search(r"\d", left)
            or not re.search(r"\d", right)
        ):
            continue
        left_value = _safe_numeric(left)
        right_value = _safe_numeric(right)
        decimal_places = [
            len(digits)
            for digits in re.findall(r"\d+\.(\d+)", f"{left} {right}")
        ]
        absolute_tolerance = (
            max(1e-10, 0.51 * 10 ** (-min(decimal_places)))
            if decimal_places
            else 1e-10
        )
        if (
            left_value is not None
            and right_value is not None
            and not math.isclose(
                left_value,
                right_value,
                rel_tol=1e-10,
                abs_tol=absolute_tolerance,
            )
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
    revision = re.compile(
        r"修正|更正|重算|重新(?:计算|核算|推导|检查)|不符|矛盾|错误|有误|反复验证|"
        r"\b(?:correction|corrected|incorrect|wrong|conflict)\b|"
        r"\b(?:recheck(?:ed|ing)?|re-evaluat(?:e|ed|ing))\b"
        r"(?=[^.\n]{0,60}\b(?:error|wrong|correct(?:ion|ed)?)\b)",
        re.IGNORECASE,
    )
    if not revision.search(text):
        return False
    conditional_prefix = re.compile(
        r"(?:^|[.;；。!?\n])\s*(?:"
        r"if\b|when\b|(?:(?:in|for)\s+)?case\s+\w+|suppose\b|assume\b|"
        r"若|如果|假设|设若|当[^,，:：\n]{0,40}时|"
        r"情形\s*\w*|分支\s*\w*|在(?:情形|情况|分支)[^,，:：\n]{0,30}(?:下|中)"
        r")",
        re.IGNORECASE,
    )
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
    assertions: dict[str, list[tuple[str, int]]] = {}
    offset = 0
    for line in text.splitlines(keepends=True):
        match = assignment.search(line)
        if not match:
            offset += len(line)
            continue
        # Values introduced only inside a case, hypothesis, or rejected
        # branch are alternatives, not revisions of one asserted scalar.
        if conditional_prefix.search(line[:match.start()]):
            offset += len(line)
            continue
        terminal_match = terminal.search(match.group("rest"))
        if not terminal_match:
            offset += len(line)
            continue
        lhs = re.sub(r"[\s{}\\]", "", match.group("lhs")).casefold()
        rhs = re.sub(
            r"^\\boxed\s*\{(.*)\}$",
            r"\1",
            terminal_match.group("rhs").strip(),
        )
        end = offset + match.end()
        assertions.setdefault(lhs, []).append((rhs, end))
        offset += len(line)
    for values in assertions.values():
        if len(values) < 2:
            continue
        for previous, current in zip(values, values[1:]):
            previous_rhs, previous_end = previous
            current_rhs, current_end = current
            if equivalent_answers(previous_rhs, current_rhs):
                continue
            revision_context = text[previous_end:current_end]
            if revision.search(revision_context):
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
        if not re.search(
            r"是|否|正确|错误|成立|不成立|可|不可|收敛|发散|"
            r"必胜|必败|先手胜|后手胜|"
            r"true|false|yes|no|holds?|does\s+not|converges?|diverges?|"
            r"\b(?:wins?|loses?|winning|losing)\b|"
            r"is\s+(?:not\s+)?(?:an?\s+)?solution|satisf(?:y|ies)\s+(?:the\s+)?equation",
            value,
            re.IGNORECASE,
        ):
            return False
        subject = spec.answer_frame.subject
        bare = re.fullmatch(r"\s*(?:是|否|正确|错误|成立|不成立|true|false|yes|no)[。.!]?\s*", value, re.IGNORECASE)
        return not (bare and subject)
    if shape == "roots":
        return bool(re.search(r"[A-Za-z](?:\s*\([^)]*\))?\s*=|解集|无解|不存在|\\varnothing|no solutions?", value, re.IGNORECASE))
    if shape == "interval":
        return bool(re.search(r"[\[(].+[,，].+[\])]|区间|定义域|范围|\\cup", value, re.DOTALL))
    if shape == "matrix":
        return bool(re.search(r"\\begin\{[pbvBV]?matrix\}|\[\s*\[", value))
    if shape == "count":
        if re.search(r"[-+]?\d|\\frac|\\sqrt|\\pi|π|∞|\\infty", value):
            return True
        # A count may be requested as a formula in parameters rather than as
        # one literal integer (for example a graph-density lower bound).  Keep
        # this deliberately narrow: require a compact mathematical expression
        # with an actual operator, not arbitrary prose containing a variable.
        compact = value.strip().strip("$").strip()
        boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}\s*[。.]?", compact, re.DOTALL)
        if boxed:
            compact = boxed.group(1).strip()
        return bool(
            1 <= len(compact) <= 240
            and re.fullmatch(r"[\sA-Za-z0-9_{}()[\],.+\-*/^=\\]+", compact)
            and re.search(r"\^|\\(?:binom|sum|prod|lambda|cdot|times)\b|[*/]", compact)
            and re.search(r"[A-Za-z]|\\lambda", compact)
        )
    if shape in {"number", "probability"}:
        return bool(re.search(r"[-+]?\d|\\frac|\\sqrt|\\pi|π|∞|\\infty", value))
    if shape == "proof":
        return _has_proof_support(value) and len(value) >= 24
    return bool(re.search(r"[\w\u4e00-\u9fff\\=+\-*/^]", value))


def _has_proof_support(value: str) -> bool:
    if re.search(
        r"因为|由于|根据|由.*得|由[^。；;\n]{0,60}(?:定义|定理|引理|性质)|"
        r"所以|故|因此|从而|推出|假设|反设|若.*则|矛盾|"
        r"\bby\b[^.\n]{0,80}\b(?:definition|theorem|lemma|property)\b|"
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
    def evaluate(node: ast.AST, depth: int = 0) -> float:
        if depth > 32:
            raise ValueError("numeric expression is too deep")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("unsupported constant")
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand, depth + 1)
            if isinstance(node.op, ast.UAdd):
                value = operand
            elif isinstance(node.op, ast.USub):
                value = -operand
            else:
                raise ValueError("unsupported unary operator")
        elif isinstance(node, ast.BinOp):
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.Mod):
                value = left % right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > 100 or (abs(left) > 1e100 and right > 1):
                    raise OverflowError("power outside numeric-check limits")
                value = left**right
            else:
                raise ValueError("unsupported binary operator")
        else:
            raise ValueError("unsupported syntax")
        if not math.isfinite(value) or abs(value) > 1e300:
            raise OverflowError("non-finite numeric-check result")
        return value

    try:
        return evaluate(tree)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None
