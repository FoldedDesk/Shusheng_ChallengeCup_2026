"""Deterministic candidate validation and selection for submitted answers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
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
    certificate_method: str = ""
    certificate_checks: tuple[str, ...] = ()
    certificate_issues: tuple[str, ...] = ()
    support: str = ""


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
    explicit_result_value = (
        extracted_value.answer
        if extracted_value.valid and extracted_value.answer
        else semantic_value
    )
    formatting_reasons = Finalizer.validate_structure(value)
    contract = getattr(spec, "answer_contract", None)
    proof_contract = getattr(contract, "mode", "") == "proof"
    task_kind = str(
        getattr(spec.profile, "task_kind", "")
        or getattr(spec.profile, "problem_type", "")
    )
    reasoning_required = proof_contract or task_kind in {
        "proof", "derivation", "explanation",
    }
    construction_required = (
        task_kind == "construction"
        or any(goal.kind == "construction" for goal in spec.goals)
        or "construction_validation" in set(getattr(spec, "risk_flags", ()))
    )
    construction_verified = (
        not construction_required
        or _has_construction_verification(semantic_value)
        or _has_self_certifying_construction(semantic_value, spec)
    )
    support_required = reasoning_required or construction_required
    construction_object_present = (
        not construction_required or _has_construction_object(semantic_value)
    )
    result_coverage = tuple(
        _result_covered(semantic_value, goal)
        and all(
            requirement.matches(explicit_result_value)
            for requirement in goal.result_requirements
            if requirement.name.startswith("parameter_dependency_")
        )
        and construction_object_present
        for goal in spec.goals
    )
    support_coverage = tuple(
        _support_covered(
            semantic_value,
            goal,
            task_kind,
            construction_required=False,
            skip_construction_check=True,
        )
        and construction_verified
        for goal in spec.goals
    )
    format_coverage = tuple(_format_covered(semantic_value, goal) for goal in spec.goals)
    coverage = tuple(
        result_ok and format_ok and (support_ok if support_required else True)
        for result_ok, support_ok, format_ok in zip(
            result_coverage, support_coverage, format_coverage
        )
    )
    complete = bool(coverage) and all(coverage)
    reasoning_missing = reasoning_required and not _has_reasoning_support(semantic_value)
    construction_verification_missing = (
        construction_required and not construction_verified
    )
    construction_object_missing = construction_required and not construction_object_present
    if reasoning_missing or construction_verification_missing or construction_object_missing:
        complete = False
    missing_strict = any(
        requirement.strict
        and (
            requirement.category in {"result", "format"}
            or (support_required and requirement.category == "support")
        )
        and not requirement.matches(
            explicit_result_value
            if requirement.name.startswith("parameter_dependency_")
            else semantic_value
        )
        for goal in spec.goals
        for requirement in goal.requirements
    )
    parameter_result_missing = any(
        requirement.name.startswith("parameter_dependency_")
        and not requirement.matches(explicit_result_value)
        for goal in spec.goals
        for requirement in goal.result_requirements
    )
    shape_valid = (
        _valid_shape(semantic_value, spec.profile.answer_shape)
        and _frame_valid(semantic_value, spec)
    )
    tool_status = _tool_status(value, evidence)
    reasons = list(extraction_reasons) + list(formatting_reasons)
    reasons.extend(_unlabelled_body_reasons(
        value,
        task_kind=task_kind,
        extraction_method=extraction_method,
        explicit_answer=explicit_answer,
    ))
    reasons.extend(candidate_consistency_reasons(value, spec))
    if not shape_valid:
        reasons.append("invalid_answer_shape")
    if tool_status == "conflict":
        reasons.append("tool_conflict")
    if reasoning_missing:
        if task_kind == "proof" or proof_contract:
            reasons.append("missing_proof_reasoning")
        else:
            reasons.append("missing_required_reasoning")
    if construction_verification_missing:
        reasons.append("missing_construction_verification")
    if construction_object_missing:
        reasons.append("missing_construction_object")
    if reasoning_missing or construction_verification_missing:
        reasons.append("missing_required_support")
    if parameter_result_missing:
        reasons.append("missing_parameter_dependency")
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
        "final_conclusion_conflict",
        "verification_unresolved", "missing_verification_certificate",
        "unlabelled_process_body", "unlabelled_intermediate_result",
        "unlabelled_future_action", "unlabelled_unfinished_body",
    }
    hard_rejected = bool(set(reasons) & hard_format_reasons) or tool_status == "conflict"
    tier = "rejected" if hard_rejected else (
        "complete" if complete and shape_valid and not missing_required and not formatting_reasons else "degraded"
    )
    score = (12 if tier == "complete" else (2 if tier == "degraded" else -20))
    score += (4 if complete else -2) + (3 if shape_valid else -2) + (3 if not formatting_reasons else -3)
    score += 4 if tool_status == "pass" else (2 if tool_status == "partial_pass" else (-8 if tool_status == "conflict" else 0))
    score += sum(support_coverage)
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
        result_coverage=result_coverage,
        support_coverage=support_coverage,
        format_coverage=format_coverage,
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
            agreement[id(item)],
            sum(item.support_coverage) if agreement[id(item)] > 0 else 0,
            item.score if agreement[id(item)] > 0 else 0,
            -len(item.answer) if agreement[id(item)] > 0 else 0,
            item.verification_verdict == "corrected",
            _source_stage(item.source) == "verify",
            item.tool_status == "partial_pass",
            not item.raw_has_meta,
            item.score,
            item.explicit_answer,
        ),
    )


_REASONING_SIGNAL = re.compile(
    r"(?:因为|由于|依据|根据|利用|所以|故|因此|从而|推出|可得|可知|"
    r"(?<![自理缘])由(?!来)|(?<!原)因(?![子式素变量果])|"
    r"\b(?:because|since|therefore|hence|thus|by|using|implies?|follows?\s+from)\b|"
    r"(?:=>|⇒|⟹|\\implies|\\Rightarrow))",
    re.IGNORECASE,
)

_CONSTRUCTION_VERIFICATION_SIGNAL = re.compile(
    r"(?:满足|验证|检验|代入|检查|符合|可行|成立|逐项核对|"
    r"\b(?:satisf(?:y|ies|ied|ying)|verif(?:y|ies|ied|ication)|"
    r"check(?:s|ed|ing)?|substitut(?:e|es|ed|ing)|meets?|fulfils?|holds?)\b)",
    re.IGNORECASE,
)


def _has_reasoning_support(answer: str) -> bool:
    value = str(answer or "")
    if _REASONING_SIGNAL.search(value):
        return True
    # A displayed chain can be a complete derivation without prose connective.
    equality_steps = len(re.findall(r"(?<![<>!])=(?!=)", value))
    return equality_steps >= 2 or bool(re.search(
        r"(?:=>|⇒|⟹|\\implies|\\Rightarrow)", value,
        re.IGNORECASE,
    ))


def _has_construction_object(answer: str) -> bool:
    value = str(answer or "")
    return bool(re.search(
        r"(?:(?:^|[\s，,；;。:：])(?:取|令|设|选取|例如)\s*\S|"
        r"(?:构造|对象|例子|反例)\s*(?:为|是|如下|[:：])\s*\S|"
        r"\b(?:take|let|choose|define|set)\s+\S|"
        r"\b(?:construction|object|example|counterexample)\s*(?:is|:|=)\s*\S|"
        r"(?<![<>!])=(?!=)|\\?\{|\\?\[)",
        value,
        re.IGNORECASE,
    ))


def _has_construction_verification(answer: str) -> bool:
    value = str(answer or "")
    if _CONSTRUCTION_VERIFICATION_SIGNAL.search(value):
        return True

    # Certified constructions often verify several requested properties by
    # direct equalities/inequalities rather than by using the word "verify".
    relation_count = len(re.findall(
        r"(?<![<>!])=(?!=)|(?:\\(?:neq|geq|leq|to)|≠|≥|≤|→|<|>)",
        value,
        re.IGNORECASE,
    ))
    has_checking_clause = bool(re.search(
        r"(?:[；;。.]\s*(?:则|它|其|该|此)|(?:，|,)\s*(?:则|且|故|从而)|"
        r"\b(?:then|and|so|hence|therefore|it|they|this)\b)",
        value,
        re.IGNORECASE,
    ))
    return relation_count >= 2 and has_checking_clause


def _has_self_certifying_construction(answer: str, spec) -> bool:
    """Accept an explicitly listed finite family as the construction itself."""
    problem = getattr(spec, "problem_text", "")
    if not re.search(
        r"(?:family|blocks?|set system|族|区组).{0,80}(?:explicit|明确|列出)|"
        r"(?:give|list|写出|给出).{0,40}(?:family|blocks?|族|区组).{0,40}(?:explicit|明确)?",
        problem,
        re.IGNORECASE | re.DOTALL,
    ):
        return False
    normalized = str(answer or "").replace(r"\{", "{").replace(r"\}", "}")
    members = re.findall(r"\{[^{}]+\}", normalized)
    return len(members) >= 2


def _unlabelled_body_reasons(
    answer: str,
    *,
    task_kind: str,
    extraction_method: str,
    explicit_answer: bool,
) -> tuple[str, ...]:
    """Reject unfinished scratch prose that escaped marker extraction.

    Proof-like tasks legitimately return a short argument body. Non-proof
    tasks are prompted to label the result, so an unlabelled process or
    intermediate-value body is not a reliable final answer.
    """
    if extraction_method != "whole_response" or explicit_answer:
        return ()

    value = str(answer or "").strip()
    reasons: list[str] = []
    future_action = re.search(
        r"(?:接下来|下一步|随后(?:还|再)?(?:需要|应当|将要)?|"
        r"还需|仍需|尚需|需要继续|继续(?:分析|计算|求解|推导|检查)|待(?:计算|求解|验证)|"
        r"\bnext\s*(?:step\b|,)|\b(?:will\s+continue|need(?:s)?\s+to\s+continue|"
        r"remain(?:s)?\s+to\s+be|to\s+be\s+continued|will\s+(?:now\s+)?"
        r"(?:calculate|analy[sz]e|derive|solve|check))\b)",
        value,
        re.IGNORECASE,
    )
    if future_action:
        reasons.append("unlabelled_future_action")

    theoretical_body = task_kind in {"proof", "derivation", "explanation"}
    if not theoretical_body:
        lines = [line.rstrip() for line in value.splitlines() if line.strip()]
        if len(lines) > 1:
            # Non-proof stages are required to mark their final answer.  A
            # multi-line unlabelled body is indistinguishable from an
            # unfinished scratchpad, even when it happens to contain numbers.
            reasons.append("unlabelled_process_body")
        if re.search(
            r"(?im)^\s*(?:(?:计算|求解|推导|分析|证明)\s*)?过程(?:如下|为)?\s*[:：]?|"
            r"^\s*(?:solution|calculation|derivation|analysis)\s+(?:process|steps?)\s*[:：]",
            value,
            re.IGNORECASE,
        ):
            reasons.append("unlabelled_process_body")
        if re.search(
            r"(?:中间(?:值|量|结果|步骤)|暂(?:时)?得到|目前(?:得到|算得)|"
            r"尚未(?:完成|求得)|"
            r"\b(?:intermediate\s+(?:value|quantity|result|step)|partial\s+result|"
            r"work\s+in\s+progress|not\s+yet\s+(?:complete|finished|solved))\b)",
            value,
            re.IGNORECASE,
        ):
            reasons.append("unlabelled_intermediate_result")

    lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    if len(lines) > 1 and re.search(
        r"(?:[:：,，;；、=+\-*/（({\[]|以及|并且|然后|because|since|and|then)\s*$",
        lines[-1],
        re.IGNORECASE,
    ):
        reasons.append("unlabelled_unfinished_body")
    return tuple(dict.fromkeys(reasons))


def _requirements_covered(answer: str, requirements) -> bool:
    return all(requirement.matches(answer) for requirement in requirements)


def _result_covered(answer: str, goal) -> bool:
    if not answer:
        return False
    compact = _compact(answer)
    if goal.result_requirements:
        return _requirements_covered(answer, goal.result_requirements)
    if goal.required_terms:
        return all(_compact(term) in compact for term in goal.required_terms)
    if goal.answer_shape == "number":
        numeric = bool(re.search(
            r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
            r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|"
            r"(?:π|\\pi|∞|\\infty)",
            answer,
        ))
        symbolic = bool(
            re.search(r"(?:\\[A-Za-z]+|[A-Za-z])", answer)
            and re.search(r"[\^_{}+*/]", answer)
            and not re.search(r"(?i)\b(?:because|therefore|answer|unknown)\b", answer)
        )
        return numeric or symbolic
    if goal.kind == "proof":
        # The support contract separately checks the deduction.  The result
        # side only needs a non-empty mathematical conclusion, in either
        # language.
        return bool(re.search(r"[\w\u4e00-\u9fff]", answer))
    if goal.kind == "truth_judgement":
        return bool(re.search(r"(?:是|否|正确|错误|成立|不成立|收敛|发散|可约|不可约)", answer))
    if goal.kind == "domain_or_interval":
        return bool(re.search(r"(?:区间|定义域|[\[(][^\n,，]+[,，][^\n)\]]+[)\]])", answer))
    if goal.kind == "formula":
        return bool(re.search(
            r"[=+\-*/^]|\\(?:frac|sum|int|sqrt|begin|operatorname\s*\{diag\}|mathrm\s*\{diag\})",
            answer,
        ))
    if goal.kind == "comparison":
        return len(re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", answer)) >= 2 or bool(
            re.search(r"(?:大于|小于|相等|误差|放大|不变|趋于)", answer)
        )
    if goal.kind == "construction":
        return bool(re.search(r"(?:取|令|构造|例如|=|\{|\[)", answer))
    return bool(re.search(r"[\w\u4e00-\u9fff=+\-*/^\\]", answer))


def _support_covered(
    answer: str,
    goal,
    task_kind: str = "",
    *,
    construction_required: bool = False,
    skip_construction_check: bool = False,
) -> bool:
    if goal.support_requirements and not _requirements_covered(
        answer, goal.support_requirements
    ):
        return False
    checks = []
    if task_kind in {"proof", "derivation", "explanation"} or goal.kind == "proof":
        checks.append(_has_reasoning_support(answer))
    if not skip_construction_check and (
        construction_required or task_kind == "construction" or goal.kind == "construction"
    ):
        checks.append(_has_construction_verification(answer))
    return all(checks) if checks else True


def _format_covered(answer: str, goal) -> bool:
    if not goal.format_requirements:
        return True
    return _requirements_covered(answer, goal.format_requirements)


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
        "audit_retry": "verify",
        "last_chance": "rescue",
    }.get(raw, raw)


def _tool_status(answer: str, evidence: tuple[ToolEvidence, ...]) -> str:
    whole = [item for item in evidence if item.scope == "whole_goal" and item.verified]
    normalized = _compact(answer)
    for item in whole:
        expected = _compact(item.result)
        same_tool_payload = (
            normalized.replace("∞", "infty")
            == expected.replace("∞", "infty")
        )
        if expected and (
            expected == normalized
            or same_tool_payload
            or equivalent_answers(answer, item.result)
        ):
            return "pass"
    if whole:
        return "conflict"
    partial = [item for item in evidence if item.scope == "subexpression" and item.verified]
    for item in partial:
        if item.operation != "lz78_encoding_check":
            continue
        lz78_status = _lz78_fixed_width_status(answer, item.result)
        if lz78_status:
            return lz78_status
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


def _lz78_fixed_width_status(answer: str, evidence: str) -> str:
    """Validate a model bit string against its stated standard LZ78 phrases.

    A generic "Lempel-Ziv" prompt may omit the index-width convention, so all
    feasible fixed widths are accepted.  A candidate is rejected only when it
    uses the exact verified phrase sequence but its bits cannot encode the
    corresponding (prefix-index, symbol) pairs under any fixed width.
    """
    evidence_text = str(evidence or "")
    phrase_match = re.search(r"Phrases:\s*([^;]+);", evidence_text, re.IGNORECASE)
    pair_match = re.search(r"pairs:\s*([^;]+);", evidence_text, re.IGNORECASE)
    chunk_match = re.search(
        r"candidate encoded string:\s*([01]+(?:\s+[01]+)+)",
        evidence_text,
        re.IGNORECASE,
    )
    if not (phrase_match and pair_match and chunk_match):
        return ""

    expected_phrases = tuple(
        part.strip() for part in phrase_match.group(1).split(",") if part.strip()
    )
    pairs = tuple(
        (int(index), symbol)
        for index, symbol in re.findall(
            r"\(\s*(\d+)\s*,\s*([A-Za-z0-9])\s*\)",
            pair_match.group(1),
        )
    )
    reference_chunks = tuple(chunk_match.group(1).split())
    if not expected_phrases or len(pairs) != len(expected_phrases):
        return ""
    if len(reference_chunks) != len(pairs) or len({len(chunk) for chunk in reference_chunks}) != 1:
        return ""

    extracted = Finalizer.extract_result(str(answer or ""))
    candidate = extracted.answer if extracted.valid and extracted.answer else str(answer or "")
    before_bits = candidate.split(";", 1)[0]
    before_bits = re.sub(
        r"^(?:phrases?|短语(?:分解)?)\s*[:：]?\s*",
        "",
        before_bits.strip(),
        flags=re.IGNORECASE,
    )
    candidate_phrases = tuple(
        part.strip(" `'$\\{}")
        for part in re.split(r"[,，]", before_bits)
        if part.strip(" `'$\\{}")
    )
    if candidate_phrases != expected_phrases:
        return ""

    bit_matches = re.findall(r"(?<![01])(?:[01][01\s]{8,}[01])(?![01])", candidate)
    if not bit_matches:
        return ""
    candidate_bits = re.sub(r"\s+", "", bit_matches[-1])
    if not candidate_bits or len(candidate_bits) % len(pairs):
        return "conflict"

    reference_width = len(reference_chunks[0])
    symbol_codes: dict[str, str] = {}
    letter_width = 0
    for index_width in range(1, reference_width):
        codes: dict[str, str] = {}
        valid = True
        for (index, symbol), chunk in zip(pairs, reference_chunks):
            if int(chunk[:index_width], 2) != index:
                valid = False
                break
            code = chunk[index_width:]
            if symbol in codes and codes[symbol] != code:
                valid = False
                break
            codes[symbol] = code
        if valid and len(set(codes.values())) == len(codes):
            symbol_codes = codes
            letter_width = reference_width - index_width
            break
    if not symbol_codes or letter_width <= 0:
        return ""

    chunk_width = len(candidate_bits) // len(pairs)
    index_width = chunk_width - letter_width
    minimum_width = max(1, max(index for index, _ in pairs).bit_length())
    if index_width < minimum_width:
        return "conflict"
    expected_bits = "".join(
        f"{index:0{index_width}b}{symbol_codes[symbol]}"
        for index, symbol in pairs
    )
    return "partial_pass" if candidate_bits == expected_bits else "conflict"


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
            r"属于|不属于|可导|不可导|不可复可导|调和|不调和|为解|不是解|"
            r"改变|不变|变化|"
            r"位于[^。；;]*(?:内|外)|"
            r"\b(?:yes|no|true|false|inside|outside|is\s+a\s+solution|"
            r"is\s+not\s+a\s+solution|changes?|unchanged)\b)"
        )
        judgement = re.search(verdict_pattern, answer)
        if not judgement:
            return False
        bare = re.fullmatch(verdict_pattern, _compact(answer))
        return not bare or not frame.subject or _compact(frame.subject) in _compact(answer)
    return True


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")


def candidate_consistency_reasons(answer: str, spec=None) -> tuple[str, ...]:
    """Return hard contradictions found inside one model candidate.

    The checks are intentionally closed-world.  Arithmetic is evaluated only
    when both sides of ``=`` contain numeric literals and elementary
    operators.  Conclusion comparison requires an explicit final label, or a
    single standalone box for a single-goal problem, plus comparable targets.
    """
    value = str(answer or "")
    reasons: list[str] = []
    problem_context = " ".join((
        str(getattr(spec, "problem_text", "") or ""),
        *(str(getattr(goal, "instruction", "") or "") for goal in getattr(spec, "goals", ())),
    ))
    boolean_algebra = bool(re.search(
        r"布尔代数|Boolean\s+algebra",
        problem_context,
        re.IGNORECASE,
    ))
    if not boolean_algebra and (
        _has_false_numeric_identity(value) or _has_false_binomial_identity(value)
    ):
        reasons.append("numeric_identity_conflict")

    labelled = [
        result.answer
        for result in Finalizer.extract_explicit_results(value)
        if result.valid
        and result.answer
        and result.method in {"label", "label_boxed", "bracket_label"}
    ]
    for index, first in enumerate(labelled):
        if any(_conclusions_conflict(first, other) for other in labelled[index + 1:]):
            reasons.append("final_conclusion_conflict")
            break

    terminal = _last_terminal_conclusion(value)
    if terminal:
        explicit_finals = list(labelled)
        if not explicit_finals:
            standalone = _single_standalone_box(value, spec)
            if standalone:
                explicit_finals.append(standalone)
        if any(_conclusions_conflict(item, terminal) for item in explicit_finals):
            reasons.append("final_conclusion_conflict")
    concluding_box = _last_concluding_box(value)
    if concluding_box and any(
        _conclusions_conflict(item, concluding_box) for item in labelled
    ):
        reasons.append("final_conclusion_conflict")
    return tuple(dict.fromkeys(reasons))


def _numeric_plain_text(value: str) -> str:
    text = str(value or "").replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\(?:cdot|times)", "*", text)
    text = re.sub(r"\\div", "/", text)
    fraction = re.compile(
        r"\\(?:d?frac)\s*\{\s*([0-9.\s()+\-*/^]+)\s*\}"
        r"\s*\{\s*([0-9.\s()+\-*/^]+)\s*\}"
    )
    for _ in range(6):
        updated = fraction.sub(r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    text = text.replace(r"\(", " ").replace(r"\)", " ")
    text = text.replace(r"\[", " ").replace(r"\]", " ").replace("$", " ")
    text = text.replace("{", "(").replace("}", ")")
    return re.sub(r"(?<!\*)\^(?!\*)", "**", text)


def _eval_numeric_expression(expression: str) -> Fraction | None:
    value = str(expression or "").strip().rstrip(".").strip()
    if (
        not value
        or len(value) > 120
        or not re.search(r"\d", value)
        or not re.fullmatch(r"[0-9.\s()+\-*/]+", value)
    ):
        return None
    try:
        root = ast.parse(value, mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def evaluate(node) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported numeric syntax")
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 20:
                raise ValueError("unsupported exponent")
            if left == 0 and right.numerator < 0:
                raise ZeroDivisionError
            return left ** right.numerator
        raise ValueError("unsupported numeric operator")

    try:
        return evaluate(root)
    except (ArithmeticError, OverflowError, ValueError):
        return None


def _numeric_rounding_tolerance(left: str, right: str) -> Fraction:
    decimal_places = [
        len(match.group(1))
        for match in re.finditer(r"(?<![\d.])\d+\.(\d+)", f"{left} {right}")
    ]
    if not decimal_places:
        return Fraction(0)
    # Treat the least precise displayed decimal as a possible rounded value.
    # This guard is for clear contradictions, not for policing notation such
    # as 1/3=0.333 or 0.3333=0.333 in a numerical derivation.
    return Fraction(1, 2 * (10 ** min(decimal_places)))


def _has_false_numeric_identity(answer: str) -> bool:
    """Check only closed elementary equalities, never variable assignments."""
    value = _numeric_plain_text(answer)
    for match in re.finditer(r"(?<![<>=!])=(?!=)", value):
        left_tail = re.search(r"([0-9.\s()+\-*/]+)$", value[: match.start()])
        right_head = re.match(r"([0-9.\s()+\-*/]+)", value[match.end() :])
        if not left_tail or not right_head:
            continue
        prefix = value[: left_tail.start(1)].rstrip()
        if prefix and (
            prefix[-1].isalnum()
            or prefix[-1] in "\\_,!)]}'\u2032\u2033"
        ):
            # The trailing digits in x_0=1, C(9,3), binom(9)(3), f(2),
            # f'(0), or 9! are identifiers/arguments, not a closed numeric
            # left-hand side.
            continue
        left_text = left_tail.group(1).strip()
        right_text = right_head.group(1).strip().rstrip(". ")
        suffix = value[match.end() + right_head.end(1):].lstrip()
        if suffix and (suffix[0].isalnum() or suffix[0] in r"\_"):
            # In 12*5=2E or 2=2*pi, the numeric prefix is a coefficient,
            # not the complete right-hand side.
            continue
        left_value = _eval_numeric_expression(left_text)
        right_value = _eval_numeric_expression(right_text)
        if left_value is None or right_value is None:
            continue
        tolerance = _numeric_rounding_tolerance(left_text, right_text)
        if abs(left_value - right_value) > tolerance:
            return True
    return False


def _strip_final_wrapper(value: str) -> str:
    text = str(value or "").strip().strip("` ")
    boxed = Finalizer._last_boxed(text)
    if boxed:
        text = boxed
    text = re.sub(
        r"^\s*(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is|equals|[:=])|"
        r"(?:最终)?答案\s*(?:为|是|[:：=])|(?:结论|结果)\s*(?:为|是|[:：=]))\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip().rstrip("。.!！?？；;").strip()


def _assignment_values(value: str) -> dict[str, tuple[str, ...]]:
    text = _strip_final_wrapper(value)
    text = re.sub(
        r"\s+(?:and|且|以及)\s+(?=[A-Za-z\\][A-Za-z0-9_{}\\()]*\s*=)",
        ",",
        text,
        flags=re.IGNORECASE,
    )
    assignments: dict[str, list[str]] = {}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"((?:E|P|Pr|Var|Cov)\s*(?:\[[^,，;\n=]+\]|\([^,，;\n=]*\))"
        r"|\\operatorname\{(?:Var|Cov)\}\s*\([^,，;\n=]*\)"
        r"|[A-Za-z](?:_[{]?[A-Za-z0-9+\-]+[}]?)?(?:\([^,，;\n=]*\))?"
        r"|\\[A-Za-z]+(?:_[{][^{}]+[}])?)"
        r"\s*=\s*([^,，;\n]+)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        target = re.sub(r"[\s{}\\]", "", match.group(1)).lower()
        result = _strip_final_wrapper(match.group(2))
        if target and result and "=" not in result:
            assignments.setdefault(target, []).append(result)
    return {key: tuple(values) for key, values in assignments.items()}


def _simple_final_atom(value: str) -> bool:
    text = _strip_final_wrapper(value)
    if not text or len(text) > 160:
        return False
    if _eval_numeric_expression(_numeric_plain_text(text)) is not None:
        return True
    if re.fullmatch(
        r"(?:[A-E](?:\s*[,，、]\s*[A-E])*|正确|错误|是|否|成立|不成立|"
        r"无解|不存在|no\s+solutions?|true|false)",
        text,
        re.IGNORECASE,
    ):
        return True
    if not re.search(r"[0-9=+\-*/^<>≤≥∈\\]", text):
        return False
    if re.search(r"\b(?:because|since|therefore|hence|thus)\b|因为|由于|因此|所以", text, re.IGNORECASE):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_{}()[\].,+\-*/^=<>≤≥∈\\\s]+", text))


def _same_conclusion_value(first: str, second: str) -> bool:
    return bool(
        _compact(first) == _compact(second)
        or equivalent_answers(first, second)
    )


def _closed_numeric_value(value: str) -> str:
    """Return a standalone numeric value, including a chain's final RHS.

    This deliberately excludes symbolic assignments.  It is used only by the
    contradiction guard, where a false positive is more damaging than failing
    to notice an unusual prose contradiction.
    """
    text = _strip_final_wrapper(value)
    if not text:
        return ""
    parts = re.split(r"(?<![<>=!])=(?!=)", text)
    candidate = _strip_final_wrapper(parts[-1])
    # Keep LaTeX braces intact: stripping ``{}`` would turn ``\frac{1}{8}``
    # into an invalid fragment before the numeric parser sees it.
    candidate = candidate.strip().strip("$ `").rstrip(".,;:，。；：").strip()
    if not candidate:
        return ""
    return candidate if _eval_numeric_expression(_numeric_plain_text(candidate)) is not None else ""


def _closed_categorical_value(value: str) -> str:
    candidate = _strip_final_wrapper(value).strip()
    if re.fullmatch(
        r"(?:[A-E]|正确|错误|是|否|成立|不成立|true|false|yes|no)",
        candidate,
        re.IGNORECASE,
    ):
        return candidate.lower()
    return ""


def _assignment_target_conflicts(
    first_values: tuple[str, ...], second_values: tuple[str, ...]
) -> bool:
    comparable = [
        (left, right)
        for left in first_values
        for right in second_values
        if _simple_final_atom(left) and _simple_final_atom(right)
    ]
    return bool(comparable) and not any(
        _same_conclusion_value(left, right) for left, right in comparable
    )


def _conclusions_conflict(first: str, second: str) -> bool:
    first_value = _strip_final_wrapper(first)
    second_value = _strip_final_wrapper(second)
    if not first_value or not second_value or _same_conclusion_value(first_value, second_value):
        return False
    first_assignments = _assignment_values(first_value)
    second_assignments = _assignment_values(second_value)
    overlap = set(first_assignments) & set(second_assignments)
    if overlap:
        return any(
            _assignment_target_conflicts(
                first_assignments[target], second_assignments[target]
            )
            for target in overlap
        )
    if first_assignments and second_assignments:
        return False
    if (
        len(first_assignments) == 1
        and not second_assignments
        and "=" not in second_value
        and _closed_numeric_value(second_value)
    ):
        only_values = next(iter(first_assignments.values()))
        numeric_values = tuple(filter(None, map(_closed_numeric_value, only_values)))
        return bool(numeric_values) and not any(
            _same_conclusion_value(item, second_value) for item in numeric_values
        )
    if (
        len(second_assignments) == 1
        and not first_assignments
        and "=" not in first_value
        and _closed_numeric_value(first_value)
    ):
        only_values = next(iter(second_assignments.values()))
        numeric_values = tuple(filter(None, map(_closed_numeric_value, only_values)))
        return bool(numeric_values) and not any(
            _same_conclusion_value(first_value, item) for item in numeric_values
        )

    first_numeric = _closed_numeric_value(first_value)
    second_numeric = _closed_numeric_value(second_value)
    if (
        not first_assignments
        and not second_assignments
        and first_numeric
        and second_numeric
    ):
        return not _same_conclusion_value(first_numeric, second_numeric)
    first_category = _closed_categorical_value(first_value)
    second_category = _closed_categorical_value(second_value)
    return bool(
        not first_assignments
        and not second_assignments
        and first_category
        and second_category
        and first_category != second_category
    )


def _last_terminal_conclusion(answer: str) -> str:
    pattern = re.compile(
        r"(?:^|[\n，,。.!?！？；;])\s*"
        r"(?:因此|所以|故而?|综上(?:所述)?|从而|可得|therefore\b|hence\b|thus\b)"
        r"[ \t]*[,，:：]?[ \t]*([^\n。！？；;]+)",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(str(answer or "")))
    if not matches:
        return ""
    value = matches[-1].group(1).strip()
    value = re.split(r"\.(?=\s+[A-Z\u4e00-\u9fff])", value, maxsplit=1)[0]
    value = re.sub(r"^(?:有|得到|可知|知)\s*", "", value)
    stated_value = re.search(
        r"(?:答案|结论|结果|数量|个数|置换数|顶点数|边数|概率|近似值|精确值)"
        r"\s*(?:为|是|等于|=)\s*(.+)$|"
        r"(?:answer|result|number|count|probability|value)"
        r"\s*(?:is|equals|=|:)\s*(.+)$",
        value,
        re.IGNORECASE,
    )
    if stated_value:
        value = next(group for group in stated_value.groups() if group is not None).strip()
    if re.search(
        r"(?:中间(?:量|结果|步骤)?|校验量|检验量|辅助量|上界|下界|"
        r"\b(?:intermediate|check\s+value|auxiliary\s+value|upper\s+bound|lower\s+bound)\b)",
        value,
        re.IGNORECASE,
    ):
        return ""
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", value):
        # A connective followed by a display opener ("所以：\n\\[")
        # introduces the next calculation; it is not a terminal conclusion.
        return ""
    return _strip_final_wrapper(value)


def _last_concluding_box(answer: str) -> str:
    """Return a terminal correction box, excluding ordinary check boxes."""
    text = str(answer or "")
    boxes = [item for item in Finalizer._boxed_values(text) if item[2]]
    labelled_before_box = bool(re.search(
        r"(?im)^\s*(?:FINAL(?:\s+ANSWER)?|(?:最终\s*)?答案|结论)\s*[:：=]",
        text[: boxes[-1][0]] if boxes else "",
    ))
    if not boxes or (len(boxes) < 2 and not labelled_before_box):
        return ""
    position, result, _ = boxes[-1]
    marker = re.search(r"\\boxed\s*\{", text[position:])
    if not marker:
        return ""
    opening = position + marker.end() - 1
    depth = 0
    closing = -1
    for index in range(opening, len(text)):
        character = text[index]
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                closing = index + 1
                break
    if closing < 0 or not re.fullmatch(
        r"[\s$\\\[\].。!！]*", text[closing:]
    ):
        return ""
    prefix = text[:position]
    # Only the immediately governing clause can turn a later box into a
    # correction.  A generic mention several sentences earlier commonly
    # introduces an intermediate/check value instead.
    clauses = []
    for clause in re.split(r"[\n。！？!?；;]", prefix):
        cleaned_clause = clause.strip()
        if not cleaned_clause or re.fullmatch(r"[\s$\\\[\]()]*", cleaned_clause):
            continue
        clauses.append(cleaned_clause)
    context = clauses[-1][-180:] if clauses else ""
    if re.search(
        r"(?:中间(?:量|结果|步骤)?|校验量|检验量|辅助量|上界|下界|"
        r"\b(?:intermediate|check\s+value|auxiliary\s+value|upper\s+bound|lower\s+bound)\b)",
        context,
        re.IGNORECASE,
    ):
        return ""
    correction_marker = re.search(
        r"(?:修正(?:后|答案|结论)?|更正(?:后|答案|结论)?|"
        r"正确(?:答案|结论)\s*(?:为|是|=|[:：])|"
        r"重算(?:后|得)|复算(?:后|得)|重新(?:计算|核验)后|核验发现[^。\n]{0,40}错误|"
        r"\b(?:corrected\s+answer|correction|rechecking\s+(?:shows|gives|finds)|"
        r"after\s+recomput(?:ing|ation))\b)",
        context,
        re.IGNORECASE,
    )
    # A later boxed value may be a check, bound, or intermediate result.  It
    # supersedes an explicit FINAL only when the prose explicitly says this is
    # a correction; generic connectives such as "therefore" are insufficient.
    if not correction_marker:
        return ""
    return result


def _single_standalone_box(answer: str, spec=None) -> str:
    boxes = [item for item in Finalizer._boxed_values(str(answer or "")) if item[2]]
    goal_count = len(getattr(spec, "goals", ())) if spec is not None else 1
    if len(boxes) != 1 or goal_count != 1:
        return ""
    position, result, _ = boxes[0]
    text = str(answer or "")
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    line = text[line_start : len(text) if line_end < 0 else line_end].strip()
    if not re.fullmatch(
        r"(?:\\\[|\$\$?)?\s*\\boxed\{.*\}\s*(?:\\\]|\$\$?)?[。.!]?",
        line,
        re.DOTALL,
    ):
        return ""
    return result


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
            prefix = value[: match.start()].rstrip()
            if prefix and re.search(
                r"(?:\\(?:cdot|times)|[+\-*×·/−]|[0-9)}\]])$",
                prefix,
                re.IGNORECASE,
            ):
                # In products such as binom(5,2)binom(4,1)=40, the
                # matched final factor is not asserted to equal the rhs.
                continue
            n, k, stated = map(int, match.groups())
            expected = math.comb(n, k) if 0 <= k <= n else 0
            if stated != expected:
                return True
    return False
