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
    certificate_method: str = ""
    certificate_checks: tuple[str, ...] = ()
    certificate_issues: tuple[str, ...] = ()


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
        _result_covered(semantic_value, goal) and construction_object_present
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
    reasons.extend(_unlabelled_body_reasons(
        value,
        task_kind=task_kind,
        extraction_method=extraction_method,
        explicit_answer=explicit_answer,
    ))
    if _has_false_binomial_identity(value):
        reasons.append("numeric_identity_conflict")
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
            agreement[id(item)],
            sum(item.goal_coverage),
            item.complete_goals,
            item.formatting_valid,
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
