"""Accuracy-first, bounded orchestration for unseen mathematics problems."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
from time import monotonic
from typing import Iterable

from classifier.problem_spec import ProblemSpec, build_problem_spec
from core.execution_limits import SUBMISSION_SOFT_ITEM_SECONDS
from core.model_response import ModelCallResult, coerce_model_response
from core.runtime_failure import is_recoverable_runtime_failure
from core.stage_budget import StageBudget, plan_stage_budget
from rag.card_retriever import CardRetriever, RetrievalBundle
from reasoning.candidate_selector import (
    CandidateAssessment,
    CheckResult,
    ToolEvidence,
    assess_candidate,
    choose_candidate,
)
from reasoning.finalizer import ExtractionResult, Finalizer
from reasoning.math_equivalence import equivalent_answers
from tools.deterministic_math_tool import DeterministicMathTool
from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, problem_fingerprint


class SubmissionAgent:
    """Solve one statement without relying on metadata, order, or answer keys."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.deterministic = DeterministicMathTool()
        self.retriever = CardRetriever()
        self.prompt = self._load_prompt()

    def solve(self, problem: str, metadata: dict) -> dict:
        del metadata
        started_at = monotonic()
        statement = str(problem or "").strip()
        spec = build_problem_spec(statement)
        cards = self.retriever.retrieve(spec)
        raw_tool_results = tuple((
            *self.sympy.results_for(statement),
            *self.deterministic.results_for(statement),
        ))
        whole_tool = self._whole_tool_result(raw_tool_results, spec)
        evidence = self._tool_evidence(raw_tool_results, spec, whole_tool)
        deep_reasoning = self._deep_reasoning(spec)
        budget = plan_stage_budget(
            spec,
            has_whole_tool_answer=whole_tool is not None,
            deep_reasoning=deep_reasoning,
        )
        trace: list[dict] = [
            {"step": "blueprint", "content": spec.trace_content()},
            {"step": "retrieval", "content": cards.trace_content()},
            {"step": "tools", "content": self._tool_trace(raw_tool_results, whole_tool)},
            {"step": "budget", "content": budget.trace_content()},
        ]

        if whole_tool is not None:
            direct = self._assess_value(
                whole_tool.result,
                source="sympy_verified",
                spec=spec,
                evidence=evidence,
                extraction_method="certified_tool",
                explicit=True,
                method_id=whole_tool.operation,
                independence_group="local_tool",
            )
            if direct.validation_tier == "complete" and direct.tool_status == "pass":
                answer = self._render_submission(direct.answer, spec)
                trace.append({
                    "step": "selection",
                    "content": self._candidate_trace(direct, route="certified_tool"),
                })
                return {"final_response": answer, "trace": trace}
            trace.append({
                "step": "tool_fallback",
                "content": {"reason": "candidate_contract_rejected"},
            })
            evidence = tuple(
                ToolEvidence(
                    result=item.result,
                    scope="goal_result" if item.scope == "whole_goal" else item.scope,
                    operation=item.operation,
                    verified=item.verified,
                    certificate_method=item.certificate_method,
                    certificate_checks=item.certificate_checks,
                    certificate_issues=tuple((*item.certificate_issues, "whole_route_rejected")),
                    support=item.support,
                )
                for item in evidence
            )
            budget = plan_stage_budget(spec, False, deep_reasoning=deep_reasoning)

        supported_tool = self._supported_tool_candidate(
            raw_tool_results,
            spec,
            evidence,
        )
        if supported_tool is not None:
            answer = self._render_submission(supported_tool.answer, spec)
            trace.append({
                "step": "selection",
                "content": self._candidate_trace(
                    supported_tool,
                    route="certified_tool_with_support",
                    model_calls=0,
                ),
            })
            return {"final_response": answer, "trace": trace}

        call_count = 0
        candidates: list[CandidateAssessment] = []
        first_raw, first_result = self._call(
            self._primary_request(statement, spec, cards, evidence),
            stage="primary",
            max_tokens=budget.solve_tokens,
            temperature=0.2,
            thinking_mode=deep_reasoning,
            trace=trace,
        )
        call_count += 1
        first_candidates = self._assess_response(
            first_raw,
            source="primary",
            spec=spec,
            evidence=evidence,
            method_id=spec.primary_method,
            independence_group="model_a",
        )
        first_truncated = self._truncated(first_result, first_raw)
        first_candidates = self._transport_admissible(first_candidates, first_truncated)
        candidates.extend(first_candidates)
        first_best = choose_candidate(first_candidates)
        first_usable = self._complete_after_transport(first_best, first_truncated)

        need_independent = bool(
            budget.allow_review
            and (
                budget.require_independent_review
                or not first_usable
                or (first_truncated and not self._objectively_checked(first_best))
            )
        )
        trace.append({
            "step": "review_admission",
            "content": {
                "admitted": need_independent,
                "first_usable": first_usable,
                "first_truncated": first_truncated,
                "reason": self._review_reason(budget, first_usable, first_truncated),
            },
        })

        second_raw = ""
        second_result = ModelCallResult("")
        second_candidates: list[CandidateAssessment] = []
        if (
            need_independent
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at) >= budget.review_min_remaining_seconds
        ):
            second_raw, second_result = self._call(
                self._independent_request(statement, spec, cards, evidence),
                stage="independent",
                max_tokens=budget.review_tokens,
                temperature=0.2,
                thinking_mode=False,
                trace=trace,
            )
            call_count += 1
            second_candidates = self._assess_response(
                second_raw,
                source="independent",
                spec=spec,
                evidence=evidence,
                method_id=spec.alternative_method,
                independence_group="model_b",
            )
        second_truncated = self._truncated(second_result, second_raw) if second_raw else False
        second_candidates = self._transport_admissible(second_candidates, second_truncated)
        candidates.extend(second_candidates)

        first_best = choose_candidate(first_candidates)
        second_best = choose_candidate(second_candidates)
        conflict = self._conflict(first_best, second_best, spec)
        second_usable = self._complete_after_transport(second_best, second_truncated)
        no_complete = not (first_usable or second_usable)
        needs_recovery = bool(
            no_complete
            or (budget.require_independent_review and need_independent and not second_usable)
            or (
                budget.require_independent_review
                and first_truncated
                and not first_usable
            )
        )
        trace.append({
            "step": "cross_check",
            "content": {
                "independent_available": bool(second_raw),
                "second_usable": second_usable,
                "second_truncated": second_truncated,
                "conflict": conflict,
                "needs_recovery": needs_recovery,
            },
        })

        adjudicated: CandidateAssessment | None = None
        decision = ""
        if (
            budget.allow_repair
            and call_count < budget.max_calls
            and (conflict or needs_recovery)
            and self._remaining_seconds(started_at) >= budget.repair_min_remaining_seconds
        ):
            if conflict and first_best is not None and second_best is not None:
                third_request = self._arbitration_request(
                    statement, spec, evidence, first_best, second_best
                )
                stage = "arbitration"
                third_tokens = budget.repair_tokens
                third_thinking = False
            else:
                third_request = self._recovery_request(
                    statement,
                    spec,
                    cards,
                    evidence,
                    drafts=(first_raw, second_raw),
                )
                stage = "recovery"
                third_tokens = budget.emergency_tokens
                third_thinking = False
            third_raw, third_result = self._call(
                third_request,
                stage=stage,
                max_tokens=third_tokens,
                temperature=0.1,
                thinking_mode=third_thinking,
                trace=trace,
            )
            call_count += 1
            third_candidates = self._assess_response(
                third_raw,
                source=stage,
                spec=spec,
                evidence=evidence,
                method_id=stage,
                independence_group="model_c",
            )
            third_candidates = self._transport_admissible(
                third_candidates,
                self._truncated(third_result, third_raw),
            )
            if stage == "arbitration":
                decision, adjudicated = self._apply_arbitration(
                    third_raw,
                    third_result,
                    third_candidates,
                    first_best,
                    second_best,
                    spec,
                )
            else:
                candidates.extend(third_candidates)

        trace.append({
            "step": "candidate_audit",
            "content": [self._candidate_trace(item) for item in candidates[:12]],
        })
        complete_candidates = [
            item for item in candidates if item.validation_tier == "complete"
        ]
        selected = adjudicated or choose_candidate(complete_candidates)
        if selected is None:
            selected = self._best_degraded(candidates, spec)
        certified_fallback = False
        if selected is None:
            selected = self._certified_goal_fallback(raw_tool_results, spec, evidence)
            certified_fallback = selected is not None
        if selected is None:
            answer = self._shape_fallback(spec)
            trace.append({
                "step": "selection",
                "content": {
                    "route": "shape_fallback",
                    "model_calls": call_count,
                    "reason": "no_structurally_usable_candidate",
                },
            })
        else:
            answer = self._render_submission(selected.answer, spec)
            trace.append({
                "step": "selection",
                "content": self._candidate_trace(
                    selected,
                    route=(
                        "certified_goal_fallback"
                        if certified_fallback
                        else "arbitrated" if adjudicated else "ranked_candidates"
                    ),
                    model_calls=call_count,
                    arbitration_decision=decision,
                ),
            })
        if not answer.strip() or Finalizer.validate_structure(answer):
            answer = self._shape_fallback(spec)
            trace.append({
                "step": "final_guard",
                "content": {"status": "fallback_after_render_validation"},
            })
        return {"final_response": answer, "trace": trace}

    def emergency_solve(self, problem: str) -> dict:
        """One answer-first attempt for entry-point failures before a valid result."""
        statement = str(problem or "").strip()
        spec = build_problem_spec(statement)
        prompt = self._recovery_request(statement, spec, RetrievalBundle((), ()), ())
        raw, result = self._call(
            prompt,
            stage="entrypoint_recovery",
            max_tokens=2048,
            temperature=0.1,
            thinking_mode=False,
            trace=[],
        )
        candidates = self._assess_response(
            raw,
            source="entrypoint_recovery",
            spec=spec,
            evidence=(),
            method_id="emergency",
            independence_group="emergency",
        )
        complete_candidates = [
            item for item in candidates if item.validation_tier == "complete"
        ]
        selected = choose_candidate(complete_candidates) or self._best_degraded(candidates, spec)
        answer = self._render_submission(selected.answer, spec) if selected else self._shape_fallback(spec)
        return {
            "final_response": answer,
            "trace": [{
                "step": "entrypoint_recovery",
                "content": {
                    "finish_reason": result.finish_reason or "unavailable",
                    "candidate_found": selected is not None,
                },
            }],
        }

    def _call(
        self,
        request: str,
        *,
        stage: str,
        max_tokens: int,
        temperature: float,
        thinking_mode: bool,
        trace: list[dict],
    ) -> tuple[str, ModelCallResult]:
        kwargs = {
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": request},
            ],
            "temperature": temperature,
            "max_tokens": max(256, int(max_tokens)),
        }
        chat_result = getattr(self.client, "chat_result", None)
        call = chat_result if callable(chat_result) else self.client.chat
        if self._supports_keyword(call, "thinking_mode"):
            kwargs["thinking_mode"] = bool(thinking_mode)
        started = monotonic()
        try:
            result = coerce_model_response(call(**kwargs))
            status = "ok" if result.content.strip() else "empty"
        except BaseException as error:
            if not is_recoverable_runtime_failure(error):
                raise
            result = ModelCallResult("")
            status = "failed"
            failure_type = type(error).__name__
        content = {
            "stage": stage,
            "status": status,
            "max_tokens": kwargs["max_tokens"],
            "thinking_mode": kwargs.get("thinking_mode", "client_default"),
            "finish_reason": result.finish_reason or "unavailable",
            "provider_truncated": result.provider_truncated,
            "elapsed_ms": round((monotonic() - started) * 1000),
        }
        if status == "failed":
            content["failure_type"] = failure_type
        trace.append({"step": "model_call", "content": content})
        return result.content.strip(), result

    def _assess_response(
        self,
        raw: str,
        *,
        source: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        method_id: str,
        independence_group: str,
    ) -> list[CandidateAssessment]:
        text = str(raw or "").strip()
        if not text:
            return []
        support_mode = spec.answer_contract.mode != "answer_only"
        extracted: list[ExtractionResult] = []
        if support_mode:
            for block in Finalizer.extract_tagged_submissions(text):
                extracted.append(ExtractionResult(
                    block,
                    "tagged_solution",
                    True,
                    (),
                    Finalizer.contains_meta(text),
                    True,
                ))
            if not extracted and not Finalizer.contains_meta(text):
                cleaned = Finalizer.extract_solution(text)
                reasons = Finalizer.validate_structure(cleaned)
                extracted.append(ExtractionResult(
                    cleaned,
                    "whole_solution",
                    not reasons,
                    reasons,
                    False,
                    False,
                ))
            # A complete explicit conclusion remains a useful last resort if
            # the surrounding proof is truncated or contaminated by meta text.
            # It is assessed separately and can never outrank a complete proof.
            extracted.extend(Finalizer.extract_explicit_results(text))
        else:
            extracted.extend(Finalizer.extract_explicit_results(text))
            if not extracted:
                extracted.append(Finalizer.extract_result(text))

        assessments: list[CandidateAssessment] = []
        seen: set[str] = set()
        for item in extracted:
            value = self._normalize_candidate(item.answer, spec)
            if not value or value in seen:
                continue
            seen.add(value)
            checks = tuple(
                CheckResult(check.name, check.status, "sympy", check.detail)
                for check in self.sympy.verify_candidate(spec.problem_text, value, spec)
            )
            assessments.append(assess_candidate(
                value,
                source,
                spec,
                evidence,
                extraction_method=item.method,
                extraction_reasons=item.rejected_reasons,
                raw_has_meta=item.raw_has_meta,
                explicit_answer=item.explicit_answer,
                method_id=method_id,
                independence_group=independence_group,
                math_checks=checks,
            ))
        return assessments

    def _assess_value(
        self,
        value: str,
        *,
        source: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        extraction_method: str,
        explicit: bool,
        method_id: str,
        independence_group: str,
    ) -> CandidateAssessment:
        normalized = self._normalize_candidate(value, spec)
        checks = tuple(
            CheckResult(check.name, check.status, "sympy", check.detail)
            for check in self.sympy.verify_candidate(spec.problem_text, normalized, spec)
        )
        return assess_candidate(
            normalized,
            source,
            spec,
            evidence,
            extraction_method=extraction_method,
            explicit_answer=explicit,
            method_id=method_id,
            independence_group=independence_group,
            math_checks=checks,
        )

    def _whole_tool_result(
        self,
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
    ) -> ToolResult | None:
        if not spec.tool_can_answer_whole:
            return None
        strict_requirements = tuple(
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
            if requirement.strict
        )
        eligible = []
        for result in results:
            contract = result.contract
            if not result.whole_answer_eligible or contract is None:
                continue
            if result.certificate.source_fingerprint != problem_fingerprint(spec.problem_text):
                continue
            if contract.covers(
                len(spec.goals),
                strict_requirements,
                task_kind=spec.profile.task_kind,
                answer_shape=spec.profile.answer_shape,
                problem_facts=(),
            ):
                eligible.append(result)
        if len(eligible) != 1:
            return None
        return eligible[0]

    def _supported_tool_candidate(
        self,
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> CandidateAssessment | None:
        fingerprint = problem_fingerprint(spec.problem_text)
        eligible = [
            result
            for result in results
            if result.supported_submission_eligible
            and result.certificate.source_fingerprint == fingerprint
            and self._certifies_goal_result(result, spec)
        ]
        if len(eligible) != 1:
            return None
        result = eligible[0]
        conclusion = result.result
        if spec.answer_contract.wrapper == "boxed":
            conclusion = rf"\boxed{{{conclusion}}}"
        candidate = self._assess_value(
            f"FINAL: {conclusion}\n{result.support}",
            source="certified_goal_result",
            spec=spec,
            evidence=evidence,
            extraction_method="certified_tool_with_support",
            explicit=True,
            method_id=result.operation,
            independence_group="local_tool",
        )
        if candidate.validation_tier != "complete" or candidate.tool_status != "pass":
            return None
        return candidate

    def _certified_goal_fallback(
        self,
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> CandidateAssessment | None:
        """Prefer a certified conclusion with partial support over an unrelated shape sentinel."""
        fingerprint = problem_fingerprint(spec.problem_text)
        eligible = [
            result for result in results
            if result.verified
            and result.goal_result_eligible
            and result.certificate.source_fingerprint == fingerprint
            and self._certifies_goal_result(result, spec)
            and result.support
        ]
        if len(eligible) != 1:
            return None
        result = eligible[0]
        conclusion = result.result
        if spec.answer_contract.wrapper == "boxed":
            conclusion = rf"\boxed{{{conclusion}}}"
        candidate = self._assess_value(
            f"FINAL: {conclusion}\n{result.support}",
            source="certified_goal_fallback",
            spec=spec,
            evidence=evidence,
            extraction_method="certified_goal_fallback",
            explicit=True,
            method_id=result.operation,
            independence_group="local_tool",
        )
        if (
            candidate.validation_tier not in {"complete", "degraded"}
            or candidate.tool_status != "pass"
            or candidate.failed_check
            or not candidate.result_coverage
            or not all(candidate.result_coverage)
        ):
            return None
        return candidate

    @staticmethod
    def _tool_evidence(
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        whole: ToolResult | None,
    ) -> tuple[ToolEvidence, ...]:
        fingerprint = problem_fingerprint(spec.problem_text)
        return tuple(
            ToolEvidence(
                result=result.result,
                scope=(
                    "whole_goal"
                    if result is whole
                    else "goal_result"
                    if SubmissionAgent._certifies_goal_result(result, spec)
                    else "subexpression"
                ),
                operation=result.operation,
                verified=bool(
                    result.verified
                    and result.certificate.source_fingerprint == fingerprint
                ),
                certificate_method=result.certificate.method,
                certificate_checks=result.certificate.checks,
                certificate_issues=result.certificate.issues,
                support=result.support,
            )
            for result in results
        )

    @staticmethod
    def _certifies_goal_result(result: ToolResult, spec: ProblemSpec) -> bool:
        contract = result.contract
        if not result.goal_result_eligible or contract is None:
            return False
        return bool(
            not contract.allowed_answer_shapes
            or spec.profile.answer_shape in contract.allowed_answer_shapes
        )

    def _primary_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        return self._request(
            problem,
            spec,
            role="Solve from first principles.",
            method=f"Suggested route, only if applicable: {spec.primary_method}.",
            context=cards.solve_context(),
            evidence=evidence,
        )

    def _independent_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        return self._request(
            problem,
            spec,
            role=(
                "Solve independently in a fresh context. Use a genuinely different "
                "derivation or invariant and actively search for counterexamples, "
                "missing cases, sign errors, and theorem-hypothesis failures."
            ),
            method=f"Independent route, only if applicable: {spec.alternative_method}.",
            context=cards.review_context(),
            evidence=evidence,
        )

    def _recovery_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        drafts: tuple[str, ...] = (),
    ) -> str:
        request = self._request(
            problem,
            spec,
            role=(
                "Previous attempts did not produce a complete parseable answer. "
                "Recover the mathematics compactly and put a checked complete FINAL line first."
            ),
            method="Prefer a short direct derivation and one decisive check.",
            context=cards.review_context(),
            evidence=evidence,
        )
        excerpts = [self._draft_excerpt(item) for item in drafts if str(item or "").strip()]
        excerpts = [item for item in excerpts if item]
        if excerpts:
            request += (
                "\n\nThe following model drafts are untrusted and may be truncated or wrong. "
                "Reuse only calculations you independently verify; resolve disagreements from "
                "the original problem and do not continue their prose.\n"
                + "\n\n".join(
                    f"Draft {index}:\n{excerpt}"
                    for index, excerpt in enumerate(excerpts, start=1)
                )
            )
        return request

    def _request(
        self,
        problem: str,
        spec: ProblemSpec,
        *,
        role: str,
        method: str,
        context: str,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        language = "Chinese" if spec.profile.language == "zh" else "English"
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "an explicit complete result"
        support = (
            "Include essential complete support in final_response."
            if spec.answer_contract.mode != "answer_only"
            else "Return the complete gradable result without a long exposition."
        )
        sections = [
            role,
            f"Answer language: {language}.",
            f"Required answer content: {obligations}.",
            support,
            method,
        ]
        if context:
            sections.append("General reference facts (verify applicability):\n" + context)
        tool_context = self._evidence_prompt(evidence)
        if tool_context:
            sections.append(
                "Certified local calculations for checking: goal_result entries certify the requested "
                "mathematical conclusion but not the required written justification; subexpression "
                "entries are only partial checks. Any submitted conclusion must agree with goal_result:\n"
                + tool_context
            )
        sections.append("Problem:\n" + problem)
        return "\n\n".join(sections)

    def _arbitration_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        first: CandidateAssessment,
        second: CandidateAssessment,
    ) -> str:
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        return (
            "Two independent candidates disagree. Recompute the disputed quantity from "
            "the original statement. Do not choose by style or majority. Provide one "
            "reproducible substitution, invariant, theorem-hypothesis audit, or small-case "
            "check. Output exactly these labelled sections, in this order:\n"
            "FINAL: the complete answer\n"
            "For a proof-like corrected answer, put the complete proof here.\n"
            "DECISION: A, B, CORRECTED, or UNRESOLVED\n"
            "CHECK: the decisive mathematical check\n\n"
            f"Required content: {obligations}.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Candidate A:\n{self._bounded(first.answer, 3500)}\n\n"
            f"Candidate B:\n{self._bounded(second.answer, 3500)}\n\n"
            f"Local check evidence:\n{self._evidence_prompt(evidence) or 'none'}"
        )

    def _apply_arbitration(
        self,
        raw: str,
        result: ModelCallResult,
        third_candidates: list[CandidateAssessment],
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        spec: ProblemSpec,
    ) -> tuple[str, CandidateAssessment | None]:
        if self._truncated(result, raw):
            return "truncated", None
        decision_match = re.search(
            r"(?im)^\s*DECISION\s*[:：]\s*(A|B|CORRECTED|UNRESOLVED)\b",
            raw,
        )
        check_match = re.search(r"(?im)^\s*CHECK\s*[:：]\s*(\S[^\n]{12,})$", raw)
        if not decision_match or not check_match:
            return "invalid_protocol", None
        if not self._decisive_check(check_match.group(1)):
            return "non_reproducible_check", None
        decision = decision_match.group(1).upper()
        submitted = choose_candidate(third_candidates)
        if decision == "A" and self._is_result_usable(first):
            if submitted is not None and self._same_conclusion(submitted, first, spec):
                return decision, first if self._is_complete(first) else submitted
            return "final_mismatches_A", None
        if decision == "B" and self._is_result_usable(second):
            if submitted is not None and self._same_conclusion(submitted, second, spec):
                return decision, second if self._is_complete(second) else submitted
            return "final_mismatches_B", None
        if decision == "CORRECTED":
            corrected = choose_candidate(third_candidates)
            if corrected is not None and corrected.validation_tier == "complete":
                return decision, corrected
        return decision, None

    @staticmethod
    def _conflict(
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        spec: ProblemSpec,
    ) -> bool:
        if not SubmissionAgent._is_result_usable(first) or not SubmissionAgent._is_result_usable(second):
            return False
        left = SubmissionAgent._comparison_value(first.answer)
        right = SubmissionAgent._comparison_value(second.answer)
        if equivalent_answers(left, right):
            return False
        if spec.profile.answer_shape == "choice":
            from classifier.choice import answer_choice_labels

            return set(answer_choice_labels(left)) != set(answer_choice_labels(right))
        if spec.profile.answer_shape in {"truth", "proof"}:
            left_polarity = SubmissionAgent._conclusion_polarity(left)
            right_polarity = SubmissionAgent._conclusion_polarity(right)
            if left_polarity and right_polarity:
                return left_polarity != right_polarity
            left_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", left)
            right_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", right)
            return bool(left_numbers and right_numbers and left_numbers != right_numbers)
        return True

    @staticmethod
    def _conclusion_polarity(value: str) -> str:
        text = str(value or "")
        if re.search(
            r"不成立|错误|为假|不是|不能|不可|不存在|无解|"
            r"\b(?:false|not|cannot|can't|does not|do not|no solutions?|impossible)\b",
            text,
            re.IGNORECASE,
        ):
            return "negative"
        if re.search(
            r"成立|正确|为真|是|能|可|存在|"
            r"\b(?:true|yes|holds?|exists?|is|are|can)\b",
            text,
            re.IGNORECASE,
        ):
            return "positive"
        return ""

    @staticmethod
    def _comparison_value(value: str) -> str:
        result = Finalizer.extract_result(value)
        return result.answer if result.valid and result.answer else str(value or "")

    @staticmethod
    def _is_usable(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier in {"complete", "degraded"}
            and not candidate.failed_check
        )

    @staticmethod
    def _is_complete(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier == "complete"
            and not candidate.failed_check
        )

    @staticmethod
    def _is_result_usable(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier in {"complete", "degraded"}
            and candidate.shape_valid
            and candidate.formatting_valid
            and bool(candidate.result_coverage)
            and all(candidate.result_coverage)
            and not candidate.failed_check
        )

    @staticmethod
    def _complete_after_transport(
        candidate: CandidateAssessment | None,
        truncated: bool,
    ) -> bool:
        return bool(
            SubmissionAgent._is_complete(candidate)
            and (not truncated or candidate.explicit_answer)
        )

    @staticmethod
    def _transport_admissible(
        candidates: list[CandidateAssessment],
        truncated: bool,
    ) -> list[CandidateAssessment]:
        if not truncated:
            return candidates
        return [item for item in candidates if item.explicit_answer]

    @staticmethod
    def _same_conclusion(
        left: CandidateAssessment,
        right: CandidateAssessment,
        spec: ProblemSpec,
    ) -> bool:
        left_value = SubmissionAgent._comparison_value(left.answer)
        right_value = SubmissionAgent._comparison_value(right.answer)
        if equivalent_answers(left_value, right_value):
            return True
        if spec.profile.answer_shape in {"truth", "proof"}:
            left_polarity = SubmissionAgent._conclusion_polarity(left_value)
            right_polarity = SubmissionAgent._conclusion_polarity(right_value)
            return bool(left_polarity and left_polarity == right_polarity)
        return False

    @staticmethod
    def _decisive_check(value: str) -> bool:
        return bool(re.search(
            r"[=<>≤≥]|代入|验证|边界|端点|反例|矛盾|定义|定理|不变量|归一化|余数|"
            r"\b(?:substitut|verify|boundary|endpoint|counterexample|contradiction|"
            r"definition|theorem|invariant|normalization|residual|differentiat|integrat)\w*\b",
            str(value or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def _objectively_checked(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and (candidate.tool_status == "pass" or candidate.passed_check_count)
        )

    @staticmethod
    def _best_degraded(
        candidates: Iterable[CandidateAssessment],
        spec: ProblemSpec,
    ) -> CandidateAssessment | None:
        usable = [
            item for item in candidates
            if item.answer
            and item.validation_tier == "degraded"
            and not item.failed_check
            and item.formatting_valid
            and (
                item.shape_valid
                or (
                    spec.answer_contract.mode != "answer_only"
                    and bool(item.result_coverage)
                    and all(item.result_coverage)
                    and item.explicit_answer
                )
            )
        ]
        method_priority = {
            "tagged_solution": 4,
            "label_boxed": 3,
            "label": 3,
            "bracket_label": 3,
            "whole_solution": 2,
            "boxed": 1,
        }
        agreement: dict[int, int] = {}
        for candidate in usable:
            groups = {
                other.independence_group
                for other in usable
                if other is not candidate
                and other.independence_group != candidate.independence_group
                and equivalent_answers(
                    SubmissionAgent._comparison_value(candidate.answer),
                    SubmissionAgent._comparison_value(other.answer),
                )
            }
            agreement[id(candidate)] = len(groups)
        source_priority = {
            "arbitration": 4,
            "recovery": 3,
            "independent": 2,
            "primary": 1,
        }
        return max(usable, key=lambda item: (
            agreement[id(item)] > 0,
            agreement[id(item)],
            item.tool_status == "pass",
            item.passed_check_count,
            bool(item.result_coverage) and all(item.result_coverage),
            item.shape_valid,
            source_priority.get(item.source, 0),
            method_priority.get(item.extraction_method, 0),
            item.score,
        )) if usable else None

    @staticmethod
    def _normalize_candidate(value: str, spec: ProblemSpec) -> str:
        answer = str(value or "").strip()
        if spec.profile.answer_shape == "truth" and re.fullmatch(
            r"(?:是|否|正确|错误|成立|不成立|true|false|yes|no)[。.!]?",
            answer,
            re.IGNORECASE,
        ):
            subject = spec.answer_frame.subject.strip()
            if subject:
                separator = "：" if spec.profile.language == "zh" else ": "
                answer = f"{subject}{separator}{answer}"
        return answer

    @staticmethod
    def _render_submission(answer: str, spec: ProblemSpec) -> str:
        value = str(answer or "").strip()
        if spec.answer_contract.mode != "answer_only":
            if (
                spec.answer_contract.wrapper == "boxed"
                and "\n" not in value
                and not re.search(r"\\boxed\s*\{", value)
                and not re.search(r"^(?:结论|Conclusion)\s*[:：]", value, re.IGNORECASE)
            ):
                return rf"\boxed{{{value}}}"
            replacement = "结论：" if spec.profile.language == "zh" else "Conclusion: "
            value = re.sub(
                r"(?i)^\s*(?:FINAL(?:\s+ANSWER)?|【\s*最终答案\s*】|最终答案|答案|CONCLUSION)\s*[:：=]?\s*",
                replacement,
                value,
                count=1,
            )
            value = re.sub(
                r"(?im)^\s*(?:DECISION|CHECK)\s*[:：].*$",
                "",
                value,
            )
            value = re.sub(r"\n{3,}", "\n\n", value).strip()
            return value
        if spec.answer_contract.wrapper == "boxed" and not re.fullmatch(
            r"\\boxed\s*\{.*\}", value, re.DOTALL
        ):
            return rf"\boxed{{{value}}}"
        return value

    @staticmethod
    def _shape_fallback(spec: ProblemSpec) -> str:
        shape = spec.profile.answer_shape
        if shape == "choice":
            from classifier.choice import option_labels

            labels = option_labels(spec.problem_text)
            return labels[0] if labels else "A"
        if shape == "truth":
            subject = spec.answer_frame.subject or ("该命题" if spec.profile.language == "zh" else "The statement")
            return f"{subject}不成立。" if spec.profile.language == "zh" else f"{subject} is false."
        if shape in {"roots", "interval"}:
            return r"\varnothing"
        if shape == "matrix":
            return r"\begin{pmatrix}0\end{pmatrix}"
        return r"\boxed{0}" if spec.answer_contract.wrapper == "boxed" else "0"

    @staticmethod
    def _truncated(result: ModelCallResult, raw: str) -> bool:
        if result.provider_truncated:
            return True
        reasons = set(Finalizer.validate_structure(raw)) if raw else {"empty"}
        return bool(reasons.intersection({
            "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
            "unclosed_display_latex", "unclosed_latex_environment",
            "unclosed_latex_brace", "unclosed_group_delimiter",
            "trailing_fragment", "truncated_sentence",
        }))

    @staticmethod
    def _deep_reasoning(spec: ProblemSpec) -> bool:
        return bool(
            spec.profile.difficulty == "hard"
            or spec.profile.task_kind in {"proof", "derivation", "explanation", "construction"}
            or spec.risk_score >= 4
        )

    @staticmethod
    def _remaining_seconds(started_at: float) -> float:
        return max(0.0, SUBMISSION_SOFT_ITEM_SECONDS - (monotonic() - started_at))

    @staticmethod
    def _supports_keyword(call, keyword: str) -> bool:
        try:
            signature = inspect.signature(call)
        except (TypeError, ValueError):
            return False
        return keyword in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _review_reason(budget: StageBudget, usable: bool, truncated: bool) -> str:
        if budget.require_independent_review:
            return "risk_requires_independent_solution"
        if not usable:
            return "no_complete_primary_candidate"
        if truncated:
            return "unverified_truncated_primary"
        return "primary_candidate_sufficient"

    @staticmethod
    def _evidence_prompt(evidence: tuple[ToolEvidence, ...]) -> str:
        lines = []
        for item in evidence:
            if not item.verified:
                continue
            lines.append(
                f"- {item.operation} ({item.scope}): {item.support or item.result}"
            )
        return "\n".join(lines)

    @staticmethod
    def _tool_trace(results: tuple[ToolResult, ...], whole: ToolResult | None) -> dict:
        return {
            "result_count": len(results),
            "direct_operation": whole.operation if whole else "none",
            "results": [result.trace_content() for result in results],
        }

    @staticmethod
    def _candidate_trace(candidate: CandidateAssessment, **extra) -> dict:
        content = {
            "source": candidate.source,
            "extraction": candidate.extraction_method,
            "validation_tier": candidate.validation_tier,
            "correctness_tier": candidate.correctness_tier,
            "complete_goals": candidate.complete_goals,
            "tool_status": candidate.tool_status,
            "checks": [item.trace_content() for item in candidate.math_checks],
            "rejected_reasons": list(candidate.rejected_reasons),
        }
        content.update(extra)
        return content

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[:limit] + "\n[content shortened]"

    @staticmethod
    def _draft_excerpt(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        explicit = [
            item.answer
            for item in Finalizer.extract_explicit_results(text)
            if item.answer and item.valid
        ]
        labelled = "\n".join(f"Explicit candidate: {item}" for item in explicit[-2:])
        if len(text) <= 6500:
            body = text
        else:
            body = text[:900] + "\n[earlier draft shortened]\n" + text[-5600:]
        return (labelled + "\n" + body).strip() if labelled else body

    @staticmethod
    def _load_prompt() -> str:
        path = Path(__file__).resolve().parent.parent / "prompts" / "submission.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return (
                "Solve the mathematics. Put a checked complete answer on the first "
                "line as FINAL: <answer>; include essential proof when required."
            )
