"""Bounded, retrieval-assisted implementation used by the public entry point."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
from time import monotonic

from classifier.problem_spec import ProblemSpec, build_problem_spec
from core.model_response import coerce_model_response
from core.runtime_failure import is_recoverable_runtime_failure
from core.stage_budget import StageBudget, plan_stage_budget
from rag.card_retriever import CardRetriever, RetrievalBundle
from reasoning.candidate_selector import (
    CandidateAssessment,
    ToolEvidence,
    assess_candidate,
    candidate_consistency_reasons,
    choose_candidate,
)
from reasoning.finalizer import ExtractionResult, Finalizer
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, result_from_legacy_hint


SUBMISSION_SOFT_BUDGET_SECONDS = 270
OPTIONAL_CALL_MIN_REMAINING_SECONDS = 135


class SubmissionAgent:
    """Score-first solve path with bounded independent verification."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.retriever = CardRetriever()
        self.prompt = self._load_prompt()

    def solve(self, problem: str, metadata: dict) -> dict:
        del metadata
        started_at = monotonic()
        text = str(problem or "").strip()
        spec = build_problem_spec(text)
        cards = self.retriever.retrieve(spec) if self._should_retrieve(spec) else RetrievalBundle((), ())
        evidence = self._tool_evidence(self.sympy.results_for(text), spec)
        tool_answer = self._whole_tool_answer(evidence)
        direct_tool_route = bool(tool_answer)
        deep_reasoning = self._use_deep_reasoning(spec, text)
        deep_verification = self._use_deep_verification(spec, deep_reasoning)
        budget = plan_stage_budget(spec, direct_tool_route, deep_reasoning=deep_reasoning)
        trace = [
            {"step": "spec", "content": spec.trace_content()},
            {"step": "risk_assessment", "content": {
                "score": spec.risk_score,
                "verification_required": spec.verification_required,
                "reasons": list(spec.risk_flags),
            }},
            {"step": "stage_budget", "content": budget.trace_content()},
            {"step": "call_plan", "content": {
                "route": "certified_tool" if direct_tool_route else "model",
                "max_model_calls": budget.max_calls,
                "deep_solve": deep_reasoning,
                "deep_verification": deep_verification,
                "independent_review_required": budget.require_independent_review,
            }},
            {"step": "retrieval", "content": cards.trace_content()},
            {"step": "tool_evidence", "content": self._evidence_trace(evidence)},
        ]

        first = ""
        first_truncated = False
        if not direct_tool_route:
            first, first_truncated = self._call(
                self._solve_request(text, spec, cards, evidence),
                "solve",
                budget.solve_tokens,
                trace,
                started_at,
                thinking_mode=deep_reasoning,
            )
        first_candidates = self._assess_candidates(
            first, "", "", tool_answer, spec, evidence, first_truncated=first_truncated
        )
        review_mode, review_reason = self._review_decision(
            spec, first_candidates, first, budget, started_at, text, first_truncated
        )
        trace.append({
            "step": "review_admission",
            "content": {
                "admitted": bool(review_mode),
                "mode": review_mode or "none",
                "reason": review_reason,
                "remaining_budget_ms": self._remaining_ms(started_at),
            },
        })
        second = ""
        second_truncated = False
        second_call_minimum = (
            0
            if review_mode in {"continue", "rescue"}
            else OPTIONAL_CALL_MIN_REMAINING_SECONDS
        )
        if review_mode and self._can_call(
            trace, budget, started_at, second_call_minimum
        ):
            independent = review_mode == "verify" or (
                review_mode == "rescue" and budget.require_independent_review
            )
            continuing = review_mode == "continue"
            request = self._solve_request(text, spec, cards, evidence) if continuing else (
                self._verification_request(text, spec, cards, evidence, first_candidates)
                if independent else self._last_chance_request(text, spec, cards, evidence)
            )
            second, second_truncated = self._call(
                request,
                review_mode,
                budget.review_tokens if independent else budget.emergency_tokens,
                trace,
                started_at,
                prior_response=first if continuing else "",
                followup=self._continuation_instruction(spec, final_attempt=False) if continuing else "",
                thinking_mode=(
                    deep_verification if independent else False
                ),
            )

        second_source = review_mode or "rescue"
        candidates = self._assess_candidates(
            first, second, "", tool_answer, spec, evidence, second_source,
            first_truncated=first_truncated,
            second_truncated=second_truncated,
        )
        conflict = self._candidate_conflict(candidates)
        selected_before_repair = self._select(candidates)
        usable_second = any(
            item.validation_tier in {"complete", "degraded"}
            and self._raw_source(item.source) == second_source
            for item in candidates
        )
        if conflict:
            repair_mode = "arbitration"
        elif review_mode == "continue" and usable_second:
            # Answer recovery is an extraction step, not a correctness check.
            # Challenge it in a fresh bounded context with explicit failure
            # modes instead of repeating another correlated long solve.
            repair_mode = "verify_recovered"
        elif review_mode in {"verify", "rescue"} and not usable_second:
            if selected_before_repair is not None:
                # A second unfinished long solve is correlated evidence. Audit
                # the usable candidate in a fresh bounded context instead of
                # spending the final call continuing another draft.
                repair_mode = "retry_verify"
            else:
                repair_mode = "last_chance"
        elif selected_before_repair is None:
            # Replaying two unfinished drafts caused repeated token exhaustion;
            # the final fallback intentionally starts from a clean context.
            repair_mode = "last_chance"
        else:
            repair_mode = ""
        arbitration = ""
        third_truncated = False
        verification_completion = ""
        fourth_truncated = False
        optional_arbitration_allowed = (
            repair_mode != "arbitration"
            or self._remaining_ms(started_at) >= budget.repair_min_remaining_seconds * 1000
        )
        proposed_correction = None
        decisive_single_correction = False
        if (
            repair_mode
            and budget.allow_repair
            and optional_arbitration_allowed
            and self._can_call(
                trace,
                budget,
                started_at,
                0 if selected_before_repair is None else OPTIONAL_CALL_MIN_REMAINING_SECONDS,
            )
        ):
            continuing_verifier = repair_mode == "continue_verify"
            targeted_auditor = repair_mode in {"verify_recovered", "retry_verify"}
            independent_verifier = False
            if targeted_auditor:
                repair_request = self._candidate_audit_request(
                    text, spec, cards, evidence, candidates
                )
            elif continuing_verifier:
                repair_request = self._verification_request(
                    text, spec, cards, evidence, candidates
                )
            elif repair_mode == "arbitration":
                repair_request = self._arbitration_request(text, spec, cards, candidates, evidence)
            else:
                repair_request = self._last_chance_request(text, spec, cards, evidence)
            arbitration, third_truncated = self._call(
                repair_request,
                repair_mode,
                budget.repair_tokens if targeted_auditor else (
                    budget.review_tokens if independent_verifier else (
                    budget.repair_tokens if repair_mode == "arbitration"
                    else budget.emergency_tokens
                    )
                ),
                trace,
                started_at,
                prior_response=second if continuing_verifier else "",
                followup=(
                    self._continuation_instruction(spec, final_attempt=True)
                    if continuing_verifier else ""
                ),
                thinking_mode=(
                    deep_verification if independent_verifier else False
                ),
            )
            candidates = self._assess_candidates(
                first, second, arbitration, tool_answer, spec, evidence, second_source,
                repair_mode,
                first_truncated,
                second_truncated,
                third_truncated,
            )
            conflict = self._candidate_conflict(candidates)
            usable_third = any(
                item.validation_tier in {"complete", "degraded"}
                and (not targeted_auditor or item.validation_tier == "complete")
                and self._raw_source(item.source) == repair_mode
                for item in candidates
            )
            proposed_correction = self._best_stage_candidate(
                [
                    item for item in candidates
                    if self._raw_source(item.source) == repair_mode
                    and item.verification_verdict == "corrected"
                    and item.validation_tier == "complete"
                ],
                {"verify"},
            )
            correction_changes_complete_candidate = bool(
                targeted_auditor
                and proposed_correction is not None
                and selected_before_repair is not None
                and selected_before_repair.validation_tier == "complete"
                and not equivalent_answers(
                    proposed_correction.answer, selected_before_repair.answer
                )
            )
            decisive_single_correction = bool(
                correction_changes_complete_candidate
                and self._certifies_minimum_power_divisibility_correction(
                    text,
                    selected_before_repair.answer,
                    proposed_correction.answer,
                    arbitration,
                )
            )
            if (
                targeted_auditor
                and arbitration.strip()
                and not usable_third
                and self._can_call(
                    trace, budget, started_at, OPTIONAL_CALL_MIN_REMAINING_SECONDS
                )
            ):
                retry_request = (
                    self._correction_confirmation_request(
                        text,
                        selected_before_repair.answer,
                        proposed_correction.answer,
                        spec,
                    )
                    if correction_changes_complete_candidate
                    else self._audit_retry_request(repair_request, arbitration, spec)
                )
                verification_completion, fourth_truncated = self._call(
                    retry_request,
                    "audit_retry",
                    budget.emergency_tokens,
                    trace,
                    started_at,
                    thinking_mode=False,
                )
                candidates.extend(self._assess_candidates(
                    "", "", verification_completion, "", spec, evidence,
                    third_source="audit_retry",
                    third_truncated=fourth_truncated,
                ))
                conflict = self._candidate_conflict(candidates)
        third_source = repair_mode if arbitration.strip() else "arbitration"
        trace.append({
            "step": "candidate_diagnostics",
            "content": {
                "solve": self._safe_trace_candidate(first),
                second_source: self._safe_trace_candidate(second),
                third_source: self._safe_trace_candidate(arbitration),
                "verification_completion": self._safe_trace_candidate(verification_completion),
            },
        })
        arbitration_selection = None
        arbitration_disposition = "not_used"
        arbitration_decision = ""
        if repair_mode == "arbitration" and arbitration.strip():
            (
                arbitration_selection,
                arbitration_disposition,
                arbitration_decision,
            ) = self._resolve_arbitration(arbitration, candidates)
        deterministically_certified_corrections = (
            [proposed_correction]
            if decisive_single_correction and proposed_correction is not None
            else []
        )
        if selected_before_repair is not None and verification_completion.strip():
            retry_correction = self._best_stage_candidate(
                [
                    item for item in candidates
                    if self._raw_source(item.source) == "audit_retry"
                    and item.verification_verdict == "corrected"
                    and item.validation_tier == "complete"
                ],
                {"verify"},
            )
            if (
                retry_correction is not None
                and not equivalent_answers(
                    retry_correction.answer, selected_before_repair.answer
                )
                and self._certifies_minimum_power_divisibility_correction(
                    text,
                    selected_before_repair.answer,
                    retry_correction.answer,
                    verification_completion,
                )
            ):
                deterministically_certified_corrections.append(retry_correction)
        selection_candidates = self._without_uncorroborated_corrections(
            candidates,
            selected_before_repair,
            tuple(deterministically_certified_corrections),
            spec,
        )
        selected = arbitration_selection or self._select(selection_candidates)
        verified_recovery = None
        if repair_mode in {"verify_recovered", "continue_verify", "retry_verify"}:
            verified_recovery = self._select([
                item for item in selection_candidates
                if self._raw_source(item.source) in {repair_mode, "continue_verify", "audit_retry"}
                and item.verification_verdict in {"confirmed", "corrected"}
            ])
            if verified_recovery is not None and (
                verified_recovery.accepted or selected is None
            ):
                # A fresh independent computation, or its bounded completion,
                # supersedes an answer recovered from an unfinished solve.
                selected = verified_recovery
        degraded_reason = ""
        if selected:
            answer = selected.answer
        else:
            answer, degraded_reason = self._best_effort_answer(
                (
                    (verification_completion, fourth_truncated),
                    (arbitration, third_truncated),
                    (second, second_truncated),
                    (first, first_truncated),
                ),
                spec,
                text,
            )
        answer = self._render_submission(answer, spec, text)
        trace.append({
            "step": "equivalence", "content": {
                "conflict": conflict,
                "accepted_sources": [item.source for item in candidates if item.accepted],
                "arbitration_used": bool(arbitration.strip() and repair_mode == "arbitration"),
                "third_stage_used": bool(arbitration.strip()),
                "final_verification_completion_used": bool(verification_completion.strip()),
                "repair_mode": repair_mode or "none",
                "final_repair_truncated": third_truncated,
                "arbitration_decision": arbitration_decision or "none",
                "arbitration_disposition": arbitration_disposition,
                "model_candidate_pairs": self._equivalence_pairs(candidates),
            },
        })
        trace.append({
            "step": "validation",
            "content": {item.source: self._assessment_trace(item) for item in candidates},
        })
        trace.append({
            "step": "selection", "content": {
                "source": selected.source if selected else "fallback",
                "score": selected.score if selected else 0,
                "degraded_reason": degraded_reason,
                "arbitration_selection_applied": bool(arbitration_selection),
                "recovered_answer_verification": (
                    verified_recovery.validation_tier if verified_recovery else "unresolved"
                ) if repair_mode in {
                    "verify_recovered", "continue_verify", "retry_verify"
                } else "not_required",
                "changed_correction_corroborated": bool(
                    verified_recovery
                    and selected_before_repair
                    and not equivalent_answers(
                        verified_recovery.answer, selected_before_repair.answer
                    )
                    and self._correction_is_corroborated(
                        verified_recovery, candidates, selected_before_repair
                    )
                ),
                "changed_correction_deterministically_certified": bool(
                    verified_recovery
                    and any(
                        verified_recovery is item
                        for item in deterministically_certified_corrections
                    )
                ),
            },
        })
        trace.append({
            "step": "finalize", "content": {
                "non_empty": bool(answer),
                "answer_shape": spec.profile.answer_shape,
                "source": selected.source if selected else "fallback",
                "contract_wrapper": self._contract_value(spec, "wrapper", "none"),
                "model_call_count": self._model_call_count(trace),
                "model_call_limit": budget.max_calls,
                "elapsed_ms": int((monotonic() - started_at) * 1000),
            },
        })
        self._append_proof_trace(
            trace, spec, selected, first, second, arbitration, verification_completion
        )
        return {"final_response": answer, "trace": trace}

    def _call(
        self,
        request: str,
        stage: str,
        max_tokens: int,
        trace: list[dict],
        started_at: float,
        prior_response: str = "",
        followup: str = "",
        thinking_mode: bool | None = None,
    ) -> tuple[str, bool]:
        stage_started = monotonic()
        try:
            messages = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": request},
            ]
            if prior_response.strip():
                messages.extend((
                    {"role": "assistant", "content": prior_response},
                    {"role": "user", "content": followup},
                ))
            call_kwargs = {
                "messages": messages,
                "temperature": 0.1 if stage in {
                    "arbitration", "last_chance", "rescue", "continue", "continue_last",
                    "continue_verify",
                } else 0.2,
                "max_tokens": max_tokens,
            }
            chat_result = getattr(self.client, "chat_result", None)
            call = chat_result if callable(chat_result) else self.client.chat
            if thinking_mode is None:
                thinking_mode = stage not in {
                    "continue", "continue_last", "last_chance", "arbitration", "rescue"
                }
            if self._supports_keyword(call, "thinking_mode"):
                call_kwargs["thinking_mode"] = thinking_mode
            raw_response = call(**call_kwargs)
            result = coerce_model_response(raw_response)
            value = result.content
            incomplete = {
                "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
                "unclosed_display_latex", "unclosed_latex_environment", "unclosed_latex_brace",
                "trailing_fragment", "truncated_sentence",
            }
            structural_truncation = bool(value.strip()) and bool(
                set(Finalizer.validate_structure(value)) & incomplete
            )
            truncation_signal = "provider_length" if result.provider_truncated else (
                "structural" if structural_truncation else (
                    "near_budget" if len(value) >= max_tokens * 3 else "none"
                )
            )
            trace.append({
                "step": f"model_call_{stage}",
                "content": {
                    "status": "completed",
                    "response_non_empty": bool(value.strip()),
                    "elapsed_ms": int((monotonic() - stage_started) * 1000),
                    "remaining_budget_ms": self._remaining_ms(started_at),
                    "max_tokens": max_tokens,
                    "finish_reason": result.finish_reason or "unavailable",
                    "provider_truncated": result.provider_truncated,
                    "response_near_budget": len(value) >= max_tokens * 3,
                    "truncation_signal": truncation_signal,
                    "usage": dict(result.usage),
                    "thinking_mode": thinking_mode if "thinking_mode" in call_kwargs else "client_default",
                    "routing_role": "deep_reasoning" if thinking_mode else "quick_response",
                },
            })
            return value, bool(result.provider_truncated or structural_truncation)
        except BaseException as exc:  # The platform client owns retries and limits.
            if not is_recoverable_runtime_failure(exc):
                raise
            trace.append({
                "step": f"model_call_{stage}",
                "content": {
                    "status": "failed",
                    "type": type(exc).__name__,
                    "failure_kind": (
                        "provider_timeout"
                        if "FunctionTimedOut" in {
                            base.__name__ for base in type(exc).__mro__
                        }
                        else "recoverable_client_error"
                    ),
                    "elapsed_ms": int((monotonic() - stage_started) * 1000),
                    "remaining_budget_ms": self._remaining_ms(started_at),
                },
            })
            return "", False

    @staticmethod
    def _use_deep_reasoning(spec: ProblemSpec, problem: str = "") -> bool:
        """Reserve long reasoning for tasks where it has positive expected value."""
        if (
            spec.profile.answer_shape in {"choice", "truth"}
            and spec.profile.difficulty != "hard"
            and len(spec.goals) == 1
        ):
            return False
        if spec.profile.problem_type in {"proof", "derivation", "explanation"}:
            return True
        if SubmissionAgent._is_simple_multi_blank(spec, problem):
            return False
        if len(spec.goals) > 1:
            return True
        if getattr(spec.profile, "topic", "general").startswith("olympiad_"):
            return True
        if spec.profile.difficulty == "hard":
            return True
        if re.search(
            r"\b(?:find|determine|classify)\s+all\s+(?:functions?|polynomials?|sequences?|sets?)\b|"
            r"\bnumber\s+of\s+(?:ordered\s+)?(?:real\s+)?(?:triples?|tuples?|solutions?)\b|"
            r"fourier\s*(?:transform|transformation|变换)|傅里叶变换",
            str(problem or ""),
            re.IGNORECASE,
        ):
            return True
        if (
            len(str(problem or "")) >= 180
            and spec.profile.answer_shape not in {"choice", "truth"}
        ):
            return True
        return False

    @staticmethod
    def _use_deep_verification(spec: ProblemSpec, solve_deep: bool) -> bool:
        """Keep bounded option/verdict rechecks from repeating a long solve."""
        return bool(
            solve_deep
            and spec.profile.answer_shape not in {"choice", "truth"}
        )

    @staticmethod
    def _is_simple_multi_blank(spec: ProblemSpec, problem: str = "") -> bool:
        """Recognize short recall-style fill-ins without weakening coupled math tasks."""
        if (
            spec.profile.problem_type != "calculation"
            or spec.profile.difficulty == "hard"
            or not 2 <= len(spec.goals) <= 4
        ):
            return False
        text = str(problem or "")
        semantic_text = re.sub(
            r"remember\s+to\s+put\s+your\s+final\s+answer\s+within\s+\\boxed\s*\{\s*\}\s*\.?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if len(semantic_text) > 160:
            return False
        blanks = re.findall(
            r"_{2,}|"
            r"[（(\[]\s*(?:\\(?:quad|qquad|blank))?\s*[）)\]]|"
            r"【\s*】",
            semantic_text,
            re.IGNORECASE,
        )
        if len(blanks) < len(spec.goals):
            return False
        return not bool(re.search(
            r"证明|推导|构造|分裂域|伽罗瓦|本原多项式|不可约|最小多项式|"
            r"(?:域|群|环)(?:扩张|同态|同构|生成元)|微分方程|"
            r"\b(?:prove|derive|construct|splitting\s+field|galois|primitive\s+polynomial|"
            r"irreducible|minimal\s+polynomial|field\s+extension|group\s+homomorphism|"
            r"differential\s+equation)\b",
            semantic_text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _supports_keyword(call, keyword: str) -> bool:
        try:
            parameters = inspect.signature(call).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    @staticmethod
    def _load_prompt() -> str:
        try:
            return (Path("prompts") / "submission.txt").read_text(encoding="utf-8")
        except OSError:
            return "Complete the decisive check before writing one FINAL: \\boxed{...} line."

    @staticmethod
    def _stage_answer_instruction(
        spec: ProblemSpec,
        problem: str,
        *,
        independent: bool = False,
    ) -> str:
        """Place FINAL after visible work only when hidden reasoning is disabled."""
        english = SubmissionAgent._answer_language(spec) == "en"
        deep = SubmissionAgent._use_deep_reasoning(spec, problem)
        if independent:
            deep = SubmissionAgent._use_deep_verification(spec, deep)
        if deep:
            return (
                "Finish the solution in hidden reasoning before emitting text. First line: "
                "FINAL: \\boxed{the complete answer}. Then give only essential support."
                if english else
                "先在隐藏推理中完成求解再输出文字。第一行必须写 FINAL: \\boxed{完整答案}，"
                "之后只保留必要依据。"
            )
        return (
            "First write at most three compact lines containing the decisive calculation or check. "
            "Then put the complete answer on the last line exactly as FINAL: \\boxed{...}. "
            "Do not state a provisional answer before that line."
            if english else
            "先用至多三行写出决定性计算或核验，再在最后一行恰好写 FINAL: \\boxed{完整答案}。"
            "在该行之前不得先写暂定答案。"
        )

    @staticmethod
    def _solve_request(problem: str, spec: ProblemSpec, cards: RetrievalBundle, evidence: tuple[ToolEvidence, ...]) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        header = "Problem" if english else "题目"
        obligations = "Required answer obligations" if english else "必须覆盖的作答要求"
        content = f"{header}:\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}"
        content += f"\n{obligations}:\n" + SubmissionAgent._goal_context(spec)
        content += (
            "\nInternal solution protocol (perform silently; do not print these labels):\n"
            if english else
            "\n内部解题协议（仅在隐藏推理中执行，不要输出这些标签）：\n"
        ) + SubmissionAgent._reasoning_protocol(spec, english)
        content += (
            f"\nRecommended method: {spec.primary_method}.\n"
            f"Checks required before committing to FINAL:\n{SubmissionAgent._audit_checklist(spec, True)}"
            if english else
            f"\n建议方法：{spec.primary_method}。\n"
            f"写下 FINAL 前必须完成的核验：\n{SubmissionAgent._audit_checklist(spec, False)}"
        )
        if cards.solve_context():
            content += (
                "\nCurated domain facts: apply every fact that directly matches the problem, and check its assumptions.\n"
                if english else
                "\n经本地校订的领域事实：直接适用时必须使用，并核对其前提。\n"
            ) + cards.solve_context()
        if evidence:
            content += ("\nVerified local evidence:\n" if english else "\n已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence)
        content += "\n" + SubmissionAgent._stage_answer_instruction(spec, problem)
        return content

    @staticmethod
    def _rescue_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidate: str,
    ) -> str:
        del candidate
        return SubmissionAgent._last_chance_request(problem, spec, cards, evidence)

    @staticmethod
    def _verification_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidates: list[CandidateAssessment],
    ) -> str:
        del candidates
        english = SubmissionAgent._answer_language(spec) == "en"
        alternative = str(getattr(spec, "alternative_method", "") or "independent_check")
        choice_check = ""
        if spec.profile.answer_shape == "choice":
            choice_check = (
                "Evaluate every option separately as true or false before returning the complete label set.\n"
                if english else
                "请先逐项判断每个选项的真伪，再返回完整的正确选项集合。\n"
            )
        content = (
            f"Problem:\n{problem}\n\nSolve independently from scratch using a different method where possible. "
            "Do not assume or reconstruct another solver's answer. Check every requested value, root, condition, unit, and extremal case.\n"
            f"Preferred independent method: {alternative}.\n{choice_check}"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
            f"Independent internal protocol (perform silently):\n"
            f"{SubmissionAgent._reasoning_protocol(spec, True, verification=True)}\n"
            if english else
            f"题目：\n{problem}\n\n请从头独立重算，尽量采用不同方法，不要猜测或沿用另一位求解者的答案。"
            f"逐项检查数值、全部根、条件、单位和极端情形。\n"
            f"优先采用的独立方法：{alternative}。\n{choice_check}"
            f"必须覆盖的作答要求：\n{SubmissionAgent._goal_context(spec)}\n"
            f"独立核验协议（仅在隐藏推理中执行）：\n"
            f"{SubmissionAgent._reasoning_protocol(spec, False, verification=True)}\n"
        )
        review_context = cards.review_context()
        if review_context:
            content += ("Alternative method hint:\n" if english else "备选方法提示：\n") + review_context + "\n"
        use_domain_fact = (
            spec.profile.answer_shape in {"choice", "truth"}
            or "definition_or_structure_conditions" in set(spec.risk_flags)
        )
        domain_fact = cards.verification_fact_context() if use_domain_fact else ""
        if domain_fact:
            content += (
                "One curated domain fact (check its assumptions independently):\n"
                if english else
                "一条经校订的领域事实（仍须独立核对其前提）：\n"
            ) + domain_fact + "\n"
        if evidence:
            content += ("Verified local evidence:\n" if english else "已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence) + "\n"
        content += SubmissionAgent._stage_answer_instruction(
            spec, problem, independent=True
        )
        return content

    @staticmethod
    def _candidate_audit_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidates: list[CandidateAssessment],
    ) -> str:
        """Build a bounded adversarial audit after a deep draft was truncated."""
        primary = choose_candidate([
            item for item in candidates
            if SubmissionAgent._source_stage(item.source) in {"solve", "rescue"}
            and item.validation_tier in {"complete", "degraded"}
        ])
        candidate_answer = primary.answer if primary is not None else ""
        extracted = Finalizer.extract_result(candidate_answer)
        if extracted.valid and extracted.answer:
            candidate_answer = extracted.answer
        candidate_answer = candidate_answer[:1200] or "[no usable candidate]"
        english = SubmissionAgent._answer_language(spec) == "en"
        checklist = SubmissionAgent._audit_checklist(spec, english)
        if english:
            content = (
                f"Problem:\n{problem}\n\nCandidate recovered from a truncated deep draft:\n"
                f"{candidate_answer}\n\nAdversarially audit this candidate. Do not echo it by default and do not "
                "restart a long full solution. Recompute the smallest decisive step that can falsify it; search "
                "for a missing branch, counterexample, factor-of-two error, or failed boundary case.\n"
                f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
                f"Mandatory audit checklist:\n{checklist}\n"
            )
        else:
            content = (
                f"题目：\n{problem}\n\n从截断的深度草稿中恢复出的候选答案：\n{candidate_answer}\n\n"
                "请对该候选做对抗性核验，不得默认照抄，也不要重新展开冗长完整解答。只重算最有判别力的一步，"
                "主动寻找遗漏分支、反例、二倍因子错误或边界失败。\n"
                f"必须覆盖的作答要求：\n{SubmissionAgent._goal_context(spec)}\n"
                f"强制核验清单：\n{checklist}\n"
            )
        review_context = cards.review_context()
        if review_context:
            content += ("Checked method hint:\n" if english else "经校订的方法提示：\n") + review_context + "\n"
        if evidence:
            content += ("Verified local evidence:\n" if english else "已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence) + "\n"
        return content + (
            "Return exactly three compact lines. First: CHECK: the decisive computation, counterexample, invariant, or two-sided bound. "
            "Second: VERDICT: CONFIRMED, VERDICT: CORRECTED, or VERDICT: UNRESOLVED. "
            "Third and last: FINAL: \\boxed{the actual complete answer}. "
            "CONFIRMED/CORRECTED without a concrete CHECK is invalid."
            if english else
            "严格只返回三行。第一行：CHECK: 决定性计算、反例、不变量或双向界。第二行："
            "VERDICT: CONFIRMED、VERDICT: CORRECTED 或 VERDICT: UNRESOLVED。"
            "第三行且最后一行：FINAL: \\boxed{实际完整答案}。"
            "没有具体 CHECK 的 CONFIRMED/CORRECTED 无效。"
        )

    @staticmethod
    def _audit_checklist(spec: ProblemSpec, english: bool) -> str:
        risks = set(spec.risk_flags)
        requirement_names = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
        }
        items: list[str] = []
        if any(name.startswith("parameter_dependency_") for name in requirement_names):
            items.append(
                "Retain every free parameter in FINAL and test at least two legal parameter values."
                if english else
                "FINAL 中必须保留全部自由参数，并至少代入两个合法参数值反检。"
            )
        if "exhaustive_result" in requirement_names or "exhaustiveness_required" in risks:
            items.append(
                "Actively search for an omitted second branch; a singleton needs a no-other-values check."
                if english else
                "主动搜索遗漏的第二分支；若答案只有一个值，必须核验不存在其他值。"
            )
        if "extremal_two_sided_bound" in risks:
            items.append(
                "Certify both directions: the universal bound and a matching construction/equality case."
                if english else
                "同时核验两个方向：普遍成立的界，以及达到该界的构造或等号情形。"
            )
            if re.search(
                r"\bno\s+matter\s+how\b[\s\S]{0,1000}\bpossible\s+to\s+choose\b|"
                r"无论|任意给定[\s\S]{0,1000}(?:可以|存在).*(?:选择|选出)",
                getattr(spec, "problem_text", ""),
                re.IGNORECASE,
            ):
                items.append(
                    "For the lower bound, exhibit one explicit input and minimize over every allowed choice, not one convenient choice."
                    if english else
                    "下界必须给出一个明确输入，并对其全部允许选择取最小值，不能只检查一个方便的选择。"
                )
        if "global_connectivity" in risks:
            items.append(
                "Check global connectivity and exclude disconnected cycles/subtours; local degree states are insufficient."
                if english else
                "检查全局连通并排除额外闭环/子回路；仅核对局部度数状态不充分。"
            )
        if "double_counting" in risks:
            items.append(
                "Check orientation, symmetry, endpoints, wrap-around, and every possible factor of two."
                if english else
                "检查方向、对称、端点、首尾闭合以及所有可能的二倍因子。"
            )
        if "endpoint_error" in risks or "domain_or_substitution" in risks:
            items.append(
                "Substitute the result back and test endpoints, excluded values, and equality cases."
                if english else
                "代回结果并检查端点、排除值和等号情形。"
            )
        if not items:
            items.append(
                "Use a direct substitution, small case, independent identity, or counterexample."
                if english else
                "使用直接代入、小规模情形、独立恒等式或反例核验。"
            )
        prefix = "- "
        return "\n".join(prefix + item for item in items)

    @staticmethod
    def _reasoning_protocol(
        spec: ProblemSpec,
        english: bool,
        *,
        verification: bool = False,
    ) -> str:
        """Give the model a compact, type-specific hidden worksheet.

        The protocol decomposes one public problem into checkable internal
        obligations without creating extra model calls or encouraging a long
        visible chain of thought.
        """
        task = getattr(spec.profile, "task_kind", spec.profile.problem_type)
        shape = spec.profile.answer_shape
        requirements = {
            item.name for goal in spec.goals for item in goal.requirements
        }
        steps = [
            (
                "Restate the exact target, givens, domains, and quantifiers; distinguish requested output from intermediate quantities."
                if english else
                "重述所求对象、已知条件、定义域与量词，区分最终所求和中间量。"
            ),
            (
                "Derive a candidate from the stated assumptions; do not import an unstated convention."
                if english else
                "仅从题设前提推出候选，不引入题面未给出的默认约定。"
            ),
        ]
        if shape == "choice":
            steps.append(
                "Build a truth table for every option from its definition, then form the complete label set."
                if english else
                "按定义逐项建立真值表，再合成完整选项集合。"
            )
        elif shape == "roots":
            steps.append(
                "Solve all branches, enforce the original domain, and substitute every root back to remove extraneous roots."
                if english else
                "求尽全部分支，执行原定义域限制，并逐根代回排除伪根。"
            )
        elif shape == "interval":
            steps.append(
                "Track inequality direction and test every endpoint and excluded singular point."
                if english else
                "跟踪不等号方向，逐一检查端点与被排除的奇点。"
            )
        elif task == "construction" or any(goal.kind == "construction" for goal in spec.goals):
            steps.append(
                "Write one explicit object, then verify every condition against that same object."
                if english else
                "写出一个明确构造对象，再对同一对象逐条核对全部条件。"
            )
        elif task in {"proof", "derivation", "explanation"}:
            steps.append(
                "Identify the smallest sufficient lemma, verify its hypotheses, and connect it to the claimed conclusion without a logical gap."
                if english else
                "确定足够且最小的关键引理，核对其前提，并无跳步地连接到待证结论。"
            )
        elif shape in {"count", "probability"}:
            steps.append(
                "Define exactly what is counted, including order, repetition, symmetry, and boundary cases; then check integrality or the range [0,1]."
                if english else
                "明确计数对象及有序性、重复、对称与边界，再检查整数性或概率是否落在[0,1]。"
            )
        else:
            steps.append(
                "Carry units, parameters, signs, and normalization through the decisive computation."
                if english else
                "在决定性计算中始终保留单位、参数、符号与归一化因子。"
            )
        if "exhaustive_result" in requirements:
            steps.append(
                "Prove exhaustiveness: test for a missing branch and justify why no other cases exist."
                if english else
                "证明穷尽性：主动寻找遗漏分支，并说明为何不存在其他情形。"
            )
        if verification:
            steps.append(
                "Try to falsify the candidate with a different identity, a smallest legal case, a boundary case, or direct substitution."
                if english else
                "用不同恒等式、最小合法情形、边界情形或直接代入尝试否证候选。"
            )
        else:
            steps.append(
                "Before FINAL, perform one disconfirming check rather than merely repeating the derivation."
                if english else
                "写 FINAL 前做一次否证式检查，不得只重复原推导。"
            )
        steps.append(
            "Assemble FINAL from the required-answer checklist only after every required item has a concrete value or conclusion."
            if english else
            "仅在每个必答项都有明确值或结论后，按清单组装 FINAL。"
        )
        return "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))

    @staticmethod
    def _audit_retry_request(base_request: str, failed_response: str, spec: ProblemSpec) -> str:
        """Escalate a failed audit without continuing its unfinished context."""
        english = SubmissionAgent._answer_language(spec) == "en"
        risks = set(spec.risk_flags)
        requirement_names = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
        }
        failures: list[str] = []
        if any(name.startswith("parameter_dependency_") for name in requirement_names):
            failures.append(
                "A constant FINAL that drops a requested free parameter is invalid; substitute two legal parameter values."
                if english else
                "若 FINAL 丢失题目要求的自由参数，则常数答案无效；必须代入两个合法参数值。"
            )
        if "exhaustive_result" in requirement_names:
            failures.append(
                "A singleton is not exhaustive merely because a formula holds for all indices; find a second branch or prove no other value exists."
                if english else
                "某公式对所有下标成立不等于单值答案已穷尽；必须寻找第二分支或证明无其他值。"
            )
        if "extremal_two_sided_bound" in risks:
            failures.append(
                "An extremum needs two separately checkable directions: a universal bound and explicit data/formula attaining it. The phrase 'achieved by a construction' is not a construction."
                if english else
                "极值必须有两个可分别核验的方向：普遍界，以及达到该界的明确数据/公式；只说“可由构造达到”不算构造。"
            )
        if "global_connectivity" in risks:
            failures.append(
                "Local degree counts do not count Hamilton paths. Give a connectivity-aware DP/recurrence that excludes every subtour, or return UNRESOLVED."
                if english else
                "局部度数计数不能代表 Hamilton 路径数；必须给出排除全部子回路的连通性 DP/递推，否则返回 UNRESOLVED。"
            )
        previous = str(failed_response or "").strip()[-1600:]
        if english:
            return (
                f"{base_request}\n\nThe previous audit below was rejected and must not be repeated:\n"
                f"{previous}\n\nWhy it is uncertified:\n- " + "\n- ".join(failures or [
                    "It did not provide the mandatory concrete certificate."
                ]) + "\nReturn a corrected three-line audit now; otherwise use VERDICT: UNRESOLVED."
            )
        return (
            f"{base_request}\n\n下面的上一轮核验已被拒绝，不得照抄：\n{previous}\n\n"
            "未通过核证的原因：\n- " + "\n- ".join(failures or [
                "没有给出强制要求的具体证书。"
            ]) + "\n现在返回修正后的三行核验；仍无法核证时必须写 VERDICT: UNRESOLVED。"
        )

    @staticmethod
    def _correction_confirmation_request(
        problem: str,
        original_answer: str,
        corrected_answer: str,
        spec: ProblemSpec,
    ) -> str:
        """Ask a fresh stage to arbitrate a value-changing correction."""
        english = SubmissionAgent._answer_language(spec) == "en"
        if english:
            return (
                f"Problem:\n{problem}\n\nA completed candidate says:\n{original_answer[:1200]}\n\n"
                f"A later audit proposes a different answer:\n{corrected_answer[:1200]}\n\n"
                "Independently recompute the smallest decisive identity, enumeration, substitution, or boundary case. "
                "Do not trust either label and do not copy either derivation.\n"
                f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
                "Return exactly three compact lines: CHECK: an independently reproducible calculation; "
                "VERDICT: CONFIRMED if the later correction is right, CORRECTED if a different result is right, "
                "or UNRESOLVED; FINAL: \\boxed{the complete answer}."
            )
        return (
            f"题目：\n{problem}\n\n已有完整候选：\n{original_answer[:1200]}\n\n"
            f"后续核验提出了不同答案：\n{corrected_answer[:1200]}\n\n"
            "请独立重算最有判别力的恒等式、枚举、代入或边界情形；不得相信任一标签，也不得照抄任一推导。\n"
            f"必须覆盖的作答要求：\n{SubmissionAgent._goal_context(spec)}\n"
            "严格只返回三行：CHECK: 可独立复现的计算；若后续修正正确则写 VERDICT: CONFIRMED，"
            "若应为第三个结果则写 VERDICT: CORRECTED，无法核证则写 VERDICT: UNRESOLVED；"
            "最后写 FINAL: \\boxed{完整答案}。"
        )

    @staticmethod
    def _arbitration_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        candidates: list[CandidateAssessment],
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        stage_groups = (
            ("A", {"solve"}),
            ("B", {"verify", "rescue"}),
        )
        rendered_items = []
        for label, stages in stage_groups:
            eligible = [
                item for item in candidates
                if item.validation_tier in {"complete", "degraded"}
                and item.answer
                and item.shape_valid
                and item.formatting_valid
                and SubmissionAgent._source_stage(item.source) in stages
            ]
            if not eligible:
                continue
            chosen = max(
                eligible,
                key=lambda item: (
                    item.validation_tier == "complete",
                    sum(item.goal_coverage),
                    item.score,
                    item.explicit_answer,
                ),
            )
            rendered_items.append(
                f"Candidate {label}: {chosen.answer[:1800]}"
                if english else f"候选{label}：{chosen.answer[:1800]}"
            )
        rendered = "\n".join(rendered_items)
        content = (
            f"Problem:\n{problem}\n\nThe independently produced candidates conflict. Recompute only the decisive step and choose A or B by mathematics, never by length or style. "
            "If decisive recomputation proves both candidates wrong, return the corrected answer; otherwise do not invent a third answer. "
            "If no result can be certified, mark the decision UNRESOLVED.\n"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n{rendered}\n"
            if english else
            f"题目：\n{problem}\n\n两个独立候选发生实质冲突。请只重算有判别力的关键步骤，并按数学正确性选择A或B，"
            f"不得按长度或措辞选择；只有决定性重算证明A、B均错时才可给出修正答案，否则不得发明第三个答案。"
            f"若无法核证任何结果，标记UNRESOLVED。\n"
            f"必须覆盖的作答要求：\n{SubmissionAgent._goal_context(spec)}\n{rendered}\n"
        )
        if evidence:
            content += ("Verified local evidence:\n" if english else "已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence) + "\n"
        if cards.solve_context():
            content += (
                "Curated domain facts (apply when directly relevant):\n"
                if english else
                "经本地校订的领域事实（直接相关时必须使用）：\n"
            ) + cards.solve_context() + "\n"
        return content + (
            "Return exactly three lines. First, CHECK: the decisive computation, definition, counterexample, or verified fact. "
            "Second, DECISION: A, DECISION: B, DECISION: CORRECTED, or DECISION: UNRESOLVED. "
            "Third and last, FINAL: \\boxed{the exact complete answer}. Never put only the candidate label A or B inside FINAL. "
            "A corrected decision must prove both A and B wrong."
            if english else
            "严格返回三行。第一行以 CHECK: 开头，写决定性计算、定义、反例或已核验事实。"
            "第二行写 DECISION: A、DECISION: B、DECISION: CORRECTED 或 DECISION: UNRESOLVED。"
            "第三行且最后一行写 FINAL: \\boxed{完整答案}，FINAL框内不得只写候选标签A或B；"
            "若选择CORRECTED，必须证明A、B均错。"
        )

    @staticmethod
    def _last_chance_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        content = (
            f"Problem:\n{problem}\n\nReturn the actual mathematical answer in at most six lines. "
            f"Recompute the decisive step first; put FINAL: \\boxed{{complete answer}} on the last line. "
            f"Do not state a provisional answer before it. Include only explicitly required support.\n"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
            if english else
            f"题目：\n{problem}\n\n最多六行：先重算决定性步骤，最后一行写 FINAL: \\boxed{{完整答案}}；"
            f"此前不得写暂定答案，只保留题目明确要求的依据。\n必须覆盖：\n{SubmissionAgent._goal_context(spec)}\n"
        )
        if evidence:
            content += ("Verified local evidence:\n" if english else "已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence) + "\n"
        if cards.solve_context():
            content += (
                "Curated domain facts:\n" if english else "经本地校订的领域事实：\n"
            ) + cards.solve_context() + "\n"
        return content

    @staticmethod
    def _continuation_instruction(spec: ProblemSpec, final_attempt: bool) -> str:
        del final_attempt
        english = SubmissionAgent._answer_language(spec) == "en"
        proof = spec.profile.problem_type in {"proof", "derivation", "explanation"}
        contract_mode = SubmissionAgent._contract_value(spec, "mode", "answer_only")
        needs_support = (
            proof
            or contract_mode in {"proof", "answer_with_support"}
            or len(spec.goals) > 1
            or any(goal.kind == "construction" for goal in spec.goals)
        )
        if english:
            support = (
                "Then give at most eight lines containing the essential proof."
                if proof else (
                    "The box must include every requested result or construction; then use at most six lines "
                    "for only the explicitly requested method, formula, or verification."
                    if needs_support else "Output no derivation after that line."
                )
            )
            return (
                "The preceding deep draft hit its token limit. Do not continue, restart, or explain the scratch work. "
                "Use its completed calculations, check the decisive recurrence or invariant once, and state the best "
                f"concrete result now. First line exactly FINAL: \\boxed{{complete answer}}. {support}"
            )
        support = (
            "之后至多用八行给出必要证明。"
            if proof else (
                "框内必须包含全部结论或构造；之后至多六行，只写题目明确要求的方法、公式或核验。"
                if needs_support else "该行之后不要输出推导。"
            )
        )
        return (
            "上面的深度草稿触及 token 上限。不要继续、重启或解释草稿；利用其中已完成的计算，"
            "核验一次决定性的递推或不变量，并立即给出最佳具体结果。"
            f"第一行必须恰为 FINAL: \\boxed{{完整答案}}。{support}"
        )

    @staticmethod
    def _continuation_context(first: str, second: str) -> str:
        """Keep enough prior derivation for a third request without overflowing context."""
        first_tail = str(first or "")[-8000:]
        second_value = str(second or "")[-12000:]
        return f"{first_tail}\n\n[continued draft]\n{second_value}".strip()

    @staticmethod
    def _goal_context(spec: ProblemSpec) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        checks = ({
            "proof": "key justification, necessary derivation, explicit conclusion",
            "construction": "constructed object and verification of every condition",
            "truth_judgement": "verdict, object being judged, decisive check",
            "domain_or_interval": "domain/interval, endpoints, excluded values",
            "formula": "complete formula and initial conditions",
            "comparison": "all requested values and the comparison",
            "equation_roots": "all roots with domain/extraneous-root checks",
            "scalar_or_result": "the requested result with any unit",
            "alternative_result": "the extremal value, or a proof that it does not exist",
        } if english else {
            "proof": "关键依据、必要推导、明确结论",
            "construction": "构造对象、逐项验证题设条件",
            "truth_judgement": "判断结论、被判断对象、关键检验",
            "domain_or_interval": "定义域/区间、端点和排除值",
            "formula": "完整公式、变量含义和题设初值",
            "comparison": "各个数值、误差或大小比较",
            "equation_roots": "全部根、定义域和伪根检查",
            "scalar_or_result": "明确结果及题目要求的单位/对象",
            "alternative_result": "极值结果，或其不存在的证明",
        })
        rendered = []
        risks = set(spec.risk_flags)
        for goal in spec.goals:
            suffix = checks.get(goal.kind, "完整可判分结论")
            parameter_names = [
                requirement.name.removeprefix("parameter_dependency_")
                for requirement in goal.requirements
                if requirement.name.startswith("parameter_dependency_")
            ]
            if parameter_names:
                suffix += (
                    "; FINAL must explicitly remain a function of " + ", ".join(parameter_names)
                    if english else
                    "；FINAL 必须显式保留参数 " + "、".join(parameter_names)
                )
            if any(
                requirement.name == "exhaustive_result"
                for requirement in goal.requirements
            ):
                suffix += (
                    "; enumerate every possibility and state that there are no others"
                    if english else
                    "；列全所有可能并明确无其他情形"
                )
            if "extremal_two_sided_bound" in risks:
                suffix += (
                    "; internally check both the universal bound and a matching construction/equality case"
                    if english else
                    "；内部须同时核验普遍界和达到该界的构造/等号情形"
                )
            if "global_connectivity" in risks:
                suffix += (
                    "; enforce one connected spanning path and exclude every detached cycle/subtour"
                    if english else
                    "；保证形成单条连通的遍历路径并排除额外闭环/子回路"
                )
            if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", goal.instruction, re.IGNORECASE):
                suffix += "；右侧区间须从初始点开始向右延伸，不得包含初始点左侧"
            rendered.append(
                f"- {goal.instruction} ({suffix})" if english else f"- {goal.instruction}（{suffix}）"
            )
        return "\n".join(rendered)

    @staticmethod
    def _spec_context(spec: ProblemSpec) -> str:
        goals = "\n".join(f"{goal.id}: {goal.instruction}" for goal in spec.goals)
        constraints = "、".join(spec.constraints) or "无额外显式约束"
        risks = "、".join(spec.risk_flags) or "常规完整性检查"
        return (
            f"目标清单：\n{goals}\n"
            f"题型：{spec.profile.subject} / {spec.profile.problem_type}；答案形态：{spec.profile.answer_shape}\n"
            f"约束：{constraints}\n首选方法：{spec.primary_method}；备选方法：{spec.alternative_method}\n"
            f"高风险检查：{risks}\n"
        )

    @staticmethod
    def _direct_instruction(spec: ProblemSpec) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        if any(re.search(
            r"最大右侧存在区间|maximal right(?:-hand)? interval", goal.instruction, re.IGNORECASE
        ) for goal in spec.goals):
            return ("Determine the solution from the initial value, then give the maximal right-hand interval starting at the initial point."
                    if english else
                "先通过初值确定解，再给出从初始自变量值开始向右延伸的最大存在区间；"
                "不要用包含初始点左侧的双侧最大区间替代右侧区间。")
        if spec.profile.problem_type in {"proof", "derivation", "explanation"}:
            return ("Give a concise self-contained proof with the decisive conditions and derivation, followed by the explicit conclusion."
                    if english else
                "请给出可直接提交的简洁证明：写明关键依据和必要推导，"
                "再用一句话明确结论。不要只给结论，也不要输出思考过程。")
        if english:
            if spec.profile.answer_shape == "roots":
                return "Give every root and check the domain and extraneous roots."
            if spec.profile.answer_shape == "interval":
                return "Check both endpoints and give the complete interval or union."
            if len(spec.goals) > 1:
                return "Answer every requested part in order."
            if any(item.strict for goal in spec.goals for item in goal.requirements):
                return "Give the result plus every explicitly required formula, intermediate value, method, or verification."
            return "Solve completely and state the requested answer concisely."
        if len(spec.goals) > 1:
            return "请按题目顺序完整回答每个子问。"
        frame = spec.answer_frame
        if frame.question_kind == "age" and frame.subject:
            return f"请完整求解；最终答案必须是可读句子，例如“{frame.subject}14岁。”，不要只写数值。"
        if frame.question_kind == "count":
            return "请完整求解；最终答案必须是可读句子，例如“所求数量为16个。”，不要只写数值。"
        if frame.question_kind == "probability":
            return "请完整求解；最终答案必须是可读句子，例如“所求概率为1/2。”，不要只写数值。"
        if frame.question_kind == "truth":
            return "请完整求解；给出完整判断句，并写出题目明确要求的关键检验、导数或数值，不要只写“是”或“否”。"
        if spec.profile.answer_shape == "roots":
            return "请给出全部解并检查定义域；离散根不要写成区间。"
        if spec.profile.answer_shape == "interval":
            return "请检查端点并用区间或并集给出解集。"
        if any(item.strict for goal in spec.goals for item in goal.requirements):
            return "请给出最终结论，并保留题目明确要求的方法、公式、中间量或验证步骤。"
        return "请直接完整求解，避免冗长解释。"

    @staticmethod
    def _tool_evidence(
        results: list[ToolResult] | list[str], spec: ProblemSpec
    ) -> tuple[ToolEvidence, ...]:
        """Translate deterministic results without letting labels self-certify."""
        evidence = []
        for raw in results:
            result = raw if isinstance(raw, ToolResult) else result_from_legacy_hint(raw)
            if result is None or not result.result.strip():
                continue
            support_required = bool(
                SubmissionAgent._contract_value(spec, "mode", "") == "proof"
                or getattr(spec.profile, "task_kind", spec.profile.problem_type)
                in {"proof", "derivation", "explanation", "construction"}
                or any(goal.kind in {"proof", "construction"} for goal in spec.goals)
                or any(
                    requirement.strict and requirement.category == "support"
                    for goal in spec.goals
                    for requirement in goal.requirements
                )
            )
            submission_result = (
                result.support.strip()
                if support_required and result.support.strip()
                else result.result.strip()
            )
            whole = bool(
                result.whole_answer_eligible
                and SubmissionAgent._tool_contract_covers_spec(result, spec)
            )
            evidence.append(ToolEvidence(
                submission_result,
                "whole_goal" if whole else "subexpression",
                result.operation,
                result.verified,
                result.certificate.method,
                result.certificate.checks,
                result.certificate.issues,
                result.support.strip(),
            ))
        whole = [item for item in evidence if item.scope == "whole_goal" and item.verified]
        if len(whole) > 1 and not all(
            equivalent_answers(whole[0].result, item.result) for item in whole[1:]
        ):
            evidence = [
                ToolEvidence(
                    item.result,
                    "subexpression" if item.scope == "whole_goal" else item.scope,
                    item.operation,
                    item.verified,
                    item.certificate_method,
                    item.certificate_checks,
                    tuple(dict.fromkeys((*item.certificate_issues, "conflicting_whole_tool_results"))),
                    item.support,
                )
                for item in evidence
            ]
        return tuple(evidence)

    @staticmethod
    def _tool_contract_covers_spec(result: ToolResult, spec: ProblemSpec) -> bool:
        """Require an explicit operation contract to cover every answer obligation."""
        tool_contract = result.contract
        if tool_contract is None or not result.verified:
            return False
        # Generic symbolic handlers deliberately solve only a narrow syntactic
        # fragment.  Their static contract is necessary but not sufficient:
        # ProblemSpec also rejects higher derivatives, constrained-domain
        # equations, numerical-method prompts, and other embedded operations.
        generic_symbolic_operations = {
            "calculate",
            "solve_equation",
            "derivative",
            "definite_integral",
            "integral",
            "limit",
        }
        if result.operation in generic_symbolic_operations and not spec.tool_can_answer_whole:
            return False
        if SubmissionAgent._has_uncovered_tool_obligation(
            spec.problem_text, result.operation
        ):
            return False
        answer_contract = getattr(spec, "answer_contract", None)
        parts = tuple(getattr(answer_contract, "parts", ())) if answer_contract is not None else ()
        goal_count = max(len(spec.goals), len(parts))
        requirements = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
            if getattr(requirement, "category", "result") != "format"
        }
        if not tool_contract.covers(
            goal_count,
            requirements,
            task_kind=getattr(spec.profile, "task_kind", spec.profile.problem_type),
            answer_shape=getattr(spec.profile, "result_kind", spec.profile.answer_shape),
            problem_facts=result.certificate.checks,
        ):
            return False
        if result.operation in {"spike_sequence_construction", "dependent_bernoulli_construction"}:
            return any(goal.kind == "construction" for goal in spec.goals)
        return True

    @staticmethod
    def _has_uncovered_tool_obligation(
        problem: str, operation: str = ""
    ) -> bool:
        """Reject direct routes when the prompt transforms or filters a matched result."""
        text = str(problem or "")
        if re.search(
            r"\b(?:and\s+then|then)\s+(?:add|subtract|multiply|divide|square|cube|double|"
            r"take|report|return|compute|evaluate|find)\b|"
            r"\b(?:twice|double|half\s+of|one\s+plus|two\s+times)\b|"
            r"\b(?:answer|result|requested\s+(?:quantity|value|number|probability)|"
            r"(?:quantity|value|number)\s+obtained)\b[^.!?\n]{0,24}"
            r"\b(?:plus|minus|times|multiplied\s+by|divided\s+by|modulo|mod|squared|cubed)\b|"
            r"\b(?:also|additionally|in\s+addition)\b[^.!?\n]{0,16}"
            r"\b(?:require|subject\s+to|impose|assume)\b|"
            r"\bonly\s+(?:count|include|report|return)\b|\breport\s+only\b|"
            r"\b(?:vertex\s+\d+|every\s+(?:set|member)|each\s+(?:set|member))\s+must\b|"
            r"(?:再|然后|所得结果|上述结果|原式)[^。！？!?\n]{0,24}(?:加|减|乘|除|平方|立方|倍)|"
            r"(?:答案|结果|所求(?:数|值|量|数量|概率|结果)|求得(?:数|值|量|结果))"
            r"[^。！？!?\n]{0,20}(?:加上?|减去?|乘以?|除以?|平方|立方|取模|余数)|"
            r"(?:另|另外|额外)(?:还|再)?(?:要求|限制|满足|规定)|"
            r"(?:只|仅)(?:统计|计算|计入|报告)|(?:顶点\s*\d+|每个集合|每个成员)必须",
            text,
            re.IGNORECASE,
        ):
            return True

        # These deterministic families deliberately compute a closed standard
        # problem.  A retained trigger plus one extra filter must not inherit a
        # whole-answer certificate.  The narrow per-operation guards avoid
        # disabling unrelated exact routes that naturally contain words such
        # as "such that" or "exactly".
        patterns = {
            "complete_multipartite_spanning_trees":
                r"(?:contain|include|exclude|avoid|must\s+use|required\s+edge)"
                r"[^.!?。！？\n]{0,50}\bedge\b|"
                r"(?:包含|含有|必须经过|不经过|指定)[^。！？\n]{0,30}边|"
                r"\b(?:also|additionally)\s+(?:delete|remove)\b|(?:再|另外|额外)(?:删|删除)",
            "digit_permutation_divisibility":
                r"\b(?:leading|first|last|final)\s+digit\b|"
                r"(?:首位|末位|第一位|最后一位)(?:数字|数码)?",
            "adjacent_surjection_count":
                r"f\s*\(\s*\d+\s*\)\s*(?:\\?ne|!=|≠|=)"
                r"\s*f\s*\(\s*1\s*\)|"
                r"(?:first|last|endpoints?)[^.!?\n]{0,40}(?:different|equal)|"
                r"(?:首项|末项|首尾)[^。！？\n]{0,30}(?:不同|相同|相等)",
            "binary_run_avoidance_count":
                r"\b(?:exactly|at\s+most|at\s+least)\s+\w+\s+(?:ones?|zeros?)\b|"
                r"(?:恰有|正好有|至多|至少)\s*[^。！？\n]{0,12}(?:个)?\s*[01一零](?:\s|$|个)",
            "bracelet_no_adjacent_count":
                r"\b(?:red|blue|green|third\s+colou?r|three\s+colou?rs?)\b|"
                r"(?:红|蓝|绿|第三种颜色|三种颜色)",
            "strip_lattice_path_count":
                r"\b(?:pass(?:es)?\s+through|go(?:es)?\s+through|avoid(?:s)?|"
                r"must\s+visit|contains?\s+the\s+point)\b|"
                r"(?:经过|必经|必须经过|避开|不经过)[^。！？\n]{0,30}(?:点|坐标|边)",
            "bipartite_matching_deletion_trees":
                r"\b(?:also|additionally)\s+(?:delete|remove)\b|"
                r"(?:再|另外|额外)(?:删|删除)",
            "complete_graph_cycle_deletion_trees":
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:delete|remove|contain|include|avoid)\b|"
                r"(?:再|另外|额外)[^。！？\n]{0,40}(?:删|删除|包含|经过|避开)",
            "finite_subtraction_game":
                r"\b(?:consecutive|previous|last)\s+(?:turn|move)|"
                r"\b(?:may|can)\s+not\s+(?:be\s+)?(?:repeat|use).{0,25}(?:move|size)|"
                r"(?:连续两次|上一(?:步|次)|不能重复|不得连续)[^。！？\n]{0,25}(?:取法|步数|数量)?",
            "wythoff_losing_position_count":
                r"\b(?:also|additionally|except|unless)\b[^.!?\n]{0,80}"
                r"\b(?:move|remove|heap|position|count|require)\b",
            "bounded_generalized_pell_count":
                r"\\?gcd\s*\(|\bcoprime\b|\bprimitive\b|互素|本原解",
            "punctured_domino_tilings":
                r"\b(?:vertical|horizontal)\s+domino(?:es)?\b|"
                r"(?:竖直|水平|横放|竖放)[^。！？\n]{0,20}多米诺",
            "positive_sum_two_squares":
                r"\bx\s*(?:<|≤|\\leq?|!=|\\ne|≠)\s*y\b|"
                r"\by\s*(?:<|≤|\\leq?|!=|\\ne|≠)\s*x\b",
            "odd_fiber_functions": r"f\s*\(\s*\d+\s*\)\s*=\s*\d+",
            "two_point_gauss_legendre_monomial":
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:exact\s+integral|error|difference|compare)\b|"
                r"(?:并|另外|额外)[^。！？\n]{0,50}(?:精确积分|误差|差值|比较)",
            "exponential_l1_sequence":
                r"(?:并|另外|额外)[^。！？\n]{0,60}(?:证明|推导|解释|依测度|一致|弱收敛|上确界)|"
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:prove|derive|explain|measure|uniform|weak|supremum)\b",
            "cauchy_location_fisher_information":
                r"(?:并|另外|额外)[^。！？\n]{0,60}(?:方差|估计|极限分布|证明|推导)|"
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:variance|estimate|distribution|prove|derive)\b",
            "one_dimensional_wald_statistic":
                r"(?:并|另外|额外)[^。！？\n]{0,60}(?:p\s*[- ]?值|拒绝|分布|证明|推导)|"
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:p\s*[- ]?value|reject|distribution|prove|derive)\b",
            "diagonal_gls_estimate":
                r"(?:并|另外|额外)[^。！？\n]{0,60}(?:拟合值|残差|方差|标准误|证明|推导)|"
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:fitted|residual|variance|standard\s+error|prove|derive)\b",
            "normal_variance_confidence_interval":
                r"(?:并|另外|额外)[^。！？\n]{0,60}(?:均值|检验|p\s*[- ]?值|证明|推导)|"
                r"\b(?:also|additionally)\b[^.!?\n]{0,60}"
                r"\b(?:mean|test|p\s*[- ]?value|prove|derive)\b",
        }
        if operation == "quadratic_congruence_count":
            congruences = re.findall(
                r"\\equiv|≡|\bcongruent\b", text, re.IGNORECASE
            )
            if len(congruences) > 1:
                return True
        pattern = patterns.get(str(operation or ""))
        return bool(pattern and re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _whole_tool_answer(evidence: tuple[ToolEvidence, ...]) -> str:
        whole = [
            item.result for item in evidence
            if item.scope == "whole_goal" and item.verified
        ]
        if not whole or not all(equivalent_answers(whole[0], item) for item in whole[1:]):
            return ""
        return whole[0]

    @staticmethod
    def _evidence_context(evidence: tuple[ToolEvidence, ...]) -> str:
        return "\n".join(
            f"- {item.operation} ({item.scope}): {item.result}"
            + (f"\n  certificate support: {item.support}" if item.support else "")
            for item in evidence
        )

    @staticmethod
    def _evidence_trace(evidence: tuple[ToolEvidence, ...]) -> dict:
        return {
            "whole_goal_count": sum(item.scope == "whole_goal" for item in evidence),
            "subexpression_count": sum(item.scope == "subexpression" for item in evidence),
            "verified_subexpression_count": sum(
                item.scope == "subexpression" and item.verified for item in evidence
            ),
            "operations": [item.operation for item in evidence],
            "certificates": [
                {
                    "operation": item.operation,
                    "scope": item.scope,
                    "passed": item.verified,
                    "method": item.certificate_method,
                    "checks": list(item.certificate_checks),
                    "issues": list(item.certificate_issues),
                    "result": item.result,
                    "support": item.support,
                }
                for item in evidence
            ],
        }

    def _review_decision(
        self,
        spec: ProblemSpec,
        first_candidates: list[CandidateAssessment],
        first_response: str,
        budget: StageBudget,
        started_at: float,
        problem: str = "",
        provider_truncated: bool = False,
    ) -> tuple[str, str]:
        if not budget.allow_review:
            return "", "review_disabled"
        usable = [
            candidate for candidate in first_candidates
            if candidate.validation_tier in {"complete", "degraded"}
            and candidate.shape_valid
            and candidate.formatting_valid
        ]
        needs_rescue = (
            not usable
            or all(candidate.coverage_uncertain for candidate in usable)
        )
        if needs_rescue:
            if provider_truncated and first_response.strip():
                return "continue", "truncated_without_complete_result"
            return "rescue", "missing_complete_result"
        if self._remaining_ms(started_at) < OPTIONAL_CALL_MIN_REMAINING_SECONDS * 1000:
            return "", "insufficient_optional_review_time"
        if (
            not budget.require_independent_review
            and self._remaining_ms(started_at) < budget.review_min_remaining_seconds * 1000
        ):
            return "", "insufficient_optional_review_time"
        if budget.require_independent_review:
            return "verify", "high_risk_independent_check"
        if spec.verification_required:
            return "verify", "problem_contract_requires_verification"
        if getattr(spec.profile, "confidence", "high") == "low":
            return "verify", "low_classification_confidence"
        contest_contract = bool(re.search(
            r"remember\s+to\s+put\s+your\s+final\s+answer|\\boxed\s*\{\s*\}",
            str(problem or ""),
            re.IGNORECASE,
        ))
        if contest_contract:
            return "verify", "contest_contract_independent_check"
        if len(str(problem or "")) >= 180:
            return "verify", "long_problem_independent_check"
        return "", "complete_low_risk_result"

    @staticmethod
    def _model_call_count(trace: list[dict]) -> int:
        return sum(
            str(item.get("step", "")).startswith("model_call_")
            for item in trace
        )

    @staticmethod
    def _can_call(
        trace: list[dict],
        budget: StageBudget,
        started_at: float | None = None,
        min_remaining_seconds: int = 0,
    ) -> bool:
        if SubmissionAgent._model_call_count(trace) >= budget.max_calls:
            return False
        if started_at is None:
            return min_remaining_seconds <= 0
        return (
            SubmissionAgent._remaining_ms(started_at)
            >= max(0, min_remaining_seconds) * 1000
        )

    @staticmethod
    def _response_near_budget(response: str, max_tokens: int) -> bool:
        return len(str(response or "")) >= max_tokens * 3

    def _assess_candidates(
        self,
        first: str,
        second: str,
        arbitration: str,
        tool_answer: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        second_source: str = "rescue",
        third_source: str = "arbitration",
        first_truncated: bool = False,
        second_truncated: bool = False,
        third_truncated: bool = False,
    ) -> list[CandidateAssessment]:
        candidates: list[CandidateAssessment] = []

        def add_response(
            response: str,
            stage: str,
            verdict: str = "",
            provider_truncated: bool = False,
        ) -> None:
            if not response.strip():
                return
            candidate_response = response
            if stage in {"verify_recovered", "retry_verify", "audit_retry"}:
                # VERDICT/CHECK are internal audit controls. The check content
                # may satisfy an explicitly requested method obligation, but
                # control labels must never leak into final_response.
                candidate_response = re.sub(
                    r"(?im)^\s*VERDICT\s*[:：].*(?:\n|$)",
                    "",
                    candidate_response,
                )
                candidate_response = re.sub(
                    r"(?im)^\s*CHECK\s*[:：]\s*",
                    "",
                    candidate_response,
                ).strip()
            has_unclosed_box = any(
                not complete
                for _, _, complete in Finalizer._boxed_values(candidate_response)
            )
            stable_prefix = (
                self._stable_truncated_prefix(response, spec)
                if provider_truncated else ""
            )
            consistency_reasons = candidate_consistency_reasons(
                candidate_response, spec
            )
            results = [self._finalize(candidate_response, spec)]
            for explicit in Finalizer.extract_explicit_results(candidate_response):
                if not explicit.answer:
                    continue
                normalized = self._render_answer(self._normalize_answer(explicit.answer, spec), spec)
                reasons = Finalizer.validate_structure(normalized)
                results.append(ExtractionResult(
                    normalized if not reasons else "",
                    explicit.method,
                    explicit.valid and not reasons,
                    tuple(dict.fromkeys((*explicit.rejected_reasons, *reasons))),
                    explicit.raw_has_meta,
                    True,
                ))
            seen: set[tuple[str, tuple[str, ...]]] = set()
            for result in results:
                key = (result.answer, result.rejected_reasons)
                if key in seen or not (result.answer or result.rejected_reasons):
                    continue
                seen.add(key)
                source = stage if len(seen) == 1 else f"{stage}#{len(seen)}"
                reasons = tuple(dict.fromkeys((
                    *result.rejected_reasons,
                    *consistency_reasons,
                )))
                if provider_truncated and not result.explicit_answer:
                    reasons = tuple(dict.fromkeys((*reasons, "provider_truncated_without_explicit_answer")))
                stable_explicit = bool(
                    stable_prefix
                    and result.answer
                    and equivalent_answers(result.answer, stable_prefix)
                )
                if (
                    provider_truncated
                    and not stable_explicit
                    and (
                        has_unclosed_box
                        or (
                            result.method in {"boxed", "boxed_unclosed"}
                            and not SubmissionAgent._is_just_boxed(response)
                        )
                    )
                ):
                    reasons = tuple(dict.fromkeys((*reasons, "provider_truncated_ambiguous_box")))
                if (
                    provider_truncated
                    and stage in {
                        "verify", "verify_recovered", "continue_verify", "retry_verify", "audit_retry"
                    }
                    and not stable_explicit
                ):
                    # A cut verifier may state one value near the start and
                    # contradict it later.  It is unresolved evidence, not a
                    # safe replacement for a complete primary answer.
                    reasons = tuple(dict.fromkeys((
                        *reasons, "provider_truncated_ambiguous_box"
                    )))
                if stage in {"verify_recovered", "retry_verify", "audit_retry"}:
                    if verdict == "unresolved":
                        reasons = tuple(dict.fromkeys((*reasons, "verification_unresolved")))
                    elif verdict not in {"confirmed", "corrected"} or not self._has_audit_support(response, spec):
                        reasons = tuple(dict.fromkeys((*reasons, "missing_verification_certificate")))
                candidates.append(assess_candidate(
                    result.answer, source, spec, evidence, result.method, reasons,
                    result.raw_has_meta, result.explicit_answer, verdict,
                ))

        # A verifier often places a complete answer before a long independent
        # check. The prefix remains usable after provider truncation only when
        # nothing later retracts it or introduces a conflicting conclusion.

        add_response(first, "solve", provider_truncated=first_truncated)
        add_response(
            second, second_source, self._verification_verdict(second), second_truncated
        )
        add_response(
            arbitration, third_source, self._verification_verdict(arbitration), third_truncated
        )
        if tool_answer:
            answer = self._render_answer(self._normalize_answer(tool_answer, spec), spec)
            candidates.append(assess_candidate(answer, "sympy_verified", spec, evidence, "tool"))
        return candidates

    @staticmethod
    def _stable_truncated_prefix(response: str, spec: ProblemSpec) -> str:
        text = str(response or "")
        indexed_lines = [
            (index, line.strip())
            for index, line in enumerate(text.splitlines())
            if line.strip()
        ]
        if not indexed_lines:
            return ""
        first_index, first_line = indexed_lines[0]
        if not re.match(
            r"^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:(?:最终\s*)?答案|结论|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion))"
            r"\s*(?:is|equals|[:：为=]))",
            first_line,
            re.IGNORECASE,
        ):
            return ""
        first_result = Finalizer.extract_result(first_line)
        if not (
            first_result.valid
            and first_result.explicit_answer
            and first_result.answer
            and not Finalizer.validate_structure(first_result.answer)
            and all(complete for _, _, complete in Finalizer._boxed_values(first_line))
        ):
            return ""

        tail = "\n".join(text.splitlines()[first_index + 1:]).strip()
        if re.search(
            r"(?:\b(?:correction|corrected|actually|wait|however|but|yet|reconsider|revise[dt]?)\b|"
            r"修正|更正|等等|但是|然而|不过|重新(?:计算|检查|考虑|核算))",
            tail,
            re.IGNORECASE,
        ):
            return ""
        if any(not complete for _, _, complete in Finalizer._boxed_values(tail)):
            return ""
        for later in Finalizer.extract_explicit_results(tail):
            if later.answer and later.valid and not equivalent_answers(
                first_result.answer, later.answer
            ):
                return ""

        normalized = SubmissionAgent._render_answer(
            SubmissionAgent._normalize_answer(first_result.answer, spec), spec
        )
        normalized = SubmissionAgent._normalize_answer(normalized, spec)
        return normalized if not Finalizer.validate_structure(normalized) else ""

    @staticmethod
    def _verification_verdict(response: str) -> str:
        value = str(response or "")
        match = re.search(
            r"(?:VERDICT|【\s*校验\s*】|校验|复核)\s*[:：]?\s*"
            r"(CONFIRMED|CORRECTED|UNRESOLVED|通过|一致|修正|错误|未解决)",
            value,
            re.IGNORECASE,
        )
        if not match:
            return ""
        verdict = match.group(1).upper()
        if verdict in {"CONFIRMED", "通过", "一致"}:
            return "confirmed"
        if verdict in {"CORRECTED", "修正", "错误"}:
            return "corrected"
        return "unresolved"

    @staticmethod
    def _correction_is_corroborated(
        candidate: CandidateAssessment,
        candidates: list[CandidateAssessment],
        baseline: CandidateAssessment,
    ) -> bool:
        """Require two audit stages before replacing a complete value."""
        if equivalent_answers(candidate.answer, baseline.answer):
            return True
        if candidate.tool_status == "pass":
            return True
        supporting_stages = {
            SubmissionAgent._raw_source(item.source)
            for item in candidates
            if item.answer
            and item.verification_verdict in {"confirmed", "corrected"}
            and item.validation_tier == "complete"
            and item.complete_goals
            and item.shape_valid
            and item.formatting_valid
            and not item.rejected_reasons
            and item.tool_status != "conflict"
            and equivalent_answers(item.answer, candidate.answer)
            and not equivalent_answers(item.answer, baseline.answer)
        }
        has_complete_support = any(
            item.validation_tier == "complete"
            and item.verification_verdict in {"confirmed", "corrected"}
            and equivalent_answers(item.answer, candidate.answer)
            for item in candidates
        )
        return has_complete_support and len(supporting_stages) >= 2

    @staticmethod
    def _without_uncorroborated_corrections(
        candidates: list[CandidateAssessment],
        baseline: CandidateAssessment | None,
        deterministically_certified: tuple[CandidateAssessment, ...] = (),
        spec: ProblemSpec | None = None,
    ) -> list[CandidateAssessment]:
        """Keep a usable baseline unless a changed audit result is corroborated."""
        if (
            baseline is None
            or baseline.validation_tier not in {"complete", "degraded"}
            or not baseline.answer
            or not baseline.shape_valid
            or not baseline.formatting_valid
        ):
            return candidates
        if baseline.validation_tier == "degraded" and spec is not None:
            missing_results = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.result_requirements
                if not requirement.matches(baseline.answer)
            }
            # A concrete missing dependency, unit, or requested component is
            # mechanically checkable.  A corrected audit may repair it in one
            # pass.  Exhaustiveness is semantic: merely saying "all" is not
            # enough evidence to replace a usable singleton or finite set.
            if missing_results and missing_results != {"exhaustive_result"}:
                return candidates
        audit_sources = {
            "verify_recovered", "continue_verify", "retry_verify", "audit_retry"
        }
        return [
            item for item in candidates
            if (
                SubmissionAgent._raw_source(item.source) not in audit_sources
                or equivalent_answers(item.answer, baseline.answer)
                or any(
                    equivalent_answers(item.answer, certified.answer)
                    for certified in deterministically_certified
                )
                or SubmissionAgent._correction_is_corroborated(
                    item, candidates, baseline
                )
            )
        ]

    @staticmethod
    def _certifies_minimum_power_divisibility_correction(
        problem: str,
        original_answer: str,
        corrected_answer: str,
        response: str,
    ) -> bool:
        """Certify the narrow minimum positive n with modulus dividing n^k task."""
        text = str(problem or "").replace(r"\mid", " divides ")
        target = re.search(
            r"(?:least|minimum|smallest)\s+positive\s+integer\s+([A-Za-z])"
            r"[\s\S]{0,100}?\1\s*\^\s*\{?(\d+)\}?\s+is\s+divisible\s+by\s+(\d+)|"
            r"最小(?:的)?正整数\s*([A-Za-z])?[\s\S]{0,80}?"
            r"([A-Za-z])\s*\^\s*\{?(\d+)\}?[\s\S]{0,30}?被\s*(\d+)\s*整除",
            text,
            re.IGNORECASE,
        )
        if not target or not SubmissionAgent._has_audit_support(response):
            return False
        if target.group(1):
            exponent, modulus = int(target.group(2)), int(target.group(3))
        else:
            exponent, modulus = int(target.group(6)), int(target.group(7))

        def integer(value: str) -> int | None:
            match = re.fullmatch(
                r"(?:\\boxed\s*\{)?\s*(\d+)\s*(?:\})?",
                str(value or "").strip(),
            )
            return int(match.group(1)) if match else None

        original = integer(original_answer)
        corrected = integer(corrected_answer)
        if (
            original is None
            or corrected is None
            or not 2 <= exponent <= 20
            or not 2 <= modulus <= 10**9
            or not 1 <= corrected <= 100_000
        ):
            return False
        return bool(
            pow(corrected, exponent, modulus) == 0
            and pow(original, exponent, modulus) != 0
            and all(pow(value, exponent, modulus) != 0 for value in range(1, corrected))
        )

    @staticmethod
    def _has_audit_support(response: str, spec: ProblemSpec | None = None) -> bool:
        match = re.search(
            r"(?ims)^\s*CHECK\s*[:：]\s*(.+)$",
            str(response or ""),
        )
        if not match:
            return False
        support = match.group(1).strip()
        if len(support) < 8 or Finalizer.contains_meta(support):
            return False
        if Finalizer.validate_structure(support):
            return False
        # A bare number or a phrase such as "the calculation gives 8" is not
        # independently checkable evidence.  Require either a reproducible
        # mathematical relation, an explicit logical implication, or a named
        # validation method applied to a concrete object.
        relation = bool(re.search(
            r"(?:[A-Za-z][A-Za-z_0-9{}\\]*|\d+(?:\.\d+)?|[)\]}])\s*"
            r"(?:=|!=|≠|<|>|≤|≥|\\(?:ne|neq|leq?|geq?|mid|nmid)\b)\s*"
            r"(?:[A-Za-z\\]|[-+]?\d|[({[])",
            support,
            re.IGNORECASE,
        ))
        option_table = bool(re.search(
            r"(?:\b[A-H]\s*=\s*(?:true|false|对|错)\b[^\n,;]*[,;]\s*){2,}",
            support,
            re.IGNORECASE,
        ))
        logical_chain = bool(
            re.search(
                r"\b(?:because|since|from|by|using)\b|因为|由于|根据|由",
                support,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:therefore|hence|thus|implies?|so|follows?)\b|"
                r"因此|所以|从而|故|推出|可得",
                support,
                re.IGNORECASE,
            )
        )
        validation_method = bool(re.search(
            r"\b(?:counterexample|substitut|enumerat|invariant|upper\s+bound|lower\s+bound|"
            r"equality|boundary|branch|bijection|recurrence|connected|subtour|cycle)\b|"
            r"反例|代入|枚举|不变量|上界|下界|等号|边界|分支|双射|递推|连通|子回路|闭环",
            support,
            re.IGNORECASE,
        ))
        concrete_object = bool(re.search(
            r"\d|[A-Za-z]\s*(?:=|∈|\\in\b)|\\(?:frac|binom|sum|prod|int|lim)\b|"
            r"\{[^{}]+\}|\([^()]+\)",
            support,
            re.IGNORECASE,
        ))
        decisive = relation or option_table or logical_chain or (
            validation_method and concrete_object
        )
        if not decisive:
            return False
        risks = set(getattr(spec, "risk_flags", ())) if spec is not None else set()
        if "extremal_two_sided_bound" in risks:
            bound_direction = bool(re.search(
                r"\b(?:upper|lower)\s+bound\b|\bat\s+(?:most|least)\b|"
                r"\b(?:every|all|no\s+(?:smaller|larger)|minimal|maximal)\b|"
                r"上界|下界|至多|至少|任意|所有|不存在更[小大]|最小性|最大性",
                support,
                re.IGNORECASE,
            ))
            attainment = bool(re.search(
                r"\b(?:achiev(?:e|ed|es)|attain(?:ed|s)?|equality|construction|example|"
                r"works?|satisf(?:y|ies|ied))\b|"
                r"达到|取等|等号|构造|例子|可行|满足",
                support,
                re.IGNORECASE,
            ))
            relation_count = len(re.findall(r"(?<![<>!])=(?!=)|[<>≤≥]", support))
            concrete = relation_count >= 2 or bool(re.search(
                r"\\?\{[^{}]*\d[^{}]*\}|"
                r"(?:[A-Za-z]\s*=\s*[-+]?\d[^,;\n]*[,;]\s*){2,}",
                support,
            ))
            if not (bound_direction and attainment and concrete):
                return False
        if "global_connectivity" in risks:
            connectivity = bool(re.search(r"connect(?:ed|ivity)|连通", support, re.IGNORECASE))
            subtour = bool(re.search(
                r"subtour|detached\s+cycle|extra\s+cycle|disconnected\s+cycle|"
                r"子回路|额外闭环|独立闭环",
                support,
                re.IGNORECASE,
            ))
            exact_check = bool(re.search(
                r"dynamic\s+program|\bdp\b|recurrence|transfer\s+matrix|state\s+table|"
                r"动态规划|递推|转移矩阵|状态表",
                support,
                re.IGNORECASE,
            ))
            if not (connectivity and subtour and exact_check):
                return False
        return True

    @staticmethod
    def _candidate_conflict(candidates: list[CandidateAssessment]) -> bool:
        by_stage: dict[str, CandidateAssessment] = {}
        for item in candidates:
            stage = SubmissionAgent._source_stage(item.source)
            if (
                item.validation_tier != "complete"
                or not item.answer
                or not item.shape_valid
                or not item.formatting_valid
                or stage not in {"solve", "verify", "rescue"}
            ):
                continue
            previous = by_stage.get(stage)
            if previous is None or (
                item.validation_tier == "complete",
                item.score,
                item.explicit_answer,
            ) > (
                previous.validation_tier == "complete",
                previous.score,
                previous.explicit_answer,
            ):
                by_stage[stage] = item
        first = by_stage.get("solve")
        second = by_stage.get("verify") or by_stage.get("rescue")
        return bool(first and second and not equivalent_answers(first.answer, second.answer))

    @staticmethod
    def _equivalence_pairs(candidates: list[CandidateAssessment]) -> list[dict]:
        model_candidates = [
            item for item in candidates
            if item.validation_tier in {"complete", "degraded"}
            and item.answer
            and SubmissionAgent._source_stage(item.source) in {
                "solve", "verify", "rescue", "arbitration"
            }
        ]
        return [
            {
                "left": left.source,
                "right": right.source,
                "equivalent": equivalent_answers(left.answer, right.answer),
            }
            for index, left in enumerate(model_candidates)
            for right in model_candidates[index + 1:]
        ]

    @staticmethod
    def _select(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
        usable_bases = [
            item for item in candidates
            if SubmissionAgent._source_stage(item.source) != "arbitration"
            and item.validation_tier in {"complete", "degraded"}
        ]
        eligible = list(usable_bases)
        for item in candidates:
            if SubmissionAgent._source_stage(item.source) != "arbitration":
                continue
            if item.tool_status == "pass" or any(
                equivalent_answers(item.answer, base.answer) for base in usable_bases
            ):
                eligible.append(item)
        return choose_candidate(eligible)

    @staticmethod
    def _select_rejected_consensus(
        candidates: list[CandidateAssessment],
        selected: CandidateAssessment | None,
    ) -> CandidateAssessment | None:
        """Recover an explicit conclusion corroborated by independent stages.

        Provider truncation makes a lone early box ambiguous.  Two independent
        stages agreeing on the same explicit, structurally valid conclusion is
        stronger evidence than a selected prose fragment that omits the actual
        requested value.  Only truncation/certificate reasons may be waived;
        mathematical, shape, coverage, and formatting failures remain hard.
        """
        recoverable_reasons = {
            "provider_truncated_ambiguous_box",
            "missing_verification_certificate",
        }
        eligible = [
            item for item in candidates
            if item.answer
            and item.explicit_answer
            and item.shape_valid
            and item.formatting_valid
            and item.complete_goals
            and item.tool_status != "conflict"
            and item.rejected_reasons
            and set(item.rejected_reasons) <= recoverable_reasons
        ]
        corroborated: list[tuple[int, CandidateAssessment]] = []
        for item in eligible:
            stages = {
                SubmissionAgent._source_stage(other.source)
                for other in eligible
                if other is not item
                and equivalent_answers(item.answer, other.answer)
            }
            stages.add(SubmissionAgent._source_stage(item.source))
            if len(stages) >= 2:
                corroborated.append((len(stages), item))
        if not corroborated:
            # A selected support body can contain the decisive scalar while a
            # malformed marker extracted only generic prose (for example
            # "an integer").  Prefer one unambiguous explicit numeric sentence
            # already present in that same complete body.
            return SubmissionAgent._explicit_numeric_from_selected(selected, candidates)

        _, recovered = max(
            corroborated,
            key=lambda pair: (
                pair[0],
                pair[1].verification_verdict in {"confirmed", "corrected"},
                pair[1].extraction_method in {"label_boxed", "boxed"},
                -len(pair[1].answer),
            ),
        )
        if selected is None:
            return recovered
        selected_variants = [
            result.answer
            for result in Finalizer.extract_explicit_results(selected.answer)
            if result.valid and result.answer
        ]
        selected_is_incomplete_fragment = bool(
            selected.validation_tier == "degraded"
            or not selected.complete_goals
            or (
                selected.explicit_answer
                and len(selected.answer) > len(recovered.answer) * 3
                and any(
                    equivalent_answers(variant, recovered.answer)
                    for variant in selected_variants
                )
            )
        )
        return recovered if selected_is_incomplete_fragment else None

    @staticmethod
    def _explicit_numeric_from_selected(
        selected: CandidateAssessment | None,
        candidates: list[CandidateAssessment],
    ) -> CandidateAssessment | None:
        if selected is None or selected.validation_tier != "complete":
            return None
        numbers = re.findall(
            r"(?im)^\s*(?:the\s+(?:answer|integer|value|number)|"
            r"(?:最终)?(?:答案|整数|数值|数量))\s*(?:is|equals|为|是|=|[:：])\s*"
            r"(-?\d+(?:/\d+)?)\s*[。.]?\s*$",
            selected.answer,
        )
        unique = tuple(dict.fromkeys(numbers))
        if len(unique) != 1:
            return None
        value = unique[0]
        variants = [
            item for item in candidates
            if item.answer == value
            and item.explicit_answer
            and item.shape_valid
            and item.formatting_valid
        ]
        if variants:
            return max(variants, key=lambda item: item.score)
        return None

    @staticmethod
    def _select_deadline_recovery(
        candidates: list[CandidateAssessment],
    ) -> CandidateAssessment | None:
        """Keep one uncontradicted explicit result when no call window remains."""
        eligible = [
            item for item in candidates
            if item.answer
            and item.explicit_answer
            and item.shape_valid
            and item.formatting_valid
            and item.complete_goals
            and item.tool_status != "conflict"
            and set(item.rejected_reasons) == {"provider_truncated_ambiguous_box"}
        ]
        if not eligible:
            return None
        representative = eligible[0]
        if any(
            not equivalent_answers(representative.answer, item.answer)
            for item in eligible[1:]
        ):
            return None
        return max(
            eligible,
            key=lambda item: (
                item.extraction_method in {"label_boxed", "label"},
                -len(item.answer),
            ),
        )

    @staticmethod
    def _best_stage_candidate(
        candidates: list[CandidateAssessment], stages: set[str]
    ) -> CandidateAssessment | None:
        return choose_candidate(SubmissionAgent._usable_stage_candidates(
            candidates, stages
        ))

    @staticmethod
    def _usable_stage_candidates(
        candidates: list[CandidateAssessment], stages: set[str]
    ) -> list[CandidateAssessment]:
        return [
            item for item in candidates
            if SubmissionAgent._source_stage(item.source) in stages
            and item.validation_tier in {"complete", "degraded"}
            and item.answer
            and item.shape_valid
            and item.formatting_valid
        ]

    @staticmethod
    def _declared_stage_matches(
        arbitration_variants: list[CandidateAssessment],
        target_variants: list[CandidateAssessment],
    ) -> bool:
        """Match an arbiter's concise result to any usable variant of a stage."""
        if any(
            equivalent_answers(arb.answer, target.answer)
            for arb in arbitration_variants
            for target in target_variants
        ):
            return True

        # Proof/explanation stages commonly produce two candidates: a complete
        # supported body and a concise FINAL conclusion.  The latter can be
        # degraded solely because it omits support, but it is still the right
        # object for checking an A/B declaration.  Keep this fallback narrow:
        # both sides must be explicit extractions and their entire normalized
        # mathematical conclusions must coincide.
        return any(
            arb.explicit_answer
            and target.explicit_answer
            and SubmissionAgent._same_refined_conclusion(
                arb.answer, target.answer
            )
            for arb in arbitration_variants
            for target in target_variants
        )

    @staticmethod
    def _same_refined_conclusion(left: str, right: str) -> bool:
        def normalize(value: str) -> str:
            text = str(value or "").lower()
            for _ in range(3):
                reduced = re.sub(
                    r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text,
                    flags=re.IGNORECASE,
                )
                if reduced == text:
                    break
                text = reduced
            text = re.sub(
                r"(?:对任意|任意|for\s+(?:any|every))\s*"
                r"(?:\\epsilon|epsilon|ε)\s*(?:>|\\gt)\s*0",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"(?:almost\s+everywhere|a\s*\.\s*e\s*\.|几乎处处)",
                "ae",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\b(?:and|thus|hence|therefore|consequently)\b|且|并且|从而|因此|故",
                "",
                text,
                flags=re.IGNORECASE,
            )
            replacements = {
                r"\geq": ">=", r"\ge": ">=", r"\leq": "<=", r"\le": "<=",
                r"\epsilon": "epsilon", "ε": "epsilon", r"\mu": "mu",
                r"\left": "", r"\right": "",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            text = re.sub(
                r"^(?:最终答案|答案|结论|final\s+answer|answer|conclusion)\s*[:：]?",
                "",
                text,
                flags=re.IGNORECASE,
            )
            return re.sub(r"[\s{}\\,，。；;：:`'$]", "", text)

        left_value = normalize(left)
        right_value = normalize(right)
        if not left_value or left_value != right_value:
            return False
        return bool(re.search(r"(?:=|<=|>=|<|>)", left_value))

    @staticmethod
    def _resolve_arbitration(
        response: str, candidates: list[CandidateAssessment]
    ) -> tuple[CandidateAssessment | None, str, str]:
        """Use arbitration as an A/B certificate or a checked correction."""
        variants_a = SubmissionAgent._usable_stage_candidates(candidates, {"solve"})
        variants_b = SubmissionAgent._usable_stage_candidates(
            candidates, {"verify", "rescue"}
        )
        arbitration_variants = SubmissionAgent._usable_stage_candidates(
            candidates, {"arbitration"}
        )
        candidate_a = SubmissionAgent._best_stage_candidate(candidates, {"solve"})
        candidate_b = SubmissionAgent._best_stage_candidate(
            candidates, {"verify", "rescue"}
        )
        arb = SubmissionAgent._best_stage_candidate(candidates, {"arbitration"})
        match = re.search(
            r"(?:DECISION|裁决)\s*[:：]\s*(A|B|CORRECTED|修正|UNRESOLVED|未解决)",
            str(response or ""),
            re.IGNORECASE,
        )
        decision = match.group(1).upper() if match else ""
        if decision == "未解决":
            decision = "UNRESOLVED"
        elif decision == "修正":
            decision = "CORRECTED"

        # An unresolved arbiter supplies no evidence for replacing A with a
        # conflicting verifier answer.  Keep the primary unless only B exists;
        # explicit A/B support, a certified correction, or tool evidence below
        # can still override it.
        fallback = candidate_a or candidate_b
        if decision in {"A", "B"}:
            target = candidate_a if decision == "A" else candidate_b
            target_variants = variants_a if decision == "A" else variants_b
            if target and SubmissionAgent._declared_stage_matches(
                arbitration_variants, target_variants
            ):
                return target, f"supports_{decision.lower()}", decision
            decision_label_only = any(
                SubmissionAgent._source_stage(item.source) == "arbitration"
                and re.fullmatch(decision, str(item.answer or "").strip(), re.IGNORECASE)
                for item in candidates
            )
            if target and decision_label_only:
                return target, f"label_supports_{decision.lower()}", decision
            if candidate_a and SubmissionAgent._declared_stage_matches(
                arbitration_variants, variants_a
            ):
                return candidate_a, "decision_answer_mismatch", decision
            if candidate_b and SubmissionAgent._declared_stage_matches(
                arbitration_variants, variants_b
            ):
                return candidate_b, "decision_answer_mismatch", decision
            return fallback, "decision_answer_mismatch", decision
        if decision == "UNRESOLVED":
            return fallback, "unresolved_fallback", decision
        if decision == "CORRECTED":
            if arb and candidate_a and equivalent_answers(arb.answer, candidate_a.answer):
                return candidate_a, "corrected_matches_a", decision
            if arb and candidate_b and equivalent_answers(arb.answer, candidate_b.answer):
                return candidate_b, "corrected_matches_b", decision
            if (
                arb
                and arb.validation_tier == "complete"
                and arb.shape_valid
                and arb.formatting_valid
                and SubmissionAgent._has_corrected_arbitration_support(response, match)
            ):
                return arb, "corrected_novel_answer", decision
            return fallback, "uncertified_correction_fallback", decision
        if arb and candidate_a and equivalent_answers(arb.answer, candidate_a.answer):
            return candidate_a, "implicit_supports_a", "A"
        if arb and candidate_b and equivalent_answers(arb.answer, candidate_b.answer):
            return candidate_b, "implicit_supports_b", "B"
        if arb and arb.tool_status == "pass":
            certified = next(
                (item for item in candidates if item.tool_status == "pass" and item.source == "sympy_verified"),
                arb,
            )
            return certified, "certified_tool_result", "TOOL"
        return fallback, "rejected_novel_answer", ""

    @staticmethod
    def _has_corrected_arbitration_support(
        response: str, decision_match: re.Match[str] | None
    ) -> bool:
        if decision_match is None:
            return False
        text = str(response or "")
        check = re.search(
            r"(?ims)^\s*CHECK\s*[:：]\s*(.+?)(?=^\s*(?:DECISION|裁决)\s*[:：])",
            text,
        )
        if not check or check.start() > decision_match.start():
            return False
        support = check.group(1).strip()
        return SubmissionAgent._has_audit_support(f"CHECK: {support}")

    @staticmethod
    def _source_stage(source: str) -> str:
        stage = SubmissionAgent._raw_source(source)
        return {
            "continue": "solve",
            "continue_last": "solve",
            "verify_recovered": "verify",
            "continue_verify": "verify",
            "retry_verify": "verify",
            "audit_retry": "verify",
            "last_chance": "rescue",
        }.get(stage, stage)

    @staticmethod
    def _raw_source(source: str) -> str:
        return str(source or "").split("#", 1)[0]

    @staticmethod
    def _assessment_trace(item: CandidateAssessment) -> dict:
        return {
            "answer_preview": item.answer[:600],
            "score": item.score,
            "complete_goals": item.complete_goals,
            "shape_valid": item.shape_valid,
            "formatting_valid": item.formatting_valid,
            "goal_coverage": list(item.goal_coverage),
            "result_coverage": list(item.result_coverage),
            "support_coverage": list(item.support_coverage),
            "format_coverage": list(item.format_coverage),
            "coverage_uncertain": item.coverage_uncertain,
            "tool_status": item.tool_status,
            "extraction_method": item.extraction_method,
            "raw_has_meta": item.raw_has_meta,
            "explicit_answer": item.explicit_answer,
            "verification_verdict": item.verification_verdict,
            "validation_tier": item.validation_tier,
            "rejected_reasons": list(item.rejected_reasons),
        }

    @staticmethod
    def _safe_trace_candidate(value: str) -> dict:
        text = str(value or "").strip()
        extraction = Finalizer.extract_result(text)
        return {
            "non_empty": bool(text),
            "length": len(text),
            "raw_has_meta": extraction.raw_has_meta,
            "extraction_method": extraction.method,
            "explicit_answer": extraction.explicit_answer,
            "tail": text[-1200:] if text else "",
        }

    @staticmethod
    def _finalize(response: str, spec: ProblemSpec):
        if not response.strip():
            return Finalizer.extract_result("")
        extracted = Finalizer.extract_result(response)
        explicit = extracted.answer if extracted.valid and extracted.explicit_answer else ""
        proof = spec.answer_frame.style == "proof"
        multipart = len(spec.goals) > 1
        tagged_block = SubmissionAgent._best_tagged_block(response, spec)
        support_required = any(
            requirement.category == "support"
            for goal in spec.goals
            for requirement in goal.requirements
        ) or (
            getattr(spec.profile, "task_kind", "") == "construction"
            or any(goal.kind == "construction" for goal in spec.goals)
        )
        explicit_incomplete = bool(explicit) and not assess_candidate(
            SubmissionAgent._normalize_answer(explicit, spec),
            "explicit_result_probe",
            spec,
            (),
        ).complete_goals
        safe_full_support = bool(
            not extracted.raw_has_meta
            and not Finalizer.validate_structure(response)
        )
        proof_block = tagged_block if proof else ""
        first_line = next(
            (line for line in response.splitlines() if line.strip()),
            "",
        )
        marker_only = bool(
            first_line
            and not SubmissionAgent._strip_leading_answer_marker(first_line)
        )
        if not proof and tagged_block and (
            multipart or support_required or explicit_incomplete
        ):
            support_source = response if safe_full_support else tagged_block
            answer = SubmissionAgent._support_submission(
                support_source,
                Finalizer.extract_result(tagged_block).answer or explicit,
                spec,
                max_chars=800 if explicit_incomplete else 240,
            )
            reasons = Finalizer.validate_structure(answer)
            result = ExtractionResult(
                answer if not reasons else "",
                "tagged_multipart_body" if multipart else "tagged_answer_body",
                not reasons,
                reasons,
                extracted.raw_has_meta,
                True,
            )
        elif multipart and tagged_block:
            # A multipart answer is already a bounded, validated block.  Do not
            # run the single-conclusion proof rewriter over it: that rewriter
            # intentionally replaces the last box and would collapse two boxes
            # into one value.
            answer = SubmissionAgent._strip_leading_answer_marker(tagged_block)
            reasons = Finalizer.validate_structure(answer)
            result = ExtractionResult(
                answer if not reasons else "",
                "tagged_multipart_body",
                not reasons,
                reasons,
                extracted.raw_has_meta,
                True,
            )
        elif tagged_block and "\n" in tagged_block and (marker_only or not extracted.valid):
            answer = SubmissionAgent._strip_leading_answer_marker(tagged_block)
            reasons = Finalizer.validate_structure(answer)
            result = ExtractionResult(
                answer if not reasons else "",
                "tagged_multiline_body",
                not reasons,
                reasons,
                extracted.raw_has_meta,
                True,
            )
        elif proof and (proof_block or not extracted.raw_has_meta) and not SubmissionAgent._is_just_boxed(response):
            # A normal proof may put its answer marker at the end, so retaining
            # only the tagged suffix would discard the actual argument. The
            # suffix is a safety boundary only when a meta preamble exists.
            proof_source = proof_block if extracted.raw_has_meta else response
            block_explicit = Finalizer.extract_result(proof_block).answer if proof_block else explicit
            answer = SubmissionAgent._proof_submission(
                proof_source,
                block_explicit or explicit,
                SubmissionAgent._answer_language(spec) == "en",
            )
            reasons = Finalizer.validate_structure(answer)
            recoverable = bool(explicit) and bool(reasons) and set(reasons) <= {
                "trailing_fragment", "truncated_sentence"
            }
            result = ExtractionResult(
                explicit if recoverable else (answer if not reasons else ""),
                "proof_conclusion_fallback" if recoverable else "proof_body",
                recoverable or not reasons,
                ("proof_body_truncated_recovered",) if recoverable else reasons,
                extracted.raw_has_meta,
                bool(recoverable),
            )
        else:
            result = extracted
        normalized = SubmissionAgent._render_answer(SubmissionAgent._normalize_answer(result.answer, spec), spec)
        normalized = SubmissionAgent._normalize_answer(normalized, spec)
        normalized_reasons = Finalizer.validate_structure(normalized)
        return ExtractionResult(
            normalized if not normalized_reasons else "",
            result.method,
            result.valid and not normalized_reasons,
            tuple(dict.fromkeys((*result.rejected_reasons, *normalized_reasons))),
            result.raw_has_meta, result.explicit_answer,
        )

    @staticmethod
    def _best_tagged_block(response: str, spec: ProblemSpec) -> str:
        blocks = Finalizer.extract_tagged_submissions(response)
        if not blocks:
            return ""
        ranked = []
        for index, block in enumerate(blocks):
            explicit = Finalizer.extract_result(block).answer
            answer = SubmissionAgent._proof_submission(
                block,
                explicit,
                SubmissionAgent._answer_language(spec) == "en",
            )
            normalized = SubmissionAgent._normalize_answer(answer, spec)
            assessment = assess_candidate(normalized, "tagged_block", spec, ())
            ranked.append((
                assessment.complete_goals,
                assessment.accepted,
                assessment.formatting_valid,
                assessment.score,
                len(normalized),
                index,
                block,
            ))
        return max(ranked)[-1]

    @staticmethod
    def _should_retrieve(spec: ProblemSpec) -> bool:
        problem_text = " ".join((
            getattr(spec, "problem_text", ""),
            *(goal.instruction for goal in spec.goals),
        ))
        if spec.profile.answer_shape in {"choice", "truth"}:
            return True
        if re.search(
            r"时间序列|回归|异方差|统计|中心差分|有限差分|有限元|辛普森|高斯求积|"
            r"Lempel[- ]?Ziv|LZ78|newton(?:'s)? method|simpson(?:'s)? rule|"
            r"gauss[- ]legendre|quadrature|interpolat",
            problem_text,
            re.IGNORECASE,
        ):
            return True
        if getattr(spec.profile, "topic", "general").startswith("olympiad_"):
            return True
        theoretical = {
            "抽象代数", "拓扑学", "泛函分析", "复分析", "常微分方程",
            "数学分析", "离散数学", "数论", "初等几何", "高等代数", "数值分析",
        }
        risk_driven = {"multiple_goals", "theorem_scope", "definition_or_structure_conditions", "construction_validation"}
        return spec.profile.subject in theoretical and (
            spec.profile.difficulty == "hard" or bool(set(spec.risk_flags) & risk_driven)
        )

    @staticmethod
    def _append_proof_trace(
        trace: list[dict],
        spec: ProblemSpec,
        selected: CandidateAssessment | None,
        first: str,
        second: str,
        arbitration: str,
        verification_completion: str = "",
    ) -> None:
        if spec.profile.problem_type not in {"proof", "derivation", "explanation"} or not selected:
            return
        raw_source = SubmissionAgent._raw_source(selected.source)
        raw = (verification_completion or arbitration) if raw_source in {"continue_verify", "audit_retry"} else (
            arbitration if raw_source in {
                "arbitration", "continue_last", "last_chance", "verify_recovered",
                "retry_verify",
            } else (
                second if raw_source in {"rescue", "verify", "continue"} else first
            )
        )
        if raw.strip():
            trace.append({"step": "proof_summary", "content": Finalizer.extract_solution(raw)[:1600]})

    @staticmethod
    def _normalize_answer(answer: str, spec: ProblemSpec) -> str:
        value = str(answer or "").strip().replace("\x08ar", r"\bar").replace(r"\infty", "∞")
        value = re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", value)
        value = re.sub(r"\\left(?=\s*[^A-Za-z])", "", value)
        value = re.sub(r"\\right(?=\s*[^A-Za-z])", "", value)
        profile = getattr(spec, "profile", spec)
        if profile.answer_shape == "roots":
            matched = re.fullmatch(r"\[\s*(-?\d+(?:/\d+)?)\s*[,，]\s*(-?\d+(?:/\d+)?)\s*\]", value)
            if matched:
                return f"x={matched.group(1)} 或 x={matched.group(2)}"
        return value

    @staticmethod
    def _render_answer(answer: str, spec: ProblemSpec) -> str:
        """Turn bare scalar answers into the wording requested by a text problem."""
        value = str(answer or "").strip()
        contract = getattr(spec, "answer_contract", None)
        if (
            contract is not None
            and getattr(contract, "mode", "") == "answer_only"
            and getattr(contract, "wrapper", "") == "boxed"
        ):
            return value
        if not value or spec.answer_frame.style != "sentence":
            return value
        frame = spec.answer_frame
        if frame.question_kind == "age":
            if frame.subject and frame.subject in value and frame.unit in value:
                return SubmissionAgent._as_sentence(value)
            scalar = SubmissionAgent._last_scalar(value)
            return SubmissionAgent._as_sentence(f"{frame.subject}{scalar}{frame.unit}") if scalar else SubmissionAgent._as_sentence(value)
        if frame.question_kind == "count":
            if frame.unit in value:
                return SubmissionAgent._as_sentence(value)
            return SubmissionAgent._as_sentence(f"所求数量为{value}{frame.unit}")
        if frame.question_kind == "probability":
            if re.search(r"概率|\bprobability\b|P\s*\(", value, re.IGNORECASE):
                return SubmissionAgent._as_sentence(value)
            prefix = (
                "The requested probability is "
                if SubmissionAgent._answer_language(spec) == "en"
                else "所求概率为"
            )
            return SubmissionAgent._as_sentence(f"{prefix}{value}")
        if frame.question_kind == "truth":
            compact = re.sub(r"[\s。.]", "", value)
            normalized = {
                "是": "是", "正确": "是", "成立": "是", "可以": "可以",
                "否": "否", "错误": "否", "不成立": "否", "不可以": "不可以",
            }
            # Preserve required checks attached to a verdict.  Collapsing a
            # certified statement such as "否；逐点极限为0，但||f_n||_1=1"
            # to the first word would discard gradable obligations.
            judgement = normalized.get(compact, value)
            if compact not in normalized:
                return SubmissionAgent._as_sentence(value)
            if frame.subject and compact in normalized:
                return SubmissionAgent._as_sentence(f"{frame.subject}：{judgement}")
            return SubmissionAgent._as_sentence(judgement)
        return value

    @staticmethod
    def _proof_submission(response: str, explicit: str, english: bool = False) -> str:
        """Keep the argument while replacing model-only final-answer markers."""
        value = Finalizer.extract_solution(response)
        conclusion_label = "Conclusion: " if english else "结论："
        boxes = [item for item in Finalizer._boxed_values(value) if item[2]]
        labelled_boxes = []
        for item in boxes:
            line_start = value.rfind("\n", 0, item[0]) + 1
            prefix = value[line_start:item[0]]
            if re.search(
                r"(?:【\s*(?:最终答案|答案|结论)\s*】|(?:最终\s*)?答案|结论|"
                r"FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION)\s*[:：=]?\s*"
                r"(?:\\\[|\$\$?|\\\()?\s*$",
                prefix,
                re.IGNORECASE,
            ):
                labelled_boxes.append(item)
        # If an explicit answer label exists, rewrite its box only.  A later
        # box can be an intermediate check and must retain its own value.
        selected_box = labelled_boxes[-1] if labelled_boxes else (boxes[-1] if boxes else None)
        boxed_at = selected_box[0] if selected_box else -1
        if boxed_at >= 0 and explicit:
            brace_at = boxed_at + len(r"\boxed")
            depth = 0
            boxed_end = -1
            for index in range(brace_at, len(value)):
                if value[index] == "{":
                    depth += 1
                elif value[index] == "}":
                    depth -= 1
                    if depth == 0:
                        boxed_end = index + 1
                        break
            if boxed_end > boxed_at:
                line_start = value.rfind("\n", 0, boxed_at) + 1
                prefix = value[line_start:boxed_at]
                labelled = bool(re.search(
                    r"(?:【\s*(?:最终答案|答案|结论)\s*】|(?:最终\s*)?答案|结论|"
                    r"FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION)\s*[:：]?\s*$",
                    prefix,
                    re.IGNORECASE,
                ))
                replacement = explicit if labelled or prefix.strip() else f"{conclusion_label}{explicit}"
                value = value[:boxed_at] + replacement + value[boxed_end:]
        value = re.sub(
            r"(?im)^\s*(?:【最终答案】|(?:最终)?答案|结论|FINAL(?:\s*ANSWER)?|ANSWER|CONCLUSION)\s*[:：]?\s*",
            conclusion_label,
            value,
        )
        return value.strip()

    @staticmethod
    def _support_submission(
        response: str,
        explicit: str,
        spec: ProblemSpec,
        max_chars: int = 240,
    ) -> str:
        """Keep only the shortest contract-complete support for a non-proof answer."""
        english = SubmissionAgent._answer_language(spec) == "en"
        full = (
            SubmissionAgent._strip_leading_answer_marker(response)
            if len(spec.goals) > 1
            else SubmissionAgent._proof_submission(response, explicit, english)
        )
        explicit_value = str(explicit or "").strip()
        if not explicit_value:
            return full

        label = "Conclusion: " if english else "结论："
        base = f"{label}{explicit_value}"
        explicit_compact = re.sub(r"\s+", "", explicit_value).casefold()
        units = tuple(
            unit for unit in SubmissionAgent._support_units(response)
            if re.sub(r"\s+", "", unit).casefold() != explicit_compact
        )

        def multipart_items(value: str) -> int:
            # Generic formula requirements can otherwise let one formula appear
            # to cover every part. Newlines and explicit list separators are the
            # stable, language-independent evidence available at this stage.
            return 1 + len(re.findall(r"\n|[、；;]", str(value or "")))

        def complete(value: str) -> bool:
            if not value or Finalizer.validate_structure(value):
                return False
            if len(spec.goals) > 1 and multipart_items(value) < len(spec.goals):
                return False
            normalized = SubmissionAgent._normalize_answer(value, spec)
            candidate_complete = assess_candidate(
                normalized, "support_compaction", spec, ()
            ).accepted
            explicit_support_complete = all(
                requirement.matches(normalized)
                for goal in spec.goals
                for requirement in goal.support_requirements
            )
            return candidate_complete and explicit_support_complete

        templated = bool(re.search(
            r"(?im)^\s*(?:#{1,6}\s*|\d+[.)、]\s*|[-+*]\s+|\*{0,2}(?:"
            r"解题步骤|解题过程|步骤|问题描述|solution\s+steps?|steps?)\b)",
            str(response or ""),
        ))
        if len(spec.goals) == 1 and len(full) <= max_chars and not templated and complete(full):
            return full

        def add_unit(value: str, unit: str) -> str:
            return f"{value}\n{unit}" if unit else value

        compact_candidates: list[str] = []
        if complete(base):
            compact_candidates.append(base)
        for unit in units:
            candidate = add_unit(base, unit)
            if len(candidate) <= max_chars and complete(candidate):
                compact_candidates.append(candidate)

        # A multipart contract or a compound support obligation may need two
        # separate statements. Limit pair search to bounded, already-clean units.
        bounded_units = [unit for unit in units if len(unit) <= max_chars][:32]
        for left_index, left in enumerate(bounded_units):
            for right in bounded_units[left_index + 1:]:
                candidate = add_unit(add_unit(base, left), right)
                if len(candidate) <= max_chars and complete(candidate):
                    compact_candidates.append(candidate)

        if compact_candidates:
            return min(
                compact_candidates,
                key=lambda value: (len(value), value.count("\n")),
            )

        # Some contracts combine more than two independently stated facts.
        # Greedily retain only lines that improve requirement/goal coverage.
        current = base
        remaining = list(bounded_units)

        def progress(value: str) -> tuple[int, int]:
            requirements = sum(
                requirement.matches(value)
                for goal in spec.goals
                for requirement in goal.requirements
            )
            coverage = sum(
                all(requirement.matches(value) for requirement in goal.requirements)
                and all(
                    re.sub(r"\s+", "", term).lower()
                    in re.sub(r"\s+", "", value).lower()
                    for term in goal.required_terms
                )
                for goal in spec.goals
            )
            return coverage, requirements

        while remaining and len(current) < max_chars:
            before = progress(current)
            ranked = sorted(
                (
                    (progress(add_unit(current, unit)), -len(unit), index, unit)
                    for index, unit in enumerate(remaining)
                    if len(add_unit(current, unit)) <= max_chars
                ),
                reverse=True,
            )
            if not ranked or ranked[0][0] <= before:
                break
            _, _, index, unit = ranked[0]
            current = add_unit(current, unit)
            remaining.pop(index)
            if complete(current):
                return current

        # Preserve the previous behavior when no bounded subset can satisfy the
        # contract. Completeness is more important than cosmetic compression.
        return base if Finalizer.validate_structure(full) else full

    @staticmethod
    def _support_units(response: str) -> tuple[str, ...]:
        """Split a tagged response into clean method/equation statements."""
        value = Finalizer.extract_solution(response).replace("\r\n", "\n")
        value = re.sub(r"```[^\n]*\n.*?```", "", value, flags=re.DOTALL)
        units: list[str] = []
        seen: set[str] = set()
        answer_marker = re.compile(
            r"^(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:(?:最终\s*)?答案|结论|(?:the\s+)?final(?:\s+answer)?|"
            r"answer|conclusion)\s*[:：为=])",
            re.IGNORECASE,
        )
        template_heading = re.compile(
            r"^(?:解题步骤|解题过程|步骤|过程|问题描述|分析|计算(?:过程)?|"
            r"solution\s+steps?|steps?|working|derivation)\s*[:：]?\s*(.*)$",
            re.IGNORECASE,
        )
        method_heading = re.compile(r"^(?:方法|method)\s*[:：]\s*(.*)$", re.IGNORECASE)

        for raw_line in value.splitlines():
            line = raw_line.strip()
            if not line or line in {r"\[", r"\]", "```"}:
                continue
            if answer_marker.match(line):
                continue
            line = re.sub(r"^\s*(?:#{1,6}\s*|\d+[.)、]\s*|[-+*]\s+)", "", line)
            line = line.replace("**", "").replace("__", "").strip()
            line = re.sub(r"^\\\[\s*|\s*\\\]$", "", line).strip()

            heading = template_heading.match(line)
            if heading:
                line = heading.group(1).strip()
                if not line:
                    continue
            method = method_heading.match(line)
            if method:
                line = method.group(1).strip()
                if not line:
                    continue
            line = line.rstrip("：:").strip()
            if not line:
                continue

            # Long prose paragraphs are only useful through their individual
            # assertions; splitting also prevents a heading from dragging in a
            # full derivation. Mathematical display lines remain intact.
            parts = re.split(r"(?<=[。！？!?；;])\s*", line)
            for part in parts:
                unit = re.sub(r"\s+", " ", part).strip()
                if not unit or answer_marker.match(unit):
                    continue
                key = unit.casefold()
                if key in seen:
                    continue
                seen.add(key)
                units.append(unit)
        return tuple(units)

    @staticmethod
    def _strip_leading_answer_marker(value: str) -> str:
        return re.sub(
            r"(?im)^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)\s*(?:is|equals|[:：=])|"
            r"(?:最终\s*)?答案\s*[:：为=]|结论\s*[:：为=])"
            r"[ \t]*(?:\*{1,3}|_{1,3})?[ \t]*",
            "",
            str(value or ""),
            count=1,
        ).strip()

    @staticmethod
    def _last_scalar(value: str) -> str:
        numbers = re.findall(r"[+-]?(?:\d+(?:/\d+)?|\d*\.\d+)", value)
        return numbers[-1] if numbers else ""

    @staticmethod
    def _as_sentence(value: str) -> str:
        cleaned = str(value or "").strip()
        return cleaned if cleaned.endswith(("。", "！", "？", ".", "!", "?")) else f"{cleaned}。"

    @staticmethod
    def _review_evidence(candidate: str) -> str:
        value = str(candidate or "").strip()
        extraction = Finalizer.extract_result(value)
        if not extraction.valid:
            return "（第一轮无可提交结论；请独立重解。）"
        if extraction.raw_has_meta:
            return extraction.answer
        return value[-3800:]

    @staticmethod
    def _best_effort_answer(
        responses: tuple[tuple[str, bool], ...],
        spec: ProblemSpec,
        problem: str,
    ) -> tuple[str, str]:
        """Return a gradable fragment instead of a guaranteed-invalid message."""
        hard_reasons = {
            "placeholder", "meaningless_fragment", "markup_fragment", "meta_text",
            "unclosed_inline_math", "unclosed_inline_latex", "unclosed_display_latex",
            "unclosed_latex_environment", "unclosed_latex_brace", "unclosed_code_fence",
        }
        for response, provider_truncated in responses:
            if not str(response or "").strip():
                continue
            if provider_truncated:
                stable_prefix = SubmissionAgent._stable_truncated_prefix(
                    str(response), spec
                )
                if stable_prefix:
                    return stable_prefix, "degraded_stable_final_prefix"
            has_unclosed_box = any(
                not complete
                for _, _, complete in Finalizer._boxed_values(str(response))
            )
            if provider_truncated and has_unclosed_box:
                continue
            finalized = SubmissionAgent._finalize(str(response), spec)
            ambiguous_truncated_box = (
                provider_truncated
                and finalized.method in {"boxed", "boxed_unclosed"}
                and not SubmissionAgent._is_just_boxed(str(response))
            )
            if finalized.answer and (
                finalized.explicit_answer
                or (
                    not provider_truncated
                    and SubmissionAgent._is_standalone_unlabelled_result(str(response))
                )
            ) and (
                not provider_truncated or not ambiguous_truncated_box
            ):
                return finalized.answer, "degraded_finalized_candidate"
            for block in reversed(Finalizer.extract_tagged_submissions(str(response))):
                result = Finalizer.extract_result(block)
                if result.answer and not (set(Finalizer.validate_structure(result.answer)) & hard_reasons):
                    return result.answer, "degraded_explicit_block"
            # Do not mine an arbitrary mathematical-looking tail line from a
            # failed draft.  It is commonly a hypothesis, intermediate value,
            # or abandoned branch.  Explicit blocks and validated standalone
            # responses have already been handled above.

        shape = spec.profile.answer_shape
        if shape == "choice":
            neutral = "A"
        elif shape == "truth":
            neutral = r"\text{No}" if SubmissionAgent._answer_language(spec) == "en" else "否"
        elif shape in {"roots", "interval"}:
            neutral = r"\varnothing"
        elif shape == "matrix":
            neutral = r"\begin{pmatrix}0\end{pmatrix}"
        else:
            neutral = "0"
        return neutral, "degraded_all_empty"

    @staticmethod
    def _is_standalone_unlabelled_result(value: str) -> bool:
        """Accept a bare scalar/formula, never a prose derivation fragment."""
        text = str(value or "").strip().strip("`")
        if not text or len(text) > 240 or "\n" in text or Finalizer.contains_meta(text):
            return False
        if re.fullmatch(
            r"(?:正确|错误|是|否|成立|不成立|无解|不存在|"
            r"true|false|yes|no|no\s+solutions?)",
            text,
            re.IGNORECASE,
        ):
            return True
        return bool(re.fullmatch(
            r"[A-Za-z0-9_{}()[\].,+\-*/^=<>≤≥∈|\\\s]+",
            text,
        )) and bool(re.search(r"\d|[=<>≤≥∈]|\\(?:frac|sqrt|boxed|varnothing)", text))

    @staticmethod
    def _contract_value(spec: ProblemSpec, name: str, default: str = "") -> str:
        contract = getattr(spec, "answer_contract", None) or getattr(spec, "contract", None)
        aliases = {
            "wrapper": ("wrapper", "output_wrapper"),
            "mode": ("mode", "output_mode"),
            "language": ("language",),
        }
        for field in aliases.get(name, (name,)):
            value = getattr(contract, field, "") if contract is not None else ""
            if value:
                return str(value)
        return default

    @staticmethod
    def _answer_language(spec: ProblemSpec) -> str:
        """Use the answer contract, whose language inference ignores math notation."""
        return SubmissionAgent._contract_value(spec, "language", spec.profile.language)

    @staticmethod
    def _render_submission(answer: str, spec: ProblemSpec, problem: str) -> str:
        value = str(answer or "").strip()
        if not value:
            value = "0"
        wrapper = SubmissionAgent._contract_value(spec, "wrapper", "")
        if not wrapper and re.search(
            r"(?:within|inside)\s+\\boxed\s*\{\s*\}|put.*final answer.*\\boxed|\\boxed\s*\{\s*\}",
            problem,
            re.IGNORECASE,
        ):
            wrapper = "boxed"
        if wrapper not in {"boxed", "box"}:
            return value

        if SubmissionAgent._is_just_boxed(value) and not Finalizer.validate_structure(value):
            return value

        mode = SubmissionAgent._contract_value(spec, "mode", "answer_only")
        multipart = len(spec.goals) > 1
        if mode in {"answer_only", "answer_with_support"} and multipart and "\n" in value:
            content = re.sub(
                r"(?im)^\s*(?:FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION|最终答案|答案|结论)\s*[:：]?\s*",
                "",
                value,
                count=1,
            ).strip()
            unwrapped_lines = []
            for line in content.splitlines():
                line_result = Finalizer.extract_result(line.strip())
                if (
                    SubmissionAgent._is_just_boxed(line)
                    and line_result.valid
                    and line_result.explicit_answer
                ):
                    unwrapped_lines.append(line_result.answer)
                else:
                    unwrapped_lines.append(line)
            content = "\n".join(unwrapped_lines).strip()
            return f"\\boxed{{{content}}}"
        extracted = Finalizer.extract_result(value)
        explicit = extracted.answer if extracted.answer and extracted.explicit_answer else ""
        if mode in {"proof", "answer_with_support"} and "\n" in value:
            conclusion = explicit or next(
                (line.strip() for line in reversed(value.splitlines()) if line.strip()),
                value,
            )
            if r"\boxed{" in value:
                return value
            return f"{value}\n\\boxed{{{conclusion}}}"

        content = explicit or value
        content = re.sub(
            r"(?i)^\s*(?:FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION|最终答案|答案|结论)\s*[:：]?\s*",
            "",
            content,
        ).strip()
        inner = Finalizer.extract_result(content)
        if inner.answer and inner.explicit_answer:
            content = inner.answer
        return f"\\boxed{{{content}}}"

    @staticmethod
    def _is_just_boxed(response: str) -> bool:
        value = response.strip()
        boxes = Finalizer._boxed_values(value)
        return bool(
            len(boxes) == 1
            and boxes[0][2]
            and re.fullmatch(
                r"(?:\\\[|\$\$?)?\s*\\boxed\{.*\}\s*(?:\\\]|\$\$?)?",
                value,
                re.DOTALL,
            )
        )

    @staticmethod
    def _remaining_ms(started_at: float) -> int:
        return max(0, int((SUBMISSION_SOFT_BUDGET_SECONDS - (monotonic() - started_at)) * 1000))
