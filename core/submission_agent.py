"""Bounded, retrieval-assisted implementation used by the public entry point."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
from time import monotonic

from classifier.problem_spec import ProblemSpec, build_problem_spec
from core.model_response import coerce_model_response
from core.stage_budget import StageBudget, plan_stage_budget
from rag.card_retriever import CardRetriever, RetrievalBundle
from reasoning.candidate_selector import (
    CandidateAssessment,
    ToolEvidence,
    assess_candidate,
    choose_candidate,
)
from reasoning.finalizer import ExtractionResult, Finalizer
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool


SUBMISSION_SOFT_BUDGET_SECONDS = 270
CERTIFIED_WHOLE_OPERATIONS = {
    "curve_speed",
    "first_fundamental_form",
    "graph_gaussian_curvature",
    "minimum_degree_path_proof",
    "nonadjacent_binary_string_count",
    "precedence_permutation_count",
    "surjection_count",
    "planar_euler_faces",
    "paraboloid_curvature",
    "ordered_positive_triples",
    "pde_verification",
    "propositional_implication_chain",
    "simple_random_walk_moments",
    "complete_graph_cover_time",
    "two_venue_capacity",
    "circle_laplacian",
    "central_difference",
    "number_writing_game",
    "path_independent_set_partition",
    "rational_f2_constraint",
    "digit_sum_window",
    "lz78_encoding",
    "spike_sequence_construction",
    "dependent_bernoulli_construction",
}
CERTIFIED_MULTI_GOAL_OPERATIONS = {
    "curve_speed", "lz78_encoding", "dependent_bernoulli_construction",
}


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
        evidence = self._tool_evidence(self.sympy.hints_for(text), spec)
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
        review_mode = self._review_mode(
            spec, first_candidates, first, budget, started_at, text, first_truncated
        )
        trace.append({
            "step": "review_admission",
            "content": {
                "admitted": bool(review_mode),
                "mode": review_mode or "none",
                "remaining_budget_ms": self._remaining_ms(started_at),
            },
        })
        second = ""
        second_truncated = False
        if review_mode:
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
            # Verify it with a fresh solve that never sees the candidate.
            repair_mode = "verify_recovered"
        elif review_mode in {"verify", "rescue"} and not usable_second:
            if second.strip() and second_truncated:
                # Recover the independently computed verifier conclusion without
                # starting another long reasoning chain.
                repair_mode = "continue_verify"
            elif selected_before_repair is not None:
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
        if repair_mode and budget.allow_repair and optional_arbitration_allowed:
            continuing_verifier = repair_mode == "continue_verify"
            independent_verifier = repair_mode in {"verify_recovered", "retry_verify"}
            if continuing_verifier or independent_verifier:
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
                budget.review_tokens if independent_verifier else (
                    budget.repair_tokens if repair_mode == "arbitration"
                    else budget.emergency_tokens
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
                and self._raw_source(item.source) == repair_mode
                for item in candidates
            )
            if (
                independent_verifier
                and third_truncated
                and arbitration.strip()
                and not usable_third
            ):
                verification_completion, fourth_truncated = self._call(
                    repair_request,
                    "continue_verify",
                    budget.emergency_tokens,
                    trace,
                    started_at,
                    prior_response=arbitration,
                    followup=self._continuation_instruction(spec, final_attempt=True),
                    thinking_mode=False,
                )
                candidates.extend(self._assess_candidates(
                    "", "", verification_completion, "", spec, evidence,
                    third_source="continue_verify",
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
        selected = self._select(candidates)
        verified_recovery = None
        if repair_mode in {"verify_recovered", "continue_verify", "retry_verify"}:
            verified_recovery = self._select([
                item for item in candidates
                if self._raw_source(item.source) in {repair_mode, "continue_verify"}
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
                "recovered_answer_verification": (
                    verified_recovery.validation_tier if verified_recovery else "unresolved"
                ) if repair_mode in {
                    "verify_recovered", "continue_verify", "retry_verify"
                } else "not_required",
            },
        })
        trace.append({
            "step": "finalize", "content": {
                "non_empty": bool(answer),
                "answer_shape": spec.profile.answer_shape,
                "source": selected.source if selected else "fallback",
                "contract_wrapper": self._contract_value(spec, "wrapper", "none"),
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
        except Exception as exc:  # The platform client owns retries and limits.
            trace.append({
                "step": f"model_call_{stage}",
                "content": {
                    "status": "failed",
                    "type": type(exc).__name__,
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
            return "Put the complete mathematical answer first as FINAL: \\boxed{...}."

    @staticmethod
    def _solve_request(problem: str, spec: ProblemSpec, cards: RetrievalBundle, evidence: tuple[ToolEvidence, ...]) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        header = "Problem" if english else "题目"
        obligations = "Required answer obligations" if english else "必须覆盖的作答要求"
        content = f"{header}:\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}"
        content += f"\n{obligations}:\n" + SubmissionAgent._goal_context(spec)
        if cards.solve_context():
            content += (
                "\nCurated domain facts: apply every fact that directly matches the problem, and check its assumptions.\n"
                if english else
                "\n经本地校订的领域事实：直接适用时必须使用，并核对其前提。\n"
            ) + cards.solve_context()
        if evidence:
            content += ("\nVerified local evidence:\n" if english else "\n已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence)
        content += (
            "\nFirst line: FINAL: \\boxed{the complete answer}. Then give only essential support."
            if english else
            "\n第一行必须写 FINAL: \\boxed{完整答案}，之后只保留必要依据。"
        )
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
        content = (
            f"Problem:\n{problem}\n\nSolve independently from scratch using a different method where possible. "
            "Do not assume or reconstruct another solver's answer. Check every requested value, root, condition, unit, and extremal case.\n"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
            if english else
            f"题目：\n{problem}\n\n请从头独立重算，尽量采用不同方法，不要猜测或沿用另一位求解者的答案。"
            f"逐项检查数值、全部根、条件、单位和极端情形。\n必须覆盖的作答要求：\n{SubmissionAgent._goal_context(spec)}\n"
        )
        if cards.solve_context():
            content += (
                "Curated domain facts: use directly applicable facts and check each option or definition against them.\n"
                if english else
                "经本地校订的领域事实：必须用直接适用的事实逐项核对定义或选项。\n"
            ) + cards.solve_context() + "\n"
        if cards.review_context():
            content += ("Alternative method hint:\n" if english else "备选方法提示：\n") + cards.review_context() + "\n"
        if evidence:
            content += ("Verified local evidence:\n" if english else "已核验的本地证据：\n") + SubmissionAgent._evidence_context(evidence) + "\n"
        content += (
            "First line: FINAL: \\boxed{the complete independently computed answer}. Then give only essential checks."
            if english else
            "第一行必须写 FINAL: \\boxed{独立算出的完整答案}，之后只写必要核验。"
        )
        return content

    @staticmethod
    def _arbitration_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        candidates: list[CandidateAssessment],
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        english = SubmissionAgent._answer_language(spec) == "en"
        rendered = "\n".join(
            f"Candidate {index + 1}: {item.answer}" if english else f"候选{index + 1}：{item.answer}"
            for index, item in enumerate(candidates)
            if item.validation_tier in {"complete", "degraded"}
            and item.answer
            and item.shape_valid
            and item.formatting_valid
            and SubmissionAgent._source_stage(item.source) in {"solve", "verify", "rescue"}
        )
        content = (
            f"Problem:\n{problem}\n\nThe candidate answers conflict. Recompute the decisive steps and adjudicate mathematically; never choose by length or style.\n"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n{rendered}\n"
            if english else
            f"题目：\n{problem}\n\n候选答案发生实质冲突。请重算关键步骤并按数学正确性裁决，不要按长度或措辞选择。\n"
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
            "First line: FINAL: \\boxed{the adjudicated complete answer}."
            if english else "第一行必须写 FINAL: \\boxed{裁决后的完整答案}。"
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
            f"Problem:\n{problem}\n\nReturn the actual mathematical answer immediately. Use at most six lines. "
            f"First line: FINAL: \\boxed{{complete answer}}. Include only explicitly required support.\n"
            f"Required answer obligations:\n{SubmissionAgent._goal_context(spec)}\n"
            if english else
            f"题目：\n{problem}\n\n立即给出实际数学答案，最多六行。第一行必须写 FINAL: \\boxed{{完整答案}}，"
            f"其余只保留题目明确要求的依据。\n必须覆盖：\n{SubmissionAgent._goal_context(spec)}\n"
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
        for goal in spec.goals:
            suffix = checks.get(goal.kind, "完整可判分结论")
            if any(
                requirement.name == "exhaustive_result"
                for requirement in goal.requirements
            ):
                suffix += (
                    "; enumerate every possibility and state that there are no others"
                    if english else
                    "；列全所有可能并明确无其他情形"
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
    def _tool_evidence(hints: list[str], spec: ProblemSpec) -> tuple[ToolEvidence, ...]:
        supported = {
            "SymPy 计算": "calculate", "SymPy 方程解": "solve_equation", "SymPy 导数": "derivative",
            "SymPy 定积分": "definite_integral", "SymPy 不定积分": "integral", "SymPy 极限": "limit",
            "SymPy 递推通项": "recurrence_solution", "SymPy 曲线速度": "curve_speed",
            "SymPy 第一基本形式": "first_fundamental_form",
            "本地高斯曲率公式": "graph_gaussian_curvature",
            "本地图论路径证明": "minimum_degree_path_proof",
            "本地不相邻二进制串计数": "nonadjacent_binary_string_count",
            "本地不相邻二进制串核验": "nonadjacent_binary_string_count_check",
            "本地排列条件计数": "precedence_permutation_count",
            "本地排列条件计数核验": "precedence_permutation_count_check",
            "本地满射容斥计数": "surjection_count",
            "本地满射容斥核验": "surjection_count_check",
            "本地平面图欧拉答案": "planar_euler_faces",
            "本地平面图欧拉核验": "planar_euler_faces_check",
            "本地抛物面曲率答案": "paraboloid_curvature",
            "本地抛物面曲率核验": "paraboloid_curvature_check",
            "SymPy PDE核验": "pde_verification",
            "本地有序三元组计数": "ordered_positive_triples",
            "本地随机游走矩": "simple_random_walk_moments",
            "本地完全图覆盖时间": "complete_graph_cover_time",
            "本地二项分布容量": "two_venue_capacity",
            "本地圆周拉普拉斯": "circle_laplacian",
            "本地中心差分": "central_difference",
            "本地有理数约束传播答案": "rational_f2_constraint",
            "本地有理数约束传播核验": "rational_f2_constraint_check",
            "本地数位和窗口答案": "digit_sum_window",
            "本地数位和窗口最小性核验": "digit_sum_window_check",
            "本地取数博弈答案": "number_writing_game",
            "本地取数博弈状态核验": "number_writing_game_state_check",
            "本地路径配分函数答案": "path_independent_set_partition",
            "本地路径配分函数递推核验": "path_partition_recurrence_check",
            "本地LZ78编码答案": "lz78_encoding",
            "本地LZ78编码核验": "lz78_encoding_check",
            "本地尖峰函数构造答案": "spike_sequence_construction",
            "本地尖峰函数构造核验": "spike_sequence_construction_check",
            "本地Bernoulli依赖构造答案": "dependent_bernoulli_construction",
            "本地Bernoulli依赖构造核验": "dependent_bernoulli_construction_check",
            "本地命题逻辑推导": "propositional_implication_chain",
        }
        evidence = []
        for hint in hints:
            label, separator, result = hint.partition(": ")
            if not separator or not result.strip():
                continue
            operation = supported.get(label, "local_hint")
            certified_whole = (
                operation in CERTIFIED_WHOLE_OPERATIONS
                and SubmissionAgent._certified_operation_covers_contract(operation, spec)
            )
            whole = certified_whole or (
                operation != "local_hint"
                and not operation.endswith("_check")
                and spec.tool_can_answer_whole
            )
            # A supported local operation is verified evidence even when it is
            # only a substep. Scope, not verification, controls whether it may
            # become the complete submitted answer.
            evidence.append(ToolEvidence(
                result.strip(), "whole_goal" if whole else "subexpression", operation,
                operation != "local_hint",
            ))
        return tuple(evidence)

    @staticmethod
    def _certified_operation_covers_contract(operation: str, spec: ProblemSpec) -> bool:
        """Prevent a verified subcalculation from bypassing unrelated answer parts."""
        contract = getattr(spec, "answer_contract", None)
        parts = tuple(getattr(contract, "parts", ())) if contract is not None else ()
        goal_count = max(len(spec.goals), len(parts))
        if operation == "lz78_encoding":
            requirements = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
                if requirement.strict
            }
            return (
                1 <= goal_count <= 2
                and requirements == {"phrase_decomposition", "encoded_string"}
            )
        if operation == "spike_sequence_construction":
            requirements = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            return (
                1 <= goal_count <= 2
                and any(goal.kind == "construction" for goal in spec.goals)
                and all(goal.kind in {"construction", "formula"} for goal in spec.goals)
                and bool(requirements & {"integral_result", "integral_value"})
                and requirements <= {"integral_result", "integral_value", "pointwise_limit"}
            )
        if operation == "dependent_bernoulli_construction":
            requirements = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            has_probability_target = "target_p" in requirements or any(
                re.search(r"P\s*\(\s*X\s*=\s*Y\s*\)", goal.instruction, re.IGNORECASE)
                for goal in spec.goals
            )
            return (
                1 <= goal_count <= 2
                and any(goal.kind == "construction" for goal in spec.goals)
                and has_probability_target
                and requirements <= {"target_p"}
            )
        if operation in {
            "nonadjacent_binary_string_count",
            "precedence_permutation_count",
            "surjection_count",
            "planar_euler_faces",
            "paraboloid_curvature",
        }:
            requirements = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            allowed = {
                "nonadjacent_binary_string_count": {"position_selection"},
                "precedence_permutation_count": {"counting_method"},
                "surjection_count": {"inclusion_exclusion"},
                "planar_euler_faces": {"euler_formula_check"},
                "paraboloid_curvature": {
                    "principal_curvatures", "gaussian_curvature", "surface_second_derivatives",
                },
            }[operation]
            max_goals = 2 if operation == "paraboloid_curvature" else 1
            return (
                1 <= goal_count <= max_goals
                and spec.profile.answer_shape in {"number", "expression"}
                and requirements <= allowed
            )
        return goal_count <= 1 or operation in CERTIFIED_MULTI_GOAL_OPERATIONS

    @staticmethod
    def _whole_tool_answer(evidence: tuple[ToolEvidence, ...]) -> str:
        for item in evidence:
            if item.scope == "whole_goal" and item.verified:
                return item.result
        return ""

    @staticmethod
    def _evidence_context(evidence: tuple[ToolEvidence, ...]) -> str:
        return "\n".join(f"- {item.operation} ({item.scope}): {item.result}" for item in evidence)

    @staticmethod
    def _evidence_trace(evidence: tuple[ToolEvidence, ...]) -> dict:
        return {
            "whole_goal_count": sum(item.scope == "whole_goal" for item in evidence),
            "subexpression_count": sum(item.scope == "subexpression" for item in evidence),
            "verified_subexpression_count": sum(
                item.scope == "subexpression" and item.verified for item in evidence
            ),
            "operations": [item.operation for item in evidence],
        }

    def _review_mode(
        self,
        spec: ProblemSpec,
        first_candidates: list[CandidateAssessment],
        first_response: str,
        budget: StageBudget,
        started_at: float,
        problem: str = "",
        provider_truncated: bool = False,
    ) -> str:
        del started_at
        if not budget.allow_review:
            return ""
        needs_rescue = (
            not any(candidate.accepted for candidate in first_candidates)
            or any(candidate.coverage_uncertain for candidate in first_candidates)
            or self._response_near_budget(first_response, budget.solve_tokens)
        )
        if needs_rescue:
            if provider_truncated and first_response.strip():
                return "continue"
            return "rescue"
        contest_contract = bool(re.search(
            r"remember\s+to\s+put\s+your\s+final\s+answer|\\boxed\s*\{\s*\}",
            problem,
            re.IGNORECASE,
        ))
        if budget.require_independent_review or contest_contract or len(problem) >= 180:
            return "verify"
        return ""

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
            has_unclosed_box = any(
                not complete
                for _, _, complete in Finalizer._boxed_values(response)
            )
            results = [self._finalize(response, spec)]
            for explicit in Finalizer.extract_explicit_results(response):
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
                reasons = result.rejected_reasons
                if provider_truncated and not result.explicit_answer:
                    reasons = tuple(dict.fromkeys((*reasons, "provider_truncated_without_explicit_answer")))
                if (
                    provider_truncated
                    and (
                        has_unclosed_box
                        or (
                            result.method in {"boxed", "boxed_unclosed"}
                            and not SubmissionAgent._is_just_boxed(response)
                        )
                    )
                ):
                    reasons = tuple(dict.fromkeys((*reasons, "provider_truncated_ambiguous_box")))
                candidates.append(assess_candidate(
                    result.answer, source, spec, evidence, result.method, reasons,
                    result.raw_has_meta, result.explicit_answer, verdict,
                ))

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
    def _verification_verdict(response: str) -> str:
        match = re.search(r"【\s*校验\s*】\s*(通过|一致|修正|错误)", str(response or ""))
        if not match:
            return ""
        return "confirmed" if match.group(1) in {"通过", "一致"} else "corrected"

    @staticmethod
    def _candidate_conflict(candidates: list[CandidateAssessment]) -> bool:
        by_stage: dict[str, CandidateAssessment] = {}
        for item in candidates:
            stage = SubmissionAgent._source_stage(item.source)
            if (
                item.validation_tier not in {"complete", "degraded"}
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
        return choose_candidate(candidates)

    @staticmethod
    def _source_stage(source: str) -> str:
        stage = SubmissionAgent._raw_source(source)
        return {
            "continue": "solve",
            "continue_last": "solve",
            "verify_recovered": "verify",
            "continue_verify": "verify",
            "retry_verify": "verify",
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
            requirement.strict and requirement.name not in {
                "judgement",
                "integral_value",
                "pointwise_limit",
                "domain",
                "solution_formula",
            }
            for goal in spec.goals
            for requirement in goal.requirements
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
        if not proof and tagged_block and (multipart or support_required):
            answer = SubmissionAgent._support_submission(
                tagged_block,
                Finalizer.extract_result(tagged_block).answer or explicit,
                spec,
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
        if spec.profile.answer_shape in {"choice", "truth"}:
            return True
        if any(re.search(
            r"时间序列|回归|异方差|统计|中心差分|有限差分|有限元|Lempel[- ]?Ziv|LZ78",
            goal.instruction,
            re.IGNORECASE,
        ) for goal in spec.goals):
            return True
        if getattr(spec.profile, "topic", "general").startswith("olympiad_"):
            return True
        theoretical = {
            "抽象代数", "拓扑学", "泛函分析", "复分析", "常微分方程",
            "数学分析", "离散数学", "数论", "初等几何", "高等代数",
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
        raw = (verification_completion or arbitration) if raw_source == "continue_verify" else (
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
            if "概率" in value:
                return SubmissionAgent._as_sentence(value)
            return SubmissionAgent._as_sentence(f"所求概率为{value}")
        if frame.question_kind == "truth":
            compact = re.sub(r"[\s。.]", "", value)
            normalized = {
                "是": "是", "正确": "是", "成立": "是", "可以": "可以",
                "否": "否", "错误": "否", "不成立": "否", "不可以": "不可以",
            }
            judgement = normalized.get(compact, value)
            if frame.subject and compact in normalized:
                return SubmissionAgent._as_sentence(f"{frame.subject}：{judgement}")
            return SubmissionAgent._as_sentence(judgement)
        return value

    @staticmethod
    def _proof_submission(response: str, explicit: str, english: bool = False) -> str:
        """Keep the argument while replacing model-only final-answer markers."""
        value = Finalizer.extract_solution(response)
        conclusion_label = "Conclusion: " if english else "结论："
        boxed_at = value.rfind(r"\boxed{")
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
            return assess_candidate(
                normalized, "support_compaction", spec, ()
            ).accepted

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
        return full

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
                not provider_truncated
                or (finalized.explicit_answer and not ambiguous_truncated_box)
            ):
                return finalized.answer, "degraded_finalized_candidate"
            for block in reversed(Finalizer.extract_tagged_submissions(str(response))):
                result = Finalizer.extract_result(block)
                if result.answer and not (set(Finalizer.validate_structure(result.answer)) & hard_reasons):
                    return result.answer, "degraded_explicit_block"
            if provider_truncated:
                continue
            for line in reversed(str(response).splitlines()):
                fragment = re.sub(
                    r"(?i)^\s*(?:FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION|最终答案|答案|结论)\s*[:：]?\s*",
                    "",
                    line,
                ).strip(" `")
                if not fragment or Finalizer.contains_meta(fragment):
                    continue
                reasons = set(Finalizer.validate_structure(fragment))
                if reasons & hard_reasons:
                    continue
                if re.search(r"\\(?:boxed|frac|sqrt|sum|prod)|[=<>≤≥]|(?<!\w)[-+]?\d", fragment):
                    return fragment, "degraded_typed_fragment"

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
