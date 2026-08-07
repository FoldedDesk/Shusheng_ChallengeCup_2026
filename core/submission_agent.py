"""Bounded, retrieval-assisted implementation used by the public entry point."""

from __future__ import annotations

from pathlib import Path
import re
from time import monotonic

from classifier.problem_spec import ProblemSpec, build_problem_spec
from core.stage_budget import StageBudget, plan_stage_budget
from rag.card_retriever import CardRetriever, RetrievalBundle
from reasoning.candidate_selector import (
    CandidateAssessment,
    ToolEvidence,
    assess_candidate,
    choose_candidate,
)
from reasoning.finalizer import ExtractionResult, Finalizer
from tools.sympy_tool import SympyTool


SUBMISSION_SOFT_BUDGET_SECONDS = 115


class SubmissionAgent:
    """Two-call solve/review path with local retrieval and deterministic checks."""

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
        direct_tool_route = bool(tool_answer and spec.profile.difficulty != "hard")
        budget = plan_stage_budget(spec, direct_tool_route)
        trace = [
            {"step": "spec", "content": spec.trace_content()},
            {"step": "stage_budget", "content": budget.trace_content()},
            {"step": "retrieval", "content": cards.trace_content()},
            {"step": "tool_evidence", "content": self._evidence_trace(evidence)},
        ]

        first = ""
        if not direct_tool_route:
            first = self._call(
                self._solve_request(text, spec, cards, evidence),
                "solve",
                budget.solve_tokens,
                trace,
                started_at,
            )
        first_candidates = self._assess_candidates(first, "", "", spec, evidence)
        review_admitted = self._review_admitted(spec, first_candidates, budget, started_at)
        trace.append({
            "step": "review_admission",
            "content": {
                "admitted": review_admitted,
                "remaining_budget_ms": self._remaining_ms(started_at),
            },
        })
        second = ""
        if review_admitted:
            second = self._call(
                self._review_request(text, spec, cards, evidence, first),
                "review",
                budget.review_tokens,
                trace,
                started_at,
            )

        candidates = self._assess_candidates(first, second, tool_answer, spec, evidence)
        repair = ""
        if (
            not any(candidate.accepted for candidate in candidates)
            and budget.allow_repair
            and self._remaining_ms(started_at) >= budget.repair_min_remaining_seconds * 1000
        ):
            repair = self._call(
                self._repair_request(text, spec), "repair", budget.repair_tokens, trace, started_at
            )
            candidates = self._assess_candidates(first, second, tool_answer, spec, evidence, repair)
        trace.append({
            "step": "candidate_diagnostics",
            "content": {
                "solve": self._safe_trace_candidate(first),
                "review": self._safe_trace_candidate(second),
                "repair": self._safe_trace_candidate(repair),
            },
        })
        selected = self._select(candidates)
        answer = selected.answer if selected else "未能生成可验证的数学答案。"
        trace.append({
            "step": "validation",
            "content": {item.source: self._assessment_trace(item) for item in candidates},
        })
        trace.append({
            "step": "selection", "content": {
                "source": selected.source if selected else "fallback",
                "score": selected.score if selected else 0,
            },
        })
        trace.append({
            "step": "finalize", "content": {
                "non_empty": bool(answer),
                "answer_shape": spec.profile.answer_shape,
                "source": selected.source if selected else "fallback",
                "elapsed_ms": int((monotonic() - started_at) * 1000),
            },
        })
        self._append_proof_trace(trace, spec, selected, first, second)
        return {"final_response": answer, "trace": trace}

    def _call(self, request: str, stage: str, max_tokens: int, trace: list[dict], started_at: float) -> str:
        stage_started = monotonic()
        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": request},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            value = str(response or "")
            trace.append({
                "step": f"model_call_{stage}",
                "content": {
                    "status": "completed",
                    "response_non_empty": bool(value.strip()),
                    "elapsed_ms": int((monotonic() - stage_started) * 1000),
                    "remaining_budget_ms": self._remaining_ms(started_at),
                },
            })
            return value
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
            return ""

    @staticmethod
    def _load_prompt() -> str:
        try:
            return (Path("prompts") / "submission.txt").read_text(encoding="utf-8")
        except OSError:
            return "直接给出简洁数学答案，最后写 \\boxed{最终答案}。"

    @staticmethod
    def _solve_request(problem: str, spec: ProblemSpec, cards: RetrievalBundle, evidence: tuple[ToolEvidence, ...]) -> str:
        content = f"题目：\n{problem}\n\n"
        content += SubmissionAgent._direct_instruction(spec)
        content += "\n" + SubmissionAgent._spec_context(spec)
        if cards.solve_context():
            content += "\n可用定理/方法（仅在前提满足时使用）：\n" + cards.solve_context()
        if evidence:
            content += "\n本地计算证据（只核对相应子目标）：\n" + SubmissionAgent._evidence_context(evidence)
        if spec.answer_frame.style == "proof":
            content += "\n\n给出精炼论证和完整结论。不要输出 Thinking Process 或元话语；最后一行必须写【最终答案】完整结论。"
        else:
            content += "\n\n先在内部完成推理，输出时只保留所有必答项组成的最终结论。不要输出 Thinking Process 或元话语；最后一行必须写【最终答案】完整答案。"
        return content

    @staticmethod
    def _review_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidate: str,
    ) -> str:
        if not candidate.strip():
            return (
                f"题目：\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}"
                "\n第一轮未产生可用答案。请从头独立求解，不要讨论失败原因；"
                "覆盖题目全部必答项并在最后一行写【最终答案】完整答案。"
            )
        content = f"题目：\n{problem}\n\n" + SubmissionAgent._direct_instruction(spec)
        content += "\n" + SubmissionAgent._spec_context(spec)
        if cards.review_context():
            content += "\n审查卡：\n" + cards.review_context()
        if evidence:
            content += "\n本地计算证据：\n" + SubmissionAgent._evidence_context(evidence)
        content += (
            "\n\n第一轮完整候选如下。不要默认它正确：逐项检查目标清单、最终公式或数值、定义域和单位。"
            "若缺少子问、格式残缺或结论不完整，必须从头修正；只输出可提交答案，最后一行写【最终答案】完整答案。\n\n"
            f"第一轮候选：\n{SubmissionAgent._review_evidence(candidate)}"
        )
        return content

    @staticmethod
    def _repair_request(problem: str, spec: ProblemSpec) -> str:
        return (
            f"题目：\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}\n"
            + SubmissionAgent._spec_context(spec)
            + "\n前两轮没有可提交结论。请从头独立作答，不要复述题目、提示词或思考过程。"
            "只输出最终可判分答案，最后一行必须为【最终答案】加完整结论。"
        )

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
        if spec.profile.problem_type in {"proof", "derivation", "explanation"}:
            return (
                "请给出可直接提交的简洁证明：写明关键依据和必要推导，"
                "再用一句话明确结论。不要只给结论，也不要输出思考过程。"
            )
        frame = spec.answer_frame
        if frame.question_kind == "age" and frame.subject:
            return f"请完整求解；最终答案必须是可读句子，例如“{frame.subject}14岁。”，不要只写数值。"
        if frame.question_kind == "count":
            return "请完整求解；最终答案必须是可读句子，例如“所求数量为16个。”，不要只写数值。"
        if frame.question_kind == "probability":
            return "请完整求解；最终答案必须是可读句子，例如“所求概率为1/2。”，不要只写数值。"
        if frame.question_kind == "truth":
            return "请完整求解；最终答案必须以“是。”或“否。”等完整判断句作答。"
        if spec.profile.answer_shape == "roots":
            return "请给出全部解并检查定义域；离散根不要写成区间。"
        if spec.profile.answer_shape == "interval":
            return "请检查端点并用区间或并集给出解集。"
        if len(spec.goals) > 1:
            return "请按题目顺序完整回答每个子问。"
        return "请直接完整求解，避免冗长解释。"

    @staticmethod
    def _tool_evidence(hints: list[str], spec: ProblemSpec) -> tuple[ToolEvidence, ...]:
        supported = {
            "SymPy 计算": "calculate", "SymPy 方程解": "solve_equation", "SymPy 导数": "derivative",
            "SymPy 定积分": "definite_integral", "SymPy 不定积分": "integral", "SymPy 极限": "limit",
        }
        evidence = []
        for hint in hints:
            label, separator, result = hint.partition(": ")
            if not separator or not result.strip():
                continue
            operation = supported.get(label, "local_hint")
            whole = (
                operation != "local_hint"
                and spec.tool_can_answer_whole
            )
            evidence.append(ToolEvidence(result.strip(), "whole_goal" if whole else "subexpression", operation, whole))
        return tuple(evidence)

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
            "operations": [item.operation for item in evidence],
        }

    def _review_admitted(
        self,
        spec: ProblemSpec,
        first_candidates: list[CandidateAssessment],
        budget: StageBudget,
        started_at: float,
    ) -> bool:
        if not budget.allow_review:
            return False
        needs_review = (
            not any(candidate.accepted for candidate in first_candidates)
            or any(candidate.coverage_uncertain for candidate in first_candidates)
            or spec.profile.difficulty == "hard"
        )
        return needs_review and self._remaining_ms(started_at) >= budget.review_min_remaining_seconds * 1000

    def _assess_candidates(
        self,
        first: str,
        second: str,
        tool_answer: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        repair: str = "",
    ) -> list[CandidateAssessment]:
        candidates = []
        first_result = self._finalize(first, spec)
        second_result = self._finalize(second, spec)
        if first_result.answer or first_result.rejected_reasons:
            candidates.append(assess_candidate(
                first_result.answer, "solve", spec, evidence, first_result.method, first_result.rejected_reasons,
                first_result.raw_has_meta, first_result.explicit_answer,
            ))
        if second_result.answer or second_result.rejected_reasons:
            candidates.append(assess_candidate(
                second_result.answer, "review", spec, evidence, second_result.method, second_result.rejected_reasons,
                second_result.raw_has_meta, second_result.explicit_answer,
            ))
        repair_result = self._finalize(repair, spec)
        if repair_result.answer or repair_result.rejected_reasons:
            candidates.append(assess_candidate(
                repair_result.answer, "repair", spec, evidence, repair_result.method, repair_result.rejected_reasons,
                repair_result.raw_has_meta, repair_result.explicit_answer,
            ))
        if tool_answer:
            answer = self._render_answer(self._normalize_answer(tool_answer, spec), spec)
            candidates.append(assess_candidate(answer, "sympy_verified", spec, evidence, "tool"))
        return candidates

    @staticmethod
    def _select(candidates: list[CandidateAssessment]) -> CandidateAssessment | None:
        return choose_candidate(candidates)

    @staticmethod
    def _assessment_trace(item: CandidateAssessment) -> dict:
        return {
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
        if proof and not extracted.raw_has_meta and not SubmissionAgent._is_just_boxed(response):
            answer = SubmissionAgent._proof_submission(response, explicit)
            reasons = Finalizer.validate_structure(answer)
            result = ExtractionResult(
                answer if not reasons else "", "proof_body", not reasons, reasons,
                extracted.raw_has_meta, False,
            )
        else:
            result = extracted
        normalized = SubmissionAgent._render_answer(SubmissionAgent._normalize_answer(result.answer, spec), spec)
        return ExtractionResult(
            normalized, result.method, result.valid, result.rejected_reasons,
            result.raw_has_meta, result.explicit_answer,
        )

    @staticmethod
    def _should_retrieve(spec: ProblemSpec) -> bool:
        theoretical = {"抽象代数", "拓扑学", "泛函分析", "复分析", "常微分方程", "数学分析", "离散数学"}
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
    ) -> None:
        if spec.profile.problem_type not in {"proof", "derivation", "explanation"} or not selected:
            return
        raw = second if selected.source == "review" else first
        if raw.strip():
            trace.append({"step": "proof_summary", "content": Finalizer.extract_solution(raw)[:1600]})

    @staticmethod
    def _normalize_answer(answer: str, spec: ProblemSpec) -> str:
        value = str(answer or "").strip().replace(r"\infty", "∞")
        value = re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", value)
        value = value.replace(r"\left", "").replace(r"\right", "")
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
            normalized = {"正确": "是", "成立": "是", "可以": "可以", "错误": "否", "不成立": "否", "不可以": "不可以"}
            judgement = normalized.get(compact, value)
            if frame.subject and compact in normalized:
                return SubmissionAgent._as_sentence(f"{frame.subject}：{judgement}")
            return SubmissionAgent._as_sentence(judgement)
        return value

    @staticmethod
    def _proof_submission(response: str, explicit: str) -> str:
        """Keep the argument while replacing model-only final-answer markers."""
        value = Finalizer.extract_solution(response)
        boxed_at = value.rfind(r"\boxed{")
        if boxed_at >= 0 and explicit:
            value = value[:boxed_at].rstrip() + f"\n结论：{explicit}"
        value = re.sub(
            r"(?im)^\s*(?:【最终答案】|(?:最终)?答案|结论|FINAL(?:\s*ANSWER)?|ANSWER)\s*[:：]?\s*",
            "结论：",
            value,
        )
        return value.strip()

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
    def _is_just_boxed(response: str) -> bool:
        value = response.strip()
        return bool(re.fullmatch(r"(?:\\\[|\$\$?)?\s*\\boxed\{.*\}\s*(?:\\\]|\$\$?)?", value, re.DOTALL))

    @staticmethod
    def _remaining_ms(started_at: float) -> int:
        return max(0, int((SUBMISSION_SOFT_BUDGET_SECONDS - (monotonic() - started_at)) * 1000))
