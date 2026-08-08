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
from reasoning.math_equivalence import equivalent_answers
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
        if not direct_tool_route:
            first = self._call(
                self._solve_request(text, spec, cards, evidence),
                "solve",
                budget.solve_tokens,
                trace,
                started_at,
            )
        first_candidates = self._assess_candidates(first, "", "", "", spec, evidence)
        review_mode = self._review_mode(spec, first_candidates, first, budget, started_at)
        trace.append({
            "step": "review_admission",
            "content": {
                "admitted": bool(review_mode),
                "mode": review_mode or "none",
                "remaining_budget_ms": self._remaining_ms(started_at),
            },
        })
        second = ""
        if review_mode:
            request = (
                self._verification_request(text, spec, cards, evidence, first_candidates)
                if review_mode == "verify"
                else self._rescue_request(text, spec, cards, evidence, first)
            )
            second = self._call(
                request,
                review_mode,
                budget.review_tokens,
                trace,
                started_at,
            )

        second_source = review_mode or "rescue"
        candidates = self._assess_candidates(first, second, "", tool_answer, spec, evidence, second_source)
        conflict = self._candidate_conflict(candidates)
        repair_mode = "arbitration" if conflict else (
            "last_chance" if not any(item.accepted for item in candidates) else ""
        )
        arbitration = ""
        if (
            repair_mode
            and budget.allow_repair
            and self._remaining_ms(started_at) >= budget.repair_min_remaining_seconds * 1000
        ):
            arbitration = self._call(
                self._arbitration_request(text, spec, candidates, evidence)
                if repair_mode == "arbitration"
                else self._last_chance_request(text, spec, evidence),
                repair_mode,
                budget.repair_tokens,
                trace,
                started_at,
            )
            candidates = self._assess_candidates(
                first, second, arbitration, tool_answer, spec, evidence, second_source
            )
        trace.append({
            "step": "candidate_diagnostics",
            "content": {
                "solve": self._safe_trace_candidate(first),
                second_source: self._safe_trace_candidate(second),
                "arbitration": self._safe_trace_candidate(arbitration),
            },
        })
        selected = self._select(candidates)
        answer = selected.answer if selected else "未能生成可验证的数学答案。"
        trace.append({
            "step": "equivalence", "content": {
                "conflict": conflict,
                "accepted_sources": [item.source for item in candidates if item.accepted],
                "arbitration_used": bool(arbitration.strip()),
                "repair_mode": repair_mode or "none",
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
        self._append_proof_trace(trace, spec, selected, first, second, arbitration)
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
                    "max_tokens": max_tokens,
                    "response_near_budget": len(value) >= max_tokens * 3,
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
        content = f"题目：\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}"
        content += "\n必须覆盖以下目标：\n" + SubmissionAgent._goal_context(spec)
        if cards.solve_context():
            content += "\n可用依据：\n" + cards.solve_context()
        if evidence:
            content += "\n本地核验：\n" + SubmissionAgent._evidence_context(evidence)
        content += "\n第一行先写【最终答案】并给出全部结论，再写必要依据。"
        return content

    @staticmethod
    def _rescue_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidate: str,
    ) -> str:
        if not candidate.strip():
            return (
                f"题目：\n{problem}\n\n{SubmissionAgent._direct_instruction(spec)}"
                f"\n必须覆盖以下目标：\n{SubmissionAgent._goal_context(spec)}"
                "\n第一轮未产生可用答案。请从头独立求解，不要讨论失败原因；"
                "覆盖题目全部必答项，第一行先写【最终答案】完整答案。"
            )
        content = f"题目：\n{problem}\n\n" + SubmissionAgent._direct_instruction(spec)
        content += "\n必须覆盖以下目标：\n" + SubmissionAgent._goal_context(spec)
        if cards.review_context():
            content += "\n审查卡：\n" + cards.review_context()
        if evidence:
            content += "\n本地计算证据：\n" + SubmissionAgent._evidence_context(evidence)
        content += (
            "\n\n独立重做并补齐缺失项，第一行先写【最终答案】完整答案。\n\n"
            f"首轮可用摘要：\n{SubmissionAgent._review_evidence(candidate)}"
        )
        return content

    @staticmethod
    def _verification_request(
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        candidates: list[CandidateAssessment],
    ) -> str:
        first = next((item.answer for item in candidates if item.source == "solve" and item.answer), "（无可用答案）")
        content = (
            f"题目：\n{problem}\n\n你是数学答案审查者。先独立重算，再检查候选；"
            "不得因为候选表述流畅就默认正确。逐项检查必答项、公式、数值、定义域、单位、定理前提和构造条件。\n"
            f"目标清单：\n{SubmissionAgent._goal_context(spec)}\n"
            f"待审候选：\n{first}\n"
        )
        if cards.review_context():
            content += "审查依据：\n" + cards.review_context() + "\n"
        if evidence:
            content += "本地核验：\n" + SubmissionAgent._evidence_context(evidence) + "\n"
        content += (
            "若候选完全正确，第一行写【校验】通过；若有任何错误或缺项，第一行写【校验】修正。"
            "第二行必须写【最终答案】完整答案，随后只写最短核验依据。"
        )
        return content

    @staticmethod
    def _arbitration_request(
        problem: str,
        spec: ProblemSpec,
        candidates: list[CandidateAssessment],
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        rendered = "\n".join(
            f"候选{index + 1}：{item.answer}"
            for index, item in enumerate(candidates)
            if item.accepted and item.source in {"solve", "verify", "rescue"}
        )
        content = (
            f"题目：\n{problem}\n\n两个答案发生实质冲突。请重新计算并裁决，不要按长度或措辞选择。\n"
            f"目标清单：\n{SubmissionAgent._goal_context(spec)}\n{rendered}\n"
        )
        if evidence:
            content += "本地证据：\n" + SubmissionAgent._evidence_context(evidence) + "\n"
        return content + "第一行写【校验】修正，第二行写【最终答案】裁决后的完整答案。"

    @staticmethod
    def _last_chance_request(
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        content = (
            f"题目：\n{problem}\n\n前两轮没有形成可提交内容。停止分析格式和措辞，"
            "直接给出实际数学结论，限一行，不解释、不复述题目。\n"
            f"必须覆盖：\n{SubmissionAgent._goal_context(spec)}\n"
        )
        if evidence:
            content += "本地核验：\n" + SubmissionAgent._evidence_context(evidence) + "\n"
        return content

    @staticmethod
    def _goal_context(spec: ProblemSpec) -> str:
        checks = {
            "proof": "关键依据、必要推导、明确结论",
            "construction": "构造对象、逐项验证题设条件",
            "truth_judgement": "判断结论、被判断对象、关键检验",
            "domain_or_interval": "定义域/区间、端点和排除值",
            "formula": "完整公式、变量含义和题设初值",
            "comparison": "各个数值、误差或大小比较",
            "equation_roots": "全部根、定义域和伪根检查",
            "scalar_or_result": "明确结果及题目要求的单位/对象",
        }
        rendered = []
        for goal in spec.goals:
            requirement_names = "、".join(item.name for item in goal.requirements)
            suffix = checks.get(goal.kind, "完整可判分结论")
            if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", goal.instruction, re.IGNORECASE):
                suffix += "；右侧区间须从初始点开始向右延伸，不得包含初始点左侧"
            if requirement_names:
                suffix += f"；必查字段：{requirement_names}"
            rendered.append(f"- {goal.id} [{goal.kind}] {goal.instruction}（{suffix}）")
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
        if any(re.search(
            r"最大右侧存在区间|maximal right(?:-hand)? interval", goal.instruction, re.IGNORECASE
        ) for goal in spec.goals):
            return (
                "先通过初值确定解，再给出从初始自变量值开始向右延伸的最大存在区间；"
                "不要用包含初始点左侧的双侧最大区间替代右侧区间。"
            )
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
            # A supported local operation is verified evidence even when it is
            # only a substep. Scope, not verification, controls whether it may
            # become the complete submitted answer.
            evidence.append(ToolEvidence(
                result.strip(), "whole_goal" if whole else "subexpression", operation,
                operation != "local_hint",
            ))
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
    ) -> str:
        if not budget.allow_review:
            return ""
        needs_rescue = (
            not any(candidate.accepted for candidate in first_candidates)
            or any(candidate.coverage_uncertain for candidate in first_candidates)
            or self._response_near_budget(first_response, budget.solve_tokens)
        )
        if self._remaining_ms(started_at) < budget.review_min_remaining_seconds * 1000:
            return ""
        if needs_rescue:
            return "rescue"
        return "verify" if spec.verification_required else ""

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
    ) -> list[CandidateAssessment]:
        candidates = []
        first_result = self._finalize(first, spec)
        second_result = self._finalize(second, spec)
        arbitration_result = self._finalize(arbitration, spec)
        if first.strip() and (first_result.answer or first_result.rejected_reasons):
            candidates.append(assess_candidate(
                first_result.answer, "solve", spec, evidence, first_result.method, first_result.rejected_reasons,
                first_result.raw_has_meta, first_result.explicit_answer,
            ))
        if second.strip() and (second_result.answer or second_result.rejected_reasons):
            candidates.append(assess_candidate(
                second_result.answer, second_source, spec, evidence, second_result.method, second_result.rejected_reasons,
                second_result.raw_has_meta, second_result.explicit_answer, self._verification_verdict(second),
            ))
        if arbitration.strip() and (arbitration_result.answer or arbitration_result.rejected_reasons):
            candidates.append(assess_candidate(
                arbitration_result.answer, "arbitration", spec, evidence,
                arbitration_result.method, arbitration_result.rejected_reasons,
                arbitration_result.raw_has_meta, arbitration_result.explicit_answer,
                self._verification_verdict(arbitration),
            ))
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
        model_candidates = [
            item for item in candidates
            if item.accepted and item.source in {"solve", "verify", "rescue"}
        ]
        return len(model_candidates) >= 2 and not equivalent_answers(
            model_candidates[0].answer, model_candidates[1].answer
        )

    @staticmethod
    def _equivalence_pairs(candidates: list[CandidateAssessment]) -> list[dict]:
        model_candidates = [
            item for item in candidates
            if item.accepted and item.source in {"solve", "verify", "rescue", "arbitration"}
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
        proof_block = Finalizer.extract_tagged_submission(response) if proof else ""
        if proof and (proof_block or not extracted.raw_has_meta) and not SubmissionAgent._is_just_boxed(response):
            # A normal proof may put its answer marker at the end, so retaining
            # only the tagged suffix would discard the actual argument. The
            # suffix is a safety boundary only when a meta preamble exists.
            proof_source = proof_block if extracted.raw_has_meta else response
            answer = SubmissionAgent._proof_submission(proof_source, explicit)
            reasons = Finalizer.validate_structure(answer)
            result = ExtractionResult(
                answer if not reasons else "", "proof_body", not reasons, reasons,
                extracted.raw_has_meta, False,
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
        arbitration: str,
    ) -> None:
        if spec.profile.problem_type not in {"proof", "derivation", "explanation"} or not selected:
            return
        raw = arbitration if selected.source == "arbitration" else (
            second if selected.source in {"rescue", "verify"} else first
        )
        if raw.strip():
            trace.append({"step": "proof_summary", "content": Finalizer.extract_solution(raw)[:1600]})

    @staticmethod
    def _normalize_answer(answer: str, spec: ProblemSpec) -> str:
        value = str(answer or "").strip().replace("\x08ar", r"\bar").replace(r"\infty", "∞")
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
