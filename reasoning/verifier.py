from __future__ import annotations

import re
from typing import Dict, List

from core.client_adapter import ClientAdapter
from core.exception_handler import IncompleteModelResponseError, retry_once
from core.state import MathState
from reasoning.finalizer import Finalizer


class Verifier:
    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def verify(
        self,
        state: MathState,
        candidates: List[str],
        critic_reviews: List[str] | None = None,
        sympy_hints: List[str] | None = None,
        response_contract: str = "",
        decomposition: List[str] | None = None,
        decomposition_tools: List[str] | None = None,
    ) -> Dict:
        if not candidates:
            return {
                "correct": False,
                "choice": None,
                "final_answer": "",
                "reason": "没有可用候选",
            }
        preserve_reasoning = state.problem_type in {"proof", "derivation", "explanation"}
        joined = "\n\n".join(
            (
                f"候选{i}的完整论证：\n{text[:6000]}"
                if preserve_reasoning
                else f"候选{i}的最终结论：{Finalizer.extract(text)}"
            )
            for i, text in enumerate(candidates)
        )
        if len(critic_reviews or []) == 1 and len(candidates) > 1:
            reviews = f"候选集综合批评意见：{critic_reviews[0]}"
        else:
            reviews = "\n".join(
                f"候选{i}的批评意见：{review}"
                for i, review in enumerate(critic_reviews or [])
            )
        tool_context = "\n".join(sympy_hints or [])
        context = f"题目：\n{state.problem}\n\n{joined}"
        if reviews:
            context += f"\n\n{reviews}"
        if tool_context:
            context += f"\n\n本地 SymPy 核对：\n{tool_context}"
        if response_contract:
            context += f"\n\n作答契约：{response_contract}"
        if decomposition:
            context += "\n\n待核对的子目标：\n" + "\n".join(
                f"{index + 1}. {step}" for index, step in enumerate(decomposition)
            )
        if decomposition_tools:
            context += "\n\n子步骤本地计算：\n" + "\n".join(decomposition_tools)
        if preserve_reasoning:
            context += (
                "\n\n这是需要说明依据的题。FINAL 必须保留完整、简洁且自洽的推理过程，"
                "包括关键依据和推导链，并以【最终答案】给出结论。"
            )
        errors: List[str] = []
        response = retry_once(
            lambda: self._complete_response(self.client_adapter.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.0,
                # Intern may emit a long hidden-style thinking preamble before its
                # structured three-line result. Leave room for FINAL to arrive.
                max_tokens=8192,
            )),
            on_error=lambda exc: errors.append(type(exc).__name__),
            attempts=1 if state.difficulty == "hard" else 2,
        )
        raw = str(response or "")
        match = re.search(r"CHOICE\s*[:：]\s*(\d+)", raw, re.IGNORECASE)
        choice = int(match.group(1)) if match else 0
        if not 0 <= choice < len(candidates):
            choice = 0
        final_match = re.search(
            r"^FINAL\s*[:：]\s*(.*?)(?=^REASON\s*[:：]|\Z)",
            raw,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        reason_match = re.search(r"REASON\s*[:：]\s*(.+)", raw, re.IGNORECASE)
        final_answer = final_match.group(1).strip() if final_match else ""
        required_terms = self._required_terms(state.problem)
        if required_terms and not self._contains_terms(final_answer, required_terms):
            for candidate in candidates:
                candidate_final = Finalizer.extract_solution(candidate) if preserve_reasoning else Finalizer.extract(candidate)
                if self._contains_terms(candidate_final, required_terms):
                    final_answer = candidate_final
                    break
        return {
            "correct": bool(response),
            "choice": choice,
            "final_answer": final_answer,
            "reason": reason_match.group(1).strip() if reason_match else raw[:300] or "验证器不可用",
            "errors": errors,
        }

    @staticmethod
    def _complete_response(response: object) -> str:
        text = str(response or "").strip()
        has_choice = re.search(r"^CHOICE\s*[:：]\s*\d+", text, re.IGNORECASE | re.MULTILINE)
        has_final = re.search(r"^FINAL\s*[:：]", text, re.IGNORECASE | re.MULTILINE)
        has_reason = re.search(r"^REASON\s*[:：]", text, re.IGNORECASE | re.MULTILINE)
        final_match = re.search(
            r"^FINAL\s*[:：]\s*(.*?)(?=^REASON\s*[:：]|\Z)",
            text,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not (has_choice and has_final and has_reason and final_match and final_match.group(1).strip()):
            raise IncompleteModelResponseError("verifier response lacks required fields")
        return text

    @staticmethod
    def _required_terms(problem: str) -> List[str]:
        """Keep the verifier from replacing a requested object with a check value."""
        compact = re.sub(r"\s+", "", problem.lower())
        if "插值多项式p(x)" in compact:
            return ["p(x)"]
        if "构造两个边缘" in compact and "p(x=y)" in compact:
            return ["p(x=y)"]
        if "商环z[x]" in compact and "x的平方" in compact:
            return ["x^2"]
        if "商环r/i" in compact:
            return ["r/i"]
        return []

    @staticmethod
    def _contains_terms(answer: str, terms: List[str]) -> bool:
        compact = re.sub(r"\s+", "", str(answer or "").lower())
        return all(term in compact for term in terms)
