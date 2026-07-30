from __future__ import annotations

from typing import Dict, List

from core.client_adapter import ClientAdapter
from core.exception_handler import retry_once
from core.state import MathState


class Solver:
    """Generate one simple candidate or three independent hard candidates."""

    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def generate(self, state: MathState, plan: Dict) -> List[str]:
        count = 3 if state.difficulty == "hard" else 1
        references = "\n".join(plan.get("references", [])[:5])
        sympy_hints = "\n".join(plan.get("sympy_hints", []))
        candidates = []
        for index in range(count):
            role = "独立核对条件与计算" if index else "完整求解"
            content = (
                f"题目：\n{state.problem}\n\n"
                f"题型：{state.subject}，{state.problem_type}。\n"
                f"本次要求：{role}。"
            )
            if references:
                content += f"\n本地参考：\n{references}"
            if sympy_hints:
                content += f"\n本地 SymPy 计算结果（仅作核对，仍须结合题目作答）：\n{sympy_hints}"
            response = retry_once(lambda content=content: self.client_adapter.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.2 + index * 0.15,
                max_tokens=4096,
            ))
            if isinstance(response, str) and response.strip():
                candidates.append(response.strip())
                state.trace.append({"step": f"solver_{index}", "content": "candidate_generated"})
            else:
                state.trace.append({"step": f"solver_{index}", "content": "candidate_failed"})
        if not candidates:
            fallback = self._fallback(plan)
            if fallback:
                candidates.append(f"【最终答案】{fallback}")
                state.trace.append({"step": "solver_fallback", "content": "sympy_hint_used"})
        state.candidate_answers.extend(candidates)
        return candidates

    @staticmethod
    def _fallback(plan: Dict) -> str:
        """Use a deterministic local result only after all model attempts fail."""
        for hint in plan.get("sympy_hints", []):
            _, separator, result = hint.partition(": ")
            if separator and result.strip():
                return result.strip()
        return ""
