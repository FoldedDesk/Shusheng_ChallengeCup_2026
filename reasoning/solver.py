from __future__ import annotations

import re
from typing import Dict, List

from core.client_adapter import ClientAdapter
from core.exception_handler import IncompleteModelResponseError, retry_once
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
        decomposition = "\n".join(
            f"{index + 1}. {step}" for index, step in enumerate(plan.get("decomposition", []))
        )
        decomposition_tools = "\n".join(plan.get("decomposition_tools", []))
        candidates = []
        for index in range(count):
            role = "独立核对条件与计算" if index else "完整求解"
            content = (
                f"题目：\n{state.problem}\n\n"
                f"题型：{state.subject}，{state.problem_type}。\n"
                f"本次要求：{role}。\n"
                f"作答契约：{plan.get('response_contract', '')}"
            )
            if references:
                content += f"\n本地参考：\n{references}"
            if sympy_hints:
                content += f"\n本地 SymPy 计算结果（仅作核对，仍须结合题目作答）：\n{sympy_hints}"
            if decomposition:
                content += f"\n分解后的子目标：\n{decomposition}"
            if decomposition_tools:
                content += f"\n分解步骤的本地计算结果：\n{decomposition_tools}"
            if state.problem_type in {"proof", "derivation", "explanation"}:
                content += (
                    "\n这是需要说明依据的题。必须给出自洽的推理过程：明确使用的定义、定理或条件，"
                    "逐步说明推理，最后单独写出【最终答案】。不得只给结论。"
                )
            errors: List[str] = []
            response = retry_once(
                lambda content=content: self._complete_response(
                    self.client_adapter.chat(
                        messages=[
                            {"role": "system", "content": self.prompt},
                            {"role": "user", "content": content},
                        ],
                        temperature=0.2 + index * 0.15,
                        # Hard calculation problems can also consume a long
                        # internal preamble before emitting the required final
                        # marker. Keep the same completion budget as proofs.
                        max_tokens=8192 if state.difficulty == "hard" else 4096,
                    )
                ),
                on_error=lambda exc: errors.append(type(exc).__name__),
                # A malformed scratchpad has no usable conclusion. Hard tasks
                # have independent candidates, but each one gets one recovery
                # attempt so a transient format failure cannot erase all of
                # them and produce TRUNCATED_ALL.
                attempts=2,
            )
            if isinstance(response, str) and response.strip():
                candidates.append(response.strip())
                state.trace.append({"step": f"solver_{index}", "content": "candidate_generated"})
            else:
                state.trace.append({
                    "step": f"solver_{index}",
                    "content": {"status": "candidate_failed", "errors": errors},
                })
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

    @staticmethod
    def _complete_response(response: object) -> str:
        text = str(response or "").strip()
        if re.search(r"(?im)^\s*(thinking process|analysis|drafting)", text):
            raise IncompleteModelResponseError("solver response contains a reasoning scratchpad")
        if "【最终答案】" not in text:
            raise IncompleteModelResponseError("solver response lacks final-answer marker")
        return text
