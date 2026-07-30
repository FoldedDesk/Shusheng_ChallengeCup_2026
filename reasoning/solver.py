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
        state.candidate_answers.extend(candidates)
        return candidates
