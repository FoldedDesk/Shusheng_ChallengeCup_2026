from __future__ import annotations

from typing import Dict, List

from core.state import MathState


class Planner:
    def plan(self, state: MathState, references: List[str]) -> Dict:
        return {
            "subject": state.subject or "进阶数学",
            "problem_type": state.problem_type or "calculation",
            "difficulty": state.difficulty or "medium",
            "references": references[:5],
            "response_contract": self._response_contract(state),
        }

    @staticmethod
    def _response_contract(state: MathState) -> str:
        if state.problem_type == "proof":
            return (
                "证明必须依次写明：题设与目标；所用定义、定理或关键条件；"
                "每一步如何推出下一步；条件为何足够；最后的完整结论。"
            )
        if state.problem_type == "derivation":
            return "推导必须交代起始公式、变形依据、关键中间式以及最终公式的适用条件。"
        if state.problem_type == "explanation":
            return "说明题必须给出结论成立的关键依据和完整因果链，不能只报数值或结论。"
        return "答案必须覆盖题目全部所求对象、数值、条件和结论，不能只给中间量。"
