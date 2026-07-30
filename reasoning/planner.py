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
        }
