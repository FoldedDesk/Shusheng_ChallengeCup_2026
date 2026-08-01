from __future__ import annotations

import re
from typing import Dict, List

from core.client_adapter import ClientAdapter
from core.exception_handler import retry_once
from core.state import MathState
from tools.sympy_tool import SympyTool


class Decomposer:
    """Turn a hard problem into a small, bounded evidence plan."""

    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def decompose(self, state: MathState) -> Dict[str, List[str]]:
        if state.difficulty != "hard":
            return {"steps": [], "tool_specs": []}
        response = retry_once(lambda: self.client_adapter.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"题目：\n{state.problem}"},
            ],
            temperature=0.0,
            max_tokens=2048,
        ), attempts=1)
        raw = str(response or "")
        steps = self._steps(raw) or self._fallback_steps(state)
        return {"steps": steps, "tool_specs": self._tool_specs(raw)}

    @staticmethod
    def execute_tools(tool: SympyTool, specs: List[str]) -> List[str]:
        results = []
        for spec in specs[:3]:
            parts = [part.strip() for part in spec.split("|")]
            operation = parts[0].lower()
            try:
                if operation == "derivative" and len(parts) == 3:
                    value = tool.derivative(tool._latex_to_sympy(parts[1]), parts[2])
                elif operation == "integral" and len(parts) == 3:
                    value = tool.integral(tool._latex_to_sympy(parts[1]), parts[2])
                elif operation == "limit" and len(parts) == 4:
                    value = tool.limit(tool._latex_to_sympy(parts[1]), parts[2], tool._latex_to_sympy(parts[3]))
                elif operation == "equation" and len(parts) == 3:
                    roots = tool.solve_equation(tool._latex_to_sympy(parts[1]), parts[2])
                    value = ", ".join(f"{parts[2]}={root}" for root in roots) if roots else None
                elif operation == "calculate" and len(parts) == 2:
                    value = tool.evaluate(tool._latex_to_sympy(parts[1]))
                else:
                    continue
            except Exception:
                continue
            if value is not None:
                results.append(f"分解工具 {operation}: {value}")
        return results

    @staticmethod
    def _steps(raw: str) -> List[str]:
        steps = []
        for match in re.finditer(r"^\s*(?:步骤\s*)?\d+[.、:：]\s*(.+)$", raw, re.MULTILINE):
            value = match.group(1).strip()
            if value and len(value) <= 400:
                steps.append(value)
        return steps[:5]

    @staticmethod
    def _tool_specs(raw: str) -> List[str]:
        specs = re.findall(r"^\s*TOOL\s*:\s*([a-z_]+(?:\|[^\n|]+){1,3})\s*$", raw, re.IGNORECASE | re.MULTILINE)
        return [spec.strip() for spec in specs[:3] if len(spec) <= 300]

    @staticmethod
    def _fallback_steps(state: MathState) -> List[str]:
        if state.problem_type in {"proof", "derivation"}:
            return [
                "明确题设、待证结论及可直接使用的定义或定理。",
                "将关键条件逐步代入或变形，说明每一步的依据。",
                "检查边界条件和量词，整理为完整结论。",
            ]
        return [
            "列出已知量、未知量和所求对象。",
            "选择与题设匹配的公式、定理或计算方法。",
            "核对结果是否满足题目全部条件。",
        ]
