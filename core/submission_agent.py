"""Minimal, platform-safe implementation used by the public entry point."""

from __future__ import annotations

from pathlib import Path

from classifier import classify_problem_type
from reasoning.finalizer import Finalizer
from tools.sympy_tool import SympyTool


class SubmissionAgent:
    """Solve one problem with one official-client request and local fallback."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.prompt = self._load_prompt()

    def solve(self, problem: str, metadata: dict) -> dict:
        text = str(problem or "").strip()
        problem_type = classify_problem_type(text)
        hints = self.sympy.hints_for(text)
        request = self._request(text, problem_type, hints)
        trace = [
            {"step": "plan", "content": {"problem_type": problem_type, "sympy_hint_count": len(hints)}},
        ]
        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": request},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            trace.append({"step": "model_call", "content": "completed"})
        except Exception as exc:  # The platform client owns retries and limits.
            response = ""
            trace.append({"step": "model_call", "content": {"status": "failed", "type": type(exc).__name__}})

        answer = self._finalize(str(response or ""), problem_type, hints)
        trace.append({"step": "finalize", "content": {"non_empty": bool(answer)}})
        return {"final_response": answer, "trace": trace}

    @staticmethod
    def _load_prompt() -> str:
        try:
            return (Path("prompts") / "solver.txt").read_text(encoding="utf-8")
        except OSError:
            return "请用中文解答数学题，给出必要推导，并在末尾写【最终答案】。"

    @staticmethod
    def _request(problem: str, problem_type: str, hints: list[str]) -> str:
        content = f"题目：\n{problem}\n\n"
        if hints:
            content += "本地符号计算仅供核对：\n" + "\n".join(hints) + "\n\n"
        if problem_type in {"proof", "derivation", "explanation"}:
            content += (
                "请给出简洁、自洽的证明或推导：说明关键定义、定理或条件，展示必要步骤，"
                "最后单独写【最终答案】。"
            )
        else:
            content += "请完整求解，覆盖题目全部所求对象，最后单独写【最终答案】。"
        return content

    @staticmethod
    def _finalize(response: str, problem_type: str, hints: list[str]) -> str:
        if response.strip():
            answer = (
                Finalizer.extract_solution(response)
                if problem_type in {"proof", "derivation", "explanation"}
                else Finalizer.extract(response)
            )
            if answer.strip():
                return answer.strip()
        for hint in hints:
            _, separator, result = hint.partition(": ")
            if separator and result.strip():
                return result.strip()
        return "未能生成可验证的数学答案。"
