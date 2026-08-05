"""Minimal, platform-safe implementation used by the public entry point."""

from __future__ import annotations

from pathlib import Path
import re

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
            trace.append({
                "step": "model_call",
                "content": {"status": "completed", "response_non_empty": bool(str(response or "").strip())},
            })
        except Exception as exc:  # The platform client owns retries and limits.
            response = ""
            trace.append({"step": "model_call", "content": {"status": "failed", "type": type(exc).__name__}})

        answer, source = self._finalize(str(response or ""), problem_type, hints)
        trace.append({"step": "finalize", "content": {"non_empty": bool(answer), "source": source}})
        return {"final_response": answer, "trace": trace}

    @staticmethod
    def _load_prompt() -> str:
        try:
            return (Path("prompts") / "submission.txt").read_text(encoding="utf-8")
        except OSError:
            return "直接给出简洁中文答案，不要输出思维草稿，最后写 \\boxed{最终答案}。"

    @staticmethod
    def _request(problem: str, problem_type: str, hints: list[str]) -> str:
        content = f"题目：\n{problem}\n\n"
        if hints:
            content += "本地符号计算仅供核对：\n" + "\n".join(hints) + "\n\n"
        if problem_type in {"proof", "derivation", "explanation"}:
            content += (
                "请给出简洁、自洽的证明或推导：说明关键定义、定理或条件，展示必要步骤，"
                "最后一行写 \\boxed{最终结论}。"
            )
        else:
            content += "请完整求解，覆盖题目全部所求对象，最后一行写 \\boxed{最终答案}。"
        return content

    @staticmethod
    def _finalize(response: str, problem_type: str, hints: list[str]) -> tuple[str, str]:
        if response.strip():
            if problem_type in {"proof", "derivation", "explanation"}:
                explicit = SubmissionAgent._explicit_answer(response)
                answer = explicit if SubmissionAgent._is_scratchpad(response) and explicit else Finalizer.extract_solution(response)
            else:
                answer = SubmissionAgent._explicit_answer(response) or Finalizer.extract(response)
            if answer.strip():
                return answer.strip(), "model"
        for hint in hints:
            _, separator, result = hint.partition(": ")
            if separator and result.strip():
                return result.strip(), "sympy"
        return "未能生成可验证的数学答案。", "fallback"

    @staticmethod
    def _explicit_answer(response: str) -> str:
        """Accept common final-answer labels used by public chat clients."""
        boxed = SubmissionAgent._last_boxed(response)
        if boxed:
            return Finalizer.extract(boxed)
        matches = re.findall(
            r"(?:【最终答案】|(?:最终)?答案|结论|FINAL)\s*[:：]?\s*([^\n]+)",
            response,
            flags=re.IGNORECASE,
        )
        return Finalizer.extract(matches[-1]) if matches else ""

    @staticmethod
    def _last_boxed(response: str) -> str:
        marker = r"\boxed{"
        position = response.rfind(marker)
        if position < 0:
            return ""
        start = position + len(marker)
        depth = 1
        for index in range(start, len(response)):
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
                if depth == 0:
                    return response[start:index].strip()
        return ""

    @staticmethod
    def _is_scratchpad(response: str) -> bool:
        return bool(re.search(r"(?im)^\s*(thinking process|analysis|drafting)", response))
