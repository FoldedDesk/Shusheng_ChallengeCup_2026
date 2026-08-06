"""Minimal, platform-safe implementation used by the public entry point."""

from __future__ import annotations

from pathlib import Path
import re

from classifier import classify_problem_type
from reasoning.finalizer import Finalizer
from tools.sympy_tool import SympyTool


class SubmissionAgent:
    """Solve and verify each problem with at most two official-client calls."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.prompt = self._load_prompt()

    def solve(self, problem: str, metadata: dict) -> dict:
        text = str(problem or "").strip()
        problem_type = classify_problem_type(text)
        hints = self.sympy.hints_for(text)
        trace = [
            {
                "step": "plan",
                "content": {"problem_type": problem_type, "sympy_hint_count": len(hints), "max_model_calls": 2},
            },
        ]
        first = self._call(self._request(text, problem_type, hints), "solve", trace)
        second = self._call(self._review_request(text, problem_type, hints, first), "verify", trace)

        first_answer, first_source = self._finalize(first, problem_type, hints)
        second_answer, second_source = self._finalize(second, problem_type, hints)
        answer, source = self._select_answer(
            first_answer,
            first_source,
            second_answer,
            second_source,
        )
        trace.append({"step": "finalize", "content": {"non_empty": bool(answer), "source": source}})
        return {"final_response": answer, "trace": trace}

    def _call(self, request: str, stage: str, trace: list[dict]) -> str:
        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": request},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            value = str(response or "")
            trace.append({
                "step": f"model_call_{stage}",
                "content": {"status": "completed", "response_non_empty": bool(value.strip())},
            })
            return value
        except Exception as exc:  # The platform client owns retries and limits.
            trace.append({
                "step": f"model_call_{stage}",
                "content": {"status": "failed", "type": type(exc).__name__},
            })
            return ""

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
    def _review_request(problem: str, problem_type: str, hints: list[str], candidate: str) -> str:
        evidence = SubmissionAgent._review_evidence(candidate)
        content = (
            f"题目：\n{problem}\n\n"
            "下面是第一阶段候选。请独立核验并重新求解；候选正确则保留，错误则纠正。"
            "不要输出思维草稿，直接给出可判分答案，最后一行写 \\boxed{最终答案}。\n\n"
            f"第一阶段候选：\n{evidence}"
        )
        if hints:
            content += "\n\n本地符号计算仅供核对：\n" + "\n".join(hints)
        if problem_type in {"proof", "derivation", "explanation"}:
            content += "\n\n证明题须保留关键依据与简洁推导，再给出盒装结论。"
        return content

    @staticmethod
    def _finalize(response: str, problem_type: str, hints: list[str]) -> tuple[str, str]:
        if response.strip():
            explicit = SubmissionAgent._explicit_answer(response)
            if problem_type in {"proof", "derivation", "explanation"}:
                if SubmissionAgent._is_scratchpad(response):
                    answer = explicit
                    source = "model_explicit"
                else:
                    answer = Finalizer.extract_solution(response)
                    source = "model_unstructured"
            else:
                if explicit:
                    answer = explicit
                    source = "model_explicit"
                elif SubmissionAgent._is_scratchpad(response):
                    answer = ""
                    source = ""
                else:
                    answer = Finalizer.extract(response)
                    source = "model_unstructured"
            if answer.strip():
                return answer.strip(), source
        for hint in hints:
            _, separator, result = hint.partition(": ")
            if separator and result.strip():
                return result.strip(), "sympy"
        return "未能生成可验证的数学答案。", "fallback"

    @staticmethod
    def _select_answer(
        first_answer: str,
        first_source: str,
        second_answer: str,
        second_source: str,
    ) -> tuple[str, str]:
        for answer, source in (
            (second_answer, second_source),
            (first_answer, first_source),
        ):
            if source == "model_explicit" and answer.strip():
                return answer, source
        for answer, source in (
            (second_answer, second_source),
            (first_answer, first_source),
        ):
            if source == "model_unstructured" and answer.strip():
                return answer, source
        for answer, source in (
            (second_answer, second_source),
            (first_answer, first_source),
        ):
            if source == "sympy" and answer.strip():
                return answer, source
        return second_answer or first_answer or "未能生成可验证的数学答案。", "fallback"

    @staticmethod
    def _review_evidence(candidate: str) -> str:
        explicit = SubmissionAgent._explicit_answer(candidate)
        if explicit:
            return explicit
        return candidate.strip()[-2400:] or "（第一阶段未生成可用候选）"

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
        return bool(
            re.search(
                r"(?im)^\s*(thinking process|analysis|drafting|思考过程|分析过程|推理过程|<think>)",
                response,
            )
        )
