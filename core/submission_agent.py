"""Minimal, platform-safe implementation used by the public entry point."""

from __future__ import annotations

from pathlib import Path
import re

from classifier import ProblemProfile, classify_profile
from reasoning.finalizer import Finalizer
from tools.sympy_tool import SympyTool


class SubmissionAgent:
    """Solve each problem with one safe tool route or a bounded review route."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.prompt = self._load_prompt()

    def solve(self, problem: str, metadata: dict) -> dict:
        text = str(problem or "").strip()
        profile = classify_profile(text)
        hints = self.sympy.hints_for(text)
        tool_answer = self._tool_answer(hints, profile)
        one_call_route = bool(tool_answer and profile.problem_type == "calculation")
        trace = [
            {
                "step": "classification",
                "content": {
                    **profile.trace_content(),
                    "sympy_hint_count": len(hints),
                    "route": "tool_assisted" if one_call_route else "solve_and_verify",
                },
            },
        ]
        trace.append({
            "step": "tool",
            "content": {
                "hint_count": len(hints),
                "eligible": profile.tool_eligible,
                "verified_candidate": bool(tool_answer),
            },
        })
        first = self._call(self._request(text, profile, hints), "solve", trace)
        second = ""
        if not one_call_route:
            second = self._call(self._review_request(text, profile, hints, first), "verify", trace)

        first_answer, first_source = self._finalize(first, profile)
        second_answer, second_source = self._finalize(second, profile)
        answer, source = self._select_answer(
            first_answer,
            first_source,
            second_answer,
            second_source,
            tool_answer,
            profile,
        )
        trace.append({
            "step": "selection",
            "content": {
                "first_source": first_source or "none",
                "second_source": second_source or "none",
                "selected_source": source,
            },
        })
        trace.append({
            "step": "finalize",
            "content": {"non_empty": bool(answer), "source": source, "tool_verified": bool(tool_answer)},
        })
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
    def _request(problem: str, profile: ProblemProfile, hints: list[str]) -> str:
        content = f"题目：\n{problem}\n\n"
        if hints:
            content += "本地符号计算仅供核对：\n" + "\n".join(hints) + "\n\n"
        if profile.problem_type in {"proof", "derivation", "explanation"}:
            content += (
                "请给出简洁、自洽的证明或推导：说明关键定义、定理或条件，展示必要步骤，"
                "最后一行写 \\boxed{最终结论}。"
            )
        else:
            content += SubmissionAgent._calculation_instruction(profile)
        content += "\n" + SubmissionAgent._subject_instruction(profile.subject)
        return content

    @staticmethod
    def _review_request(problem: str, profile: ProblemProfile, hints: list[str], candidate: str) -> str:
        evidence = SubmissionAgent._review_evidence(candidate)
        content = (
            f"题目：\n{problem}\n\n"
            "下面是第一阶段候选。请独立核验并重新求解；候选正确则保留，错误则纠正。"
            "不要输出思维草稿，直接给出可判分答案，最后一行写 \\boxed{最终答案}。\n\n"
            f"第一阶段候选：\n{evidence}"
        )
        if hints:
            content += "\n\n本地符号计算仅供核对：\n" + "\n".join(hints)
        if profile.problem_type in {"proof", "derivation", "explanation"}:
            content += "\n\n证明题须保留关键依据与简洁推导，再给出盒装结论。"
        elif profile.answer_shape == "roots":
            content += "\n\n请逐个代回检查所有根；多个离散根不要写成区间。"
        elif profile.answer_shape == "interval":
            content += "\n\n请检查端点是否包含，并以区间或并集形式给出不等式解集。"
        content += "\n" + SubmissionAgent._subject_instruction(profile.subject)
        return content

    @staticmethod
    def _finalize(response: str, profile: ProblemProfile) -> tuple[str, str]:
        if response.strip():
            explicit = SubmissionAgent._explicit_answer(response)
            if profile.problem_type in {"proof", "derivation", "explanation"}:
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
                return SubmissionAgent._normalize_answer(answer, profile), source
        return "", ""

    @staticmethod
    def _select_answer(
        first_answer: str,
        first_source: str,
        second_answer: str,
        second_source: str,
        tool_answer: str,
        profile: ProblemProfile,
    ) -> tuple[str, str]:
        # A complete deterministic result is more reliable than an unverified
        # model disagreement for elementary calculation tasks.
        if tool_answer and profile.problem_type == "calculation":
            return tool_answer, "sympy_verified"
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
    def _tool_answer(hints: list[str], profile: ProblemProfile) -> str:
        if not profile.tool_eligible or profile.confidence != "high":
            return ""
        supported = ("SymPy 计算", "SymPy 方程解", "SymPy 导数", "SymPy 定积分", "SymPy 不定积分", "SymPy 极限")
        for hint in hints:
            label, separator, result = hint.partition(": ")
            if separator and label in supported and result.strip():
                return SubmissionAgent._normalize_answer(result, profile)
        return ""

    @staticmethod
    def _calculation_instruction(profile: ProblemProfile) -> str:
        if profile.answer_shape == "roots":
            return "请完整求解并逐根检查。多个离散根写为方程根或集合，不能写成区间；最后一行写 \\boxed{最终答案}。"
        if profile.answer_shape == "interval":
            return "请解出不等式并检查端点，使用区间或并集表示解集；最后一行写 \\boxed{最终答案}。"
        if profile.answer_shape == "matrix":
            return "请检查矩阵维度与所求对象，最后一行写 \\boxed{最终答案}。"
        return "请完整求解，覆盖题目全部所求对象，最后一行写 \\boxed{最终答案}。"

    @staticmethod
    def _subject_instruction(subject: str) -> str:
        instructions = {
            "抽象代数": "代数结构题请明确运算、子结构或同态条件，检查定义域和值域。",
            "高等代数": "线性或多项式题请检查维度、可逆性、根的重数或矩阵维度。",
            "线性代数": "线性代数题请检查维度、基、秩和矩阵乘法顺序。",
            "概率论": "概率题请说明样本空间和事件关系，并检查结果在 [0,1] 内。",
            "离散数学": "计数题请检查对象是否重复或遗漏，并核对边界情形。",
            "数学分析": "分析题请检查定义域、收敛条件及极限或积分的适用前提。",
            "常微分方程": "微分方程题请代回方程和初值条件核验常数。",
            "复分析": "复分析题请明确解析区域、奇点和积分路径等必要条件。",
        }
        return instructions.get(subject, "请检查题目中的全部条件、量词和边界情形。")

    @staticmethod
    def _normalize_answer(answer: str, profile: ProblemProfile) -> str:
        value = answer.strip().replace(r"\infty", "∞")
        value = re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", value)
        value = value.replace(r"\left", "").replace(r"\right", "")
        if profile.answer_shape == "roots":
            matched = re.fullmatch(r"\[\s*(-?\d+(?:/\d+)?)\s*[,，]\s*(-?\d+(?:/\d+)?)\s*\]", value)
            if matched:
                return f"x={matched.group(1)} 或 x={matched.group(2)}"
        return value

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
            r"(?:【最终答案】|(?:最终)?答案|结论|FINAL(?:\s*ANSWER)?|ANSWER)\s*[:：]?\s*([^\n]+)",
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
