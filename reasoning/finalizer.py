from __future__ import annotations

import re

from tools.latex_parser import normalize_latex


class Finalizer:
    @staticmethod
    def extract(candidate: str) -> str:
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", candidate)
        for match in reversed(matches):
            answer = Finalizer._clean(match)
            if Finalizer._is_submittable(answer):
                return answer
        recovered = Finalizer._recover_quoted_chinese_answer(candidate)
        if recovered:
            return recovered
        boxed = re.findall(r"\\boxed\{(.+)\}", candidate)
        if boxed:
            return Finalizer._clean(boxed[-1])
        return Finalizer._clean(candidate.strip())

    @staticmethod
    def extract_solution(candidate: str) -> str:
        """Keep a proof's reasoning while applying only display cleanup."""
        return Finalizer._clean(candidate.strip())

    @staticmethod
    def _clean(answer: str) -> str:
        value = re.sub(r"^```(?:latex|text|markdown)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
        value = normalize_latex(value).strip().strip('"“”')
        # A dangling inline-math delimiter is a common model formatting error.
        if value.count("$") % 2:
            value += "$"
        return value

    @staticmethod
    def _is_submittable(answer: str) -> bool:
        if not answer or re.search(r"<[^>]+>", answer):
            return False
        english_words = re.findall(r"\b[A-Za-z]{3,}\b", answer)
        if len(english_words) >= 5:
            return False
        return True

    @staticmethod
    def _recover_quoted_chinese_answer(candidate: str) -> str:
        quoted = re.findall(r"[\"“]([^\"”\n]+)[\"”]", candidate)
        for value in reversed(quoted):
            answer = Finalizer._clean(value)
            if re.search(r"[\u4e00-\u9fff]", answer) and Finalizer._is_submittable(answer):
                return answer
        fragments = re.findall(r"[\u4e00-\u9fff][^\"“”\n]*", candidate)
        for value in reversed(fragments):
            answer = Finalizer._clean(value)
            if (
                re.search(r"[=＝\d]", answer)
                and "<" not in answer
                and Finalizer._is_submittable(answer)
            ):
                return answer.rstrip(".。")
        return ""
