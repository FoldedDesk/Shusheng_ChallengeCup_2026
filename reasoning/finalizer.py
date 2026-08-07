from __future__ import annotations

from dataclasses import dataclass
import re

from tools.latex_parser import normalize_latex


@dataclass(frozen=True)
class ExtractionResult:
    answer: str
    method: str
    valid: bool
    rejected_reasons: tuple[str, ...] = ()
    raw_has_meta: bool = False
    explicit_answer: bool = False


class Finalizer:
    """Extract explicit answer candidates without silently repairing malformed text."""

    _LABEL = re.compile(
        r"(?im)^\s*(?:【\s*(?:最终答案|答案|结论)\s*】|(?:最终\s*)?答案|结论|final\s+answer|answer)\s*[:：]?\s*([^\n]+?)\s*$"
    )
    _PLACEHOLDER = re.compile(
        r"^(?:最终答案|完整答案|完整结论|答案|final(?:\s+answer)?|answer|check\s+format(?:ting)?|format(?:ting)?|(?:final\s+)?(?:conclusion|response|done)|[.。…`'\"，,]+)$",
        re.IGNORECASE,
    )
    _META = re.compile(
        r"(?:<think\b|thinking process|(?im:^\s*(?:analysis|drafting)\s*[:：])|check formatting|check spacing|"
        r"system prompt|prompt instruction|final answer should|最后一行必须|思考过程|分析过程|推理过程|"
        r"格式检查|检查格式|提示词)",
        re.IGNORECASE,
    )

    @staticmethod
    def extract(candidate: str) -> str:
        result = Finalizer.extract_result(candidate)
        return result.answer if result.valid else ""

    @staticmethod
    def extract_result(candidate: str) -> ExtractionResult:
        text = str(candidate or "").strip()
        if not text:
            return ExtractionResult("", "empty", False, ("empty",))
        text = re.sub(r"<\|(?:assistant|user|system|endoftext)\|>", "", text, flags=re.IGNORECASE).strip()
        raw_has_meta = Finalizer.contains_meta(text)

        labels = Finalizer._LABEL.findall(text)
        if labels:
            return Finalizer._result(labels[-1], "label", raw_has_meta=raw_has_meta, explicit=True)
        boxed = Finalizer._last_boxed(text)
        if boxed is not None:
            if not boxed and r"\boxed{" in text:
                # Preserve the malformed source so structural validation can
                # report the actual truncation rather than only an empty value.
                return Finalizer._result(text, "boxed_unclosed", raw_has_meta=raw_has_meta, explicit=True)
            return Finalizer._result(boxed, "boxed", raw_has_meta=raw_has_meta, explicit=True)
        final_lines = re.findall(r"(?im)^\s*final\s*[:：]\s*([^\n]+)", text)
        if final_lines:
            return Finalizer._result(final_lines[-1], "final_marker", raw_has_meta=raw_has_meta, explicit=True)
        if raw_has_meta:
            return ExtractionResult("", "meta_without_explicit_answer", False, ("meta_without_explicit_answer",), True, False)
        recovered = Finalizer._recover_quoted_chinese_answer(text)
        if recovered:
            return Finalizer._result(recovered, "quoted")
        return Finalizer._result(text, "whole_response")

    @staticmethod
    def contains_meta(value: str) -> bool:
        return bool(Finalizer._META.search(str(value or "")))

    @staticmethod
    def extract_solution(candidate: str) -> str:
        """Keep a proof's reasoning while applying only display cleanup."""
        return Finalizer._clean(str(candidate or "").strip())

    @staticmethod
    def validate_structure(answer: str) -> tuple[str, ...]:
        value = str(answer or "").strip()
        reasons: list[str] = []
        if not value:
            return ("empty",)
        if Finalizer._PLACEHOLDER.fullmatch(value):
            reasons.append("placeholder")
        if re.fullmatch(r"(?:final\s+)?(?:conclusion|response|done)[.。!?！]?", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.fullmatch(
            r"(?:final\s+)?(?:check|checking)(?:\s+(?:on|the|all|format(?:ting)?|constraints?|answer|result)){0,4}\s*[:：]?",
            value,
            re.IGNORECASE,
        ):
            reasons.append("placeholder")
        if Finalizer._META.search(value):
            reasons.append("meta_text")
        if not re.search(r"[\w\u4e00-\u9fff=+\-*/^\\]", value):
            reasons.append("meaningless_fragment")
        if re.search(r"<[^>]+>", value):
            reasons.append("markup_fragment")
        if value.count("```") % 2:
            reasons.append("unclosed_code_fence")
        if value.count("$") % 2:
            reasons.append("unclosed_inline_math")
        if value.count(r"\(") != value.count(r"\)"):
            reasons.append("unclosed_inline_latex")
        if value.count(r"\[") != value.count(r"\]"):
            reasons.append("unclosed_display_latex")
        for environment in re.findall(r"\\begin\{([^}]+)\}", value):
            if len(re.findall(rf"\\end\{{{re.escape(environment)}\}}", value)) < len(
                re.findall(rf"\\begin\{{{re.escape(environment)}\}}", value)
            ):
                reasons.append("unclosed_latex_environment")
                break
        if not Finalizer._balanced_braces(value):
            reasons.append("unclosed_latex_brace")
        return tuple(reasons)

    @staticmethod
    def _result(
        value: str,
        method: str,
        *,
        raw_has_meta: bool = False,
        explicit: bool = False,
    ) -> ExtractionResult:
        answer = Finalizer._clean(value)
        reasons = Finalizer.validate_structure(answer)
        return ExtractionResult(answer if not reasons else "", method, not reasons, reasons, raw_has_meta, explicit)

    @staticmethod
    def _clean(answer: str) -> str:
        value = re.sub(r"^```(?:latex|text|markdown)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
        return normalize_latex(value).strip().strip('"“”')

    @staticmethod
    def _last_boxed(text: str) -> str | None:
        marker = r"\boxed{"
        position = text.rfind(marker)
        if position < 0:
            return None
        start = position + len(marker)
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index].strip()
        return ""

    @staticmethod
    def _balanced_braces(value: str) -> bool:
        depth = 0
        escaped = False
        for char in value:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == "{" and not escaped:
                depth += 1
            elif char == "}" and not escaped:
                depth -= 1
                if depth < 0:
                    return False
            escaped = False
        return depth == 0

    @staticmethod
    def _recover_quoted_chinese_answer(candidate: str) -> str:
        quoted = re.findall(r"[\"“]([^\"”\n]+)[\"”]", candidate)
        for value in reversed(quoted):
            answer = Finalizer._clean(value)
            if re.search(r"[\u4e00-\u9fff]", answer) and not Finalizer.validate_structure(answer):
                return answer
        return ""
