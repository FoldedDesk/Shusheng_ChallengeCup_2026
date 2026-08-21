"""Parse grounded, declarative operation hints from model responses.

The model may select one whitelisted operation and copy its source expression
verbatim from the problem.  It cannot supply the computed result.  SymPy
recomputes that result through the existing restricted parser and binds the
certificate to the canonical problem statement.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult


class OperationLocator:
    """Turn one optional ``TOOL_JSON`` block into a certified local result."""

    _MARKER = re.compile(r"(?i)\bTOOL_JSON\s*:\s*")
    _ALLOWED = {
        "calculate",
        "solve_equation",
        "derivative",
        "definite_integral",
        "limit",
        "finite_sum",
        "solve_linear_system",
        "matrix_determinant",
        "matrix_rank",
        "matrix_inverse",
        "matrix_eigenvalues",
    }
    _MAX_JSON_CHARS = 2_000
    _MAX_SOURCE_CHARS = 900

    def __init__(self, symbolic: SympyTool | None = None) -> None:
        self.symbolic = symbolic or SympyTool()

    def result_from_response(
        self,
        problem: str,
        response: str,
        spec,
    ) -> ToolResult | None:
        """Recompute the last valid grounded operation, or abstain."""
        payloads = self._payloads(response)
        for payload in reversed(payloads):
            if set(payload) - {"operation", "source", "variable"}:
                continue
            operation = self._short_string(payload.get("operation"), 40).casefold()
            source = self._short_string(
                payload.get("source"), self._MAX_SOURCE_CHARS
            )
            raw_variable = self._short_string(payload.get("variable", ""), 16)
            variable = "" if raw_variable.casefold() in {"", "none", "null"} else raw_variable
            if operation not in self._ALLOWED or not source:
                continue
            if variable and not re.fullmatch(r"[A-Za-z]", variable):
                continue
            result = self.symbolic.result_from_located_fragment(
                problem,
                operation,
                source,
                variable,
                spec=spec,
            )
            if result is not None and result.verified:
                return result
        return None

    @classmethod
    def strip_blocks(cls, value: str) -> str:
        """Remove complete locator blocks before answer extraction."""
        text = str(value or "")
        spans = cls._payload_spans(text)
        if not spans:
            return text.strip()
        pieces: list[str] = []
        cursor = 0
        for start, end in spans:
            pieces.append(text[cursor:start])
            cursor = end
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
            if cursor < len(text) and text[cursor] == "\r":
                cursor += 1
            if cursor < len(text) and text[cursor] == "\n":
                cursor += 1
        pieces.append(text[cursor:])
        return re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()

    @classmethod
    def prompt_instruction(cls, language: str) -> str:
        operations = "|".join(sorted(cls._ALLOWED))
        if language == "zh":
            return (
                "若且仅若整题是下列一种直接运算，请紧接 FINAL 行输出一行 "
                "TOOL_JSON；source 必须逐字复制题面中的连续数学片段，不得改写，"
                f"JSON 中不得写计算结果。允许的 operation：{operations}。"
                "否则不要输出 TOOL_JSON。格式：\n"
                'TOOL_JSON: {"operation":"一个允许的名称",'
                '"source":"题面原片段","variable":"x或NONE"}'
            )
        return (
            "If and only if the whole problem is one direct operation below, emit one "
            "TOOL_JSON line immediately after FINAL. The source must be one contiguous "
            "verbatim mathematical fragment from the problem; never put a computed result "
            f"in the JSON. Allowed operations: {operations}. Otherwise omit TOOL_JSON. Format:\n"
            'TOOL_JSON: {"operation":"one allowed name",'
            '"source":"verbatim problem fragment","variable":"x or NONE"}'
        )

    @classmethod
    def _payloads(cls, value: str) -> tuple[dict[str, Any], ...]:
        text = str(value or "")
        payloads: list[dict[str, Any]] = []
        for start, end in cls._payload_spans(text):
            marker = cls._MARKER.search(text, start, end)
            if marker is None:
                continue
            raw = text[marker.end():end].strip()
            if len(raw) > cls._MAX_JSON_CHARS:
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
        return tuple(payloads)

    @classmethod
    def _payload_spans(cls, value: str) -> tuple[tuple[int, int], ...]:
        text = str(value or "")
        spans: list[tuple[int, int]] = []
        for marker in cls._MARKER.finditer(text):
            brace = text.find("{", marker.end())
            if brace < 0 or brace - marker.end() > 16:
                continue
            end = cls._balanced_object_end(text, brace)
            if end is not None:
                spans.append((marker.start(), end))
        return tuple(spans)

    @classmethod
    def _balanced_object_end(cls, value: str, start: int) -> int | None:
        depth = 0
        quoted = escaped = False
        for index in range(start, min(len(value), start + cls._MAX_JSON_CHARS + 1)):
            char = value[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
        return None

    @staticmethod
    def _short_string(value: Any, limit: int) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip()
        return text if 0 < len(text) <= limit else ""
