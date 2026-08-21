"""Safe, model-supplied certificates for candidate-level math checks.

The model may describe a small deterministic check as JSON.  This module
parses only a fixed schema and delegates expression handling to the existing
whitelisted SymPy frontend.  Missing or malformed certificates are ignored;
only a grounded, candidate-relevant computation can pass or fail a candidate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from tools.sympy_tool import SympyTool, ToolCheck


class StructuredVerificationTool:
    """Execute compact verification certificates without executing code."""

    _MARKER = re.compile(r"(?i)\bVERIFY_JSON\s*:\s*")
    _MAX_CERTIFICATE_CHARS = 6_000
    _MAX_CHECKS = 3

    def __init__(self, symbolic: SympyTool | None = None) -> None:
        self.symbolic = symbolic or SympyTool()
        self.sp = self.symbolic.sympy

    def verify_response(
        self,
        problem: str,
        raw_response: str,
        candidate: str,
    ) -> tuple[ToolCheck, ...]:
        """Return checks that are both executable and relevant to *candidate*."""
        if not self.sp:
            return ()
        payloads = self._payloads(raw_response)
        checks: list[ToolCheck] = []
        for payload in payloads:
            for item in self._items(payload):
                result = self._execute(problem, candidate, item)
                if result is not None:
                    checks.append(result)
                if len(checks) >= self._MAX_CHECKS:
                    return tuple(checks)
        return tuple(checks)

    @classmethod
    def prompt_instruction(cls, language: str) -> str:
        """Describe the optional declarative certificate, never executable code."""
        kinds = (
            "exact_value|equation_roots|derivative|antiderivative|"
            "definite_integral|identity|substitution"
        )
        if language == "zh":
            return (
                "若最终答案含可机械复核的代数或微积分结论，可在 FINAL 答案之后额外"
                "输出一行 VERIFY_JSON；否则省略。它只能是 JSON 数据，禁止 Python 或"
                "伪代码。source 必须逐字复制题面中的连续数学片段，claim 必须逐字出现"
                f"在 FINAL 答案中，kind 只能是 {kinds} 之一，最多 3 个检查。"
                "单检查格式：VERIFY_JSON: {\"kind\":\"exact_value\","
                "\"source\":\"题面原片段\",\"claim\":\"FINAL中的结论\"}。"
                "方程根可另用 roots 字符串数组和单字母 variable；定积分可另用"
                " variable/lower/upper；代入可另用 assignments 对象。不要声称检查已通过。"
            )
        return (
            "If FINAL contains an algebraic or calculus claim that can be mechanically "
            "checked, you may emit one VERIFY_JSON line after FINAL; otherwise omit it. "
            "It must be JSON data only, never Python or pseudocode. source must be one "
            "contiguous verbatim mathematical fragment from the problem and claim must "
            f"appear verbatim in FINAL. kind must be one of {kinds}; at most 3 checks. "
            "Single-check form: VERIFY_JSON: {\"kind\":\"exact_value\","
            "\"source\":\"verbatim problem fragment\",\"claim\":\"claim in FINAL\"}. "
            "Equation roots may additionally use a string array roots and one-letter "
            "variable; definite integrals may use variable/lower/upper; substitutions may "
            "use an assignments object. Never claim that the check passed."
        )

    @classmethod
    def strip_certificates(cls, value: str) -> str:
        """Remove certificate lines/blocks before answer extraction."""
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
    def _payloads(cls, value: str) -> tuple[dict[str, Any], ...]:
        text = str(value or "")
        payloads: list[dict[str, Any]] = []
        for start, end in cls._payload_spans(text):
            marker = cls._MARKER.search(text, start, end)
            if marker is None:
                continue
            raw = text[marker.end():end].strip()
            if len(raw) > cls._MAX_CERTIFICATE_CHARS:
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
            if brace < 0 or brace - marker.end() > 32:
                continue
            end = cls._balanced_object_end(text, brace)
            if end is not None:
                spans.append((marker.start(), end))
        return tuple(spans)

    @staticmethod
    def _balanced_object_end(value: str, start: int) -> int | None:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, min(len(value), start + 6_001)):
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

    @classmethod
    def _items(cls, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
        raw_items = payload.get("checks")
        if isinstance(raw_items, list):
            items = raw_items[: cls._MAX_CHECKS]
        else:
            items = [payload]
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("kind"), str):
                yield item

    def _execute(
        self,
        problem: str,
        candidate: str,
        item: dict[str, Any],
    ) -> ToolCheck | None:
        kind = str(item.get("kind", "")).strip().lower()
        handlers = {
            "exact_value": self._exact_value,
            "equation_roots": self._equation_roots,
            "derivative": self._derivative,
            "antiderivative": self._antiderivative,
            "definite_integral": self._definite_integral,
            "identity": self._identity,
            "substitution": self._substitution,
        }
        handler = handlers.get(kind)
        if handler is None:
            return None
        try:
            return handler(problem, candidate, item)
        except Exception:
            # A certificate is untrusted input.  Unsupported syntax is an
            # abstention, not evidence against the candidate.
            return None

    def _exact_value(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        if not source:
            return None
        expression = self._math_body(source)
        if re.search(r"(?<![<>!])=(?!=)|\\(?:sum|int|lim)", expression):
            return None
        actual = self.sp.simplify(self.symbolic._parse(expression))
        expected = self.symbolic._parse(claim)
        passed = self.sp.simplify(actual - expected) == 0
        return self._result("structured_exact_value", passed, decisive=True)

    def _equation_roots(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        roots = item.get("roots")
        variable = self._short_string(item.get("variable"), 16)
        if not source or not variable or not isinstance(roots, list) or len(roots) > 20:
            return None
        if len(re.findall(r"(?<![<>!])=(?!=)", source)) != 1:
            return None
        left, right = re.split(r"(?<![<>!])=(?!=)", source, maxsplit=1)
        symbol = self.sp.Symbol(variable)
        expression = self.symbolic._parse(self._math_body(left)) - self.symbolic._parse(
            self._math_body(right)
        )
        if expression.free_symbols != {symbol}:
            return None
        proposed = tuple(self.symbolic._parse(self._short_string(root, 300)) for root in roots)
        solved = tuple(self.sp.solve(expression, symbol))
        passed = self._same_expression_set(proposed, solved) and all(
            self.sp.simplify(expression.subs(symbol, root)) == 0 for root in proposed
        )
        return self._result("structured_equation_roots", passed, decisive=True)

    def _derivative(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        variable = self._short_string(item.get("variable"), 16)
        if not source or not variable:
            return None
        symbol = self.sp.Symbol(variable)
        expression = self.symbolic._parse(self._math_body(source))
        expected = self.symbolic._parse(claim)
        passed = self.sp.simplify(self.sp.diff(expression, symbol) - expected) == 0
        return self._result("structured_derivative", passed, decisive=True)

    def _antiderivative(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        variable = self._short_string(item.get("variable"), 16)
        if not source or not variable:
            return None
        symbol = self.sp.Symbol(variable)
        integrand = self.symbolic._parse(self._math_body(source))
        proposed = self.symbolic._parse(claim)
        passed = self.sp.simplify(self.sp.diff(proposed, symbol) - integrand) == 0
        return self._result("structured_antiderivative", passed, decisive=True)

    def _definite_integral(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        variable = self._short_string(item.get("variable"), 16)
        lower = self._short_string(item.get("lower"), 300)
        upper = self._short_string(item.get("upper"), 300)
        if not source or not variable or not lower or not upper:
            return None
        symbol = self.sp.Symbol(variable)
        integrand = self.symbolic._parse(self._math_body(source))
        actual = self.sp.integrate(
            integrand,
            (symbol, self.symbolic._parse(lower), self.symbolic._parse(upper)),
        )
        expected = self.symbolic._parse(claim)
        if actual.has(self.sp.Integral):
            return None
        passed = self.sp.simplify(actual - expected) == 0
        return self._result("structured_definite_integral", passed, decisive=True)

    def _identity(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        left = self._short_string(item.get("left"), 1_000)
        right = self._short_string(item.get("right"), 1_000)
        if not left or not right:
            return None
        rendered = f"{left}={right}"
        if not self._claim_matches(candidate, rendered):
            return None
        passed = self.sp.simplify(
            self.symbolic._parse(left) - self.symbolic._parse(right)
        ) == 0
        return self._result("structured_identity", passed, decisive=False)

    def _substitution(
        self, problem: str, candidate: str, item: dict[str, Any]
    ) -> ToolCheck | None:
        source, claim = self._source_and_claim(problem, candidate, item)
        assignments = item.get("assignments")
        if not source or not isinstance(assignments, dict) or len(assignments) > 8:
            return None
        substitutions = {}
        for name, value in assignments.items():
            variable = self._short_string(name, 16)
            scalar = self._short_string(value, 300)
            if not re.fullmatch(r"[A-Za-z]", variable) or not scalar:
                return None
            substitutions[self.sp.Symbol(variable)] = self.symbolic._parse(scalar)
        actual = self.sp.simplify(
            self.symbolic._parse(self._math_body(source)).subs(substitutions)
        )
        expected = self.symbolic._parse(claim)
        passed = self.sp.simplify(actual - expected) == 0
        return self._result("structured_substitution", passed, decisive=True)

    def _source_and_claim(
        self,
        problem: str,
        candidate: str,
        item: dict[str, Any],
    ) -> tuple[str, str]:
        source = self._short_string(item.get("source"), 1_000)
        claim = self._short_string(item.get("claim"), 1_000)
        if (
            not source
            or not claim
            or not self._grounded(problem, source)
            or not self._claim_matches(candidate, claim)
        ):
            return "", ""
        return source, claim

    @staticmethod
    def _result(name: str, passed: bool, *, decisive: bool) -> ToolCheck:
        return ToolCheck(
            name,
            "pass" if passed else "fail",
            "grounded structured certificate recomputed locally",
            decisive,
        )

    def _same_expression_set(self, left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for item in left:
            match = next(
                (
                    index
                    for index, other in enumerate(unmatched)
                    if self.sp.simplify(item - other) == 0
                ),
                None,
            )
            if match is None:
                return False
            unmatched.pop(match)
        return not unmatched

    @classmethod
    def _grounded(cls, problem: str, source: str) -> bool:
        needle = cls._compact_math(source)
        haystack = cls._compact_math(problem)
        return bool(len(needle) >= 3 and needle in haystack)

    def _claim_matches(self, candidate: str, claim: str) -> bool:
        needle = self._compact_math(claim)
        haystack = self._compact_math(candidate)
        if not needle or not haystack:
            return False
        if needle == haystack or needle in haystack:
            return True
        try:
            candidate_scalar = self.symbolic._first_scalar_result(candidate)
            claim_scalar = self.symbolic._parse(self._math_body(claim))
            return bool(
                candidate_scalar is not None
                and self.sp.simplify(candidate_scalar - claim_scalar) == 0
            )
        except Exception:
            return False

    @staticmethod
    def _short_string(value: Any, limit: int) -> str:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return ""
        text = str(value).strip()
        return text if 0 < len(text) <= limit else ""

    @staticmethod
    def _math_body(value: str) -> str:
        text = str(value or "").strip()
        for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
            if text.startswith(left) and text.endswith(right):
                return text[len(left):-len(right)].strip()
        return text

    @classmethod
    def _compact_math(cls, value: str) -> str:
        text = cls._math_body(str(value or ""))
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = re.sub(r"\s+", "", text)
        return text.strip("。.!;；,，")
