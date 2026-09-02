"""Compact, untrusted method discovery for one unseen mathematics problem."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


_ANSWER_MARKER = re.compile(
    r"\\boxed\s*\{|\bFINAL(?:\s+ANSWER)?\s*[:：]|(?:最终)?答案\s*[:：]",
    re.IGNORECASE,
)
_GENERIC_STEP = re.compile(
    r"^(?:solve|calculate|compute|prove|derive|analyze|use the method|"
    r"求解|计算|证明|推导|分析)(?:\s+(?:the|this)\s+(?:problem|claim))?[.!。]?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScoutRoute:
    approach: str
    critical_insight: str
    first_nontrivial_step: str
    failure_condition: str


@dataclass(frozen=True)
class MethodScout:
    routes: tuple[ScoutRoute, ...]
    valid: bool
    source: str = "model"

    @classmethod
    def fallback(cls) -> "MethodScout":
        return cls((), False, "local_fallback")

    @classmethod
    def parse(cls, response: str) -> "MethodScout":
        payload = _extract_payload(response)
        if payload is None:
            return cls.fallback()
        raw_routes = payload if isinstance(payload, list) else payload.get("routes")
        if not isinstance(raw_routes, list) or not 2 <= len(raw_routes) <= 3:
            return cls.fallback()

        routes: list[ScoutRoute] = []
        for raw in raw_routes:
            if not isinstance(raw, dict):
                return cls.fallback()
            values = tuple(
                _field(raw.get(name), limit)
                for name, limit in (
                    ("approach", 240),
                    ("critical_insight", 500),
                    ("first_nontrivial_step", 600),
                    ("failure_condition", 360),
                )
            )
            if (
                not all(values)
                or any(_ANSWER_MARKER.search(value) for value in values)
                or _GENERIC_STEP.fullmatch(values[2])
            ):
                return cls.fallback()
            route = ScoutRoute(*values)
            if any(_same_route(route, existing) for existing in routes):
                return cls.fallback()
            routes.append(route)
        return cls(tuple(routes), True)

    def prompt_context(self) -> str:
        if not self.valid:
            return ""
        lines = [
            "UNTRUSTED METHOD SCOUTS: These suggestions may all be wrong. Before spending "
            "the solve budget, independently test each proposed first step against the "
            "original statement. Choose at most one route that gives concrete progress, "
            "or discard all of them. Never force a theorem whose hypotheses fail, and do "
            "not discuss the scouting process in the answer."
        ]
        for index, route in enumerate(self.routes, start=1):
            lines.extend((
                f"Route {index} approach: {route.approach}",
                f"Route {index} possible decisive insight: {route.critical_insight}",
                f"Route {index} proposed first step: {route.first_nontrivial_step}",
                f"Route {index} rejection test: {route.failure_condition}",
            ))
        return "\n".join(lines)

    def trace_content(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "source": self.source,
            "route_count": len(self.routes),
            "routes_distinct": bool(self.valid and len(self.routes) >= 2),
            "has_concrete_first_steps": bool(
                self.valid and all(route.first_nontrivial_step for route in self.routes)
            ),
        }


def scouting_system_prompt() -> str:
    return (
        "You are a bounded mathematical method scout, not the final solver. Search for "
        "materially different ways to obtain the decisive breakthrough. Preserve every "
        "coefficient, quantifier, domain, endpoint, initial condition, and requested "
        "method. Do not solve the whole problem, state or guess the final answer, cite a "
        "remembered source, or emit FINAL/boxed text. A route is useful only if it gives "
        "a precise critical insight, one concrete first nontrivial mathematical step, and "
        "a condition that would falsify or reject the route. Return exactly one JSON "
        "object with no prose: {\"routes\":[{\"approach\":\"...\","
        "\"critical_insight\":\"...\",\"first_nontrivial_step\":\"...\","
        "\"failure_condition\":\"...\"}, {...}, {...}]}"
    )


def scouting_request(problem: str, spec) -> str:
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "complete requested result"
    return (
        f"Problem:\n{problem}\n\n"
        f"Locally classified subject: {spec.profile.primary_subject}. "
        f"Task kind: {spec.profile.task_kind}. Required result: {obligations}.\n"
        "Find exactly three short, method-diverse breakthrough candidates. Stop after "
        "recording their first concrete steps and rejection conditions; do not carry any "
        "route through to the final answer."
    )


def _extract_payload(response: str) -> dict[str, Any] | list[Any] | None:
    text = str(response or "").strip().strip("`")
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [value for value in (object_start, array_start) if value >= 0]
    start = min(starts) if starts else -1
    if start < 0:
        return None
    end = _balanced_json_end(text, start)
    if end is None or text[end:].strip().strip("`"):
        return None
    encoded = text[start:end]
    try:
        payload = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        # Models commonly place LaTeX such as ``\frac`` in JSON strings
        # without doubling the backslash. Escape only sequences that JSON
        # itself does not define; valid JSON escapes remain untouched.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', encoded)
        try:
            payload = json.loads(repaired)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return payload if isinstance(payload, (dict, list)) else None


def _balanced_json_end(text: str, start: int) -> int | None:
    opening = text[start] if 0 <= start < len(text) else ""
    if opening not in "{[":
        return None
    stack: list[str] = []
    quoted = False
    escaped = False
    for index in range(start, min(len(text), start + 20_001)):
        char = text[index]
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
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or (stack[-1], char) not in {("{", "}"), ("[", "]")}:
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


def _field(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _same_route(first: ScoutRoute, second: ScoutRoute) -> bool:
    def tokens(value: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", value.casefold()))

    left = tokens(first.approach + " " + first.critical_insight)
    right = tokens(second.approach + " " + second.critical_insight)
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= 0.8
