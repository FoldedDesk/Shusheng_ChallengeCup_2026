"""Bounded, untrusted method search for one unseen mathematics problem.

The graph is planning data only.  It never becomes an answer candidate and it
never certifies that a theorem or lemma applies.  The original problem remains
the authoritative source for every coefficient, domain, and quantifier.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any

from reasoning.solve_plan import SUBJECT_CODES, SolvePlan


_OBLIGATION_KINDS = {
    "definition",
    "calculation",
    "condition_check",
    "lemma",
    "theorem_application",
    "proof",
    "construction",
}
_DIFFICULTIES = {"easy", "normal", "hard"}
_VERIFICATION_MODES = {
    "symbolic",
    "matrix",
    "finite",
    "residual",
    "boundary",
    "counterexample",
    "theorem_conditions",
    "none",
}
_ANSWER_MARKERS = re.compile(
    r"\\boxed\s*\{|\bFINAL(?:\s+ANSWER)?\s*[:：]|"
    r"(?:最终)?答案\s*[:：]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObligationNode:
    id: str
    kind: str
    claim: str
    depends_on: tuple[str, ...]
    difficulty: str
    verification: str


@dataclass(frozen=True)
class MethodRoute:
    id: str
    method: str
    key_lemma: str
    checkpoint: str
    obstacle: str


@dataclass(frozen=True)
class MathematicalObligationGraph:
    """A compact route portfolio for the hardest obligation."""

    subject: str
    objects: tuple[str, ...]
    assumptions: tuple[str, ...]
    obligations: tuple[ObligationNode, ...]
    hardest_id: str
    routes: tuple[MethodRoute, ...]
    valid: bool
    source: str = "model"

    @classmethod
    def fallback(cls, spec) -> "MathematicalObligationGraph":
        target = str(getattr(spec.semantics, "target", "") or spec.problem_text)
        node = ObligationNode(
            id="O1",
            kind=_fallback_kind(spec),
            claim=target[:600],
            depends_on=(),
            difficulty=(
                "hard" if spec.profile.difficulty == "hard" or spec.risk_score >= 4
                else "normal"
            ),
            verification=_fallback_verification(spec),
        )
        primary = MethodRoute(
            "A",
            str(spec.primary_method or "direct derivation")[:500],
            "derive the decisive intermediate claim from the original statement",
            str(spec.alternative_method or "independent boundary or hypothesis check")[:500],
            "; ".join(spec.risk_flags)[:500] or "unverified theorem hypotheses",
        )
        secondary_method = str(
            spec.alternative_method or "counterexample and theorem-condition audit"
        )[:500]
        secondary = MethodRoute(
            "B",
            secondary_method,
            "rederive the hardest implication without using route A",
            "attempt a boundary case, substitution, or counterexample",
            "shared assumptions with the direct route",
        )
        return cls(
            subject=getattr(spec.profile, "primary_subject", spec.profile.subject),
            objects=(),
            assumptions=tuple(spec.constraints[:6]),
            obligations=(node,),
            hardest_id=node.id,
            routes=(primary, secondary),
            valid=False,
            source="local_fallback",
        )

    @classmethod
    def parse(cls, response: str, spec) -> "MathematicalObligationGraph":
        fallback = cls.fallback(spec)
        payload = _extract_payload(response)
        if payload is None:
            return fallback

        objects = _string_list(payload.get("objects"), maximum=6, limit=220)
        assumptions = _string_list(payload.get("assumptions"), maximum=8, limit=320)
        raw_obligations = payload.get("obligations")
        if not isinstance(raw_obligations, list) or not 1 <= len(raw_obligations) <= 6:
            return fallback

        obligations: list[ObligationNode] = []
        ids: set[str] = set()
        for raw in raw_obligations:
            if not isinstance(raw, dict):
                return fallback
            node_id = _field(raw.get("id"), 12)
            kind = _field(raw.get("kind"), 40).casefold()
            claim = _field(raw.get("claim"), 600)
            difficulty = _field(raw.get("difficulty"), 20).casefold()
            verification = _field(raw.get("verification"), 40).casefold()
            dependencies = _string_list(raw.get("depends_on"), maximum=5, limit=12)
            if (
                not re.fullmatch(r"O[1-9][0-9]?", node_id)
                or node_id in ids
                or kind not in _OBLIGATION_KINDS
                or not claim
                or difficulty not in _DIFFICULTIES
                or verification not in _VERIFICATION_MODES
                or _ANSWER_MARKERS.search(claim)
            ):
                return fallback
            ids.add(node_id)
            obligations.append(ObligationNode(
                node_id,
                kind,
                claim,
                dependencies,
                difficulty,
                verification,
            ))
        if any(dependency not in ids for node in obligations for dependency in node.depends_on):
            return fallback

        hardest_id = _field(payload.get("hardest"), 12)
        if hardest_id not in ids:
            return fallback
        raw_routes = payload.get("routes")
        if not isinstance(raw_routes, list) or not 2 <= len(raw_routes) <= 3:
            return fallback
        routes: list[MethodRoute] = []
        for raw in raw_routes:
            if not isinstance(raw, dict):
                continue
            route = MethodRoute(
                id=_field(raw.get("id"), 8),
                method=_field(raw.get("method"), 500),
                key_lemma=_field(raw.get("key_lemma"), 600),
                checkpoint=_field(raw.get("checkpoint"), 500),
                obstacle=_field(raw.get("obstacle"), 400),
            )
            if (
                route.id
                and route.method
                and route.key_lemma
                and route.checkpoint
                and not any(
                    _ANSWER_MARKERS.search(value)
                    for value in (
                        route.method,
                        route.key_lemma,
                        route.checkpoint,
                        route.obstacle,
                    )
                )
                and not any(_same_method(route, existing) for existing in routes)
            ):
                routes.append(route)
            if len(routes) == 2:
                break
        if len(routes) != 2:
            return fallback

        raw_subject = _field(payload.get("subject"), 80)
        subject = SUBJECT_CODES.get(raw_subject.casefold(), raw_subject)
        if subject not in set(SUBJECT_CODES.values()):
            subject = fallback.subject
        return cls(
            subject=subject,
            objects=objects,
            assumptions=assumptions,
            obligations=tuple(obligations),
            hardest_id=hardest_id,
            routes=tuple(routes),
            valid=True,
        )

    def route_plan(self, spec, index: int) -> SolvePlan:
        route = self.routes[min(max(index, 0), len(self.routes) - 1)]
        hardest = next(
            (node for node in self.obligations if node.id == self.hardest_id),
            self.obligations[0],
        )
        local_target = str(getattr(spec.semantics, "target", "") or spec.problem_text)
        output_obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        dependencies = ", ".join(hardest.depends_on) or "none"
        assigned_method = (
            f"Hardest obligation {hardest.id} ({hardest.kind}, dependencies: {dependencies}): "
            f"{hardest.claim}. Assigned method: {route.method}. "
            f"Candidate key lemma: {route.key_lemma}."
        )
        return SolvePlan(
            subject=self.subject,
            target=local_target,
            obligations=output_obligations,
            method=assigned_method[:1500],
            check=route.checkpoint[:900],
            risks=(route.obstacle or "; ".join(spec.risk_flags))[:700],
            valid=self.valid,
            source=f"mog_route_{route.id}",
        )

    def trace_content(self) -> dict:
        kinds = Counter(node.kind for node in self.obligations)
        difficulties = Counter(node.difficulty for node in self.obligations)
        verifications = Counter(node.verification for node in self.obligations)
        hardest = next(
            (node for node in self.obligations if node.id == self.hardest_id),
            self.obligations[0],
        )
        return {
            "source": self.source,
            "valid": self.valid,
            "object_count": len(self.objects),
            "assumption_count": len(self.assumptions),
            "obligation_count": len(self.obligations),
            "dependency_count": sum(len(node.depends_on) for node in self.obligations),
            "obligation_kinds": dict(kinds),
            "difficulty_counts": dict(difficulties),
            "verification_modes": dict(verifications),
            "hardest_kind": hardest.kind,
            "hardest_difficulty": hardest.difficulty,
            "route_count": len(self.routes),
            "routes_distinct": (
                len(self.routes) == 2 and not _same_method(self.routes[0], self.routes[1])
            ),
        }


def planning_system_prompt(spec) -> str:
    families = " | ".join(_method_families(spec))
    return (
        "You are a mathematical method-search planner, not an answer solver. Build a small "
        "dependency graph for the proof or computation and propose exactly two genuinely "
        "different routes to its hardest obligation. Do not calculate or state the final "
        "answer, do not emit FINAL or boxed text, and do not identify a contest source. "
        "Preserve coefficients, domains, quantifiers, initial/boundary conditions, and "
        "requested methods exactly. A route is useful only when it names a concrete key "
        "lemma and a falsifiable checkpoint. Suggested method families for this subject: "
        f"{families}. Return exactly one MOG_JSON object and no prose. Schema:\n"
        "MOG_JSON: {\"subject\":\"one subject code\",\"objects\":[\"...\"],"
        "\"assumptions\":[\"...\"],\"obligations\":[{\"id\":\"O1\","
        "\"kind\":\"definition|calculation|condition_check|lemma|theorem_application|proof|construction\","
        "\"claim\":\"...\",\"depends_on\":[],\"difficulty\":\"easy|normal|hard\","
        "\"verification\":\"symbolic|matrix|finite|residual|boundary|counterexample|theorem_conditions|none\"}],"
        "\"hardest\":\"O1\",\"routes\":[{\"id\":\"A\",\"method\":\"...\","
        "\"key_lemma\":\"...\",\"checkpoint\":\"...\",\"obstacle\":\"...\"},"
        "{\"id\":\"B\",\"method\":\"...\",\"key_lemma\":\"...\","
        "\"checkpoint\":\"...\",\"obstacle\":\"...\"}]}"
    )


def planning_request(problem: str, spec) -> str:
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "complete requested result"
    return (
        f"Problem:\n{problem}\n\n"
        f"Locally classified subject: {getattr(spec.profile, 'primary_subject', spec.profile.subject)}.\n"
        f"Task kind: {spec.profile.task_kind}. Required output content: {obligations}.\n"
        "Plan only. Decompose logical dependencies, identify the hardest mathematical "
        "obligation, and produce two method-diverse routes without giving the result."
    )


def _extract_payload(response: str) -> dict[str, Any] | None:
    text = str(response or "").strip()
    marker = re.search(r"(?i)\bMOG_JSON\s*:\s*", text)
    start = text.find("{", marker.end() if marker else 0)
    if start < 0:
        return None
    end = _balanced_object_end(text, start)
    if end is None or text[end:].strip().strip("`"):
        return None
    try:
        payload = json.loads(text[start:end])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _balanced_object_end(value: str, start: int) -> int | None:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, min(len(value), start + 20_001)):
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


def _field(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _string_list(value: Any, *, maximum: int, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        return ()
    items = tuple(_field(item, limit) for item in value)
    return tuple(item for item in items if item)


def _same_method(left: MethodRoute, right: MethodRoute) -> bool:
    def tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{2,}", value.casefold()))

    left_tokens = tokens(left.method)
    right_tokens = tokens(right.method)
    if not left_tokens or not right_tokens:
        return left.method.casefold() == right.method.casefold()
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return overlap >= 0.8


def _fallback_kind(spec) -> str:
    task = str(spec.profile.task_kind)
    return task if task in _OBLIGATION_KINDS else (
        "calculation" if spec.answer_contract.mode == "answer_only" else "proof"
    )


def _fallback_verification(spec) -> str:
    shape = str(spec.profile.answer_shape)
    if shape in {"number", "expression", "roots", "matrix"}:
        return "symbolic"
    if shape in {"count", "probability"}:
        return "finite"
    return "theorem_conditions"


def _method_families(spec) -> tuple[str, ...]:
    subject = getattr(spec.profile, "primary_subject", spec.profile.subject)
    families = {
        "离散数学": (
            "direct or contradiction", "induction or minimal counterexample",
            "double counting or inclusion-exclusion", "invariant or extremal principle",
            "recurrence or generating function", "explicit construction",
        ),
        "抽象代数": (
            "definition chase", "kernel/image and isomorphism theorem",
            "order/coset/quotient", "explicit homomorphism or counterexample",
            "universal property",
        ),
        "测度积分": (
            "definition and simple functions", "MCT or Fatou", "DCT",
            "Tonelli or Fubini", "truncation and almost-everywhere argument",
        ),
        "概率论": (
            "conditioning", "indicator variables and linearity", "generating transform",
            "coupling", "martingale or stopping", "direct density integration",
        ),
        "数值分析": (
            "Taylor/error equation", "contraction", "stability and consistency",
            "interpolation or quadrature remainder", "residual and boundary check",
        ),
        "复分析": (
            "Cauchy integral or residues", "argument principle", "maximum principle",
            "power/Laurent series", "analytic continuation or normal families",
        ),
        "常微分方程": (
            "separation or integrating factor", "phase portrait", "Lyapunov function",
            "variation of constants", "existence-uniqueness and comparison",
        ),
        "偏微分方程": (
            "energy estimate", "characteristics", "separation/Fourier expansion",
            "maximum principle", "weak formulation and test functions",
        ),
        "微分几何": (
            "coordinate computation", "moving frame", "Gauss-Codazzi",
            "geodesic/variational argument", "curvature invariant",
        ),
    }.get(subject)
    return families or (
        "definition/direct derivation", "contradiction or counterexample",
        "structural theorem with hypothesis audit", "construction or invariant",
        "coordinate or symbolic computation",
    )
