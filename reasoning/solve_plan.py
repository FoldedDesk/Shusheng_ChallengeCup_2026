"""Bounded, untrusted route plans for unseen mathematics problems."""

from __future__ import annotations

from dataclasses import dataclass
import re


SUBJECT_CODES = {
    "discrete": "离散数学",
    "numerical": "数值分析",
    "measure": "测度积分",
    "differential_geometry": "微分几何",
    "probability": "概率论",
    "abstract_algebra": "抽象代数",
    "stochastic_process": "随机过程",
    "complex_analysis": "复分析",
    "ode": "常微分方程",
    "statistical_inference": "统计推断",
    "functional_analysis": "泛函分析",
    "linear_regression": "线性回归",
    "pde": "偏微分方程",
    "advanced_linear_algebra": "高等代数",
    "operations_research": "运筹学",
    "analysis": "数学分析",
    "topology": "拓扑学",
    "advanced": "非基础及进阶课程",
    "number_theory": "数论",
    "euclidean_geometry": "欧氏几何",
    "general": "进阶数学",
}

_SUBJECT_ALIASES = {
    **SUBJECT_CODES,
    **{value.casefold(): value for value in SUBJECT_CODES.values()},
    "combinatorics": "离散数学",
    "graph_theory": "离散数学",
    "linear_algebra": "高等代数",
    "statistics": "统计推断",
    "geometry": "欧氏几何",
    "real_analysis": "数学分析",
    "algebra": "抽象代数",
}


@dataclass(frozen=True)
class SolvePlan:
    """A route hint, never an answer or a correctness certificate."""

    subject: str
    target: str
    obligations: str
    method: str
    check: str
    risks: str
    valid: bool
    source: str = "model"

    @classmethod
    def fallback(cls, spec) -> "SolvePlan":
        subject = getattr(spec.profile, "primary_subject", spec.profile.subject)
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        return cls(
            subject=subject,
            target=getattr(spec.semantics, "target", "") or spec.problem_text,
            obligations=obligations,
            method=spec.primary_method,
            check=spec.alternative_method,
            risks="; ".join(spec.risk_flags),
            valid=False,
            source="local_fallback",
        )

    @classmethod
    def parse(cls, response: str, spec) -> "SolvePlan":
        fallback = cls.fallback(spec)
        text = str(response or "").strip()
        if not text:
            return fallback
        fields: dict[str, str] = {}
        aliases = {
            "subject": "subject",
            "target": "target",
            "obligations": "obligations",
            "method": "method",
            "check": "check",
            "risks": "risks",
        }
        for line in text.splitlines():
            match = re.match(r"^\s*([A-Za-z_]+)\s*[:：]\s*(.*?)\s*$", line)
            if not match:
                continue
            key = aliases.get(match.group(1).casefold())
            value = _clean_field(match.group(2))
            if key and value and key not in fields:
                fields[key] = value
        raw_subject = fields.get("subject", "").casefold().strip("`'\"[]()")
        raw_subject = re.split(r"[/,(，;；]+", raw_subject, maxsplit=1)[0].strip()
        raw_subject = raw_subject.replace(" ", "_")
        subject = _SUBJECT_ALIASES.get(raw_subject, "")
        required = ("target", "check")
        valid = bool(subject and all(fields.get(item) for item in required))
        if not valid:
            return fallback
        return cls(
            subject=subject,
            target=fields["target"],
            obligations=fields.get("obligations", fallback.obligations),
            # A model-generated route can anchor the solver on a plausible but
            # wrong theorem.  Planning is used only to target later checks.
            method=fallback.method,
            check=fields["check"],
            risks=fields.get("risks", "none"),
            valid=True,
        )

    def effective_subject(self, spec) -> str:
        """Prefer explicit local terminology; use the planner to resolve ambiguity."""
        local = getattr(spec.profile, "primary_subject", spec.profile.subject)
        confidence = getattr(spec.profile, "subject_confidence", "low")
        if confidence == "high" and local not in {"进阶数学", "非基础及进阶课程"}:
            return local
        return self.subject if self.valid else local

    def prompt_context(self, spec) -> str:
        subject = self.effective_subject(spec)
        # The planner is untrusted.  Preserve the locally copied target and
        # obligations so a paraphrase cannot alter coefficients, domains, or
        # quantifiers in the statement seen by the solver.
        target = getattr(spec.semantics, "target", "") or spec.problem_text
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or self.obligations
        return "\n".join((
            f"Planned subject: {subject}",
            f"Exact target: {target}",
            f"Output obligations: {obligations or 'complete requested result'}",
            f"Assigned route: {self.method}",
            f"Decisive verification: {self.check}",
            f"Known risks: {self.risks or 'none'}",
        ))

    def independent_context(self, spec) -> str:
        """Expose the target and falsifier without anchoring on the first route."""
        return "\n".join((
            f"Subject: {self.effective_subject(spec)}",
            f"Exact target: {self.target}",
            f"Output obligations: {self.obligations or 'complete requested result'}",
            f"Assigned independent method: {self.method}",
            f"Preferred independent checkpoint: {self.check}",
            f"Risks to attack: {self.risks or 'none'}",
            "Do not reuse the primary route merely because it was planned; derive the result independently.",
        ))

    def trace_content(self, spec) -> dict:
        return {
            "source": self.source,
            "valid": self.valid,
            "subject": self.effective_subject(spec),
            "local_subject": getattr(spec.profile, "primary_subject", spec.profile.subject),
            "target": self.target[:500],
            "method": self.method[:500],
            "check": self.check[:500],
            "risks": self.risks[:300],
        }


def planning_system_prompt() -> str:
    codes = "|".join(SUBJECT_CODES)
    return (
        "You are a route planner for one unseen mathematics problem. Do not solve the "
        "problem and do not guess its final answer. Identify one target, one best route, "
        "and one falsifiable verification. Preserve all coefficients, signs, domains, "
        "quantifiers, initial or boundary data, and requested methods. Output exactly five "
        "single-line fields and no prose before or after them:\n"
        f"SUBJECT: one of {codes}\n"
        "TARGET: the exact requested conclusion\n"
        "OBLIGATIONS: what must appear in the submitted answer\n"
        "CHECK: one independent substitution, invariant, normalization, boundary case, "
        "small enumeration, or counterexample test\n"
        "RISKS: the most likely sign, quantifier, branch, normalization, or hypothesis error"
    )


def _clean_field(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?i)\b(?:FINAL(?: ANSWER)?|最终答案)\s*[:：].*$", "", text).strip()
    return text[:900]
