"""Contracts and reproducible certificates for general local math tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Optional


GENERIC_OPERATIONS = frozenset({
    "calculate",
    "solve_equation",
    "derivative",
    "definite_integral",
    "limit",
    "matrix_determinant",
    "matrix_rank",
    "matrix_inverse",
    "matrix_eigenvalues",
    "graph_spanning_trees",
    "scaled_cauchy_kernel_limit",
    "surface_gaussian_curvature",
    "gauss_legendre_error",
    "uniform_maximum_spacing_expectation",
    "birth_death_hitting_probability",
    "contour_residue_integral",
    "linear_ode_ivp",
    "intercept_gls",
    "poisson_disk_harmonic_measure",
    "abelian_presentation_snf",
    "candidate_symbolic_check",
    "candidate_range_check",
})


def problem_fingerprint(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", str(problem or "")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


@dataclass(frozen=True)
class ToolContract:
    operation: str
    result_kind: str
    certificate_method: str
    whole_answer_capable: bool = False
    written_support_capable: bool = False
    max_goals: int = 1
    allowed_requirements: tuple[str, ...] = ()
    required_requirements: tuple[str, ...] = ()
    allowed_task_kinds: tuple[str, ...] = ("calculation", "fill_blank")
    allowed_answer_shapes: tuple[str, ...] = ()
    required_problem_facts: tuple[str, ...] = ()
    forbidden_problem_facts: tuple[str, ...] = ("unparsed_suffix", "extra_obligation")

    @property
    def certified(self) -> bool:
        return self.operation in GENERIC_OPERATIONS and bool(self.certificate_method)

    def covers(
        self,
        goal_count: int,
        requirements: Iterable[str] = (),
        *,
        task_kind: str = "",
        answer_shape: str = "",
        problem_facts: Optional[Iterable[str]] = None,
    ) -> bool:
        requested = set(requirements)
        facts = set(problem_facts or ())
        return bool(
            self.certified
            and self.whole_answer_capable
            and goal_count == 1
            and goal_count <= self.max_goals
            and set(self.required_requirements) <= requested
            and requested <= set(self.allowed_requirements)
            and (not task_kind or task_kind in self.allowed_task_kinds)
            and (not answer_shape or not self.allowed_answer_shapes or answer_shape in self.allowed_answer_shapes)
            and set(self.required_problem_facts) <= facts
            and not set(self.forbidden_problem_facts).intersection(facts)
        )


@dataclass(frozen=True)
class ToolCertificate:
    passed: bool
    method: str
    checks: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    source_fingerprint: str = ""

    def trace_content(self) -> dict:
        return {
            "passed": self.passed,
            "method": self.method,
            "checks": list(self.checks),
            "issues": list(self.issues),
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class ToolResult:
    result: str
    operation: str
    label: str
    contract: Optional[ToolContract]
    certificate: ToolCertificate
    support: str = ""

    @property
    def verified(self) -> bool:
        return bool(self.certificate.passed and self.contract and self.contract.certified)

    @property
    def whole_answer_eligible(self) -> bool:
        return bool(self.verified and self.contract and self.contract.whole_answer_capable)

    @property
    def goal_result_eligible(self) -> bool:
        """Whether the result certifies the requested conclusion, excluding prose obligations."""
        return self.whole_answer_eligible

    @property
    def supported_submission_eligible(self) -> bool:
        return bool(
            self.goal_result_eligible
            and self.contract
            and self.contract.written_support_capable
            and self.support
        )

    def to_hint(self) -> str:
        return f"{self.label}: {self.support or self.result}"

    def trace_content(self) -> dict:
        return {
            "operation": self.operation,
            "result_kind": self.contract.result_kind if self.contract else "unknown",
            "whole_answer_eligible": self.whole_answer_eligible,
            "goal_result_eligible": self.goal_result_eligible,
            "supported_submission_eligible": self.supported_submission_eligible,
            "certificate": self.certificate.trace_content(),
            "support": self.support,
        }


def make_tool_result(
    *,
    problem: str,
    operation: str,
    result: str,
    result_kind: str,
    method: str,
    whole: bool,
    written_support: bool = False,
    checks: Iterable[str],
    issues: Iterable[str] = (),
    support: str = "",
    answer_shapes: tuple[str, ...] = (),
    requirements: tuple[str, ...] = ("result_present",),
) -> ToolResult:
    clean_result = str(result or "").strip()
    check_tuple = tuple(dict.fromkeys(str(item) for item in checks if str(item)))
    issue_tuple = tuple(dict.fromkeys(str(item) for item in issues if str(item)))
    contract = ToolContract(
        operation=operation,
        result_kind=result_kind,
        certificate_method=method,
        whole_answer_capable=whole,
        written_support_capable=written_support,
        allowed_requirements=requirements,
        allowed_answer_shapes=answer_shapes,
    )
    passed = bool(
        operation in GENERIC_OPERATIONS
        and clean_result
        and check_tuple
        and not issue_tuple
    )
    return ToolResult(
        result=clean_result,
        operation=operation,
        label=f"local.{operation}",
        contract=contract,
        certificate=ToolCertificate(
            passed=passed,
            method=method,
            checks=check_tuple,
            issues=issue_tuple,
            source_fingerprint=problem_fingerprint(problem),
        ),
        support=str(support or "").strip(),
    )


_LEGACY_PREFIXES = {
    "SymPy 计算": ("calculate", "scalar", "sympy_exact", ("number", "expression")),
    "SymPy 方程解": ("solve_equation", "solution_set", "sympy_substitution", ("roots",)),
    "SymPy 导数": ("derivative", "expression", "sympy_differentiate", ("expression",)),
    "SymPy 定积分": ("definite_integral", "scalar", "sympy_exact", ("number", "expression")),
    "SymPy 极限": ("limit", "expression", "sympy_limit", ("number", "expression")),
}


def result_from_legacy_hint(
    hint: str,
    *,
    trusted_source: bool = False,
    extra_checks: Iterable[str] = (),
    source_problem: str = "",
) -> Optional[ToolResult]:
    """Compatibility parser limited to the generic operations above."""
    if not trusted_source:
        return None
    label, separator, value = str(hint or "").partition(":")
    spec = _LEGACY_PREFIXES.get(label.strip())
    if not separator or spec is None or not value.strip():
        return None
    operation, result_kind, method, shapes = spec
    return make_tool_result(
        problem=source_problem,
        operation=operation,
        result=value.strip(),
        result_kind=result_kind,
        method=method,
        whole=False,
        checks=("trusted_generic_producer", *tuple(extra_checks)),
        answer_shapes=shapes,
    )
