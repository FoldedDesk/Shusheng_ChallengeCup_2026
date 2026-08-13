from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


PROBLEM = (
    r"Apply the two-point Gauss-Legendre quadrature rule on $[0,1]$ to "
    r"approximate $\int_0^1 x^4\,dx$. Find the resulting approximation as "
    r"an exact fraction. Remember to put your final answer within \boxed{}."
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified quadrature unexpectedly called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def test_two_point_gauss_legendre_monomial_is_exact_certified_route():
    evidence = _evidence(PROBLEM)
    matching = [item for item in evidence if item.operation == "two_point_gauss_legendre_monomial"]

    assert len(matching) == 1
    assert matching[0].scope == "whole_goal"
    assert matching[0].result == "7/36"

    result = ReasoningAgent(_NoModelClient()).solve(PROBLEM, {})
    assert result["final_response"] == r"\boxed{7/36}"
    assert next(
        step for step in result["trace"] if step["step"] == "selection"
    )["content"]["source"] == "sympy_verified"


def test_gauss_legendre_route_rejects_changed_contracts():
    variants = (
        PROBLEM.replace("two-point", "three-point"),
        PROBLEM.replace("$[0,1]$", "$[0,2]$"),
        PROBLEM.replace("exact fraction", "decimal"),
        PROBLEM.replace(r"x^4", r"x^4+1"),
        PROBLEM.replace(
            r"approximate $\int_0^1 x^4\,dx$.",
            r"approximate $\int_0^1 x^4\,dx$ and $\int_0^1 x^2\,dx$.",
        ),
        PROBLEM.replace(
            "Find the resulting approximation",
            "Find the nodes, weights, and resulting approximation",
        ),
        PROBLEM + " Also report the exact integral and signed error.",
        PROBLEM + " Then add 1 to the resulting approximation.",
        PROBLEM + " Multiply the result by 2.",
    )

    for problem in variants:
        assert not any(
            item.operation == "two_point_gauss_legendre_monomial"
            and item.scope == "whole_goal"
            for item in _evidence(problem)
        )


def test_gauss_legendre_route_rejects_composite_and_weighted_variants():
    variants = (
        PROBLEM.replace("two-point", "composite two-point"),
        PROBLEM.replace("two-point", "two-panel two-point"),
        PROBLEM.replace(
            "on $[0,1]$",
            "on each of four subintervals obtained by partitioning $[0,1]$",
        ),
        PROBLEM.replace("on $[0,1]$", "using four equal intervals on $[0,1]$"),
        PROBLEM.replace(
            "two-point Gauss-Legendre quadrature rule",
            "weighted two-point Gauss-Legendre quadrature rule with weight function $w(x)=x$",
        ),
        PROBLEM.replace(
            "two-point Gauss-Legendre quadrature rule",
            "two-point Gauss-Legendre quadrature rule with weight $w(x)=x$",
        ),
        (
            r"在 $[0,1]$ 上用复合二点 Gauss-Legendre 求积公式近似计算 "
            r"$\int_0^1 x^4\,dx$，结果写成精确分数。"
        ),
    )

    for problem in variants:
        assert not any(
            item.operation == "two_point_gauss_legendre_monomial"
            for item in SympyTool().results_for(problem)
        )
