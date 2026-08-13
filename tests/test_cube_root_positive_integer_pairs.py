from pathlib import Path
import sys

import pytest
import sympy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


ENGLISH = (
    "Find all pairs of positive integers $(a,b)$ that satisfy "
    r"$\sqrt[3]{7a^2+ab+b^2}=a+1$."
)
CHINESE = (
    r"求所有正整数对$(a,b)$，使得$\sqrt[3]{7a^2+ab+b^2}=a+1$。"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified Diophantine route called the model: {kwargs}")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == "cube_root_positive_integer_pairs"
    ]


@pytest.mark.parametrize("problem", [ENGLISH, CHINESE])
def test_cube_root_pair_route_is_bilingual_certified_and_complete(problem):
    evidence = _matching(problem)

    assert len(evidence) == 1
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"
    assert "n^2+3n+2" in evidence[0].result
    assert "n^3+4n^2+3n-1" in evidence[0].result
    assert r"\mathbb Z_{\ge 1}" in evidence[0].result
    assert {
        "discriminant_square_condition",
        "square_factorization_descent",
        "symbolic_substitution_identity",
        "parameter_domain_exhausted",
    } <= set(evidence[0].certificate_checks)


def test_parameterization_substitutes_into_the_original_identity():
    n = sympy.symbols("n", integer=True, positive=True)
    a = n**2 + 3 * n + 2
    b = n**3 + 4 * n**2 + 3 * n - 1

    assert sympy.expand(7 * a**2 + a * b + b**2 - (a + 1) ** 3) == 0
    for value in range(1, 20):
        av, bv = int(a.subs(n, value)), int(b.subs(n, value))
        assert av > 0 and bv > 0
        assert 7 * av**2 + av * bv + bv**2 == (av + 1) ** 3


def test_cube_root_pair_route_bypasses_the_model_end_to_end():
    result = ReasoningAgent(_NoModelClient()).solve(
        ENGLISH + r" Remember to put your final answer within \boxed{}.", {}
    )

    assert "n^2+3n+2" in result["final_response"]
    assert r"\boxed{" in result["final_response"]
    assert not any(
        str(step.get("step", "")).startswith("model_call_")
        for step in result["trace"]
    )


@pytest.mark.parametrize(
    "problem",
    [
        ENGLISH.replace("positive integers", "integers"),
        ENGLISH.replace("7a^2", "8a^2"),
        ENGLISH.replace("ab", "2ab"),
        ENGLISH.replace("b^2", "2b^2"),
        ENGLISH.replace("a+1", "a+2"),
        ENGLISH.replace("Find all pairs", "Find one pair"),
        ENGLISH + " Also require $\\gcd(a,b)=1$.",
        ENGLISH + " Find only the pairs with $a<100$.",
        CHINESE.replace("所有正整数对", "所有整数对"),
        CHINESE.replace("7a^2", "6a^2"),
        CHINESE + "并求其中互素的解。",
    ],
)
def test_cube_root_pair_route_rejects_changed_contract(problem):
    assert not _matching(problem)
