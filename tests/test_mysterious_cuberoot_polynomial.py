import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


PROBLEM = r"""We call a real number $x$ 'mysterious' if it is a solution to
$A(x) = \frac{1}{\sqrt[3]{3}}x$ for some polynomial $A(x)$ with rational coefficients.
Find all polynomials $A(x)$ with rational coefficients of lowest possible degree such that
$\sqrt[3]{3} + \sqrt[3]{9}$ is mysterious.

Remember to put your final answer within \boxed{}."""


def _route(problem: str):
    return [
        result
        for result in SympyTool().results_for(problem)
        if result.operation == "mysterious_cuberoot_polynomial"
    ]


def test_actual_family_derives_and_certifies_the_unique_quadratic():
    results = _route(PROBLEM)

    assert len(results) == 1
    result = results[0]
    assert result.result == r"A(x)=\frac{1}{2}(x^2-x-4)"
    assert result.verified
    assert result.certificate.method == "cubic_field_basis_identity_and_minimality"
    assert {
        "exact_polynomial_identity",
        "degree_two_minimality",
        "unique_reduced_representative",
    } <= set(result.certificate.checks)

    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(PROBLEM))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


def test_harmless_wording_whitespace_and_equivalent_fraction_form_match():
    variant = r"""A real number x is called mysterious if it is a root of
      $A(x)=\frac{x}{\sqrt[3]{3}}$ for a polynomial A(x) with rational coefficients.
      Determine all polynomials A(x) with rational coefficients of minimum degree such that
      $\sqrt[3]{3}+\sqrt[3]{9}$ is mysterious."""

    assert _route(variant)[0].result == r"A(x)=\frac{1}{2}(x^2-x-4)"


def test_consistently_changed_noncube_radicand_is_recomputed():
    changed = PROBLEM.replace(r"\sqrt[3]{3}", r"\sqrt[3]{5}").replace(
        r"\sqrt[3]{9}", r"\sqrt[3]{25}"
    )

    assert _route(changed)[0].result == r"A(x)=\frac{1}{4}(x^2-x-6)"


def test_supported_nonzero_integer_scale_is_recomputed_not_reused():
    changed = PROBLEM.replace(r"\frac{1}{\sqrt[3]{3}}x", r"\frac{2}{\sqrt[3]{3}}x")

    assert _route(changed)[0].result == "A(x)=x^2-x-4"


@pytest.mark.parametrize(
    "mutator",
    [
        # A single changed radicand no longer defines the same pure cubic field.
        lambda text: text.replace(r"\sqrt[3]{3}", r"\sqrt[3]{5}", 1),
        lambda text: text.replace(r"\frac{1}{\sqrt[3]{3}}x", r"\frac{0}{\sqrt[3]{3}}x"),
        lambda text: text.replace(r"\frac{1}{\sqrt[3]{3}}x", r"\frac{1}{\sqrt[3]{3}}x+1"),
        lambda text: text.replace(r"\frac{1}{\sqrt[3]{3}}x", r"\frac{1}{\sqrt[3]{3}}x^2"),
        lambda text: text.replace(r"\sqrt[3]{3} + \sqrt[3]{9}", r"\sqrt[3]{3} - \sqrt[3]{9}"),
        lambda text: text.replace(r"\sqrt[3]{3} + \sqrt[3]{9}", r"\sqrt[3]{3} + 2\sqrt[3]{9}"),
        lambda text: text.replace("some polynomial $A(x)$ with rational coefficients", "some polynomial $A(x)$ with real coefficients"),
        lambda text: text.replace("polynomials $A(x)$ with rational coefficients", "polynomials $A(x)$ with integer coefficients", 1),
        lambda text: text.replace("of lowest possible degree", "of degree at most two"),
        lambda text: text.replace("Find all polynomials", "Find a polynomial"),
        lambda text: text.replace("is mysterious.", "is mysterious and $A(0)=0$."),
        lambda text: text.replace(r"\sqrt[3]{3}", r"\sqrt[3]{8}").replace(r"\sqrt[3]{9}", r"\sqrt[3]{64}"),
    ],
)
def test_semantic_contract_changes_do_not_receive_the_original_whole_answer(mutator):
    assert _route(mutator(PROBLEM)) == []


class NoCallClient:
    def chat_result(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")

    def chat(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")


def test_fully_matched_problem_bypasses_model_with_boxed_polynomial():
    result = SubmissionAgent(NoCallClient()).solve(PROBLEM, {})

    assert result["final_response"] == r"\boxed{A(x)=\frac{1}{2}(x^2-x-4)}"
    plan = next(item for item in result["trace"] if item["step"] == "call_plan")
    assert plan["content"]["route"] == "certified_tool"


def test_removed_all_quantifier_is_not_integration_eligible():
    changed = PROBLEM.replace("Find all polynomials", "Find a polynomial")
    operations = {result.operation for result in SympyTool().results_for(changed)}

    assert "mysterious_cuberoot_polynomial" not in operations
