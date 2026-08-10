from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from tools.tool_contract import result_from_legacy_hint


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def _operations(problem: str) -> set[str]:
    return {result.operation for result in SympyTool().results_for(problem)}


def test_result_transformations_and_unparsed_filters_cannot_be_whole_answers():
    cases = (
        "Find the derivative of f(x)=x^3, then add 1.",
        r"Evaluate twice $\int_0^1 x^2\,dx$.",
        r"Find one plus $\lim_{x\to0}\sin(x)/x$.",
        "Solve x^2-1=0, then square every root.",
        (
            "The vertices of a regular $11$-gon are labeled cyclically by $0,1,...,10$, "
            "and four labeled colors are available. In how many vertex colorings do any two "
            "vertices at cyclic distance $1$ or $2$ receive different colors? Vertex 0 must "
            "use the first color."
        ),
        (
            "Determine the number of ordered pairs of positive integers $(x,y)$ satisfying "
            "$x^2-5y^2=-4$ and $x <= 10^6$. Only count even x."
        ),
        (
            r"Determine all real numbers (a) for which "
            r"(x^4+ax^3+6x^2+ax+1\ge0) holds for every real (x). Report only positive parameters."
        ),
    )
    for problem in cases:
        evidence = _evidence(problem)
        assert SubmissionAgent._whole_tool_answer(evidence) == ""
        assert all(item.scope == "subexpression" for item in evidence)


def test_conflicting_whole_goal_tools_are_downgraded_together():
    spec = build_problem_spec("Calculate 1+1.")
    results = [
        result_from_legacy_hint("SymPy 计算: 45", trusted_source=True),
        result_from_legacy_hint("SymPy 计算: 32", trusted_source=True),
    ]
    evidence = SubmissionAgent._tool_evidence(
        [result for result in results if result is not None], spec
    )

    assert len(evidence) >= 2
    assert SubmissionAgent._whole_tool_answer(evidence) == ""
    assert all(item.scope == "subexpression" for item in evidence)
    assert all(
        "conflicting_whole_tool_results" in item.certificate_issues
        for item in evidence
    )


def test_handlers_reject_unparsed_expression_parts_and_wrong_requested_quantity():
    cases = (
        (
            r"Let (M=\dfrac{10!}{2\cdot5!5!}). Find the greatest integer (k) "
            r"such that (2^k\mid M).",
            "factorial_quotient_valuation",
        ),
        (
            r"Find the least residue of (17^{3^{10}}+29^{5^{8}}+1) modulo (10^6).",
            "nested_modular_power_sum",
        ),
        (
            r"Find the maximum value of (x^2+2xy+4yz+4zx) over all real triples "
            r"with (x^2+y^2+z^2=1).",
            "quadratic_form_maximum",
        ),
        (
            "Three circles of radii 36, 9, and 4 are pairwise externally tangent. A fourth "
            "circle lies in the bounded gap and is externally tangent to all three. Find the "
            "curvature of the fourth circle.",
            "descartes_inner_circle",
        ),
        (
            r"标准 Brownian 运动从 $1$ 出发，记 $\tau$ 为首次离开区间 $(-2,3)$ 的时刻。"
            r"求 $2\mathbb E[\tau]$。",
            "brownian_exit_expectation",
        ),
    )
    for problem, forbidden in cases:
        assert forbidden not in _operations(problem)
