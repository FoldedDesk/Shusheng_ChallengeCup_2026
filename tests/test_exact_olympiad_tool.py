import json
from fractions import Fraction
import math
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool


DATASET = Path(__file__).parents[1] / "sample_data" / "judge1_style_112_hard_v1.jsonl"
ROWS = {row["idx"]: row for row in map(json.loads, DATASET.read_text().splitlines())}

EXPECTED_OPERATIONS = {
    5000: "cyclic_nonadjacent_selection",
    5003: "finite_subtraction_game",
    5006: "wheel_coloring",
    5007: "grid_poset_extensions",
    5010: "hypercube_spanning_trees",
    5031: "cycle_distance_two_coloring",
    5032: "intersecting_antichain_maximum",
    5035: "punctured_domino_tilings",
    5038: "bipartite_matching_deletion_trees",
    5039: "square_subtraction_game",
    5040: "odd_fiber_functions",
    5042: "complete_intersection_maximum",
    5046: "couples_unlabeled_groups",
    5048: "bounded_generalized_pell_count",
    5051: "integer_polynomial_divisibility",
    5053: "bounded_divisor_count",
    5054: "primitive_pythagorean_count",
    5056: "inverse_totient",
    5058: "nested_modular_power_sum",
    5059: "gcd_sum",
    5060: "positive_sum_two_squares",
    5061: "factorial_quotient_valuation",
    5063: "pell_fundamental_solution",
    5064: "least_integer_with_divisor_count",
    5065: "factorable_binary_quadratic",
    5070: "root_polynomial_product",
    5071: "reciprocal_quartic_nonnegative",
    5073: "affine_recurrence_determinant",
    5078: "cevian_length",
    5082: "descartes_inner_circle",
    5088: "smith_normal_form",
    5090: "rotation_necklace_fixed_weight",
    5092: "bose_einstein_integral",
    5099: "bernoulli_likelihood_ratio",
    5108: "complete_graph_cover_time",
    5109: "brownian_exit_expectation",
}


def _directed_cylinder_problem(width: int) -> str:
    upper = width - 1
    total = 3 * width
    return (
        rf"Let \(S\) be the set of all ordered pairs \((x,y)\) with "
        rf"\(0 \leq x \leq {upper}\) and \(0 \leq y \leq 2\). "
        rf"Compute the number of permutations \((x_1,y_1)\), ..., \((x_{total},y_{total})\) "
        rf"of the elements of \(S\) such that \(y_1=2\), \(y_{{{total}}}=0\), and for all "
        rf"\(1 \leq i \leq {total - 1}\), exactly one of the following holds: "
        rf"\(x_i=x_{{i+1}}\) and \(|y_i-y_{{i+1}}|=1\), or "
        rf"\(y_i=y_{{i+1}}\) and \(x_i-x_{{i+1}}\) is \(-1\) or \({upper}\). "
        r"Remember to put your final answer within \boxed{}."
    )


def _sorted_triangle_problem(count: int) -> str:
    return (
        rf"Find the minimum value of an integer $N$ satisfying this condition. Given {count} "
        "non-degenerate triangles, each triangle has one side colored green, one side colored purple, "
        "and one side colored orange. "
        rf"Let $g_1 \ge g_2 \ge \cdots \ge g_{{{count}}}$, "
        rf"$p_1 \ge p_2 \ge \cdots \ge p_{{{count}}}$, and "
        rf"$o_1 \ge o_2 \ge \cdots \ge o_{{{count}}}$ be the separately sorted side lengths. "
        rf"The number of $1 \le a \le {count}$ such that $g_a,p_a,o_a$ do not form the sides of a "
        r"triangle is always less than or equal to $N$."
    )


@pytest.mark.parametrize("idx", EXPECTED_OPERATIONS)
def test_exact_family_result_is_certified_equivalent_and_covers_whole_goal(idx):
    row = ROWS[idx]
    spec = build_problem_spec(row["problem"])
    results = SympyTool().results_for(row["problem"])
    matching = [result for result in results if result.operation == EXPECTED_OPERATIONS[idx]]

    assert len(matching) == 1
    assert matching[0].verified
    assert matching[0].whole_answer_eligible
    assert equivalent_answers(matching[0].result, row["answer"])

    evidence = SubmissionAgent._tool_evidence(matching, spec)
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize(
    "problem, forbidden_operation",
    [
        (
            "Fifteen labeled seats are arranged in a row. How many ways can five seats be selected "
            "so that no two selected seats are adjacent?",
            "cyclic_nonadjacent_selection",
        ),
        (
            "A heap has n stones. Players remove exactly 1, 3, or 4 stones, and the player taking "
            "the last stone loses. How many losing positions have 1 <= n <= 100?",
            "finite_subtraction_game",
        ),
        (
            "A graph contains the cycle C_8 and one isolated vertex. How many proper vertex "
            "colorings use four labeled colors?",
            "wheel_coloring",
        ),
        (
            r"The vertices of Q_4 are binary strings; vertices are adjacent when they differ in "
            r"two coordinates. Find its number of spanning trees.",
            "hypercube_spanning_trees",
        ),
        (
            "A heap move removes a positive perfect cube. How many initial positions up to 500 are losing?",
            "square_subtraction_game",
        ),
        (
            "How many functions from {1,2,...,13} to {1,2,3,4,5} have odd nonempty fibers, "
            "where a zero fiber is also considered odd?",
            "odd_fiber_functions",
        ),
        (
            "Six married couples are partitioned into four labeled groups of three, with no spouses together.",
            "couples_unlabeled_groups",
        ),
        (
            "How many positive integers n <= 10000 have at most 48 positive divisors?",
            "bounded_divisor_count",
        ),
        (
            "How many Pythagorean triples, not necessarily primitive, have c <= 2026?",
            "primitive_pythagorean_count",
        ),
        (
            r"Find one positive integer n satisfying \varphi(n)=72.",
            "inverse_totient",
        ),
        (
            r"Evaluate \sum_{k=1}^{100}\gcd(k,120).",
            "gcd_sum",
        ),
        (
            r"Determine integer pairs, allowing zero, satisfying x^2+y^2=5^4\cdot13^2.",
            "positive_sum_two_squares",
        ),
        (
            r"For M=\dfrac{100!}{40!60!}, find the remainder modulo 210.",
            "factorial_quotient_valuation",
        ),
        (
            r"Find the smallest solution of the negative Pell equation x^2-109y^2=-1.",
            "pell_fundamental_solution",
        ),
        (
            "Find the greatest positive integer having exactly 2025 divisors.",
            "least_integer_with_divisor_count",
        ),
        (
            r"Determine all integer pairs satisfying x^2+4xy+2y^2=2025.",
            "factorable_binary_quadratic",
        ),
        (
            "Three tangent circles have radii 36, 9, and 4. Find the radius of a circle enclosing all three.",
            "descartes_inner_circle",
        ),
        (
            "用两种颜色给正十边形顶点着色，恰有三个红点；旋转或反射重合均视为相同。",
            "rotation_necklace_fixed_weight",
        ),
        (
            r"Evaluate \int_0^{\infty}x^2/(e^x-1)\,dx.",
            "bose_einstein_integral",
        ),
        (
            r"Bernoulli样本量为20，观察到15次成功，求似然比统计量的渐近分布。",
            "bernoulli_likelihood_ratio",
        ),
        (
            r"带漂移 Brownian 运动从 $1$ 出发，求首次离开 $(-2,3)$ 的期望。",
            "brownian_exit_expectation",
        ),
        (
            "The vertices of a regular 11-gon have four labeled colors. Adjacent vertices "
            "must receive different colors.",
            "cycle_distance_two_coloring",
        ),
        (
            "A 6 by 8 rectangular board has two marked squares. How many tilings use 1 by 3 trominoes?",
            "punctured_domino_tilings",
        ),
        (
            "A family of 6-element subsets of {1,2,...,12} has every three distinct members "
            "sharing at least three elements. Find its maximum size.",
            "complete_intersection_maximum",
        ),
        (
            r"Determine the number of ordered pairs of positive integers satisfying x^2-5y^2=-4, with y\le10^6.",
            "bounded_generalized_pell_count",
        ),
        (
            r"Find the complete set of real numbers n for which $n^2-n+2$ divides $n^4+5n+10$.",
            "integer_polynomial_divisibility",
        ),
        (
            r"Determine all real a for which x^4+ax^3+6x^2-ax+1\ge0 for every real x.",
            "reciprocal_quartic_nonnegative",
        ),
        (
            r"Let a_0=1, a_1=4 and a_{n+2}=6a_{n+1}+a_n+4. Evaluate "
            r"(a_{2026}+1)(a_{2024}+1)-(a_{2025}+1)^2.",
            "affine_recurrence_determinant",
        ),
        (
            r"Let \alpha_1,\ldots,\alpha_5 be the distinct roots of x^5-2x+3=0. "
            r"Evaluate \prod_{j=1}^5(\alpha_j^3+\alpha_j+1).",
            "root_polynomial_product",
        ),
        (
            "In triangle ABC, AB=13, AC=15 and BC=14. A point D on the extension of BC "
            "satisfies BD:DC=3:4. Find AD.",
            "cevian_length",
        ),
        (
            r"求有理矩阵 $A=\begin{pmatrix}1/2&0\\0&2\end{pmatrix}$ 的 Smith 标准形。",
            "smith_normal_form",
        ),
        (
            "A family F of subsets of {1,2,...,10} is intersecting if every three members "
            "have nonempty intersection, and is an antichain if no member contains another. "
            "Find its maximum size.",
            "intersecting_antichain_maximum",
        ),
        (
            r"From the complete bipartite graph $K_{7,8}$ delete seven edges $u_iv_i$ for "
            r"$1\le i\le7$, and also delete $u_1v_2$. Find the number of spanning trees.",
            "bipartite_matching_deletion_trees",
        ),
    ],
)
def test_exact_family_handlers_reject_changed_or_missing_rules(problem, forbidden_operation):
    operations = {result.operation for result in SympyTool().results_for(problem)}
    assert forbidden_operation not in operations


def test_descartes_reference_is_independently_certified_by_curvatures():
    curvatures = [Fraction(1, radius) for radius in (36, 9, 4)]
    pair_sum = sum(
        (curvatures[left] * curvatures[right] for left, right in ((0, 1), (0, 2), (1, 2))),
        Fraction(),
    )
    fourth_curvature = sum(curvatures, Fraction()) + 2 * Fraction(
        math.isqrt(pair_sum.numerator), math.isqrt(pair_sum.denominator)
    )

    assert 1 / fourth_curvature == Fraction(9, 7)


def test_new_exact_families_have_precise_answer_shapes_and_exhaustive_contracts():
    assert build_problem_spec(ROWS[5051]["problem"]).profile.answer_shape == "roots"
    assert build_problem_spec(ROWS[5071]["problem"]).profile.answer_shape == "interval"
    for idx in (5051, 5071):
        spec = build_problem_spec(ROWS[idx]["problem"])
        names = {requirement.name for goal in spec.goals for requirement in goal.requirements}
        assert "exhaustive_result" in names


@pytest.mark.parametrize(
    "width, expected",
    [(3, 12), (4, 12), (8, 120), (20, 20460)],
)
def test_directed_three_row_cylinder_hamilton_count_is_exact(width, expected):
    problem = _directed_cylinder_problem(width)
    results = [
        result for result in SympyTool().results_for(problem)
        if result.operation == "directed_cylinder_hamilton_paths"
    ]

    assert len(results) == 1
    assert results[0].result == str(expected)
    assert results[0].verified
    assert results[0].whole_answer_eligible
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace(r"\(0 \leq y \leq 2\)", r"\(0 \leq y \leq 3\)"),
        lambda text: text.replace(r"\(y_{60}=0\)", r"\(y_{60}=1\)"),
        lambda text: text.replace(r"\(1 \leq i \leq 59\)", r"\(1 \leq i \leq 58\)"),
        lambda text: text.replace(r"\(-1\) or \(19\)", r"\(-1\) or \(18\)"),
        lambda text: text.replace(r"x_i-x_{i+1}", r"x_{i+1}-x_i"),
    ],
)
def test_directed_three_row_cylinder_handler_rejects_changed_contract(mutator):
    operations = {
        result.operation
        for result in SympyTool().results_for(mutator(_directed_cylinder_problem(20)))
    }

    assert "directed_cylinder_hamilton_paths" not in operations


def test_directed_three_row_cylinder_bypasses_the_model_only_after_full_match():
    class NoCallClient:
        def chat_result(self, **kwargs):
            raise AssertionError(f"unexpected model call: {kwargs}")

    result = SubmissionAgent(NoCallClient()).solve(_directed_cylinder_problem(20), {})

    assert result["final_response"] == r"\boxed{20460}"
    call_plan = next(item for item in result["trace"] if item["step"] == "call_plan")
    assert call_plan["content"]["route"] == "certified_tool"


@pytest.mark.parametrize("count", [1, 2, 10, 2025])
def test_sorted_triangle_failure_bound_is_n_minus_one(count):
    problem = _sorted_triangle_problem(count)
    results = [
        result for result in SympyTool().results_for(problem)
        if result.operation == "sorted_triangle_failure_bound"
    ]

    assert len(results) == 1
    assert results[0].result == str(count - 1)
    assert results[0].verified
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("non-degenerate triangles", "triples of positive numbers"),
        lambda text: text.replace(r"g_1 \ge g_2", r"g_1 \le g_2"),
        lambda text: text.replace("do not form the sides", "form the sides"),
        lambda text: text.replace("always less than or equal to", "is sometimes less than"),
        lambda text: text.replace("minimum value", "maximum value"),
    ],
)
def test_sorted_triangle_failure_handler_rejects_changed_contract(mutator):
    operations = {
        result.operation
        for result in SympyTool().results_for(mutator(_sorted_triangle_problem(20)))
    }

    assert "sorted_triangle_failure_bound" not in operations
