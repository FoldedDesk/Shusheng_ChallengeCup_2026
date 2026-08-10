import json
from pathlib import Path

import pytest

from tools.sympy_tool import SympyTool


DATASET = Path(__file__).parents[1] / "sample_data" / "judge1_style_strategy_probe_12.jsonl"

EXPECTED_OPERATIONS = {
    5001: "complete_multipartite_spanning_trees",
    5002: "quadratic_congruence_count",
    5005: "digit_permutation_divisibility",
    5008: "adjacent_surjection_count",
    5011: "multiset_no_adjacent_count",
    5015: "complete_multipartite_spanning_trees",
    5020: "binary_run_avoidance_count",
    5028: "bracelet_no_adjacent_count",
    5030: "strip_lattice_path_count",
    5049: "nested_modular_power_sum",
    5062: "quadratic_congruence_count",
    5072: "quadratic_form_maximum",
}


def _cases():
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["idx"]))
def test_hard_probe_has_one_certified_whole_answer(case):
    results = SympyTool().results_for(case["problem"])
    matching = [result for result in results if result.operation == EXPECTED_OPERATIONS[case["idx"]]]

    assert len(matching) == 1
    assert matching[0].verified
    assert matching[0].whole_answer_eligible
    assert matching[0].result


@pytest.mark.parametrize(
    "problem, forbidden_operation",
    [
        (
            "Find the number of spanning trees of the graph obtained from the complete "
            "bipartite graph $K_{5,7}$ by deleting a Hamiltonian cycle.",
            "complete_multipartite_spanning_trees",
        ),
        (
            r"How many integers satisfy $x^2\equiv2\pmod{840}$?",
            "quadratic_congruence_count",
        ),
        (
            "How many eight-digit integers use digits 0,1,...,7 with repetition and are divisible by 11?",
            "digit_permutation_divisibility",
        ),
        (
            r"How many functions $f:\{1,2,\ldots,10\}\to\{1,2,3,4\}$ satisfy "
            r"$f(i)\ne f(i+1)$?",
            "adjacent_surjection_count",
        ),
        (
            "A bracelet has $18$ positions. Exactly six positions are black. "
            "Two colorings are identified by rotation or reflection. How many are possible?",
            "bracelet_no_adjacent_count",
        ),
        (
            "A monotone lattice path from $(0,0)$ to $(15,11)$ uses steps $(1,0)$ and $(0,1)$. "
            "How many such paths are there?",
            "strip_lattice_path_count",
        ),
    ],
)
def test_specialized_handlers_reject_missing_or_changed_constraints(problem, forbidden_operation):
    operations = {result.operation for result in SympyTool().results_for(problem)}
    assert forbidden_operation not in operations


def test_unevaluated_sympy_integral_is_not_certified_as_a_definite_integral():
    problem = r"Evaluate $\int_{0}^{\infty} x^3/(e^x-1)\,dx$."
    results = SympyTool().results_for(problem)

    assert all(result.operation != "definite_integral" for result in results)


@pytest.mark.parametrize(
    "result",
    [
        "Integral(x**3/(exp(x) - 1), (x, 0, oo))",
        "Derivative(f(x), x)",
        "Limit(sin(x)/x, x, 0)",
        "RootSum(x**5 - x + 1)",
    ],
)
def test_inert_symbolic_objects_are_not_evaluated_results(result):
    assert not SympyTool._is_evaluated_result(result)
