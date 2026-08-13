import json
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


DATASET = Path(__file__).parents[1] / "sample_data" / "judge1_style_112_hard_v1.jsonl"
ROWS = {
    row["idx"]: row
    for row in map(json.loads, DATASET.read_text(encoding="utf-8").splitlines())
}

CASES = {
    "complete_multipartite_spanning_trees": (5001, "Count only trees containing edge u_2v_2."),
    "quadratic_congruence_count": (5002, r"Also require $x\equiv1\pmod 8$."),
    "digit_permutation_divisibility": (5005, "Also require the leading digit to be even."),
    "adjacent_surjection_count": (5008, r"Also require $f(10)\ne f(1)$."),
    "binary_run_avoidance_count": (5020, "Also require exactly ten ones."),
    "bracelet_no_adjacent_count": (5028, "Nonblack positions may use either white or red."),
    "strip_lattice_path_count": (5030, "Also require the path to pass through (8,5)."),
    "bipartite_matching_deletion_trees": (5038, "Also delete edge u_1v_2."),
    "finite_subtraction_game": (5003, "The same move may not be used on consecutive turns."),
    "bounded_generalized_pell_count": (5048, r"Also require $\gcd(x,y)=1$."),
    "punctured_domino_tilings": (5035, "Also require exactly ten vertical dominoes."),
    "positive_sum_two_squares": (5060, r"Also require $x<y$."),
    "odd_fiber_functions": (5040, r"Also require $f(1)=1$."),
}


def _problem(idx):
    return ROWS[idx]["problem"].split("Remember to put", 1)[0].strip()


def _matching(problem, operation):
    return [
        result for result in SympyTool().results_for(problem)
        if result.operation == operation
    ]


@pytest.mark.parametrize("operation", CASES)
def test_standard_exact_route_remains_a_whole_answer(operation):
    idx, _ = CASES[operation]
    problem = _problem(idx)
    results = _matching(problem, operation)

    assert len(results) == 1
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize("operation", CASES)
def test_retained_trigger_with_extra_constraint_never_bypasses_model(operation):
    idx, extra = CASES[operation]
    changed = _problem(idx) + " " + extra
    results = _matching(changed, operation)

    # The old deterministic matcher still recognizes the standard core; only
    # the whole-answer authorization must be removed.
    assert len(results) == 1
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(changed))
    assert len(evidence) == 1
    assert evidence[0].scope == "subexpression"
    assert SubmissionAgent._whole_tool_answer(evidence) == ""


@pytest.mark.parametrize(
    "suffix",
    (
        " 最终只需给出所求量加1后的值。",
        " Report the requested quantity plus 1 as the final answer.",
        " 另外要求答案满足一个额外限制。",
        " Additionally require the answer to satisfy an extra restriction.",
    ),
)
def test_generic_output_transform_or_added_restriction_downgrades_every_known_whole_route(suffix):
    root = Path(__file__).parents[1]
    datasets = (
        root / "sample_data" / "official_distribution_112_hard.jsonl",
        root / "sample_data" / "judge1_style_112_hard_v1.jsonl",
    )
    sympy = SympyTool()
    whole_route_count = 0

    for dataset in datasets:
        rows = map(json.loads, dataset.read_text(encoding="utf-8").splitlines())
        for row in rows:
            problem = str(row["problem"])
            original = SubmissionAgent._tool_evidence(
                sympy.results_for(problem), build_problem_spec(problem)
            )
            if not SubmissionAgent._whole_tool_answer(original):
                continue
            whole_route_count += 1
            changed = problem + suffix
            changed_evidence = SubmissionAgent._tool_evidence(
                sympy.results_for(changed), build_problem_spec(changed)
            )
            assert SubmissionAgent._whole_tool_answer(changed_evidence) == "", row["idx"]

    assert whole_route_count == 99


@pytest.mark.parametrize("idx", (4001, 4004, 4010, 4014, 4016, 4021))
def test_exact_route_preserves_support_when_problem_explicitly_requires_a_method(idx):
    dataset = Path(__file__).parents[1] / "sample_data" / "official_distribution_112_hard.jsonl"
    rows = {
        row["idx"]: row
        for row in map(json.loads, dataset.read_text(encoding="utf-8").splitlines())
    }
    problem = rows[idx]["problem"]
    spec = build_problem_spec(problem)
    results = SympyTool().results_for(problem)
    evidence = SubmissionAgent._tool_evidence(results, spec)
    whole_answer = SubmissionAgent._whole_tool_answer(evidence)

    assert any(
        requirement.strict and requirement.category == "support"
        for goal in spec.goals
        for requirement in goal.requirements
    )
    assert len(evidence) == 1
    assert evidence[0].support
    assert evidence[0].result == evidence[0].support
    assert whole_answer == evidence[0].support
