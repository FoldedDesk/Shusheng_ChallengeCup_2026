from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


EVEN_SUBSETS = (
    "设集合A有n个元素，求满足B⊆A且|B|为偶数的子集个数，"
    "并说明n≥1时为何两类子集数相等。"
)
DELETED_EDGE_PATHS = (
    "在完全二部图K_{4,5}中删去一条边后，求从左部指定顶点到右部指定非邻接顶点"
    "的长度为3的简单路径数。"
)
BOUNDED_COMPOSITIONS = (
    "求满足x_1+x_2+x_3+x_4=13且每个x_i为正整数、x_1≥3的解数，"
    "需通过变量平移化为隔板问题。"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified exact-count route called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def _whole(problem: str, operation: str):
    return next(
        item for item in _evidence(problem)
        if item.operation == operation and item.scope == "whole_goal" and item.verified
    )


def test_official_style_examples_are_certified_whole_answers_without_model_calls():
    cases = (
        (EVEN_SUBSETS, "even_subset_count", r"2^{n-1}"),
        (DELETED_EDGE_PATHS, "deleted_edge_bipartite_length_three_paths", "12"),
        (BOUNDED_COMPOSITIONS, "positive_composition_lower_bounds", "120"),
    )
    for problem, operation, expected in cases:
        evidence = _whole(problem, operation)
        solved = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert evidence.certificate_method
        assert evidence.certificate_checks
        assert expected in evidence.result or equivalent_answers(evidence.result, expected)
        assert solved["final_response"]
        assert expected in solved["final_response"] or equivalent_answers(
            solved["final_response"], expected
        )
        assert not any(
            str(step.get("step", "")).startswith("model_call_")
            for step in solved["trace"]
        )
        assert next(
            step for step in solved["trace"] if step["step"] == "selection"
        )["content"]["source"] == "sympy_verified"


def test_even_subset_route_keeps_the_requested_bijection_support():
    evidence = _whole(EVEN_SUBSETS, "even_subset_count")
    answer = ReasoningAgent(_NoModelClient()).solve(EVEN_SUBSETS, {})["final_response"]

    assert evidence.result == evidence.support
    assert "双射" in evidence.support
    assert r"2^{n-1}" in answer
    assert "偶数基数" in answer and "奇数基数" in answer


def test_parameter_changes_are_recomputed_instead_of_memorized():
    cases = (
        (
            "Let A be a set with 6 elements. How many subsets of A have even cardinality?",
            "even_subset_count",
            "32",
        ),
        (
            "In the complete bipartite graph K_{3,7}, delete one edge uv. Find the number of "
            "simple paths of length 3 from u to v.",
            "deleted_edge_bipartite_length_three_paths",
            "12",
        ),
        (
            "Find the number of positive integer solutions to x_1+x_2+x_3=9 with "
            "x_1>=2 and x_3>=4, using a variable shift and stars and bars.",
            "positive_composition_lower_bounds",
            "6",
        ),
    )
    for problem, operation, expected in cases:
        evidence = _whole(problem, operation)
        answer = ReasoningAgent(_NoModelClient()).solve(problem, {})["final_response"]

        assert evidence.result == expected
        assert equivalent_answers(answer, expected)


def test_impossible_lower_bounds_return_zero_with_a_valid_certificate():
    problem = (
        "求满足x_1+x_2+x_3=5且每个x_i为正整数、x_1≥4、x_2≥2的解数，"
        "需通过变量平移化为隔板问题。"
    )
    evidence = _whole(problem, "positive_composition_lower_bounds")
    answer = ReasoningAgent(_NoModelClient()).solve(problem, {})["final_response"]

    assert "0" in evidence.result
    assert "下界之和" in evidence.support
    assert "0" in answer
    assert "下界之和" in answer


def test_even_subset_near_neighbors_do_not_trigger():
    cases = (
        "设集合A有0个元素，求满足B⊆A且|B|为偶数的子集个数。",
        "设集合A有n个元素，求满足B⊆A且|B|为奇数的子集个数，其中n≥1。",
        "设集合A有n个元素，求满足B⊆A且|B|=4的子集个数，其中n≥4。",
        "Let A be a set with n elements. How many subsets of A have even cardinality?",
    )
    for problem in cases:
        assert not any(item.operation == "even_subset_count" for item in _evidence(problem))


def test_deleted_edge_path_near_neighbors_do_not_trigger():
    cases = (
        "在完全二部图K_{4,5}中删去两条边后，求两个非邻接顶点间长度为3的简单路径数。",
        "在有向完全二部图K_{4,5}中删去一条边，求缺边两端之间长度为3的简单路径数。",
        "在完全二部图K_{4,5}中删去一条边，求缺边两端之间长度为3的游走数。",
        "In K_{4,5}, delete one edge uv and count simple paths of length 5 from u to v.",
    )
    for problem in cases:
        assert not any(
            item.operation == "deleted_edge_bipartite_length_three_paths"
            for item in _evidence(problem)
        )


def test_positive_composition_near_neighbors_do_not_trigger():
    cases = (
        "求x_1+x_2+x_3=9的非负整数解数，其中x_1≥2。",
        "求正整数解x_1+x_2+x_3=9的个数，其中x_1≤2。",
        "求正整数解2x_1+x_2+x_3=9的个数，其中x_1≥2。",
        "求正整数解x_1+x_2+x_3=9的个数，其中x_1>x_2且x_3≥2。",
        "求正整数解x_1+x_2+x_3=9的个数。",
    )
    for problem in cases:
        assert not any(
            item.operation == "positive_composition_lower_bounds"
            for item in _evidence(problem)
        )
