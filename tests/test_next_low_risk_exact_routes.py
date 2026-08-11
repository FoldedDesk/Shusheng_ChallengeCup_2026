from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


BINOMIAL_EQUATION = (
    "求所有满足二项式系数C(n,2)=45的正整数n，并说明为何二次方程的另一根应舍去。"
)
CYCLIC_SUBGROUPS = (
    "设G为6阶循环群，求其所有子群的个数并说明子群与6的正因子的对应关系。"
)
LINEAR_NONADJACENT = (
    "在集合{1,2,…,10}中任选4个元素，求其中不含相邻整数的选法数，并用位置压缩法计算。"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified deterministic route called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def _whole(problem: str, operation: str):
    return next(
        item for item in _evidence(problem)
        if item.operation == operation and item.scope == "whole_goal" and item.verified
    )


def test_chinese_reference_style_problems_bypass_the_model_with_certificates():
    cases = (
        (BINOMIAL_EQUATION, "binomial_choose_two_positive_root", "n=10"),
        (CYCLIC_SUBGROUPS, "finite_cyclic_subgroup_count", "4"),
        (LINEAR_NONADJACENT, "linear_nonadjacent_selection", "35"),
    )
    for problem, operation, expected in cases:
        evidence = _whole(problem, operation)
        solved = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert evidence.certificate_method
        assert evidence.certificate_checks
        assert expected in evidence.result or equivalent_answers(evidence.result, expected)
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


def test_dynamic_english_parameters_are_recomputed():
    cases = (
        (
            r"Find all positive integers m satisfying \binom{m}{2}=21 and explain why the "
            r"other quadratic root is discarded.",
            "binomial_choose_two_positive_root",
            "m=7",
        ),
        (
            "Let H be a finite cyclic group of order 72. Find the total number of its subgroups "
            "and explain the correspondence with positive divisors of 72.",
            "finite_cyclic_subgroup_count",
            "12",
        ),
        (
            "From the set {1,2,...,15}, choose exactly 5 elements with no two chosen elements "
            "consecutive. Find the number of such selections using position compression.",
            "linear_nonadjacent_selection",
            "462",
        ),
    )
    for problem, operation, expected in cases:
        evidence = _whole(problem, operation)
        solved = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert expected in evidence.result or equivalent_answers(evidence.result, expected)
        assert expected in solved["final_response"] or equivalent_answers(
            solved["final_response"], expected
        )


def test_exact_no_solution_branches_are_certified_without_model_calls():
    cases = (
        (
            "求所有满足二项式系数C(n,2)=20的正整数n。",
            "binomial_choose_two_positive_root",
            "无正整数解",
        ),
        (
            "在集合{1,2,…,10}中任选6个元素，求其中不含相邻整数的选法数，"
            "并用位置压缩法计算。",
            "linear_nonadjacent_selection",
            "0",
        ),
    )
    for problem, operation, expected in cases:
        evidence = _whole(problem, operation)
        solved = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert expected in evidence.result or equivalent_answers(evidence.result, expected)
        assert expected in solved["final_response"] or equivalent_answers(
            solved["final_response"], expected
        )


def test_binomial_equation_near_neighbors_do_not_trigger():
    cases = (
        "求所有满足C(n,3)=20的正整数n。",
        "求所有满足C(n,2)=45的整数n。",
        "求满足C(n,2)≥45的最小正整数n。",
        "求满足C(n,2)=45的正整数解的个数。",
        r"Find all real n satisfying \binom{n}{2}=21.",
    )
    for problem in cases:
        assert not any(
            item.operation == "binomial_choose_two_positive_root"
            for item in _evidence(problem)
        )


def test_cyclic_group_near_neighbors_do_not_trigger():
    cases = (
        "设G为6阶群，求其所有子群的个数。",
        "设G为6阶循环群，求其所有真子群的个数。",
        "设G为6阶循环群，列出其所有子群。",
        "设G为6阶循环群，求其生成元个数。",
        "Let G be an infinite cyclic group. How many subgroups does it have?",
    )
    for problem in cases:
        assert not any(
            item.operation == "finite_cyclic_subgroup_count"
            for item in _evidence(problem)
        )


def test_linear_nonadjacent_near_neighbors_do_not_trigger():
    cases = (
        "在圆周上排列的集合{1,2,…,10}中任选4个元素，求不含相邻元素的选法数。",
        "在集合{1,2,…,10}中至多选4个元素，求不含相邻整数的选法数。",
        "在集合{1,2,…,10}中任选4个元素，要求任意两数之差至少为3，求选法数。",
        "在集合{1,2,…,10}中可重复选取4个元素，求不含相邻整数的选法数。",
        "求长度为10且恰有4个1、不含相邻1的二进制串数。",
    )
    for problem in cases:
        assert not any(
            item.operation == "linear_nonadjacent_selection"
            for item in _evidence(problem)
        )
