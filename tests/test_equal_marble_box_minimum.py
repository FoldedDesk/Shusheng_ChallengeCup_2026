from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


def _english(count: int) -> str:
    return (
        "Consider a game where you start with $" + str(count)
        + "$ boxes, each containing a single marble. A move consists of selecting two "
        "boxes, removing an equal number of marbles from each, and creating a new box "
        "with the combined marbles. What is the minimum number of non-empty boxes that "
        "can be achieved through a finite sequence of such moves?"
    )


def _chinese(count: int) -> str:
    return (
        "考虑如下游戏：开始时有" + str(count)
        + "个盒子，每个盒子恰好有一颗弹珠。每次操作选择两个不同的非空盒子，从每个盒子中"
        "取出相同的正数量的弹珠，并新建一个盒子放入取出的全部弹珠。问经过有限次操作后"
        "非空盒子的最少数量是多少？"
    )


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError("certified box invariant route called the model")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == "equal_marble_box_minimum"
    ]


@pytest.mark.parametrize(
    "count, expected",
    [(1, "1"), (2, "1"), (3, "2"), (4, "1"), (7, "2"), (8, "1"), (2025, "2")],
)
def test_equal_marble_box_route_recomputes_parameter(count, expected):
    evidence = _matching(_english(count))

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"
    assert {
        "odd_common_divisor_reverse_invariant",
        "power_of_two_single_box_criterion",
        "two_box_construction_for_non_power_of_two",
    } <= set(evidence[0].certificate_checks)


@pytest.mark.parametrize("count, expected", [(4, "1"), (15, "2"), (16, "1")])
def test_equal_marble_box_route_is_bilingual(count, expected):
    assert _matching(_chinese(count))[0].result == expected


def test_equal_marble_box_route_bypasses_model_and_preserves_box():
    result = ReasoningAgent(_NoModelClient()).solve(
        _english(2025) + r" Remember to put your final answer within \boxed{}.", {}
    )

    assert result["final_response"] == r"\boxed{2}"
    assert not any(
        str(step.get("step", "")).startswith("model_call_")
        for step in result["trace"]
    )


@pytest.mark.parametrize(
    "problem",
    [
        _english(9).replace("a single marble", "two marbles"),
        _english(9).replace("selecting two boxes", "selecting three boxes"),
        _english(9).replace("an equal number", "a different number"),
        _english(9).replace(
            "creating a new box",
            "discarding the removed marbles and creating a new box",
        ),
        _english(9).replace("minimum number of non-empty boxes", "minimum number of moves"),
        _english(9) + " Use at most five moves.",
        _chinese(9).replace("相同的正数量", "不相同数量"),
        _chinese(9).replace("选择两个", "选择三个"),
        _chinese(9).replace("最少数量", "最多数量"),
    ],
)
def test_equal_marble_box_route_rejects_changed_rules(problem):
    assert not _matching(problem)
