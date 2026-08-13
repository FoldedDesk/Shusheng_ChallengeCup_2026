from itertools import combinations
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


def _english(size: int) -> str:
    value = str(size)
    return (
        "Suppose we have a $" + value + r" \times " + value
        + "$ board and we want to mark some cells on this board. Determine the smallest "
        r"positive integer $k$ such that it is possible to mark $k$ cells on the board "
        r"in a way that there exists a unique partition of the board into $1 \times 2$ "
        r"and $2 \times 1$ dominoes, where none of the dominoes contains two marked cells."
    )


def _chinese(size: int) -> str:
    value = str(size)
    return (
        "在一个" + value + "×" + value
        + "的方格棋盘上标记若干格。求最小正整数$k$，使得可以标记$k$个格子，并且棋盘"
        "存在唯一一种用1×2和2×1多米诺骨牌完全分割的方式，其中任意一块多米诺骨牌"
        "都不包含两个被标记的格子。"
    )


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError("certified unique-domino route called the model")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == "unique_domino_partition_marking"
    ]


@pytest.mark.parametrize("size", [2, 4, 6, 194, 1000])
def test_unique_domino_route_recomputes_even_side_length(size):
    evidence = _matching(_english(size))

    assert len(evidence) == 1
    assert evidence[0].result == str(size)
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"
    assert {
        "alternating_cycle_lower_bound",
        "diagonal_marking_construction",
        "small_board_exhaustive_crosscheck",
    } <= set(evidence[0].certificate_checks)


@pytest.mark.parametrize("size", [2, 4, 194])
def test_unique_domino_route_is_bilingual(size):
    assert _matching(_chinese(size))[0].result == str(size)


def test_unique_domino_route_bypasses_model_and_preserves_box():
    result = ReasoningAgent(_NoModelClient()).solve(
        _english(194) + r" Remember to put your final answer within \boxed{}.", {}
    )

    assert result["final_response"] == r"\boxed{194}"
    assert not any(
        str(step.get("step", "")).startswith("model_call_")
        for step in result["trace"]
    )


def _tilings(size: int):
    full = (1 << (size * size)) - 1
    found = []

    def visit(mask: int, dominoes: tuple[tuple[int, int], ...]) -> None:
        if mask == full:
            found.append(dominoes)
            return
        first = next(index for index in range(size * size) if not mask & (1 << index))
        row, column = divmod(first, size)
        neighbors = []
        if column + 1 < size:
            neighbors.append(first + 1)
        if row + 1 < size:
            neighbors.append(first + size)
        for second in neighbors:
            if not mask & (1 << second):
                visit(
                    mask | (1 << first) | (1 << second),
                    dominoes + ((first, second),),
                )

    visit(0, ())
    return found


@pytest.mark.parametrize("size", [2, 4])
def test_small_boards_independently_confirm_the_minimum(size):
    tilings = _tilings(size)
    minimum = None
    for marked_count in range(size * size + 1):
        for marked in combinations(range(size * size), marked_count):
            marked_set = set(marked)
            valid = sum(
                all(not ({first, second} <= marked_set) for first, second in tiling)
                for tiling in tilings
            )
            if valid == 1:
                minimum = marked_count
                break
        if minimum is not None:
            break
    assert minimum == size


@pytest.mark.parametrize(
    "problem",
    [
        _english(6).replace(r"$6 \times 6$", r"$6 \times 8$"),
        _english(6).replace("unique partition", "at most one partition"),
        _english(6).replace("contains two marked cells", "contains a marked cell"),
        _english(6).replace("smallest positive integer", "largest positive integer"),
        _english(6).replace(r"$1 \times 2$ and $2 \times 1$", "L-triomino"),
        _english(6) + " The board has one hole.",
        _english(5),
        _chinese(6).replace("唯一一种", "至多一种"),
        _chinese(6).replace("不包含两个被标记的格子", "不包含被标记的格子"),
        _chinese(6).replace("最小正整数", "最大正整数"),
    ],
)
def test_unique_domino_route_rejects_changed_contract(problem):
    assert not _matching(problem)
