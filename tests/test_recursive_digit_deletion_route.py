from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


ENGLISH = (
    'A positive integer $m$ consisting of distinct digits is considered "good" if it is '
    "a single-digit number, or if removing one of its digits results in a divisor of "
    "$m$ that is also a good number. A deletion may not leave a leading zero. "
    "Find the largest good number."
)
CHINESE = (
    "一个由互不相同的数字组成的正整数$m$被称为“好数”，如果它是一位数，或者删去"
    "其中一个数字后得到的数是$m$的约数且也是好数。删去后不允许前导零。求最大的好数。"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified recursive search called the model: {kwargs}")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == "recursive_digit_deletion_maximum"
    ]


@pytest.mark.parametrize("problem", [ENGLISH, CHINESE])
def test_recursive_digit_deletion_route_is_certified_and_bilingual(problem):
    evidence = _matching(problem)

    assert len(evidence) == 1
    assert evidence[0].result == "146250"
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"
    assert {
        "one_digit_canonical_deletion",
        "complete_finite_state_enumeration",
        "all_ten_decimal_digits_exhausted",
        "maximality_by_empty_longer_layers",
    } <= set(evidence[0].certificate_checks)


def test_recursive_digit_deletion_route_bypasses_model_and_preserves_box():
    result = ReasoningAgent(_NoModelClient()).solve(
        ENGLISH + r" Remember to put your final answer within \boxed{}.", {}
    )

    assert result["final_response"] == r"\boxed{146250}"
    assert not any(
        str(step.get("step", "")).startswith("model_call_")
        for step in result["trace"]
    )


def test_returned_number_has_a_complete_recursive_divisor_chain():
    chain = (146250, 14625, 1625, 125, 25, 5)

    for value, divisor in zip(chain, chain[1:]):
        assert value % divisor == 0
        before, after = str(value), str(divisor)
        assert len(before) == len(after) + 1
        assert any(before[:index] + before[index + 1 :] == after for index in range(len(before)))
    assert all(len(set(str(value))) == len(str(value)) for value in chain)


@pytest.mark.parametrize(
    "problem",
    [
        ENGLISH.replace("distinct digits", "digits, with repetition allowed"),
        ENGLISH.replace("removing one of its digits", "removing two of its digits"),
        ENGLISH.replace("largest good number", "smallest good number"),
        ENGLISH.replace("Find the largest good number", "Find how many good numbers there are"),
        ENGLISH.replace(
            "A deletion may not leave a leading zero.",
            "Leading zeroes are allowed after deletion.",
        ),
        ENGLISH.replace("positive integer", "positive base-8 integer"),
        CHINESE.replace("最大的好数", "最小的好数"),
        CHINESE.replace("删去其中一个数字", "删去其中两个数字"),
        CHINESE.replace("删去后不允许前导零", "删去后允许前导零"),
        ENGLISH.replace("A deletion may not leave a leading zero. ", ""),
        CHINESE.replace("删去后不允许前导零。", ""),
    ],
)
def test_recursive_digit_deletion_route_rejects_changed_contract(problem):
    assert not _matching(problem)
