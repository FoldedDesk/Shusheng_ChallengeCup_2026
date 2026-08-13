from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


PROBLEM = (
    "Two heaps contain $a$ and $b$ stones. A move removes any positive number "
    "of stones from exactly one heap, or removes the same positive number from "
    "both heaps; the player making the last move wins. Among the positions with "
    "$0 <= a <= b <= 100$, how many are losing positions for the player whose "
    "turn it is?"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified Wythoff count unexpectedly called the model: {kwargs}")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
    return [item for item in evidence if item.operation == "wythoff_losing_position_count"]


def test_wythoff_position_count_uses_exact_beatty_enumeration():
    evidence = _matching(PROBLEM)

    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "39"
    assert "39" in ReasoningAgent(_NoModelClient()).solve(PROBLEM, {})["final_response"]


def test_wythoff_route_recomputes_bound_and_rejects_changed_rules():
    assert _matching(PROBLEM.replace("100", "5"))[0].result == "3"

    variants = (
        PROBLEM.replace("same positive number", "different positive numbers"),
        PROBLEM.replace("last move wins", "last move loses"),
        PROBLEM + " The same move may not be repeated.",
        PROBLEM + " Count only positions for which a+b is even.",
        PROBLEM + " Among those positions, include only coprime pairs.",
        PROBLEM + " Exclude positions where the two heaps are equal.",
        PROBLEM.replace(
            "how many are losing positions",
            "how many losing positions have equal heap sizes",
        ),
    )
    for problem in variants:
        assert not any(item.scope == "whole_goal" for item in _matching(problem))
