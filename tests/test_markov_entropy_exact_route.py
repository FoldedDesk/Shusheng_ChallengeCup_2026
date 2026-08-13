from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


PROBLEM = (
    r"A stationary two-state Markov information source has transition matrix "
    r"$\begin{pmatrix}3/4&1/4\\1/2&1/2\end{pmatrix}$. Using logarithms to "
    r"base $2$, determine its entropy rate."
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified entropy route unexpectedly called the model: {kwargs}")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
    return [item for item in evidence if item.operation == "two_state_markov_entropy_rate"]


def test_two_state_markov_entropy_route_computes_stationary_weighted_row_entropy():
    evidence = _matching(PROBLEM)
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == (
        r"\frac{2}{3}H_2\!\left(\frac{1}{4}\right)+\frac{1}{3}"
    )
    response = ReasoningAgent(_NoModelClient()).solve(PROBLEM, {})["final_response"]
    assert "H_2" in response and r"\frac{2}{3}" in response


def test_two_state_markov_entropy_route_rejects_changed_contracts():
    variants = (
        PROBLEM.replace("stationary", "nonstationary"),
        PROBLEM.replace("base $2$", "base $e$"),
        PROBLEM.replace("3/4&1/4", "3/4&1/2"),
        PROBLEM + " Also determine the excess entropy.",
        PROBLEM.replace("entropy rate", "stationary distribution"),
    )
    for problem in variants:
        assert not _matching(problem)
