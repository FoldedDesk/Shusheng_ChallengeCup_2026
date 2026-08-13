from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


PROBLEM = (
    r"Start with the complete graph $K_9$ on labeled vertices $1,2,...,9$, "
    r"and delete the nine edges of the Hamiltonian cycle $1-2-3-...-9-1$. "
    r"How many spanning trees does the resulting graph have?"
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified graph count unexpectedly called the model: {kwargs}")


def _matching(problem: str):
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
    return [item for item in evidence if item.operation == "complete_graph_cycle_deletion_trees"]


def test_complete_graph_minus_hamilton_cycle_uses_exact_matrix_tree_count():
    evidence = _matching(PROBLEM)

    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "412164"

    result = ReasoningAgent(_NoModelClient()).solve(PROBLEM, {})
    assert "412164" in result["final_response"]


def test_cycle_deletion_route_recomputes_order_and_rejects_residual_constraints():
    changed = PROBLEM.replace("K_9", "K_7").replace("nine edges", "seven edges").replace(
        "...-9-1", "...-7-1"
    ).replace("1,2,...,9", "1,2,...,7")
    assert _matching(changed)[0].result == "1183"

    variants = (
        PROBLEM.replace("nine edges", "eight edges"),
        PROBLEM.replace("Hamiltonian cycle", "Hamiltonian path"),
        PROBLEM + " Also delete edge 1-3.",
        PROBLEM + " Count only spanning trees that contain edge 1-3.",
        PROBLEM + " How many of those trees avoid vertex 4 as a leaf?",
        PROBLEM + " Report the answer modulo 100.",
        PROBLEM + " Count only trees whose maximum vertex degree is 3.",
    )
    for problem in variants:
        assert not any(item.scope == "whole_goal" for item in _matching(problem))
