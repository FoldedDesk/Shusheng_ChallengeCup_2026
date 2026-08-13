from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


RESIDUES = (
    r"How many residue classes $x$ modulo $1000$ satisfy "
    r"$x^5\equiv x\pmod{1000}$?"
)
BRACELETS = (
    "A bracelet is made from six black beads and six white beads. Two arrangements "
    "are considered the same if one can be obtained from the other by a rotation or "
    "a reflection. How many distinct bracelets are there?"
)
TREES = (
    "Among all labeled trees on vertex set {1,2,...,12}, how many have vertex 1 of "
    "degree 1, vertex 2 of degree 2, and vertex 3 of degree 3, with no restrictions "
    "on the remaining degrees?"
)
PERMUTATIONS = (
    "How many permutations of {1,2,...,12} have exactly four cycles in their "
    "disjoint-cycle decomposition and have every cycle of odd length? Cycles and "
    "their order are interpreted in the usual permutation sense, so cyclic rotations "
    "do not create new cycles."
)
RECIPROCAL_SUM = (
    "For every unordered pair of positive integers {x,y} satisfying "
    "$1/x+1/y=1/60$, form the value $x+y$. Find the sum of these values over "
    "all distinct unordered solutions."
)
GRID_TRIANGLES = (
    r"Let $S=\{(i,j):i,j\in\{0,1,2,3,4\}\}$. Determine the number of "
    r"nondegenerate triangles whose three vertices are distinct points of $S$."
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified route unexpectedly called the model: {kwargs}")


def _matching(problem: str, operation: str):
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
    return [item for item in evidence if item.operation == operation]


def test_power_fixed_residue_route_exhaustively_counts_all_classes():
    evidence = _matching(RESIDUES, "power_fixed_residue_count")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "25"
    assert "25" in ReasoningAgent(_NoModelClient()).solve(RESIDUES, {})["final_response"]


def test_fixed_weight_bracelet_route_uses_reflections_and_rotations():
    evidence = _matching(BRACELETS, "fixed_weight_binary_bracelets")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "50"
    assert "50" in ReasoningAgent(_NoModelClient()).solve(BRACELETS, {})["final_response"]


def test_fixed_weight_bracelet_burnside_formula_handles_odd_and_even_orders():
    odd = BRACELETS.replace("six black beads and six white beads", "three black beads and four white beads")
    even = BRACELETS.replace("six black beads and six white beads", "two black beads and six white beads")

    assert _matching(odd, "fixed_weight_binary_bracelets")[0].result == "4"
    assert _matching(even, "fixed_weight_binary_bracelets")[0].result == "4"


def test_specified_degree_tree_route_counts_free_prufer_symbols():
    evidence = _matching(TREES, "specified_degree_labeled_trees")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "1721868840"
    assert "1721868840" in ReasoningAgent(_NoModelClient()).solve(TREES, {})["final_response"]


def test_odd_cycle_permutation_route_uses_exact_recurrence():
    evidence = _matching(PERMUTATIONS, "odd_cycle_permutations")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "30633856"
    assert "30633856" in ReasoningAgent(_NoModelClient()).solve(PERMUTATIONS, {})["final_response"]


def test_reciprocal_pair_sum_route_uses_complete_divisor_pairing():
    evidence = _matching(RECIPROCAL_SUM, "reciprocal_pair_sum")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "15313"
    assert "15313" in ReasoningAgent(_NoModelClient()).solve(RECIPROCAL_SUM, {})["final_response"]


def test_integer_grid_triangle_route_checks_every_triple_by_determinant():
    evidence = _matching(GRID_TRIANGLES, "integer_grid_nondegenerate_triangles")
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].result == "2148"
    assert "2148" in ReasoningAgent(_NoModelClient()).solve(GRID_TRIANGLES, {})["final_response"]


def test_new_routes_reject_changed_or_extra_constraints():
    variants = (
        (RESIDUES.replace(r"\pmod{1000}", r"\pmod{999}"), "power_fixed_residue_count"),
        (RESIDUES + " Count only even representatives.", "power_fixed_residue_count"),
        (BRACELETS.replace("rotation or a reflection", "rotation"), "fixed_weight_binary_bracelets"),
        (BRACELETS + " Adjacent black beads are forbidden.", "fixed_weight_binary_bracelets"),
        (TREES.replace("no restrictions", "an additional restriction"), "specified_degree_labeled_trees"),
        (TREES + " The tree must contain edge 1-2.", "specified_degree_labeled_trees"),
        (PERMUTATIONS.replace("odd length", "even length"), "odd_cycle_permutations"),
        (PERMUTATIONS + " Element 1 must be fixed.", "odd_cycle_permutations"),
        (RECIPROCAL_SUM.replace("unordered", "ordered", 1), "reciprocal_pair_sum"),
        (RECIPROCAL_SUM + " Require x and y to be distinct.", "reciprocal_pair_sum"),
        (GRID_TRIANGLES.replace("distinct points", "points"), "integer_grid_nondegenerate_triangles"),
        (GRID_TRIANGLES + " Count only triangles of integer area.", "integer_grid_nondegenerate_triangles"),
    )
    for problem, operation in variants:
        assert not _matching(problem, operation)
