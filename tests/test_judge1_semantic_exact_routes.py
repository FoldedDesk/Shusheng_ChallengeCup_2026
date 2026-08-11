import json
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


ROOT = Path(__file__).parents[1]
PROBE_ROWS = {
    str(row["idx"]): row
    for row in map(
        json.loads,
        (ROOT / "sample_data" / "judge1_hard_probe_0_4_judge7.jsonl").read_text().splitlines(),
    )
}

GRAPH_HOMOMORPHISM_PROBLEM = r"""Let $n,s,$ and $t$ be positive integers and $0<\lambda<1.$ A simple graph on
$n$ vertices with at least $\lambda n^2$ edges is given. We say that
$(x_1,\ldots,x_s,y_1,\ldots,y_t)$ is a good intersection if the letters denote not
necessarily distinct vertices and every $x_i y_j$ is an edge of the graph
$(1\leq i\leq s, 1\leq j\leq t)$. Find the minimum number of good intersections."""

TANGENTIAL_POLYGON_PROBLEM = r"""A convex $m$-gon $Q$, where $m > 3$, is divided into
identical triangles by diagonals that do not intersect within it. For which values of $m$ is it
possible for $Q$ to be circumscribed? Remember to put your final answer within \boxed{}."""

FORMAL_ADJOINT_PROBLEM = r"""设 $\Omega$ 为开区域. 算子
$Lu := \sum_{i,j=1}^n \partial_i(a_{ij}\partial_j u)+\sum_{j=1}^n b_j\partial_j u+cu$,
定义域为 $C_0^\infty(\Omega)$. 其中 $a_{ij},b_j,c$ 为实有界光滑函数. 求 $L$ 在
$L^2(\Omega)$ 上的伴随算子 $L^*$."""


def _grid_compression_problem(x_base=2, y_base=3, z_base=5):
    return rf"""Let $a,b,c$ be positive integers and
$Q=\{{(x,y,z)\in\mathbb{{Z}}^3:0\le x\le a,0\le y\le b,0\le z\le c\}}$.
Initially a total of $M$ identical pieces are distributed among the points in $Q$.
(1) Remove {x_base} pieces from a point $(x,y,z)$ and place one piece on the point
$(x-1,y,z)$, provided $x>0$.
(2) Remove {y_base} pieces from a point $(x,y,z)$ and place one piece on the point
$(x,y-1,z)$, provided $y>0$.
(3) Remove {z_base} pieces from a point $(x,y,z)$ and place one piece on the point
$(x,y,z-1)$, provided $z>0$.
Find the smallest positive integer $M$ such that, regardless of the initial distribution,
one can always perform a sequence of operations to place at least one piece on the point
$(0,0,0)$."""


def _results(problem: str):
    return {result.operation: result for result in SympyTool().results_for(problem)}


@pytest.mark.parametrize(
    "problem, operation, expected",
    [
        (PROBE_ROWS["1"]["problem"], "sparkling_tuple_pair_sum", "2-2m"),
        (PROBE_ROWS["2"]["problem"], "five_number_ratio_gap", "1/2"),
        (
            PROBE_ROWS["3"]["problem"],
            "nested_nonnegative_sequence_values",
            r"\{2026,2030\}",
        ),
        (
            GRAPH_HOMOMORPHISM_PROBLEM,
            "complete_bipartite_homomorphism_bound",
            r"\lambda^{st}n^{s+t}",
        ),
        (
            TANGENTIAL_POLYGON_PROBLEM,
            "tangential_identical_triangulation_polygon",
            "4",
        ),
        (
            FORMAL_ADJOINT_PROBLEM,
            "formal_l2_adjoint",
            r"L^*v=\sum_{i,j=1}^n\partial_j(a_{ij}\partial_i v)-\sum_{j=1}^n\partial_j(b_jv)+cv",
        ),
        (
            _grid_compression_problem(),
            "mixed_radix_grid_compression",
            "2^a3^b5^c",
        ),
    ],
)
def test_judge1_semantic_family_is_certified_and_covers_whole_goal(problem, operation, expected):
    result = _results(problem)[operation]

    assert result.result == expected
    assert result.verified
    evidence = SubmissionAgent._tool_evidence([result], build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


def test_sparkling_bound_scales_with_the_stated_adjacent_lower_bound():
    changed = PROBE_ROWS["1"]["problem"].replace(r"\geqslant-4", r"\geqslant-6")
    assert _results(changed)["sparkling_tuple_pair_sum"].result == "3-3m"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace(r"m\ge 3", r"m\ge 2"),
        lambda text: text.replace("real numbers", "integers"),
        lambda text: text.replace("for each permutation", "for some permutation"),
        lambda text: text.replace("T=T(m)", "T=4"),
        lambda text: text.replace(r"p< q", r"p\le q"),
    ],
)
def test_sparkling_handler_rejects_changed_theorem_contract(mutator):
    assert "sparkling_tuple_pair_sum" not in _results(mutator(PROBE_ROWS["1"]["problem"]))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("five distinct positive", "four distinct positive"),
        lambda text: text.replace("five distinct positive", "five positive"),
        lambda text: text.replace("choose four distinct", "choose three distinct"),
        lambda text: text.replace("Tfh", "Tfg"),
        lambda text: text.replace("minimum value", "maximum value"),
    ],
)
def test_five_number_ratio_handler_rejects_changed_contract(mutator):
    assert "five_number_ratio_gap" not in _results(mutator(PROBE_ROWS["2"]["problem"]))


def _sequence_problem(index: int) -> str:
    return rf"""Let $a_0,a_1,\ldots$ be a sequence of non-negative integers. Suppose that
for all non-negative integers $p$, $a_{{a_{{a_p}}}}=a_{{p+1}}+1$. Find all possible
values of $a_{{{index}}}$."""


@pytest.mark.parametrize(
    "index, expected",
    [
        (8, "9"),
        (9, r"\{10,14\}"),
        (10, "11"),
        (11, r"\{8,12\}"),
    ],
)
def test_nested_sequence_classification_handles_every_residue(index, expected):
    assert _results(_sequence_problem(index))["nested_nonnegative_sequence_values"].result == expected


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("non-negative integers", "positive integers", 1),
        lambda text: text.replace(r"a_{a_{a_p}}", r"a_{a_p}"),
        lambda text: text.replace(r"a_{p+1} + 1", r"a_{p+1} + 2"),
        lambda text: text.replace(r"a_{p+1} + 1", r"a_{p+2} + 1"),
        lambda text: text.replace("for all non-negative integers", "for all positive integers"),
        lambda text: text.replace("Find all possible", r"Also $a_0=1$. Find all possible"),
    ],
)
def test_nested_sequence_handler_rejects_changed_contract(mutator):
    assert "nested_nonnegative_sequence_values" not in _results(mutator(PROBE_ROWS["3"]["problem"]))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("not\nnecessarily distinct", "distinct"),
        lambda text: text.replace(r"\lambda n^2", r"\lambda\binom n2"),
        lambda text: text.replace("every $x_i y_j$", "some $x_i y_j$"),
        lambda text: text.replace("minimum number", "maximum number"),
        lambda text: text.replace("positive integers", "non-negative integers"),
    ],
)
def test_graph_homomorphism_handler_rejects_changed_contract(mutator):
    assert "complete_bipartite_homomorphism_bound" not in _results(
        mutator(GRAPH_HOMOMORPHISM_PROBLEM)
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("identical triangles", "similar triangles"),
        lambda text: text.replace("do not intersect", "may intersect"),
        lambda text: text.replace("circumscribed", "inscribed"),
        lambda text: text.replace("m > 3", "m > 2"),
    ],
)
def test_tangential_polygon_handler_rejects_changed_contract(mutator):
    assert "tangential_identical_triangulation_polygon" not in _results(
        mutator(TANGENTIAL_POLYGON_PROBLEM)
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("实有界光滑", "复值有界光滑"),
        lambda text: text.replace(r"C_0^\infty", r"C^\infty"),
        lambda text: text.replace(r"\partial_i(a_{ij}\partial_j u)", r"a_{ij}\partial_i\partial_j u"),
        lambda text: text.replace(r"+\sum_{j=1}^n b_j\partial_j u", r"-\sum_{j=1}^n b_j\partial_j u"),
        lambda text: text.replace(r"L^2(\Omega)", r"weighted L^2(\Omega)"),
    ],
)
def test_formal_adjoint_handler_rejects_changed_contract(mutator):
    assert "formal_l2_adjoint" not in _results(mutator(FORMAL_ADJOINT_PROBLEM))


def test_grid_compression_reads_each_carry_base_from_the_problem():
    result = _results(_grid_compression_problem(4, 6, 7))["mixed_radix_grid_compression"]
    assert result.result == "4^a6^b7^c"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda text: text.replace("positive integers", "real numbers"),
        lambda text: text.replace("x-1,y,z", "x+1,y,z"),
        lambda text: text.replace("provided $y>0$", "provided $y=0$"),
        lambda text: text.replace("regardless of the initial distribution", "for some initial distribution"),
        lambda text: text.replace("point\n$(0,0,0)$", "point\n$(1,0,0)$"),
    ],
)
def test_grid_compression_handler_rejects_changed_contract(mutator):
    assert "mixed_radix_grid_compression" not in _results(
        mutator(_grid_compression_problem())
    )
