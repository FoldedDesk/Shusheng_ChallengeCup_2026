from __future__ import annotations

import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


HEXAGON = (
    "Consider a regular hexagon with side length $100$ that is divided into "
    "equilateral triangles with side length $1$ by lines parallel to its sides. "
    "Additionally, there are two circles with radii $99$ and $101$, respectively. "
    "Find the number of regular hexagons all of whose vertices are among the vertices "
    "of the equilateral triangles of the regular hexagon with side length $100$."
)

LINE_COVER = (
    "Let $S$ be a subset of $2012$ points on the plane. There does not exist $2012$ "
    "lines such that every element of $S$ lies on one of them. For all $X \\in S$ "
    "there exists $2012$ lines such that every element of $S-\\{X\\}$ lies on one "
    "of them. Find the maximum possible value of $|S|$."
)

PAIR_PARITY = (
    "Find all even integers $d$ such that the number of ordered integer pairs $(x,y)$ "
    "satisfying $(x+2y-d)^2=xy$ is even."
)

INVARIANT_RING = (
    r"Find all $f \in \mathbb{C}[x,y]$ such that for all complex numbers $a,b$, "
    r"$f(a^2,b^2)=f\left(\frac{(a-b)^2}{2},\frac{(a+b)^2}{2}\right)$."
)

NORMAL_PARAMETERS = (
    "正态分布的两个参数分别是（）\n"
    "A.均值和方差B.均值和标准差C.中位数和方差D.中位数和标准差"
)

DOMINO = (
    r"A domino is a $2 \times 1$ or $1 \times 2$ tile. Place exactly $k^2$ "
    r"dominoes on a $2k \times 2k$ chessboard without overlapping. Every "
    r"$2 \times 2$ square contains at least two uncovered unit squares in the "
    "same row or column. Determine how many ways the dominoes can be placed."
)

RED_BLUE = (
    "In the plane, 7 red points and 8 blue points are marked so that no three "
    "marked points are collinear. Draw k lines not passing through the marked "
    "points so that no region contains points of both colors. Find the minimal "
    "k for every possible configuration of 15 points."
)

CLUSTERED = (
    r"Let $a$ be a positive integer greater than or equal to $3$. A finite set "
    r"$X$ of positive integers is clustered if for any three elements, at least "
    r"one of their pairwise gcd values is not equal to $1$. Find the maximum "
    r"possible value of $|X|$ when the difference between the maximum and minimum "
    r"elements is less than or equal to $a$."
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified route called the model: {kwargs}")


def _operation(problem: str, operation: str):
    return [
        result
        for result in SympyTool().results_for(problem)
        if result.operation == operation
    ]


def _whole(problem: str, operation: str) -> bool:
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
    return any(item.operation == operation and item.scope == "whole_goal" for item in evidence)


@pytest.mark.parametrize(
    ("problem", "operation", "expected"),
    (
        (HEXAGON, "triangular_lattice_regular_hexagons", r"\boxed{25502500}"),
        (LINE_COVER, "critical_line_cover_point_set", r"\boxed{2027091}"),
        (
            PAIR_PARITY,
            "even_quadratic_pair_count_parameters",
            r"\boxed{d\in14\mathbb{Z}\setminus\{0\}}",
        ),
        (
            INVARIANT_RING,
            "quadratic_transform_invariant_polynomials",
            r"\boxed{f(x,y)=g\!(x+y,xy(x-y)^2),\quad g\in\mathbb{C}[u,v]}",
        ),
        (NORMAL_PARAMETERS, "normal_distribution_parameters", r"\boxed{B}"),
        (DOMINO, "sparse_domino_placements", r"\boxed{\binom{2k}{k}^2}"),
        (RED_BLUE, "red_blue_line_separation", r"\boxed{7}"),
        (
            CLUSTERED,
            "clustered_interval_maximum",
            r"\boxed{\lfloor\frac{a+2}{2}\rfloor+\lfloor\frac{a+2}{3}\rfloor-\lfloor\frac{a+2}{6}\rfloor}",
        ),
    ),
)
def test_certified_accuracy_floor_routes_bypass_model(problem, operation, expected):
    result = ReasoningAgent(_NoModelClient()).solve(
        problem + r" Remember to put your final answer within \boxed{}.",
        {},
    )

    assert result["final_response"] == expected
    selection = next(step for step in result["trace"] if step["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"
    matched = _operation(problem, operation)
    assert len(matched) == 1
    assert matched[0].certificate.passed


@pytest.mark.parametrize(
    ("problem", "operation"),
    (
        (HEXAGON.replace("side length $1$", "side length $2$"), "triangular_lattice_regular_hexagons"),
        (HEXAGON.replace("number of regular hexagons", "number of unit regular hexagons"), "triangular_lattice_regular_hexagons"),
        (HEXAGON.replace("lines parallel", "lines perpendicular"), "triangular_lattice_regular_hexagons"),
        (HEXAGON + " Count only hexagons inside the first circle.", "triangular_lattice_regular_hexagons"),
        (HEXAGON + " Count only regular hexagons of side length 1.", "triangular_lattice_regular_hexagons"),
        (HEXAGON + " Return the result modulo 7.", "triangular_lattice_regular_hexagons"),
        (LINE_COVER.replace("exists $2012$ lines", "exists $2011$ lines", 1), "critical_line_cover_point_set"),
        (LINE_COVER.replace("maximum", "minimum"), "critical_line_cover_point_set"),
        (LINE_COVER.replace("$2012$ lines", "at most $2012$ lines"), "critical_line_cover_point_set"),
        (LINE_COVER + " Assume all points are collinear.", "critical_line_cover_point_set"),
        (LINE_COVER + " Then add 1 to the maximum.", "critical_line_cover_point_set"),
        (PAIR_PARITY.replace("even integers", "odd integers"), "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY.replace("ordered integer pairs", "unordered integer pairs"), "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY.replace("x+2y-d", "x+3y-d"), "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY.replace("is even", "is odd"), "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY + " Require x and y to be positive.", "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY.replace("even integers", "positive even integers"), "even_quadratic_pair_count_parameters"),
        (PAIR_PARITY.replace("=xy", "=xy+1"), "even_quadratic_pair_count_parameters"),
        (INVARIANT_RING.replace(r"\mathbb{C}", r"\mathbb{R}"), "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING.replace("for all complex numbers", "for all real numbers"), "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING.replace(r"\frac{(a-b)^2}{2}", r"\frac{(a-b)^2}{3}"), "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING.replace("Find all", "Find one"), "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING + " Assume f is homogeneous of degree 6.", "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING + " Also require f(0,0)=1.", "quadratic_transform_invariant_polynomials"),
        (INVARIANT_RING + " Also require f(1,1)=0.", "quadratic_transform_invariant_polynomials"),
        (NORMAL_PARAMETERS.replace("正态分布", "泊松分布"), "normal_distribution_parameters"),
        (NORMAL_PARAMETERS.replace("D.中位数和标准差", ""), "normal_distribution_parameters"),
        (NORMAL_PARAMETERS.replace("D.中位数和标准差", "D.均值和标准差"), "normal_distribution_parameters"),
        (NORMAL_PARAMETERS.replace("A.均值和方差", "A.众数和方差"), "normal_distribution_parameters"),
        (DOMINO.replace("exactly", "at most"), "sparse_domino_placements"),
        (DOMINO.replace("without overlapping", "overlaps are allowed"), "sparse_domino_placements"),
        (DOMINO.replace("at least two", "at least one"), "sparse_domino_placements"),
        (DOMINO.replace("same row or column", "different rows and columns"), "sparse_domino_placements"),
        (DOMINO + " Return the result modulo 5.", "sparse_domino_placements"),
        (DOMINO.replace("$k^2$", "$k^2+1$"), "sparse_domino_placements"),
        (DOMINO + " Exactly ten dominoes must be horizontal.", "sparse_domino_placements"),
        (RED_BLUE.replace("8 blue", "9 blue").replace("15 points", "16 points"), "red_blue_line_separation"),
        (RED_BLUE.replace("no three", "no four"), "red_blue_line_separation"),
        (RED_BLUE.replace("not passing through", "possibly passing through"), "red_blue_line_separation"),
        (RED_BLUE.replace("every possible configuration", "some configuration"), "red_blue_line_separation"),
        (RED_BLUE + " Then add 1.", "red_blue_line_separation"),
        (RED_BLUE + " All drawn lines must be parallel.", "red_blue_line_separation"),
        (CLUSTERED.replace("any three", "any two"), "clustered_interval_maximum"),
        (CLUSTERED.replace("not equal to $1$", "equal to $1$"), "clustered_interval_maximum"),
        (CLUSTERED.replace("less than or equal to $a$", "strictly less than $a$"), "clustered_interval_maximum"),
        (CLUSTERED.replace("positive integers", "integers"), "clustered_interval_maximum"),
        (CLUSTERED + " Return the result modulo 7.", "clustered_interval_maximum"),
        (CLUSTERED + " Every element of X must be odd.", "clustered_interval_maximum"),
    ),
)
def test_accuracy_floor_routes_reject_changed_contracts(problem, operation):
    assert not _whole(problem, operation)


@pytest.mark.parametrize(
    ("problem", "operation", "result"),
    (
        (
            "边长为3的正六边形由平行于各边的直线划分成边长为1的等边三角形。"
            "求顶点均为这些等边三角形顶点的正六边形个数。",
            "triangular_lattice_regular_hexagons",
            "36",
        ),
        (
            "求所有偶整数d，使方程(x+2y-d)^2=xy的有序整数对(x,y)解的个数为偶数。",
            "even_quadratic_pair_count_parameters",
            r"d\in14\mathbb{Z}\setminus\{0\}",
        ),
        (
            r"求所有$f\in\mathbb{C}[x,y]$，使任意复数a,b满足"
            r"$f(a^2,b^2)=f\left(\frac{(a-b)^2}{2},\frac{(a+b)^2}{2}\right)$。",
            "quadratic_transform_invariant_polynomials",
            r"f(x,y)=g\!\left(x+y,xy(x-y)^2\right),\quad g\in\mathbb{C}[u,v]",
        ),
        (
            "The two parameters of a normal distribution are "
            "A. mean and variance B. mean and standard deviation "
            "C. median and variance D. median and standard deviation",
            "normal_distribution_parameters",
            "B",
        ),
        (
            "多米诺骨牌是2×1或1×2的骨牌。在2k×2k棋盘上恰好放置k^2个多米诺，"
            "互不重叠。每个2×2小方块至少有两个未覆盖的单位方格位于同一行或同一列。"
            "共有多少种放置方案？",
            "sparse_domino_placements",
            r"\binom{2k}{k}^2",
        ),
        (
            "平面上标有7个红点和8个蓝点，无三点共线。画若干条不经过这些点的直线，"
            "使每个区域不同时包含两种颜色。求对任意构型都可实现时直线条数k的最小值，"
            "共15个点。",
            "red_blue_line_separation",
            "7",
        ),
        (
            r"设S为平面上的有限点集。不存在5条直线覆盖S。对任意点X\in S，"
            r"删去该点后存在5条直线覆盖S-\{X\}。求|S|的最大值。",
            "critical_line_cover_point_set",
            "21",
        ),
        (
            r"设a为正整数且a大于等于3。正整数有限集X称为聚集的，如果任意三个元素中，"
            r"至少一对的最大公约数不等于1。若X最大元素与最小元素之差不超过a，求|X|最大值。",
            "clustered_interval_maximum",
            r"\left\lfloor\frac{a+2}{2}\right\rfloor+\left\lfloor\frac{a+2}{3}\right\rfloor-\left\lfloor\frac{a+2}{6}\right\rfloor",
        ),
    ),
)
def test_accuracy_floor_routes_accept_bilingual_equivalents(problem, operation, result):
    matched = _operation(problem, operation)

    assert len(matched) == 1
    assert matched[0].result == result
    assert _whole(problem, operation)
