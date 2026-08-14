from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


REMINDER = r"Remember to put your final answer within \boxed{}."

CUBIC = r"""Let $p, q, r, s$ be constants such that the equation $py^3 + qy^2 + ry + s = 0$ has three distinct real roots. Find all possible values for the number of distinct real roots of the equation
$$\left(pz^{3}+qz^{2}+rz+s\right)(6pz+2q)=\left(3pz^{2}+2qz+r\right)^{2}.$$"""

NAPKIN = r"""On a large chessboard of 2011 by 2011 squares, a finite number of square tiles are placed. Each tile covers a square area of 52 by 52 cells. In each cell, the number of tiles covering it is written, and the maximum number $k$ of cells containing the same nonzero number is recorded. Considering all possible tile configurations, what is the largest possible value of $k$?"""

GAME = r"""Two players, Alice and Bob, play a game in which they take turns choosing positive integers less than or equal to a positive integer $n$. The rules of the game are:

(i) A player cannot choose a number that has been chosen by either player on any previous turn.

(ii) A player cannot choose a number consecutive to any of those the player has already chosen on any previous turn.

(iii) The game is a draw if all numbers have been chosen; otherwise the player who cannot choose a number anymore loses the game.

Alice takes the first turn. Find the largest value of $n$ such that the game ends in a draw."""

HOTEL = r"""A sports tournament is being organized for $256$ players. Every pair of players must play exactly one match against each other. The tournament is scheduled such that each day only one match is played. Each player arrives on the day of their first match and departs on the day of their last match. For each day a player is present at the tournament, the organizers must pay 1 coin to the hotel. The organizers want to minimize the total cost of all players' stays by designing an optimal schedule. Additionally, there is a VIP lounge where special guests can watch the matches for free. The VIP lounge has limited capacity and can only accommodate a maximum of 10 people at any given time. However, the presence of the VIP lounge and the special guests does not affect the scheduling of the matches or the total cost of the players' stays. Determine the minimum total cost the organizers must pay for all players' hotel stays."""

FIBONACCI = r"""The Lucas numbers $L_{0}, L_{1}, L_{2}, \ldots$ are defined inductively by $L_{0}=2, L_{1}=1$, and $L_{n+1}=L_{n}+L_{n-1}$ for $n \geqslant 1$. The Fibonacci numbers $F_{0}, F_{1}, F_{2}, \ldots$ are defined inductively by $F_{0}=0, F_{1}=1$, and $F_{n+1}=F_{n}+F_{n-1}$ for $n \geqslant 1$. Determine the smallest size of a set $S$ of integers such that for every $k=2,3, \ldots, 125$ there exist some $x, y \in S$ such that $x-y=F_{k}$. Also, there exist some $a, b \in T$ for some set $T$ such that $a-b = L_{100}$."""

TRANSLATION = r"""Let $\mathbb{Z}_{\geqslant 0}$ be the set of non-negative integers, and let $f: \mathbb{Z}_{\geqslant 0} \times \mathbb{Z}_{\geqslant 0} \rightarrow \mathbb{Z}_{\geqslant 0}$ be a bijection such that whenever $f\left(x_{1}, y_{1}\right)>f\left(x_{2}, y_{2}\right)$, we have $f\left(x_{1}+1, y_{1}\right)>f\left(x_{2}+1, y_{2}\right)$ and $f\left(x_{1}, y_{1}+1\right)>f\left(x_{2}, y_{2}+1\right)$. Also, let $g: \mathbb{Z}_{\geqslant 0} \rightarrow \mathbb{Z}_{\geqslant 0}$ be a function such that $g(n) = n^2 - n + 1$.

Let $N$ be the number of pairs of integers $(x, y)$, with $0 \leqslant x, y<100$, such that $f(x, y)$ is odd. Let the smallest and largest possible value of $N$ be $a,b$, find the product $ab$."""

SPARSE = r"""Let $s$ be positive integers such that $s<5625$. Initially, one cell out of an $n \times n$ grid is coloured green. On each turn, we pick some green cell $c$ and colour green some $s$ out of the $5625$ cells in the $75 \times 75$ square centred at $c$. No cell may be coloured green twice. We say that $s$ is sparse if there exists some positive number $C$ such that, for every positive integer $n$, the total number of green cells after any number of turns is always going to be at most $Cn$. Find the least sparse integer $s$."""

TRIANGLE = r"""Given right triangle $ XYZ$ with hypothenuse $ XZ$ and $ \angle X = 50^{\circ}$. Points $ P$ and $ Q$ on the side $ YZ$ are such that $ \angle PXZ = \angle QXY = 10^{\circ}$. Compute the ratio $2 \times YQ/ZP$."""

QUADRILATERAL = r"""Let $PQRS$ be a convex quadrilateral with perimeter $3$ and $PR=QS=1$. Determine the maximum possible area of $PQRS$."""

SEMIGROUP = r"""For a positive integer $n \geq 2$, let the set $C_n$ be the set of integers $2^n - 2^i$ for integers $i$ such that $0 \leq i < n$. Find the smallest positive integer that cannot be expressed as a sum of numbers in $C_n$ (where the same number can be used multiple times)."""

RECURRENCE = r"""For the integer sequence $(a_n)$ defined by $a_1=10$ and $a_{n+1}=6a_n - 2^{n+2} - 3^{n+1} +5$, find all positive numbers that are relatively prime to every number in $(a_n)$."""

FIELD = r"""$x^4+5\in\mathbb{Q}[x]$在$\mathbb{Q}$上的分裂域(记为$E$)是$(\quad)$.
$[E:\mathbb{Q}]=(\quad)$.
$E/\mathbb{Q}$ $(\quad)$(填“是”或“否”.)为Galois扩张."""

FLOOD = r"""Let $\gamma \geq 1$ be a real number. Sun Wukong and the Sea God play a turn-based game on an infinite grid of unit squares. Before the game starts, the Sea God chooses a finite number of cells to be flooded with seawater. Sun Wukong is building a magical barrier, which is a subset of unit edges of the grid (called walls) forming a connected, non-self-intersecting path or loop. Additionally, there is a magical artifact that randomly generates a finite number of extra walls on the grid, with no specific pattern or distribution.

The game then begins with Sun Wukong moving first. On each of Sun Wukong's turns, he adds one or more walls to the magical barrier, as long as the total length of the barrier is at most $\gamma n$ after his $n$th turn. On each of the Sea God's turns, every cell which is adjacent to an already flooded cell and with no wall between them becomes flooded as well. Sun Wukong wins if the magical barrier forms a closed loop such that all flooded cells are contained in the interior of the loop — hence stopping the flood and saving the world. What is the largest constant $C$ such that for all $\gamma > C$ can Sun Wukong guarantee victory in a finite number of turns no matter how the Sea God chooses the initial cells to flood?"""

PRIME_EXPONENTIAL = r"""Find all pairs $(a, b)$ of positive real numbers such that for every prime number $p$ and real number $x$ satisfying
\[
    2^{2^{p + 1}x} = 2^px + 1,
\]
we have
\[
    \frac{a^x + b^x + 1}{3} \ge x + 1.
\]"""


def _matching(problem: str, operation: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == operation
    ]


@pytest.mark.parametrize(
    "problem, operation, expected",
    [
        (CUBIC, "cubic_log_derivative_real_root_count", "0"),
        (NAPKIN, "napkin_equal_coverage_maximum", "3986729"),
        (GAME, "personal_consecutive_number_game", "6"),
        (HOTEL, "round_robin_hotel_cost_minimum", "4202432"),
        (FIBONACCI, "fibonacci_difference_basis_minimum", "64"),
        (TRANSLATION, "translation_order_odd_count_product", "18750000"),
        (SPARSE, "sparse_green_neighborhood_threshold", "4181"),
        (TRIANGLE, "right_triangle_two_cevian_ratio", "1"),
        (QUADRILATERAL, "equal_diagonal_quadrilateral_maximum_area", r"\frac{1}{2}"),
        (SEMIGROUP, "power_difference_semigroup_smallest_gap", "1"),
        (RECURRENCE, "recurrence_universal_coprime_set", r"\{1\}"),
        (
            PRIME_EXPONENTIAL,
            "prime_exponential_inequality_parameter_region",
            r"\{(a,b)\in\mathbb{R}_{>0}^2:ab\le e^3\}",
        ),
        (FLOOD, "flood_barrier_critical_speed", "2"),
        (
            FIELD,
            "quartic_plus_five_splitting_field",
            r"E=\mathbb{Q}(\sqrt[4]{-5},i),\ [E:\mathbb{Q}]=8,\ \text{是}",
        ),
    ],
)
def test_new_certified_routes_cover_complete_goals(problem, operation, expected):
    evidence = _matching(problem + "\n" + REMINDER, operation)

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize(
    "problem, operation",
    [
        (CUBIC.replace("three distinct real roots", "three real roots"), "cubic_log_derivative_real_root_count"),
        (NAPKIN.replace("nonzero number", "number"), "napkin_equal_coverage_maximum"),
        (GAME.replace("the player has already chosen", "either player has already chosen"), "personal_consecutive_number_game"),
        (HOTEL.replace("only one match", "two matches"), "round_robin_hotel_cost_minimum"),
        (FIBONACCI.replace("x-y=F_{k}", "x+y=F_{k}"), "fibonacci_difference_basis_minimum"),
        (TRANSLATION.replace(">f\\left(x_{2}+1", "<f\\left(x_{2}+1"), "translation_order_odd_count_product"),
        (SPARSE.replace("No cell may be coloured green twice", "A cell may be coloured green twice"), "sparse_green_neighborhood_threshold"),
        (TRIANGLE.replace("YQ/ZP", "ZP/YQ"), "right_triangle_two_cevian_ratio"),
        (QUADRILATERAL.replace("convex", "nonconvex"), "equal_diagonal_quadrilateral_maximum_area"),
        (SEMIGROUP.replace("smallest", "largest"), "power_difference_semigroup_smallest_gap"),
        (RECURRENCE.replace("+5", "-5"), "recurrence_universal_coprime_set"),
        (
            PRIME_EXPONENTIAL + " Assume additionally that $a=b$.",
            "prime_exponential_inequality_parameter_region",
        ),
        (
            PRIME_EXPONENTIAL.replace(r"\ge x + 1", r"\le x + 1"),
            "prime_exponential_inequality_parameter_region",
        ),
        (FLOOD.replace("total length", "new length"), "flood_barrier_critical_speed"),
        (FIELD.replace("x^4+5", "x^4-5"), "quartic_plus_five_splitting_field"),
    ],
)
def test_new_certified_routes_reject_changed_mathematics(problem, operation):
    assert not _matching(problem, operation)


def test_parameterized_theorem_families_recompute_instead_of_memorizing_numbers():
    hotel = HOTEL.replace("$256$ players", "$8$ players")
    assert _matching(hotel, "round_robin_hotel_cost_minimum")[0].result == "134"

    fibonacci = FIBONACCI.replace("\\ldots, 125$", "\\ldots, 9$")
    assert _matching(fibonacci, "fibonacci_difference_basis_minimum")[0].result == "6"

    translation = TRANSLATION.replace("y<100$", "y<20$")
    assert _matching(translation, "translation_order_odd_count_product")[0].result == "30000"

    sparse = SPARSE.replace("5625", "49").replace("75 \\times 75", "7 \\times 7")
    assert _matching(sparse, "sparse_green_neighborhood_threshold")[0].result == "33"


@pytest.mark.parametrize(
    "problem, subject, method_fragment",
    [
        (
            "Two contestants take turns selecting unused vertices of a finite path; a player unable to move loses. Determine the winning strategy.",
            "离散数学",
            "minimax",
        ),
        (
            r"求多项式 $x^5-2$ 在有理数域上的 splitting field、扩张次数以及是否为 Galois 扩张。",
            "抽象代数",
            "field_tower",
        ),
        (
            r"计算 $e^{-|x|}$ 的 Fourier 变换，并说明所采用的归一化。",
            "数学分析",
            "fourier",
        ),
        (
            "The Fibonacci numbers satisfy their usual recurrence. Determine the smallest size of an integer set realizing the first n values as differences.",
            "离散数学",
            "recurrence",
        ),
        (
            "Square tiles are placed on an N by N chessboard. Maximize the cells having the same nonzero covering multiplicity.",
            "离散数学",
            "coverage_multiplicity",
        ),
        (
            "In a tournament each pair plays once. Each player arrives for the first match and departs after the last; minimize the hotel stay cost.",
            "离散数学",
            "arrival_departure",
        ),
        (
            "A bijection of the nonnegative integer lattice preserves strict order after either coordinate translation. Bound the odd images in a square.",
            "离散数学",
            "translation_invariant_order",
        ),
        (
            "Green cells spread through a finite grid neighborhood each turn. Find the sharp threshold between linear and quadratic growth.",
            "离散数学",
            "isoperimetric_boundary",
        ),
    ],
)
def test_unseen_paraphrases_receive_domain_specific_methods(problem, subject, method_fragment):
    spec = build_problem_spec(problem)

    assert spec.profile.primary_subject == subject
    assert method_fragment in spec.primary_method


def test_submission_suffix_does_not_hide_an_added_constraint():
    changed = GAME + "\n" + REMINDER + " Bob must choose an even number first."
    assert not _matching(changed, "personal_consecutive_number_game")


@pytest.mark.parametrize(
    "problem",
    [
        (
            "Start with integer-valued 17-tuples and repeatedly add or take coordinatewise maxima. "
            "Determine how many generators suffice to produce every integer-valued 23-tuple."
        ),
        (
            "Two players choose one of two operations optimally. Additionally a random coin forces "
            "the second player to use one operation, although that player wants to minimize the score."
        ),
        (
            "Minimize the players' hotel cost. Additionally, a VIP lounge is available, but its "
            "presence does not affect the schedule or total cost."
        ),
        (
            "Find the largest table satisfying two conditions. In addition to the given constraints, "
            "a randomly selected column is forced to contain a broken marker."
        ),
    ],
)
def test_semantically_suspicious_statements_require_integrity_audit(problem):
    spec = build_problem_spec(problem)

    assert "statement_integrity_audit" in spec.risk_flags
    checklist = SubmissionAgent._audit_checklist(spec, True)
    assert "smallest coherent repair" in checklist


def test_ordinary_coherent_additional_condition_remains_authoritative():
    spec = build_problem_spec(
        "Find the maximum of x+y subject to x^2+y^2=1. Additionally, x and y are positive."
    )

    assert "statement_integrity_audit" not in spec.risk_flags
