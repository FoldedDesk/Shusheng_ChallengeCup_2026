from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


DIGIT_PROBLEM = r"""Determine the number of natural numbers $n$ that that has at most 16 digits satisfying the following conditions:
i) $3|n.$
ii) The digits of $n$ in decimal representation are in the set $\{2,0,1,8\}$."""

FLOOR_PROBLEM = r"""Let $p$ be a prime greater than $100$. Find the $9$th largest positive integer $n$ less than $p$ such that
\[
    nk + k \ge p \left\lfloor \frac{nk + n}{p} \right\rfloor
\]
for all $k = 0, 1, \ldots, p - 2$."""

FUNCTION_PROBLEM = (
    r"Find all functions $A:\mathbb{R}\rightarrow\mathbb{R}$ such that "
    r"$A(p)A(q)+A(-pq)=A(p+q)+2pq+1$ holds for all real numbers $p$ and $q$."
)

NICE_FUNCTION_PROBLEM = (
    'A function $C$ from the set of positive integers to itself is called "nice" if for all '
    r"positive integers $a, b$, $C(a+b) - C(a) - C(C(b)) + 1 \ge 0$. "
    r"Find all possible values of $C(1234)$ for a nice function "
    r"$C: \mathbb{N} \rightarrow \mathbb{N}$."
)

OPEN_INTERVAL_PROBLEM = r"""Find the smallest positive integer $n$ such that there exist real numbers $x_1, \ldots, x_n$ between $-1$ and 1 satisfying
\[
\sum_{i=1}^n x_i^2 + \left(\sum_{i=1}^n x_i\right)^2 = 20, \quad |x_1 + \ldots + x_n| < 1.
\]"""

SUBSET_CARD_PROBLEM = (
    "A card deck consists of 1024 cards. On each card, a set of distinct decimal digits is written "
    "in such a way that no two of these sets coincide, including an empty card. Two players "
    "alternately take cards from the deck, one card per turn. After the deck is empty, each player "
    "checks if he can throw out one of his cards so that each of the ten digits occurs on an even "
    "number of his remaining cards. If one player can do this but the other one cannot, the one who "
    "can is the winner; otherwise, a draw is declared. Determine all possible first moves of the "
    "first player after which the opponent has a winning strategy."
)

CIRCLE_PARAMETER_PROBLEM = r"""Let $k$ be a positive real number. Triangle XYZ is acute and scalene, O is its circumcenter and XD, YE, ZF are the internal bisectors. On the rays XD, YE, ZF, respectively, let points P, Q, R such that $\frac{XP}{XD} = \frac{YQ}{YE} = \frac{ZR}{ZF} = k$. Denote $(C_1), (C_2), (C_3)$ be respectively the circle through P and touches OX at X, the circle through Q and touches OY at Y, the circle through R and touches OZ at Z. Find all values of k such that three circles $(C_1), (C_2), (C_3)$ have exactly two common points."""

ODD_PART_PROBLEM = r"""For a given positive integer $n$, let $m$ be the exponent of 2 in the prime factorization of $n$. Define $f(n) = \frac{n}{2^m}$. Find all positive integers $u$ for which there exists a positive integer $v$ such that
(Condition) $f(u+v) - f(u), f(u+v+1) - f(u+1), \cdots, f(u+2v-1) - f(u+v-1)$ are all multiples of 4."""

MUTUAL_HISTOGRAM_PROBLEM = r"""A sequence of integers $a_0, \ldots, a_{1000}$ is called a good sequence if there exists a sequence of integers $b_0, \ldots, b_{1000}$ such that
\[
\prod_{k=0}^{1000} (x-a_k)=\prod_{k=0}^{1000}(x-k)^{b_k},\quad
\prod_{k=0}^{1000} (x-b_k)=\prod_{k=0}^{1000}(x-k)^{a_k}
\]
for all $x$. Find all the possible values of $\sum_{i=0}^{1000}(i+1)a_i^2$ for good sequences $a_0, \ldots, a_{1000}$."""

SIGNED_SUBSEQUENCE_PROBLEM = r"""A $\pm 1$-sequence is a sequence of 2022 numbers $a_1, \ldots, a_{2022}$, each equal to either +1 or -1. Determine the largest $C$ so that, for any $\pm 1$-sequence, there exists an integer $k$ and indices $1 \leqslant t_1<\ldots<t_k \leqslant 2022$ so that $t_{i+1}-t_i \leqslant 2$ for all $i$, and
\[
\left|\sum_{i=1}^{k}a_{t_{i}}\right|\geqslant C.
\]"""

MERCHANT_PROBLEM = r"""In a marketplace, $7396$ stalls are arranged in a straight line. Each of two merchants sells $k$ distinct items numbered from 1 to $k$; each item is sold at a lower-numbered stall and bought at a higher-numbered stall. For each merchant, and for any $i$ and $j$ with $1 \leqslant i<j \leqslant k$, the stall where item $j$ is sold is higher than the stall where item $i$ is sold; similarly, the stall where item $j$ is bought is higher than the stall where item $i$ is bought. Say that two stalls are connected by some merchant if one can start from the lower-numbered stall and reach the higher-numbered stall by buying and selling one or more items from that merchant. Determine the smallest $k$ for which one can guarantee that there are two stalls that are connected by both merchants."""

POLYOMINO_COLOR_PROBLEM = r"""A polyomino is a figure which consists of unit squares joined together by their sides. Consider a grid of unit square cells which extends to infinity in all directions. Find the greatest positive integer $C$ which satisfies the following condition: For every colouring of the cells of the grid in $36$ colours, there is some polyomino within the grid which contains at most $35$ colours and whose area is at least $C$."""

KOREAN_SEQUENCE_PROBLEM = r"""A sequence of positive integers $a_1, a_2, \ldots, a_n$ is called a Korean sequence if $a_1<a_2<\ldots<a_n$. For each $1 \leq k<n$, let $A_k=\{a_1,\ldots,a_k\}$ and $B_k=\{a_{k+1},\ldots,a_n\}$. A partition is good if the least common multiple of the elements in $A_k$ is equal to the greatest common divisor of the elements in $B_k$. Determine the minimum value of $n$ such that there exists a Korean sequence of length $n$ with exactly $2015$ good partitions."""

FLIP_PROBLEM = r"""In a research lab, scientists are studying bacteria on a $64 \times 64$ square petri dish. The dish is divided into small square sections, each of which is a $1 \times 1$ square and is either infected or sterile. Initially, there are exactly $k$ infected sections, and the rest are sterile. The bacteria spread according to two rules:
1) If a $2 \times 2$ square section has exactly three infected sections, the last sterile section gets infected.
2) If a $2 \times 2$ square has exactly two infected sections, infected sections become sterile, and sterile sections become infected.
Determine the smallest number of initially infected sections $k$ such that, no matter how the infection starts, the entire dish can be infected after a sequence of operations."""

BEZOUT_PROBLEM = r"""Let $k>l$ be given coprime positive integers greater than 1. Define a function $f: \mathbb{Z}\rightarrow \mathbb{Z}$ as follows: for $x$, $f(x)$ is the smallest value of $|a|+|b|$ among all integers $a,b$ satisfying $ka+lb = x$. An integer $x$ is called 'nice' if $f(x)\geq \max (f(x-a),f(x+a),f(x-b),f(x+b))$. Denote by $F(k,l)$ the number of nice integers when both $k$ and $l$ are odd, and denote by $G(k,l)$ the number of nice integers when either $k$ or $l$ is even. Suppose that there exist polynomials $p(k,l)$ and $q(k,l)$ such that $F(k,l)=p(k,l)$ for all odd integers $k,l$ and $G(k,l)=q(k,l)$ whenever at least one of $k$ or $l$ is even. Evaluate $p(k,l)^2 + q(k,l)^2$."""

CYCLIC_QUARTIC_PROBLEM = r"""Find number of triples $(x,y,z)$ of real numbers satisfying
\[x^2+y^2+z^2=xy^3+yz^3+zx^3=3.\]"""


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified route called the model: {kwargs}")


def _matching(problem: str, operation: str):
    spec = build_problem_spec(problem)
    return [
        item
        for item in SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)
        if item.operation == operation
    ]


@pytest.mark.parametrize(
    "problem, expected",
    [
        (DIGIT_PROBLEM, "1431655765"),
        (DIGIT_PROBLEM.replace("natural numbers", "positive integers"), "1431655764"),
        (
            "求至多4位的自然数$n$的个数，其中(1) $7|n.$ "
            r"(2) $n$的十进制表示中每一位数字都属于集合$\{0,1,2\}$。",
            str(sum(value % 7 == 0 and set(str(value)) <= set("012") for value in range(2223))),
        ),
    ],
)
def test_bounded_digit_set_route_is_exact_certified_and_bilingual(problem, expected):
    evidence = _matching(problem, "bounded_digit_set_divisibility_count")

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"
    assert {
        "modular_remainder_dynamic_programming",
        "state_mass_invariant",
    } <= set(evidence[0].certificate_checks)


@pytest.mark.parametrize(
    "problem",
    [
        DIGIT_PROBLEM.replace("at most 16", "exactly 16"),
        DIGIT_PROBLEM.replace("The digits of", "The distinct digits of"),
        DIGIT_PROBLEM.replace("ii)", "ii) $n$ is even. iii)"),
        DIGIT_PROBLEM.replace("decimal representation", "base-8 representation"),
        DIGIT_PROBLEM.replace("Determine the number", "Determine the sum"),
        DIGIT_PROBLEM.replace(r"3|n", r"3\nmid n"),
    ],
)
def test_bounded_digit_set_route_rejects_changed_contract(problem):
    assert not _matching(problem, "bounded_digit_set_divisibility_count")


def test_prime_floor_rank_route_has_symbolic_certificate_and_bypasses_model():
    evidence = _matching(FLOOR_PROBLEM, "prime_floor_inequality_rank")

    assert len(evidence) == 1
    assert evidence[0].result == r"\left\lfloor\frac{p}{9}\right\rfloor"
    assert evidence[0].scope == "whole_goal"
    assert "floor_quotient_set_characterization" in evidence[0].certificate_checks

    result = ReasoningAgent(_NoModelClient()).solve(
        FLOOR_PROBLEM + r" Remember to put your final answer within \boxed{}.", {}
    )
    assert result["final_response"] == r"\boxed{\lfloor\frac{p}{9}\rfloor}"
    assert not any(step["step"].startswith("model_call_") for step in result["trace"])


@pytest.mark.parametrize(
    "problem",
    [
        FLOOR_PROBLEM.replace("a prime", "an integer"),
        FLOOR_PROBLEM.replace("greater than $100$", "greater than $20$"),
        FLOOR_PROBLEM.replace("for all", "for some"),
        FLOOR_PROBLEM.replace("p - 2", "p - 3"),
        FLOOR_PROBLEM.replace(r"\ge", ">"),
        FLOOR_PROBLEM.replace("9$th largest", "smallest"),
        FLOOR_PROBLEM.replace("positive integer $n$", "positive prime $n$"),
        FLOOR_PROBLEM.replace("greater than $100$", "greater than $1$").replace(
            "$9$th largest", "$2$nd largest"
        ),
    ],
)
def test_prime_floor_rank_route_rejects_changed_contract(problem):
    assert not _matching(problem, "prime_floor_inequality_rank")


@pytest.mark.parametrize(
    "problem",
    [
        FUNCTION_PROBLEM,
        (
            r"求所有函数$A:\mathbb{R}\rightarrow\mathbb{R}$，使得对任意实数$p,q$均有"
            r"$A(p)A(q)+A(-pq)=A(p+q)+2pq+1$。"
        ),
    ],
)
def test_real_functional_equation_route_returns_all_three_branches(problem):
    evidence = _matching(problem, "real_functional_equation_three_solutions")

    assert len(evidence) == 1
    assert evidence[0].result == r"A(x)=1-x,\ A(x)=1+2x,\ A(x)=1-x^2"
    assert evidence[0].scope == "whole_goal"
    assert "all_function_branches_exhausted" in evidence[0].certificate_checks


@pytest.mark.parametrize(
    "problem",
    [
        FUNCTION_PROBLEM.replace(r"\mathbb{R}", r"\mathbb{Q}", 1),
        FUNCTION_PROBLEM.replace("2pq+1", "2pq-1"),
        FUNCTION_PROBLEM.replace("for all real numbers", "for all positive real numbers"),
        FUNCTION_PROBLEM.replace("Find all functions", "Find one function"),
        FUNCTION_PROBLEM + " Also find A(10).",
    ],
)
def test_real_functional_equation_route_rejects_changed_contract(problem):
    assert not _matching(problem, "real_functional_equation_three_solutions")


def test_nice_positive_integer_function_route_returns_complete_parameterized_range():
    evidence = _matching(NICE_FUNCTION_PROBLEM, "nice_positive_integer_function_value_set")

    assert len(evidence) == 1
    assert evidence[0].result == r"\{1,2,\ldots,1235\}"
    assert evidence[0].scope == "whole_goal"
    assert {
        "growth_bootstrap_upper_bound",
        "all_values_have_explicit_constructions",
    } <= set(evidence[0].certificate_checks)

    small = NICE_FUNCTION_PROBLEM.replace("1234", "7")
    assert _matching(small, "nice_positive_integer_function_value_set")[0].result == r"\{1,2,\ldots,8\}"


@pytest.mark.parametrize(
    "problem",
    [
        NICE_FUNCTION_PROBLEM.replace("positive integers to itself", "nonnegative integers to itself"),
        NICE_FUNCTION_PROBLEM.replace("C(C(b))", "C(b)"),
        NICE_FUNCTION_PROBLEM.replace(r"\ge 0", r"\le 0"),
        NICE_FUNCTION_PROBLEM.replace("for all positive integers", "for some positive integers"),
        NICE_FUNCTION_PROBLEM.replace("all possible values", "the maximum value"),
        NICE_FUNCTION_PROBLEM + " Assume also that C is injective.",
    ],
)
def test_nice_positive_integer_function_route_rejects_changed_contract(problem):
    assert not _matching(problem, "nice_positive_integer_function_value_set")


def test_real_functional_equation_route_rejects_collapsed_quantifier_variables():
    collapsed = FUNCTION_PROBLEM.replace("$p$ and $q$", "$p$ and $p$")
    assert not _matching(collapsed, "real_functional_equation_three_solutions")


@pytest.mark.parametrize(
    "problem, operation, expected",
    [
        (OPEN_INTERVAL_PROBLEM, "open_interval_quadratic_minimum_dimension", "21"),
        (
            SUBSET_CARD_PROBLEM,
            "subset_xor_card_game_losing_first_move",
            r"\text{taking the empty card}",
        ),
        (
            CIRCLE_PARAMETER_PROBLEM,
            "angle_bisector_three_circle_parameter",
            r"\left\{\frac12,1\right\}",
        ),
        (ODD_PART_PROBLEM, "odd_part_block_congruence_values", r"\{1,3,5\}"),
    ],
)
def test_exact_theorem_routes_cover_the_whole_goal(problem, operation, expected):
    evidence = _matching(problem, operation)

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].verified
    assert evidence[0].scope == "whole_goal"


@pytest.mark.parametrize(
    "problem, operation",
    [
        (
            OPEN_INTERVAL_PROBLEM.replace("between $-1$ and 1", "between $-1$ and 1 inclusive"),
            "open_interval_quadratic_minimum_dimension",
        ),
        (
            OPEN_INTERVAL_PROBLEM.replace("= 20", "= 21"),
            "open_interval_quadratic_minimum_dimension",
        ),
        (
            OPEN_INTERVAL_PROBLEM.replace(
                r"\sum_{i=1}^n x_i^2",
                r"2\sum_{i=1}^n x_i^2",
                1,
            ),
            "open_interval_quadratic_minimum_dimension",
        ),
        (
            SUBSET_CARD_PROBLEM.replace("1024 cards", "1023 cards"),
            "subset_xor_card_game_losing_first_move",
        ),
        (
            SUBSET_CARD_PROBLEM.replace("even number", "odd number"),
            "subset_xor_card_game_losing_first_move",
        ),
        (
            CIRCLE_PARAMETER_PROBLEM.replace("acute and scalene", "right and scalene"),
            "angle_bisector_three_circle_parameter",
        ),
        (
            CIRCLE_PARAMETER_PROBLEM.replace("exactly two", "exactly one"),
            "angle_bisector_three_circle_parameter",
        ),
        (
            CIRCLE_PARAMETER_PROBLEM.replace(
                "On the rays XD, YE, ZF, respectively, let points P, Q, R",
                "In the interiors of segments XD, YE, ZF, respectively, let points P, Q, R",
            ),
            "angle_bisector_three_circle_parameter",
        ),
        (
            ODD_PART_PROBLEM.replace("multiples of 4", "multiples of 8"),
            "odd_part_block_congruence_values",
        ),
        (
            ODD_PART_PROBLEM.replace("there exists a positive integer $v$", "for every positive integer $v$"),
            "odd_part_block_congruence_values",
        ),
    ],
)
def test_exact_theorem_routes_reject_changed_contracts(problem, operation):
    assert not _matching(problem, operation)


@pytest.mark.parametrize(
    "problem, operation, expected",
    [
        (
            MUTUAL_HISTOGRAM_PROBLEM,
            "mutual_histogram_weighted_values",
            r"\{995018,995026,997008\}",
        ),
        (SIGNED_SUBSEQUENCE_PROBLEM, "gap_two_signed_subsequence_guarantee", "506"),
        (MERCHANT_PROBLEM, "two_monotone_merchant_common_connection", "7311"),
        (POLYOMINO_COLOR_PROBLEM, "missing_color_polyomino_area", "2450"),
        (KOREAN_SEQUENCE_PROBLEM, "korean_sequence_good_partition_minimum", "3024"),
    ],
)
def test_parameterized_olympiad_routes_are_certified(problem, operation, expected):
    evidence = _matching(problem, operation)

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].verified


def test_two_color_polyomino_boundary_uses_checkerboard_sharp_value():
    problem = POLYOMINO_COLOR_PROBLEM.replace("$36$", "$2$").replace("$35$", "$1$")
    evidence = _matching(problem, "missing_color_polyomino_area")

    assert len(evidence) == 1
    assert evidence[0].result == "1"
    assert evidence[0].verified


@pytest.mark.parametrize(
    "problem, operation",
    [
        (
            MUTUAL_HISTOGRAM_PROBLEM.replace("(i+1)a_i^2", "(i+1)a_i^3"),
            "mutual_histogram_weighted_values",
        ),
        (
            MUTUAL_HISTOGRAM_PROBLEM.replace("b_{1000}", "b_{999}", 1),
            "mutual_histogram_weighted_values",
        ),
        (
            SIGNED_SUBSEQUENCE_PROBLEM.replace(r"\leqslant 2", r"\leqslant 3"),
            "gap_two_signed_subsequence_guarantee",
        ),
        (
            MERCHANT_PROBLEM.replace("Each of two merchants", "Each of three merchants"),
            "two_monotone_merchant_common_connection",
        ),
        (
            MERCHANT_PROBLEM.replace("7396", "7395"),
            "two_monotone_merchant_common_connection",
        ),
        (
            POLYOMINO_COLOR_PROBLEM.replace("at most $35$", "at most $34$"),
            "missing_color_polyomino_area",
        ),
        (
            KOREAN_SEQUENCE_PROBLEM.replace("exactly $2015$", "at least $2015$"),
            "korean_sequence_good_partition_minimum",
        ),
    ],
)
def test_parameterized_olympiad_routes_reject_changed_contracts(problem, operation):
    assert not _matching(problem, operation)


@pytest.mark.parametrize(
    "problem, operation, expected",
    [
        (FLIP_PROBLEM, "two_by_two_flip_closure_minimum", "1057"),
        (BEZOUT_PROBLEM, "bezout_l1_nice_count_polynomial", r"5(l-1)^2"),
        (CYCLIC_QUARTIC_PROBLEM, "cyclic_quartic_equality_triple_count", "8"),
    ],
)
def test_strict_exact_matchers_keep_original_contracts(problem, operation, expected):
    evidence = _matching(problem, operation)

    assert len(evidence) == 1
    assert evidence[0].result == expected
    assert evidence[0].verified


@pytest.mark.parametrize(
    "problem, operation",
    [
        (
            FLIP_PROBLEM.replace(
                "infected sections become sterile, and sterile sections become infected",
                "infected sections remain infected, while the sterile sections, being sterile, become infected",
            ),
            "two_by_two_flip_closure_minimum",
        ),
        (
            BEZOUT_PROBLEM.replace(
                "f(x-a),f(x+a),f(x-b),f(x+b)",
                "f(x),f(x),f(x),f(x)",
            ),
            "bezout_l1_nice_count_polynomial",
        ),
    ],
)
def test_strict_exact_matchers_reject_rule_mutations(problem, operation):
    assert not _matching(problem, operation)


def test_submission_reminder_does_not_hide_a_later_constraint():
    reminder = r"Remember to put your final answer within \boxed{}."
    unchanged = _matching(
        CYCLIC_QUARTIC_PROBLEM + "\n" + reminder,
        "cyclic_quartic_equality_triple_count",
    )
    changed = _matching(
        CYCLIC_QUARTIC_PROBLEM
        + "\n"
        + reminder
        + " Additionally require $x=y=z=0$.",
        "cyclic_quartic_equality_triple_count",
    )

    assert len(unchanged) == 1
    assert not changed


def test_exact_tool_has_no_greedy_submission_reminder_removal():
    source = (
        Path(__file__).resolve().parents[1] / "tools" / "exact_olympiad_tool.py"
    ).read_text(encoding="utf-8")

    assert r"final\s+answer[\s\S]*$" not in source
