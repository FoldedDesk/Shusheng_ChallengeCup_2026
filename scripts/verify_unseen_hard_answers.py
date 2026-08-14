"""Deterministically verify the answer key for the isolated hard holdout set.

This module is an offline authoring check.  It is never imported by the agent
and its results are never included in a model request.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v1.jsonl"


def _load_answers(path: Path) -> dict[int, str]:
    return {
        int(row["idx"]): str(row["answer"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _regular_binary_matrices(size: int) -> int:
    masks = [sum(1 << column for column in pair) for pair in combinations(range(size), 2)]
    states: dict[tuple[int, ...], int] = {(0,) * size: 1}
    for _ in range(size):
        next_states: dict[tuple[int, ...], int] = defaultdict(int)
        for state, count in states.items():
            for mask in masks:
                candidate = tuple(
                    state[column] + ((mask >> column) & 1)
                    for column in range(size)
                )
                if max(candidate) <= 2:
                    next_states[candidate] += count
        states = next_states
    return states.get((2,) * size, 0)


def _unicyclic_graphs(vertices: int) -> int:
    total = 0
    for cycle_size in range(3, vertices + 1):
        cycles = math.comb(vertices, cycle_size) * math.factorial(cycle_size - 1) // 2
        forests = (
            1
            if cycle_size == vertices
            else cycle_size * vertices ** (vertices - cycle_size - 1)
        )
        total += cycles * forests
    return total


def _vieta_pairs(bound: int) -> set[tuple[int, int]]:
    brute = {
        (x, y)
        for x in range(1, bound + 1)
        for y in range(1, bound + 1)
        if x * x + y * y + 1 == 3 * x * y
    }
    recurrence: set[tuple[int, int]] = set()
    left = right = 1
    while left <= bound and right <= bound:
        recurrence.add((left, right))
        recurrence.add((right, left))
        left, right = right, 3 * right - left
    assert brute == recurrence
    return brute


def _competing_pattern_probability(first: str, second: str) -> sp.Rational:
    prefixes = {""}
    for word in (first, second):
        prefixes.update(word[:index] for index in range(1, len(word)))
    ordered = sorted(prefixes, key=lambda value: (len(value), value))
    variables = sp.symbols(f"p0:{len(ordered)}")
    equations = []
    for state, variable in zip(ordered, variables):
        rhs = 0
        for outcome in "HT":
            combined = state + outcome
            if combined.endswith(first):
                value = sp.Integer(1)
            elif combined.endswith(second):
                value = sp.Integer(0)
            else:
                suffix = max(
                    (prefix for prefix in ordered if combined.endswith(prefix)),
                    key=len,
                )
                value = variables[ordered.index(suffix)]
            rhs += value / 2
        equations.append(sp.Eq(variable, rhs))
    solution = sp.solve(equations, variables, dict=True)[0]
    return sp.factor(solution[variables[ordered.index("")]])


def _polya_sequence_probability() -> Fraction:
    total = Fraction(0)
    for outcomes in product("RB", repeat=7):
        if outcomes.count("R") != 4:
            continue
        red, blue = 2, 3
        probability = Fraction(1)
        for outcome in outcomes:
            probability *= Fraction(red if outcome == "R" else blue, red + blue)
            if outcome == "R":
                red += 1
            else:
                blue += 1
        total += probability
    return total


def verify(path: Path = DATASET) -> dict:
    answers = _load_answers(path)
    checks: dict[int, bool] = {}

    checks[90001] = _regular_binary_matrices(5) == 2040 and answers[90001] == "2040"
    checks[90002] = _unicyclic_graphs(8) == 1_436_568 and answers[90002] == "1436568"
    checks[90003] = bool(_vieta_pairs(500)) and "q_{m+2}=3q_{m+1}-q_m" in answers[90003]

    modular_solutions = [
        value for value in range(1, 10_001)
        if (pow(2, value, value) + 1) % value == 0
    ]
    expected_modular = [1, 3, 9, 27, 81, 171, 243, 513, 729, 1539, 2187, 3249, 4617, 6561, 9747]
    checks[90004] = modular_solutions == expected_modular and all(
        str(value) in answers[90004] for value in expected_modular
    )

    x = sp.symbols("x")
    alpha = sp.sqrt(2) + sp.sqrt(3) + sp.sqrt(5)
    polynomial = x**8 - 40 * x**6 + 352 * x**4 - 960 * x**2 + 576
    checks[90005] = (
        sp.minimal_polynomial(alpha, x) == polynomial
        and sp.factor(polynomial) == polynomial
        and answers[90005].replace(" ", "") == "x^8-40x^6+352x^4-960x^2+576"
    )

    ring_images = [
        value for value in range(840)
        if value * value % 840 == value and 360 * value % 840 == 0
    ]
    checks[90006] = len(ring_images) == 8 and answers[90006] == "8"

    blocks = (4, 2, 2, 1)
    formula_dimension = sum(min(left, right) for left in blocks for right in blocks)
    jordan = sp.diag(*[sp.zeros(size) for size in blocks])
    offset = 0
    for size in blocks:
        for index in range(size - 1):
            jordan[offset + index, offset + index + 1] = 1
        offset += size
    coefficients = sp.symbols("c0:81")
    matrix = sp.Matrix(9, 9, coefficients)
    commutator_equations = list(matrix * jordan - jordan * matrix)
    coefficient_matrix, _ = sp.linear_eq_to_matrix(commutator_equations, coefficients)
    rank_dimension = 81 - coefficient_matrix.rank()
    checks[90007] = formula_dimension == rank_dimension == 27 and answers[90007] == "27"

    checks[90008] = (
        _competing_pattern_probability("HTHH", "THTH") == sp.Rational(5, 14)
        and answers[90008] == r"\frac{5}{14}"
    )
    beta_binomial = sp.binomial(7, 4) * sp.rf(2, 4) * sp.rf(3, 3) / sp.rf(5, 7)
    checks[90009] = (
        beta_binomial == sp.Rational(5, 33)
        and _polya_sequence_probability() == Fraction(5, 33)
        and answers[90009] == r"\frac{5}{33}"
    )

    median_squared = sp.Rational(1, 4) * (2 * 13**2 + 2 * 20**2 - 11**2)
    centroid_distance = sp.Rational(2, 3) * sp.sqrt(median_squared)
    checks[90010] = (
        sp.simplify(centroid_distance - sp.sqrt(113)) == 0
        and answers[90010] == r"\sqrt{113}"
    )

    contact_x = sp.Rational(25, 13)
    contact_y = 3 * sp.sqrt(1 - contact_x**2 / 25)
    ellipse_area = sp.simplify(contact_y * (13 - contact_x))
    checks[90011] = (
        ellipse_area == sp.Rational(5184, 169)
        and answers[90011] == r"\frac{5184}{169}"
    )

    parameter = sp.symbols("a", real=True)
    beta_integral = sp.pi / sp.sin(sp.pi * parameter)
    integral_value = sp.simplify(sp.diff(beta_integral, parameter, 2).subs(parameter, sp.Rational(1, 2)))
    checks[90012] = integral_value == sp.pi**3 and answers[90012] == r"\pi^3"

    derivative = sp.symbols("d")
    cayley_derivative = sp.simplify(sp.diff((1 + derivative * x) / (1 - derivative * x), x).subs(x, 0))
    checks[90013] = cayley_derivative.subs(derivative, 1) == 2 and answers[90013] == "2"

    p = sp.symbols("p", positive=True)
    scaled_integral = sp.beta((p + 1) / 2, (p - 1) / 2) / 2
    checks[90014] = (
        sp.simplify(sp.expand_func(scaled_integral.subs(p, 2)) - sp.pi / 4) == 0
        and answers[90014].replace(" ", "") == r"1<p<\infty"
    )

    missing = sorted(set(answers) - set(checks))
    failed = sorted(index for index, passed in checks.items() if not passed)
    return {
        "rows": len(answers),
        "verified": len(checks) - len(failed),
        "passed": not failed and not missing and len(checks) == len(answers),
        "failed_indices": failed,
        "unchecked_indices": missing,
        "runtime_answer_exposure": False,
    }


def main() -> int:
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
