"""Offline-only deterministic verification for the sealed V2 hard holdout.

The answer key is read only by this authoring script.  Nothing here is imported
by the submission agent or included in a model request.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product
import json
import math
from pathlib import Path

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v2.jsonl"


def _answers(path: Path) -> dict[int, str]:
    return {
        int(row["idx"]): str(row["answer"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _diagonal_latin_squares() -> tuple[int, int]:
    symbols = tuple(range(4))
    rows = tuple(permutations(symbols))
    normalized = total = 0
    for square in product(rows, repeat=4):
        if not all(len({square[i][j] for i in range(4)}) == 4 for j in range(4)):
            continue
        if len({square[i][i] for i in range(4)}) != 4:
            continue
        if len({square[i][3 - i] for i in range(4)}) != 4:
            continue
        total += 1
        normalized += square[0] == symbols
    return normalized, total


def _k5_flow_counts() -> tuple[int, int]:
    vertices = range(5)
    edges = tuple((i, j) for i in vertices for j in vertices if i < j)
    direct = 0
    for values in product((1, 2), repeat=len(edges)):
        balance = [0] * 5
        for (left, right), value in zip(edges, values):
            balance[left] = (balance[left] + value) % 3
            balance[right] = (balance[right] - value) % 3
        direct += all(value == 0 for value in balance)

    chords = tuple(edge for edge in edges if edge[0] != 0)
    cycle_coordinates = 0
    for chord_values in product(range(3), repeat=len(chords)):
        if any(value == 0 for value in chord_values):
            continue
        chord_balance = [0] * 5
        for (left, right), value in zip(chords, chord_values):
            chord_balance[left] = (chord_balance[left] + value) % 3
            chord_balance[right] = (chord_balance[right] - value) % 3
        tree_values = tuple(chord_balance[vertex] for vertex in range(1, 5))
        cycle_coordinates += all(value != 0 for value in tree_values)
    return direct, cycle_coordinates


def _valuation_two(value: int) -> int:
    exponent = 0
    while value and value % 2 == 0:
        exponent += 1
        value //= 2
    return exponent


def _central_binomial_product_valuation(limit: int) -> tuple[int, int]:
    digit_sum = sum(value.bit_count() for value in range(1, limit + 1))
    legendre = 0
    for value in range(1, limit + 1):
        numerator = math.factorial(2 * value)
        denominator = math.factorial(value) ** 2
        legendre += _valuation_two(numerator // denominator)
    return digit_sum, legendre


def _quadric_points(prime: int) -> int:
    square_counts = [0] * prime
    for value in range(prime):
        square_counts[value * value % prime] += 1
    return sum(
        square_counts[left]
        * square_counts[right]
        * square_counts[(1 - left - right) % prime]
        for left in range(prime)
        for right in range(prime)
    )


def _rref(rows: tuple[tuple[int, ...], ...], prime: int) -> tuple[tuple[int, ...], ...]:
    matrix = [list(row) for row in rows if any(value % prime for value in row)]
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next(
            (index for index in range(pivot_row, len(matrix)) if matrix[index][column] % prime),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column] % prime, -1, prime)
        matrix[pivot_row] = [(value * inverse) % prime for value in matrix[pivot_row]]
        for index in range(len(matrix)):
            if index == pivot_row:
                continue
            factor = matrix[index][column] % prime
            matrix[index] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(matrix[index], matrix[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return tuple(tuple(row) for row in matrix[:pivot_row])


def _two_dimensional_subspaces() -> set[tuple[tuple[int, ...], ...]]:
    vectors = tuple(product(range(3), repeat=4))
    spaces: set[tuple[tuple[int, ...], ...]] = set()
    for first in vectors[1:]:
        for second in vectors[1:]:
            basis = _rref((first, second), 3)
            if len(basis) == 2:
                spaces.add(basis)
    return spaces


def _involution_count() -> tuple[int, int]:
    gl = lambda dimension: math.prod(3**dimension - 3**index for index in range(dimension))
    orbit = gl(4) // (gl(2) * gl(2))
    spaces = _two_dimensional_subspaces()
    decompositions = sum(
        len(_rref(left + right, 3)) == 4
        for left in spaces
        for right in spaces
    )
    return orbit, decompositions


def _irreducible_matrix_count() -> tuple[int, int]:
    nonsquares = {2, 3}
    brute = 0
    for a, b, c, d in product(range(5), repeat=4):
        discriminant = ((a + d) ** 2 - 4 * (a * d - b * c)) % 5
        brute += discriminant in nonsquares
    irreducible_quadratics = (5**2 - 5) // 2
    gl2 = (5**2 - 1) * (5**2 - 5)
    orbit_sum = irreducible_quadratics * gl2 // (5**2 - 1)
    return brute, orbit_sum


def _disphenoid_volume() -> tuple[sp.Expr, sp.Expr]:
    x, y, z = sp.sqrt(12), sp.sqrt(8), sp.sqrt(5)
    points = (
        sp.Matrix((x, y, z)),
        sp.Matrix((x, -y, -z)),
        sp.Matrix((-x, y, -z)),
        sp.Matrix((-x, -y, z)),
    )
    coordinate = sp.Abs(sp.det(sp.Matrix.hstack(
        points[1] - points[0], points[2] - points[0], points[3] - points[0]
    ))) / 6
    squared = (
        (0, 52, 68, 80),
        (52, 0, 80, 68),
        (68, 80, 0, 52),
        (80, 68, 52, 0),
    )
    cayley = sp.ones(5, 5)
    cayley[0, 0] = 0
    for i in range(4):
        for j in range(4):
            cayley[i + 1, j + 1] = squared[i][j]
    cayley_menger = sp.sqrt(sp.det(cayley) / 288)
    return sp.simplify(coordinate), sp.simplify(cayley_menger)


def _projective_parameter() -> tuple[sp.Expr, sp.Expr]:
    parameter = sp.symbols("t")
    tangent_one = sp.Matrix((1, -2, 1))
    tangent_two = sp.Matrix((1, -4, 4))
    intersection = tangent_one.cross(tangent_two)
    point_three = sp.Matrix((9, 3, 1))
    secant = intersection.cross(point_three)
    restriction = sp.factor(secant.dot(sp.Matrix((parameter**2, parameter, 1))))
    roots = sp.solve(restriction, parameter)
    second = next(root for root in roots if root != 3)

    polar_intersection = sp.Matrix((4, 3, 2))
    polar_line = polar_intersection.cross(point_three)
    polar_restriction = sp.factor(polar_line.dot(sp.Matrix((parameter**2, parameter, 1))))
    polar_second = next(root for root in sp.solve(polar_restriction, parameter) if root != 3)
    return sp.simplify(second), sp.simplify(polar_second)


def _branching_extinction() -> tuple[sp.Expr, float]:
    q = sp.symbols("q")
    fixed = sp.factor(sp.Rational(1, 4) + q / 4 + q**3 / 2 - q)
    root = min(
        (value for value in sp.solve(fixed, q) if value.is_real and 0 <= float(value) <= 1),
        key=float,
    )
    exact = sp.simplify(root**3)
    iterate = 0.0
    for _ in range(300):
        iterate = 0.25 + 0.25 * iterate + 0.5 * iterate**3
    return exact, iterate**3


def _halton(index: int, base: int) -> float:
    fraction = 1.0
    value = 0.0
    while index:
        fraction /= base
        value += fraction * (index % base)
        index //= base
    return value


def _dirichlet_probability_check(samples: int = 50_000) -> float:
    accepted = 0
    for index in range(1, samples + 1):
        exponentials = [-math.log(_halton(index, base)) for base in (2, 3, 5, 7, 11)]
        accepted += 5 * max(exponentials) < 2 * sum(exponentials)
    return accepted / samples


def _integral_operator_power_norm(size: int = 80, iterations: int = 250) -> float:
    points = [(index + 0.5) / size for index in range(size)]
    vector = [1 / math.sqrt(size)] * size
    eigenvalue = 0.0
    for _ in range(iterations):
        updated = [
            sum(min(x, y) * value for y, value in zip(points, vector)) / size
            for x in points
        ]
        norm = math.sqrt(sum(value * value for value in updated))
        vector = [value / norm for value in updated]
        eigenvalue = sum(
            vector[i]
            * sum(min(points[i], y) * vector[j] for j, y in enumerate(points))
            / size
            for i in range(size)
        )
    return eigenvalue


def verify(path: Path = DATASET) -> dict:
    answers = _answers(path)
    checks: dict[int, bool] = {}

    normalized, latin_total = _diagonal_latin_squares()
    checks[91001] = normalized == 2 and latin_total == 48 and answers[91001] == "48"

    direct_flows, cycle_flows = _k5_flow_counts()
    checks[91002] = direct_flows == cycle_flows == 24 and answers[91002] == "24"

    digit_valuation, factorial_valuation = _central_binomial_product_valuation(1024)
    checks[91003] = digit_valuation == factorial_valuation == 5121 and answers[91003] == "5121"

    quadric_formula = 1009**2 + 1009
    checks[91004] = _quadric_points(1009) == quadric_formula == 1_019_090 and answers[91004] == "1019090"

    orbit_involutions, decomposed_involutions = _involution_count()
    checks[91005] = orbit_involutions == decomposed_involutions == 10_530 and answers[91005] == "10530"

    brute_irreducible, orbit_irreducible = _irreducible_matrix_count()
    checks[91006] = brute_irreducible == orbit_irreducible == 200 and answers[91006] == "200"

    coordinate_volume, cayley_volume = _disphenoid_volume()
    expected_volume = 32 * sp.sqrt(30) / 3
    checks[91007] = coordinate_volume == cayley_volume == expected_volume and answers[91007] == r"\frac{32\sqrt{30}}{3}"

    line_parameter, polar_parameter = _projective_parameter()
    checks[91008] = line_parameter == polar_parameter == sp.Rational(5, 3) and answers[91008] == r"\frac{5}{3}"

    extinction_exact, extinction_iteration = _branching_extinction()
    expected_extinction = (3 * sp.sqrt(3) - 5) / 4
    checks[91009] = (
        sp.simplify(extinction_exact - expected_extinction) == 0
        and abs(extinction_iteration - float(expected_extinction)) < 1e-12
        and answers[91009] == r"\frac{3\sqrt3-5}{4}"
    )

    simplex_exact = 1 - 5 * Fraction(3, 5) ** 4 + 10 * Fraction(1, 5) ** 4
    simplex_numeric = _dirichlet_probability_check()
    checks[91010] = (
        simplex_exact == Fraction(46, 125)
        and abs(simplex_numeric - float(simplex_exact)) < 0.002
        and answers[91010] == r"\frac{46}{125}"
    )

    z = sp.symbols("z")
    roots = sp.nroots(z**7 + 5 * z**3 + 1, n=40, maxsteps=200)
    annulus_roots = sum(1 < abs(complex(root)) < 2 for root in roots)
    checks[91011] = annulus_roots == 4 and answers[91011] == "4"

    spectral_numeric = _integral_operator_power_norm()
    checks[91012] = (
        abs(spectral_numeric - 4 / math.pi**2) < 5e-5
        and answers[91012] == r"\frac{4}{\pi^2}"
    )

    checks[91013] = math.gcd(2, 3) == 1 and answers[91013] == r"H_1(X;\mathbb Z)\cong\mathbb Z"

    variable = sp.symbols("lambda")
    polynomial = (65 - 63 * variable + 15 * variable**2 - variable**3) / 65
    mapped = (variable - 5) / 4
    chebyshev = sp.chebyshevt(3, mapped) / sp.chebyshevt(3, sp.Rational(-5, 4))
    extrema = [1, 3, 7, 9]
    alternating = [sp.simplify(polynomial.subs(variable, value)) for value in extrema]
    checks[91014] = (
        sp.simplify(polynomial - chebyshev) == 0
        and alternating == [sp.Rational(16, 65), sp.Rational(-16, 65), sp.Rational(16, 65), sp.Rational(-16, 65)]
        and answers[91014] == r"p(\lambda)=\frac{65-63\lambda+15\lambda^2-\lambda^3}{65}"
    )

    failed = sorted(index for index, passed in checks.items() if not passed)
    unchecked = sorted(set(answers) - set(checks))
    return {
        "rows": len(answers),
        "verified": len(checks) - len(failed),
        "passed": not failed and not unchecked and len(checks) == len(answers),
        "failed_indices": failed,
        "unchecked_indices": unchecked,
        "runtime_answer_exposure": False,
        "agent_executed": False,
    }


def main() -> int:
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
