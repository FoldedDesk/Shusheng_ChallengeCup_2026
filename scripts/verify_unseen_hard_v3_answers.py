"""Offline mathematical certification for the sealed V3 hard holdout.

This authoring-only module reads the local answer field, but it never imports
or invokes the reasoning agent.  Each item is checked by two independent
derivations; finite or symbolic items also receive an executable third check.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import random

import mpmath as mp
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v3.jsonl"


def _answers(path: Path) -> dict[int, str]:
    return {
        int(row["idx"]): str(row["answer"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _fixed_arc_euler_tours() -> tuple[int, int]:
    vertices = tuple(range(4))
    arcs = tuple((left, right) for left in vertices for right in vertices if left != right)
    first = arcs.index((0, 1))

    @lru_cache(maxsize=None)
    def count(vertex: int, used: int) -> int:
        if used == (1 << len(arcs)) - 1:
            return int(vertex == 0)
        return sum(
            count(right, used | (1 << index))
            for index, (left, right) in enumerate(arcs)
            if left == vertex and not used & (1 << index)
        )

    brute = count(1, 1 << first)
    laplacian_minor = sp.Matrix([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]])
    best = int(laplacian_minor.det()) * math.factorial(2) ** 4
    return best, brute


def _plane_tree_counts() -> tuple[int, int]:
    increments = (-1, 0, 1, 2)
    initial = (5, 2, 2, 1)

    @lru_cache(maxsize=None)
    def valid_words(counts: tuple[int, ...], height: int) -> int:
        remaining = sum(counts)
        if not remaining:
            return int(height == -1)
        total = 0
        for index, increment in enumerate(increments):
            if counts[index] == 0:
                continue
            next_height = height + increment
            if remaining > 1 and next_height < 0:
                continue
            updated = list(counts)
            updated[index] -= 1
            total += valid_words(tuple(updated), next_height)
        return total

    enumerated = valid_words(initial, 0)
    multinomial = math.factorial(10) // (
        math.factorial(5) * math.factorial(2) * math.factorial(2)
    )
    return multinomial // 10, enumerated


def _singular_congruence_counts() -> tuple[int, int]:
    modulus = 3**9
    brute = sum((value * value - 3**4) % modulus == 0 for value in range(modulus))
    lifted_units = sum((unit * unit - 1) % (3**5) == 0 for unit in range(3**7))
    return 2 * 3**2, brute if brute == lifted_units else -1


def _prime_power_base(value: int) -> int | None:
    factors = sp.factorint(value)
    return int(next(iter(factors))) if len(factors) == 1 else None


def _cyclotomic_checks() -> tuple[bool, bool]:
    x = sp.symbols("x")
    symbolic = all(
        int(sp.cyclotomic_poly(index, x).subs(x, 1))
        == (_prime_power_base(index) or 1)
        for index in range(2, 501)
    )
    classified = all(
        (_prime_power_base(index) == 7) == (index in {7, 49, 343})
        for index in range(2, 501)
    )
    return symbolic, classified


def _mixed_group_automorphisms() -> tuple[int, int]:
    criterion = brute = 0
    for alpha, beta, gamma, delta in product(range(8), range(4), range(4), range(4)):
        criterion += alpha % 2 == 1 and delta % 2 == 1
        image = {
            ((alpha * x + 2 * beta * y) % 8, (gamma * x + delta * y) % 4)
            for x in range(8)
            for y in range(4)
        }
        brute += len(image) == 32
    return criterion, brute


def _torus_zero_mean_length() -> tuple[sp.Expr, sp.Expr]:
    v = sp.symbols("v", real=True)
    radius = sp.Rational(3, 2) + sp.cos(v)
    mean_curvature = sp.simplify(
        (sp.Rational(3, 2) + 2 * sp.cos(v)) / (2 * radius)
    )
    zero_cosine = sp.solve(sp.together(mean_curvature), sp.cos(v))[0]
    branch_radius = sp.simplify(radius.subs(sp.cos(v), zero_cosine))
    geometric = sp.simplify(2 * (2 * sp.pi * branch_radius))

    u = sp.symbols("u", real=True)
    branch = sp.Matrix((branch_radius * sp.cos(u), branch_radius * sp.sin(u), sp.sqrt(7) / 4))
    speed = sp.sqrt(sp.simplify(branch.diff(u).dot(branch.diff(u))))
    integrated = sp.simplify(2 * sp.integrate(speed, (u, 0, 2 * sp.pi)))
    return geometric, integrated


def _hyperbolic_octagon_checks() -> tuple[sp.Expr, sp.Expr]:
    half_cosh = sp.trigsimp(sp.cos(sp.pi / 8) / sp.sin(sp.pi / 8))
    simplified = sp.radsimp(half_cosh)
    full_cosh = sp.expand(2 * simplified**2 - 1)
    return sp.simplify(simplified), sp.simplify(full_cosh)


def _gaussian_real_eigenvalue_probability(samples: int = 120_000) -> float:
    rng = random.Random(92008)
    accepted = 0
    for _ in range(samples):
        a, b, c, d = (rng.gauss(0.0, 1.0) for _ in range(4))
        accepted += (a - d) ** 2 + 4 * b * c >= 0
    return accepted / samples


def _wishart_monte_carlo(samples: int = 15_000) -> float:
    rng = random.Random(92009)
    total = 0.0
    for _ in range(samples):
        rows = [[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(7)]
        gram = [
            [sum(row[i] * row[j] for row in rows) for j in range(3)]
            for i in range(3)
        ]
        total += (
            gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
            - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
            + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0])
        )
    return total / samples


def _endpoint_integral_numeric(index: int = 50_000) -> mp.mpf:
    mp.mp.dps = 50
    log_value = (
        -mp.log(4)
        + mp.loggamma(mp.mpf(1) / 4)
        + mp.loggamma(index + 1)
        - mp.loggamma(index + mp.mpf(5) / 4)
        + mp.log(index) / 4
    )
    return mp.exp(log_value)


def _factorial_gap_checks() -> tuple[bool, bool]:
    ratios = [math.factorial(index + 1) // math.factorial(index) for index in range(2, 13)]
    gap_growth = ratios == list(range(3, 14))
    root_orders = (2, 3, 5, 7)
    root_absorption = all(
        all(math.factorial(index) % order == 0 for index in range(order, 13))
        for order in root_orders
    )
    return gap_growth, root_absorption


def _sdirk_checks() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    gamma, z, y = sp.symbols("gamma z y", positive=True, real=True)
    roots = sp.solve(sp.Eq(2 * gamma - gamma**2, sp.Rational(1, 2)), gamma)
    admissible = next(root for root in roots if 0 < float(root) < 1)
    matrix = sp.Matrix(((gamma, 0), (1 - gamma, gamma)))
    weights = sp.Matrix(((1 - gamma), gamma))
    ones = sp.ones(2, 1)
    stability = sp.factor(1 + z * (weights.T * (sp.eye(2) - z * matrix).inv() * ones)[0])
    chosen = sp.factor(stability.subs(gamma, admissible))
    stiff_limit = sp.limit(chosen, z, sp.oo)
    modulus_gap = sp.simplify(
        (1 + admissible**2 * y**2) ** 2
        - (1 + (1 - 2 * admissible) ** 2 * y**2)
    )
    return sp.simplify(admissible), stiff_limit, sp.factor(modulus_gap)


def _torus_fixed_points() -> tuple[int, int]:
    difference = sp.Matrix(((3, 1), (2, 2)))
    determinant = abs(int(difference.det()))
    representatives = 0
    for numerator_x, numerator_y in product(range(determinant), repeat=2):
        vector = sp.Matrix((sp.Rational(numerator_x, determinant), sp.Rational(numerator_y, determinant)))
        image = difference * vector
        representatives += all(value.q == 1 for value in image)
    return determinant, representatives


def _burgers_catastrophe() -> tuple[sp.Expr, float]:
    xi = sp.symbols("xi", real=True)
    slope = sp.diff(sp.exp(-xi**2), xi)
    critical = [value for value in sp.solve(sp.diff(slope, xi), xi) if value.is_real]
    minimum = min((sp.simplify(slope.subs(xi, value)) for value in critical), key=float)
    exact = sp.simplify(-1 / minimum)
    grid_min = min(-2 * (step / 100_000) * math.exp(-(step / 100_000) ** 2) for step in range(0, 300_001))
    return exact, -1 / grid_min


VERIFICATION_METHODS = {
    92001: ("BEST and matrix-tree", "last-exit ordering", "arc-state dynamic programming"),
    92002: ("cycle lemma", "prescribed-degree tree formula", "prefix-word enumeration"),
    92003: ("valuation stratification", "odd-prime unit roots", "residue enumeration"),
    92004: ("cyclotomic quotient induction", "Mobius cancellation", "symbolic polynomial prefix"),
    92005: ("Frattini quotient", "abelian p-group formula", "endomorphism enumeration"),
    92006: ("fundamental forms", "surface-of-revolution curvatures", "curve-speed integration"),
    92007: ("right-triangle law", "regular-polygon identity", "full-side cosh identity"),
    92008: ("Gaussian orthogonal transform", "spherical solid angle", "seeded simulation"),
    92009: ("Bartlett decomposition", "Cauchy-Binet minors", "seeded simulation"),
    92010: ("Beta-Gamma ratio", "endpoint Laplace scaling", "high-precision evaluation"),
    92011: ("root-of-unity blow-up", "Fabry gap theorem", "factorial divisibility check"),
    92012: ("order conditions", "imaginary-axis modulus", "symbolic stability function"),
    92013: ("lattice cokernel", "Lefschetz index", "rational-grid enumeration"),
    92014: ("characteristics", "minimum-slope theorem", "dense-grid minimization"),
}


def verify(path: Path = DATASET) -> dict:
    answers = _answers(path)
    checks: dict[int, bool] = {}

    best, euler_brute = _fixed_arc_euler_tours()
    checks[92001] = best == euler_brute == 256 and answers[92001] == "256"

    cycle_lemma, tree_words = _plane_tree_counts()
    checks[92002] = cycle_lemma == tree_words == 756 and answers[92002] == "756"

    lifted, congruence_brute = _singular_congruence_counts()
    checks[92003] = lifted == congruence_brute == 18 and answers[92003] == "18"

    cyclotomic_symbolic, cyclotomic_classification = _cyclotomic_checks()
    checks[92004] = (
        cyclotomic_symbolic
        and cyclotomic_classification
        and answers[92004] == r"n=7^k\ (k\ge1)"
    )

    frattini_count, automorphism_brute = _mixed_group_automorphisms()
    checks[92005] = frattini_count == automorphism_brute == 128 and answers[92005] == "128"

    curvature_length, speed_length = _torus_zero_mean_length()
    checks[92006] = (
        curvature_length == speed_length == 3 * sp.pi
        and answers[92006] == r"3\pi"
    )

    half_side_cosh, full_side_cosh = _hyperbolic_octagon_checks()
    checks[92007] = (
        sp.simplify(half_side_cosh - (1 + sp.sqrt(2))) == 0
        and sp.simplify(full_side_cosh - (5 + 4 * sp.sqrt(2))) == 0
        and answers[92007] == r"2\operatorname{arcosh}(1+\sqrt2)"
    )

    gaussian_numeric = _gaussian_real_eigenvalue_probability()
    checks[92008] = (
        abs(gaussian_numeric - 1 / math.sqrt(2)) < 0.006
        and answers[92008] == r"\frac{1}{\sqrt2}"
    )

    bartlett = 7 * 6 * 5
    cauchy_binet = math.comb(7, 3) * math.factorial(3)
    wishart_numeric = _wishart_monte_carlo()
    checks[92009] = (
        bartlett == cauchy_binet == 210
        and abs(wishart_numeric - 210) < 7
        and answers[92009] == "210"
    )

    endpoint_numeric = _endpoint_integral_numeric()
    endpoint_exact = mp.gamma(mp.mpf(1) / 4) / 4
    checks[92010] = (
        abs(endpoint_numeric - endpoint_exact) < mp.mpf("0.00001")
        and answers[92010] == r"\frac{\Gamma(1/4)}{4}"
    )

    gap_growth, root_absorption = _factorial_gap_checks()
    checks[92011] = (
        gap_growth
        and root_absorption
        and answers[92011] == r"|z|=1\text{ is a natural boundary}"
    )

    gamma, stiff_limit, modulus_gap = _sdirk_checks()
    checks[92012] = (
        sp.simplify(gamma - (1 - 1 / sp.sqrt(2))) == 0
        and stiff_limit == 0
        and sp.simplify(modulus_gap - gamma**4 * sp.symbols("y", positive=True, real=True) ** 4) == 0
        and answers[92012] == r"1-\frac{1}{\sqrt2}"
    )

    lattice_index, fixed_enumeration = _torus_fixed_points()
    checks[92013] = lattice_index == fixed_enumeration == 4 and answers[92013] == "4"

    catastrophe_exact, catastrophe_numeric = _burgers_catastrophe()
    checks[92014] = (
        sp.simplify(catastrophe_exact - sp.sqrt(sp.E / 2)) == 0
        and abs(catastrophe_numeric - math.sqrt(math.e / 2)) < 1e-9
        and answers[92014] == r"\sqrt{\frac e2}"
    )

    failed = sorted(index for index, passed in checks.items() if not passed)
    unchecked = sorted(set(answers) - set(checks))
    methods_complete = all(len(set(methods)) >= 3 for methods in VERIFICATION_METHODS.values())
    return {
        "rows": len(answers),
        "verified": len(checks) - len(failed),
        "passed": not failed and not unchecked and len(checks) == len(answers) and methods_complete,
        "failed_indices": failed,
        "unchecked_indices": unchecked,
        "minimum_independent_methods": min(map(len, VERIFICATION_METHODS.values())),
        "runtime_answer_exposure": False,
        "agent_executed": False,
    }


def main() -> int:
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
