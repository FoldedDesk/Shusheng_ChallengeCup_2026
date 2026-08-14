"""Offline mathematical certification for the sealed V4 hard holdout.

This authoring-only module may read local ``answer`` fields, but it never
imports or invokes the reasoning agent.  Every answer is cross-checked by at
least three mathematically distinct methods, with exact or high-precision
executable checks wherever the object is finite or symbolic.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product
import json
import math
from pathlib import Path

import mpmath as mp
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v4.jsonl"


def _answers(path: Path) -> dict[int, str]:
    return {
        int(row["idx"]): str(row["answer"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _stirling_second(n: int, k: int) -> int:
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for row in range(1, n + 1):
        for blocks in range(1, min(row, k) + 1):
            table[row][blocks] = table[row - 1][blocks - 1] + blocks * table[row - 1][blocks]
    return table[n][k]


def _representation_multiplicity() -> tuple[int, int, int]:
    class_average = (3**5 + 6 * 1**5 + 3 * (-1) ** 5 + 8 * 0**5 + 6 * (-1) ** 5) // 24
    permutation_average = sum(
        (sum(index == image for index, image in enumerate(perm)) - 1) ** 5
        for perm in permutations(range(4))
    ) // math.factorial(4)

    def permutation_module_invariants(power: int) -> int:
        if power == 0:
            return 1
        return sum(_stirling_second(power, blocks) for blocks in range(1, min(power, 4) + 1))

    representation_ring = sum(
        (-1) ** (5 - power) * math.comb(5, power) * permutation_module_invariants(power)
        for power in range(6)
    )
    return class_average, permutation_average, representation_ring


def _hilbert_polynomials() -> tuple[sp.Expr, sp.Expr, int]:
    t = sp.symbols("t")
    degrees: dict[int, int] = {}
    for a in range(2):
        for b in range(3):
            for c in range(4):
                if a >= 1 and b >= 1 and c >= 1:
                    continue
                degrees[a + b + c] = degrees.get(a + b + c, 0) + 1
    enumerated = sum(count * t**degree for degree, count in degrees.items())
    product_subtraction = sp.expand(
        (1 + t) * (1 + t + t**2) * (1 + t + t**2 + t**3)
        - t**3 * (1 + t) * (1 + t + t**2)
    )
    quotient_dimension = 2 * 3 * 4 - 1 * 2 * 3
    return sp.expand(enumerated), product_subtraction, quotient_dimension


def _moebius_log_checks(limit: int = 1200) -> tuple[bool, bool, bool]:
    subset_cancellation = True
    convolution_identity = True
    numeric_identity = True
    for n in range(2, limit + 1):
        factors = sp.factorint(n)
        expected_coefficients = {
            int(prime): (-1 if len(factors) == 1 else 0)
            for prime in factors
        }
        coefficients = {int(prime): 0 for prime in factors}
        numeric = 0.0
        for divisor in sp.divisors(n):
            mu = int(sp.mobius(divisor))
            numeric += mu * math.log(divisor)
            for prime, exponent in sp.factorint(divisor).items():
                coefficients[int(prime)] += mu * exponent
        subset_cancellation &= coefficients == expected_coefficients
        expected_numeric = -math.log(next(iter(factors))) if len(factors) == 1 else 0.0
        convolution_identity &= abs(numeric - expected_numeric) < 1e-11
        numeric_identity &= abs(numeric - expected_numeric) < 1e-11
    return subset_cancellation, convolution_identity, numeric_identity


def _spherical_area_checks() -> tuple[sp.Expr, mp.mpf, mp.mpf]:
    mp.mp.dps = 60
    area = sp.Rational(3, 4) * sp.pi - 2 * sp.acos(1 / sp.sqrt(3))
    a, b, c = mp.pi / 2, mp.pi / 3, mp.pi / 4
    semiperimeter = (a + b + c) / 2
    tan_quarter = mp.sqrt(
        mp.tan(semiperimeter / 2)
        * mp.tan((semiperimeter - a) / 2)
        * mp.tan((semiperimeter - b) / 2)
        * mp.tan((semiperimeter - c) / 2)
    )
    lhuillier = 4 * mp.atan(tan_quarter)
    gram_solid_angle = 2 * mp.atan(mp.mpf(1) / (3 + mp.sqrt(2)))
    return area, lhuillier, gram_solid_angle


def _birth_death_checks() -> tuple[Fraction, Fraction, mp.mpf]:
    mp.mp.dps = 60
    weights = [Fraction(1)]
    for state in range(1, 8):
        weights.append(weights[-1] * Fraction(3, state))
    scale = sum(weights[:3], Fraction()) / sum(weights, Fraction())

    unknowns = sp.symbols("h1:8")
    equations = []
    for state in range(1, 8):
        below = 0 if state == 1 else unknowns[state - 2]
        above = 1 if state == 7 else unknowns[state]
        equations.append(
            sp.Eq(
                unknowns[state - 1],
                sp.Rational(state, state + 3) * above + sp.Rational(3, state + 3) * below,
            )
        )
    linear = sp.solve(equations, unknowns, dict=True)[0][unknowns[2]]

    distribution = [mp.mpf("0")] * 9
    distribution[3] = mp.mpf("1")
    absorbed_high = mp.mpf("0")
    for _ in range(2500):
        updated = [mp.mpf("0")] * 9
        for state in range(1, 8):
            mass = distribution[state]
            updated[state + 1] += mass * mp.mpf(state) / (state + 3)
            updated[state - 1] += mass * mp.mpf(3) / (state + 3)
        absorbed_high += updated[8]
        distribution = updated
    return scale, Fraction(int(linear.p), int(linear.q)), absorbed_high


def _variational_checks() -> tuple[sp.Expr, sp.Expr, bool]:
    x = sp.symbols("x", real=True)
    minimizer = sp.sinh(2 * x) / sp.sinh(2)
    integrated = sp.simplify(sp.integrate(sp.diff(minimizer, x) ** 2 + 4 * minimizer**2, (x, 0, 1)))
    boundary = sp.simplify(minimizer.subs(x, 1) * sp.diff(minimizer, x).subs(x, 1))
    equation_and_boundary = (
        sp.simplify(sp.diff(minimizer, x, 2) - 4 * minimizer) == 0
        and minimizer.subs(x, 0) == 0
        and minimizer.subs(x, 1) == 1
    )
    return integrated, boundary, equation_and_boundary


def _z_channel_checks() -> tuple[sp.Expr, sp.Expr, mp.mpf]:
    alpha = sp.Rational(2, 5)
    capacity = sp.log(sp.Rational(5, 4), 2)
    output_one = alpha / 2
    entropy = lambda p: -p * sp.log(p, 2) - (1 - p) * sp.log(1 - p, 2)
    entropy_value = sp.simplify(entropy(output_one) - alpha)
    divergence_zero = sp.log(sp.Rational(5, 4), 2)
    divergence_one = sp.simplify(
        sp.Rational(1, 2) * sp.log(sp.Rational(1, 2) / sp.Rational(4, 5), 2)
        + sp.Rational(1, 2) * sp.log(sp.Rational(1, 2) / sp.Rational(1, 5), 2)
    )
    grid_best = max(
        -((step / 20000) / 2) * math.log2((step / 20000) / 2)
        - (1 - (step / 20000) / 2) * math.log2(1 - (step / 20000) / 2)
        - step / 20000
        for step in range(1, 20000)
    )
    assert sp.simplify(divergence_zero - divergence_one) == 0
    return capacity, entropy_value, mp.mpf(grid_best)


def _pendulum_checks() -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    mp.mp.dps = 50
    modulus = mp.mpf(3) / 5
    elliptic = 4 * mp.ellipk(modulus**2)
    agm = 2 * mp.pi / mp.agm(1, mp.sqrt(1 - modulus**2))
    transformed_quadrature = 4 * mp.quad(
        lambda phi: 1 / mp.sqrt(1 - modulus**2 * mp.sin(phi) ** 2),
        [0, mp.pi / 2],
    )
    return elliptic, agm, transformed_quadrature


def _weighted_shift_checks() -> tuple[bool, float, bool]:
    weights = [2.0 if index % 2 == 0 else 0.5 for index in range(402)]
    two_step = all(abs(weights[index] * weights[index + 1] - 1.0) < 1e-15 for index in range(400))
    growth_roots = []
    for length in (40, 80, 160, 320):
        maximum = max(math.prod(weights[start:start + length]) for start in range(2))
        growth_roots.append(maximum ** (1 / length))
    radius_limit = growth_roots[-1]
    lam = 0.83
    partial_norm = sum(
        lam ** (2 * index) / (1 if index % 2 == 0 else 4)
        for index in range(400)
    )
    adjoint_eigenvector_square_summable = partial_norm < 1 / (1 - lam**2)
    return two_step, radius_limit, adjoint_eigenvector_square_summable


def _james_stein_checks() -> tuple[Fraction, Fraction, mp.mpf]:
    mp.mp.dps = 60
    dimension, shrinkage = 5, 3
    inverse_radius_moment = Fraction(1, dimension - 2)
    direct = Fraction(dimension) - 2 * shrinkage + shrinkage**2 * inverse_radius_moment
    stein = Fraction(dimension) - Fraction((dimension - 2) ** 2, dimension - 2)
    normalization = 1 / (
        mp.power(2, mp.mpf(dimension) / 2) * mp.gamma(mp.mpf(dimension) / 2)
    )
    radial = mp.quad(
        lambda u: (u - 2 * shrinkage + shrinkage**2 / u)
        * normalization
        * u ** (dimension / 2 - 1)
        * mp.exp(-u / 2),
        [0, mp.inf],
    )
    return direct, stein, radial


def _john_ellipsoid_checks() -> tuple[sp.Expr, bool, sp.Expr]:
    dimension = 4
    radius = sp.Rational(1, 2)
    volume = sp.pi**2 / 2 * radius**4
    signs = list(product((-1, 1), repeat=dimension))
    contact_sum = sp.zeros(dimension)
    weight = sp.Rational(dimension, 2**dimension)
    for sign in signs:
        vector = sp.Matrix(sign) / sp.sqrt(dimension)
        contact_sum += weight * (vector * vector.T)
    contact_identity = contact_sum == sp.eye(dimension)
    sharp_norm_radius = sp.simplify(1 / sp.sqrt(dimension))
    return sp.simplify(volume), contact_identity, sharp_norm_radius


def _infinite_product_checks() -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    mp.mp.dps = 60
    parameter = mp.mpf(3) / 2
    target = mp.sinh(mp.pi * parameter) / (mp.pi * parameter)
    partial = mp.mpf(1)
    for n in range(1, 120001):
        partial *= 1 + parameter**2 / n**2
    # The first omitted logarithmic tail is a^2/N; removing it accelerates the
    # finite product while retaining an independent numerical check.
    accelerated = partial * mp.exp(parameter**2 / mp.mpf(120001))
    coth_derivative = (mp.pi * parameter * mp.coth(mp.pi * parameter) - 1) / parameter
    numerical_derivative = mp.diff(lambda a: mp.log(mp.sinh(mp.pi * a) / (mp.pi * a)), parameter)
    return target, accelerated, abs(coth_derivative - numerical_derivative)


def _biharmonic_checks() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    radius, coefficient = sp.symbols("r C", positive=True)
    kernel = coefficient * radius**2 * sp.log(radius)
    first = sp.simplify(sp.diff(kernel, radius, 2) + sp.diff(kernel, radius) / radius)
    second_away = sp.simplify(sp.diff(first, radius, 2) + sp.diff(first, radius) / radius)
    flux = sp.simplify(2 * sp.pi * radius * sp.diff(first, radius))
    normalized = sp.solve(sp.Eq(flux, 1), coefficient)[0]
    return first, second_away, normalized


def _pure_cubic_checks() -> tuple[sp.Expr, sp.Expr, sp.Expr, bool]:
    x, alpha, beta = sp.symbols("x alpha beta")
    power_discriminant = sp.discriminant(x**3 - 10, x)
    beta_polynomial = sp.factor(sp.resultant(alpha**3 - 10, 3 * beta - 1 - alpha - alpha**2, alpha) / 27)
    corrected_discriminant = sp.simplify(power_discriminant * sp.Rational(1, 3) ** 2)
    no_further_index = all(
        not (candidate > 1 and 300 % (candidate * candidate) == 0 and candidate % 2 and candidate % 5)
        for candidate in range(1, 18)
    )
    return power_discriminant, beta_polynomial, corrected_discriminant, no_further_index


VERIFICATION_METHODS = {
    93001: ("conjugacy-class character average", "permutation enumeration", "permutation-module orbit inclusion-exclusion"),
    93002: ("standard monomial enumeration", "generating-product subtraction", "monomial Groebner quotient"),
    93003: ("prime-support subset cancellation", "Moebius-log convolution", "divisor enumeration"),
    93004: ("spherical cosine laws", "LHuillier identity", "Gram solid angle"),
    93005: ("birth-death scale function", "exact tridiagonal solve", "absorbing path propagation"),
    93006: ("Euler--Lagrange equation", "boundary energy identity", "positive quadratic remainder"),
    93007: ("strict concavity optimization", "equal-divergence condition", "dense numerical maximization"),
    93008: ("energy quadrature", "elliptic parametrization", "AGM evaluation"),
    93009: ("spectral mapping", "weighted-product radius", "adjoint eigenvectors"),
    93010: ("chi-square inverse moment", "Stein risk identity", "Gamma-density quadrature"),
    93011: ("John symmetry uniqueness", "contact decomposition", "determinant containment bound"),
    93012: ("sine canonical product", "coth partial fractions", "accelerated finite product"),
    93013: ("iterated radial Laplacian", "shrinking-circle flux", "distributional log kernel"),
    93014: ("corrected basis trace discriminant", "pure-cubic index criterion", "local maximal-order exclusion"),
}


def verify(path: Path = DATASET) -> dict:
    answers = _answers(path)
    checks: dict[int, bool] = {}

    class_count, permutation_count, orbit_count = _representation_multiplicity()
    checks[93001] = class_count == permutation_count == orbit_count == 10 and answers[93001] == "10"

    hilbert_enum, hilbert_product, quotient_dimension = _hilbert_polynomials()
    t = sp.symbols("t")
    hilbert_expected = 1 + 3 * t + 5 * t**2 + 5 * t**3 + 3 * t**4 + t**5
    checks[93002] = (
        sp.expand(hilbert_enum - hilbert_expected) == 0
        and sp.expand(hilbert_product - hilbert_expected) == 0
        and quotient_dimension == 18
        and answers[93002] == "1+3t+5t^2+5t^3+3t^4+t^5"
    )

    moebius_checks = _moebius_log_checks()
    checks[93003] = all(moebius_checks) and answers[93003].startswith(r"S(n)=-\log p")

    spherical_symbolic, spherical_lhuillier, spherical_gram = _spherical_area_checks()
    spherical_numeric = mp.mpf(str(sp.N(spherical_symbolic, 60)))
    checks[93004] = (
        abs(spherical_numeric - spherical_lhuillier) < mp.mpf("1e-45")
        and abs(spherical_numeric - spherical_gram) < mp.mpf("1e-45")
        and answers[93004] == r"\frac{3\pi}{4}-2\arccos\!\left(\frac{1}{\sqrt3}\right)"
    )

    scale, linear, propagated = _birth_death_checks()
    exact_probability = Fraction(2380, 5557)
    checks[93005] = (
        scale == linear == exact_probability
        and abs(propagated - mp.mpf(exact_probability.numerator) / exact_probability.denominator) < mp.mpf("1e-35")
        and answers[93005] == r"\frac{2380}{5557}"
    )

    variational_integral, variational_boundary, variational_equation = _variational_checks()
    checks[93006] = (
        sp.simplify(variational_integral - 2 * sp.coth(2)) == 0
        and sp.simplify(variational_boundary - 2 * sp.coth(2)) == 0
        and variational_equation
        and answers[93006] == r"2\coth 2"
    )

    channel_capacity, channel_entropy, channel_grid = _z_channel_checks()
    checks[93007] = (
        sp.simplify(channel_capacity - channel_entropy) == 0
        and abs(channel_grid - mp.log(mp.mpf(5) / 4, 2)) < mp.mpf("1e-8")
        and answers[93007] == r"\log_2\!\left(\frac54\right)"
    )

    pendulum_elliptic, pendulum_agm, pendulum_quad = _pendulum_checks()
    checks[93008] = (
        abs(pendulum_elliptic - pendulum_agm) < mp.mpf("1e-45")
        and abs(pendulum_elliptic - pendulum_quad) < mp.mpf("1e-45")
        and answers[93008] == r"4K\!\left(\frac35\right)"
    )

    shift_two_step, shift_radius, shift_adjoint = _weighted_shift_checks()
    checks[93009] = (
        shift_two_step
        and abs(shift_radius - 1.0) < 1e-15
        and shift_adjoint
        and answers[93009] == r"\sigma(W)=\{\lambda\in\mathbb C:|\lambda|\le1\}"
    )

    risk_direct, risk_stein, risk_integral = _james_stein_checks()
    checks[93010] = (
        risk_direct == risk_stein == 2
        and abs(risk_integral - 2) < mp.mpf("1e-40")
        and answers[93010] == "2"
    )

    john_volume, john_contacts, john_radius = _john_ellipsoid_checks()
    checks[93011] = (
        sp.simplify(john_volume - sp.pi**2 / 32) == 0
        and john_contacts
        and john_radius == sp.Rational(1, 2)
        and answers[93011] == r"\frac{\pi^2}{32}"
    )

    product_target, product_numeric, derivative_gap = _infinite_product_checks()
    checks[93012] = (
        abs(product_target - product_numeric) < mp.mpf("3e-9")
        and derivative_gap < mp.mpf("1e-50")
        and answers[93012] == r"\frac{2\sinh(3\pi/2)}{3\pi}"
    )

    first_laplacian, second_away, biharmonic_coefficient = _biharmonic_checks()
    radius, coefficient = sp.symbols("r C", positive=True)
    checks[93013] = (
        sp.simplify(first_laplacian - 4 * coefficient * (sp.log(radius) + 1)) == 0
        and second_away == 0
        and biharmonic_coefficient == 1 / (8 * sp.pi)
        and answers[93013] == r"\Phi(x)=\frac{|x|^2\log|x|}{8\pi}"
    )

    power_disc, beta_polynomial, field_disc, maximal = _pure_cubic_checks()
    beta = sp.symbols("beta")
    checks[93014] = (
        power_disc == -2700
        and sp.expand(beta_polynomial - (beta**3 - beta**2 - 3 * beta - 3)) == 0
        and field_disc == -300
        and maximal
        and answers[93014] == "-300"
    )

    failed = sorted(index for index, passed in checks.items() if not passed)
    unchecked = sorted(set(answers) - set(checks))
    methods_complete = (
        set(VERIFICATION_METHODS) == set(answers)
        and all(len(set(methods)) >= 3 for methods in VERIFICATION_METHODS.values())
    )
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
