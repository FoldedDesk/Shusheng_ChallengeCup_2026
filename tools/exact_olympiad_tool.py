"""Strict deterministic handlers for common finite olympiad problem families."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import math
import re
from typing import Optional


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _small_number(value: str) -> Optional[int]:
    token = str(value or "").strip().lower().strip("$()")
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token, _CHINESE_NUMBERS.get(token))


def _positive_product(expression: str) -> Optional[int]:
    value = str(expression or "").replace(r"\cdot", "*").replace(" ", "")
    if not value:
        return None
    total = 1
    for part in value.split("*"):
        match = re.fullmatch(r"(\d+)(?:\^\{?(\d+)\}?)?", part)
        if not match:
            return None
        base = int(match.group(1))
        exponent = int(match.group(2) or 1)
        if base <= 0 or exponent < 0 or exponent > 10000:
            return None
        total *= base**exponent
    return total


def _prime_factors(value: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            remaining //= divisor
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def _divisors(value: int) -> list[int]:
    result = [1]
    for prime, exponent in _prime_factors(value).items():
        powers = [prime**power for power in range(exponent + 1)]
        result = [left * right for left in result for right in powers]
    return sorted(result)


def _totient(value: int) -> int:
    result = value
    for prime in _prime_factors(value):
        result -= result // prime
    return result


def _integer_polynomial(expression: str, variable: str = "n") -> Optional[list[int]]:
    """Parse a deliberately small integer-polynomial grammar, lowest degree first."""
    text = str(expression or "").replace(" ", "").replace("{", "").replace("}", "")
    text = text.replace(r"\cdot", "").replace("*", "").replace("−", "-")
    if not text or not re.fullmatch(rf"[0-9{re.escape(variable)}+\-^]+", text):
        return None
    if text[0] not in "+-":
        text = "+" + text
    terms = re.findall(r"[+-][^+-]+", text)
    if "".join(terms) != text:
        return None
    coefficients: dict[int, int] = defaultdict(int)
    for term in terms:
        sign = -1 if term[0] == "-" else 1
        body = term[1:]
        if variable in body:
            match = re.fullmatch(rf"(\d*){re.escape(variable)}(?:\^(\d+))?", body)
            if not match:
                return None
            coefficient = int(match.group(1) or 1)
            exponent = int(match.group(2) or 1)
        elif body.isdigit():
            coefficient, exponent = int(body), 0
        else:
            return None
        if exponent > 32:
            return None
        coefficients[exponent] += sign * coefficient
    degree = max(coefficients, default=0)
    result = [coefficients[index] for index in range(degree + 1)]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _polynomial_remainder(dividend: list[int], divisor: list[int]) -> Optional[list[int]]:
    if len(divisor) < 2 or abs(divisor[-1]) != 1:
        return None
    remainder = list(dividend)
    while len(remainder) >= len(divisor):
        factor = remainder[-1] // divisor[-1]
        offset = len(remainder) - len(divisor)
        for index, coefficient in enumerate(divisor):
            remainder[offset + index] -= factor * coefficient
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    return remainder


def _polynomial_value(coefficients: list[int], value: int) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


class ExactOlympiadTool:
    """Return answers only when every rule needed by an exact algorithm is present."""

    _HANDLERS: tuple[str, ...] = (
        "_cycle_distance_two_coloring",
        "_punctured_domino_tilings",
        "_complete_intersection_maximum",
        "_bounded_generalized_pell_count",
        "_integer_polynomial_divisibility",
        "_reciprocal_quartic_nonnegative",
        "_affine_recurrence_determinant",
        "_root_polynomial_product",
        "_cevian_length",
        "_smith_normal_form",
        "_intersecting_antichain_maximum",
        "_bipartite_matching_deletion_trees",
        "_cyclic_nonadjacent_selection",
        "_finite_subtraction_game",
        "_square_subtraction_game",
        "_wheel_coloring",
        "_grid_poset_extensions",
        "_hypercube_spanning_trees",
        "_odd_fiber_functions",
        "_couples_unlabeled_groups",
        "_bounded_divisor_count",
        "_primitive_pythagorean_count",
        "_inverse_totient",
        "_gcd_sum",
        "_positive_sum_two_squares",
        "_factorial_quotient_valuation",
        "_pell_fundamental_solution",
        "_least_integer_with_divisor_count",
        "_factorable_binary_quadratic",
        "_descartes_inner_circle",
        "_rotation_necklace_fixed_weight",
        "_bose_einstein_integral",
        "_bernoulli_likelihood_ratio",
        "_brownian_exit_expectation",
    )

    def hints_for(self, problem: str) -> list[str]:
        hints = []
        for name in self._HANDLERS:
            try:
                hint = getattr(self, name)(str(problem or ""))
            except Exception:
                hint = None
            if hint:
                hints.append(hint)
        return hints

    @staticmethod
    def _cycle_distance_two_coloring(problem: str) -> Optional[str]:
        polygon = re.search(r"regular\s+\$?(\d+)\$?-gon", problem, re.IGNORECASE)
        colors = re.search(r"(\d+|[A-Za-z]+)\s+labeled\s+colors", problem, re.IGNORECASE)
        if (
            not polygon or not colors
            or not re.search(r"vertices?.{0,30}labeled\s+cyclically", problem, re.IGNORECASE)
            or not re.search(
                r"any\s+two\s+vertices\s+at\s+cyclic\s+distance\s+\$?1\$?\s+or\s+\$?2\$?"
                r"\s+receive\s+different\s+colors",
                problem,
                re.IGNORECASE,
            )
        ):
            return None
        size = int(polygon.group(1))
        color_count = _small_number(colors.group(1))
        if color_count is None or not 5 <= size <= 500 or not 3 <= color_count <= 30:
            return None

        # Color symmetry fixes the first two colors to 0 and 1.
        states = {(0, 1): 1}
        for _ in range(2, size):
            updated: dict[tuple[int, int], int] = defaultdict(int)
            for (previous_two, previous), count in states.items():
                for color in range(color_count):
                    if color not in {previous_two, previous}:
                        updated[(previous, color)] += count
            states = updated
        canonical = sum(
            count
            for (penultimate, last), count in states.items()
            if last != 0 and penultimate != 0 and last != 1
        )
        result = color_count * (color_count - 1) * canonical
        return f"本地循环距离二染色计数: {result}"

    @staticmethod
    def _punctured_domino_tilings(problem: str) -> Optional[str]:
        board = re.search(
            r"(?:a\s+)?\$?(\d+)\$?\s+by\s+\$?(\d+)\$?\s+rectangular\s+board",
            problem,
            re.IGNORECASE,
        )
        holes = re.findall(
            r"square\s+in\s+row\s+\$?(\d+)\$?\s*,\s*column\s+\$?(\d+)\$?",
            problem,
            re.IGNORECASE,
        )
        if (
            not board or not holes
            or not re.search(r"\bremove\b", problem, re.IGNORECASE)
            or not re.search(r"remaining\s+board\s+be\s+tiled", problem, re.IGNORECASE)
            or not re.search(r"\$?1\$?\s+by\s+\$?2\$?\s+dominoes", problem, re.IGNORECASE)
        ):
            return None
        rows, columns = map(int, board.groups())
        removed = {(int(row) - 1, int(column) - 1) for row, column in holes}
        if (
            len(removed) != len(holes)
            or not rows or not columns
            or any(not (0 <= row < rows and 0 <= column < columns) for row, column in removed)
            or rows * columns > 240
            or min(rows, columns) > 14
        ):
            return None
        if (rows * columns - len(removed)) % 2:
            return "本地障碍多米诺铺法计数: 0"
        if columns > rows:
            rows, columns = columns, rows
            removed = {(column, row) for row, column in removed}
        blocked = [0] * rows
        for row, column in removed:
            blocked[row] |= 1 << column

        states = {0: 1}
        full = (1 << columns) - 1
        for row in range(rows):
            updated: dict[int, int] = defaultdict(int)
            for incoming, count in states.items():
                if incoming & blocked[row]:
                    continue
                occupied = incoming | blocked[row]

                def fill(mask: int, next_mask: int) -> None:
                    if mask == full:
                        updated[next_mask] += count
                        return
                    column = next(index for index in range(columns) if not mask & (1 << index))
                    bit = 1 << column
                    if column + 1 < columns and not mask & (bit << 1):
                        fill(mask | bit | (bit << 1), next_mask)
                    if row + 1 < rows and not blocked[row + 1] & bit:
                        fill(mask | bit, next_mask | bit)

                fill(occupied, 0)
            states = updated
        return f"本地障碍多米诺铺法计数: {states.get(0, 0)}"

    @staticmethod
    def _complete_intersection_maximum(problem: str) -> Optional[str]:
        family = re.search(
            r"family\s+of\s+\$?(\d+)\$?-element\s+subsets\s+of\s+"
            r"\{\s*1\s*,\s*2\s*,\s*(?:\.\.\.|\\ldots|\\dots)\s*,?\s*(\d+)\s*\}",
            problem,
            re.IGNORECASE,
        )
        intersection = re.search(
            r"intersection\s+of\s+every\s+two\s+distinct\s+members\s+has\s+at\s+least\s+"
            r"(\d+|[A-Za-z]+)\s+elements",
            problem,
            re.IGNORECASE,
        )
        if (
            not family or not intersection
            or not re.search(r"maximum\s+possible\s+value\s+of\s+\$?\|A\|\$?", problem, re.IGNORECASE)
        ):
            return None
        subset_size, universe_size = map(int, family.groups())
        threshold = _small_number(intersection.group(1))
        if (
            threshold is None
            or not 1 <= threshold <= subset_size <= universe_size <= 500
        ):
            return None
        best = 0
        for index in range(min(subset_size - threshold, (universe_size - threshold) // 2) + 1):
            distinguished = threshold + 2 * index
            required = threshold + index
            size = 0
            for chosen in range(required, min(subset_size, distinguished) + 1):
                outside = subset_size - chosen
                if 0 <= outside <= universe_size - distinguished:
                    size += math.comb(distinguished, chosen) * math.comb(
                        universe_size - distinguished, outside
                    )
            best = max(best, size)
        return f"本地完全交集族最大值: {best}"

    @staticmethod
    def _bounded_generalized_pell_count(problem: str) -> Optional[str]:
        equation = re.search(r"x\^2\s*-\s*(\d+)\s*y\^2\s*=\s*([-+]?\d+)", problem)
        bound = re.search(r"x\s*(?:<=|\\le)\s*(10\^\{?\d+\}?|\d+)", problem)
        if (
            not equation or not bound
            or not re.search(r"ordered\s+pairs?\s+of\s+positive\s+integers", problem, re.IGNORECASE)
            or not re.search(r"(?:determine|find)\s+the\s+number", problem, re.IGNORECASE)
        ):
            return None
        nonsquare, target = map(int, equation.groups())
        bound_text = bound.group(1).replace("{", "").replace("}", "")
        x_bound = 10 ** int(bound_text.split("^")[1]) if "^" in bound_text else int(bound_text)
        root = math.isqrt(nonsquare)
        if nonsquare <= 1 or root * root == nonsquare or x_bound < 1:
            return None
        maximum_y_squared = (x_bound * x_bound - target) // nonsquare
        if maximum_y_squared < 1:
            return "本地受界广义Pell解计数: 0"
        maximum_y = math.isqrt(maximum_y_squared)
        if maximum_y > 2_000_000:
            return None
        count = 0
        for y_value in range(1, maximum_y + 1):
            x_squared = nonsquare * y_value * y_value + target
            if x_squared <= 0:
                continue
            x_value = math.isqrt(x_squared)
            count += x_value <= x_bound and x_value * x_value == x_squared
        return f"本地受界广义Pell解计数: {count}"

    @staticmethod
    def _integer_polynomial_divisibility(problem: str) -> Optional[str]:
        match = re.search(r"\$([^$]+)\$\s+divides\s+\$([^$]+)\$", problem, re.IGNORECASE)
        if (
            not match
            or not re.search(
                r"(?:complete\s+set\s+of|determine\s+all|find\s+all)\s+integers?\s+\$?n\$?",
                problem,
                re.IGNORECASE,
            )
        ):
            return None
        divisor = _integer_polynomial(match.group(1))
        dividend = _integer_polynomial(match.group(2))
        if (
            divisor is None or dividend is None
            or len(divisor) < 2 or len(dividend) <= len(divisor)
            or abs(divisor[-1]) != 1
        ):
            return None
        remainder = _polynomial_remainder(dividend, divisor)
        if remainder is None or all(coefficient == 0 for coefficient in remainder):
            return None
        lower_sum = sum(abs(coefficient) for coefficient in divisor[:-1])
        remainder_sum = sum(abs(coefficient) for coefficient in remainder)
        bound = lower_sum + remainder_sum + 1
        if bound > 1_000_000:
            return None
        solutions = []
        for integer in range(-bound, bound + 1):
            divisor_value = _polynomial_value(divisor, integer)
            if divisor_value and _polynomial_value(dividend, integer) % divisor_value == 0:
                solutions.append(integer)
        answer = r"\varnothing" if not solutions else r"\{" + ",".join(map(str, solutions)) + r"\}"
        return f"本地整数多项式整除解集: {answer}"

    @staticmethod
    def _reciprocal_quartic_nonnegative(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        compact = compact.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        compact = compact.replace("{", "").replace("}", "").replace(r"\geq", ">=").replace(r"\ge", ">=")
        match = re.search(r"x\^4\+ax\^3\+([-+]?\d+)x\^2\+ax\+1>=0", compact)
        if (
            not match
            or not re.search(r"determine\s+all\s+real\s+numbers\s+.*for\s+which", problem, re.IGNORECASE)
            or not re.search(r"for\s+every\s+real\s+.*x", problem, re.IGNORECASE)
        ):
            return None
        middle = int(match.group(1))
        if middle < -2:
            answer = r"\varnothing"
        elif middle <= 6:
            endpoint = Fraction(middle + 2, 2)
            endpoint_text = _fraction_text(endpoint)
            answer = f"[-{endpoint_text},{endpoint_text}]"
        else:
            radicand = middle - 2
            root = math.isqrt(radicand)
            endpoint_text = str(2 * root) if root * root == radicand else rf"2\sqrt{{{radicand}}}"
            answer = f"[-{endpoint_text},{endpoint_text}]"
        return f"本地回文四次式非负参数: {answer}"

    @staticmethod
    def _affine_recurrence_determinant(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        compact = compact.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        initial_zero = re.search(r"a_?\{?0\}?=([-+]?\d+)", compact)
        initial_one = re.search(r"a_?\{?1\}?=([-+]?\d+)", compact)
        recurrence = re.search(
            r"a_?\{?n\+2\}?=([-+]?\d+)a_?\{?n\+1\}?-a_?\{?n\}?([+-]\d+)",
            compact,
        )
        target = re.search(
            r"\(a_?\{?(\d+)\}?([+-]\d+)\)\(a_?\{?(\d+)\}?([+-]\d+)\)"
            r"-\(a_?\{?(\d+)\}?([+-]\d+)\)\^2",
            compact,
        )
        if (
            not initial_zero or not initial_one or not recurrence or not target
            or not re.search(r"\bevaluate\b", problem, re.IGNORECASE)
        ):
            return None
        first_index, first_shift, last_index, last_shift, middle_index, middle_shift = map(int, target.groups())
        multiplier, constant = map(int, recurrence.groups())
        if (
            first_index != middle_index + 1
            or last_index != middle_index - 1
            or len({first_shift, middle_shift, last_shift}) != 1
            or multiplier == 2
            or Fraction(constant, multiplier - 2) != first_shift
        ):
            return None
        shifted_zero = Fraction(int(initial_zero.group(1)) + first_shift)
        shifted_one = Fraction(int(initial_one.group(1)) + first_shift)
        shifted_two = multiplier * shifted_one - shifted_zero
        invariant = shifted_two * shifted_zero - shifted_one * shifted_one
        return f"本地仿射递推行列式不变量: {_fraction_text(invariant)}"

    @staticmethod
    def _root_polynomial_product(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        compact = compact.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        compact = compact.replace(r"\displaystyle", "")
        roots = re.search(
            r"\\alpha_1,\\ldots,\\alpha_(\d+)bethecomplexroots,countedwithmultiplicity,"
            r"of([0-9x+\-^]+)=0",
            compact,
            re.IGNORECASE,
        )
        product = re.search(
            r"Evaluate\\prod_\{j=1\}\^\{?(\d+)\}?\(([^()]+)\)",
            compact,
            re.IGNORECASE,
        )
        if not roots or not product:
            return None
        root_count = int(roots.group(1))
        product_count = int(product.group(1))
        polynomial = _integer_polynomial(roots.group(2), "x")
        factor_text = product.group(2).replace(r"\alpha_j", "x")
        factor = _integer_polynomial(factor_text, "x")
        if (
            polynomial is None or factor is None
            or len(polynomial) - 1 != root_count or product_count != root_count
            or polynomial[-1] != 1
            or not 1 <= root_count <= 12
        ):
            return None
        try:
            import sympy

            variable = sympy.Symbol("x")
            left = sympy.Poly.from_list(list(reversed(polynomial)), gens=variable)
            right = sympy.Poly.from_list(list(reversed(factor)), gens=variable)
            result = int(sympy.resultant(left.as_expr(), right.as_expr(), variable))
        except Exception:
            return None
        return f"本地根上多项式乘积: {result}"

    @staticmethod
    def _cevian_length(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        compact = compact.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        sides = re.search(
            r"IntriangleABC,AB=(\d+),AC=(\d+),andBC=(\d+)",
            compact,
            re.IGNORECASE,
        )
        ratio = re.search(
            r"ApointDonBCsatisfiesBD:DC=(\d+):(\d+)",
            compact,
            re.IGNORECASE,
        )
        if not sides or not ratio or not re.search(r"FindAD(?:\.|Remember|$)", compact, re.IGNORECASE):
            return None
        side_ab, side_ac, side_bc = map(int, sides.groups())
        left_ratio, right_ratio = map(int, ratio.groups())
        if (
            min(side_ab, side_ac, side_bc, left_ratio, right_ratio) <= 0
            or side_ab + side_ac <= side_bc
            or side_ab + side_bc <= side_ac
            or side_ac + side_bc <= side_ab
        ):
            return None
        left_segment = Fraction(side_bc * left_ratio, left_ratio + right_ratio)
        right_segment = Fraction(side_bc * right_ratio, left_ratio + right_ratio)
        length_squared = (
            Fraction(side_ac * side_ac) * left_segment
            + Fraction(side_ab * side_ab) * right_segment
        ) / side_bc - left_segment * right_segment
        if length_squared <= 0:
            return None
        numerator_root = math.isqrt(length_squared.numerator)
        denominator_root = math.isqrt(length_squared.denominator)
        if numerator_root**2 == length_squared.numerator and denominator_root**2 == length_squared.denominator:
            answer = _fraction_text(Fraction(numerator_root, denominator_root))
        elif denominator_root**2 == length_squared.denominator:
            answer = (
                rf"\sqrt{{{length_squared.numerator}}}"
                if denominator_root == 1
                else rf"\frac{{\sqrt{{{length_squared.numerator}}}}}{{{denominator_root}}}"
            )
        else:
            answer = rf"\sqrt{{{_fraction_text(length_squared)}}}"
        return f"本地三角形劈线长度: {answer}"

    @staticmethod
    def _smith_normal_form(problem: str) -> Optional[str]:
        matrix_match = re.search(
            r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}",
            problem,
            re.DOTALL,
        )
        if (
            not matrix_match
            or not re.search(r"整数矩阵", problem)
            or not re.search(r"Smith\s*标准形", problem, re.IGNORECASE)
        ):
            return None
        rows = [
            [cell.strip() for cell in row.split("&")]
            for row in re.split(r"\\\\", matrix_match.group(1))
        ]
        if (
            not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows)
            or len(rows) > 6 or len(rows[0]) > 6
            or any(not re.fullmatch(r"[-+]?\d+", cell) for row in rows for cell in row)
        ):
            return None
        values = [[int(cell) for cell in row] for row in rows]
        try:
            import sympy

            matrix = sympy.Matrix(values)
            previous = 1
            invariants: list[int] = []
            for size in range(1, min(matrix.rows, matrix.cols) + 1):
                minor_gcd = 0
                for row_indices in combinations(range(matrix.rows), size):
                    for column_indices in combinations(range(matrix.cols), size):
                        determinant = abs(int(matrix.extract(row_indices, column_indices).det()))
                        minor_gcd = math.gcd(minor_gcd, determinant)
                if minor_gcd == 0:
                    break
                if minor_gcd % previous:
                    return None
                invariants.append(minor_gcd // previous)
                previous = minor_gcd
        except Exception:
            return None
        invariants.extend([0] * (min(len(rows), len(rows[0])) - len(invariants)))
        answer = r"\operatorname{diag}(" + ",".join(map(str, invariants)) + ")"
        return f"本地Smith标准形: {answer}"

    @staticmethod
    def _intersecting_antichain_maximum(problem: str) -> Optional[str]:
        universe = re.search(
            r"family\s+\$?F\$?\s+of\s+subsets\s+of\s+"
            r"\{\s*1\s*,\s*2\s*,\s*(?:\.\.\.|\\ldots|\\dots)\s*,?\s*(\d+)\s*\}",
            problem,
            re.IGNORECASE,
        )
        if (
            not universe
            or not re.search(
                r"called\s+intersecting\s+if\s+every\s+two\s+members\s+have\s+nonempty\s+intersection",
                problem,
                re.IGNORECASE,
            )
            or not re.search(
                r"(?:is\s+)?an\s+antichain\s+if\s+no\s+member\s+contains\s+another",
                problem,
                re.IGNORECASE,
            )
            or not re.search(r"maximum\s+possible\s+value\s+of\s+\$?\|F\|\$?", problem, re.IGNORECASE)
        ):
            return None
        size = int(universe.group(1))
        if not 1 <= size <= 10000:
            return None
        result = math.comb(size, size // 2 + 1)
        return f"本地相交反链最大值: {result}"

    @staticmethod
    def _bipartite_matching_deletion_trees(problem: str) -> Optional[str]:
        graph = re.search(
            r"complete\s+bipartite\s+graph\s+\$?K_?\{?(\d+)\s*,\s*(\d+)\}?\$?",
            problem,
            re.IGNORECASE,
        )
        deletion = re.search(
            r"delete\s+the\s+(\d+|[A-Za-z]+)\s+independent\s+edges\s+\$?u_i\s*v_i\$?",
            problem,
            re.IGNORECASE,
        )
        bound = re.search(r"1\s*(?:<=|\\le)\s*i\s*(?:<=|\\le)\s*(\d+)", problem)
        if (
            not graph or not deletion or not bound
            or not re.search(r"number\s+of\s+spanning\s+trees", problem, re.IGNORECASE)
        ):
            return None
        left, right = map(int, graph.groups())
        deleted_count = _small_number(deletion.group(1))
        index_bound = int(bound.group(1))
        if (
            deleted_count is None or deleted_count != index_bound
            or not 1 <= deleted_count <= min(left, right)
            or left + right > 60
        ):
            return None
        try:
            import sympy

            order = left + right
            laplacian = sympy.zeros(order)
            for left_index in range(left):
                for right_index in range(right):
                    if left_index == right_index and left_index < deleted_count:
                        continue
                    target = left + right_index
                    laplacian[left_index, left_index] += 1
                    laplacian[target, target] += 1
                    laplacian[left_index, target] = -1
                    laplacian[target, left_index] = -1
            result = int(laplacian[:-1, :-1].det(method="domain-ge"))
        except Exception:
            return None
        return f"本地二部图删匹配生成树: {result}"

    @staticmethod
    def _cyclic_nonadjacent_selection(problem: str) -> Optional[str]:
        seats = re.search(r"(\d+|[A-Za-z]+)\s+labeled\s+seats", problem, re.IGNORECASE)
        selected = re.search(
            r"how\s+many\s+ways\s+can\s+(\d+|[A-Za-z]+)\s+of\s+the\s+seats\s+be\s+selected",
            problem,
            re.IGNORECASE,
        )
        if (
            not seats or not selected
            or not re.search(r"circular\s+table", problem, re.IGNORECASE)
            or not re.search(r"no\s+two\s+selected\s+seats\s+are\s+adjacent", problem, re.IGNORECASE)
        ):
            return None
        n, k = _small_number(seats.group(1)), _small_number(selected.group(1))
        if n is None or k is None or not 0 <= k <= n <= 100:
            return None
        count = 1 if k == 0 else (0 if n < 2 * k else n * math.comb(n - k - 1, k - 1) // k)
        return f"本地圆周不相邻选择计数: {count}"

    @staticmethod
    def _finite_subtraction_game(problem: str) -> Optional[str]:
        rules = re.search(r"removing\s+exactly\s+(.+?)\s+stones", problem, re.IGNORECASE)
        bound = re.search(r"1\s*\\le\s*n\s*\\le\s*(\d+)", problem)
        if (
            not rules or not bound
            or not re.search(r"two\s+players\s+alternate", problem, re.IGNORECASE)
            or not re.search(r"taking\s+the\s+last\s+stone\s+wins", problem, re.IGNORECASE)
            or not re.search(
                r"(?:losing\s+positions?|position\s+losing|initial\s+position\s+losing)",
                problem,
                re.IGNORECASE,
            )
        ):
            return None
        moves = sorted(set(map(int, re.findall(r"\d+", rules.group(1)))))
        limit = int(bound.group(1))
        if not moves or moves[0] <= 0 or len(moves) > 20 or limit > 10**6:
            return None
        losing = [True] + [False] * limit
        for size in range(1, limit + 1):
            losing[size] = not any(move <= size and losing[size - move] for move in moves)
        return f"本地减法博弈必败态计数: {sum(losing[1:])}"

    @staticmethod
    def _square_subtraction_game(problem: str) -> Optional[str]:
        bound = re.search(r"1\s*(?:<=|\\le)\s*n\s*(?:<=|\\le)\s*(\d+)", problem)
        if (
            not bound
            or not re.search(r"heap\s+initially\s+contains\s+\$?n\$?\s+stones", problem, re.IGNORECASE)
            or not re.search(r"positive\s+perfect[- ]square", problem, re.IGNORECASE)
            or not re.search(r"no\s+legal\s+move\s+loses", problem, re.IGNORECASE)
            or not re.search(
                r"(?:losing\s+positions?|position\s+losing|initial\s+position\s+losing)",
                problem,
                re.IGNORECASE,
            )
        ):
            return None
        limit = int(bound.group(1))
        if not 1 <= limit <= 10**6:
            return None
        moves = [value * value for value in range(1, math.isqrt(limit) + 1)]
        losing = [True] + [False] * limit
        for size in range(1, limit + 1):
            losing[size] = not any(move <= size and losing[size - move] for move in moves)
        return f"本地平方减法博弈计数: {sum(losing[1:])}"

    @staticmethod
    def _wheel_coloring(problem: str) -> Optional[str]:
        cycle = re.search(r"cycle\s+\$?C_?\{?(\d+)\}?\$?", problem, re.IGNORECASE)
        colors = re.search(r"using\s+(\d+|[A-Za-z]+)\s+labeled\s+colors", problem, re.IGNORECASE)
        if (
            not cycle or not colors
            or not re.search(r"joining\s+one\s+new\s+vertex\s+to\s+every\s+vertex", problem, re.IGNORECASE)
            or not re.search(r"proper\s+vertex\s+colorings", problem, re.IGNORECASE)
        ):
            return None
        n, q = int(cycle.group(1)), _small_number(colors.group(1))
        if q is None or not 3 <= q <= 100 or not 3 <= n <= 10**6:
            return None
        count = q * ((q - 2) ** n + (-1) ** n * (q - 2))
        return f"本地轮图正常着色计数: {count}"

    @staticmethod
    def _grid_poset_extensions(problem: str) -> Optional[str]:
        match = re.search(
            r"poset\s+\$?\\\{([0-9,\s]+)\\\}\\times\\\{([0-9,\s]+)\\\}",
            problem,
            re.IGNORECASE,
        )
        if (
            not match
            or not re.search(r"exactly\s+when\s+\$?i\\le\s*k\$?\s+and\s+\$?j\\le", problem, re.IGNORECASE)
            or not re.search(r"linear\s+extensions", problem, re.IGNORECASE)
        ):
            return None
        first_values = [int(item) for item in match.group(1).split(",")]
        second_values = [int(item) for item in match.group(2).split(",")]
        if first_values != list(range(1, len(first_values) + 1)) or second_values != list(
            range(1, len(second_values) + 1)
        ):
            return None
        rows, columns = len(first_values), len(second_values)
        if not 1 <= rows * columns <= 10000:
            return None
        result = math.factorial(rows * columns)
        for row in range(rows):
            for column in range(columns):
                result //= rows + columns - row - column - 1
        return f"本地网格偏序线性扩张计数: {result}"

    @staticmethod
    def _hypercube_spanning_trees(problem: str) -> Optional[str]:
        match = re.search(r"(?:hypercube\s+\$?Q_?\{?|\$?Q_?\{?)(\d+)\}?\$?", problem, re.IGNORECASE)
        if (
            not match
            or not re.search(r"binary\s+strings", problem, re.IGNORECASE)
            or not re.search(r"differ\s+in\s+exactly\s+one\s+coordinate", problem, re.IGNORECASE)
            or not re.search(r"spanning\s+trees", problem, re.IGNORECASE)
        ):
            return None
        dimension = int(match.group(1))
        if not 1 <= dimension <= 20:
            return None
        result = 2 ** (2**dimension - dimension - 1)
        for index in range(1, dimension + 1):
            result *= index ** math.comb(dimension, index)
        return f"本地超立方体生成树计数: {result}"

    @staticmethod
    def _odd_fiber_functions(problem: str) -> Optional[str]:
        domain = re.search(r"functions\s+from\s+\{1,2,(?:\.\.\.|\\ldots|\\dots),?(\d+)\}", problem, re.IGNORECASE)
        codomain = re.search(r"\s+to\s+\{([0-9,\s]+)\}", problem, re.IGNORECASE)
        if (
            not domain or not codomain
            or not re.search(r"every\s+fiber\s+has\s+odd\s+cardinality", problem, re.IGNORECASE)
            or not re.search(r"fiber\s+of\s+cardinality\s+zero\s+is\s+not\s+considered\s+odd", problem, re.IGNORECASE)
            or not re.search(r"codomain\s+elements\s+are\s+labeled", problem, re.IGNORECASE)
        ):
            return None
        n = int(domain.group(1))
        values = [int(item) for item in codomain.group(1).split(",")]
        if values != list(range(1, len(values) + 1)) or n > 200:
            return None
        states = {0: 1}
        for _ in values:
            updated: dict[int, int] = defaultdict(int)
            for used, count in states.items():
                for size in range(1, n - used + 1, 2):
                    updated[used + size] += count * math.comb(n - used, size)
            states = updated
        return f"本地奇数纤维函数计数: {states.get(n, 0)}"

    @staticmethod
    def _couples_unlabeled_groups(problem: str) -> Optional[str]:
        couples_match = re.search(r"(\d+|[A-Za-z]+)\s+married\s+couples", problem, re.IGNORECASE)
        groups_match = re.search(
            r"partitioned\s+into\s+(\d+|[A-Za-z]+)\s+unlabeled\s+groups\s+of\s+(\d+|[A-Za-z]+)",
            problem,
            re.IGNORECASE,
        )
        if (
            not couples_match or not groups_match
            or not re.search(r"all\s+.*people\s+distinct", problem, re.IGNORECASE)
            or not re.search(r"no\s+group\s+contain\s+both\s+members", problem, re.IGNORECASE)
        ):
            return None
        couples = _small_number(couples_match.group(1))
        group_count = _small_number(groups_match.group(1))
        group_size = _small_number(groups_match.group(2))
        if None in {couples, group_count, group_size}:
            return None
        assert couples is not None and group_count is not None and group_size is not None
        if 2 * couples != group_count * group_size or couples > 12 or group_count > 8:
            return None
        states = {(0,) * group_count: 1}
        for _ in range(couples):
            updated: dict[tuple[int, ...], int] = defaultdict(int)
            for capacities, count in states.items():
                for first in range(group_count):
                    if capacities[first] >= group_size:
                        continue
                    for second in range(group_count):
                        if second == first or capacities[second] >= group_size:
                            continue
                        new = list(capacities)
                        new[first] += 1
                        new[second] += 1
                        updated[tuple(new)] += count
            states = updated
        labeled = states.get((group_size,) * group_count, 0)
        return f"本地夫妻分组计数: {labeled // math.factorial(group_count)}"

    @staticmethod
    def _bounded_divisor_count(problem: str) -> Optional[str]:
        match = re.search(
            r"how\s+many\s+positive\s+integers\s+\$?n\s*(?:<=|\\le)\s*(\d+)\$?\s+"
            r"have\s+exactly\s+\$?(\d+)\$?\s+positive\s+divisors",
            problem,
            re.IGNORECASE,
        )
        if not match:
            return None
        bound, target = map(int, match.groups())
        if not 1 <= bound <= 2_000_000 or not 1 <= target <= 10000:
            return None
        count = 0
        for value in range(1, bound + 1):
            divisor_count = 1
            for exponent in _prime_factors(value).values():
                divisor_count *= exponent + 1
            count += divisor_count == target
        return f"本地约数个数范围计数: {count}"

    @staticmethod
    def _primitive_pythagorean_count(problem: str) -> Optional[str]:
        bound = re.search(r"primitive\s+Pythagorean\s+triples\s+have\s+\$?c\s*(?:<=|\\le)\s*(\d+)", problem, re.IGNORECASE)
        if (
            not bound
            or not re.search(r"a\s*<\s*b\s*<\s*c", problem)
            or not re.search(r"gcd\s*\(a,b,c\)\s*=\s*1", problem, re.IGNORECASE)
        ):
            return None
        limit = int(bound.group(1))
        if not 5 <= limit <= 10**8:
            return None
        count = 0
        for larger in range(2, math.isqrt(limit) + 1):
            for smaller in range(1, larger):
                if larger * larger + smaller * smaller > limit:
                    break
                if (larger - smaller) % 2 and math.gcd(larger, smaller) == 1:
                    count += 1
        return f"本地本原勾股三元组计数: {count}"

    @staticmethod
    def _inverse_totient(problem: str) -> Optional[str]:
        match = re.search(r"(?:\\varphi|φ)\s*\(\s*n\s*\)\s*=\s*(\d+)", problem)
        if not match or not re.search(r"determine\s+all\s+positive\s+integers", problem, re.IGNORECASE):
            return None
        target = int(match.group(1))
        bound = max(2, 2 * target * target)
        if target < 1 or bound > 2_000_000:
            return None
        solutions = [value for value in range(1, bound + 1) if _totient(value) == target]
        answer = r"\{" + ",".join(map(str, solutions)) + r"\}"
        return f"本地欧拉函数逆像: {answer}"

    @staticmethod
    def _gcd_sum(problem: str) -> Optional[str]:
        match = re.search(
            r"\\sum_\{k=1\}\^\{(\d+)\}\\gcd\s*\(\s*k\s*,\s*(\d+)\s*\)",
            problem,
        )
        if not match or match.group(1) != match.group(2) or not re.search(r"evaluate", problem, re.IGNORECASE):
            return None
        value = int(match.group(1))
        if not 1 <= value <= 10**12:
            return None
        result = sum(divisor * _totient(value // divisor) for divisor in _divisors(value))
        return f"本地最大公约数求和: {result}"

    @staticmethod
    def _positive_sum_two_squares(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem).replace(r"\(", "").replace(r"\)", "")
        match = re.search(r"x\^2\+y\^2=([0-9^{}\\cdot*]+)", compact)
        if (
            not match
            or not re.search(r"orderedpairs", compact, re.IGNORECASE)
            or not re.search(r"positiveintegers", compact, re.IGNORECASE)
        ):
            return None
        target = _positive_product(match.group(1))
        if target is None or target > 10**12:
            return None
        count = 0
        for x_value in range(1, math.isqrt(target) + 1):
            remainder = target - x_value * x_value
            if remainder <= 0:
                continue
            y_value = math.isqrt(remainder)
            count += y_value > 0 and y_value * y_value == remainder
        return f"本地正整数平方和计数: {count}"

    @staticmethod
    def _factorial_quotient_valuation(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem).replace(r"\,", "")
        quotient = re.search(r"M=\\d?frac\{(\d+)!\}\{([^{}]+)\}", compact)
        request = re.search(r"(\d+)\^k\\midM", compact)
        if not quotient or not request or not re.search(r"greatestinteger", compact, re.IGNORECASE):
            return None
        numerator = int(quotient.group(1))
        denominator_text = quotient.group(2)
        denominators = [int(item) for item in re.findall(r"(\d+)!", denominator_text)]
        unparsed_denominator = re.sub(r"\d+!", "", denominator_text)
        unparsed_denominator = (
            unparsed_denominator.replace(r"\,", "")
            .replace(r"\cdot", "")
            .replace("*", "")
        )
        base = int(request.group(1))
        if (
            not denominators or unparsed_denominator
            or any(item > numerator for item in denominators)
            or numerator > 10**9 or base <= 1
        ):
            return None

        def factorial_valuation(limit: int, prime: int) -> int:
            total = 0
            while limit:
                limit //= prime
                total += limit
            return total

        exponents = []
        for prime, base_exponent in _prime_factors(base).items():
            valuation = factorial_valuation(numerator, prime) - sum(
                factorial_valuation(item, prime) for item in denominators
            )
            if valuation < 0:
                return None
            exponents.append(valuation // base_exponent)
        return f"本地阶乘商复合估值: {min(exponents)}"

    @staticmethod
    def _pell_fundamental_solution(problem: str) -> Optional[str]:
        match = re.search(r"x\^2\s*-\s*(\d+)\s*y\^2\s*=\s*1", problem)
        if (
            not match
            or not re.search(r"Pell\s+equation", problem, re.IGNORECASE)
            or not re.search(r"smallest\s+possible.{0,12}x", problem, re.IGNORECASE)
            or not re.search(r"ordered\s+pair", problem, re.IGNORECASE)
        ):
            return None
        nonsquare = int(match.group(1))
        root = math.isqrt(nonsquare)
        if nonsquare <= 1 or root * root == nonsquare or nonsquare > 10**6:
            return None
        m_value, denominator, coefficient = 0, 1, root
        p_prev, p_value = 1, coefficient
        q_prev, q_value = 0, 1
        iterations = 0
        while p_value * p_value - nonsquare * q_value * q_value != 1:
            m_value = denominator * coefficient - m_value
            denominator = (nonsquare - m_value * m_value) // denominator
            coefficient = (root + m_value) // denominator
            p_prev, p_value = p_value, coefficient * p_value + p_prev
            q_prev, q_value = q_value, coefficient * q_value + q_prev
            iterations += 1
            if iterations > 2_000_000:
                return None
        return f"本地Pell基本解: ({p_value},{q_value})"

    @staticmethod
    def _least_integer_with_divisor_count(problem: str) -> Optional[str]:
        match = re.search(r"least\s+positive\s+integer\s+having\s+exactly\s+\\?\(?\s*(\d+)\s*\\?\)?\s+positive\s+divisors", problem, re.IGNORECASE)
        if not match:
            return None
        target = int(match.group(1))
        if not 1 <= target <= 100000:
            return None
        primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
        best: Optional[int] = None

        def search(index: int, remaining: int, max_exponent: int, value: int) -> None:
            nonlocal best
            if remaining == 1:
                best = value if best is None else min(best, value)
                return
            if index >= len(primes):
                return
            for factor in _divisors(remaining):
                exponent = factor - 1
                if factor <= 1 or exponent > max_exponent:
                    continue
                updated = value * primes[index] ** exponent
                if best is not None and updated >= best:
                    continue
                search(index + 1, remaining // factor, exponent, updated)

        search(0, target, target - 1, 1)
        return None if best is None else f"本地最小约数数目整数: {best}"

    @staticmethod
    def _factorable_binary_quadratic(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        match = re.search(r"x\^2\+(\d+)xy\+(\d+)y\^2=(\d+)", compact)
        if not match or not re.search(r"allorderedpairsofintegers", compact, re.IGNORECASE):
            return None
        middle, last, target = map(int, match.groups())
        roots = [value for value in range(1, middle) if value * (middle - value) == last]
        if not roots:
            return None
        first, second = min(roots), max(roots)
        if second - first != 1:
            return None
        answer = (
            r"\left\{\left("
            f"{second}d-\\frac{{{first * target}}}{{d}},\\frac{{{target}}}{{d}}-d"
            r"\right):d\in\mathbb Z,\ d\mid"
            f"{target}"
            r"\right\}"
        )
        return f"本地可分解二次型整数解: {answer}"

    @staticmethod
    def _descartes_inner_circle(problem: str) -> Optional[str]:
        normalized = problem.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        radii = re.search(
            r"radii\s+(\d+)\s*,\s*(\d+)\s*,\s*and\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        if (
            not radii
            or not re.search(r"pairwise\s+externally\s+tangent", problem, re.IGNORECASE)
            or not re.search(r"bounded\s+gap", problem, re.IGNORECASE)
            or not re.search(r"externally\s+tangent\s+to\s+all\s+three", problem, re.IGNORECASE)
            or not re.search(r"find\s+the\s+radius\s+of\s+the\s+fourth\s+circle", problem, re.IGNORECASE)
        ):
            return None
        values = tuple(map(int, radii.groups()))
        curvatures = tuple(Fraction(1, radius) for radius in values if radius > 0)
        if len(curvatures) != 3:
            return None
        pair_sum = sum((curvatures[i] * curvatures[j] for i, j in combinations(range(3), 2)), Fraction())
        numerator_root = math.isqrt(pair_sum.numerator)
        denominator_root = math.isqrt(pair_sum.denominator)
        if numerator_root**2 != pair_sum.numerator or denominator_root**2 != pair_sum.denominator:
            return None
        fourth_curvature = sum(curvatures, Fraction()) + 2 * Fraction(numerator_root, denominator_root)
        radius = 1 / fourth_curvature
        answer = str(radius.numerator) if radius.denominator == 1 else f"{radius.numerator}/{radius.denominator}"
        return f"本地Descartes内切圆半径: {answer}"

    @staticmethod
    def _rotation_necklace_fixed_weight(problem: str) -> Optional[str]:
        polygon = re.search(r"正([一二两三四五六七八九十\d]+)边形", problem)
        red = re.search(r"恰有([一二两三四五六七八九十\d]+)个顶点为红色", problem)
        if (
            not polygon or not red
            or not re.search(r"只把旋转后重合", problem)
            or re.search(r"反射|翻转|reflection", problem, re.IGNORECASE)
        ):
            return None
        n, k = _small_number(polygon.group(1)), _small_number(red.group(1))
        if n is None or k is None or not 0 <= k <= n <= 30 or math.comb(n, k) > 2_000_000:
            return None
        representatives = set()
        for selected in combinations(range(n), k):
            bits = tuple(int(index in selected) for index in range(n))
            representatives.add(min(bits[offset:] + bits[:offset] for offset in range(n)))
        return f"本地定重旋转项链计数: {len(representatives)}"

    @staticmethod
    def _bose_einstein_integral(problem: str) -> Optional[str]:
        compact = re.sub(r"\s+", "", problem)
        match = re.search(
            r"\\int_0\^\{?\\infty\}?(?:\\d?frac)?\{?x\^(\d+)\}?\{?e\^x-1\}?\\,?dx",
            compact,
        )
        if not match or not re.search(r"广义积分|improper\s+integral|evaluate", problem, re.IGNORECASE):
            return None
        power = int(match.group(1))
        if power < 1 or power % 2 == 0 or power > 15:
            return None
        order = power + 1

        bernoulli = [Fraction(0) for _ in range(order + 1)]
        work = [Fraction(0) for _ in range(order + 1)]
        for m_value in range(order + 1):
            work[m_value] = Fraction(1, m_value + 1)
            for j_value in range(m_value, 0, -1):
                work[j_value - 1] = j_value * (work[j_value - 1] - work[j_value])
            bernoulli[m_value] = work[0]
        half_order = order // 2
        coefficient = (
            (-1) ** (half_order + 1)
            * bernoulli[order]
            * Fraction(2 ** (order - 1), order)
        )
        if coefficient <= 0:
            return None
        if coefficient.denominator == 1:
            answer = f"{coefficient.numerator}*pi**{order}"
        elif coefficient.numerator == 1:
            answer = f"pi**{order}/{coefficient.denominator}"
        else:
            answer = f"{coefficient.numerator}*pi**{order}/{coefficient.denominator}"
        return f"本地Bose积分: {answer}"

    @staticmethod
    def _bernoulli_likelihood_ratio(problem: str) -> Optional[str]:
        sample = re.search(r"Bernoulli\s*样本量为\s*\$?(\d+)\$?", problem, re.IGNORECASE)
        successes = re.search(r"观察到\s*\$?(\d+)\$?\s*次成功", problem)
        null = re.search(r"H_0\s*:\s*p\s*=\s*(\d+)\s*/\s*(\d+)", problem)
        if (
            not sample or not successes or not null
            or not re.search(r"H_1\s*:\s*p\s*\\ne", problem)
            or not re.search(r"-2\\log\\Lambda", problem)
            or not re.search(r"精确表达式", problem)
        ):
            return None
        count, success = int(sample.group(1)), int(successes.group(1))
        probability = Fraction(int(null.group(1)), int(null.group(2)))
        if not 0 < success < count or not 0 < probability < 1 or count > 10**7:
            return None
        arguments = (
            (2 * success, Fraction(success, count) / probability),
            (2 * (count - success), Fraction(count - success, count) / (1 - probability)),
        )
        coefficients: dict[int, int] = defaultdict(int)
        for multiplier, argument in arguments:
            for prime, exponent in _prime_factors(argument.numerator).items():
                coefficients[prime] += multiplier * exponent
            for prime, exponent in _prime_factors(argument.denominator).items():
                coefficients[prime] -= multiplier * exponent
        terms: list[tuple[int, int]] = []
        for prime in sorted(coefficients, reverse=True):
            coefficient = coefficients[prime]
            if coefficient:
                terms.append((coefficient, prime))
        rendered = []
        for index, (coefficient, prime) in enumerate(terms):
            sign = "-" if coefficient < 0 else ("+" if index else "")
            rendered.append(f"{sign}{abs(coefficient)}\\ln {prime}")
        answer = "".join(rendered)
        return f"本地Bernoulli似然比: {answer}"

    @staticmethod
    def _brownian_exit_expectation(problem: str) -> Optional[str]:
        start = re.search(
            r"Brownian\s+运动从\s*\$?([-+]?\d+(?:/\d+)?)\$?\s*出发",
            problem,
            re.IGNORECASE,
        )
        interval = re.search(
            r"首次离开区间\s*\$?\(\s*([-+]?\d+(?:/\d+)?)\s*,\s*"
            r"([-+]?\d+(?:/\d+)?)\s*\)\$?",
            problem,
        )
        if (
            not start or not interval
            or not re.search(r"标准\s*Brownian", problem, re.IGNORECASE)
            or not re.search(
                r"求\s*\$?\s*(?:\\mathbb\s*\{?E\}?|E)\s*\[\s*\\tau\s*\]\s*\$?",
                problem,
            )
        ):
            return None
        point = Fraction(start.group(1))
        lower, upper = map(Fraction, interval.groups())
        if not lower < point < upper:
            return None
        result = (point - lower) * (upper - point)
        answer = str(result.numerator) if result.denominator == 1 else f"{result.numerator}/{result.denominator}"
        return f"本地Brownian离区间期望: {answer}"
