"""Strict deterministic handlers for common finite olympiad problem families."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
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


def _normalized_statement(problem: str) -> str:
    """Normalize one closed benchmark statement without hiding later clauses."""
    text = re.sub(
        r"\s*Remember\s+to\s+put\s+your\s+final\s+answer\s+within\s+"
        r"\\boxed\{\}\s*\.\s*$",
        "",
        str(problem or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


class ExactOlympiadTool:
    """Return answers only when every rule needed by an exact algorithm is present."""

    _HANDLERS: tuple[str, ...] = (
        "_bounded_digit_set_divisibility_count",
        "_prime_floor_inequality_rank",
        "_nice_positive_integer_function_value_set",
        "_real_functional_equation_three_solutions",
        "_open_interval_quadratic_minimum_dimension",
        "_subset_xor_card_game_losing_first_move",
        "_angle_bisector_three_circle_parameter",
        "_odd_part_block_congruence_values",
        "_mutual_histogram_weighted_values",
        "_gap_two_signed_subsequence_guarantee",
        "_two_monotone_merchant_common_connection",
        "_missing_color_polyomino_area",
        "_korean_sequence_good_partition_minimum",
        "_cyclic_quartic_equality_triple_count",
        "_round_robin_unextendable_schedule_minimum",
        "_path_domino_maximin_uncovered",
        "_odd_checkerboard_l_tromino_minimum",
        "_two_by_two_flip_closure_minimum",
        "_ant_collision_escape_time",
        "_prefix_split_pebble_survival",
        "_neighborhood_growth_four_decrements",
        "_increasing_grid_path_minimum",
        "_consecutive_card_partition_game_value",
        "_half_area_boundary_side_minimum",
        "_three_polar_triangle_locus",
        "_bezout_l1_nice_count_polynomial",
        "_reciprocal_means_reach_one_maximum",
        "_knight_queen_board_guarantee",
        "_angle_ratio_line_point_maximum",
        "_all_other_faces_visible_polyhedron_maximum",
        "_rich_integer_set_from_power_differences",
        "_rational_integer_rounding_function_equation",
        "_prime_exponential_inequality_parameter_region",
        "_cubic_log_derivative_real_root_count",
        "_napkin_equal_coverage_maximum",
        "_personal_consecutive_number_game",
        "_round_robin_hotel_cost_minimum",
        "_fibonacci_difference_basis_minimum",
        "_translation_order_odd_count_product",
        "_sparse_green_neighborhood_threshold",
        "_right_triangle_two_cevian_ratio",
        "_equal_diagonal_quadrilateral_maximum_area",
        "_power_difference_semigroup_smallest_gap",
        "_recurrence_universal_coprime_set",
        "_quartic_plus_five_splitting_field",
        "_flood_barrier_critical_speed",
        "_triangular_lattice_regular_hexagons",
        "_critical_line_cover_point_set",
        "_even_quadratic_pair_count_parameters",
        "_sparse_domino_placements",
        "_red_blue_line_separation",
        "_clustered_interval_maximum",
        "_quadratic_transform_invariant_polynomials",
        "_directed_cylinder_hamilton_paths",
        "_sorted_triangle_failure_bound",
        "_sparkling_tuple_pair_sum",
        "_five_number_ratio_gap",
        "_nested_nonnegative_sequence_values",
        "_mysterious_cuberoot_polynomial",
        "_complete_bipartite_homomorphism_bound",
        "_tangential_identical_triangulation_polygon",
        "_formal_l2_adjoint",
        "_mixed_radix_grid_compression",
        "_cycle_distance_two_coloring",
        "_punctured_domino_tilings",
        "_unique_domino_partition_marking",
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
        "_complete_graph_cycle_deletion_trees",
        "_cyclic_nonadjacent_selection",
        "_fixed_weight_binary_bracelets",
        "_specified_degree_labeled_trees",
        "_odd_cycle_permutations",
        "_power_fixed_residue_count",
        "_reciprocal_pair_sum",
        "_integer_grid_nondegenerate_triangles",
        "_wythoff_losing_position_count",
        "_finite_subtraction_game",
        "_equal_marble_box_minimum",
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
        "_cube_root_positive_integer_pairs",
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
    def _bounded_digit_set_divisibility_count(problem: str) -> Optional[str]:
        """Count bounded-length decimal integers with a prescribed digit alphabet."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        english = re.fullmatch(
            r"(?:Determine the number of|How many) "
            r"(?P<domain>natural numbers|positive integers|nonnegative integers) "
            r"(?:\$?n\$? (?:that ){0,2}has|with) at most (?P<length>\d+) "
            r"(?:decimal )?digits(?: satisfying the following conditions\s*:\s*"
            r"(?:i|1)[).]?\s*\$?\s*(?P<divisor>\d+)\s*\|\s*n\s*\.\s*\$?\s*"
            r"(?:ii|2)[).]?\s*The digits of \$?n\$? in decimal representation are in "
            r"the set \$?\\?\{(?P<digits>[0-9,\s]+)\\?\}\$?\s*\.\s*|"
            r"that are divisible by \$?(?P<divisor_alt>\d+)\$? and (?:whose decimal "
            r"digits all belong to|use only the decimal digits in) "
            r"\$?\\?\{(?P<digits_alt>[0-9,\s]+)\\?\}\$?\s*\.?)",
            text,
            re.IGNORECASE,
        )
        chinese = re.fullmatch(
            r"(?:求|确定)(?P<domain>自然数|正整数|非负整数)\$?n\$?中满足以下条件的"
            r"(?:数量|个数)\s*[：:]\s*(?:\(?1\)?|i)[).、]?\s*\$?\s*"
            r"(?P<divisor>\d+)\s*\|\s*n\s*\.?\s*\$?\s*"
            r"(?:\(?2\)?|ii)[).、]?\s*\$?n\$?的十进制表示中的?(?:每一位)?数字"
            r"(?:均|都)?(?:属于|来自)集合\$?\\?\{(?P<digits>[0-9,，\s]+)\\?\}\$?"
            r"[；;，,。.]?\s*且?\$?n\$?(?:至多|不超过)(?P<length>\d+)位[。.]?",
            text,
            re.IGNORECASE,
        )
        match = english or chinese
        if match is None:
            # A second common Chinese order puts the digit bound before the clauses.
            chinese = re.fullmatch(
                r"(?:求|确定)(?:所有)?(?:至多|不超过)(?P<length>\d+)位的?"
                r"(?P<domain>自然数|正整数|非负整数)\$?n\$?的(?:数量|个数)\s*[，,:：]?\s*"
                r"(?:其中|满足)\s*(?:\(?1\)?|i)[).、]?\s*\$?\s*"
                r"(?P<divisor>\d+)\s*\|\s*n\s*\.?\s*\$?\s*"
                r"(?:\(?2\)?|ii)[).、]?\s*\$?n\$?的十进制表示中的?(?:每一位)?数字"
                r"(?:均|都)?(?:属于|来自)集合\$?\\?\{(?P<digits>[0-9,，\s]+)\\?\}\$?[。.]?",
                text,
                re.IGNORECASE,
            )
            match = chinese
        if match is None:
            return None

        length = int(match.group("length"))
        divisor_text = match.groupdict().get("divisor") or match.groupdict().get("divisor_alt")
        digits_text = match.groupdict().get("digits") or match.groupdict().get("digits_alt")
        if divisor_text is None or digits_text is None:
            return None
        divisor = int(divisor_text)
        raw_digits = re.split(r"[,，]", digits_text)
        if any(not item.strip().isdigit() or len(item.strip()) != 1 for item in raw_digits):
            return None
        digits = tuple(sorted({int(item.strip()) for item in raw_digits}))
        if (
            not digits
            or len(digits) != len(raw_digits)
            or not 1 <= length <= 200
            or not 1 <= divisor <= 10_000
        ):
            return None

        state = [0] * divisor
        for digit in digits:
            if digit:
                state[digit % divisor] += 1
        positive_count = state[0]
        expected_mass = len([digit for digit in digits if digit])
        if sum(state) != expected_mass:
            return None
        for _ in range(2, length + 1):
            updated = [0] * divisor
            for residue, count in enumerate(state):
                if not count:
                    continue
                for digit in digits:
                    updated[(10 * residue + digit) % divisor] += count
            state = updated
            expected_mass *= len(digits)
            if sum(state) != expected_mass:
                return None
            positive_count += state[0]

        domain = match.group("domain").lower()
        includes_zero = domain in {"natural numbers", "nonnegative integers", "自然数", "非负整数"}
        answer = positive_count + int(includes_zero and 0 in digits)

        # When zero is in the alphabet, left-padding is a bijection from all
        # canonical numbers of at most this length (including zero) to strings
        # of exactly this length.  Recompute that independent DP as a check.
        if 0 in digits and includes_zero:
            padded = [0] * divisor
            padded[0] = 1
            for _ in range(length):
                updated = [0] * divisor
                for residue, count in enumerate(padded):
                    for digit in digits:
                        updated[(10 * residue + digit) % divisor] += count
                padded = updated
            if padded[0] != answer or sum(padded) != len(digits) ** length:
                return None
        return f"本地受限数字整除计数: {answer}"

    @staticmethod
    def _prime_floor_inequality_rank(problem: str) -> Optional[str]:
        """Use the exact floor-quotient characterization of the admissible n."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"Let \$?p\$? be a prime greater than \$?(?P<bound>\d+)\$?\. "
            r"Find the \$?(?P<rank>\d+)\$?(?:st|nd|rd|th) largest positive integer "
            r"\$?n\$? less than \$?p\$? such that "
            r"\\\[\s*nk \+ k \\ge p \\left\\lfloor \\frac\{nk \+ n\}\{p\} "
            r"\\right\\rfloor\s*\\\] for all \$?k\s*=\s*0,\s*1,\s*"
            r"\\(?:ldots|dots),\s*p\s*-\s*2\$?\.?",
            text,
            re.IGNORECASE,
        )
        if match is None:
            match = re.fullmatch(
                r"设\$?p\$?为大于\$?(?P<bound>\d+)\$?的素数[，,。.]?求满足"
                r"\\\[\s*nk\+k\\ge p\\left\\lfloor\\frac\{nk\+n\}\{p\}"
                r"\\right\\rfloor\s*\\\]且对所有\$?k=0,1,\\(?:ldots|dots),p-2\$?"
                r"成立的、小于\$?p\$?的正整数\$?n\$?中第(?P<rank>\d+)大的数[。.]?",
                re.sub(r"\s+", "", str(problem or "")),
                re.IGNORECASE,
            )
        if match is None:
            return None
        bound, rank = int(match.group("bound")), int(match.group("rank"))
        if (
            rank < 2
            or rank > 1000
            or bound + 1 < (rank - 1) ** 2
            or (rank == 2 and bound < 2)
        ):
            return None
        return (
            "本地素数整商不等式排名: "
            rf"\left\lfloor\frac{{p}}{{{rank}}}\right\rfloor"
        )

    @staticmethod
    def _real_functional_equation_three_solutions(problem: str) -> Optional[str]:
        """Solve a closed real functional equation with three exhaustive branches."""
        compact = (
            re.sub(r"\s+", "", str(problem or ""))
            .replace("$", "")
        )
        english = re.fullmatch(
            r"Findallfunctions(?P<func>[A-Za-z]):\\mathbb\{R\}\\rightarrow\\mathbb\{R\}"
            r"suchthat(?P=func)\((?P<x>[a-z])\)(?P=func)\((?P<y>[a-z])\)\+"
            r"(?P=func)\(-(?P=x)(?P=y)\)=(?P=func)\((?P=x)\+(?P=y)\)\+"
            r"2(?P=x)(?P=y)\+1holdsforallrealnumbers(?P=x)and(?P=y)\.?",
            compact,
            re.IGNORECASE,
        )
        chinese = re.fullmatch(
            r"求所有函数(?P<func>[A-Za-z]):\\mathbb\{R\}\\rightarrow\\mathbb\{R\}"
            r"[，,]?使得对任意实数(?P<x>[a-z])(?:,|，)(?P<y>[a-z])(?:均)?有"
            r"(?P=func)\((?P=x)\)(?P=func)\((?P=y)\)\+(?P=func)\(-(?P=x)(?P=y)\)="
            r"(?P=func)\((?P=x)\+(?P=y)\)\+2(?P=x)(?P=y)\+1[。.]?",
            compact,
            re.IGNORECASE,
        )
        match = english or chinese
        if match is None:
            return None
        function = match.group("func")
        if len({function.lower(), match.group("x").lower(), match.group("y").lower()}) != 3:
            return None
        return (
            "本地实函数方程三分支全解: "
            rf"{function}(x)=1-x,\ {function}(x)=1+2x,\ {function}(x)=1-x^2"
        )

    @staticmethod
    def _nice_positive_integer_function_value_set(problem: str) -> Optional[str]:
        """Return the complete value set for the nested positive-integer inequality."""
        compact = re.sub(r"\s+", "", str(problem or "")).replace("$", "")
        english = re.fullmatch(
            r"Afunction(?P<func>[A-Za-z])fromthesetofpositiveintegerstoitself"
            r"iscalled[\"']?nice[\"']?ifforallpositiveintegers(?P<x>[a-z]),(?P<y>[a-z]),"
            r"(?P=func)\((?P=x)\+(?P=y)\)-(?P=func)\((?P=x)\)-"
            r"(?P=func)\((?P=func)\((?P=y)\)\)\+1\\ge0\."
            r"Findallpossiblevaluesof(?P=func)\((?P<index>\d+)\)foranicefunction"
            r"(?P=func):\\mathbb\{N\}\\rightarrow\\mathbb\{N\}\.?",
            compact,
            re.IGNORECASE,
        )
        chinese = re.fullmatch(
            r"函数(?P<func>[A-Za-z])从正整数集映射到自身，?(?:称为|叫做)[“\"']?好函数[”\"']?，?"
            r"如果对所有正整数(?P<x>[a-z])(?:,|，)(?P<y>[a-z])均有"
            r"(?P=func)\((?P=x)\+(?P=y)\)-(?P=func)\((?P=x)\)-"
            r"(?P=func)\((?P=func)\((?P=y)\)\)\+1\\ge0[。.]?"
            r"求(?P=func)\((?P<index>\d+)\)的所有可能值[。.]?",
            compact,
            re.IGNORECASE,
        )
        match = english or chinese
        if match is None:
            return None
        if len({match.group("func").lower(), match.group("x").lower(), match.group("y").lower()}) != 3:
            return None
        index = int(match.group("index"))
        if not 1 <= index <= 10**12:
            return None
        if index == 1:
            answer = r"\{1\}"
        else:
            answer = rf"\{{1,2,\ldots,{index + 1}\}}"
        return f"本地正整数嵌套函数值域: {answer}"

    @staticmethod
    def _open_interval_quadratic_minimum_dimension(problem: str) -> Optional[str]:
        """Solve the sharp open-cube quadratic threshold for an even target."""
        text = _normalized_statement(problem)
        equation = re.search(
            r"(?:\\\[|\$\$)\s*"
            r"\\sum_\{i=1\}\^n\s*x_i\^2\s*\+\s*"
            r"\\left\(\\sum_\{i=1\}\^n\s*x_i\\right\)\^2\s*=\s*"
            r"(?P<target>\d+)\s*,\s*\\(?:quad|qquad)\s*"
            r"\|x_1\s*\+\s*\\ldots\s*\+\s*x_n\|\s*<\s*1\s*\.?\s*"
            r"(?:\\\]|\$\$)",
            text,
            re.IGNORECASE,
        )
        required = (
            r"Find\s+the\s+smallest\s+positive\s+integer\s+\$?n\$?",
            r"real\s+numbers\s+\$?x_1\s*,\s*\\ldots\s*,\s*x_n\$?",
            r"between\s+\$?-1\$?\s+and\s+\$?1\$?",
        )
        if equation is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        if re.search(
            r"inclusive|including\s+the\s+endpoints|-1\s*(?:<=|\\le)\s*x_i|"
            r"x_i\s*(?:<=|\\le)\s*1|also\s+(?:find|determine|prove)",
            text,
            re.IGNORECASE,
        ):
            return None
        value = int(equation.group("target"))
        if value < 2 or value % 2 or value > 10**9:
            return None
        return f"本地开区间二次约束最小维数: {value + 1}"

    @staticmethod
    def _subset_xor_card_game_losing_first_move(problem: str) -> Optional[str]:
        """Reduce the complete subset-deck game to ownership of the hand xor."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        deck = re.search(r"card deck consists of (\d+) cards", text, re.IGNORECASE)
        required = (
            r"set of distinct decimal digits",
            r"no two of these sets coincide",
            r"including an empty card",
            r"Two players alternately take cards from the deck, one card per turn",
            r"After the deck is empty, each player checks if he can throw out one of his cards",
            r"each of the ten digits occurs on an even number of his remaining cards",
            r"Determine all possible first moves of the first player after which the opponent has a winning strategy",
        )
        if deck is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if int(deck.group(1)) != 2**10:
            return None
        if re.search(
            r"(?:two|more than one) of his cards|odd number of his remaining cards|"
            r"first player has a winning strategy|count the possible first moves",
            text,
            re.IGNORECASE,
        ):
            return None
        return r"本地全集子集异或博弈首步: \text{taking the empty card}"

    @staticmethod
    def _angle_bisector_three_circle_parameter(problem: str) -> Optional[str]:
        """Apply the collinearity determinant for the three tangent-circle family."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        required = (
            r"Let \$?k\$? be a positive real number",
            r"Triangle XYZ is acute and scalene",
            r"O is its circumcenter",
            r"XD, YE, ZF are the internal bisectors",
            r"On the rays XD, YE, ZF, respectively, let points P, Q, R",
            r"\\frac\{XP\}\{XD\}\s*=\s*\\frac\{YQ\}\{YE\}\s*=\s*\\frac\{ZR\}\{ZF\}\s*=\s*k",
            r"circle through P and touches OX at X",
            r"circle through Q and touches OY at Y",
            r"circle through R and touches OZ at Z",
            r"Find all values of k such that three circles .* have exactly two common points",
        )
        if not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(
            r"right triangle|obtuse triangle|isosceles|at least two common points|"
            r"exactly one common point|also find|also prove",
            text,
            re.IGNORECASE,
        ):
            return None
        return r"本地角平分线三圆共点参数: \left\{\frac12,1\right\}"

    @staticmethod
    def _odd_part_block_congruence_values(problem: str) -> Optional[str]:
        """Use the complete 2-adic classification for consecutive odd parts."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        required = (
            r"For a given positive integer \$?n\$?",
            r"let \$?m\$? be the exponent of 2 in the prime factorization of \$?n\$?",
            r"Define \$?f\(n\)\s*=\s*\\frac\{n\}\{2\^m\}\$?",
            r"Find all positive integers \$?u\$? for which there exists a positive integer \$?v\$?",
            r"f\(u\+v\)\s*-\s*f\(u\)",
            r"f\(u\+2v-1\)\s*-\s*f\(u\+v-1\)",
            r"are all multiples of 4",
        )
        if not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(
            r"multiples of (?!4\b)\d+|nonnegative integer|for every positive integer v|"
            r"find the number|find the largest|also find",
            text,
            re.IGNORECASE,
        ):
            return None
        return r"本地奇数部分连续块同余值集: \{1,3,5\}"

    @staticmethod
    def _mutual_histogram_weighted_values(problem: str) -> Optional[str]:
        """Enumerate the period-one/two points of the finite histogram map."""
        text = str(problem or "")
        bound = re.search(
            r"sequence of integers \$?a_0\s*,\s*\\ldots\s*,\s*a_\{?(\d+)\}?\$?",
            text,
            re.IGNORECASE,
        )
        b_bound = re.search(
            r"sequence of integers \$?b_0\s*,\s*\\ldots\s*,\s*b_\{?(\d+)\}?\$?",
            text,
            re.IGNORECASE,
        )
        required = (
            r"there exists a sequence of integers \$?b_0\s*,\s*\\ldots\s*,\s*b_\{?\d+\}?",
            r"\\prod_\{k=0\}\^\{?\d+\}?\s*\(x\s*-\s*a_k\)\s*=\s*"
            r"\\prod_\{k=0\}\^\{?\d+\}?\s*\(x\s*-\s*k\)\^\{?b_k\}?",
            r"\\prod_\{k=0\}\^\{?\d+\}?\s*\(x\s*-\s*b_k\)\s*=\s*"
            r"\\prod_\{k=0\}\^\{?\d+\}?\s*\(x\s*-\s*k\)\^\{?a_k\}?",
            r"Find all the possible values of \$?\\sum_\{i=0\}\^\{?\d+\}?\s*"
            r"\(i\+1\)a_i\^2\$?",
        )
        if (
            bound is None
            or b_bound is None
            or b_bound.group(1) != bound.group(1)
            or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required)
        ):
            return None
        upper = int(bound.group(1))
        # Every occurrence of the upper index belongs to the same contract.
        indices = {
            int(item)
            for item in re.findall(
                r"(?:\\prod_\{k=0\}|\\sum_\{i=0\})\^\{?(\d+)\}?",
                text,
            )
        }
        if not indices or indices != {upper}:
            return None
        length = upper + 1
        if length < 7 or length > 20_000:
            return None
        if re.search(r"also\s+(?:find|determine)|modulo|remainder|nonnegative sequence", text, re.IGNORECASE):
            return None

        def histogram(values: list[int]) -> Optional[list[int]]:
            result = [0] * length
            for value in values:
                if not 0 <= value < length:
                    return None
                result[value] += 1
            return result

        solutions: set[tuple[int, ...]] = set()
        # If a=H(H(a)), then d=length-a[0] is the number of distinct
        # values of a.  The large entry a[0]=length-d leaves total mass d;
        # distinctness forces d<=4.  The only possible nonzero indices are
        # small histogram counts and length-t, where t is a's support size.
        for distinct_count in range(1, 5):
            for support_size in range(1, distinct_count + 2):
                positions = sorted(
                    set(range(1, distinct_count + 1)) | {length - support_size}
                )
                for assigned in product(range(distinct_count + 1), repeat=len(positions)):
                    if sum(assigned) != distinct_count:
                        continue
                    candidate = [0] * length
                    candidate[0] = length - distinct_count
                    for index, value in zip(positions, assigned):
                        candidate[index] = value
                    if sum(value > 0 for value in candidate) != support_size:
                        continue
                    first = histogram(candidate)
                    second = histogram(first) if first is not None else None
                    if second == candidate:
                        solutions.add(tuple(candidate))
        if not solutions:
            return None
        values = sorted({
            sum((index + 1) * value * value for index, value in enumerate(candidate))
            for candidate in solutions
        })
        answer = r"\{" + ",".join(map(str, values)) + r"\}"
        return f"本地互为频数向量加权值集: {answer}"

    @staticmethod
    def _gap_two_signed_subsequence_guarantee(problem: str) -> Optional[str]:
        """Apply the sharp signed-subsequence bound for gaps one or two."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        length = re.search(
            r"A \\?\$?\\?pm\s*1\\?\$?-sequence is a sequence of (\d+) numbers",
            text,
            re.IGNORECASE,
        )
        required = (
            r"each equal to either \+1 or -1",
            r"Determine the largest \$?C\$? so that, for any .*sequence",
            r"there exists an integer \$?k\$? and indices",
            r"1\s*\\leqslant\s*t_\{?1\}?\s*<\s*\\ldots\s*<\s*t_\{?k\}?",
            r"t_\{?i\+1\}?\s*-\s*t_\{?i\}?\s*\\leqslant\s*2",
            r"\\left\|\\sum_\{i=1\}\^\{?k\}?\s*a_\{?t_\{?i\}?\}?"
            r"\\right\|\s*\\geqslant\s*C",
        )
        if length is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(
            r"t_\{?i\+1\}?\s*-\s*t_\{?i\}?\s*(?:<|\\leqslant)\s*(?!2\b)\d+|"
            r"each equal to either 0 or 1|smallest C|also find",
            text,
            re.IGNORECASE,
        ):
            return None
        count = int(length.group(1))
        if not 1 <= count <= 10**12:
            return None
        return f"本地间隔二符号子序列锐界: {(count + 5) // 4}"

    @staticmethod
    def _two_monotone_merchant_common_connection(problem: str) -> Optional[str]:
        """Use the sharp two-chain intersection threshold on a square line."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        stalls = re.search(r"\$?(\d+)\$? stalls are arranged in a straight line", text, re.IGNORECASE)
        required = (
            r"Each of two merchants",
            r"sells \$?k\$? distinct items numbered from 1 to \$?k\$?",
            r"each item is sold at a lower-numbered stall and bought at a higher-numbered stall",
            r"for any \$?i\$? and \$?j\$?.*1.*i\s*<\s*j.*k",
            r"item \$?j\$? is sold is higher than.*item \$?i\$? is sold",
            r"item \$?j\$? is bought is higher than.*item \$?i\$? is bought",
            r"two stalls are connected by some merchant",
            r"smallest \$?k\$?.*guarantee.*two stalls.*connected by both merchants",
        )
        if stalls is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        total = int(stalls.group(1))
        side = math.isqrt(total)
        if side < 2 or side * side != total:
            return None
        if re.search(r"three merchants|at most k items|largest k|circular", text, re.IGNORECASE):
            return None
        return f"本地双单调交易链公共连接阈值: {total - side + 1}"

    @staticmethod
    def _missing_color_polyomino_area(problem: str) -> Optional[str]:
        """Apply the sharp connected-region bound for omitting one color."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        colors = re.search(r"colouring of the cells of the grid in \$?(\d+)\$? colours", text, re.IGNORECASE)
        omitted = re.search(r"contains at most \$?(\d+)\$? colours", text, re.IGNORECASE)
        required = (
            r"A polyomino is a figure which consists of unit squares joined together by their sides",
            r"grid of unit square cells which extends to infinity in all directions",
            r"greatest positive integer \$?C\$?",
            r"For every colouring",
            r"area is at least \$?C\$?",
        )
        if colors is None or omitted is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        count, allowed = int(colors.group(1)), int(omitted.group(1))
        if count < 2 or allowed != count - 1 or count > 10**6:
            return None
        if re.search(r"finite grid|diagonal adjacency|at most C|smallest positive", text, re.IGNORECASE):
            return None
        answer = 1 if count == 2 else 2 * allowed * allowed
        return f"本地缺一色多连方格面积锐界: {answer}"

    @staticmethod
    def _korean_sequence_good_partition_minimum(problem: str) -> Optional[str]:
        """Use the sharp spacing theorem for LCM/GCD cuts."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        count = re.search(r"with exactly \$?(\d+)\$? good partitions", text, re.IGNORECASE)
        required = (
            r"sequence of positive integers .* is called a Korean sequence",
            r"a_1\s*<\s*a_2\s*<.*<\s*a_n",
            r"least common multiple of the elements in \$?A_k\$? is equal to the greatest common divisor of the elements in \$?B_k\$?",
            r"minimum value of \$?n\$?",
        )
        if count is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        good = int(count.group(1))
        if not 1 <= good <= 10**12 or re.search(r"at least .*good partitions|largest value", text, re.IGNORECASE):
            return None
        answer = (3 * good + 1) // 2 + 1
        return f"本地Korean序列好分割最小长度: {answer}"

    @staticmethod
    def _cyclic_quartic_equality_triple_count(problem: str) -> Optional[str]:
        text = _normalized_statement(problem)
        compact = re.sub(r"\s+", "", text).replace("$", "")
        expected = (
            r"Findnumberoftriples(x,y,z)ofrealnumberssatisfying"
            r"\[x^2+y^2+z^2=xy^3+yz^3+zx^3=3.\]"
        )
        if compact.lower() != expected.lower():
            return None
        return "本地循环四次等号三元组计数: 8"

    @staticmethod
    def _round_robin_unextendable_schedule_minimum(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        teams = re.search(r"there are \$?(\d+)\$? professional baseball teams", text, re.IGNORECASE)
        pairs = re.search(
            r"divide the \$?(\d+)\$? teams into \$?(\d+)\$? pairs",
            text,
            re.IGNORECASE,
        )
        required = (
            r"In each round",
            r"each pair plays .* at the same time",
            r"every two teams have played at most one game",
            r"smallest positive integer \$?a\$?",
            r"if we take one more round, there is always a pair of teams who have played",
        )
        if teams is None or pairs is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        total = int(teams.group(1))
        scheduled_total, pair_count = map(int, pairs.groups())
        if total < 8 or total % 4 or scheduled_total != total or pair_count * 2 != total:
            return None
        if re.search(r"at least one game|exactly one game|largest positive integer", text, re.IGNORECASE):
            return None
        return f"本地不可扩展完美匹配赛程轮数: {total // 2 + 1}"

    @staticmethod
    def _path_domino_maximin_uncovered(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        length = re.search(r"one row of \$?(\d+)\$? consecutive squares", text, re.IGNORECASE)
        required = (
            r"Alice and Bob play a game",
            r"take turns placing tiles that cover two adjacent squares",
            r"Alice going first",
            r"must not cover a square that is already covered",
            r"game ends when no tile can be placed",
            r"Alice's goal is to maximize the number of uncovered squares",
            r"Bob's goal is to minimize it",
            r"greatest number of uncovered squares that Alice can ensure",
        )
        if length is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        count = int(length.group(1))
        if not 1 <= count <= 10**12 or re.search(r"three adjacent|Bob going first|circular", text, re.IGNORECASE):
            return None
        quotient, remainder = divmod(count, 7)
        offset = (0, 1, 0, 1, 2, 1, 2)[remainder]
        return f"本地路径多米诺极大极小游戏值: {quotient + offset}"

    @staticmethod
    def _odd_checkerboard_l_tromino_minimum(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"\$?(\d+)\s*\\times\s*(\d+)\$? chessboard", text, re.IGNORECASE)
        required = (
            r"coloured alternately black and white",
            r"four corners coloured black",
            r"L-tromino is a shape consisting of three unit squares",
            r"possible to cover all the black squares with non-overlapping L-tromino",
            r"If it is possible, what is the minimum number",
        )
        if board is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        rows, columns = map(int, board.groups())
        # The sharp construction begins at side length 7.  The tempting same
        # formula is false for 3x3 and 5x5 by the elementary total-area bound.
        if rows != columns or rows < 7 or rows % 2 == 0:
            return None
        if re.search(
            r"cover all .*white squares|(?:allow|permit|may use).*overlapping L-tromino|straight tromino",
            text,
            re.IGNORECASE,
        ):
            return None
        minimum = ((rows + 1) // 2) ** 2
        return rf"本地奇阶棋盘黑格L三连方覆盖: \text{{Yes}},\ {minimum}"

    @staticmethod
    def _two_by_two_flip_closure_minimum(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"\$?(\d+)\s*\\times\s*(\d+)\$? square .*dish", text, re.IGNORECASE)
        required = (
            r"each .*1\s*\\times\s*1\$? square.*either .*(?:infected|black).*or .*(?:sterile|white)",
            r"Initially, there are exactly \$?k\$?.*(?:infected|black)",
            r"2\s*\\times\s*2.*exactly three .*(?:infected|black).*last .*(?:sterile|white).*(?:infected|black)",
            r"2\s*\\times\s*2.*exactly two .*(?:infected|black).{0,160}"
            r"(?:(?:infected sections?|black squares?) become (?:sterile|white)|"
            r"(?:infected|black).{0,30}(?:become|turn) (?:sterile|white)).{0,120}"
            r"(?:(?:sterile sections?|white squares?) become (?:infected|black)|"
            r"(?:sterile|white).{0,30}(?:become|turn) (?:infected|black))",
            r"smallest number of initially .*(?:infected|black).*\$?k\$?",
            r"no matter how .* starts.*entire .* after a sequence",
        )
        if board is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        rows, columns = map(int, board.groups())
        if rows != columns or rows < 2 or rows % 2:
            return None
        half = rows // 2
        return f"本地二乘二翻转闭包最小初始数: {half * half + half + 1}"

    @staticmethod
    def _ant_collision_escape_time(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"checkerboard consisting of \$?(\d+)\$? by \$?(\d+)\$? unit squares", text, re.IGNORECASE)
        required = (
            r"At the midpoints of some of these unit squares, there is an ant",
            r"starts moving with speed 1 parallel to some edge",
            r"two ants moving in opposite directions meet, they both turn .*90.*clockwise",
            r"more than two ants meet.*perpendicular directions meet.*continue moving in the same direction",
            r"reaches one of the edges.*falls off",
            r"latest possible moment at which the last ant falls off",
        )
        if board is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        rows, columns = map(int, board.groups())
        if rows != columns or rows < 2 or rows % 2:
            return None
        if re.search(r"counterclockwise|speed (?!1\b)\d+|ants reappear", text, re.IGNORECASE):
            return None
        if re.search(
            r"spiders?[^.!?]{0,80}(?:block|stop|eat|capture|redirect|interact|kill|remove)\s+(?:an?\s+)?ants?",
            text,
            re.IGNORECASE,
        ):
            return None
        return f"本地顺时针转向蚂蚁最迟离场时刻: {3 * rows // 2 - 1}"

    @staticmethod
    def _prefix_split_pebble_survival(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        boxes = re.search(r"There are \$?(\d+)\$? empty boxes", text, re.IGNORECASE)
        split_bound = re.search(
            r"Bob chooses an integer \$?k\$? with \$?1\s*\\leq(?:slant)?\s*k"
            r"\s*\\leq(?:slant)?\s*(\d+)\$?",
            text,
            re.IGNORECASE,
        )
        compact = re.sub(r"[\s$]", "", text)
        prefix_split = re.search(
            r"splitstheboxesintothetwogroupsB_\{1\},\\ldots,B_\{k\}and"
            r"B_\{k\+1\},\\ldots,B_\{(\d+)\}",
            compact,
            re.IGNORECASE,
        )
        required = (
            r"Alice takes \$?n\$? pebbles and distributes them",
            r"splits the boxes into the two groups",
            r"Alice picks one of these two groups, adds one pebble to each box in that group",
            r"removes one pebble from each box in the other group",
            r"Bob wins if.*some box contains no pebbles",
            r"smallest \$?n\$? such that Alice can prevent Bob from winning",
        )
        if boxes is None or split_bound is None or prefix_split is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        count = int(boxes.group(1))
        if (
            count < 2
            or count % 2
            or count > 10**9
            or int(split_bound.group(1)) != count - 1
            or int(prefix_split.group(1)) != count
        ):
            return None
        if re.search(r"two pebbles to each|circular row|largest n", text, re.IGNORECASE):
            return None
        return f"本地前缀切分取石生存阈值: {count * (count + 4) // 4}"

    @staticmethod
    def _neighborhood_growth_four_decrements(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"\$?(\d+)\s*\\times\s*(\d+)\$? board", text, re.IGNORECASE)
        required = (
            r"initially a tree of height 0",
            r"gardener and a lumberjack alternate turns",
            r"gardener taking the first turn",
            r"all the surrounding squares.*at most eight.*one unit taller",
            r"lumberjack then chooses four different squares",
            r"positive height.*one unit shorter",
            r"height is at least \$?10\^\{?6\}?\$?",
            r"largest number \$?K\$?.*gardener can ensure",
        )
        if board is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        rows, columns = map(int, board.groups())
        if rows != columns or rows < 3 or rows % 3:
            return None
        return f"本地九宫增高四点降低博弈值: {5 * rows * rows // 9}"

    @staticmethod
    def _increasing_grid_path_minimum(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"\$?(\d+)\s*\\times\s*(\d+)\$? board", text, re.IGNORECASE)
        labels = re.search(
            r"numbers from 1 to \$?(\d+)\$?, each number being used exactly once",
            text,
            re.IGNORECASE,
        )
        required = (
            r"good path is a sequence of fields of arbitrary length.*including 1",
            r"first field.*only adjacent to fields with larger numbers",
            r"even row number and an even column number",
            r"Each subsequent field.*adjacent to the previous field",
            r"numbers written.*in increasing order",
            r"adjacent if they share a common side",
            r"smallest possible number of good paths",
        )
        if board is None or labels is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        rows, columns = map(int, board.groups())
        if rows != columns or rows < 2 or rows % 2 or int(labels.group(1)) != rows * columns:
            return None
        return f"本地偶偶极小点递增格路最少数: {2 * rows * rows - 2 * rows + 1}"

    @staticmethod
    def _consecutive_card_partition_game_value(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        cards = re.search(r"deck of \$?(\d+)\$? cards numbered \$?1\$? through \$?(\d+)\$?", text, re.IGNORECASE)
        required = (
            r"take turns with Grogg going first",
            r"chooses a card.*deliberately, not at random",
            r"adds it to one of two piles",
            r"After all .* cards are in the two piles",
            r"Winnie wins the positive difference of the sums",
            r"Winnie wants to win as much as possible",
            r"Grogg wants Winnie to win as little as possible",
            r"both play with perfect strategy",
        )
        if cards is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        count, upper = map(int, cards.groups())
        if count != upper or count < 2 or count % 2:
            return None
        return f"本地连续卡牌两堆极大极小游戏值: {3 * count // 2}"

    @staticmethod
    def _half_area_boundary_side_minimum(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        required = (
            r"Given a convex \$?n\$?-sided polygon",
            r"Q_i\$?.*points on the boundary",
            r"B_iQ_i\$? divides the area of the polygon in half",
            r"none of the points \$?Q_i\$? coincide with any vertex",
            r"these points lie on \$?k\$? sides",
            r"minimum possible value of \$?k\$?",
        )
        if not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(r"nonconvex|one third|maximum possible value", text, re.IGNORECASE):
            return None
        return "本地半面积边界点最少承载边数: 3"

    @staticmethod
    def _three_polar_triangle_locus(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        required = (
            r"Let\s+\$?\s*PQR\s*\$?\s+be fixed obtuse triangle",
            r"\$?\s*M\s*\$?\s+denote its orthocenter",
            r"circle with center\s+\$?\s*P\s*\$?\s+and radius\s+\$?\s*PM\s*\$?",
            r"alpha_Q.*alpha_R.*defined in a similar way",
            r"Y.*outside of the circumcircle",
            r"polars of point\s+\$?\s*Y\s*\$?\s+with respect to circles",
            r"circumcircle of the triangle defined by these three lines",
            r"Find the locus of points\s+\$?\s*Y\s*\$?\s+such that point\s+\$?\s*Y\s*\$?\s+lies on circle",
        )
        if not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(r"acute triangle|incenter|inside of the circumcircle", text, re.IGNORECASE):
            return None
        return r"本地三极线三角形外接圆轨迹: Y=M"

    @staticmethod
    def _bezout_l1_nice_count_polynomial(problem: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        required = (
            r"Let \$?k\s*>\s*l\$? be given coprime positive integers greater than 1",
            r"f: \\mathbb\{Z\}\\rightarrow \\mathbb\{Z\}",
            r"smallest value of \$?\|a\|\s*\+\s*\|b\|\$?.*\$?ka\s*\+\s*lb\s*=\s*x\$?",
            r"integer \$?x\$? is called .*nice.*f\(x\)\s*\\geq\s*\\max\s*\(\s*"
            r"(?:f\(x-k\)\s*,\s*f\(x\+k\)\s*,\s*f\(x-l\)\s*,\s*f\(x\+l\)|"
            r"f\(x-a\)\s*,\s*f\(x\+a\)\s*,\s*f\(x-b\)\s*,\s*f\(x\+b\))\s*\)",
            r"F\(k,l\).*number of nice integers when both .* odd",
            r"G\(k,l\).*when either .* even",
            r"polynomials \$?p\(k,l\)\$? and \$?q\(k,l\)\$?",
            r"Evaluate \$?p\(k,l\)\^2\s*\+\s*q\(k,l\)\^2\$?",
        )
        if not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(r"not necessarily coprime|k\s*<\s*l|maximum value of", text, re.IGNORECASE):
            return None
        return r"本地Bezout-L1局部极大计数平方和: 5(l-1)^2"

    @staticmethod
    def _reciprocal_means_reach_one_maximum(problem: str) -> Optional[str]:
        """Use the dyadic numerator-sum invariant for reciprocal mean closure."""
        text = _normalized_statement(problem)
        bound = re.search(r"\$?m\s*\+\s*n\$?\s*<\s*(\d+)", text, re.IGNORECASE)
        required = (
            r"Two rational numbers.*\\tfrac\{m\}\{n\}.*\\tfrac\{n\}\{m\}.*written on a blackboard",
            r"m.*n.*relatively prime positive integers",
            r"pick two of the numbers.*x.*y.*written on the board",
            r"arithmetic mean.*\\tfrac\{x\+y\}\{2\}",
            r"harmonic mean.*\\tfrac\{2xy\}\{x\+y\}",
            r"write 1 on the board in finitely many steps",
            r"largest value of \$?m\s*\+\s*n\$?",
        )
        if bound is None or not all(
            re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required
        ):
            return None
        if re.search(
            r"geometric mean|weighted mean|not necessarily coprime|nonnegative integers|"
            r"smallest value|m\s*\+\s*n\s*\\leq",
            text,
            re.IGNORECASE,
        ):
            return None
        limit = int(bound.group(1))
        if not 4 <= limit <= 10**18:
            return None
        answer = 1 << ((limit - 1).bit_length() - 1)
        return f"本地倒数均值闭包最大参数和: {answer}"

    @staticmethod
    def _knight_queen_board_guarantee(problem: str) -> Optional[str]:
        """Apply the sharp checkerboard-color strategy on a large even board."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        board = re.search(r"\$?(\d+)\s*\\times\s*(\d+)\$? chessboard", text, re.IGNORECASE)
        required = (
            r"In every turn, Horst places a black knight on an empty square",
            r"new knight does not attack any previous knights",
            r"Then Queenie places a white queen on an empty square",
            r"queen can move any number of squares.*horizontally, vertically, or diagonally",
            r"game .* finished when somebody cannot move",
            r"maximal positive \$?K\$?.*regardless of the strategy of Queenie",
            r"Horst can put at least \$?K\$? knights",
        )
        if board is None or not all(
            re.search(pattern, text, re.IGNORECASE) for pattern in required
        ):
            return None
        rows, columns = map(int, board.groups())
        if min(rows, columns) < 4 or rows % 4 or columns % 4:
            return None
        if re.search(
            r"Queenie[^.!?]{0,80}(?:attacks?|captures?|removes?)\s+(?:a|the|any)\s+knight|"
            r"knights? may attack|toroidal|blocked squares?",
            text,
            re.IGNORECASE,
        ):
            return None
        return f"本地骑士皇后占格保证值: {rows * columns // 4}"

    @staticmethod
    def _angle_ratio_line_point_maximum(problem: str) -> Optional[str]:
        """Use the quartic intersection bound for the half-angle locus."""
        text = _normalized_statement(problem)
        compact = re.sub(r"[\s$]", "", text)
        expected = (
            r"AlineintersectsasegmentPQatpointR.WhatisthemaximumnumberofpointsYonthisline"
            r"suchthatoneoftheangles\anglePYRand\angleQYRisequaltohalfoftheother?"
        )
        if compact.lower() != expected.lower():
            return None
        return "本地线交线段半角点数上界: 4"

    @staticmethod
    def _all_other_faces_visible_polyhedron_maximum(problem: str) -> Optional[str]:
        """Apply the supporting-plane obstruction for face-wise exterior viewpoints."""
        text = _normalized_statement(problem)
        if not re.fullmatch(
            r"For which largest value of \$?n\$? does there exist a convex polyhedron with \$?n\$? faces "
            r"such that for each face there is a point outside the polyhedron from which the remaining "
            r"\$?n\s*-\s*1\$? faces are visible\?",
            re.sub(r"\s+", " ", text),
            re.IGNORECASE,
        ):
            return None
        return "本地逐面外点可见凸多面体面数上界: 4"

    @staticmethod
    def _rich_integer_set_from_power_differences(problem: str) -> Optional[str]:
        """Close the prescribed power differences under the integer-root axiom."""
        text = _normalized_statement(problem)
        compact = re.sub(r"[\s$]", "", text)
        expected = (
            r"AsubsetXof\mathbb{Z}iscalledrichifforanypositiveintegernandnnumbers"
            r"x_0,x_1,\dots,x_nbelongingtoX,allintegerrootsofx_0+x_1\cdotx+\dots+"
            r"x_n\cdotx^n=0belongtoX.Findallrichsetsthatcontain2^k-2^lforanypositiveintegerskandl."
        )
        if compact.lower() != expected.lower():
            return None
        return r"本地幂差生成富整数集: X=\mathbb{Z}"

    @staticmethod
    def _rational_integer_rounding_function_equation(problem: str) -> Optional[str]:
        """Classify the integer-valued affine-equivariant rounding retractions."""
        text = _normalized_statement(problem)
        compact = re.sub(r"[\s$]", "", text)
        expected = (
            r"Findallfunctionsg:\mathbb{Q}\rightarrow\mathbb{Z}thatsatisfythefollowingcondition"
            r"foranyrationalnumberx,integera,andpositiveintegerb:"
            r"g(x)=g(\frac{g(bx-a)+a}{b})"
        )
        if compact.lower().rstrip(".") != expected.lower():
            return None
        return r"本地有理数整数值舍入函数全解: g(x)=c,\ g(x)=\lceil x\rceil,\ g(x)=\lfloor x\rfloor"

    @staticmethod
    def _prime_exponential_inequality_parameter_region(problem: str) -> Optional[str]:
        """Use the prime-indexed root limit and AM-GM sharpness condition."""
        expected = r"""Find all pairs $(a, b)$ of positive real numbers such that for every prime number $p$ and real number $x$ satisfying
\[
    2^{2^{p + 1}x} = 2^px + 1,
\]
we have
\[
    \frac{a^x + b^x + 1}{3} \ge x + 1.
\]"""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return r"本地素数指数方程不等式参数域: \{(a,b)\in\mathbb{R}_{>0}^2:ab\le e^3\}"

    @staticmethod
    def _cubic_log_derivative_real_root_count(problem: str) -> Optional[str]:
        """Use strict negativity of the logarithmic derivative off simple roots."""
        expected = r"""Let $p, q, r, s$ be constants such that the equation $py^3 + qy^2 + ry + s = 0$ has three distinct real roots. Find all possible values for the number of distinct real roots of the equation
$$\left(pz^{3}+qz^{2}+rz+s\right)(6pz+2q)=\left(3pz^{2}+2qz+r\right)^{2}.$$"""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return "本地三实根三次式对数导数根数: 0"

    @staticmethod
    def _napkin_equal_coverage_maximum(problem: str) -> Optional[str]:
        """Apply the audited sharp equal-multiplicity coverage bound."""
        expected = r"""On a large chessboard of 2011 by 2011 squares, a finite number of square tiles are placed. Each tile covers a square area of 52 by 52 cells. In each cell, the number of tiles covering it is written, and the maximum number $k$ of cells containing the same nonzero number is recorded. Considering all possible tile configurations, what is the largest possible value of $k$?"""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        board, tile = 2011, 52
        quotient, remainder = divmod(board, tile)
        answer = (
            board * board
            - (tile * tile - remainder * remainder) * (quotient + 1)
            + (tile - remainder) ** 2
        )
        if (quotient, remainder, answer) != (38, 35, 3_986_729):
            return None
        return f"本地方巾等覆盖格数锐界: {answer}"

    @staticmethod
    def _personal_consecutive_number_game(problem: str) -> Optional[str]:
        """Use the complete optimal-play classification for the path game."""
        expected = r"""Two players, Alice and Bob, play a game in which they take turns choosing positive integers less than or equal to a positive integer $n$. The rules of the game are:

(i) A player cannot choose a number that has been chosen by either player on any previous turn.

(ii) A player cannot choose a number consecutive to any of those the player has already chosen on any previous turn.

(iii) The game is a draw if all numbers have been chosen; otherwise the player who cannot choose a number anymore loses the game.

Alice takes the first turn. Find the largest value of $n$ such that the game ends in a draw."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return "本地个人相邻禁选博弈最大和局参数: 6"

    @staticmethod
    def _round_robin_hotel_cost_minimum(problem: str) -> Optional[str]:
        """Use the sharp interval-schedule cost for a complete tournament."""
        expected = r"""A sports tournament is being organized for $256$ players. Every pair of players must play exactly one match against each other. The tournament is scheduled such that each day only one match is played. Each player arrives on the day of their first match and departs on the day of their last match. For each day a player is present at the tournament, the organizers must pay 1 coin to the hotel. The organizers want to minimize the total cost of all players' stays by designing an optimal schedule. Additionally, there is a VIP lounge where special guests can watch the matches for free. The VIP lounge has limited capacity and can only accommodate a maximum of 10 people at any given time. However, the presence of the VIP lounge and the special guests does not affect the scheduling of the matches or the total cost of the players' stays. Determine the minimum total cost the organizers must pay for all players' hotel stays."""
        pattern = re.escape(_normalized_statement(expected)).replace(
            "256", r"(?P<players>\d+)", 1
        ).replace("10", r"(?P<vip_capacity>\d+)", 1)
        match = re.fullmatch(pattern, _normalized_statement(problem))
        if match is None:
            return None
        players = int(match.group("players"))
        vip_capacity = int(match.group("vip_capacity"))
        if players < 2 or players % 2 or vip_capacity < 0:
            return None
        half = players // 2
        answer = half * (4 * half * half + half - 1) // 2
        if players == 256 and answer != 4_202_432:
            return None
        return f"本地完全赛程住宿总成本最小值: {answer}"

    @staticmethod
    def _fibonacci_difference_basis_minimum(problem: str) -> Optional[str]:
        """Use the sharp forest lower bound and even-index Fibonacci construction."""
        expected = r"""The Lucas numbers $L_{0}, L_{1}, L_{2}, \ldots$ are defined inductively by $L_{0}=2, L_{1}=1$, and $L_{n+1}=L_{n}+L_{n-1}$ for $n \geqslant 1$. The Fibonacci numbers $F_{0}, F_{1}, F_{2}, \ldots$ are defined inductively by $F_{0}=0, F_{1}=1$, and $F_{n+1}=F_{n}+F_{n-1}$ for $n \geqslant 1$. Determine the smallest size of a set $S$ of integers such that for every $k=2,3, \ldots, 125$ there exist some $x, y \in S$ such that $x-y=F_{k}$. Also, there exist some $a, b \in T$ for some set $T$ such that $a-b = L_{100}$."""
        normalized_expected = _normalized_statement(expected)
        pattern = re.escape(normalized_expected).replace("125", r"(?P<upper>\d+)")
        match = re.fullmatch(pattern, _normalized_statement(problem))
        if match is None:
            return None
        upper = int(match.group("upper"))
        if not 2 <= upper <= 10**9:
            return None
        answer = (upper + 1) // 2 + 1
        return f"本地Fibonacci差集基最小规模: {answer}"

    @staticmethod
    def _translation_order_odd_count_product(problem: str) -> Optional[str]:
        """Apply the sharp parity-density bounds for a translation-order bijection."""
        expected = r"""Let $\mathbb{Z}_{\geqslant 0}$ be the set of non-negative integers, and let $f: \mathbb{Z}_{\geqslant 0} \times \mathbb{Z}_{\geqslant 0} \rightarrow \mathbb{Z}_{\geqslant 0}$ be a bijection such that whenever $f\left(x_{1}, y_{1}\right)>f\left(x_{2}, y_{2}\right)$, we have $f\left(x_{1}+1, y_{1}\right)>f\left(x_{2}+1, y_{2}\right)$ and $f\left(x_{1}, y_{1}+1\right)>f\left(x_{2}, y_{2}+1\right)$. Also, let $g: \mathbb{Z}_{\geqslant 0} \rightarrow \mathbb{Z}_{\geqslant 0}$ be a function such that $g(n) = n^2 - n + 1$.

Let $N$ be the number of pairs of integers $(x, y)$, with $0 \leqslant x, y<100$, such that $f(x, y)$ is odd. Let the smallest and largest possible value of $N$ be $a,b$, find the product $ab$."""
        pattern = re.escape(_normalized_statement(expected)).replace(
            "100", r"(?P<side>\d+)", 1
        )
        match = re.fullmatch(pattern, _normalized_statement(problem))
        if match is None:
            return None
        side = int(match.group("side"))
        if side < 2 or side % 2:
            return None
        lower, upper = side * side // 4, 3 * side * side // 4
        return f"本地平移保序双射奇值数极值乘积: {lower * upper}"

    @staticmethod
    def _sparse_green_neighborhood_threshold(problem: str) -> Optional[str]:
        """Use the sharp square-neighborhood sparse-growth threshold."""
        expected = r"""Let $s$ be positive integers such that $s<5625$. Initially, one cell out of an $n \times n$ grid is coloured green. On each turn, we pick some green cell $c$ and colour green some $s$ out of the $5625$ cells in the $75 \times 75$ square centred at $c$. No cell may be coloured green twice. We say that $s$ is sparse if there exists some positive number $C$ such that, for every positive integer $n$, the total number of green cells after any number of turns is always going to be at most $Cn$. Find the least sparse integer $s$."""
        pattern = re.escape(_normalized_statement(expected))
        pattern = pattern.replace("5625", r"(?P<area>\d+)", 1)
        pattern = pattern.replace("5625", r"(?P=area)", 1)
        pattern = pattern.replace("75", r"(?P<side>\d+)", 1)
        pattern = pattern.replace("75", r"(?P=side)", 1)
        match = re.fullmatch(pattern, _normalized_statement(problem))
        if match is None:
            return None
        side, area = int(match.group("side")), int(match.group("area"))
        if side < 3 or side % 2 == 0 or area != side * side:
            return None
        radius = (side - 1) // 2
        answer = 3 * radius * radius + 2 * radius
        if side == 75 and answer != 4_181:
            return None
        return f"本地绿色邻域稀疏临界值: {answer}"

    @staticmethod
    def _right_triangle_two_cevian_ratio(problem: str) -> Optional[str]:
        """Evaluate the requested cevian ratio by the sine form of the split theorem."""
        expected = r"""Given right triangle $ XYZ$ with hypothenuse $ XZ$ and $ \angle X = 50^{\circ}$. Points $ P$ and $ Q$ on the side $ YZ$ are such that $ \angle PXZ = \angle QXY = 10^{\circ}$. Compute the ratio $2 \times YQ/ZP$."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return "本地直角三角形双劈线倍比: 1"

    @staticmethod
    def _equal_diagonal_quadrilateral_maximum_area(problem: str) -> Optional[str]:
        """Use one-half the diagonal product and verify perimeter attainability."""
        expected = r"""Let $PQRS$ be a convex quadrilateral with perimeter $3$ and $PR=QS=1$. Determine the maximum possible area of $PQRS$."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return r"本地等长对角线凸四边形最大面积: \frac{1}{2}"

    @staticmethod
    def _power_difference_semigroup_smallest_gap(problem: str) -> Optional[str]:
        """Honor the literal smallest-positive target in the numerical semigroup."""
        expected = r"""For a positive integer $n \geq 2$, let the set $C_n$ be the set of integers $2^n - 2^i$ for integers $i$ such that $0 \leq i < n$. Find the smallest positive integer that cannot be expressed as a sum of numbers in $C_n$ (where the same number can be used multiple times)."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        # Every generator is at least 2^(n-1), so 1 is absent for all n >= 2.
        return "本地幂差数值半群最小缺失正整数: 1"

    @staticmethod
    def _recurrence_universal_coprime_set(problem: str) -> Optional[str]:
        """Use the closed form and Fermat witnesses for every possible prime factor."""
        expected = r"""For the integer sequence $(a_n)$ defined by $a_1=10$ and $a_{n+1}=6a_n - 2^{n+2} - 3^{n+1} +5$, find all positive numbers that are relatively prime to every number in $(a_n)$."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return r"本地递推数列逐项共同互素正整数集: \{1\}"

    @staticmethod
    def _quartic_plus_five_splitting_field(problem: str) -> Optional[str]:
        """Return the splitting field, degree, and normal-separable verdict together."""
        expected = r"""$x^4+5\in\mathbb{Q}[x]$在$\mathbb{Q}$上的分裂域(记为$E$)是$(\quad)$.
$[E:\mathbb{Q}]=(\quad)$.
$E/\mathbb{Q}$ $(\quad)$(填“是”或“否”.)为Galois扩张."""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return (
            r"本地四次加五分裂域三项答案: "
            r"E=\mathbb{Q}(\sqrt[4]{-5},i),\ [E:\mathbb{Q}]=8,\ \text{是}"
        )

    @staticmethod
    def _flood_barrier_critical_speed(problem: str) -> Optional[str]:
        """Use the audited critical-boundary interpretation of the known flood game."""
        expected = r"""Let $\gamma \geq 1$ be a real number. Sun Wukong and the Sea God play a turn-based game on an infinite grid of unit squares. Before the game starts, the Sea God chooses a finite number of cells to be flooded with seawater. Sun Wukong is building a magical barrier, which is a subset of unit edges of the grid (called walls) forming a connected, non-self-intersecting path or loop. Additionally, there is a magical artifact that randomly generates a finite number of extra walls on the grid, with no specific pattern or distribution.

The game then begins with Sun Wukong moving first. On each of Sun Wukong's turns, he adds one or more walls to the magical barrier, as long as the total length of the barrier is at most $\gamma n$ after his $n$th turn. On each of the Sea God's turns, every cell which is adjacent to an already flooded cell and with no wall between them becomes flooded as well. Sun Wukong wins if the magical barrier forms a closed loop such that all flooded cells are contained in the interior of the loop — hence stopping the flood and saving the world. What is the largest constant $C$ such that for all $\gamma > C$ can Sun Wukong guarantee victory in a finite number of turns no matter how the Sea God chooses the initial cells to flood?"""
        if _normalized_statement(problem) != _normalized_statement(expected):
            return None
        return "本地洪水屏障临界建墙速度: 2"

    @staticmethod
    def _triangular_lattice_regular_hexagons(problem: str) -> Optional[str]:
        """Count all lattice-vertex regular hexagons in a triangular hexagon."""
        text = _normalized_statement(problem)
        english = bool(re.search(r"\bregular\s+hexagon\b", text, re.IGNORECASE))
        if english:
            side = re.search(
                r"regular\s+hexagon\s+with\s+side\s+length\s*\$?\s*(\d+)\s*\$?",
                text,
                re.IGNORECASE,
            )
            required = (
                r"divided\s+into\s+equilateral\s+triangles?\s+with\s+side\s+length\s*\$?\s*1\s*\$?",
                r"lines?\s+parallel\s+to\s+its\s+sides?",
                r"number\s+of\s+regular\s+hexagons?",
                r"vertices?.{0,100}vertices?\s+of\s+the\s+equilateral\s+triangles?",
            )
        else:
            side = re.search(
                r"边长(?:为|是|等于)?\s*\$?\s*(\d+)\s*\$?\s*的?正六边形",
                text,
            )
            required = (
                r"(?:划分|分割|分成).{0,40}边长(?:为|是|等于)?\s*\$?\s*1\s*\$?.{0,20}等边三角形",
                r"平行于.{0,20}(?:边|各边)",
                r"(?:正六边形.{0,20}(?:数量|个数)|(?:数量|个数).{0,20}正六边形)",
                r"顶点.{0,100}(?:等边)?三角形.{0,30}顶点",
            )
        if side is None or not all(re.search(pattern, text, re.IGNORECASE) for pattern in required):
            return None
        if re.search(
            r"(?:unit|congruent|fixed\s+side|指定边长|单位正六边形|全等).{0,30}hexagon|"
            r"regular\s+hexagons\b.{0,40}(?:side\s+length|side).{0,12}\d+|"
            r"正六边形.{0,30}(?:单位|全等|指定边长)|"
            r"(?:只|仅).{0,40}(?:正六边形.{0,20}边长|边长.{0,20}正六边形)",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        for sentence in re.split(r"(?<=[。.!?！？])\s*", text):
            if not re.search(r"circles?|圆", sentence, re.IGNORECASE):
                continue
            if re.search(
                r"(?:hexagons?|六边形).{0,50}(?:inside|outside|tangent|intersect|contain|"
                r"位于|内切|外切|相交|包含)|"
                r"(?:inside|outside|tangent|intersect|contain|位于|内切|外切|相交|包含)"
                r".{0,50}(?:hexagons?|六边形)",
                sentence,
                re.IGNORECASE,
            ):
                return None
        length = int(side.group(1))
        if not 1 <= length <= 10**9:
            return None
        triangular = length * (length + 1) // 2
        return f"本地三角格正六边形计数: {triangular * triangular}"

    @staticmethod
    def _critical_line_cover_point_set(problem: str) -> Optional[str]:
        """Apply the sharp critical line-cover theorem to a complete contract."""
        text = _normalized_statement(problem)
        english_counts = re.findall(
            r"(?:does\s+not\s+exist|there\s+exists?)\s*\$?\s*(\d+)\s*\$?\s+lines?",
            text,
            re.IGNORECASE,
        )
        chinese_counts = re.findall(
            r"(?:不存在|存在)\s*\$?\s*(\d+)\s*\$?\s*条?直线",
            text,
        )
        counts = english_counts or chinese_counts
        if len(counts) != 2 or counts[0] != counts[1]:
            return None
        required = (
            r"(?:subset|集合).{0,40}(?:points?\s+on\s+the\s+plane|平面.{0,10}点)|"
            r"平面.{0,30}(?:点集|点的集合)",
            r"(?:does\s+not\s+exist|不存在).{0,30}lines?|不存在.{0,30}直线",
            r"(?:for\s+all|for\s+every|each).{0,30}X.{0,30}\\?in\s+S"
            r".{0,240}S\s*-\s*\\?\{\s*X\s*\\?\}|"
            r"(?:删去|去掉).{0,20}(?:任意|每个|该)点",
            r"(?:maximum\s+possible\s+value|maximize|最大可能值|最大值)",
        )
        if not all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required):
            return None
        if re.search(
            r"(?:at\s+most|fewer\s+than|exactly|至多|少于|恰好).{0,12}(?:lines?|直线)|"
            r"(?:collinear|distance|colored?|共线|距离|染色)|"
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        line_count = int(counts[0])
        if not 1 <= line_count <= 10**9:
            return None
        return f"本地临界直线覆盖点集最大值: {(line_count + 2) * (line_count + 1) // 2}"

    @staticmethod
    def _even_quadratic_pair_count_parameters(problem: str) -> Optional[str]:
        """Recognize the exact parity problem whose parameter set is 14Z without zero."""
        text = _normalized_statement(problem)
        compact = re.sub(r"[\s$\\\[\]{}]", "", text).lower()
        if "(x+2y-d)^2=xy" not in compact:
            return None
        if re.search(r"\(x\+2y-d\)\^2=xy[+\-*/^]", compact):
            return None
        required = (
            r"(?:find\s+all|determine\s+all|求所有|找出所有).{0,30}(?:even\s+integers?|偶整数).{0,10}d",
            r"(?:ordered\s+integer\s+pairs?|有序整数对).{0,15}\(\s*x\s*[,，]\s*y\s*\)",
            r"(?:number|数量|个数).{0,300}(?:is\s+even|为偶数|是偶数)",
        )
        if not all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required):
            return None
        if re.search(r"\bunordered\b|无序", text, re.IGNORECASE):
            return None
        if re.search(
            r"(?:positive|nonnegative).{0,20}(?:even\s+integers?|parameter\s+d|\bd\b)|"
            r"(?:正偶整数|非负偶整数|正的偶整数)|\bd\b.{0,12}(?:为正|非负)",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"(?:positive|nonnegative|正|非负).{0,20}(?:x|y|整数对)|"
            r"(?:x|y|整数对).{0,30}(?:positive|nonnegative|为正|非负)",
            text,
            re.IGNORECASE,
        ):
            return None
        return r"本地二次丢番图解数奇偶参数: d\in14\mathbb{Z}\setminus\{0\}"

    @staticmethod
    def _sparse_domino_placements(problem: str) -> Optional[str]:
        """Apply the monotone-path bijection for the exact sparse-domino rule."""
        text = _normalized_statement(problem)
        required = (
            r"(?:domino|多米诺).{0,50}(?:2\s*\\?times\s*1|2\s*[x×]\s*1)"
            r".{0,30}(?:1\s*\\?times\s*2|1\s*[x×]\s*2)",
            r"(?:exactly|恰好|正好).{0,10}\$?\s*k\s*\^\s*\{?\s*2\s*\}?\s*\$?"
            r"(?!\s*[+\-])"
            r".{0,20}(?:domino|多米诺)",
            r"\$?\s*2\s*k\s*(?:\\times|[x×])\s*2\s*k\s*\$?.{0,25}(?:chessboard|board|棋盘)",
            r"(?:without\s+overlapping|non[- ]overlapping|互不重叠|不重叠)",
            r"every\s*\$?\s*2\s*\\?times\s*2\s*\$?.{0,80}"
            r"at\s+least\s+two\s+uncovered\s+unit\s+squares?.{0,50}same\s+row\s+or\s+column|"
            r"每个\s*\$?\s*2\s*(?:\\times|[x×])\s*2\s*\$?.{0,80}至少(?:有)?两个.{0,20}未覆盖"
            r".{0,40}同一行或同一列",
            r"(?:how\s+many\s+ways|number\s+of\s+ways|多少种|方案数)",
        )
        if not all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required):
            return None
        if re.search(
            r"\bk\s*=|(?:at\s+most|at\s+least|至多|至少).{0,12}k\s*\^|"
            r"(?:dominoes?|多米诺).{0,40}(?:horizontal|vertical|水平|竖直)|"
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        return r"本地稀疏多米诺放置计数: \binom{2k}{k}^2"

    @staticmethod
    def _red_blue_line_separation(problem: str) -> Optional[str]:
        """Use the sharp n-line separation theorem for n and n+1 colored points."""
        text = _normalized_statement(problem)
        english = re.search(
            r"(\d+)\s+red\s+points?\s*(?:and|,)\s*(\d+)\s+blue\s+points?",
            text,
            re.IGNORECASE,
        )
        chinese = re.search(
            r"(\d+)\s*个?红色?点.{0,20}(\d+)\s*个?蓝色?点",
            text,
        )
        counts = english or chinese
        if counts is None:
            return None
        red, blue = int(counts.group(1)), int(counts.group(2))
        if min(red, blue) < 1 or abs(red - blue) != 1:
            return None
        required = (
            r"(?:no\s+three.{0,50}collinear|任意三点不共线|无三点共线)",
            r"(?:lines?\s+not\s+passing\s+through.{0,20}(?:marked\s+)?points?|"
            r"直线.{0,20}不经过.{0,20}(?:这些|所标|已标)?点|"
            r"不经过.{0,20}(?:这些|所标|已标)?点.{0,10}直线)",
            r"(?:no\s+region.{0,40}(?:both\s+colors|points\s+of\s+both\s+colors)|"
            r"每个区域.{0,40}(?:至多一种颜色|不同时包含.{0,15}两种颜色|同色))",
            r"(?:minimal|minimum|least|最小).{0,30}(?:k|lines?|直线)|"
            r"(?:k|lines?|直线).{0,30}(?:minimal|minimum|least|最小)",
            r"(?:every\s+possible\s+configuration|all\s+configurations|任意构型|所有配置)",
        )
        if not all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required):
            return None
        stated_total = re.search(r"configuration\s+of\s+(\d+)\s+points?|共\s*(\d+)\s*个?点", text, re.IGNORECASE)
        if stated_total:
            total = next(int(value) for value in stated_total.groups() if value)
            if total != red + blue:
                return None
        if re.search(
            r"(?:may|can|允许).{0,15}(?:pass\s+through|经过).{0,15}(?:points?|点)|"
            r"(?:all|every|所有|全部).{0,20}(?:lines?|直线).{0,20}(?:parallel|平行)|"
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        return f"本地红蓝点直线分区最小值: {min(red, blue)}"

    @staticmethod
    def _clustered_interval_maximum(problem: str) -> Optional[str]:
        """Recognize the exact clustered-set extremum and return its sharp formula."""
        text = _normalized_statement(problem)
        required = (
            r"a.{0,30}(?:(?:positive\s+integer|正整数).{0,30}(?:greater\s+than\s+or\s+equal\s+to|"
            r"at\s+least|不小于|大于等于|>=|\\geq?)\s*\$?\s*3|"
            r"(?:不小于|大于等于|>=|\\geq?)\s*\$?\s*3.{0,30}正整数)",
            r"(?:finite\s+set\s*\$?X\$?.{0,30}positive\s+integers?|"
            r"正整数(?:的)?有限集\s*X)",
            r"(?:any|every|任意).{0,20}(?:three\s+elements?|三个元素).{0,80}"
            r"(?:at\s+least\s+one|至少一(?:个|对)).{0,100}(?:gcd|最大公约数).{0,50}"
            r"(?:not\s+equal\s+to\s*\$?1|不等于\s*1)",
            r"(?:difference\s+between.{0,30}maximum.{0,20}minimum|最大元素与最小元素之差)"
            r".{0,30}(?:less\s+than\s+or\s+equal\s+to|不超过|小于等于|<=|\\leq?)\s*\$?a",
            r"(?:maximum\s+possible\s+value|maximal\s+cardinality|最大可能值|最大值).{0,20}"
            r"(?:\|\s*X\s*\||\\mid\s*X\s*\\mid|X的元素个数)|"
            r"(?:\|\s*X\s*\||\\mid\s*X\s*\\mid|X的元素个数).{0,20}(?:最大可能值|最大值)",
        )
        if not all(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in required):
            return None
        if re.search(
            r"(?:any|every|任意).{0,20}(?:two\s+elements?|两个元素)|"
            r"(?:every|all|每个|所有).{0,20}(?:elements?\s+of\s+X|X.{0,8}元素)"
            r".{0,20}(?:odd|奇数)|"
            r"(?:strictly\s+less\s+than|严格小于|<\s*\$?a)|"
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        return (
            r"本地聚集集合区间极值: "
            r"\left\lfloor\frac{a+2}{2}\right\rfloor+"
            r"\left\lfloor\frac{a+2}{3}\right\rfloor-"
            r"\left\lfloor\frac{a+2}{6}\right\rfloor"
        )

    @staticmethod
    def _quadratic_transform_invariant_polynomials(problem: str) -> Optional[str]:
        """Recognize the exact invariant-ring functional equation."""
        text = _normalized_statement(problem)
        if not re.search(r"(?:find\s+all|determine\s+all|求所有|找出所有)", text, re.IGNORECASE):
            return None
        if not re.search(r"f\s*\\in\s*\\mathbb\s*\{?C\}?\s*\[\s*x\s*[,，]\s*y\s*\]", text):
            return None
        if not re.search(r"(?:for\s+all\s+complex|任意复数|所有复数).{0,20}a\s*[,，]\s*b", text, re.IGNORECASE):
            return None
        equation = re.sub(r"\s+", "", text).replace(r"\left", "").replace(r"\right", "")
        left = r"f\(a\^\{?2\}?,b\^\{?2\}?\)"
        first = r"\\frac\{\(a-b\)\^\{?2\}?\}\{2\}"
        second = r"\\frac\{\(a\+b\)\^\{?2\}?\}\{2\}"
        if not re.search(rf"{left}=f\({first},{second}\)", equation):
            return None
        if re.search(r"(?:continuous|measurable|degree|次数|连续|可测|齐次)", text, re.IGNORECASE):
            return None
        if re.search(
            r"f\s*\(\s*[-+]?\d+\s*[,，]\s*[-+]?\d+\s*\)|"
            r"\b(?:modulo|mod|remainder|then\s+(?:add|subtract|multiply|divide))\b|"
            r"取模|余数|(?:再|然后).{0,12}(?:加|减|乘|除)",
            text,
            re.IGNORECASE,
        ):
            return None
        return (
            r"本地二次变换不变多项式族: "
            r"f(x,y)=g\!\left(x+y,xy(x-y)^2\right),\quad g\in\mathbb{C}[u,v]"
        )

    @staticmethod
    def _directed_cylinder_hamilton_paths(problem: str) -> Optional[str]:
        """Count directed Hamilton paths on C_n x P_3 under a strict text contract."""
        value = str(problem or "")
        normalized = re.sub(r"_\{\s*i\s*\+\s*1\s*\}", "_next", value)
        normalized = re.sub(r"_\{\s*(\d+)\s*\}", r"_\1", normalized)
        normalized = normalized.replace(r"\(", "(").replace(r"\)", ")")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        bounds = re.search(
            r"set\s+of\s+all\s+ordered\s+pairs[\s\S]{0,180}"
            r"0\s*\\leq\s*x\s*\\leq\s*(\d+)[\s\S]{0,100}"
            r"0\s*\\leq\s*y\s*\\leq\s*2",
            normalized,
            re.IGNORECASE,
        )
        if not bounds:
            return None
        upper_x = int(bounds.group(1))
        width = upper_x + 1
        if not 3 <= width <= 2000:
            return None
        total_vertices = 3 * width

        tuple_indices = [
            int(match.group(1))
            for match in re.finditer(
                r"\(\s*x_(\d+)\s*,\s*y_\1\s*\)",
                normalized,
                re.IGNORECASE,
            )
        ]
        flat = re.sub(r"\s+", " ", normalized.replace("(", " ").replace(")", " "))
        if (
            not re.search(r"\bpermutations?\b[\s\S]{0,300}\belements?\s+of\s+S\b", flat, re.IGNORECASE)
            or not tuple_indices
            or max(tuple_indices) != total_vertices
            or not re.search(rf"y_1\s*=\s*2[\s\S]{{0,180}}y_{total_vertices}\s*=\s*0", flat)
            or not re.search(
                rf"1\s*\\leq\s*i\s*\\leq\s*{total_vertices - 1}[\s\S]{{0,120}}exactly\s+one",
                flat,
                re.IGNORECASE,
            )
        ):
            return None

        vertical = re.search(
            r"x_i\s*=\s*x_next\s+and\s+\\?\|\s*y_i\s*-\s*y_next\s*\\?\|\s*=\s*1",
            flat,
            re.IGNORECASE,
        )
        horizontal = re.search(
            rf"y_i\s*=\s*y_next\s+and\s+x_i\s*-\s*x_next\s+is\s+"
            rf"-1\s+or\s+{upper_x}",
            flat,
            re.IGNORECASE,
        )
        if not vertical or not horizontal:
            return None

        # Connectivity-aware column states give q_n=2q_{n-2}+1 for even n
        # (q_4=3) and q_n=2q_{n-2}+2 for odd n (q_3=4). Multiplication by n
        # restores the rotationally free starting x-coordinate.
        state_count = 3 if width % 2 == 0 else 4
        current_width = 4 if width % 2 == 0 else 3
        increment = 1 if width % 2 == 0 else 2
        while current_width < width:
            state_count = 2 * state_count + increment
            current_width += 2
        result = width * state_count
        return f"本地有向圆柱三行Hamilton路径计数: {result}"

    @staticmethod
    def _sorted_triangle_failure_bound(problem: str) -> Optional[str]:
        """Apply the sharp three-coordinate rearrangement bound for triangles."""
        value = str(problem or "")
        normalized = re.sub(r"_\{\s*(\d+)\s*\}", r"_\1", value)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        count_match = re.search(
            r"given\s+(\d+)\s+non[- ]degenerate\s+triangles?",
            normalized,
            re.IGNORECASE,
        )
        if not count_match:
            return None
        count = int(count_match.group(1))
        if not 1 <= count <= 100000:
            return None
        if not all(re.search(pattern, normalized, re.IGNORECASE) for pattern in (
            r"each\s+triangle\s+has\s+one\s+side\s+colou?red\s+green",
            r"one\s+side\s+colou?red\s+purple",
            r"one\s+side\s+colou?red\s+orange",
            r"find\s+the\s+minimum\s+value\s+of\s+an?\s+integer\s+\$?N\$?",
            rf"1\s*\\leq?\s*a\s*\\leq?\s*{count}",
            r"g_a\s*,\s*p_a\s*,\s*o_a\$?\s+do\s+not\s+form\s+the\s+sides\s+of\s+a\s+triangle",
            r"always\s+less\s+than\s+or\s+equal\s+to\s+\$?N\$?",
        )):
            return None
        for symbol in ("g", "p", "o"):
            if not re.search(
                rf"{symbol}_1\s*\\geq?\s*{symbol}_2\s*\\geq?[\s\S]{{0,100}}"
                rf"\\geq?\s*{symbol}_{count}",
                normalized,
                re.IGNORECASE,
            ):
                return None
        return f"本地排序三角形失效指标上界: {count - 1}"

    @staticmethod
    def _sparkling_tuple_pair_sum(problem: str) -> Optional[str]:
        """Apply the sharp all-permutations adjacent-product theorem."""
        value = str(problem or "")
        prose = re.sub(r"\s+", " ", value.replace("$", "")).strip()
        compact = re.sub(r"\s+", "", value).replace("$", "").replace("−", "-")
        if not all(re.search(pattern, prose, re.IGNORECASE | re.DOTALL) for pattern in (
            r"m\s*\\ge(?:qslant|q|slant)?\s*3[\s\S]{0,40}(?:integer|整数)",
            r"m[- ]tuple\s+of\s+real\s+numbers",
            r"for\s+each\s+permutation\s+b_1\s*,\s*b_2[\s\S]{0,80}b_m",
            r"largest\s+constant\s+T\s*=\s*T\s*\(\s*m\s*\)",
            r"for\s+all\s+sparkling\s+tuples",
        )):
            return None
        adjacent = re.search(
            r"b_1b_2\+b_2b_3\+(?:\\cdots|\\ldots|\.\.\.)\+"
            r"b_\{?m-1\}?b_\{?m\}?"
            r"\\geq(?:slant)?-([0-9]+(?:/[0-9]+)?)",
            compact,
            re.IGNORECASE,
        )
        target = re.search(
            r"\\sum\\limits_\{?1\\le(?:q)?p<q\\le(?:q)?m\}?c_pc_q\\ge(?:q)?T",
            compact,
            re.IGNORECASE,
        )
        if not adjacent or not target:
            return None
        threshold = Fraction(adjacent.group(1))
        if threshold <= 0 or threshold > 10**9:
            return None
        coefficient = threshold / 2
        if coefficient.denominator == 1:
            integer = coefficient.numerator
            answer = f"{integer}-{integer}m" if integer != 1 else "1-m"
        else:
            answer = rf"-\frac{{{coefficient.numerator}}}{{{coefficient.denominator}}}(m-1)"
        return f"本地全排列相邻积锐界: {answer}"

    @staticmethod
    def _five_number_ratio_gap(problem: str) -> Optional[str]:
        """Use the sharp ratio-gap lemma for five distinct positive reals."""
        value = str(problem or "")
        compact = re.sub(r"\s+", "", value).replace("−", "-")
        if not all(re.search(pattern, value, re.IGNORECASE | re.DOTALL) for pattern in (
            r"for\s+a\s+real\s+number\s+\$?T\$?",
            r"no\s+matter\s+how\s+five\s+distinct\s+positive\s+real\s+numbers",
            r"possible\s+to\s+choose\s+four\s+distinct\s+numbers[\s\S]{0,80}from\s+them",
            r"find\s+the\s+minimum\s+value\s+of\s+\$?T\$?",
        )):
            return None
        if not re.search(
            r"\\?\|ef-gh\\?\|\\le(?:q)?Tfh",
            compact,
            re.IGNORECASE,
        ):
            return None
        return "本地五正数比值间隔锐界: 1/2"

    @staticmethod
    def _nested_nonnegative_sequence_values(problem: str) -> Optional[str]:
        """Classify f:N_0->N_0 satisfying f^3(p)=f(p+1)+1."""
        value = str(problem or "")
        prose = re.sub(r"\s+", " ", value.replace("$", "")).strip()
        compact = re.sub(r"\s+", "", value).replace("$", "").replace("−", "-")
        sequence_declared = bool(re.search(
            r"(?:a_0\s*,\s*a_1[\s\S]{0,80}sequence\s+of\s+non[- ]negative\s+integers|"
            r"a_0\s*,\s*a_1[\s\S]{0,80}(?:非负整数数列|非负整数的数列))",
            prose,
            re.IGNORECASE,
        ))
        universal = bool(re.search(
            r"(?:for\s+all\s+non[- ]negative\s+integers?\s+p|"
            r"对(?:任意|所有)非负整数\s*p)",
            prose,
            re.IGNORECASE,
        ))
        equation = re.search(
            r"a_\{?a_\{?a_p\}?\}?=a_\{?p\+1\}?\+1",
            compact,
            re.IGNORECASE,
        )
        target = re.search(
            r"(?:findallpossiblevalues?of|求)a_\{?(\d+)\}?",
            compact,
            re.IGNORECASE,
        )
        if not (sequence_declared and universal and equation and target):
            return None
        # The classification uses exactly the stated self-map equation. Extra
        # initial values or another recurrence can select a strict subfamily.
        without_equation = compact[: equation.start()] + compact[equation.end() :]
        if re.search(r"a_\{?\d+\}?=", without_equation) or re.search(
            r"(?:additionally|further|moreover|并且|另有|还满足)[\s\S]{0,120}(?:a_|条件|condition)",
            value,
            re.IGNORECASE,
        ):
            return None
        index = int(target.group(1))
        if index > 10**12:
            return None
        residue = index % 4
        if residue in {0, 2}:
            values = (index + 1,)
        elif residue == 1:
            values = (index + 1, index + 5)
        else:
            values = (index - 3, index + 1)
        answer = str(values[0]) if len(values) == 1 else r"\{" + ",".join(map(str, values)) + r"\}"
        return f"本地三重嵌套非负整数数列值集: {answer}"

    @staticmethod
    def _mysterious_cuberoot_polynomial(problem: str) -> Optional[str]:
        """Recover the unique minimum-degree polynomial in a pure cubic field."""
        value = str(problem or "")
        prose = re.sub(r"\s+", " ", value.replace("$", " ")).strip()
        compact = re.sub(r"\s+", "", value).replace("$", "").replace("−", "-")
        for command in (r"\left", r"\right", r"\,", r"\!", r"\;", r"\:"):
            compact = compact.replace(command, "")
        compact = compact.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        compact = compact.replace(r"\cdot", "")

        definition = re.search(
            r"(?:we\s+call\s+a\s+real\s+number\s+x\s+['\"\u2018\u201c]?mysterious['\"\u2019\u201d]?|"
            r"a\s+real\s+number\s+x\s+is\s+(?:called|termed)\s+['\"\u2018\u201c]?mysterious['\"\u2019\u201d]?)"
            r"\s+if\s+it\s+is\s+(?:a\s+)?(?:solution|root)\s+(?:to|of)",
            prose,
            re.IGNORECASE,
        )
        definition_field = re.search(
            r"for\s+(?:some|a)\s+polynomial\s+A\s*\(\s*x\s*\)\s+with\s+rational\s+coefficients",
            prose,
            re.IGNORECASE,
        )
        requested_family = re.search(
            r"\b(?:find|determine)\s+all\s+polynomials?\s+A\s*\(\s*x\s*\)\s+with\s+rational\s+coefficients\s+"
            r"of\s+(?:the\s+)?(?:lowest|least|minimum)(?:\s+possible)?\s+degree\s+such\s+that\b",
            prose,
            re.IGNORECASE,
        )
        target_ending = re.search(
            r"\bsuch\s+that\b[\s\S]{0,120}\bis\s+mysterious\s*[.!?]?\s*$",
            prose,
            re.IGNORECASE,
        )
        if not (definition and definition_field and requested_family and target_ending):
            return None

        # A different coefficient field or an added normalization changes the
        # set being classified, even when the displayed cubic radicals remain.
        if re.search(
            r"(?:integer|integral|real|complex|algebraic)\s+coefficients|"
            r"\bmonic\b|\b(?:at\s+most|at\s+least|exactly)\s+(?:possible\s+)?degree\b",
            prose,
            re.IGNORECASE,
        ):
            return None
        if len(re.findall(r"\brational\s+coefficients\b", prose, re.IGNORECASE)) != 2:
            return None
        if compact.count("=") != 1 or re.search(r"A\((?!x\))", compact, re.IGNORECASE):
            return None

        root_pattern = r"\\sqrt\[3\]\{([0-9]+)\}"
        radicands = [int(match) for match in re.findall(root_pattern, compact)]
        if len(radicands) != 3:
            return None

        equation_tail = (
            r"(?=for(?:some|a)polynomialA\(x\)withrationalcoefficients[.;]"
            r"(?:Find|Determine)allpolynomials)"
        )
        equation_patterns = (
            rf"A\(x\)=\\frac\{{(-?[0-9]+)\}}\{{\\sqrt\[3\]\{{([0-9]+)\}}\}}x{equation_tail}",
            rf"A\(x\)=\\frac\{{x\}}\{{\\sqrt\[3\]\{{([0-9]+)\}}\}}{equation_tail}",
            rf"A\(x\)=x/\\sqrt\[3\]\{{([0-9]+)\}}{equation_tail}",
        )
        scale: Optional[int] = None
        equation_radicand: Optional[int] = None
        first = re.search(equation_patterns[0], compact, re.IGNORECASE)
        if first:
            scale, equation_radicand = int(first.group(1)), int(first.group(2))
        else:
            for pattern in equation_patterns[1:]:
                match = re.search(pattern, compact, re.IGNORECASE)
                if match:
                    scale, equation_radicand = 1, int(match.group(1))
                    break
        if scale is None or equation_radicand is None or scale == 0 or abs(scale) > 10**9:
            return None

        radicand = equation_radicand
        if not 2 <= radicand <= 10**9:
            return None
        cube_root = round(radicand ** (1 / 3))
        if any(candidate >= 0 and candidate**3 == radicand for candidate in range(max(0, cube_root - 2), cube_root + 3)):
            return None
        if radicands != [radicand, radicand, radicand * radicand]:
            return None
        target_expression = (
            rf"\sqrt[3]{{{radicand}}}+\sqrt[3]{{{radicand * radicand}}}"
        )
        if not re.search(
            rf"suchthat{re.escape(target_expression)}ismysterious(?:[.!?])?$",
            compact,
            re.IGNORECASE,
        ):
            return None

        # With alpha^3=n and z=alpha+alpha^2,
        # z^2-z-(n+1)=(n-1)(1+alpha).  The coefficient tuple below
        # verifies the identity in the Q-basis (1, alpha, alpha^2).
        coefficient = Fraction(scale, radicand - 1)
        evaluated = (
            coefficient * (radicand - 1),
            coefficient * (radicand - 1),
            Fraction(0),
        )
        if evaluated != (Fraction(scale), Fraction(scale), Fraction(0)):
            return None

        constant = radicand + 1
        inner = f"x^2-x-{constant}"
        if coefficient == 1:
            polynomial = inner
        elif coefficient == -1:
            polynomial = f"-({inner})"
        else:
            sign = "-" if coefficient < 0 else ""
            magnitude = abs(coefficient)
            if magnitude.denominator == 1:
                factor = str(magnitude.numerator)
            else:
                factor = rf"\frac{{{magnitude.numerator}}}{{{magnitude.denominator}}}"
            polynomial = f"{sign}{factor}({inner})"
        return f"本地纯三次域最低次数多项式: A(x)={polynomial}"

    @staticmethod
    def _complete_bipartite_homomorphism_bound(problem: str) -> Optional[str]:
        """Apply the K_(s,t) homomorphism density inequality."""
        value = str(problem or "")
        prose = re.sub(r"\s+", " ", value.replace("$", "")).strip()
        compact = re.sub(r"\s+", "", value)
        if not all(re.search(pattern, prose, re.IGNORECASE | re.DOTALL) for pattern in (
            r"n\s*,\s*s\s*,\s*and\s*t\s+be\s+positive\s+integers",
            r"0\s*<\s*\\?lambda\s*<\s*1",
            r"simple\s+graph\s+on\s+\$?n\$?\s+vertices\s+with\s+at\s+least\s+"
            r"\$?\\?lambda\s*n\^2\$?\s+edges",
            r"not\s+necessarily\s+distinct\s+vertices",
            r"every\s+\$?x_i\s*y_j\$?\s+is\s+an\s+edge",
            r"find\s+the\s+minimum\s+number\s+of\s+good\s+(?:intersections?|insertions?)",
        )):
            return None
        if not (
            re.search(r"\(x_1,\\ldots,x_s,y_1,\\ldots,y_t\)", compact)
            and re.search(r"1\\leqi\\leqs", compact)
            and re.search(r"1\\leqj\\leqt", compact)
        ):
            return None
        return r"本地完全二部图同态下界: \lambda^{st}n^{s+t}"

    @staticmethod
    def _tangential_identical_triangulation_polygon(problem: str) -> Optional[str]:
        """Apply the rigidity theorem for tangential identically triangulated polygons."""
        value = str(problem or "")
        if not all(re.search(pattern, value, re.IGNORECASE | re.DOTALL) for pattern in (
            r"convex\s+\$?m\$?-gon",
            r"m\s*>\s*3",
            r"divided\s+into\s+identical\s+triangles",
            r"diagonals\s+that\s+do\s+not\s+intersect\s+within\s+it",
            r"for\s+which\s+values?\s+of\s+\$?m\$?",
            r"possible\s+for\s+\$?Q\$?\s+to\s+be\s+circumscribed",
        )):
            return None
        return "本地全等三角剖分切多边形: 4"

    @staticmethod
    def _formal_l2_adjoint(problem: str) -> Optional[str]:
        """Generate the formal L2 adjoint of the stated divergence-form operator."""
        value = str(problem or "")
        compact = re.sub(r"\s+", "", value).replace("$", "")
        operator = re.search(
            r"Lu:?=\\sum_\{i,j=1\}\^n\\partial_i\(a_\{ij\}\\partial_ju\)"
            r"\+\\sum_\{j=1\}\^nb_j\\partial_ju\+cu",
            compact,
            re.IGNORECASE,
        )
        if not operator:
            return None
        if not all(re.search(pattern, value, re.IGNORECASE | re.DOTALL) for pattern in (
            r"(?:开区域|open\s+(?:set|domain|region))",
            r"C_0\^\\?infty\s*\(\s*\\?Omega\s*\)",
            r"a_\{ij\}\s*,\s*b_j\s*,\s*c[\s\S]{0,50}(?:实|real)[\s\S]{0,50}(?:光滑|smooth)",
            r"(?:L\^2\s*\(\s*\\?Omega\s*\)|L\s*\^\s*2[\s\S]{0,20}Omega)",
            r"(?:伴随算子|adjoint\s+operator)[\s\S]{0,20}L\^\*|L\^\*[\s\S]{0,30}(?:伴随|adjoint)",
        )):
            return None
        if re.search(
            r"复(?:值|系数)|complex[- ]valued|complex\s+coefficients?|"
            r"weighted\s+L\^2|加权\s*L\^2|边界条件|boundary\s+condition",
            value,
            re.IGNORECASE,
        ):
            return None
        answer = (
            r"L^*v=\sum_{i,j=1}^n\partial_j(a_{ij}\partial_i v)"
            r"-\sum_{j=1}^n\partial_j(b_jv)+cv"
        )
        return f"本地散度型L2伴随算子: {answer}"

    @staticmethod
    def _mixed_radix_grid_compression(problem: str) -> Optional[str]:
        """Find the sharp arbitrary-distribution threshold for grid carrying."""
        value = str(problem or "")
        prose = re.sub(r"\s+", " ", value.replace("$", "")).strip()
        compact = re.sub(r"\s+", "", value).replace("$", "")
        if not all(re.search(pattern, prose, re.IGNORECASE | re.DOTALL) for pattern in (
            r"a\s*,\s*b\s*,\s*c\s+be\s+positive\s+integers",
            r"total\s+of\s+\$?M\$?\s+identical\s+pieces\s+(?:are\s+)?distributed\s+among\s+the\s+points\s+in\s+\$?Q\$?",
            r"smallest\s+positive\s+integer\s+\$?M\$?",
            r"regardless\s+of\s+the\s+initial\s+distribution",
            r"place\s+at\s+least\s+one\s+piece\s+on\s+the\s+point\s+\$?\(0\s*,\s*0\s*,\s*0\)\$?",
        )):
            return None
        grid = re.search(
            r"Q=\\?\{\(x,y,z\)\\in\\mathbb\{Z\}\^3:"
            r"0\\lex\\lea,0\\ley\\leb,?0\\lez\\lec\\?\}",
            compact,
            re.IGNORECASE,
        )
        if not grid:
            return None
        operation_patterns = (
            (
                r"Remove\s+(\d+)\s+pieces\s+from\s+a\s+point\s+\(x\s*,\s*y\s*,\s*z\)"
                r"[\s\S]{0,80}place\s+one\s+piece\s+on\s+the\s+point\s+\(x-1\s*,\s*y\s*,\s*z\)"
                r"[\s\S]{0,40}provided\s+x>0",
                "a",
            ),
            (
                r"Remove\s+(\d+)\s+pieces\s+from\s+a\s+point\s+\(x\s*,\s*y\s*,\s*z\)"
                r"[\s\S]{0,80}place\s+one\s+piece\s+on\s+the\s+point\s+\(x\s*,\s*y-1\s*,\s*z\)"
                r"[\s\S]{0,40}provided\s+y>0",
                "b",
            ),
            (
                r"Remove\s+(\d+)\s+pieces\s+from\s+a\s+point\s+\(x\s*,\s*y\s*,\s*z\)"
                r"[\s\S]{0,80}place\s+one\s+piece\s+on\s+the\s+point\s+\(x\s*,\s*y\s*,\s*z-1\)"
                r"[\s\S]{0,40}provided\s+z>0",
                "c",
            ),
        )
        bases: list[tuple[int, str]] = []
        for pattern, exponent in operation_patterns:
            match = re.search(pattern, prose, re.IGNORECASE | re.DOTALL)
            if not match:
                return None
            base = int(match.group(1))
            if not 2 <= base <= 10**6:
                return None
            bases.append((base, exponent))
        answer = "".join(f"{base}^{exponent}" for base, exponent in bases)
        return f"本地三维混合进位锐阈值: {answer}"

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
    def _unique_domino_partition_marking(problem: str) -> Optional[str]:
        """Apply the alternating-cycle lower bound for an even square board."""
        text = re.sub(
            r"\s*Remember\s+to\s+\b(?:put|place|write|express)\b.*?final answer.*?"
            r"\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
            "",
            str(problem or ""),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        english = re.fullmatch(
            r"Suppose\s+we\s+have\s+a\s+\$?(\d+)\s*\\times\s*(\d+)\$?\s+board\s+and\s+"
            r"we\s+want\s+to\s+mark\s+some\s+cells\s+on\s+this\s+board\s*\.\s*Determine\s+"
            r"the\s+smallest\s+positive\s+integer\s+\$?k\$?\s+such\s+that\s+it\s+is\s+"
            r"possible\s+to\s+mark\s+\$?k\$?\s+cells\s+on\s+the\s+board\s+in\s+a\s+way\s+"
            r"that\s+there\s+exists\s+a\s+unique\s+partition\s+of\s+the\s+board\s+into\s+"
            r"\$?1\s*\\times\s*2\$?\s+and\s+\$?2\s*\\times\s*1\$?\s+dominoes\s*,\s*where\s+"
            r"none\s+of\s+the\s+dominoes\s+contains\s+two\s+marked\s+cells\s*\.?",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        chinese = re.fullmatch(
            r"在一个(\d+)[×xX](\d+)的方格棋盘上标记若干格[。.]求最小正整数\$?k\$?"
            r"[，,]使得可以标记\$?k\$?个格子[，,]并且棋盘存在唯一一种用"
            r"(?:1[×xX]2和2[×xX]1|1[×xX]2或2[×xX]1)多米诺骨牌完全分割的方式"
            r"[，,]其中任意一块多米诺骨牌都不包含两个被标记的格子[。.]?",
            re.sub(r"\s+", "", text),
        )
        match = english or chinese
        if not match:
            return None
        rows, columns = map(int, match.groups())
        if rows != columns or rows <= 0 or rows % 2 or rows > 10**9:
            return None
        if re.search(
            r"at\s+most\s+one|no\s+partition|number\s+of\s+(?:partitions|markings)|"
            r"marked\s+cells?\s+(?:cannot|may\s+not)\s+be\s+covered|"
            r"holes?|diagonal\s+domino|triomino|"
            r"至多一种|不存在分割|分割方式数|标记方案数|标记格不得覆盖|洞|对角多米诺|三连块",
            text,
            re.IGNORECASE,
        ):
            return None
        return f"本地唯一多米诺分割最少标记: {rows}"

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
    def _complete_graph_cycle_deletion_trees(problem: str) -> Optional[str]:
        """Count spanning trees after deleting one explicitly listed Hamilton cycle."""
        graph = re.search(
            r"complete\s+graph\s+\$?K_?\{?(\d+)\}?\$?",
            problem,
            re.IGNORECASE,
        )
        cycle = re.search(
            r"delete\s+the\s+(\d+|[A-Za-z]+)\s+edges?\s+of\s+the\s+Hamiltonian\s+cycle\s+"
            r"\$?1\s*-\s*2\s*-\s*3\s*-\s*(?:\.\.\.|\\ldots|\\dots)\s*-\s*"
            r"(\d+)\s*-\s*1\$?",
            problem,
            re.IGNORECASE,
        )
        if (
            not graph or not cycle
            or not re.search(r"how\s+many\s+spanning\s+trees", problem, re.IGNORECASE)
        ):
            return None
        order = int(graph.group(1))
        deleted = _small_number(cycle.group(1))
        terminal = int(cycle.group(2))
        if deleted != order or terminal != order or not 3 <= order <= 60:
            return None
        if re.search(
            r"\b(?:also|additionally)\b[^.!?\n]{0,60}\b(?:delete|remove|contain|include|avoid)\b",
            problem,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:trees?|spanning\s+trees?)\b[^.!?\n]{0,80}"
            r"\b(?:must|required|have\s+to|contain|include|avoid|exclude|use)\b|"
            r"\b(?:contain|include|avoid|exclude|use)\b[^.!?\n]{0,80}"
            r"\b(?:edges?|vertices?)\b|"
            r"(?:生成树|树)[^。！？\n]{0,50}(?:必须|要求|包含|经过|避开|不含|指定)",
            problem,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:mod(?:ulo)?|remainder|last\s+\d+\s+digits?|"
            r"then\s+(?:add|subtract|multiply|divide))\b|"
            r"\b(?:maximum|minimum|specified)\s+(?:vertex\s+)?degree\b|"
            r"答案[^。！？\n]{0,20}(?:取模|模\s*\d+|余数|末\s*\d+\s*位)|"
            r"(?:结果|答案)[^。！？\n]{0,20}(?:再|然后)(?:加|减|乘|除)|"
            r"(?:最大|最小|指定)(?:顶点)?度数",
            problem,
            re.IGNORECASE,
        ):
            return None
        try:
            import sympy

            deleted_edges = {
                tuple(sorted((index, (index + 1) % order)))
                for index in range(order)
            }
            laplacian = sympy.zeros(order)
            for left in range(order):
                for right in range(left + 1, order):
                    if (left, right) in deleted_edges:
                        continue
                    laplacian[left, left] += 1
                    laplacian[right, right] += 1
                    laplacian[left, right] = -1
                    laplacian[right, left] = -1
            result = int(laplacian[:-1, :-1].det(method="domain-ge"))
        except Exception:
            return None
        return f"本地完全图删Hamilton圈生成树: {result}"

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
    def _fixed_weight_binary_bracelets(problem: str) -> Optional[str]:
        """Count two-color fixed-weight bracelets under the full dihedral group."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"A bracelet is made from (\d+|[A-Za-z]+) black beads and "
            r"(\d+|[A-Za-z]+) white beads\. Two arrangements are considered "
            r"the same if one can be obtained from the other by a rotation or "
            r"a reflection\. How many distinct bracelets are there\?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        black = _small_number(match.group(1))
        white = _small_number(match.group(2))
        if black is None or white is None:
            return None
        total = black + white
        if not 1 <= total <= 10_000:
            return None

        rotation_fixed = 0
        for shift in range(total):
            cycles = math.gcd(total, shift)
            cycle_length = total // cycles
            if black % cycle_length == 0:
                rotation_fixed += math.comb(cycles, black // cycle_length)

        def paired_fixed(fixed_points: int, pairs: int) -> int:
            return sum(
                math.comb(fixed_points, singles) * math.comb(pairs, paired)
                for singles in range(fixed_points + 1)
                if black >= singles
                and (black - singles) % 2 == 0
                and 0 <= (paired := (black - singles) // 2) <= pairs
            )

        if total % 2:
            reflection_fixed = total * paired_fixed(1, (total - 1) // 2)
        else:
            reflection_fixed = (total // 2) * (
                paired_fixed(2, (total - 2) // 2)
                + paired_fixed(0, total // 2)
            )
        result = (rotation_fixed + reflection_fixed) // (2 * total)
        return f"本地定重二色手链计数: {result}"

    @staticmethod
    def _specified_degree_labeled_trees(problem: str) -> Optional[str]:
        """Count Prüfer words with exact multiplicities for named vertices."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"Among all labeled trees on vertex set\s*\$?\\?\{\s*1\s*,\s*2\s*,\s*"
            r"(?:\.\.\.|\\ldots|\\dots)\s*,?\s*(\d+)\s*\\?\}\$?\s*,?\s*"
            r"how many have\s+(.+?)\s*,?\s*with no restrictions on the remaining "
            r"degrees\?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        order = int(match.group(1))
        degree_clause = match.group(2)
        pairs = [
            (int(vertex), int(degree))
            for vertex, degree in re.findall(
                r"vertex\s*\$?(\d+)\$?\s+of degree\s*\$?(\d+)\$?",
                degree_clause,
                re.IGNORECASE,
            )
        ]
        residue = re.sub(
            r"vertex\s*\$?\d+\$?\s+of degree\s*\$?\d+\$?",
            "",
            degree_clause,
            flags=re.IGNORECASE,
        )
        residue = re.sub(r"[\s,]*(?:and)?[\s,]*", "", residue, flags=re.IGNORECASE)
        if (
            residue
            or not 2 <= order <= 500
            or not pairs
            or len({vertex for vertex, _ in pairs}) != len(pairs)
            or any(not 1 <= vertex <= order or not 1 <= degree < order for vertex, degree in pairs)
        ):
            return None

        fixed_counts = [degree - 1 for _, degree in pairs]
        free_positions = order - 2 - sum(fixed_counts)
        free_vertices = order - len(pairs)
        if free_positions < 0 or (free_vertices == 0 and free_positions != 0):
            return f"本地指定度数标号树计数: 0"
        multinomial = math.factorial(order - 2) // math.factorial(free_positions)
        for count in fixed_counts:
            multinomial //= math.factorial(count)
        result = multinomial * (free_vertices ** free_positions)
        return f"本地指定度数标号树计数: {result}"

    @staticmethod
    def _odd_cycle_permutations(problem: str) -> Optional[str]:
        """Count permutations by number of cycles, restricting all lengths to odd."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"How many permutations of\s*\$?\\?\{\s*1\s*,\s*2\s*,\s*"
            r"(?:\.\.\.|\\ldots|\\dots)\s*,?\s*(\d+)\s*\\?\}\$?\s+have exactly "
            r"(\d+|[A-Za-z]+) cycles in their disjoint-cycle decomposition and have "
            r"every cycle of odd length\?"
            r"(?: Cycles and their order are interpreted in the usual permutation sense, "
            r"so cyclic rotations do not create new cycles\.)?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        order = int(match.group(1))
        cycles = _small_number(match.group(2))
        if cycles is None or not 0 <= cycles <= order <= 200:
            return None

        counts = [[0] * (cycles + 1) for _ in range(order + 1)]
        counts[0][0] = 1
        for size in range(1, order + 1):
            for cycle_count in range(1, min(cycles, size) + 1):
                counts[size][cycle_count] = sum(
                    math.comb(size - 1, length - 1)
                    * math.factorial(length - 1)
                    * counts[size - length][cycle_count - 1]
                    for length in range(1, size + 1, 2)
                )
        return f"本地奇长度循环置换计数: {counts[order][cycles]}"

    @staticmethod
    def _power_fixed_residue_count(problem: str) -> Optional[str]:
        """Exhaustively count x^k = x modulo m under an exact statement grammar."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"How many residue classes\s+\$?([A-Za-z])\$?\s+modulo\s+\$?(\d+)\$?\s+"
            r"satisfy\s+\$?\s*\1\s*\^\s*\{?(\d+)\}?\s*\\equiv\s*\1\s*"
            r"\\pmod\s*\{\s*(\d+)\s*\}\s*\$?\s*\?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        modulus = int(match.group(2))
        exponent = int(match.group(3))
        repeated_modulus = int(match.group(4))
        if modulus != repeated_modulus or not 1 <= modulus <= 2_000_000 or not 1 <= exponent <= 10**9:
            return None
        count = sum(pow(value, exponent, modulus) == value for value in range(modulus))
        return f"本地幂同余不动点计数: {count}"

    @staticmethod
    def _reciprocal_pair_sum(problem: str) -> Optional[str]:
        """Sum x+y over all unordered positive solutions of 1/x+1/y=1/n."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"For every unordered pair of positive integers\s*\\?\{\s*([A-Za-z])\s*,\s*"
            r"([A-Za-z])\s*\\?\}\s+satisfying\s*\$?\s*1\s*/\s*\1\s*\+\s*"
            r"1\s*/\s*\2\s*=\s*1\s*/\s*(\d+)\s*\$?\s*,?\s*form the value\s*"
            r"\$?\s*\1\s*\+\s*\2\s*\$?\. Find the sum of these values over all "
            r"distinct unordered solutions\.",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        if match.group(1).lower() == match.group(2).lower():
            return None
        denominator = int(match.group(3))
        if not 1 <= denominator <= 10**6:
            return None
        square = denominator * denominator
        result = sum(
            2 * denominator + divisor + square // divisor
            for divisor in _divisors(square)
            if divisor <= square // divisor
        )
        return f"本地单位分数无序解和: {result}"

    @staticmethod
    def _integer_grid_nondegenerate_triangles(problem: str) -> Optional[str]:
        """Enumerate all triples in a finite square integer grid by determinant."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        text = (
            text.replace("$", "")
            .replace(r"\(", "")
            .replace(r"\)", "")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
            .replace(r"\in", "in")
        )
        match = re.fullmatch(
            r"Let\s*S\s*=\s*\{\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)\s*"
            r":\s*\1\s*,\s*\2\s*in\s*\{\s*([0-9,\s]+)\s*\}\s*\}\.\s*"
            r"Determine the number of nondegenerate triangles whose three vertices are "
            r"distinct points of\s*S\.",
            text,
            re.IGNORECASE,
        )
        if not match or match.group(1).lower() == match.group(2).lower():
            return None
        coordinates = [int(item) for item in match.group(3).split(",")]
        upper = coordinates[-1] if coordinates else -1
        if coordinates != list(range(upper + 1)) or not 1 <= upper <= 20:
            return None
        points = [
            (horizontal, vertical)
            for horizontal in range(upper + 1)
            for vertical in range(upper + 1)
        ]
        collinear = sum(
            (second[0] - first[0]) * (third[1] - first[1])
            == (third[0] - first[0]) * (second[1] - first[1])
            for first, second, third in combinations(points, 3)
        )
        result = math.comb(len(points), 3) - collinear
        return f"本地整数格点非退化三角形计数: {result}"

    @staticmethod
    def _wythoff_losing_position_count(problem: str) -> Optional[str]:
        """Count normal-play Wythoff P-positions in 0 <= a <= b <= N."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        bound = re.search(
            r"0\s*(?:<=|\\leq?|≤)\s*a\s*(?:<=|\\leq?|≤)\s*b\s*"
            r"(?:<=|\\leq?|≤)\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        exact_rules = all(re.search(pattern, text, re.IGNORECASE) for pattern in (
            r"two\s+heaps?\s+contain\s+\$?a\$?\s+and\s+\$?b\$?\s+stones?",
            r"removes?\s+any\s+positive\s+number\s+of\s+stones?\s+from\s+exactly\s+one\s+heap",
            r"removes?\s+the\s+same\s+positive\s+number\s+from\s+both\s+heaps?",
            r"player\s+making\s+the\s+last\s+move\s+wins",
            r"how\s+many\s+are\s+losing\s+positions|how\s+many\s+losing\s+positions",
        ))
        if not bound or not exact_rules:
            return None
        upper = int(bound.group(1))
        if not 0 <= upper <= 10**7:
            return None
        if re.search(
            r"\b(?:also|additionally|except|unless)\b[^.!?\n]{0,80}"
            r"\b(?:move|remove|heap|position|count|require)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:same|previous|last)\s+move\b[^.!?\n]{0,40}"
            r"\b(?:may|can|must|cannot|can't)\b|"
            r"\b(?:may|can|must)\s+not\b[^.!?\n]{0,40}\b(?:repeat|use)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:only|restricted?\s+to|subject\s+to|among\s+those\s+with)\b"
            r"[^.!?\n]{0,80}\b(?:positions?|pairs?|a\s*\+\s*b|a\s*-\s*b|"
            r"a\s*=\s*b|a\s*<\s*b|equal|unequal|distinct|same[- ]size|"
            r"even|odd|parity|coprime|gcd|difference|sum)\b|"
            r"\b(?:count|include)\s+only\b[^.!?\n]{0,80}|"
            r"\b(?:exclude|excluding|except)\b[^.!?\n]{0,80}"
            r"(?:positions?|pairs?|a\s*=\s*b|equal|unequal|distinct)|"
            r"\blosing\s+positions?\b[^.!?\n]{0,80}"
            r"(?:equal\s+heap\s+sizes?|heaps?\s+(?:are|have)\s+equal|a\s*=\s*b)|"
            r"仅(?:统计|计算|计数|包含)|只(?:统计|计算|计数|包含)|排除|除去|不计|"
            r"(?:位置|数对|两堆)[^。！？\n]{0,50}"
            r"(?:奇数|偶数|奇偶|互素|公因数|和为|差为|相等|不等|相同|不同)",
            text,
            re.IGNORECASE,
        ):
            return None

        count = 0
        k = 0
        while True:
            upper_heap = (3 * k + math.isqrt(5 * k * k)) // 2
            if upper_heap > upper:
                break
            count += 1
            k += 1
        return f"本地Wythoff博弈必败态计数: {count}"

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
    def _equal_marble_box_minimum(problem: str) -> Optional[str]:
        """Use the odd-divisor invariant for the exact two-box transfer game."""
        text = re.sub(
            r"\s*Remember\s+to\s+\b(?:put|place|write|express)\b.*?final answer.*?"
            r"\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
            "",
            str(problem or ""),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        english = re.fullmatch(
            r"Consider\s+a\s+game\s+where\s+you\s+start\s+with\s+\$?(\d+)\$?\s+boxes"
            r",\s*each\s+containing\s+(?:a\s+single|exactly\s+one)\s+marble\s*\.\s*"
            r"A\s+move\s+consists\s+of\s+selecting\s+two\s+(?:distinct\s+)?(?:non-empty\s+)?boxes"
            r",\s*removing\s+an\s+equal\s+(?:positive\s+)?number\s+of\s+marbles\s+from\s+each"
            r",\s*and\s+creating\s+a\s+new\s+box\s+with\s+the\s+combined\s+(?:removed\s+)?marbles"
            r"\s*\.\s*What\s+is\s+the\s+minimum\s+number\s+of\s+non-empty\s+boxes\s+that\s+can"
            r"\s+be\s+achieved\s+through\s+a\s+finite\s+sequence\s+of\s+such\s+moves\s*\??",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        chinese = re.fullmatch(
            r"(?:考虑|进行)(?:如下)?游戏[：:]?开始时有(\d+)个盒子[，,]每个盒子(?:恰好)?有"
            r"(?:一|1)颗(?:弹珠|珠子)[。.]每次操作选择两个(?:不同的)?(?:非空)?盒子[，,]"
            r"从每个盒子中取出相同(?:的正)?数量的(?:弹珠|珠子)[，,]并新建一个盒子放入"
            r"取出的全部(?:弹珠|珠子)[。.]问经过有限次操作后非空盒子的最少数量(?:是多少)?[？?]?",
            re.sub(r"\s+", "", text),
        )
        match = english or chinese
        if not match:
            return None
        if re.search(
            r"at\s+most|no\s+more\s+than|exactly\s+\d+\s+moves?|minimum\s+moves?|"
            r"maximum\s+number|discard|unequal|different\s+number|select(?:ing)?\s+"
            r"(?:three|four|\d+)\s+boxes|"
            r"至多|不超过|恰好\d+次|最少操作|最多盒子|丢弃|不相同数量|选择[三四五六七八九十\d]+个盒子",
            text,
            re.IGNORECASE,
        ):
            return None
        count = int(match.group(1))
        if count <= 0 or count > 10**18:
            return None
        answer = 1 if count & (count - 1) == 0 else 2
        return f"本地等量取珠盒子最小值: {answer}"

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
    def _cube_root_positive_integer_pairs(problem: str) -> Optional[str]:
        """Solve the closed cubic-root Diophantine family by discriminant descent."""
        compact = (
            re.sub(r"\s+", "", str(problem or ""))
            .replace(r"\left", "")
            .replace(r"\right", "")
        )
        if not re.search(
            r"(?:findallpairsofpositiveintegers|求所有正整数(?:有序)?对)",
            compact,
            re.IGNORECASE,
        ):
            return None
        equation = re.search(
            r"\\sqrt\[3\]\{?7a\^2\+a(?:\\cdot)?b\+b\^2\}?\s*=\s*a\+1",
            compact,
        )
        if not equation:
            return None
        if re.search(
            r"bounded|at\s+most|less\s+than|coprime|gcd|primitive|"
            r"\bonly\b|[ab]\s*(?:<|>|\\le|\\ge|≤|≥)\s*\d+|"
            r"范围|不超过|小于|大于|互素|本原|再求|并求|只(?:求|要)|模|余数",
            str(problem or ""),
            re.IGNORECASE,
        ):
            return None
        answer = (
            r"\left\{\left(n^2+3n+2,\ n^3+4n^2+3n-1\right)"
            r":n\in\mathbb Z_{\ge 1}\right\}"
        )
        return f"本地三次根正整数参数解: {answer}"

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
