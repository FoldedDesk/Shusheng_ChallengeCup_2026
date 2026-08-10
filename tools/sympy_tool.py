from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
from math import gcd
import math
from typing import Any, Optional

from tools.tool_contract import ToolResult, result_from_legacy_hint
from tools.exact_olympiad_tool import ExactOlympiadTool


class SympyTool:
    """Optional local symbolic helper. Tool failures never block model solving."""

    def __init__(self) -> None:
        try:
            import sympy as sympy_module
        except ImportError:
            sympy_module = None
        self.sympy = sympy_module

    def derivative(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.diff(self._parse(expression), s.Symbol(variable)))

    def integral(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.integrate(self._parse(expression), s.Symbol(variable)))

    def definite_integral(
        self,
        expression: str,
        variable: str,
        lower: str,
        upper: str,
    ) -> Optional[str]:
        return self._run(
            lambda s: s.integrate(
                self._parse(expression),
                (s.Symbol(variable), self._parse(lower), self._parse(upper)),
            )
        )

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list]:
        if not self.sympy:
            return None
        try:
            return [self._format(item) for item in self.sympy.solve(self._parse(expression), self.sympy.Symbol(variable))]
        except Exception:
            return None

    def matrix(self, rows: list[list[Any]]) -> Optional[list[list[str]]]:
        if not self.sympy:
            return None
        try:
            return [[self._format(item) for item in row] for row in self.sympy.Matrix(rows).tolist()]
        except Exception:
            return None

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(
            lambda s: s.limit(self._parse(expression), s.Symbol(variable), self._parse(point))
        )

    def evaluate(self, expression: str) -> Optional[str]:
        return self._run(lambda _: self._parse(expression))

    def hints_for(self, problem: str) -> list[str]:
        """Return safe, deterministic hints for elementary symbolic subproblems.

        This deliberately handles only unambiguous LaTex or plain-text forms.
        Anything it cannot parse is left to the model solver.
        """
        problem = re.sub(
            r"\s*Remember\s+to\s+\b(?:put|place|write|express)\b.*?final answer.*?\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
            "",
            str(problem or ""),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        hints: list[str] = ExactOlympiadTool().hints_for(problem)
        for local_hint in (
            self._complete_multipartite_tree_hint(problem),
            self._quadratic_congruence_count_hint(problem),
            self._digit_permutation_divisibility_hint(problem),
            self._adjacent_surjection_count_hint(problem),
            self._multiset_no_adjacent_hint(problem),
            self._binary_run_avoidance_hint(problem),
            self._bracelet_no_adjacent_hint(problem),
            self._strip_lattice_path_hint(problem),
            self._nested_modular_sum_hint(problem),
            self._quadratic_form_maximum_hint(problem),
            self._propositional_implication_chain_hint(problem),
            self._minimum_degree_path_hint(problem),
            self._nonadjacent_binary_string_count_hint(problem),
            self._precedence_permutation_count_hint(problem),
            self._surjection_count_hint(problem),
            self._planar_euler_face_hint(problem),
            self._paraboloid_curvature_hint(problem),
            self._ordered_positive_triple_hint(problem),
            self._simple_random_walk_hint(problem),
            self._complete_graph_cover_time_hint(problem),
            self._two_venue_capacity_hint(problem),
            self._circle_laplacian_hint(problem),
            self._central_difference_hint(problem),
            self._rational_f2_constraint_hint(problem),
            self._digit_sum_window_hint(problem),
            self._number_writing_game_hint(problem),
            self._path_independent_set_partition_hint(problem),
            self._spike_sequence_construction_hint(problem),
            self._dependent_bernoulli_construction_hint(problem),
            self._lz78_encoding_hint(problem),
            self._linear_recurrence_hint(problem),
            self._curve_speed_hint(problem),
            self._first_fundamental_form_hint(problem),
            self._graph_gaussian_curvature_hint(problem),
            self._pde_verification_hint(problem),
        ):
            if local_hint:
                hints.append(local_hint)
        if not self.sympy:
            return hints
        arithmetic = re.search(
            r"(?:计算|求值|calculate|evaluate)\s*([0-9A-Za-z_+\-*/^().,\s]+?)[。？?]?$",
            problem,
            re.IGNORECASE,
        )
        if arithmetic and not re.search(r"积分|导数|极限|方程|integral|derivative|limit|equation", problem, re.IGNORECASE):
            result = self.evaluate(arithmetic.group(1))
            if result is not None:
                hints.append(f"SymPy 计算: {result}")

        congruence = self._congruence_hint(problem)
        if congruence:
            hints.append(congruence)
        modular_power = self._modular_power_hint(problem)
        if modular_power:
            hints.append(modular_power)

        if re.search(r"导数|求导|微分|derivative|differentiate", problem, re.IGNORECASE):
            partial = re.search(
                r"f\s*\(\s*[A-Za-z]\s*,\s*[A-Za-z]\s*\)\s*=\s*(?P<expression>[^，。；;]+?)\s*关于\s*\$?(?P<variable>[A-Za-z])\$?\s*的?(?:偏导|导数)",
                problem,
            )
            match = partial or re.search(
                r"(?:f\s*\(\s*(?P<variable>[A-Za-z])\s*\)|y)\s*=\s*(?P<expression>[^，。；;]+?)(?=\s*(?:的(?:导数|微分)|[,，。；;]|$))",
                problem,
            )
            if match:
                variable = match.group("variable") or "x"
                result = self.derivative(self._latex_to_sympy(match.group("expression")), variable)
                if result is not None:
                    label = "偏导数" if partial else "导数"
                    hints.append(f"SymPy {label}: {result}")

        math_parts = re.findall(r"\$([^$]+)\$", problem)
        math_parts.extend(self._raw_latex_parts(problem))
        math_parts.extend(self._plain_equations(problem))
        if re.search(r"积分|\\int|integral|integrate", problem, re.IGNORECASE):
            for part in math_parts:
                definite = re.search(
                    r"\\int_\{?([^}\s]+)\}?\^\{?([^}\s]+)\}?\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b",
                    part,
                )
                if definite:
                    result = self.definite_integral(
                        self._latex_to_sympy(definite.group(3)),
                        definite.group(4),
                        self._latex_to_sympy(definite.group(1)),
                        self._latex_to_sympy(definite.group(2)),
                    )
                    if result is not None and self._is_evaluated_result(result):
                        hints.append(f"SymPy 定积分: {result}")
                    break
                match = re.search(r"\\int\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b", part)
                if match:
                    result = self.integral(self._latex_to_sympy(match.group(1)), match.group(2))
                    if result is not None:
                        hints.append(f"SymPy 不定积分: {result}")
                    break

        if re.search(r"极限|\\lim|\blimit\b", problem, re.IGNORECASE):
            for part in math_parts:
                match = re.search(r"\\lim_\{?\s*([A-Za-z])\s*\\to\s*([^}\s]+)\}?\s*(.+)", part)
                if match:
                    result = self.limit(
                        self._latex_to_sympy(match.group(3)), match.group(1), self._latex_to_sympy(match.group(2))
                    )
                    if result is not None:
                        hints.append(f"SymPy 极限: {result}")
                    break

        if re.search(r"方程|求解|equation|solve|roots?|zeros?|\bfind\s+[xyz]\b", problem, re.IGNORECASE):
            for part in math_parts:
                if "=" not in part or r"\begin" in part:
                    continue
                left, right = part.split("=", 1)
                variable = re.search(r"\b([xyz])\b", left + right)
                if variable:
                    expression = f"({self._latex_to_sympy(left)})-({self._latex_to_sympy(right)})"
                    result = self.solve_equation(expression, variable.group(1))
                    if result is not None:
                        if result:
                            answer = "，".join(f"{variable.group(1)}={item}" for item in result)
                        else:
                            answer = "无解"
                        hints.append(f"SymPy 方程解: {answer}")
                    break

        if re.search(r"矩阵|\\begin\{[pb]?matrix\}|\bmatrix\b", problem, re.IGNORECASE):
            for part in math_parts:
                match = re.search(r"\\begin\{[pb]?matrix\}(.+?)\\end\{[pb]?matrix\}", part, re.DOTALL)
                if match:
                    rows = [
                        [self._latex_to_sympy(cell) for cell in row.split("&")]
                        for row in re.split(r"\\\\", match.group(1))
                    ]
                    result = self.matrix(rows)
                    if result is not None:
                        hints.append(f"SymPy 矩阵: {result}")
                    break
        return hints

    def results_for(self, problem: str) -> list[ToolResult]:
        """Return structured deterministic evidence while preserving ``hints_for``.

        Callers should prefer this method when deciding whether a tool may
        answer a complete goal.  Unknown legacy labels remain unverified.
        """

        results: list[ToolResult] = []
        for hint in self.hints_for(problem):
            parsed = result_from_legacy_hint(
                hint,
                trusted_source=True,
                extra_checks=self._certificate_checks_for_hint(hint),
            )
            if parsed is not None:
                results.append(parsed)
        return results

    @staticmethod
    def _certificate_checks_for_hint(hint: str) -> tuple[str, ...]:
        label = str(hint or "").partition(": ")[0]
        if label == "本地圆周拉普拉斯":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "ambient_or_unqualified_laplacian",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "ambient_operator_selected",
                "second_derivatives_sum_to_4",
            )
        if label == "本地圆周Laplace-Beltrami":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "explicit_intrinsic_operator",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "intrinsic_operator_selected",
                "restriction_is_constant",
            )
        if label == "本地圆周拉普拉斯歧义核验":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "operator_not_disambiguated",
                "operator_ambiguity",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "operator_ambiguity_detected",
                "both_operator_cases_evaluated",
            )
        return ()

    @staticmethod
    def _is_evaluated_result(result: str) -> bool:
        """Reject inert SymPy objects that merely restate the requested work."""
        return not bool(re.search(
            r"\b(?:Integral|Derivative|Limit|Sum|Product|RootSum|ConditionSet)\s*\(",
            str(result or ""),
        ))

    @staticmethod
    def _complete_multipartite_tree_hint(problem: str) -> Optional[str]:
        """Count spanning trees of a fully specified complete multipartite graph."""
        text = str(problem or "")
        match = re.search(
            r"complete\s+(?:bi|tri|multi)?partite\s+graph\s+\$?K_?\{?"
            r"([0-9]+(?:\s*,\s*[0-9]+)+)\}?\$?",
            text,
            re.IGNORECASE,
        )
        if not match or not re.search(r"spanning\s+trees?", text, re.IGNORECASE):
            return None
        parts = tuple(int(item) for item in re.findall(r"\d+", match.group(1)))
        if len(parts) < 2 or any(item <= 0 for item in parts):
            return None
        deletion = bool(re.search(r"\b(?:delet|remov)\w*\b", text, re.IGNORECASE))
        if deletion:
            if len(parts) != 2 or not re.search(
                r"(?:one|a\s+single)\s+edge", text, re.IGNORECASE
            ):
                return None
            left, right = parts
            result = (
                left ** (right - 2)
                * right ** (left - 2)
                * (left - 1)
                * (right - 1)
            )
        else:
            total = sum(parts)
            result = total ** (len(parts) - 2)
            for part in parts:
                result *= (total - part) ** (part - 1)
        return f"本地完全多部图生成树: {result}"

    @staticmethod
    def _factor_prime_powers(value: int) -> list[tuple[int, int]]:
        factors = []
        remaining = value
        divisor = 2
        while divisor * divisor <= remaining:
            exponent = 0
            while remaining % divisor == 0:
                remaining //= divisor
                exponent += 1
            if exponent:
                factors.append((divisor, exponent))
            divisor += 1 if divisor == 2 else 2
        if remaining > 1:
            factors.append((remaining, 1))
        return factors

    @staticmethod
    def _parse_positive_product(expression: str) -> Optional[int]:
        value = str(expression or "").replace(r"\cdot", "*").replace(" ", "")
        if not value:
            return None
        result = 1
        for piece in value.split("*"):
            match = re.fullmatch(r"(\d+)(?:\^\{?(\d+)\}?)?", piece)
            if not match:
                return None
            base = int(match.group(1))
            exponent = int(match.group(2) or 1)
            if base <= 0 or exponent < 0 or exponent > 10000:
                return None
            result *= base**exponent
        return result

    @staticmethod
    def _unit_square_roots(modulus: int) -> list[int]:
        roots = [0]
        current_modulus = 1
        for prime, exponent in SympyTool._factor_prime_powers(modulus):
            prime_power = prime**exponent
            if prime == 2:
                if exponent == 1:
                    local = [1]
                elif exponent == 2:
                    local = [1, 3]
                else:
                    half = prime_power // 2
                    local = [1, prime_power - 1, half - 1, half + 1]
            else:
                local = [1, prime_power - 1]
            combined = []
            inverse = pow(current_modulus, -1, prime_power)
            for left, right in product(roots, local):
                offset = ((right - left) * inverse) % prime_power
                combined.append((left + current_modulus * offset) % (current_modulus * prime_power))
            roots = combined
            current_modulus *= prime_power
        return sorted(set(roots))

    @staticmethod
    def _quadratic_congruence_count_hint(problem: str) -> Optional[str]:
        """Count x^2=1 residue classes, optionally in a stated positive range."""
        compact = re.sub(r"\s+", "", str(problem or ""))
        match = re.search(r"x\^2\\equiv1\\pmod\{([^{}]+)\}", compact)
        if not match or not re.search(r"howmany|numberof|多少", compact, re.IGNORECASE):
            return None
        modulus = SympyTool._parse_positive_product(match.group(1))
        if modulus is None or modulus <= 1 or modulus > 10**12:
            return None
        roots = SympyTool._unit_square_roots(modulus)
        bound_match = re.search(r"1\\le(?:q)?x\\le(?:q)?(10\^\{\d+\}|\d+)", compact)
        if bound_match:
            bound = SympyTool._parse_positive_product(bound_match.group(1))
            if bound is None:
                return None
            count = sum(
                0 if residue > bound else (bound - residue) // modulus + 1
                for residue in roots
                if residue > 0
            )
            # The zero residue is not a root for modulus > 1, but retaining
            # this branch keeps the range count correct for future handlers.
            if 0 in roots:
                count += bound // modulus
        else:
            count = len(roots)
        return f"本地二次同余计数: {count}"

    @staticmethod
    def _digit_permutation_divisibility_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        digits = re.search(
            r"digits?\s*\$?0\s*,\s*1\s*,\s*(?:\\ldots|\\dots|\.\.\.)\s*,\s*(\d+)\$?",
            text,
            re.IGNORECASE,
        )
        modulus = re.search(r"divisible\s+by\s+\$?(\d+)\$?", text, re.IGNORECASE)
        if not digits or not modulus or not re.search(r"exactly\s+once", text, re.IGNORECASE):
            return None
        last = int(digits.group(1))
        divisor = int(modulus.group(1))
        if not 1 <= last <= 8 or divisor <= 0:
            return None
        count = 0
        for arrangement in permutations(range(last + 1)):
            if arrangement[0] == 0:
                continue
            residue = 0
            for digit in arrangement:
                residue = (10 * residue + digit) % divisor
            count += residue == 0
        return f"本地数字排列整除计数: {count}"

    @staticmethod
    def _adjacent_surjection_count_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        domain = re.search(r"\\\{1,2,\\(?:ldots|dots),(\d+)\\\}", text)
        codomain = re.search(r"\\to\s*\\\{([^{}]+)\\\}", text)
        if (
            not domain or not codomain
            or not re.search(r"surjective", text, re.IGNORECASE)
            or not re.search(r"f\s*\(i\)\s*\\ne\s*f\s*\(i\s*\+\s*1\)", text)
        ):
            return None
        values = [part.strip() for part in codomain.group(1).split(",")]
        if not values or any(not value.isdigit() for value in values):
            return None
        numeric_values = [int(value) for value in values]
        if numeric_values != list(range(1, len(numeric_values) + 1)):
            return None
        length, colors = int(domain.group(1)), len(numeric_values)
        if not 1 <= length <= 10**5 or not 1 <= colors <= 30:
            return None
        count = 0
        for omitted in range(colors + 1):
            available = colors - omitted
            proper = available * (available - 1) ** (length - 1) if available else 0
            count += (-1) ** omitted * math.comb(colors, omitted) * proper
        return f"本地相邻约束满射计数: {count}"

    @staticmethod
    def _multiset_no_adjacent_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        word = re.search(r"\\mathrm\{([A-Za-z]+)\}", text)
        letter = re.search(
            r"no\s+two\s+copies\s+of\s+the\s+letter\s+\$?([A-Za-z])\$?\s+adjacent",
            text,
            re.IGNORECASE,
        )
        if not word or not letter or not re.search(r"arrangements?", text, re.IGNORECASE):
            return None
        symbols = word.group(1).upper()
        separated = letter.group(1).upper()
        frequencies = Counter(symbols)
        copies = frequencies.pop(separated, 0)
        other_count = sum(frequencies.values())
        if copies <= 1 or copies > other_count + 1 or len(symbols) > 30:
            return None
        arrangements = math.factorial(other_count)
        for frequency in frequencies.values():
            arrangements //= math.factorial(frequency)
        arrangements *= math.comb(other_count + 1, copies)
        return f"本地重复字母隔位计数: {arrangements}"

    @staticmethod
    def _binary_run_avoidance_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        length = re.search(r"binary\s+strings?\s+of\s+length\s+\$?(\d+)\$?", text, re.IGNORECASE)
        forbidden = re.search(
            r"neither\s+\$?(0+)\$?\s+nor\s+\$?(1+)\$?",
            text,
            re.IGNORECASE,
        )
        if not length or not forbidden or len(forbidden.group(1)) != len(forbidden.group(2)):
            return None
        size = int(length.group(1))
        run_limit = len(forbidden.group(1))
        if not 1 <= size <= 10**6 or run_limit <= 1:
            return None
        states = {(0, 1): 1, (1, 1): 1}
        if size == 1:
            return "本地二进制游程计数: 2"
        for _ in range(1, size):
            updated: dict[tuple[int, int], int] = {}
            for (last, run), count in states.items():
                updated[(1 - last, 1)] = updated.get((1 - last, 1), 0) + count
                if run + 1 < run_limit:
                    updated[(last, run + 1)] = updated.get((last, run + 1), 0) + count
            states = updated
        return f"本地二进制游程计数: {sum(states.values())}"

    @staticmethod
    def _bracelet_no_adjacent_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        size = re.search(r"bracelet\s+has\s+\$?(\d+)\$?\s+positions", text, re.IGNORECASE)
        weight = re.search(
            r"exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty)\s+positions\s+are\s+black",
            text,
            re.IGNORECASE,
        )
        if (
            not size or not weight
            or not re.search(r"no\s+two\s+black\s+positions\s+are\s+adjacent", text, re.IGNORECASE)
            or not re.search(r"rotation\s+or\s+a?\s*reflection", text, re.IGNORECASE)
        ):
            return None
        number_words = {
            word: value
            for value, word in enumerate(
                (
                    "zero", "one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                )
            )
        }
        weight_text = weight.group(1).lower()
        n = int(size.group(1))
        k = int(weight_text) if weight_text.isdigit() else number_words[weight_text]
        if not 1 <= n <= 28 or not 0 <= k <= n or math.comb(n, k) > 2_000_000:
            return None
        representatives = set()
        for selected in combinations(range(n), k):
            chosen = set(selected)
            if any(((index + 1) % n) in chosen for index in chosen):
                continue
            bits = tuple(int(index in chosen) for index in range(n))
            reflected = tuple(reversed(bits))
            orbit = [bits[offset:] + bits[:offset] for offset in range(n)]
            orbit.extend(reflected[offset:] + reflected[:offset] for offset in range(n))
            representatives.add(min(orbit))
        return f"本地手链轨道计数: {len(representatives)}"

    @staticmethod
    def _strip_lattice_path_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        endpoint = re.search(
            r"path\s+from\s+\$?\(0\s*,\s*0\)\$?\s+to\s+\$?\((\d+)\s*,\s*(\d+)\)\$?",
            text,
            re.IGNORECASE,
        )
        strip = re.search(
            r"0\s*(?:<=|\\le)\s*x\s*-\s*y\s*(?:<=|\\le)\s*(\d+)",
            text,
        )
        if (
            not endpoint or not strip
            or not re.search(r"monotone\s+lattice\s+path", text, re.IGNORECASE)
            or not re.search(r"steps?\s+\$?\(1\s*,\s*0\).*\(0\s*,\s*1\)", text, re.IGNORECASE)
        ):
            return None
        horizontal, vertical, width = map(int, (*endpoint.groups(), strip.group(1)))
        if horizontal * vertical > 10**7:
            return None
        counts = {(0, 0): 1}
        for x_value in range(horizontal + 1):
            for y_value in range(vertical + 1):
                if (x_value, y_value) == (0, 0) or not 0 <= x_value - y_value <= width:
                    continue
                counts[(x_value, y_value)] = (
                    counts.get((x_value - 1, y_value), 0)
                    + counts.get((x_value, y_value - 1), 0)
                )
        return f"本地条带格路计数: {counts.get((horizontal, vertical), 0)}"

    @staticmethod
    def _nested_modular_sum_hint(problem: str) -> Optional[str]:
        compact = (
            re.sub(r"\s+", "", str(problem or ""))
            .replace(r"\(", "")
            .replace(r"\)", "")
            .replace("$", "")
        )
        match = re.search(
            r"(\d+)\^\{(\d+)\^\{(\d+)\}\}\+"
            r"(\d+)\^\{(\d+)\^\{(\d+)\}\}modulo(\d+(?:\^\{?\d+\}?)?)",
            compact,
            re.IGNORECASE,
        )
        if not match:
            return None
        first, inner_first, power_first, second, inner_second, power_second = map(int, match.groups()[:6])
        modulus = SympyTool._parse_positive_product(match.group(7))
        if modulus is None:
            return None
        if modulus <= 0 or max(power_first, power_second) > 10000:
            return None
        result = (
            pow(first, inner_first**power_first, modulus)
            + pow(second, inner_second**power_second, modulus)
        ) % modulus
        return f"本地嵌套模幂和: {result}"

    def _quadratic_form_maximum_hint(self, problem: str) -> Optional[str]:
        if not self.sympy:
            return None
        text = str(problem or "")
        match = re.search(
            r"maximum\s+value\s+of\s+\$?([^$]+?)\$?\s+over\s+all\s+real\s+triples",
            text,
            re.IGNORECASE,
        )
        if not match or not re.search(
            r"x\^2\s*\+\s*y\^2\s*\+\s*z\^2\s*=\s*1", text
        ):
            return None
        expression = re.sub(r"\s+|\\cdot", "", match.group(1))
        expression = expression.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        terms = re.findall(r"([+-]?\d*)(xy|yz|zx)", expression)
        if (
            not re.fullmatch(r"(?:[+-]?\d*(?:xy|yz|zx)){3}", expression)
            or {name for _, name in terms} != {"xy", "yz", "zx"}
            or len(terms) != 3
        ):
            return None
        coefficients = {}
        for raw, name in terms:
            coefficients[name] = -1 if raw == "-" else 1 if raw in {"", "+"} else int(raw)
        matrix = self.sympy.Matrix([
            [0, self.sympy.Rational(coefficients["xy"], 2), self.sympy.Rational(coefficients["zx"], 2)],
            [self.sympy.Rational(coefficients["xy"], 2), 0, self.sympy.Rational(coefficients["yz"], 2)],
            [self.sympy.Rational(coefficients["zx"], 2), self.sympy.Rational(coefficients["yz"], 2), 0],
        ])
        eigenvalues = tuple(matrix.eigenvals())
        if not eigenvalues:
            return None
        maximum = max(eigenvalues, key=lambda value: float(value.evalf()))
        return f"本地二次型最大值: {self._format(maximum)}"

    @staticmethod
    def _propositional_implication_chain_hint(problem: str) -> Optional[str]:
        """Resolve an explicit two-step implication chain using modus ponens."""
        normalized = (
            str(problem or "")
            .replace(r"\to", "→")
            .replace("->", "→")
            .replace(r"\land", "∧")
            .replace(" ", "")
        )
        chain = re.search(
            r"\(([A-Za-z])→([A-Za-z])\)∧\(\2→([A-Za-z])\)∧\1(?![A-Za-z])",
            normalized,
        )
        if not chain or not re.search(r"推理规则|inference\s+rule", problem, re.IGNORECASE):
            return None
        first, middle, conclusion = chain.group(1, 2, 3)
        return (
            "本地命题逻辑推导: "
            f"由 {first} 与 {first}→{middle} 用假言推理得 {middle}，"
            f"再由 {middle} 与 {middle}→{conclusion} 用假言推理得 {conclusion}；"
            f"故合取范式下必然推出的最简结论为 {conclusion}。"
        )

    @staticmethod
    def _minimum_degree_path_hint(problem: str) -> Optional[str]:
        """Apply the longest-path endpoint argument under explicit bounds."""
        text = str(problem or "")
        graph = re.search(
            r"简单图.*?(\d+)\s*个顶点.*?(?:每个顶点度数|最小度数).*?(?:至少|≥|>=)\s*(\d+)",
            text,
        )
        target = re.search(r"长度(?:至少)?为?\s*(\d+)\s*的路径", text)
        if not graph or not target or not re.search(r"证明|show|prove", text, re.IGNORECASE):
            return None
        vertices, minimum_degree, target_length = map(
            int, (graph.group(1), graph.group(2), target.group(1))
        )
        if vertices < minimum_degree + 1 or minimum_degree < target_length or target_length < 1:
            return None
        return (
            "本地图论路径证明: 取最长路径 P=v_0v_1...v_k；若端点 v_0 有邻点不在 P 中，"
            f"则可延长 P，故 v_0 的至少{minimum_degree}个邻点全在 P 上，于是 k≥{minimum_degree}。"
            f"因此 P 的前{target_length + 1}个顶点构成长为{target_length}的路径；"
            f"所用度数条件为最小度数 δ(G)≥{minimum_degree}。"
        )

    @staticmethod
    def _nonadjacent_binary_string_count_hint(problem: str) -> Optional[str]:
        """Count fixed-weight binary strings with no adjacent ones."""
        text = str(problem or "")
        length = re.search(
            r"长度(?:为|是)?\s*(\d+)|(?:binary\s+strings?).{0,40}?"
            r"(?:of\s+)?length\s*(\d+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        weight = re.search(
            r"恰有\s*(\d+)\s*个?\s*1|(?:exactly|with)\s*(\d+)\s*(?:ones?|1s?)",
            text,
            re.IGNORECASE,
        )
        no_adjacent = re.search(
            r"(?:不含|没有|任意)[^。.!?]{0,24}(?:相邻|连续)[^。.!?]{0,8}(?:两个)?\s*1|"
            r"no\s+(?:two\s+)?(?:ones?|1s?)\s+(?:are\s+)?(?:adjacent|consecutive)|"
            r"without\s+(?:adjacent|consecutive)\s+(?:ones?|1s?)",
            text,
            re.IGNORECASE,
        )
        asks_count = re.search(
            r"(?:串|字符串)(?:的)?(?:数|数量)|多少(?:个)?|"
            r"(?:number|count)\s+of\s+(?:such\s+)?(?:binary\s+)?strings?|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not (length and weight and no_adjacent and asks_count):
            return None
        n = int(next(group for group in length.groups() if group is not None))
        k = int(next(group for group in weight.groups() if group is not None))
        if not (0 <= k <= n <= 10000):
            return None
        gaps = n - k + 1
        result = math.comb(gaps, k) if gaps >= k else 0
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Position selection: arrange the {n-k} zeros first, giving {gaps} gaps; "
            rf"choose {k} gaps, so \(\binom{{{gaps}}}{{{k}}}={result}\)."
            if english else
            rf"插空选位置：先排{n-k}个0得到{gaps}个空位，选择其中{k}个放1，"
            rf"故 \(\binom{{{gaps}}}{{{k}}}={result}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地不相邻二进制串核验" if needs_more else "本地不相邻二进制串计数"
        return f"{label}: {answer}"

    @staticmethod
    def _precedence_permutation_count_hint(problem: str) -> Optional[str]:
        """Count a precedence condition while excluding one first element."""
        text = str(problem or "")
        size = re.search(
            r"(\d+)\s*个不同元素|(?:permutations?\s+of|among)\s*(\d+)\s+distinct\s+elements?",
            text,
            re.IGNORECASE,
        )
        before = re.search(
            r"(?:元素)?\s*([A-Za-z])\s*在\s*([A-Za-z])\s*之前|"
            r"\b(?:have\s+)?([A-Za-z])\s+(?:(?:comes?|is)\s+before|precedes|before)\s+([A-Za-z])\b",
            text,
            re.IGNORECASE,
        )
        excluded = re.search(
            r"([A-Za-z])\s*不在首位|([A-Za-z])\s+is\s+not\s+(?:in\s+)?(?:the\s+)?first(?:\s+position)?",
            text,
            re.IGNORECASE,
        )
        asks_count = re.search(
            r"排列数|多少(?:种|个)?排列|number\s+of\s+(?:such\s+)?permutations?|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not (size and before and excluded and asks_count):
            return None
        n = int(next(group for group in size.groups() if group is not None))
        left, right = (
            (before.group(1), before.group(2))
            if before.group(1) is not None else (before.group(3), before.group(4))
        )
        blocked = excluded.group(1) or excluded.group(2)
        if n < 3 or len({left.lower(), right.lower(), blocked.lower()}) != 3 or n > 1000:
            return None
        all_precedence = math.factorial(n) // 2
        blocked_first = math.factorial(n - 1) // 2
        result = all_precedence - blocked_first
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Conditional counting: \({n}!/2-({n-1})!/2={all_precedence}-{blocked_first}={result}\)."
            if english else
            rf"条件计数：\({n}!/2-({n-1})!/2={all_precedence}-{blocked_first}={result}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地排列条件计数核验" if needs_more else "本地排列条件计数"
        return f"{label}: {answer}"

    @staticmethod
    def _surjection_count_hint(problem: str) -> Optional[str]:
        """Count onto maps between two explicitly finite sets."""
        text = (
            str(problem or "")
            .replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        if not re.search(r"满射|surjections?|onto\s+(?:maps?|functions?)", text, re.IGNORECASE):
            return None
        explicit_sets = re.search(
            r"(?:从|from)\s*(?:集合\s*)?\{([^{}]+)\}\s*(?:到|to)\s*"
            r"(?:集合\s*)?\{([^{}]+)\}",
            text,
            re.IGNORECASE,
        )
        if explicit_sets:
            source_items = [item.strip() for item in re.split(r"[,，]", explicit_sets.group(1)) if item.strip()]
            target_items = [item.strip() for item in re.split(r"[,，]", explicit_sets.group(2)) if item.strip()]
            if (
                len(set(source_items)) != len(source_items)
                or len(set(target_items)) != len(target_items)
            ):
                return None
            n, m = len(source_items), len(target_items)
        else:
            size_match = re.search(
                r"从\s*([一二三四五六七八九十\d]+)\s*元素集合\s*到\s*"
                r"([一二三四五六七八九十\d]+)\s*元素集合|"
                r"from\s+(?:a\s+)?([a-z\d-]+)[ -]element\s+set\s+to\s+"
                r"(?:a\s+)?([a-z\d-]+)[ -]element\s+set",
                text,
                re.IGNORECASE,
            )
            if not size_match:
                return None
            first, second = (
                (size_match.group(1), size_match.group(2))
                if size_match.group(1) is not None else (size_match.group(3), size_match.group(4))
            )
            n = SympyTool._small_integer_word(first)
            m = SympyTool._small_integer_word(second)
        if not n or not m or not (1 <= n <= 1000 and 1 <= m <= 50):
            return None
        asks_count = re.search(
            r"(?:满射)(?:的)?(?:个数|数量)|求[^。.!?]{0,40}满射[^。.!?]{0,12}(?:个数|数量)|"
            r"number\s+of\s+(?:such\s+)?(?:surjections?|onto\s+(?:maps?|functions?))|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not asks_count:
            return None
        result = sum(
            (-1) ** omitted * math.comb(m, omitted) * (m - omitted) ** n
            for omitted in range(m + 1)
        )
        english = SympyTool._uses_english_prose(text)
        formula = rf"\sum_{{j=0}}^{{{m}}}(-1)^j\binom{{{m}}}{{j}}({m}-j)^{{{n}}}={result}"
        answer = (
            rf"Inclusion-exclusion gives \({formula}\)."
            if english else rf"由容斥原理，\({formula}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地满射容斥核验" if needs_more else "本地满射容斥计数"
        return f"{label}: {answer}"

    @staticmethod
    def _small_integer_word(value: str) -> Optional[int]:
        token = str(value or "").strip().lower()
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        if token.isdigit():
            return int(token)
        return words.get(token)

    @staticmethod
    def _planar_euler_face_hint(problem: str) -> Optional[str]:
        """Compute the face count of a connected planar graph and verify Euler."""
        text = str(problem or "")
        chinese = re.search(
            r"连通平面(?:简单)?图.*?(\d+)\s*个顶点.*?(\d+)\s*条边",
            text,
            re.DOTALL,
        )
        english_match = re.search(
            r"connected\s+(?:(?:simple\s+)?planar|planar(?:\s+simple)?)\s+graph"
            r".*?(?:has|with)\s*(\d+)\s+vertices?"
            r".*?(?:and|with)\s*(\d+)\s+edges?",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        match = chinese or english_match
        if not match or not re.search(r"面数|number\s+of\s+faces?|how\s+many\s+faces?", text, re.IGNORECASE):
            return None
        if not re.search(
            r"验证[^。.!?]{0,40}欧拉公式|verify[^.!?]{0,60}euler(?:'s)?\s+formula",
            text,
            re.IGNORECASE,
        ):
            return None
        vertices, edges = map(int, match.groups())
        if vertices < 1 or edges < 0 or vertices > 10**9 or edges > 10**12:
            return None
        faces = edges - vertices + 2
        if faces < 1:
            return None
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Euler's formula gives \(F=E-V+2={edges}-{vertices}+2={faces}\), and "
            rf"\({vertices}-{edges}+{faces}=2\)."
            if english else
            rf"由欧拉公式，\(F=E-V+2={edges}-{vertices}+2={faces}\)，且 "
            rf"\({vertices}-{edges}+{faces}=2\)。"
        )
        extra = bool(re.search(
            r"证明|推广|另求|并求|并计算|"
            r"\b(?:prove|generalize|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地平面图欧拉核验" if extra else "本地平面图欧拉答案"
        return f"{label}: {answer}"

    @staticmethod
    def _paraboloid_curvature_hint(problem: str) -> Optional[str]:
        """Return the exact curvatures of z=x^2+y^2 at the origin."""
        text = re.sub(r"\s+", "", str(problem or ""))
        surface = re.search(
            r"(?:曲面|surface)(?:z=)?x\^\{?2\}?\+y\^\{?2\}?|"
            r"z=x\^\{?2\}?\+y\^\{?2\}?",
            text,
            re.IGNORECASE,
        )
        origin = re.search(r"原点|at(?:the)?origin|\(0,0(?:,0)?\)", text, re.IGNORECASE)
        principal = re.search(r"主曲率|principalcurvatures?", text, re.IGNORECASE)
        gaussian = re.search(r"高斯曲率|gaussiancurvature", text, re.IGNORECASE)
        derivatives = re.search(r"二阶导数|second(?:order)?(?:partial)?derivatives?", text, re.IGNORECASE)
        if not (surface and origin and principal and gaussian and derivatives):
            return None
        english = SympyTool._uses_english_prose(problem)
        answer = (
            r"At the origin, \(f_x=f_y=0\) and \(f_{xx}=2,f_{xy}=0,f_{yy}=2\). "
            r"Thus the shape operator is \(\operatorname{diag}(2,2)\), so "
            r"\(\kappa_1=\kappa_2=2\) and \(K=\kappa_1\kappa_2=4\)."
            if english else
            r"原点处 \(f_x=f_y=0\)，二阶导数为 \(f_{xx}=2,f_{xy}=0,f_{yy}=2\)。"
            r"形算子为 \(\operatorname{diag}(2,2)\)，故 \(\kappa_1=\kappa_2=2\)，"
            r"\(K=\kappa_1\kappa_2=4\)。"
        )
        extra = bool(re.search(
            r"证明|推广|另求|并求|并计算|"
            r"\b(?:prove|generalize|also\s+(?:find|compute|determine))\b",
            str(problem or ""),
            re.IGNORECASE,
        ))
        label = "本地抛物面曲率核验" if extra else "本地抛物面曲率答案"
        return f"{label}: {answer}"

    @staticmethod
    def _ordered_positive_triple_hint(problem: str) -> Optional[str]:
        """Exactly count an explicitly ordered positive-integer triple."""
        normalized = (
            str(problem or "")
            .replace(r"\leq", "≤")
            .replace(r"\le", "≤")
            .replace("<=", "≤")
        )
        total_match = re.search(
            r"([a-z])\s*\+\s*([a-z])\s*\+\s*([a-z])\s*=\s*(\d+)(?!\d)",
            normalized,
            re.IGNORECASE,
        )
        if not total_match or not re.search(r"正整数|positive\s+integers?", normalized, re.IGNORECASE):
            return None
        first, second, third = total_match.group(1, 2, 3)
        if len({first.lower(), second.lower(), third.lower()}) != 3:
            return None
        ordered = re.compile(
            rf"{re.escape(first)}\s*≤\s*{re.escape(second)}\s*≤\s*{re.escape(third)}",
            re.IGNORECASE,
        )
        if not ordered.search(normalized):
            return None
        total = int(total_match.group(4))
        if not 3 <= total <= 10000:
            return None
        counts: list[tuple[int, int]] = []
        for first_value in range(1, total + 1):
            count = sum(
                1
                for second_value in range(first_value, total + 1)
                if total - first_value - second_value >= second_value
            )
            if count:
                counts.append((first_value, count))
        if not counts:
            return None
        cases = "，".join(f"{value}时{count}个" for value, count in counts)
        return f"本地有序三元组计数: 按{first}分类，{first}={cases}，共{sum(count for _, count in counts)}个"

    @staticmethod
    def _simple_random_walk_hint(problem: str) -> Optional[str]:
        """Return exact first and second moments for a named simple symmetric walk."""
        text = str(problem or "")
        if not re.search(r"简单对称随机游走|simple\s+symmetric\s+random\s+walk", text, re.IGNORECASE):
            return None
        if not re.search(r"从\s*0\s*出发|S_?\{?0\}?\s*=\s*0|starts?\s+(?:at|from)\s+0", text, re.IGNORECASE):
            return None
        expectation = re.search(r"E\s*\[\s*S_?\{?(\d+)\}?\s*\]", text, re.IGNORECASE)
        variance = re.search(r"Var\s*\(\s*S_?\{?(\d+)\}?\s*\)", text, re.IGNORECASE)
        if not expectation or not variance or expectation.group(1) != variance.group(1):
            return None
        step = int(expectation.group(1))
        return (
            f"本地随机游走矩: E[S_{step}]=0，Var(S_{step})={step}；由独立增量，"
            f"E[S_{step}]={step}E[X_1]，Var(S_{step})={step}Var(X_1)"
        )

    @staticmethod
    def _complete_graph_cover_time_hint(problem: str) -> Optional[str]:
        """Exact coupon-collector expectation for a complete-graph walk."""
        text = str(problem or "")
        numeric_graph = re.search(r"完全图\s*\$?K_?\{?(\d+)\}?\$?", text)
        if numeric_graph:
            size = int(numeric_graph.group(1))
            words = {
                "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            }
            all_vertices = re.search(r"首次访问全部([一二两三四五六七八九十\d]+)个顶点", text)
            alternatives = re.search(r"每一步等概率走向另外([一二两三四五六七八九十\d]+)个顶点", text)

            def parse_count(value: str) -> Optional[int]:
                return int(value) if value.isdigit() else words.get(value)

            if (
                2 <= size <= 10**6
                and all_vertices and parse_count(all_vertices.group(1)) == size
                and alternatives and parse_count(alternatives.group(1)) == size - 1
                and re.search(r"简单随机游走", text)
                and re.search(r"期望", text)
                and not re.search(r"懒惰|加权|自环", text)
            ):
                expectation = (size - 1) * sum(
                    (Fraction(1, index) for index in range(1, size)),
                    Fraction(),
                )
                answer = (
                    str(expectation.numerator)
                    if expectation.denominator == 1
                    else f"{expectation.numerator}/{expectation.denominator}"
                )
                return f"本地完全图覆盖时间: {answer}"
        if not re.search(
            r"(?:\$\s*)?N(?:\s*\$|\\\))?\s*个顶点的完全图|"
            r"complete\s+graph\s+(?:on|with)\s+(?:\$\s*)?N(?:\s*\$)?\s+vertices",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"首次遍访所有顶点|cover\s+time|visited?\s+all\s+vertices", text, re.IGNORECASE):
            return None
        if not re.search(r"简单随机游动|simple\s+random\s+walk", text, re.IGNORECASE):
            return None
        if re.search(r"lazy|weighted|加权|懒惰|自环|self[- ]?loop", text, re.IGNORECASE):
            return None
        if not re.search(r"(?:求|find|compute)\s*E\s*T|期望", text, re.IGNORECASE):
            return None
        return r"本地完全图覆盖时间: (N-1)\sum_{j=1}^{N-1}\frac{1}{j}"

    @staticmethod
    def _two_venue_capacity_hint(problem: str) -> Optional[str]:
        """Invert an exact symmetric binomial tail for two equiprobable venues."""
        text = str(problem or "")
        population = re.search(r"(\d+)\s*名(?:市民|观众|顾客)", text)
        threshold = re.search(r"概率不超过\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not population or not threshold:
            return None
        if not re.search(r"两个(?:剧院|场馆).*(?:独立)?等可能", text, re.DOTALL):
            return None
        if not re.search(
            r"(?:每个|各).*?有\s*(?:\\\(\s*)?[xX](?:\s*\\\))?\s*个座位|"
            r"(?:\\\(\s*)?[xX](?:\s*\\\))?\s*的最小值",
            text,
            re.DOTALL,
        ):
            return None
        count = int(population.group(1))
        if count < 1 or count > 20000:
            return None
        probability = Fraction(threshold.group(1))
        total_outcomes = 1 << count
        tail = 0
        largest_tail_index = -1
        for index in range(0, count // 2):
            proposed = tail + math.comb(count, index)
            if 2 * proposed * probability.denominator > probability.numerator * total_outcomes:
                break
            tail = proposed
            largest_tail_index = index
        capacity = count - largest_tail_index - 1
        return f"本地二项分布容量: {capacity}"

    @staticmethod
    def _circle_laplacian_hint(problem: str) -> Optional[str]:
        """Separate the ambient Laplacian from the circle Laplace--Beltrami operator."""
        text = str(problem or "")
        compact = re.sub(r"\s+", "", text).lower()
        if re.search(
            r"bi\s*[-–—]?\s*laplacian|biharmonic|双拉普拉斯|双调和|"
            r"weighted\s+laplacian|加权拉普拉斯|p\s*[-–—]?\s*laplacian|p\s*[-–—]?\s*拉普拉斯",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"拉普拉斯|laplacian|laplace\s*[-–—]?\s*beltrami", text, re.IGNORECASE):
            return None
        expression = bool(re.search(
            r"f\(x,y\)=x(?:\^\{?2\}?|²)\+y(?:\^\{?2\}?|²)", compact,
        ))
        circle = bool(
            re.search(r"圆周|circle|s\^?1", text, re.IGNORECASE)
            and re.search(
                r"x(?:\^\{?2\}?|²)\+y(?:\^\{?2\}?|²)="
                r"(?:1|[1-9]\d*(?:\^\{?2\}?|²)?|[a-z](?:\^\{?2\}?|²))",
                compact,
                re.IGNORECASE,
            )
        )
        if not (expression and circle):
            return None

        explicit_intrinsic = bool(re.search(
            r"laplace\s*[-–—]?\s*beltrami|laplacebeltrami|"
            r"拉普拉斯\s*[-–—]?\s*贝尔特拉米|拉普拉斯贝尔特拉米|"
            r"内蕴拉普拉斯|intrinsic\s+laplacian|"
            r"(?:限制函数|限制到|restriction\s+of|restricted\s+to|f\s*\|)"
            r"[^。.!?]{0,40}(?:拉普拉斯|laplacian)",
            text,
            re.IGNORECASE,
        ))
        explicit_ambient = bool(re.search(
            r"环境拉普拉斯|欧氏拉普拉斯|ambient\s+laplacian|euclidean\s+laplacian|"
            r"\\Delta_?\{?\\mathbb\s*\{?R\}?\^?2\}?",
            text,
            re.IGNORECASE,
        ))
        explicit_ambiguity = bool(
            re.search(
                r"(?:环境|欧氏|ambient|euclidean).{0,30}(?:还是|或|或者|or|versus|vs\.?)"
                r".{0,30}(?:内蕴|贝尔特拉米|intrinsic|beltrami)|"
                r"(?:内蕴|贝尔特拉米|intrinsic|beltrami).{0,30}"
                r"(?:还是|或|或者|or|versus|vs\.?).{0,30}(?:环境|欧氏|ambient|euclidean)|"
                r"(?:未说明|不明确|有歧义|unspecified|ambiguous).{0,20}"
                r"(?:拉普拉斯|laplacian)",
                text,
                re.IGNORECASE,
            )
            or (explicit_intrinsic and explicit_ambient)
        )
        needs_support = bool(re.search(
            r"证明|推导|解释|说明理由|"
            r"\b(?:prove|derive|justify|explain|show\s+why)\b",
            text,
            re.IGNORECASE,
        ))

        if explicit_ambiguity:
            return (
                "本地圆周拉普拉斯歧义核验: "
                r"若指环境算子，则 \(\Delta_{\mathbb R^2}f=f_{xx}+f_{yy}=2+2=4\)；"
                r"若指限制函数的 Laplace--Beltrami 算子，则 \(f|_{S^1}\) 为常数，故值为 \(0\)"
            )
        if explicit_intrinsic:
            label = "本地圆周Laplace-Beltrami核验" if needs_support else "本地圆周Laplace-Beltrami"
            return f"{label}: 0"
        label = "本地圆周拉普拉斯核验" if needs_support else "本地圆周拉普拉斯"
        return f"{label}: 4"

    @staticmethod
    def _central_difference_hint(problem: str) -> Optional[str]:
        """Compute the named centered first-difference formula for sin."""
        text = str(problem or "")
        if not re.search(r"中心差分|central\s+difference", text, re.IGNORECASE):
            return None
        if not re.search(r"f\s*\(\s*x\s*\)\s*=\s*(?:\\sin|sin)\s*\(\s*x\s*\)", text, re.IGNORECASE):
            return None
        point = re.search(r"x\s*=\s*(?:\\pi|π)\s*/\s*(\d+)", text, re.IGNORECASE)
        step = re.search(r"h\s*=\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not point or not step:
            return None
        denominator = int(point.group(1))
        h_value = float(step.group(1))
        if denominator <= 0 or not (0 < h_value <= 1):
            return None
        x_value = math.pi / denominator
        approximation = (
            math.sin(x_value + h_value) - math.sin(x_value - h_value)
        ) / (2 * h_value)
        return f"本地中心差分: {approximation:.4f}"

    @staticmethod
    def _rational_f2_constraint_hint(problem: str) -> Optional[str]:
        """Propagate the exact parity invariant generated by three rational involutions."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        function = re.search(
            r"([A-Za-z])\s*:\s*\\mathbb\s*\{Q\}\s*\\rightarrow\s*"
            r"\\mathbb\s*\{F\}\s*_?\s*\{?2\}?",
            text,
        )
        if not function:
            return None
        name = re.escape(function.group(1))
        if not re.search(
            rf"{name}\s*\(\s*r\s*\)\s*\+\s*{name}\s*\(\s*r['’]?\s*\)\s*=\s*1",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"distinct\s+rational\s+numbers", text, re.IGNORECASE):
            return None
        required_relations = (
            r"r\s*\+\s*r['’]?\s*=\s*0",
            r"r\s*\+\s*r['’]?\s*=\s*1",
            r"r\s*r['’]?\s*=\s*1",
        )
        if any(not re.search(pattern, text, re.IGNORECASE) for pattern in required_relations):
            return None
        seed = re.search(
            rf"{name}\s*\(\s*([+-]?\d+)\s*/\s*(\d+)\s*\)\s*=\s*([01])",
            text,
        )
        request = re.search(r"\b(?:evaluate|compute|find)\b", text, re.IGNORECASE)
        if not seed or not request:
            return None
        denominator = int(seed.group(2))
        if denominator == 0:
            return None
        seed_value = Fraction(int(seed.group(1)), denominator)
        seed_bit = int(seed.group(3))
        query = text[request.end():]
        terms = re.findall(
            rf"{name}\s*\(\s*([+-]?\d+)(?:\s*/\s*(\d+))?\s*\)",
            query,
        )
        if not terms or len(terms) > 100:
            return None
        rationals = []
        for numerator_text, denominator_text in terms:
            term_denominator = int(denominator_text or "1")
            if term_denominator == 0:
                return None
            rationals.append(Fraction(int(numerator_text), term_denominator))

        seed_colour = SympyTool._rational_involution_colour(seed_value)
        result = 0
        for value in rationals:
            result ^= seed_bit ^ seed_colour ^ SympyTool._rational_involution_colour(value)
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+why)\b",
            query,
            re.IGNORECASE,
        ))
        label = "本地有理数约束传播核验" if needs_support else "本地有理数约束传播答案"
        return f"{label}: {result}"

    @staticmethod
    def _rational_involution_colour(value: Fraction) -> int:
        """Two-colour Q for edges r~-r, r~1-r and r~1/r."""
        if value == 0:
            return 1
        flips = int(value < 0)
        numerator = abs(value.numerator)
        denominator = value.denominator
        while numerator != denominator:
            if numerator > denominator:
                # Repeated integer translations use two involutions and keep colour.
                numerator = (numerator - 1) % denominator + 1
            else:
                numerator, denominator = denominator, numerator
                flips ^= 1
        return flips

    @staticmethod
    def _digit_sum_window_hint(problem: str) -> Optional[str]:
        """Find the first window whose digit sums all avoid a requested divisor."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        if not re.search(
            r"S\s*\(\s*n\s*\)\s*\$?\s+(?:be|is).*?sum\s+of\s+the\s+digits.*?"
            r"decimal\s+representation.*?positive\s+integer",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"smallest\s+positive\s+integer\s+\$?n\$?", text, re.IGNORECASE):
            return None
        offset = re.search(
            r"S\s*\(\s*n\s*\)\s*S\s*\(\s*n\s*\+\s*1\s*\)\s*"
            r"(?:\\cdots|\\dots|\.\.\.).*?S\s*\(\s*n\s*\+\s*(\d+)\s*\)",
            text,
            re.IGNORECASE,
        )
        modulus = re.search(r"not\s+a\s+multiple\s+of\s+\$?(\d+)\$?", text, re.IGNORECASE)
        if not offset or not modulus:
            return None
        window_offset = int(offset.group(1))
        divisor = int(modulus.group(1))
        if not 1 <= window_offset <= 200 or not 2 <= divisor <= 100:
            return None
        window_size = window_offset + 1
        search_limit = 2_000_000

        def divisible_digit_sum(number: int) -> int:
            total = 0
            value = number
            while value:
                total += value % 10
                value //= 10
            return int(total % divisor == 0)

        bad = sum(divisible_digit_sum(number) for number in range(1, window_size + 1))
        answer = None
        for start in range(1, search_limit + 1):
            if bad == 0:
                answer = start
                break
            bad -= divisible_digit_sum(start)
            bad += divisible_digit_sum(start + window_size)
        if answer is None:
            return None
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+that)\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地数位和窗口最小性核验" if needs_support else "本地数位和窗口答案"
        return f"{label}: {answer}"

    @staticmethod
    def _number_writing_game_hint(problem: str) -> Optional[str]:
        """Exhaust a fully specified ``n+1``/``2n`` normal-play game.

        The matcher intentionally verifies every rule needed to identify a game
        position.  A computed value is only labelled as a complete answer when
        the prompt asks for that value alone; requested proofs or strategies
        still receive the value as local evidence for the model solver.
        """
        text = str(problem or "")
        compact = re.sub(r"\s+", " ", text).strip()
        if not re.search(
            r"two\s+players\s+[\$({]*A[\$)}]*\s+and\s+[\$({]*B[\$)}]*.*?"
            r"taking\s+turns\s+writing\s+numbers?",
            compact,
            re.IGNORECASE,
        ):
            return None
        required_rules = (
            r"set\s*\$?\\?\{\s*1\s*,\s*(?:\\dots|\\ldots|\.\.\.)\s*,\s*N\s*\\?\}\$?",
            r"\$?N\$?\s+is\s+a\s+positive\s+integer",
            r"player\s*\$?A\$?\s+starts?\s+(?:the\s+game\s+)?by\s+writing\s+(?:the\s+number\s+)?\$?1\$?",
            r"if\s+a\s+player\s+writes?\s+(?:the\s+number\s+)?\$?n\$?.*?"
            r"other\s+player\s+can\s+write\s+either\s+\$?n\s*\+\s*1\$?\s+or\s+\$?2\s*n\$?",
            r"provided\s+(?:that\s+)?the\s+number\s+does\s+not\s+exceed\s+\$?N(?!\s*[+\-*/])\$?",
            r"player\s+who\s+writes?\s+(?:the\s+number\s+)?\$?N(?!\s*[+\-*/])\$?\s+wins?",
            r"\$?N\$?\s+is\s+of\s+type\s+\$?A\$?.*?player\s+\$?A\$?\s+has\s+a\s+winning\s+strategy",
            r"(?:\$?N\$?\s+is\s+|and\s+)of\s+type\s+\$?B\$?.*?"
            r"player\s+\$?B\$?\s+has\s+a\s+winning\s+strategy",
        )
        if any(not re.search(rule, compact, re.IGNORECASE) for rule in required_rules):
            return None
        if re.search(
            r"player\s+who\s+(?:cannot|can\s+not)\s+move|no\s+legal\s+move|"
            r"may\s+also\s+write|instead\s+of",
            compact,
            re.IGNORECASE,
        ):
            return None
        request = re.search(
            r"find\s+the\s+least\s+\$?N\s*>\s*(\d+)\$?\s+such\s+that\s+"
            r"it\s+is\s+a\s+type\s+\$?([AB])\$?\s+number",
            compact,
            re.IGNORECASE,
        )
        if not request:
            return None
        threshold = int(request.group(1))
        requested_type = request.group(2).upper()
        # Keep exhaustive search predictably cheap.  Refusing a larger instance
        # is safer than presenting an unchecked heuristic as certified output.
        if not 1 <= threshold <= 1000:
            return None
        search_limit = max(2048, 2 * threshold + 1024)
        candidate = next(
            (
                limit
                for limit in range(threshold + 1, search_limit + 1)
                if SympyTool._number_game_type(limit) == requested_type
            ),
            None,
        )
        if candidate is None:
            return None
        request_tail = compact[request.end():]
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+that|give|describe)\b.*?"
            r"\b(?:strategy|reason|proof|derivation)\b",
            compact,
            re.IGNORECASE,
        ) or re.search(r"\b(?:also|in\s+addition|and\s+then)\b", request_tail, re.IGNORECASE))
        label = "本地取数博弈状态核验" if needs_support else "本地取数博弈答案"
        return f"{label}: {candidate}"

    @staticmethod
    def _number_game_type(limit: int) -> str:
        """Return the winning player's type after A has written the initial 1."""
        winning = bytearray(limit + 1)
        for current in range(limit - 1, 0, -1):
            plus_one_loses = not winning[current + 1]
            doubled_loses = 2 * current <= limit and not winning[2 * current]
            winning[current] = plus_one_loses or doubled_loses
        # At position 1 it is B's turn.  A losing position for that player means
        # the initial writer A has the winning strategy.
        return "B" if winning[1] else "A"

    @staticmethod
    def _path_independent_set_partition_hint(problem: str) -> Optional[str]:
        """Compute the hard-core partition polynomial of an explicitly named path."""
        text = str(problem or "")
        compact = re.sub(r"\s+", " ", text).strip()
        if not re.search(
            r"(?:let\s+)?\$?P_?\{?n\}?\$?\s+(?:be|is)\s+a\s+path\s+on\s+"
            r"\$?n\$?\s+vertices",
            compact,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"\$?\\lambda\$?\s+(?:be|is)\s+a\s+positive\s+real\s+number",
            compact,
            re.IGNORECASE,
        ):
            return None
        if not (
            re.search(r"define\s+\$?Z_?\{?P_?\{?n\}?\}?\s*\(\s*\\lambda\s*\)", compact, re.IGNORECASE)
            and re.search(r"=\s*\\sum_?\{?\s*I\s*\\in", compact, re.IGNORECASE)
            and re.search(r"\\lambda\s*\^\s*\{?\s*\|\s*I\s*\|\s*\}?", compact, re.IGNORECASE)
            and re.search(r"independent\s+sets?\s+of\s+\$?P_?\{?n\}?", compact, re.IGNORECASE)
        ):
            return None
        request = re.search(
            r"(?:compute|find|determine)\s+(?:the\s+value\s+of\s+)?\$?"
            r"[zZ]_?\{?(?:P_?\{?)?(\d+)\}?\}?"
            r"(?:\s*\(\s*\\lambda\s*\))?\$?\s+in\s+terms\s+of\s+\$?\\lambda\$?",
            compact,
            re.IGNORECASE,
        )
        if not request:
            return None
        vertices = int(request.group(1))
        if not 1 <= vertices <= 100:
            return None
        polynomial = SympyTool._path_partition_polynomial(vertices)
        request_tail = compact[request.end():]
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show|establish)\b|"
            r"(?:also|and)\s+(?:give|find|derive|show).*?\brecurrence\b",
            compact,
            re.IGNORECASE,
        ) or re.search(r"\b(?:also|in\s+addition|and\s+then)\b", request_tail, re.IGNORECASE))
        label = "本地路径配分函数递推核验" if needs_support else "本地路径配分函数答案"
        return f"{label}: {polynomial}"

    @staticmethod
    def _path_partition_polynomial(vertices: int) -> str:
        """Apply ``Z_n=Z_{n-1}+lambda*Z_{n-2}`` from ``Z_0,Z_1`` exactly."""
        previous_two = [1]
        previous_one = [1, 1]
        for _ in range(2, vertices + 1):
            coefficients = previous_one.copy()
            if len(coefficients) < len(previous_two) + 1:
                coefficients.append(0)
            for size, coefficient in enumerate(previous_two, start=1):
                coefficients[size] += coefficient
            previous_two, previous_one = previous_one, coefficients
        result = previous_one
        terms = [str(result[0])]
        for size, coefficient in enumerate(result[1:], start=1):
            variable = r"\lambda" if size == 1 else rf"\lambda^{{{size}}}"
            terms.append(variable if coefficient == 1 else f"{coefficient}{variable}")
        return "+".join(terms)

    @staticmethod
    def _spike_sequence_construction_hint(problem: str) -> Optional[str]:
        """Provide the canonical unit-mass spike only for its exact contract."""
        text = str(problem or "")
        english = SympyTool._uses_english_prose(text)
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace("，", ",")
        )
        construct = bool(re.search(
            r"构造|写出.*(?:函数列|例子)|\b(?:construct|exhibit|give|find)\b.*?"
            r"(?:sequence|example)",
            normalized,
            re.IGNORECASE | re.DOTALL,
        ))
        function_sequence = bool(re.search(
            r"函数列|f\s*_?\s*\{?n\}?|sequence\s+of\s+(?:nonnegative\s+)?(?:measurable\s+)?functions?",
            normalized,
            re.IGNORECASE,
        ))
        convergence_context = bool(re.search(
            r"逐点|收敛|趋于|极限|pointwise|converge|tend|limit",
            normalized,
            re.IGNORECASE,
        ))
        integral_context = bool(re.search(r"积分|\\int|\bintegrals?\b", normalized, re.IGNORECASE))
        if not (construct and function_sequence and convergence_context and integral_context):
            return None

        exact_conditions = {
            "domain": bool(re.search(r"\[\s*0\s*,\s*1\s*\]", normalized)),
            "nonnegative": bool(re.search(r"非负|non[- ]?negative", normalized, re.IGNORECASE)),
            "measurable": bool(re.search(r"可测|measurable", normalized, re.IGNORECASE)),
            "pointwise_zero": bool(re.search(
                r"逐点\s*(?:收敛|趋(?:于|向)|极限(?:为|是)?)\s*(?:到|至)?\s*0|"
                r"(?:converges?|tends?)\s+pointwise\s+to\s+0|"
                r"pointwise.{0,24}(?:converges?|tends?|limit).{0,12}0",
                normalized,
                re.IGNORECASE,
            )),
            "unit_integral": bool(re.search(
                r"积分\s*(?:恒|始终)?\s*(?:等于|为|=)\s*1(?![\d/.])|"
                r"(?:integral|\\int).{0,40}(?:equals?|equal\s+to|is|=)\s*1(?![\d/.])",
                normalized,
                re.IGNORECASE,
            )),
            "formula": bool(re.search(
                r"具体公式|显式公式|写出.{0,12}公式|explicit\s+formula|"
                r"(?:give|write|state).{0,16}(?:formula|expression)",
                normalized,
                re.IGNORECASE,
            )),
        }
        extra_obligation = bool(re.search(
            r"证明|说明(?:理由|为什么|为何)|解释|验证|推广|比较|讨论|"
            r"并\s*(?:求|计算|证明|说明|验证|比较|讨论)|"
            r"范数|上确界|下确界|一致收敛|依测度收敛|控制收敛|"
            r"\b(?:prove|justify|explain|verify|generalize|compare|discuss|also|"
            r"supremum|infimum|norm|uniform\s+convergence|convergence\s+in\s+measure|"
            r"dominated\s+convergence)\b",
            normalized,
            re.IGNORECASE,
        ))
        result = (
            (
                r"Take \(f_n(x)=n\mathbf{1}_{(0,1/n]}(x)\) for \(x\in[0,1]\). "
                r"It is nonnegative and measurable, \(f_n(x)\to0\) pointwise on \([0,1]\), "
                r"and \(\int_0^1 f_n(x)\,dx=1\)."
            )
            if english else (
                r"取 f_n(x)=n\mathbf{1}_{(0,1/n]}(x)\ (x\in[0,1])；"
                r"则 f_n\geq0 且可测，逐点 f_n(x)\to0\ (\forall x\in[0,1])，"
                r"积分为 \int_0^1 f_n(x)\,dx=1。"
            )
        )
        if not all(exact_conditions.values()) or extra_obligation:
            missing = ",".join(name for name, present in exact_conditions.items() if not present)
            reason = "存在额外证明或计算义务" if extra_obligation else f"标准条件未完整匹配({missing})"
            return f"本地尖峰函数构造核验: {result} 仅核验上述标准构造；{reason}。"
        return f"本地尖峰函数构造答案: {result}"

    @staticmethod
    def _dependent_bernoulli_construction_hint(problem: str) -> Optional[str]:
        """Construct perfectly dependent fair Bernoulli marginals when asked exactly."""
        text = str(problem or "")
        english = SympyTool._uses_english_prose(text)
        normalized = re.sub(r"\s+", " ", text).strip()
        construct = bool(re.search(r"构造|\b(?:construct|exhibit|give)\b", normalized, re.IGNORECASE))
        random_variables = bool(re.search(
            r"随机变量|random\s+variables?",
            normalized,
            re.IGNORECASE,
        ))
        bernoulli = bool(re.search(r"Bernoulli|伯努利", normalized, re.IGNORECASE))
        dependence_context = bool(re.search(
            r"独立|不(?:相互)?独立|非独立|not\s+independent|\bdependent\b|\bindependent\b",
            normalized,
            re.IGNORECASE,
        ))
        if not (construct and random_variables and bernoulli and dependence_context):
            return None

        fair = r"(?:Bernoulli|伯努利)\s*[（(]\s*(?:1\s*/\s*2|0\.5)\s*[）)]"
        two_variables = bool(re.search(
            r"两个.{0,60}随机变量|random\s+variables?\s+X\s+and\s+Y|"
            r"two\s+(?:Bernoulli\s+)?random\s+variables?",
            normalized,
            re.IGNORECASE,
        ))
        fair_marginals = bool(
            re.search(
                rf"(?:两个)?边缘(?:分布)?.{{0,12}}(?:均|都).{{0,12}}{fair}|"
                rf"both\s+(?:marginals?|marginal\s+distributions?).{{0,12}}{fair}|"
                rf"(?:marginals?|marginal\s+distributions?).{{0,12}}(?:are\s+)?both.{{0,12}}{fair}|"
                rf"X\s+and\s+Y.{{0,20}}(?:both|each).{{0,16}}{fair}",
                normalized,
                re.IGNORECASE,
            )
        )
        not_independent = bool(re.search(
            r"不(?:相互)?独立|非独立|not\s+independent|\bdependent\b",
            normalized,
            re.IGNORECASE,
        ))
        equality_probability = bool(re.search(
            r"(?:P|\\mathbb\s*\{P\})\s*[（(]\s*X\s*=\s*Y\s*[）)]",
            normalized,
            re.IGNORECASE,
        ))

        probability_events = [
            re.sub(r"\s+", "", event).upper()
            for event in re.findall(
                r"(?:P|\\mathbb\s*\{P\})\s*[（(]\s*([^）)]+)\s*[）)]",
                normalized,
                re.IGNORECASE,
            )
        ]
        extra_probability = any(event not in {"X=Y", "Y=X"} for event in probability_events)
        extra_obligation = extra_probability or bool(re.search(
            r"证明|说明(?:理由|为什么|为何)|解释|验证|协方差|相关系数|相关性|"
            r"联合分布(?:表)?|条件概率|期望|方差|熵|互信息|不相关|"
            r"\b(?:prove|justify|explain|verify|covariance|correlation|uncorrelated|"
            r"joint\s+distribution|conditional\s+probability|expectation|variance|"
            r"entropy|mutual\s+information)\b",
            normalized,
            re.IGNORECASE,
        ))
        result = (
            (
                r"Let \(P((X,Y)=(0,0))=P((X,Y)=(1,1))=1/2\), with probability zero "
                r"otherwise (equivalently, \(X\sim\operatorname{Bernoulli}(1/2)\) and \(Y=X\)). "
                r"Both marginals are \(\operatorname{Bernoulli}(1/2)\), and "
                r"\(P(X=1,Y=1)=1/2\neq1/4=P(X=1)P(Y=1)\), so they are not independent; "
                r"\(P(X=Y)=1\)."
            )
            if english else (
                r"取 P((X,Y)=(0,0))=P((X,Y)=(1,1))=1/2，其余情形概率为0（即 "
                r"X\sim\operatorname{Bernoulli}(1/2),\ Y=X）。两边缘均为 "
                r"\operatorname{Bernoulli}(1/2)，且 P(X=1,Y=1)=1/2\neq1/4="
                r"P(X=1)P(Y=1)，故 X,Y 不独立；所求概率 P=1，即 P(X=Y)=1。"
            )
        )
        exact = two_variables and fair_marginals and not_independent and equality_probability
        if not exact or extra_obligation:
            reason = "存在额外证明或计算义务" if extra_obligation else "公平Bernoulli边缘、非独立或目标概率条件未完整匹配"
            return f"本地Bernoulli依赖构造核验: {result} 仅核验上述标准构造；{reason}。"
        return f"本地Bernoulli依赖构造答案: {result}"

    @staticmethod
    def _uses_english_prose(problem: str) -> bool:
        """Choose tool-answer prose without counting one-letter math variables."""
        value = re.sub(
            r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
            " ",
            str(problem or ""),
            flags=re.DOTALL,
        )
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", value))
        english_words = len(re.findall(r"\b[A-Za-z]{2,}\b", value))
        return english_words >= 2 and english_words > chinese_chars

    @staticmethod
    def _lz78_encoding_hint(problem: str) -> Optional[str]:
        """Encode a fully specified, standard fixed-width LZ78 exercise.

        A complete answer is certified only for the empty-dictionary LZ78
        convention (empty string at index 0), an explicit one-character binary
        alphabet map, and an input that ends when a new phrase is emitted.  The
        final dictionary has ``m`` phrases, so every prefix index that an output
        pair may contain lies in ``0,...,m-1`` and needs ``ceil(log2(m))`` bits.
        Variant or incomplete statements still receive a deterministic phrase
        check, but are never presented as a complete encoded answer.
        """
        text = str(problem or "")
        if not re.search(r"\b(?:LZ\s*78|Lempel[- ]?Ziv)\b", text, re.IGNORECASE):
            return None
        if re.search(r"\bLZ\s*77\b|sliding\s+window|滑动窗口", text, re.IGNORECASE):
            return None
        asks_phrases = bool(re.search(
            r"(?:decomposition\s+into\s+phrases|phrase\s+decomposition|"
            r"decompos(?:e|ition).*?phrases?|短语分解|分解.*?短语)",
            text,
            re.IGNORECASE | re.DOTALL,
        ))
        asks_encoding = bool(re.search(
            r"(?:encoded\s+string|encode(?:d|s|ing)?\s+(?:the\s+)?(?:message|string)|"
            r"编码(?:串|结果|该?(?:消息|字符串)))",
            text,
            re.IGNORECASE,
        ))
        if not (asks_phrases and asks_encoding):
            return None

        message = SympyTool._lz78_message(text)
        if not message or len(message) > 10000:
            return None
        pairs, phrases, terminal_prefix = SympyTool._lz78_parse(message)
        if not pairs:
            return None

        pair_text = ", ".join(f"({index},{symbol})" for index, symbol in pairs)
        phrase_text = ", ".join(phrases)
        base = f"Phrases: {phrase_text}; pairs: {pair_text}"
        issues: list[str] = []
        if terminal_prefix:
            issues.append(
                "the input ends in an existing dictionary phrase "
                f"{terminal_prefix!r}, so an EOF convention is required"
            )

        mapping, mapping_issue = SympyTool._lz78_letter_mapping(text)
        if mapping_issue:
            issues.append(mapping_issue)
        missing_symbols = sorted(set(message) - set(mapping))
        if missing_symbols:
            issues.append("the explicit bit mapping omits " + ", ".join(missing_symbols))

        phrase_count = len(pairs)
        derived_width = max(1, (phrase_count - 1).bit_length())
        explicit_widths = SympyTool._lz78_explicit_index_widths(text)
        if len(explicit_widths) > 1:
            issues.append("conflicting index widths are stated")
            index_width = derived_width
        elif explicit_widths:
            index_width = next(iter(explicit_widths))
            if index_width < derived_width:
                issues.append(
                    f"the stated {index_width}-bit index cannot represent all prefix indices 0,...,{phrase_count - 1}"
                )
            elif index_width > 64:
                issues.append("the stated index width is outside the supported deterministic range")
        else:
            index_width = derived_width

        if re.search(
            r"(?:index|indices|dictionary\s+entries?).{0,24}(?:start|begin)(?:s|ning)?\s+(?:at|from)\s+1|"
            r"(?:索引|下标).{0,12}从\s*1\s*开始|"
            r"(?:preloaded|initial(?:ly)?\s+contains|initial\s+dictionary\s+(?:is\s+)?(?:not\s+empty|contains)|"
            r"预置字典|初始字典.{0,8}(?:非空|包含))|"
            r"(?:variable|dynamic|adaptive)[- ](?:width|length)\s+(?:index|code)|"
            r"(?:变长|动态|自适应).{0,8}(?:索引|编码)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            issues.append("the problem specifies a nonstandard dictionary or index convention")

        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+why)\b|证明|说明理由|解释|推导",
            text,
            re.IGNORECASE,
        ))
        if needs_support:
            issues.append("the requested justification is not covered by the deterministic encoding result")

        if issues:
            return f"本地LZ78编码核验: {base}; verification only: {'; '.join(issues)}"

        encoded_chunks = [
            f"{index:0{index_width}b}{mapping[symbol]}"
            for index, symbol in pairs
        ]
        width_reason = (
            f"stated index width: {index_width} bits"
            if explicit_widths
            else (
                f"fixed index width: ceil(log2({phrase_count}))={index_width} bits "
                f"for prefix indices 0,...,{phrase_count - 1}"
            )
        )
        return (
            f"本地LZ78编码答案: {base}; {width_reason}; "
            f"encoded string: {' '.join(encoded_chunks)}"
        )

    @staticmethod
    def _lz78_message(text: str) -> str:
        """Extract one explicit alphanumeric message token from public text."""
        patterns = (
            r"(?:consider\s+the\s+)?message\s*(?:is|=|:|：)?\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
            r"(?:encode|compress)\s+(?:the\s+)?(?:message|string)\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
            r"(?:消息|报文|字符串)\s*(?:为|是|=|:|：)\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1).lower() not in {
                "is", "the", "a", "an", "using", "obtained", "string",
            }:
                return match.group(1)
        return ""

    @staticmethod
    def _lz78_parse(message: str) -> tuple[list[tuple[int, str]], list[str], str]:
        """Return standard LZ78 output pairs and any ambiguous terminal prefix."""
        dictionary = {"": 0}
        pairs: list[tuple[int, str]] = []
        phrases: list[str] = []
        position = 0
        while position < len(message):
            cursor = position
            prefix = ""
            while cursor < len(message) and prefix + message[cursor] in dictionary:
                prefix += message[cursor]
                cursor += 1
            if cursor == len(message):
                return pairs, phrases, prefix
            symbol = message[cursor]
            phrase = prefix + symbol
            pairs.append((dictionary[prefix], symbol))
            phrases.append(phrase)
            dictionary[phrase] = len(dictionary)
            position = cursor + 1
        return pairs, phrases, ""

    @staticmethod
    def _lz78_letter_mapping(text: str) -> tuple[dict[str, str], str]:
        normalized = (
            text.replace(r"\rightarrow", "->")
            .replace(r"\longrightarrow", "->")
            .replace(r"\to", "->")
            .replace("→", "->")
        )
        entries = re.findall(
            r"(?<![A-Za-z0-9])([A-Za-z0-9])\s*->\s*([01]+)(?![01])",
            normalized,
        )
        mapping: dict[str, str] = {}
        for symbol, bits in entries:
            if symbol in mapping and mapping[symbol] != bits:
                return mapping, f"conflicting bit codes are given for {symbol}"
            mapping[symbol] = bits
        if not mapping:
            return {}, "no explicit letter-to-bit mapping was found"
        widths = {len(bits) for bits in mapping.values()}
        if len(widths) != 1:
            return mapping, "the explicit letter codes do not have one fixed width"
        if len(set(mapping.values())) != len(mapping):
            return mapping, "the explicit letter-to-bit mapping is not one-to-one"
        return mapping, ""

    @staticmethod
    def _lz78_explicit_index_widths(text: str) -> set[int]:
        widths: set[int] = set()
        patterns = (
            r"(?:dictionary\s+)?(?:index|pointer)\s*(?:field)?\s*(?:uses?|is|has|:|=)?\s*(\d+)\s*[- ]?bits?",
            r"(\d+)\s*[- ]?bit\s+(?:dictionary\s+)?(?:index|pointer)",
            r"(?:索引|下标)(?:字段)?.{0,12}?(\d+)\s*位",
        )
        for pattern in patterns:
            widths.update(int(value) for value in re.findall(pattern, text, re.IGNORECASE))
        return widths

    def _linear_recurrence_hint(self, problem: str) -> Optional[str]:
        """Solve a first-order affine recurrence only when every coefficient is explicit."""
        match = re.search(
            r"a_n\s*=\s*([+-]?\d*)\s*\*?\s*a_\{?n-1\}?\s*([+-]\s*\d+)?",
            problem,
            re.IGNORECASE,
        )
        initial = re.search(r"a_1\s*=\s*([+-]?\d+(?:/\d+)?)", problem, re.IGNORECASE)
        if not match or not initial or not self.sympy:
            return None
        coefficient_text = match.group(1).replace(" ", "")
        coefficient_text = "1" if coefficient_text in {"", "+"} else ("-1" if coefficient_text == "-" else coefficient_text)
        offset_text = (match.group(2) or "0").replace(" ", "")
        try:
            coefficient = self.sympy.Rational(coefficient_text)
            offset = self.sympy.Rational(offset_text)
            first = self.sympy.Rational(initial.group(1))
            n = self.sympy.Symbol("n", integer=True, positive=True)
            if coefficient == 1:
                expression = first + (n - 1) * offset
            else:
                fixed_point = offset / (1 - coefficient)
                expression = fixed_point + (first - fixed_point) * coefficient ** (n - 1)
            return f"SymPy 递推通项: a_n={self._format(self.sympy.simplify(expression))}"
        except Exception:
            return None

    def _curve_speed_hint(self, problem: str) -> Optional[str]:
        if not self.sympy or not re.search(r"速度长度|弧长参数", problem):
            return None
        match = re.search(
            r"(?:γ|gamma)\s*\(\s*([A-Za-z])\s*\)\s*=\s*\(([^()]+(?:\([^()]*\)[^()]*)*)\)",
            problem,
            re.IGNORECASE,
        )
        if not match:
            return None
        components = [item.strip() for item in match.group(2).split(",")]
        if len(components) not in {2, 3}:
            return None
        try:
            variable = self.sympy.Symbol(match.group(1))
            vector = [self._parse(self._latex_to_sympy(item)) for item in components]
            speed = self.sympy.simplify(self.sympy.sqrt(sum(self.sympy.diff(item, variable) ** 2 for item in vector)))
            judgement = "是弧长参数" if self.sympy.simplify(speed - 1) == 0 else "不是弧长参数"
            return f"SymPy 曲线速度: 速度长度为{self._format(speed)}，{judgement}"
        except Exception:
            return None

    def _first_fundamental_form_hint(self, problem: str) -> Optional[str]:
        if not self.sympy or not re.search(r"第一基本形式.*E\s*[,，]\s*F\s*[,，]\s*G", problem, re.IGNORECASE):
            return None
        match = re.search(
            r"X\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)\s*=\s*\(([^()]+)\)",
            problem,
        )
        if not match:
            return None
        components = [item.strip() for item in match.group(3).split(",")]
        if len(components) != 3:
            return None
        try:
            u, v = self.sympy.Symbol(match.group(1)), self.sympy.Symbol(match.group(2))
            vector = [self._parse(self._latex_to_sympy(item)) for item in components]
            xu = [self.sympy.diff(item, u) for item in vector]
            xv = [self.sympy.diff(item, v) for item in vector]
            e_value = self.sympy.simplify(sum(item * item for item in xu))
            f_value = self.sympy.simplify(sum(left * right for left, right in zip(xu, xv)))
            g_value = self.sympy.simplify(sum(item * item for item in xv))
            return (
                "SymPy 第一基本形式: "
                f"E={self._format(e_value)}，F={self._format(f_value)}，G={self._format(g_value)}"
            )
        except Exception:
            return None

    @staticmethod
    def _graph_gaussian_curvature_hint(problem: str) -> Optional[str]:
        if re.search(r"曲面.*z\s*=\s*f\s*\(\s*x\s*,\s*y\s*\).*∇f\s*=\s*0", problem, re.IGNORECASE) and re.search(
            r"高斯曲率.*Hessian|Hessian.*高斯曲率", problem, re.IGNORECASE
        ):
            return "本地高斯曲率公式: K=f_{xx}f_{yy}-f_{xy}^2"
        return None

    def _pde_verification_hint(self, problem: str) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            x, y, t = self.sympy.symbols("x y t")
            if re.search(r"热方程.*u_t\s*=\s*u_\{?xx\}?", problem, re.IGNORECASE):
                match = re.search(r"u\s*\(\s*x\s*,\s*t\s*\)\s*=\s*(.+?)(?=是否|，|。|；|;|$)", problem)
                if match:
                    expression = self._parse(self._latex_to_sympy(match.group(1)))
                    time_derivative = self.sympy.simplify(self.sympy.diff(expression, t))
                    space_derivative = self.sympy.simplify(self.sympy.diff(expression, x, 2))
                    judgement = "是解" if self.sympy.simplify(time_derivative - space_derivative) == 0 else "不是解"
                    return (
                        "SymPy PDE核验: "
                        f"u_t={self._format(time_derivative)}，u_{{xx}}={self._format(space_derivative)}，{judgement}"
                    )
            if re.search(r"拉普拉斯方程|u_\{?xx\}?\s*\+\s*u_\{?yy\}?\s*=\s*0", problem, re.IGNORECASE):
                match = re.search(r"函数\s*u\s*=\s*(.+?)(?=是否|调和|，|。|；|;|$)", problem)
                if match:
                    expression = self._parse(self._latex_to_sympy(match.group(1)))
                    u_xx = self.sympy.simplify(self.sympy.diff(expression, x, 2))
                    u_yy = self.sympy.simplify(self.sympy.diff(expression, y, 2))
                    total = self.sympy.simplify(u_xx + u_yy)
                    judgement = "是调和函数" if total == 0 else "不是调和函数"
                    return (
                        "SymPy PDE核验: "
                        f"u_{{xx}}={self._format(u_xx)}，u_{{yy}}={self._format(u_yy)}，二者之和为{self._format(total)}，{judgement}"
                    )
        except Exception:
            return None
        return None

    @staticmethod
    def _plain_equations(problem: str) -> list[str]:
        """Extract only short, ASCII-style equations outside LaTex delimiters."""
        if "$" in problem or not re.search(r"方程|求解|equation|solve|roots?|zeros?", problem, re.IGNORECASE):
            return []
        matches = re.findall(
            r"([0-9xyzXYZ(][0-9A-Za-z_+\-*/^().,\s]{0,120}=[0-9A-Za-z_+\-*/^().,\s]{1,120})",
            problem,
        )
        return [match.strip() for match in matches]

    @staticmethod
    def _congruence_hint(problem: str) -> Optional[str]:
        normalized = (
            problem.replace(r"\equiv", "≡")
            .replace(r"\pmod{", " mod ")
            .replace(r"\pmod", " mod ")
            .replace("}", "")
        )
        match = re.search(
            r"(-?\d+)\s*\*?\s*x\s*≡\s*(-?\d+)\s*(?:mod\s*|\b)(\d+)",
            normalized,
        )
        if not match:
            return None
        coefficient, constant, modulus = map(int, match.groups())
        divisor = gcd(coefficient, modulus)
        if constant % divisor:
            return "本地同余方程：无解"
        if divisor != 1:
            return None
        solution = (pow(coefficient % modulus, -1, modulus) * constant) % modulus
        return f"本地同余方程解: x={solution} (mod {modulus})"

    @staticmethod
    def _modular_power_hint(problem: str) -> Optional[str]:
        normalized = problem.replace(r"\bmod", "mod").replace("{", "").replace("}", "")
        match = re.search(r"(-?\d+)\s*\^\s*(\d+)\s*mod\s*(\d+)", normalized)
        if not match:
            return None
        base, exponent, modulus = map(int, match.groups())
        if modulus == 0:
            return None
        return f"本地模幂计算: {pow(base, exponent, modulus)}"

    @staticmethod
    def _raw_latex_parts(problem: str) -> list[str]:
        """Find standalone raw LaTex limits and integrals without `$...$`."""
        return re.findall(r"(\\(?:lim|int)[^。？?\n]+)", problem)

    def _parse(self, expression: str):
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", expression):
            raise ValueError("unsupported symbolic expression")
        identifiers = set(re.findall(r"[A-Za-z]+", expression))
        allowed = {"sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt", "pi", "oo"}
        if any(identifier not in allowed and len(identifier) != 1 for identifier in identifiers):
            raise ValueError("unsupported symbolic identifier")
        return parse_expr(
            expression,
            transformations=standard_transformations + (implicit_multiplication_application,),
        )

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        # English prose extractors may include the sentence-final period.  It
        # would turn an integer exponent such as ``x^3.`` into SymPy's ``3.0``.
        value = expression.strip().replace("$", "").rstrip("。；;，,.!?？")
        value = SympyTool._replace_fractions(value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        function_names = {
            "arctan": "atan",
            "arcsin": "asin",
            "arccos": "acos",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "ln": "log",
            "exp": "exp",
        }
        value = re.sub(
            r"\\(arctan|arcsin|arccos|sin|cos|tan|log|ln|exp)",
            lambda m: f" {function_names[m.group(1)]}",
            value,
        )
        value = (
            value.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\!", "")
            .replace(r"\pi", "pi")
            .replace(r"\infty", "oo")
            .replace(r"\,", "")
        )
        value = value.replace("^", "**").replace("{", "(").replace("}", ")")
        value = re.sub(r"(?<=[xyzXYZ])(?=[xyzXYZ])", "*", value)
        return re.sub(r"(?<![A-Za-z])e(?=\s*\*\*)", "E", value)

    @staticmethod
    def _replace_fractions(value: str) -> str:
        """Convert nested LaTex fractions without relying on a full TeX parser."""
        marker = r"\frac"
        while marker in value:
            start = value.find(marker)
            numerator = SympyTool._braced_group(value, start + len(marker))
            if numerator is None:
                break
            numerator_text, after_numerator = numerator
            denominator = SympyTool._braced_group(value, after_numerator)
            if denominator is None:
                break
            denominator_text, after_denominator = denominator
            replacement = f"({numerator_text})/({denominator_text})"
            value = value[:start] + replacement + value[after_denominator:]
        return value

    @staticmethod
    def _braced_group(value: str, start: int) -> Optional[tuple[str, int]]:
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value) or value[start] != "{":
            return None
        depth = 0
        for index in range(start, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    return value[start + 1:index], index + 1
        return None

    @staticmethod
    def _format(value: Any) -> str:
        text = str(value).replace("**", "^")
        text = re.sub(r"\blog\(", "ln(", text)
        text = re.sub(r"\batan\(", "arctan(", text)
        text = re.sub(r"\basin\(", "arcsin(", text)
        text = re.sub(r"\bacos\(", "arccos(", text)
        text = re.sub(r"\bexp\(x\)", "e^x", text)
        text = re.sub(r"\bexp\(([^()]+)\)", r"e^(\1)", text)
        return re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", text)

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return self._format(operation(self.sympy))
        except Exception:
            return None
