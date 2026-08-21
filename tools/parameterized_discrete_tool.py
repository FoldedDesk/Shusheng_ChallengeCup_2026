"""Fail-closed compilers for explicit parameterized finite computations.

These routes contain no stored problem or answer.  They parse all parameters
from the current statement, reject unsupported residue text, execute a bounded
local operation, and require that operation's independent postcondition.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from reasoning.local_tool_opportunity import (
    LocalToolOpportunityKind,
    detect_local_tool_opportunity,
)
from tools.model_math_tools import ModelMathTools, ModelToolExecution
from tools.tool_contract import (
    CertificateStatus,
    ToolResult,
    make_parameterized_tool_result,
)


class ParameterizedDiscreteTool:
    """Compile a narrow, complete statement into a deterministic operation."""

    _UNSUPPORTED_GAME = re.compile(
        r"两堆|多堆|若干堆|分成|拆分|最后取走者输|反常玩法|"
        r"\b(?:multiple\s+heaps?|two\s+heaps?|split(?:ting)?|mis[eè]re)\b",
        re.IGNORECASE,
    )
    _HEAP_PATTERNS = (
        re.compile(
            r"(?:一|1)\s*堆[^。；;\n]{0,45}?(?P<value>\d{1,6})\s*(?:枚|个)?\s*石子",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:a|one)\s+(?:heap|pile)\s+"
            r"(?:contains?|has|of|with)\s+(?P<value>\d{1,6})\s+stones?\b",
            re.IGNORECASE,
        ),
    )
    _MOVE_PATTERNS = (
        re.compile(
            r"每次[^。；;\n]{0,35}?(?:取走|取|拿走|拿|移走)\s*"
            r"(?P<values>\d{1,6}(?:\s*(?:[、,，]|或|或者)\s*\d{1,6})*)"
            r"\s*(?:枚|个)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:remov(?:e|es|ing)|tak(?:e|es|ing))\s+(?:exactly\s+)?"
            r"(?P<values>\d{1,6}(?:\s*(?:,|or|and)\s*\d{1,6})*)"
            r"\s+stones?\b",
            re.IGNORECASE,
        ),
    )
    _MODULUS = re.compile(
        r"(?:\\pmod\s*\{?\s*|\bmod(?:ulo)?\s+|模\s*)"
        r"[$\\({\[]*\s*(?P<value>\d{1,10})",
        re.IGNORECASE,
    )
    _UNSUPPORTED_CYCLE_CONSTRAINT = re.compile(
        r"至少|至多|不少于|不多于|最多|最少|奇数个循环|偶数个循环|"
        r"循环总数|只能|仅允许|包含指定|经过指定|"
        r"\b(?:at\s+least|at\s+most|no\s+(?:more|fewer)\s+than|"
        r"odd\s+number\s+of\s+cycles?|even\s+number\s+of\s+cycles?|"
        r"total\s+number\s+of\s+cycles?|only\s+(?:cycles?|cycle\s+lengths?)|"
        r"only\b[^.。;；\n]{0,30}\bcycle\s+lengths?|"
        r"containing\s+(?:a\s+)?specified|through\s+(?:a\s+)?specified)\b",
        re.IGNORECASE,
    )
    _NUMBER_WORDS = {
        "零": 0,
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
    }

    def __init__(self, runtime: ModelMathTools | None = None) -> None:
        self.runtime = runtime or ModelMathTools()

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        results: list[ToolResult] = []
        for compiler in (
            self._subtraction_game,
            self._permutation_cycle_inventory,
            self._factorial_valuation,
            self._modular_power_sum,
            self._lattice_polygon,
        ):
            try:
                result = compiler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _subtraction_game(self, text: str) -> Optional[ToolResult]:
        opportunity = detect_local_tool_opportunity(text)
        if (
            opportunity.kind is not LocalToolOpportunityKind.SUBTRACTION_GAME
            or self._UNSUPPORTED_GAME.search(text)
        ):
            return None
        heap_values = [
            int(match.group("value"))
            for pattern in self._HEAP_PATTERNS
            for match in pattern.finditer(text)
        ]
        move_matches = [
            match
            for pattern in self._MOVE_PATTERNS
            for match in pattern.finditer(text)
        ]
        if len(heap_values) != 1 or len(move_matches) != 1:
            return None
        moves = self._plain_integer_list(move_matches[0].group("values"))
        if moves is None or len(moves) != len(set(moves)):
            return None
        execution = self._execute(
            "subtraction_game_outcome",
            {"initial_heap": heap_values[0], "moves": moves},
        )
        if execution is None:
            return None
        payload = json.loads(execution.result)
        winning = bool(payload["winning"])
        winning_moves = [int(value) for value in payload["winning_moves"]]
        chinese = self._is_chinese(text)
        if chinese:
            if winning:
                rendered = "、".join(map(str, winning_moves))
                result = (
                    f"先手必胜；全部必胜第一步为取走 {rendered} 枚石子。"
                )
            else:
                result = "先手必败（后手必胜）。"
            support = (
                "令空堆为必败态，逐个堆大小应用：存在一步到必败态当且仅当当前态必胜；"
                "再用 Sprague-Grundy 递推独立复核。"
            )
        else:
            if winning:
                rendered = ", ".join(map(str, winning_moves))
                result = (
                    "The first player wins; all winning first moves remove "
                    f"{rendered} stone(s)."
                )
            else:
                result = "The first player loses (the second player wins)."
            support = (
                "Starting with the empty heap as losing, each heap was classified "
                "by whether a legal move reaches a losing state; an independent "
                "Sprague-Grundy recurrence gave the same outcome."
            )
        return self._result(
            text,
            operation="parameterized_subtraction_game",
            result=result,
            result_kind="winner_and_moves",
            method="finite_normal_play_dp_with_grundy_crosscheck",
            checks=(
                "single_heap_size_parsed",
                "complete_fixed_move_list_parsed",
                "normal_play_rule_confirmed",
                "unsupported_multiheap_split_and_misere_rules_absent",
                *execution.postconditions,
            ),
            support=support,
            shapes=("truth", "text"),
            requirements=("result_present", "judgement"),
        )

    def _permutation_cycle_inventory(self, text: str) -> Optional[ToolResult]:
        opportunity = detect_local_tool_opportunity(text)
        if (
            opportunity.kind is not LocalToolOpportunityKind.PERMUTATION_CYCLES
            or self._UNSUPPORTED_CYCLE_CONSTRAINT.search(text)
        ):
            return None

        sizes = {
            int(match.group("size"))
            for pattern in (
                r"(?P<size>\d{1,3})\s*个?(?:标号|有标号)?元素的(?:排列|置换)",
                r"\bpermutations?\s+(?:of|on)\s+(?:the\s+)?"
                r"(?P<size>\d{1,3})\s+(?:(?:labelled|labeled)\s+)?"
                r"(?:elements?|symbols?|objects?)\b",
            )
            for match in re.finditer(pattern, text, re.IGNORECASE)
        }
        if len(sizes) != 1:
            return None
        size = next(iter(sizes))

        exact_bounds: dict[int, int] = {}
        exact_matches: list[re.Match[str]] = []
        exact_patterns = (
            re.compile(
                r"(?:恰好|正好)(?:有)?\s*"
                r"(?P<count>\d{1,3}|[零一二两三四五六七八九十])\s*个?"
                r"(?:长度为|长为)\s*(?P<length>\d{1,3})\s*的?"
                r"(?:循环|轮换)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:恰好|正好)(?:有)?\s*"
                r"(?P<count>\d{1,3}|[零一二两三四五六七八九十])\s*个?"
                r"(?P<length>\d{1,3})\s*[-－—]?\s*(?:循环|轮换)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bexactly\s+"
                r"(?P<count>\d{1,3}|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s+cycles?\s+of\s+length\s+(?P<length>\d{1,3})\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bexactly\s+"
                r"(?P<count>\d{1,3}|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s+(?P<length>\d{1,3})\s*[- ]\s*cycles?\b",
                re.IGNORECASE,
            ),
        )
        for pattern in exact_patterns:
            for match in pattern.finditer(text):
                count = self._small_number(match.group("count"))
                raw_length = match.groupdict().get("length")
                if count is None or raw_length is None:
                    return None
                length = int(raw_length)
                if length in exact_bounds and exact_bounds[length] != count:
                    return None
                exact_bounds[length] = count
                exact_matches.append(match)
        if not exact_matches:
            return None

        forbidden: set[int] = set()
        fixed_point_matches = list(re.finditer(
            r"没有不动点|无不动点|不存在不动点|"
            r"\b(?:no|without)\s+fixed\s+points?\b",
            text,
            re.IGNORECASE,
        ))
        if fixed_point_matches:
            forbidden.add(1)
        forbidden_matches = list(re.finditer(
            r"(?:没有|无|不存在)\s*(?P<zh_length>\d{1,3})\s*[-－—]?\s*(?:循环|轮换)|"
            r"\b(?:no|without)\s+(?P<en_length>\d{1,3})\s*[- ]\s*cycles?\b",
            text,
            re.IGNORECASE,
        ))
        for match in forbidden_matches:
            forbidden.add(int(match.group("zh_length") or match.group("en_length")))

        # Every explicit exact/absence cycle restriction must have been parsed.
        exact_markers = len(re.findall(
            r"(?:恰好|正好)[^。；;\n]{0,45}(?:循环|轮换)|"
            r"\bexactly\b[^.。;；\n]{0,45}\bcycles?\b",
            text,
            re.IGNORECASE,
        ))
        absence_markers = len(re.findall(
            r"(?:没有|无|不存在)[^。；;\n]{0,25}(?:不动点|循环|轮换)|"
            r"\b(?:no|without)\b[^.。;；\n]{0,25}(?:fixed\s+points?|cycles?)\b",
            text,
            re.IGNORECASE,
        ))
        if exact_markers != len(exact_matches):
            return None
        if absence_markers != len(fixed_point_matches) + len(forbidden_matches):
            return None
        if any(
            length < 1
            or length > size
            or count < 0
            or count > size // length
            or (length in forbidden and count != 0)
            for length, count in exact_bounds.items()
        ):
            return None
        if any(length < 1 or length > size for length in forbidden):
            return None

        allowed_lengths = [
            length for length in range(1, size + 1) if length not in forbidden
        ]
        if not allowed_lengths:
            return None
        cycle_count_bounds = [
            {"length": length, "minimum": count, "maximum": count}
            for length, count in sorted(exact_bounds.items())
        ]
        execution = self._execute(
            "permutation_cycle_count",
            {
                "size": size,
                "allowed_cycle_lengths": allowed_lengths,
                "cycle_count_bounds": cycle_count_bounds,
            },
        )
        if execution is None:
            return None
        result = execution.result
        chinese = self._is_chinese(text)
        support = (
            "按标号置换的循环指数型生成函数，将禁用循环长度删去，并将每个指定"
            f"长度的循环个数固定后取 {size} 次项，得到 {result}；逆序卷积独立复核。"
            if chinese
            else
            "Using the labelled-permutation cycle exponential generating function, "
            "the forbidden lengths are removed and each stated cycle count is fixed; "
            f"the degree-{size} coefficient gives {result}, independently checked "
            "by reversed convolution."
        )
        return self._result(
            text,
            operation="parameterized_permutation_cycle_inventory",
            result=result,
            result_kind="permutation_count",
            method="cycle_egf_coefficient_with_reversed_convolution",
            checks=(
                "single_labelled_permutation_size_parsed",
                "all_exact_cycle_count_constraints_parsed",
                "all_cycle_absence_constraints_parsed",
                "unsupported_cycle_constraints_absent",
                *execution.postconditions,
            ),
            support=support,
            shapes=("count", "number", "expression"),
            requirements=("result_present",),
        )

    def _factorial_valuation(self, text: str) -> Optional[ToolResult]:
        opportunity = detect_local_tool_opportunity(text)
        if opportunity.kind is not LocalToolOpportunityKind.FACTORIAL_VALUATION:
            return None
        prime_values = {
            int(value)
            for pattern in (
                r"(?P<prime>\d{1,7})\s*[- ]?adic",
                r"v\s*_\s*\{?\s*(?P<prime>\d{1,7})\s*\}?",
                r"素因子\s*(?P<prime>\d{1,7})\s*的(?:指数|幂次)",
            )
            for value in (
                match.group("prime")
                for match in re.finditer(pattern, text, re.IGNORECASE)
            )
        }
        fraction = self._latex_fraction_containing_factorials(text)
        if len(prime_values) != 1 or fraction is None:
            return None
        numerator_source, denominator_source = fraction
        numerator = self._factorial_product(numerator_source)
        denominator = self._factorial_product(denominator_source)
        if numerator is None or denominator is None:
            return None
        if len(re.findall(r"\d+\s*!", text)) != len(numerator) + len(denominator):
            return None
        prime = next(iter(prime_values))
        execution = self._execute(
            "factorial_ratio_prime_valuation",
            {
                "prime": prime,
                "numerator_factorials": numerator,
                "denominator_factorials": denominator,
            },
        )
        if execution is None:
            return None
        result = execution.result
        chinese = self._is_chinese(text)
        support = (
            rf"由 Legendre 公式 $v_{{{prime}}}(n!)=\sum_{{j\ge1}}"
            rf"\lfloor n/{prime}^j\rfloor$，分子各项减去分母各项得 {result}；"
            "并以 p 进制数位和公式独立复核。"
            if chinese
            else
            rf"Legendre's formula $v_{{{prime}}}(n!)=\sum_{{j\ge1}}"
            rf"\lfloor n/{prime}^j\rfloor$ gives {result} after subtracting "
            "the denominator terms; the base-p digit-sum formula independently agrees."
        )
        return self._result(
            text,
            operation="parameterized_factorial_ratio_valuation",
            result=result,
            result_kind="integer_valuation",
            method="legendre_formula_and_digit_sum_identity",
            checks=(
                "single_prime_parsed",
                "complete_factorial_numerator_parsed",
                "complete_factorial_denominator_parsed",
                "no_unassigned_factorial_term",
                *execution.postconditions,
            ),
            support=support,
            shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _modular_power_sum(self, text: str) -> Optional[ToolResult]:
        opportunity = detect_local_tool_opportunity(text)
        if opportunity.kind is not LocalToolOpportunityKind.MODULAR_POWER:
            return None
        moduli = {int(match.group("value")) for match in self._MODULUS.finditer(text)}
        if len(moduli) != 1:
            return None
        expression = self._single_power_math_segment(text)
        if expression is None:
            return None
        terms = self._compile_power_terms(expression)
        if terms is None:
            return None
        modulus = next(iter(moduli))
        execution = self._execute(
            "modular_power_sum",
            {"terms": terms, "modulus": modulus},
        )
        if execution is None:
            return None
        result = execution.result
        chinese = self._is_chinese(text)
        support = (
            f"逐项用平方-乘算法在模 {modulus} 下计算幂并相加，所得标准余数为 {result}；"
            "逆序重算各项余数结果一致。"
            if chinese
            else
            f"Each power was evaluated modulo {modulus} by exact square-and-multiply "
            f"and the standard residue is {result}; recomputing the terms in reverse "
            "order gave the same residue."
        )
        return self._result(
            text,
            operation="parameterized_modular_power_sum",
            result=result,
            result_kind="modular_residue",
            method="bounded_power_ast_modular_exponentiation",
            checks=(
                "single_numeric_modulus_parsed",
                "complete_power_sum_expression_consumed",
                "nonnegative_exponents_confirmed",
                *execution.postconditions,
            ),
            support=support,
            shapes=("number", "expression"),
            requirements=("result_present", "numeric_result"),
        )

    def _lattice_polygon(self, text: str) -> Optional[ToolResult]:
        opportunity = detect_local_tool_opportunity(text)
        if opportunity.kind is not LocalToolOpportunityKind.LATTICE_POLYGON:
            return None
        clause = re.search(
            r"(?:顶点[^。；;\n]{0,40}?(?:为|是)|"
            r"\b(?:has|with)\s+vertices?\b|"
            r"\bvertices?\b[^.；;\n]{0,40}?(?:are|:))"
            r"(?P<vertices>[^。；;\n]+)",
            text,
            re.IGNORECASE,
        )
        if clause is None:
            return None
        vertices = [
            [int(first), int(second)]
            for first, second in re.findall(
                r"\(\s*([-+]?\d{1,7})\s*[,，]\s*([-+]?\d{1,7})\s*\)",
                clause.group("vertices"),
            )
        ]
        all_vertices = re.findall(
            r"\(\s*[-+]?\d{1,7}\s*[,，]\s*[-+]?\d{1,7}\s*\)",
            text,
        )
        if len(vertices) < 3 or len(vertices) != len(all_vertices):
            return None
        execution = self._execute(
            "lattice_polygon_interior",
            {"vertices": vertices},
        )
        if execution is None:
            return None
        payload = json.loads(execution.result)
        result = str(payload["interior_points"])
        area_twice = int(payload["area_twice"])
        boundary = int(payload["boundary_points"])
        chinese = self._is_chinese(text)
        support = (
            f"鞋带公式给出两倍面积 {area_twice}，各边坐标差的最大公约数之和给出"
            f"边界格点数 {boundary}。Pick 定理于是给出内部格点数 {result}。"
            if chinese
            else
            f"The shoelace sum gives twice the area {area_twice}, and the sum of "
            f"edge-coordinate gcds gives {boundary} boundary lattice points. "
            f"Pick's theorem therefore gives {result} interior lattice points."
        )
        return self._result(
            text,
            operation="parameterized_lattice_polygon_interior",
            result=result,
            result_kind="lattice_point_count",
            method="shoelace_boundary_gcd_pick_theorem",
            checks=(
                "complete_ordered_vertex_clause_parsed",
                "all_coordinates_integral",
                "simple_polygon_intersection_guard_passed",
                *execution.postconditions,
            ),
            support=support,
            shapes=("number", "expression", "count"),
            requirements=("result_present",),
        )

    def _execute(self, name: str, arguments: dict[str, Any]) -> ModelToolExecution | None:
        execution = self.runtime.execute_call({
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
        if (
            not execution.ok
            or execution.local_certificate_status is not CertificateStatus.CERTIFIED_TRUE
        ):
            return None
        return execution

    @staticmethod
    def _plain_integer_list(value: str) -> list[int] | None:
        text = str(value or "")
        numbers = [int(item) for item in re.findall(r"\d+", text)]
        residue = re.sub(r"\d+", "", text)
        residue = re.sub(r"[\s、,，]|或|或者|\band\b|\bor\b", "", residue, flags=re.IGNORECASE)
        return numbers if numbers and not residue else None

    @classmethod
    def _small_number(cls, value: str) -> int | None:
        normalized = str(value or "").strip().casefold()
        if normalized.isdigit():
            return int(normalized)
        return cls._NUMBER_WORDS.get(normalized)

    @classmethod
    def _latex_fraction_containing_factorials(
        cls,
        text: str,
    ) -> tuple[str, str] | None:
        candidates: list[tuple[str, str]] = []
        start = 0
        while True:
            position = text.find(r"\frac", start)
            if position < 0:
                break
            cursor = position + len(r"\frac")
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                start = cursor
                continue
            numerator_end = cls._matching_brace(text, cursor)
            if numerator_end < 0:
                return None
            denominator_start = numerator_end + 1
            while denominator_start < len(text) and text[denominator_start].isspace():
                denominator_start += 1
            if denominator_start >= len(text) or text[denominator_start] != "{":
                start = numerator_end + 1
                continue
            denominator_end = cls._matching_brace(text, denominator_start)
            if denominator_end < 0:
                return None
            numerator = text[cursor + 1:numerator_end]
            denominator = text[denominator_start + 1:denominator_end]
            if "!" in numerator or "!" in denominator:
                candidates.append((numerator, denominator))
            start = denominator_end + 1
        if len(candidates) == 1:
            return candidates[0]
        plain = re.search(
            r"(?P<numerator>(?:\d+\s*!\s*(?:[·*]\s*)?)+)\s*/\s*"
            r"\(?(?P<denominator>(?:\d+\s*!\s*(?:[·*]\s*)?)+)\)?",
            text,
        )
        return (
            (plain.group("numerator"), plain.group("denominator"))
            if not candidates and plain
            else None
        )

    @staticmethod
    def _matching_brace(text: str, start: int) -> int:
        if not (0 <= start < len(text)) or text[start] != "{":
            return -1
        depth = 0
        index = start
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        return -1

    @staticmethod
    def _factorial_product(value: str) -> list[int] | None:
        text = str(value or "")
        values = [int(item) for item in re.findall(r"(\d+)\s*!", text)]
        residue = re.sub(r"\d+\s*!", "", text)
        residue = re.sub(r"\\cdot|\\times|[·*()\s]", "", residue)
        return values if values and not residue else None

    @staticmethod
    def _single_power_math_segment(text: str) -> str | None:
        segments = [
            match.group(1).strip()
            for match in re.finditer(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", text)
            if "^" in match.group(1)
        ]
        if len(segments) == 1:
            return segments[0]
        if segments:
            return None
        capture = re.search(
            r"(?:remainder\s+of|compute|evaluate|计算|求)\s*"
            r"(?P<expression>.+?)\s*(?:modulo|mod|模)\s*[$\\({\[]*\s*\d+",
            text,
            re.IGNORECASE,
        )
        return capture.group("expression").strip() if capture else None

    @classmethod
    def _compile_power_terms(cls, value: str) -> list[dict[str, int]] | None:
        text = str(value or "").replace(r"\cdot", "*").replace(r"\times", "*")
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = re.sub(r"\s+", "", text).replace("{", "(").replace("}", ")")
        signed_terms = cls._split_top_level_terms(text)
        if not signed_terms:
            return None
        result: list[dict[str, int]] = []
        for sign, term in signed_terms:
            parsed = cls._compile_power_term(term)
            if parsed is None:
                return None
            parsed["coefficient"] *= sign
            result.append(parsed)
        return result

    @classmethod
    def _compile_power_term(cls, value: str) -> dict[str, int] | None:
        text = cls._strip_outer_parentheses(value)
        power_position = cls._top_level_operator(text, "^")
        if power_position < 0:
            return (
                {
                    "coefficient": int(text),
                    "base": 1,
                    "exponent_base": 1,
                    "exponent_power": 0,
                    "exponent_multiplier": 0,
                    "exponent_offset": 0,
                }
                if re.fullmatch(r"\d+", text)
                else None
            )
        left = text[:power_position]
        exponent_source = text[power_position + 1:]
        factors = left.split("*")
        if len(factors) == 1:
            coefficient_source, base_source = "1", factors[0]
        elif len(factors) == 2:
            coefficient_source, base_source = factors
        else:
            return None
        if not re.fullmatch(r"\d+", coefficient_source) or not re.fullmatch(
            r"\d+", base_source
        ):
            return None
        exponent = cls._compile_exponent(exponent_source)
        if exponent is None:
            return None
        return {
            "coefficient": int(coefficient_source),
            "base": int(base_source),
            **exponent,
        }

    @classmethod
    def _compile_exponent(cls, value: str) -> dict[str, int] | None:
        text = cls._strip_outer_parentheses(value)
        if re.fullmatch(r"\d+", text):
            return {
                "exponent_base": 1,
                "exponent_power": 0,
                "exponent_multiplier": 0,
                "exponent_offset": int(text),
            }
        position = cls._top_level_operator(text, "^")
        if position < 0:
            return None
        base = text[:position]
        power = cls._strip_outer_parentheses(text[position + 1:])
        if not re.fullmatch(r"\d+", base) or not re.fullmatch(r"\d+", power):
            return None
        return {
            "exponent_base": int(base),
            "exponent_power": int(power),
            "exponent_multiplier": 1,
            "exponent_offset": 0,
        }

    @staticmethod
    def _split_top_level_terms(value: str) -> list[tuple[int, str]] | None:
        if not value:
            return None
        terms: list[tuple[int, str]] = []
        depth = 0
        start = 0
        sign = 1
        if value[0] in "+-":
            sign = -1 if value[0] == "-" else 1
            start = 1
        for index in range(start, len(value)):
            character = value[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return None
            elif depth == 0 and character in "+-":
                term = value[start:index]
                if not term:
                    return None
                terms.append((sign, term))
                sign = -1 if character == "-" else 1
                start = index + 1
        if depth or not value[start:]:
            return None
        terms.append((sign, value[start:]))
        return terms

    @staticmethod
    def _strip_outer_parentheses(value: str) -> str:
        text = str(value or "")
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            wraps = True
            for index, character in enumerate(text):
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0 and index != len(text) - 1:
                        wraps = False
                        break
            if not wraps or depth:
                break
            text = text[1:-1]
        return text

    @staticmethod
    def _top_level_operator(value: str, operator: str) -> int:
        depth = 0
        for index, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    return -1
            elif character == operator and depth == 0:
                return index
        return -1

    @staticmethod
    def _is_chinese(value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))

    @staticmethod
    def _result(
        problem: str,
        *,
        operation: str,
        result: str,
        result_kind: str,
        method: str,
        checks: tuple[str, ...],
        support: str,
        shapes: tuple[str, ...],
        requirements: tuple[str, ...],
    ) -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=support,
            answer_shapes=shapes,
            requirements=requirements,
        )
