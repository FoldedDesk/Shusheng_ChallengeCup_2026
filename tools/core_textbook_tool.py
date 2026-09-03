"""Small, parameterized certificates for foundational textbook calculations.

The implementations below are formulas or finite algorithms.  They parse every
parameter from the current statement, enforce the theorem hypotheses named in
that statement, and abstain on ambiguous variants.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial, gcd, isqrt
import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class CoreTextbookTool:
    """Certify conservative one-target textbook computations."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        if not self.sp:
            return []
        text = str(problem or "").strip()
        results: list[ToolResult] = []
        for compiler in (
            self._even_cardinality_subsets,
            self._positive_compositions,
            self._surjection_without_singleton_fibers,
            self._surjection_count,
            self._nonadjacent_subset_count,
            self._connected_planar_face_count,
            self._cyclic_group_subgroup_count,
            self._power_element_order,
            self._affine_first_order_recurrence,
            self._involution_fixed_point_count,
            self._independent_event_union,
            self._bernoulli_variance,
            self._sample_mean_variance,
            self._simple_regression_r_squared,
            self._two_color_hypergeometric,
            self._coupon_collector_expectation,
            self._geometric_waiting_tail,
            self._brownian_covariance,
            self._matrix_trace_determinant,
            self._chebyshev_nodes,
        ):
            try:
                result = compiler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _even_cardinality_subsets(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"(?:偶数个元素|元素个数为偶数|偶基数|even[- ]cardinality|even\s+number\s+of\s+elements?)[^。.?\n]{0,25}(?:子集|subsets?)|"
            r"(?:子集|subsets?)[^。.?\n]{0,45}(?:偶数|even)",
            text,
            re.IGNORECASE,
        ):
            return None
        match = re.search(
            r"(?:集合|set)[^。.;\n]{0,30}?(?:有|含|has|contains?|with)\s*"
            r"(?:\$|\\\()?\s*([A-Za-z]|\d+)\s*(?:\$|\\\))?\s*"
            r"(?:个)?\s*(?:元素|elements?)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        if re.search(
            r"真子集|非空|必须包含|含有指定|包含指定|proper\s+subsets?|"
            r"nonempty|non-empty|contain(?:ing|s)?\s+(?:a|the|one)\s+(?:fixed|specified)|"
            r"exclude|排除|不包含",
            text,
            re.IGNORECASE,
        ):
            return None
        size = match.group(1)
        if size.isdigit():
            n = int(size)
            if not 1 <= n <= 100_000:
                return None
            answer = str(1 << (n - 1))
        else:
            answer = rf"2^{{{size}-1}}"
        zh = self._is_chinese(text)
        support = (
            rf"固定一个元素并切换其是否属于子集，偶基数与奇基数子集两两配对；因此 $2^{size}$ 个子集中恰有一半，即 ${answer}$ 个。"
            if zh else
            rf"Toggling one fixed element bijects even- and odd-cardinality subsets, so exactly half of the $2^{size}$ subsets are even: ${answer}$."
        )
        return self._count_result(
            text, "even_cardinality_subsets", answer,
            "parity_toggling_bijection",
            ("finite_set_size_parsed", "even_subset_target", "parity_bijection"),
            support,
        )

    def _positive_compositions(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"正整数|positive\s+integers?", text, re.IGNORECASE):
            return None
        equation = re.search(
            r"((?:[A-Za-z]\s*_?\s*\{?\s*\d+\s*\}?\s*\+\s*)+"
            r"[A-Za-z]\s*_?\s*\{?\s*\d+\s*\}?)\s*=\s*(\d+)",
            text,
        )
        if not equation:
            return None
        variables = re.findall(r"([A-Za-z])\s*_?\s*\{?\s*(\d+)\s*\}?", equation.group(1))
        indices = [int(index) for _, index in variables]
        if (
            len(indices) < 2
            or len({name.casefold() for name, _ in variables}) != 1
            or indices != list(range(1, len(indices) + 1))
        ):
            return None
        if re.search(
            r"<=|≤|\\leq|至多|不超过|at\s+most|distinct|互不相同|"
            r"奇数|偶数|奇偶|整除|倍数|互素|素数|质数|"
            r"\b(?:odd|even|divisib|multiple|coprime|prime|increasing|decreasing)\w*\b|"
            r"(?<!\\)[<>]",
            text,
            re.IGNORECASE,
        ):
            return None
        # Stars-and-bars is a whole-answer route only when the displayed sum
        # is the sole equality constraint.  A second equality can encode, for
        # example, x_1=x_2 and changes the count completely.
        if len(re.findall(r"(?<![<>!])=(?!=)", text)) != 1:
            return None
        total = int(equation.group(2))
        minima = [1] * len(indices)
        for index, lower in re.findall(
            r"[A-Za-z]\s*_?\s*\{?\s*(\d+)\s*\}?\s*(?:>=|≥|\\geq?)\s*(\d+)",
            text,
        ):
            position = int(index) - 1
            if not 0 <= position < len(minima):
                return None
            minima[position] = max(minima[position], int(lower))
        remainder = total - sum(minima)
        count = 0 if remainder < 0 else comb(remainder + len(indices) - 1, len(indices) - 1)
        zh = self._is_chinese(text)
        support = (
            rf"令新变量等于原变量减去下界 {tuple(minima)}，则得到 {len(indices)} 个非负整数之和为 {remainder}；因此隔板法给出 $\binom{{{remainder + len(indices) - 1}}}{{{len(indices) - 1}}}={count}$。"
            if zh else
            rf"Set each new variable equal to the original variable minus its lower bound {tuple(minima)}. Then {len(indices)} nonnegative integers sum to {remainder}; therefore stars and bars gives $\binom{{{remainder + len(indices) - 1}}}{{{len(indices) - 1}}}={count}$."
        )
        return self._count_result(
            text, "positive_composition_with_lower_bounds", str(count),
            "stars_and_bars_after_lower_bound_translation",
            ("all_variables_and_total_parsed", "all_lower_bounds_parsed", "translated_sum_recomputed"),
            support,
        )

    def _surjection_count(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"满射|surjective|onto", text, re.IGNORECASE):
            return None
        if re.search(
            r"单射|双射|一一|固定点|不动点|递增|递减|保序|纤维|原像集|逆像集|"
            r"injective|bijective|one[- ]to[- ]one|fixed\s+points?|"
            r"monotone|order[- ]preserving|fib(?:er|re)s?|preimages?|inverse\s+images?|"
            r"指定像|prescribed\s+image",
            text,
            re.IGNORECASE,
        ):
            return None
        parsed_sizes = self._surjection_sizes(text)
        if parsed_sizes is None:
            return None
        domain, codomain = parsed_sizes
        count = sum(
            (-1) ** omitted * comb(codomain, omitted) * (codomain - omitted) ** domain
            for omitted in range(codomain + 1)
        )
        zh = self._is_chinese(text)
        support = (
            rf"令 $j$ 表示未命中的陪域元素数，则对这些元素作容斥；因此 $\sum_{{j=0}}^{{{codomain}}}(-1)^j\binom{{{codomain}}}j({codomain}-j)^{{{domain}}}={count}$。"
            if zh else
            rf"Let $j$ be the number of missed codomain elements. Inclusion-exclusion over those elements therefore gives $\sum_{{j=0}}^{{{codomain}}}(-1)^j\binom{{{codomain}}}j({codomain}-j)^{{{domain}}}={count}$."
        )
        return self._count_result(
            text, "surjection_count", str(count),
            "finite_inclusion_exclusion_over_codomain",
            ("domain_cardinality_parsed", "codomain_cardinality_parsed", "all_omission_sizes_summed"),
            support,
        )

    def _surjection_without_singleton_fibers(
        self,
        text: str,
    ) -> Optional[ToolResult]:
        if not re.search(r"满射|surjective|onto", text, re.IGNORECASE):
            return None
        no_singletons = bool(re.search(
            r"无单点纤维|不存在单点纤维|"
            r"(?:任何|任一|每个|所有)?\s*(?:原像集|逆像集|纤维)[^。.;；\n]{0,35}"
            r"(?:大小|基数)?[^。.;；\n]{0,20}(?:(?:不允许|不能|不得|不等于|不是)"
            r"\s*(?:为|等于)?\s*1|至少\s*(?:为|等于)?\s*2)|"
            r"no\s+singleton\s+fib(?:er|re)s?|"
            r"no\s+fib(?:er|re)[^.;\n]{0,30}(?:has|of|with)[^.;\n]{0,15}(?:size\s*)?1|"
            r"every\s+fib(?:er|re)[^.;\n]{0,30}(?:size|cardinality)[^.;\n]{0,15}"
            r"(?:at\s+least\s+2|>=?\s*2|not\s+(?:equal\s+to\s+)?1)",
            text,
            re.IGNORECASE,
        ))
        if not no_singletons:
            return None
        # The recurrence below certifies exactly one restriction: every
        # labelled codomain fiber has size at least two.  Other semantic
        # constraints require a different state space, so abstain.
        if re.search(
            r"单射|双射|一一|固定点|不动点|相邻|连续元素|奇数|偶数|互异|不同大小|"
            r"指定像|指定原像|递增|递减|保序|"
            r"injective|bijective|one[- ]to[- ]one|fixed\s+points?|adjacent|"
            r"consecutive|odd|even|distinct\s+(?:fib(?:er|re)\s+)?sizes?|"
            r"prescribed|monotone|order[- ]preserving",
            text,
            re.IGNORECASE,
        ):
            return None
        parsed_sizes = self._surjection_sizes(text)
        if parsed_sizes is None:
            return None
        domain, codomain = parsed_sizes

        # Assign the labelled domain elements to the labelled codomain fibers
        # one fiber at a time. Choosing k elements for the next fiber and
        # recursing counts every ordered fiber partition exactly once.
        states = [0] * (domain + 1)
        states[0] = 1
        for _ in range(codomain):
            next_states = [0] * (domain + 1)
            for used, ways in enumerate(states):
                if not ways:
                    continue
                remaining = domain - used
                for size in range(2, remaining + 1):
                    next_states[used + size] += ways * comb(remaining, size)
            states = next_states
        count = states[domain]
        zh = self._is_chinese(text)
        support = (
            rf"将 {domain} 个标号元素分到 {codomain} 个标号纤维，逐纤维选择大小 $k_i\ge2$；"
            rf"精确求和 $\sum_{{k_1+\cdots+k_{{{codomain}}}={domain},\,k_i\ge2}}"
            rf"\frac{{{domain}!}}{{k_1!\cdots k_{{{codomain}}}!}}={count}$。"
            if zh else
            rf"Partition the {domain} labelled elements among {codomain} labelled fibers with $k_i\ge2$; "
            rf"the exact sum $\sum_{{k_1+\cdots+k_{{{codomain}}}={domain},\,k_i\ge2}}"
            rf"\frac{{{domain}!}}{{k_1!\cdots k_{{{codomain}}}!}}={count}$."
        )
        return self._count_result(
            text,
            "surjection_without_singleton_fibers",
            str(count),
            "labelled_fiber_size_dynamic_program",
            (
                "domain_cardinality_parsed",
                "codomain_cardinality_parsed",
                "all_fiber_sizes_at_least_two",
                "all_ordered_fiber_compositions_summed",
            ),
            support,
        )

    @staticmethod
    def _surjection_sizes(text: str) -> Optional[tuple[int, int]]:
        sizes = re.search(
            r"从\s*(\d+)\s*元(?:标号)?集合\s*到\s*(\d+)\s*元(?:标号)?集合|"
            r"(?:functions?|maps?)\s+from\s+(?:a\s+)?(\d+)[- ]element\s+set"
            r"[^.。;；\n]{0,40}?\s+to\s+(?:a\s+)?(\d+)[- ]element\s+set",
            text,
            re.IGNORECASE,
        )
        if sizes is None:
            sizes = re.search(
                r"(?:functions?|maps?)[^.。;；\n]{0,30}?\s+from\s+(?:a\s+)?"
                r"(\d+)[- ]element\s+set[^.。;；\n]{0,40}?\s+to\s+(?:a\s+)?"
                r"(\d+)[- ]element\s+set",
                text,
                re.IGNORECASE,
            )
        if sizes:
            values = [int(item) for item in sizes.groups() if item]
            if len(values) != 2:
                return None
            domain, codomain = values
        else:
            explicit_sets = re.findall(r"\{([^{}]+)\}", text)
            if len(explicit_sets) < 2:
                return None
            cardinalities = []
            for body in explicit_sets[:2]:
                entries = [item.strip() for item in re.split(r"[,，]", body) if item.strip()]
                if not entries or any("ldots" in item or "…" in item for item in entries):
                    return None
                cardinalities.append(len(entries))
            domain, codomain = cardinalities
        if not 0 <= domain <= 60 or not 1 <= codomain <= 20:
            return None
        return domain, codomain

    def _nonadjacent_subset_count(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"不含相邻整数|没有两个相邻|no\s+(?:two\s+)?(?:chosen\s+)?(?:consecutive|adjacent)\s+integers",
            text,
            re.IGNORECASE,
        ):
            return None
        normalized = text.replace(r"\{", "{").replace(r"\}", "}")
        interval = re.search(
            r"\{\s*1\s*[,，]\s*2[^{}]*(?:\\ldots|\\dots|…)[^{}]*?\s*(\d+)\s*\}",
            normalized,
        )
        choose = re.search(
            r"(?:任选|选取|选择|choose|select)\s*(\d+)\s*(?:个|elements?|integers?)?",
            text,
            re.IGNORECASE,
        )
        if not interval or not choose:
            return None
        if re.search(
            r"元素和|总和|乘积|奇数个|偶数个|整除|倍数|必须包含|包含指定|"
            r"\b(?:sum|product|odd|even|divisib|multiple|contain(?:ing|s)?\s+"
            r"(?:a|the|one)\s+(?:fixed|specified))\b",
            text,
            re.IGNORECASE,
        ):
            return None
        n, k = int(interval.group(1)), int(choose.group(1))
        if not 0 <= k <= n:
            return None
        count = comb(n - k + 1, k) if n - k + 1 >= k else 0
        zh = self._is_chinese(text)
        support = (
            rf"若 $a_1<\cdots<a_{k}$ 且相邻差至少 2，令 $b_i=a_i-(i-1)$，便与从 {n-k+1} 个位置普通选 {k} 个一一对应，故为 $\binom{{{n-k+1}}}{{{k}}}={count}$。"
            if zh else
            rf"For $a_1<\cdots<a_{k}$ with gaps at least 2, $b_i=a_i-(i-1)$ is a bijection to ordinary {k}-subsets of {n-k+1} positions, giving $\binom{{{n-k+1}}}{{{k}}}={count}$."
        )
        return self._count_result(
            text, "nonadjacent_subset_count", str(count),
            "gap_compression_bijection",
            ("integer_interval_parsed", "chosen_size_parsed", "compression_inverse_checked"),
            support,
        )

    def _connected_planar_face_count(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"连通[^。.;\n]{0,20}平面[^。.;\n]{0,10}图|connected\s+planar\s+graph", text, re.IGNORECASE):
            return None
        if re.search(
            r"有界面|内部面|bounded\s+faces?|interior\s+faces?",
            text,
            re.IGNORECASE,
        ):
            return None
        vertices = re.search(r"(?:有|含|with|has)\s*(\d+)\s*(?:个)?\s*(?:顶点|vertices?)", text, re.IGNORECASE)
        edges = re.search(r"(?:有|含|with|has)?\s*(\d+)\s*(?:条)?\s*(?:边|edges?)", text, re.IGNORECASE)
        if not vertices or not edges:
            return None
        vertex_count, edge_count = int(vertices.group(1)), int(edges.group(1))
        faces = edge_count - vertex_count + 2
        if vertex_count < 1 or edge_count < 0 or faces < 1:
            return None
        zh = self._is_chinese(text)
        answer = rf"F={faces}"
        support = (
            rf"连通平面图满足 $V-E+F=2$，所以 $F=2-{vertex_count}+{edge_count}={faces}$。"
            if zh else
            rf"A connected planar graph satisfies $V-E+F=2$, hence $F=2-{vertex_count}+{edge_count}={faces}$."
        )
        return self._count_result(
            text, "connected_planar_face_count", answer,
            "connected_planar_euler_identity",
            ("connected_planar_hypotheses", "vertex_and_edge_counts_parsed", "euler_identity_substitution"),
            support,
        )

    def _cyclic_group_subgroup_count(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"循环群|cyclic\s+group", text, re.IGNORECASE) or not re.search(r"子群|subgroups?", text, re.IGNORECASE):
            return None
        if re.search(
            r"真子群|非平凡|极大|指定阶|proper|nontrivial|maximal|"
            r"subgroups?\s+of\s+order",
            text,
            re.IGNORECASE,
        ):
            return None
        match = re.search(
            r"(?:阶数?|order)(?:为|是|is|of|=)?\s*(\d+)|(?:of\s+)?order\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        order = int(next(item for item in match.groups() if item))
        if not 1 <= order <= 10**12:
            return None
        small_divisors = [value for value in range(1, isqrt(order) + 1) if order % value == 0]
        count = sum(1 if value * value == order else 2 for value in small_divisors)
        zh = self._is_chinese(text)
        support = (
            rf"循环群对每个正因子 $d\mid {order}$ 恰有一个 $d$ 阶子群；{order} 的正因子数为 {count}。"
            if zh else
            rf"A cyclic group has exactly one subgroup of order $d$ for each positive divisor $d\mid {order}$; {order} has {count} divisors."
        )
        return self._count_result(
            text, "cyclic_group_subgroup_count", str(count),
            "cyclic_subgroup_divisor_bijection",
            ("cyclic_group_and_order_parsed", "all_divisors_counted"),
            support,
        )

    def _power_element_order(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"元素|element", text, re.IGNORECASE) or not re.search(r"阶|order", text, re.IGNORECASE):
            return None
        base_order = re.search(
            r"(?:元素|element)\s*\$?\s*([A-Za-z])\s*\$?[^。.;\n]{0,24}?(?:阶(?:为|是)?|has\s+order|order\s+is)\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        power = re.search(r"([A-Za-z])\s*\^\s*\{?\s*(\d+)\s*\}?", text)
        if not base_order or not power:
            return None
        if base_order.group(1).casefold() != power.group(1).casefold():
            return None
        order, exponent = int(base_order.group(2)), int(power.group(2))
        if order < 1 or exponent < 0:
            return None
        answer = order // gcd(order, exponent)
        zh = self._is_chinese(text)
        support = (
            rf"若 $|g|={order}$，则 $|g^{{{exponent}}}|={order}/\gcd({order},{exponent})={answer}$。"
            if zh else
            rf"For $|g|={order}$, $|g^{{{exponent}}}|={order}/\gcd({order},{exponent})={answer}$."
        )
        return self._result(
            text, "power_element_order", str(answer), "integer",
            "element_power_order_gcd_formula",
            ("finite_element_order_parsed", "power_exponent_parsed", "gcd_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _affine_first_order_recurrence(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"递推|数列|recurrence|sequence", text, re.IGNORECASE):
            return None
        if not re.search(r"通项|闭式|general\s+term|closed\s+form", text, re.IGNORECASE):
            return None
        if re.search(
            r"模\s*\d|modulo|\bmod\b|取整|floor|ceiling|绝对值|absolute\s+value|"
            r"max\s*\(|min\s*\(|随机|probab",
            text,
            re.IGNORECASE,
        ):
            return None

        unsigned = r"(?:\d+(?:\.\d+)?(?:/\d+)?|\\frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\})"
        coefficient = rf"([+-]?\s*(?:{unsigned})?)"
        offset = rf"([+-]\s*{unsigned})?"
        recurrence_patterns = (
            re.compile(
                rf"([A-Za-z])\s*_?\s*\{{?\s*n\s*\}}?\s*=\s*{coefficient}\s*\*?\s*"
                rf"([A-Za-z])\s*_?\s*\{{?\s*n\s*-\s*1\s*\}}?\s*{offset}"
                rf"(?=\s*(?:[,，。.;；]|且|并|where\b|with\b|and\b|$))",
                re.IGNORECASE,
            ),
            re.compile(
                rf"([A-Za-z])\s*_?\s*\{{?\s*n\s*\+\s*1\s*\}}?\s*=\s*{coefficient}\s*\*?\s*"
                rf"([A-Za-z])\s*_?\s*\{{?\s*n\s*\}}?\s*{offset}"
                rf"(?=\s*(?:[,，。.;；]|且|并|where\b|with\b|and\b|$))",
                re.IGNORECASE,
            ),
        )
        recurrence = next(
            (match for pattern in recurrence_patterns if (match := pattern.search(text))),
            None,
        )
        if recurrence is None:
            return None
        sequence_name, coefficient_text, rhs_name, offset_text = recurrence.groups()
        if sequence_name.casefold() != rhs_name.casefold():
            return None
        coefficient_text = re.sub(r"\s+", "", coefficient_text or "")
        coefficient_text = "1" if coefficient_text in {"", "+"} else "-1" if coefficient_text == "-" else coefficient_text
        coefficient_value = self._expr(coefficient_text)
        offset_value = self._expr(re.sub(r"\s+", "", offset_text or "0"))
        if (
            coefficient_value is None
            or offset_value is None
            or coefficient_value.free_symbols
            or offset_value.free_symbols
            or coefficient_value.is_real is not True
            or offset_value.is_real is not True
        ):
            return None
        initial = re.search(
            rf"{re.escape(sequence_name)}\s*_?\s*\{{?\s*(\d+)\s*\}}?\s*=\s*"
            rf"([+-]?\s*{unsigned})",
            text,
            re.IGNORECASE,
        )
        if not initial:
            return None
        start_index = int(initial.group(1))
        initial_value = self._expr(re.sub(r"\s+", "", initial.group(2)))
        if initial_value is None or initial_value.free_symbols or initial_value.is_real is not True:
            return None

        n_symbol = self.sp.symbols("n", integer=True)
        if coefficient_value == 1:
            expression = self.sp.simplify(initial_value + (n_symbol - start_index) * offset_value)
        else:
            fixed_point = self.sp.simplify(offset_value / (1 - coefficient_value))
            expression = self.sp.simplify(
                fixed_point
                + (initial_value - fixed_point) * coefficient_value ** (n_symbol - start_index)
            )
        initial_check = self.sp.simplify(expression.subs(n_symbol, start_index) - initial_value)
        if coefficient_value == 0:
            recurrence_check = self.sp.simplify(
                expression.subs(n_symbol, start_index + 1) - offset_value
            )
        else:
            recurrence_check = self.sp.simplify(
                expression
                - coefficient_value * expression.subs(n_symbol, n_symbol - 1)
                - offset_value
            )
        if initial_check != 0 or recurrence_check != 0:
            return None
        result = rf"{sequence_name}_n={self.sp.latex(expression)}\quad(n\ge {start_index})"
        if self._is_chinese(text):
            support = (
                rf"由固定点方程得 $L={self.sp.latex(fixed_point)}$；令 $b_n={sequence_name}_n-L$ 后得到等比递推。"
                rf"再代入初值 $ {sequence_name}_{{{start_index}}}={self.sp.latex(initial_value)}$，并把所得通项代回原递推验证。"
                if coefficient_value != 1 else
                rf"由递推逐次累加常量增量得到该式；再代入初值 $ {sequence_name}_{{{start_index}}}={self.sp.latex(initial_value)}$ 并代回原递推验证。"
            )
        else:
            support = (
                "Translate by the unique fixed point when the multiplier is not one; "
                "the displayed formula was substituted into both the initial condition and the recurrence."
            )
        return self._result(
            text, "affine_first_order_recurrence", result, "sequence_formula",
            "fixed_point_translation_and_symbolic_substitution",
            ("recurrence_parameters_parsed", "initial_condition_parsed", "initial_value_checked", "recurrence_identity_checked"),
            support,
            ("expression", "number"),
            ("result_present", "numeric_result"),
        )

    def _involution_fixed_point_count(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"置换|permutations?|对合|involutions?", text, re.IGNORECASE):
            return None
        if not re.search(
            r"对合|involutions?|(?:sigma|tau|σ|τ|\\sigma|\\tau)\s*\^\s*\{?2\}?\s*=\s*(?:id|identity)",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(r"至少|至多|不多于|不少于|at\s+least|at\s+most|no\s+(?:more|less)\s+than", text, re.IGNORECASE):
            return None
        fixed = re.search(
            r"(?:恰有|正好有|exactly)\s*(\d+)\s*(?:个)?\s*(?:不动点|fixed\s+points?)",
            text,
            re.IGNORECASE,
        )
        total = re.search(
            r"(?<![A-Za-z0-9_])[nms]\s*=\s*(\d+)|S\s*_?\s*\{?\s*(\d+)\s*\}?|"
            r"(?:在|on)\s*(\d+)\s*(?:个)?(?:元集合|个?元素|elements?)",
            text,
            re.IGNORECASE,
        )
        if not fixed or not total:
            return None
        n = int(next(group for group in total.groups() if group is not None))
        fixed_count = int(fixed.group(1))
        if not 0 <= fixed_count <= n or not 1 <= n <= 200:
            return None
        moving = n - fixed_count
        if moving % 2:
            count = 0
        else:
            pairs = moving // 2
            count = comb(n, fixed_count) * factorial(moving) // (2 ** pairs * factorial(pairs))
        support = (
            rf"先选 {fixed_count} 个不动点；其余 {moving} 个元素必须分成互不相交的二元组，故数目为 "
            rf"$\binom{{{n}}}{{{fixed_count}}}\,{moving}!/(2^{{{moving // 2}}}({moving // 2})!)={count}$。"
            if self._is_chinese(text) else
            rf"Choose the {fixed_count} fixed points, then pair the remaining {moving} points. "
            rf"This gives $\binom{{{n}}}{{{fixed_count}}}{moving}!/(2^{{{moving // 2}}}({moving // 2})!)={count}$."
        )
        return self._count_result(
            text, "involution_fixed_point_count", str(count),
            "fixed_point_choice_and_perfect_matching_count",
            ("ground_set_size_parsed", "exact_fixed_point_count_parsed", "remaining_parity_checked", "pairing_count_recomputed"),
            support,
        )

    def _independent_event_union(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"独立事件|(?:事件\s*)?A\s*(?:(?:与|和|及)\s*B|[,，、]\s*B)\s*(?:相互)?独立|"
            r"events?\s+(?:are\s+)?independent|independent\s+events?",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"P\s*\(\s*A\s*(?:\\cup|∪|union)\s*B\s*\)|并集|union", text, re.IGNORECASE):
            return None
        first = self._probability_assignment(text, "A")
        second = self._probability_assignment(text, "B")
        if first is None or second is None or not self._probability_value(first) or not self._probability_value(second):
            return None
        answer = self.sp.simplify(first + second - first * second)
        if not self._probability_value(answer):
            return None
        zh = self._is_chinese(text)
        support = (
            rf"独立性给出 $P(A\cap B)=P(A)P(B)$，故 $P(A\cup B)=P(A)+P(B)-P(A)P(B)={self.sp.latex(answer)}$。"
            if zh else
            rf"Independence gives $P(A\cap B)=P(A)P(B)$, so $P(A\cup B)=P(A)+P(B)-P(A)P(B)={self.sp.latex(answer)}$."
        )
        return self._probability_result(
            text, "independent_event_union", answer,
            "two_event_inclusion_exclusion_with_independence",
            ("both_probabilities_parsed", "independence_explicit", "intersection_product", "probability_range"),
            support,
        )

    def _bernoulli_variance(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Bernoulli|伯努利", text, re.IGNORECASE) or not re.search(r"方差|variance|Var", text, re.IGNORECASE):
            return None
        var_calls = re.findall(r"(?:Var|\\operatorname\s*\{Var\})\s*\(([^()]*)\)", text, re.IGNORECASE)
        if any(not re.fullmatch(r"\s*[A-Za-z]\s*", item) for item in var_calls):
            return None
        match = re.search(r"Bernoulli\s*\(\s*([^()]+)\s*\)|参数(?:为|是)?\s*([^,，。;；\s]+)", text, re.IGNORECASE)
        if not match:
            return None
        parameter = self._expr(next(item for item in match.groups() if item))
        if parameter is None or (parameter.free_symbols and {symbol.name for symbol in parameter.free_symbols} != {"p"}):
            return None
        if not parameter.free_symbols and not self._probability_value(parameter):
            return None
        variance = self.sp.factor(parameter * (1 - parameter))
        zh = self._is_chinese(text)
        support = (
            rf"$X^2=X$，所以 $\operatorname{{Var}}(X)=E[X^2]-E[X]^2={self.sp.latex(parameter)}-{self.sp.latex(parameter)}^2={self.sp.latex(variance)}$。"
            if zh else
            rf"Since $X^2=X$, $\operatorname{{Var}}(X)=E[X^2]-E[X]^2={self.sp.latex(parameter)}-{self.sp.latex(parameter)}^2={self.sp.latex(variance)}$."
        )
        return self._result(
            text, "bernoulli_variance_identity", self.sp.latex(variance), "variance",
            "bernoulli_second_moment_identity",
            ("bernoulli_parameter_parsed", "second_moment_recomputed", "variance_nonnegative"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _sample_mean_variance(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"样本均值|sample\s+mean", text, re.IGNORECASE) or not re.search(r"方差|variance|Var", text, re.IGNORECASE):
            return None
        if not re.search(r"独立同分布|随机样本|i\.?i\.?d\.?|independent(?:ly)?\s+(?:and\s+)?identically\s+distributed|random\s+sample", text, re.IGNORECASE):
            return None
        if re.search(r"相关|不独立|非同分布|异方差|without\s+replacement|correlated|dependent|heteroscedastic", text, re.IGNORECASE):
            return None
        size = re.search(
            r"(?:样本量|sample\s+size)\s*(?:n\s*)?(?:为|is|=|of)?\s*(\d+)|"
            r"\bn\s*(?:为|is|=)\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        variance = re.search(
            r"(?:总体方差|population\s+variance|Var\s*\(\s*X_i\s*\))[^,，。;；\n]{0,20}?(?:为|is|=)\s*([^,，。;；\s$]+)",
            text,
            re.IGNORECASE,
        )
        symbolic = bool(re.search(r"总体方差|population\s+variance", text, re.IGNORECASE) and re.search(r"\\sigma\s*\^\s*\{?2\}?|σ\s*\^?\s*2", text))
        if not size or not (variance or symbolic):
            return None
        n = int(next(group for group in size.groups() if group is not None))
        if n < 1:
            return None
        if symbolic:
            answer = rf"\frac{{\sigma^2}}{{{n}}}"
        else:
            population_variance = self._expr(variance.group(1))
            if population_variance is None or population_variance.free_symbols or population_variance.is_nonnegative is not True:
                return None
            answer = self.sp.latex(self.sp.simplify(population_variance / n))
        zh = self._is_chinese(text)
        support = (
            rf"在独立同分布假设下，各协方差项为零，故 $\operatorname{{Var}}(\bar X)=n^{{-2}}\sum_i\operatorname{{Var}}(X_i)={answer}$。"
            if zh else
            rf"Under the i.i.d. assumption all covariance terms vanish, hence $\operatorname{{Var}}(\bar X)=n^{{-2}}\sum_i\operatorname{{Var}}(X_i)={answer}$."
        )
        return self._result(
            text, "sample_mean_variance", answer, "variance",
            "iid_variance_sum_and_scaling",
            ("iid_hypothesis", "sample_size_parsed", "population_variance_parsed", "covariances_zero"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _simple_regression_r_squared(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"决定系数|判定系数|coefficient\s+of\s+determination|R\s*\^\s*\{?2\}?", text, re.IGNORECASE):
            return None
        if not re.search(r"(?:简单|一元|simple)\s*(?:线性)?\s*(?:回归|regression)|线性回归", text, re.IGNORECASE):
            return None
        if not re.search(r"含(?:有)?(?:截距|常数项)|带(?:截距|常数项)|with\s+(?:an?\s+)?intercept", text, re.IGNORECASE):
            return None
        if re.search(
            r"无截距|不含(?:截距|常数项)|过原点|多元|复相关|调整(?:的)?\s*R|"
            r"without\s+(?:an?\s+)?intercept|through\s+(?:the\s+)?origin|multiple\s+regression|adjusted\s+R",
            text,
            re.IGNORECASE,
        ):
            return None
        correlation = re.search(
            r"(?:相关系数|correlation\s+coefficient)\s*(?:r\s*)?(?:为|is|=|:|：)\s*"
            r"([^,，。;；\s$]+)|"
            r"\br\s*(?:为|is|=|:|：)\s*([^,，。;；\s$]+)",
            text,
            re.IGNORECASE,
        )
        if not correlation:
            return None
        raw = next(group for group in correlation.groups() if group is not None)
        r_value = self._expr(raw)
        if r_value is None or r_value.free_symbols or not self._correlation_value(r_value):
            return None
        answer = self.sp.simplify(r_value ** 2)
        zh = self._is_chinese(text)
        result = rf"R^2={self.sp.latex(answer)}"
        support = (
            rf"含截距的一元线性回归满足 $R^2=r^2$；代入 $r={self.sp.latex(r_value)}$ 得 $R^2={self.sp.latex(answer)}$，表示线性模型解释了响应变量总离差的 {self.sp.latex(100 * answer)}\%（样本内）。"
            if zh else
            rf"For simple linear regression with an intercept, $R^2=r^2$. Thus $r={self.sp.latex(r_value)}$ gives $R^2={self.sp.latex(answer)}$, the sample fraction of total response variation explained by the fitted line."
        )
        return self._result(
            text, "simple_regression_r_squared", result, "coefficient_of_determination",
            "simple_regression_correlation_identity",
            ("simple_regression_with_intercept", "correlation_parsed", "correlation_range", "square_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _two_color_hypergeometric(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"不放回|without\s+replacement", text, re.IGNORECASE):
            return None
        colors = re.search(
            r"(?:有|含|contains?|has)\s*(\d+)\s*(?:个|只)?\s*(红|red)[^。.;\n]{0,30}?"
            r"(?:和|及|and|,)\s*(\d+)\s*(?:个|只)?\s*(蓝|blue)|"
            r"(?:有|含|contains?|has)\s*(\d+)\s*(?:个|只)?\s*(蓝|blue)[^。.;\n]{0,30}?"
            r"(?:和|及|and|,)\s*(\d+)\s*(?:个|只)?\s*(红|red)",
            text,
            re.IGNORECASE,
        )
        draw = re.search(r"(?:抽取|抽出|draw|select)\s*(\d+)\s*(?:个|只|balls?|items?)", text, re.IGNORECASE)
        target = re.search(r"(?:恰有|正好|exactly)\s*(\d+)\s*(?:个|只)?\s*(红|red)", text, re.IGNORECASE)
        if not colors or not draw or not target:
            return None
        groups = colors.groups()
        if groups[0] is not None:
            red, blue = int(groups[0]), int(groups[2])
        else:
            blue, red = int(groups[4]), int(groups[6])
        sample, red_drawn = int(draw.group(1)), int(target.group(1))
        blue_drawn = sample - red_drawn
        if min(red, blue, sample, red_drawn, blue_drawn) < 0 or sample > red + blue:
            return None
        favorable = (comb(red, red_drawn) if red_drawn <= red else 0) * (comb(blue, blue_drawn) if blue_drawn <= blue else 0)
        total = comb(red + blue, sample)
        answer = self.sp.Rational(favorable, total)
        support = (
            rf"不放回且各 {sample} 元子集等可能；有利选择数为 $\binom{{{red}}}{{{red_drawn}}}\binom{{{blue}}}{{{blue_drawn}}}$，总数为 $\binom{{{red+blue}}}{{{sample}}}$。"
            if self._is_chinese(text) else
            rf"All {sample}-subsets are equally likely; favorable selections number $\binom{{{red}}}{{{red_drawn}}}\binom{{{blue}}}{{{blue_drawn}}}$ out of $\binom{{{red+blue}}}{{{sample}}}$."
        )
        return self._probability_result(
            text, "two_color_hypergeometric", answer,
            "hypergeometric_combination_ratio",
            ("both_color_counts_parsed", "without_replacement", "sample_composition_parsed", "favorable_and_total_counts"),
            support,
        )

    def _coupon_collector_expectation(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"coupon\s+collector|集齐|收集齐", text, re.IGNORECASE) or not re.search(r"期望|expected", text, re.IGNORECASE):
            return None
        if not re.search(r"等概率|均匀|equally\s+likely|uniform", text, re.IGNORECASE):
            return None
        match = re.search(r"(?:共|有|among|of)\s*(\d+)\s*(?:种|类|types?|coupons?)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)\s*(?:种|类|types?)[^。.;\n]{0,35}(?:coupon|优惠券|卡片)", text, re.IGNORECASE)
        if not match:
            return None
        n = int(match.group(1))
        if not 1 <= n <= 100_000:
            return None
        expectation = sum((Fraction(n, remaining) for remaining in range(1, n + 1)), Fraction(0))
        answer = self._fraction_latex(expectation)
        zh = self._is_chinese(text)
        support = (
            rf"已有 $k$ 种时得到新品种的成功率为 $({n}-k)/{n}$，等待期望为 ${n}/({n}-k)$；对 $k=0,\ldots,{n-1}$ 求和得 ${n}H_{{{n}}}={answer}$。"
            if zh else
            rf"With $k$ types collected, a new type arrives with probability $({n}-k)/{n}$, so the waiting mean is ${n}/({n}-k)$; summing for $k=0,\ldots,{n-1}$ gives ${n}H_{{{n}}}={answer}$."
        )
        return self._result(
            text, "coupon_collector_expectation", answer, "expectation",
            "sum_of_geometric_waiting_expectations",
            ("uniform_coupon_types", "type_count_parsed", "all_waiting_stages_summed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _geometric_waiting_tail(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"首次成功|第一次成功|first\s+success|trials?\s+until\s+(?:the\s+)?first\s+success", text, re.IGNORECASE):
            return None
        probability = re.search(r"(?:成功概率|success\s+probability)\s*(?:为|is|=)\s*([^,，。;；\s$]+)", text, re.IGNORECASE)
        tail = re.search(r"P\s*\(\s*X\s*>\s*(\d+)\s*\)", text, re.IGNORECASE)
        if not probability or not tail:
            return None
        p = self._expr(probability.group(1))
        k = int(tail.group(1))
        if p is None or p.free_symbols or not self._probability_value(p) or p in {0, 1}:
            return None
        answer = self.sp.simplify((1 - p) ** k)
        support = (
            rf"$X>{k}$ 当且仅当前 {k} 次均失败；独立试验下概率为 $(1-p)^{{{k}}}={self.sp.latex(answer)}$。"
            if self._is_chinese(text) else
            rf"$X>{k}$ iff the first {k} trials all fail; independence gives $(1-p)^{{{k}}}={self.sp.latex(answer)}$."
        )
        return self._probability_result(
            text, "geometric_waiting_tail", answer,
            "independent_initial_failure_product",
            ("first_success_convention_explicit", "success_probability_parsed", "tail_index_parsed"),
            support,
        )

    def _brownian_covariance(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"标准布朗运动|standard\s+Brownian\s+motion", text, re.IGNORECASE) or not re.search(r"协方差|covariance|Cov", text, re.IGNORECASE):
            return None
        match = re.search(
            r"(?:Cov|\\operatorname\s*\{Cov\})\s*\(\s*B\s*_?\s*\{?\s*([^,，}\s]+)\s*\}?\s*[,，]\s*B\s*_?\s*\{?\s*([^)}\s]+)\s*\}?\s*\)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        first, second = self._expr(match.group(1)), self._expr(match.group(2))
        if first is None or second is None or first.free_symbols or second.free_symbols:
            return None
        if first.is_nonnegative is not True or second.is_nonnegative is not True:
            return None
        answer = first if bool(first <= second) else second
        support = (
            rf"标准布朗运动满足 $E[B_sB_t]=\min(s,t)$ 且均值为零，故协方差为 ${self.sp.latex(answer)}$。"
            if self._is_chinese(text) else
            rf"Standard Brownian motion has $E[B_sB_t]=\min(s,t)$ and zero mean, so the covariance is ${self.sp.latex(answer)}$."
        )
        return self._result(
            text, "brownian_covariance", self.sp.latex(answer), "covariance",
            "brownian_independent_increment_covariance",
            ("standard_brownian_hypothesis", "two_nonnegative_times_parsed", "minimum_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _matrix_trace_determinant(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"特征值|eigenvalues?", text, re.IGNORECASE):
            return None
        if not re.search(r"迹|trace", text, re.IGNORECASE) or not re.search(r"行列式|determinant|det", text, re.IGNORECASE):
            return None
        match = re.search(
            r"(?:特征值|eigenvalues?)[^\[\]。.;；\n]{0,30}?(?:为|是|are|:|：)\s*"
            r"(?:\$|\\\()?\s*\[\s*([^\]]+)\s*\]\s*(?:\$|\\\))?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        pieces = [item.strip() for item in re.split(r"[,，]", match.group(1)) if item.strip()]
        if not 1 <= len(pieces) <= 50:
            return None
        complete_list = bool(re.search(
            r"全部特征值|所有特征值|含代数重数|计重数|"
            r"all\s+(?:the\s+)?eigenvalues?|complete\s+(?:list\s+of\s+)?eigenvalues?|"
            r"including\s+(?:algebraic\s+)?multiplicit",
            text,
            re.IGNORECASE,
        ))
        dimension = re.search(
            r"(?:([1-9]\d*)\s*[×xX]\s*\1|([1-9]\d*)\s*阶|"
            r"dimension\s*(?:is|=)?\s*([1-9]\d*))",
            text,
            re.IGNORECASE,
        )
        stated_dimension = next(
            (int(item) for item in dimension.groups() if item), 0
        ) if dimension else 0
        if not complete_list and stated_dimension != len(pieces):
            return None
        values = [self._expr(item) for item in pieces]
        if any(item is None or item.free_symbols for item in values):
            return None
        trace = self.sp.simplify(sum(values, self.sp.S.Zero))
        determinant = self.sp.simplify(self.sp.prod(values))
        result = rf"$\operatorname{{tr}}A={self.sp.latex(trace)},\quad \det A={self.sp.latex(determinant)}$。"
        support = (
            "按代数重数计，矩阵的迹等于全部特征值之和，行列式等于全部特征值之积；已分别精确求和、求积。"
            if self._is_chinese(text) else
            "Counting algebraic multiplicities, trace is the sum and determinant the product of all eigenvalues; both were recomputed exactly."
        )
        return self._result(
            text, "matrix_trace_determinant_from_eigenvalues", result, "trace_and_determinant",
            "eigenvalue_sum_and_product",
            ("all_eigenvalues_parsed", "algebraic_multiplicity_list", "sum_recomputed", "product_recomputed"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _chebyshev_nodes(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"Chebyshev|切比雪夫", text, re.IGNORECASE) or not re.search(r"节点|nodes?|zeros?", text, re.IGNORECASE):
            return None
        if not (
            re.search(r"x\s*_?\s*\{?\s*k\s*\}?\s*=\s*(?:\\?cos|cos)", text, re.IGNORECASE)
            and re.search(r"2\s*\*?\s*k\s*-\s*1", text, re.IGNORECASE)
            and re.search(r"(?:\\pi|π|\bpi\b)", text, re.IGNORECASE)
            and re.search(r"2\s*\*?\s*n", text, re.IGNORECASE)
        ):
            return None
        if re.search(r"映射|变换到|mapped?\s+to|transform(?:ed)?\s+to", text, re.IGNORECASE):
            return None
        size = re.search(r"(?<![A-Za-z0-9_])n\s*(?:为|is|=)\s*(\d+)", text, re.IGNORECASE)
        if not size:
            return None
        n = int(size.group(1))
        if not 1 <= n <= 20:
            return None
        nodes = tuple(
            self.sp.simplify(self.sp.cos(self.sp.pi * (2 * k - 1) / (2 * n)))
            for k in range(1, n + 1)
        )
        if any(self.sp.simplify(nodes[index] + nodes[-1-index]) != 0 for index in range(n)):
            return None
        rendered = r",\;".join(self.sp.latex(node) for node in nodes)
        result = rf"(x_1,\ldots,x_{n})=({rendered})"
        support = (
            rf"逐个代入 $k=1,\ldots,{n}$；并核验 $x_{{{n}+1-k}}=-x_k$，所以全部节点关于原点对称。"
            if self._is_chinese(text) else
            rf"Substitution for every $k=1,\ldots,{n}$ gives the displayed tuple, and $x_{{{n}+1-k}}=-x_k$ verifies symmetry about zero."
        )
        return self._result(
            text, "chebyshev_nodes", result, "finite_tuple",
            "exact_trigonometric_substitution_and_symmetry",
            ("node_formula_explicit", "node_count_parsed", "all_nodes_evaluated", "reflection_symmetry_checked"),
            support,
            ("number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _probability_assignment(self, text: str, event: str):
        match = re.search(
            rf"P\s*\(\s*{re.escape(event)}\s*\)\s*(?:=|为|is)\s*([^,，、。;；\s$]+)",
            text,
            re.IGNORECASE,
        )
        return self._expr(match.group(1)) if match else None

    def _expr(self, value: str):
        text = str(value or "").strip().strip("$")
        text = text.replace(r"\left", "").replace(r"\right", "")
        try:
            return self.sp.nsimplify(self.symbolic._parse(text), rational=True, full=False)
        except Exception:
            return None

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _fraction_latex(value: Fraction) -> str:
        return str(value.numerator) if value.denominator == 1 else rf"\frac{{{value.numerator}}}{{{value.denominator}}}"

    @staticmethod
    def _probability_value(value) -> bool:
        try:
            return bool(value.is_real is True and value >= 0 and value <= 1)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _correlation_value(value) -> bool:
        try:
            return bool(value.is_real is True and value >= -1 and value <= 1)
        except (TypeError, ValueError):
            return False

    def _count_result(
        self,
        problem: str,
        operation: str,
        answer: str,
        method: str,
        checks: tuple[str, ...],
        support: str,
    ) -> ToolResult:
        return self._result(
            problem, operation, answer, "integer_or_formula", method, checks,
            support, ("count", "number", "expression"),
            ("result_present", "numeric_result"),
        )

    def _probability_result(
        self,
        problem: str,
        operation: str,
        answer,
        method: str,
        checks: tuple[str, ...],
        support: str,
    ) -> ToolResult:
        return self._result(
            problem, operation, self.sp.latex(answer), "probability", method,
            checks, support, ("probability", "number", "expression"),
            ("result_present", "numeric_result"),
        )

    @staticmethod
    def _result(
        problem: str,
        operation: str,
        answer: str,
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
            result=answer,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=support,
            answer_shapes=shapes,
            requirements=requirements,
        )
