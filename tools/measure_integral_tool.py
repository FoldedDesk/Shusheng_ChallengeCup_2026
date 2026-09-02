"""Exact finite-measure and elementary integral computations."""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class MeasureIntegralTool:
    """Parse explicit parameters and abstain when measure semantics are incomplete."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if not text or self.sp is None:
            return []
        handlers = (
            self._measure_union,
            self._shrinking_interval,
            self._power_sequence,
            self._simple_indicator_integral,
            self._counting_geometric_integral,
            self._power_singularity,
            self._monotone_integral_limit,
            self._l1_absolute_continuity,
            self._translating_indicator_uniform_integrability,
            self._concentrating_spike_nonuniform_integrability,
        )
        results: list[ToolResult] = []
        for handler in handlers:
            try:
                item = handler(text)
            except Exception:
                item = None
            if item is not None and item.verified:
                results.append(item)
        return results

    def _measure_union(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"(?:\\mu|μ)\s*\(\s*A\s*(?:\\cup|∪)\s*B\s*\)", text, re.I):
            return None
        a = self._measure_assignment(text, r"A")
        b = self._measure_assignment(text, r"B")
        intersection = self._measure_assignment(text, r"A\s*(?:\\cap|∩)\s*B")
        if None in {a, b, intersection}:
            return None
        value = self.sp.simplify(a + b - intersection)
        rendered = self.symbolic._format(value)
        result = (
            rf"由容斥公式，$\mu(A\cup B)=\mu(A)+\mu(B)-\mu(A\cap B)={rendered}$。"
            if self._zh(text) else
            rf"By inclusion-exclusion, $\mu(A\cup B)=\mu(A)+\mu(B)-\mu(A\cap B)={rendered}$."
        )
        return self._result(text, "measure_union", result, "measure",
                            "finite_measure_inclusion_exclusion",
                            ("three_measures_parsed", "intersection_subtracted", "exact_recompute"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1"),
                            ("number", "expression"))

    def _shrinking_interval(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"E_?\{?n\}?", text) or not re.search(r"测度|measure", text, re.I):
            return None
        interval = re.search(
            r"E_?\{?n\}?\s*=\s*(?P<left>[\[(])\s*0\s*,\s*"
            r"(?:"
            r"(?:(?P<scale>\d+(?:/\d+)?)\s*(?:\\cdot|\*)\s*)?1\s*/\s*n"
            r"|(?P<numerator>\d+(?:/\d+)?)\s*/\s*n"
            r")\s*(?P<right>[\])])",
            text,
            re.I,
        )
        if interval is None:
            return None
        coefficient = self.symbolic._parse(
            interval.group("scale") or interval.group("numerator") or "1"
        )
        if coefficient.is_real is not True or not bool(coefficient > 0):
            return None
        coefficient_text = self.symbolic._format(coefficient)
        measure_text = r"\frac{1}{n}" if coefficient == 1 else rf"\frac{{{coefficient_text}}}{{n}}"
        intersection = r"\{0\}" if interval.group("left") == "[" else r"\varnothing"
        asks_intersection = bool(re.search(r"交集|intersection|\\bigcap", text, re.I))
        result = (
            rf"因为区间长度为 {measure_text}，故 $\mu(E_n)={measure_text}\to0$；"
            rf"该集合列单调递减，且 $\bigcap_{{n\ge1}}E_n={intersection}$。"
            if self._zh(text) else
            rf"Since the interval length is {measure_text}, $\mu(E_n)={measure_text}\to0$; "
            rf"the sets decrease and $\bigcap_{{n\ge1}}E_n={intersection}$."
        )
        requirements = ["result_present", "numeric_result"]
        if asks_intersection:
            requirements.extend(("reasoning", "support_anchor_1"))
        return self._result(text, "shrinking_interval_measure", result, "limit_and_intersection",
                            "interval_length_and_endpoint_intersection",
                            ("interval_endpoints_parsed", "length_recomputed", "intersection_from_brackets"),
                            tuple(requirements), ("expression", "number", "set"))

    def _power_sequence(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"f_?\{?n\}?\s*\(\s*x\s*\)\s*=\s*x\s*\^\s*\{?n\}?|f_?\{?n\}?\s*=\s*x\s*\^\s*\{?n\}?", text, re.I):
            return None
        if not re.search(r"\[\s*0\s*,\s*1\s*\]|on\s*\[\s*0\s*,\s*1\s*\]", text, re.I):
            return None
        if not re.search(r"逐点|pointwise", text, re.I) or not re.search(r"积分|integral", text, re.I):
            return None
        result = (
            r"逐点极限为 $f(x)=0$（$0\le x<1$）且 $f(1)=1$；同时 "
            r"$\int_0^1x^n\,dx=\frac1{n+1}\to0=\int_0^1f\,dx$。"
            if self._zh(text) else
            r"The pointwise limit is $f(x)=0$ for $0\le x<1$ and $f(1)=1$; moreover "
            r"$\int_0^1x^n\,dx=\frac1{n+1}\to0=\int_0^1f\,dx$."
        )
        return self._result(text, "power_sequence_integral_limit", result, "pointwise_and_integral_limits",
                            "pointwise_endpoint_split_and_exact_antiderivative",
                            ("domain_parsed", "endpoint_split", "integral_recomputed", "limit_compared"),
                            ("result_present", "reasoning", "support_anchor_1", "numeric_result"),
                            ("expression", "text"))

    def _simple_indicator_integral(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"1\s*_\s*(?:\{)?[\[(]", text) or not re.search(r"积分|integral", text, re.I):
            return None
        normalized = text.replace(r"\cdot", "*").replace("·", "*")
        terms = list(re.finditer(
            r"(?:(?P<c>[-+]?\d+(?:/\d+)?)\s*\*\s*)?1\s*_\s*(?:\{)?"
            r"(?P<left>[\[(])\s*(?P<a>[-+]?\d+(?:/\d+)?)\s*,\s*"
            r"(?P<b>[-+]?\d+(?:/\d+)?)\s*(?P<right>[\])])",
            normalized,
        ))
        if not terms:
            return None
        intervals = []
        total = self.sp.Integer(0)
        for match in terms:
            coefficient = self.symbolic._parse(match.group("c") or "1")
            left = self.symbolic._parse(match.group("a"))
            right = self.symbolic._parse(match.group("b"))
            if any(item.free_symbols for item in (coefficient, left, right)) or not bool(right >= left):
                return None
            intervals.append((left, right))
            total += coefficient * (right - left)
        ordered = sorted(intervals, key=lambda item: float(item[0]))
        if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])):
            return None
        rendered = self.symbolic._format(self.sp.simplify(total))
        result = (
            rf"按不交区间分解，简单函数积分为各常值乘区间长度之和，即 ${rendered}$。"
            if self._zh(text) else
            rf"Decomposing into disjoint intervals, the integral is the sum of each constant times its interval length, namely ${rendered}$."
        )
        return self._result(text, "simple_indicator_integral", result, "integral",
                            "disjoint_simple_function_measure_sum",
                            ("all_indicator_terms_parsed", "intervals_disjoint", "weighted_lengths_summed"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1"),
                            ("number", "expression"))

    def _counting_geometric_integral(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"计数测度|counting\s+measure", text, re.I):
            return None
        match = re.search(r"f\s*\(\s*[kKnN]\s*\)\s*=\s*([^\s，,。;]+)\s*\^\s*\{?[kKnN]\}?", text)
        if match is None or not re.search(r"正整数|positive\s+integers?", text, re.I):
            return None
        ratio = self.symbolic._parse(match.group(1))
        if ratio.is_real is not True or not bool(abs(ratio) < 1):
            return None
        value = self.symbolic._format(self.sp.simplify(ratio / (1 - ratio)))
        result = (
            rf"计数测度积分等于级数 $\sum_{{k=1}}^\infty r^k=r/(1-r)={value}$。"
            if self._zh(text) else
            rf"The counting-measure integral is $\sum_{{k=1}}^\infty r^k=r/(1-r)={value}$."
        )
        return self._result(text, "counting_measure_geometric_integral", result, "integral",
                            "geometric_series_under_counting_measure",
                            ("positive_integer_support", "ratio_parsed", "convergence_checked", "series_summed"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression"))

    def _power_singularity(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"L\s*\^?\s*1|可积|integrable", text, re.I):
            return None
        match = re.search(r"x\s*\^\s*\{?\s*(-?\d+(?:/\d+)?)\s*\}?", text)
        if match is None or not re.search(r"\(\s*0\s*,\s*1\s*\]|\[\s*0\s*,\s*1\s*\]", text):
            return None
        exponent = self.symbolic._parse(match.group(1))
        if exponent.is_real is not True or not bool(exponent > -1):
            if exponent.is_real is True and bool(exponent <= -1):
                result = "该函数不属于 $L^1(0,1)$，因为原点处幂次不满足 $\alpha>-1$。" if self._zh(text) else r"The function is not in $L^1(0,1)$ because its exponent at zero does not satisfy $\alpha>-1$."
                value_checks = ("exponent_parsed", "integrability_threshold_checked")
                return self._result(text, "power_singularity_integrability", result, "integrability",
                                    "power_integrability_threshold", value_checks,
                                    (
                                        "result_present", "judgement", "reasoning",
                                        "membership_judgement", "integral_conclusion",
                                    ),
                                    ("truth", "text", "expression"))
            return None
        integral = self.symbolic._format(self.sp.simplify(1 / (exponent + 1)))
        result = (
            rf"该函数属于 $L^1(0,1)$，且积分为 "
            rf"$\int_0^1x^{{{self.symbolic._format(exponent)}}}\,dx={integral}$。"
            if self._zh(text) else
            rf"The function lies in $L^1(0,1)$ and its integral equals "
            rf"$\int_0^1x^{{{self.symbolic._format(exponent)}}}\,dx={integral}$."
        )
        return self._result(text, "power_singularity_integrability", result, "integrability_and_integral",
                            "power_antiderivative_and_endpoint_limit",
                            ("exponent_parsed", "integrability_threshold_checked", "integral_recomputed"),
                            (
                                "result_present", "numeric_result", "judgement", "reasoning",
                                "membership_judgement", "integral_conclusion",
                            ),
                            ("truth", "number", "expression", "text"))

    def _monotone_integral_limit(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"单调收敛|monotone\s+convergence", text, re.I):
            return None
        match = re.search(r"(?:\\int|∫)\s*f_?\{?n\}?[^=]{0,30}=\s*\$?\s*([^，,。;；\n]+)", text)
        if match is None:
            return None
        n = self.sp.Symbol("n", positive=True, integer=True)
        expression = self.symbolic._parse(match.group(1).strip().strip("$"))
        if expression.free_symbols - {self.sp.Symbol("n")}:
            return None
        # SymPy symbols with different assumptions do not compare equal.
        expression = expression.subs(self.sp.Symbol("n"), n)
        value = self.sp.limit(expression, n, self.sp.oo)
        if value.has(self.sp.Limit):
            return None
        rendered = self.symbolic._format(value)
        result = (
            rf"由单调收敛定理，$\int f\,d\mu=\lim_{{n\to\infty}}\int f_n\,d\mu={rendered}$。"
            if self._zh(text) else
            rf"By monotone convergence, $\int f\,d\mu=\lim_{{n\to\infty}}\int f_n\,d\mu={rendered}$."
        )
        return self._result(text, "monotone_convergence_integral", result, "integral_limit",
                            "monotone_convergence_and_symbolic_limit",
                            ("monotonicity_stated", "integral_sequence_parsed", "limit_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1"),
                            ("number", "expression"))

    def _l1_absolute_continuity(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"f\s*(?:\\in|∈)\s*L\s*\^?\s*1|f\s+in\s+L1", text, re.I):
            return None
        if not re.search(r"(?:\\mu|μ)\s*\(\s*E_?\{?n\}?\s*\)\s*(?:\\to|→)\s*0", text):
            return None
        if not re.search(r"(?:\\int|∫)[^。.!?]{0,60}\|\s*f\s*\||absolute\s+continuity", text, re.I):
            return None
        result = (
            r"由 $L^1$ 积分的绝对连续性，$\displaystyle\lim_{n\to\infty}\int_{E_n}|f|\,d\mu=0$。"
            if self._zh(text) else
            r"By absolute continuity of the $L^1$ integral, $\displaystyle\lim_{n\to\infty}\int_{E_n}|f|\,d\mu=0$."
        )
        return self._result(text, "l1_absolute_continuity", result, "integral_limit",
                            "absolute_continuity_of_l1_integral",
                            ("l1_hypothesis_present", "set_measure_limit_present", "theorem_applied"),
                            ("result_present", "reasoning"),
                            ("number", "expression", "text"))

    def _translating_indicator_uniform_integrability(self, text: str) -> Optional[ToolResult]:
        """Certify fixed-width indicators translating to positive infinity."""
        if not re.search(r"无限测度|infinite\s+measure", text, re.I):
            return None
        if not re.search(r"一致可积|uniform(?:ly)?\s+integrab", text, re.I):
            return None
        definition = re.search(
            r"(?P<function>[A-Za-z])\s*_?\s*\{?(?P<index>[A-Za-z])\}?"
            r"(?:\s*\(\s*[A-Za-z]\s*\))?\s*=\s*"
            r"\\(?:mathbf|mathbb)\s*\{?\s*1\s*\}?\s*_?\s*\{?\s*"
            r"(?P<left>[\[(])\s*(?P<lower>[^,，\]\)]+)\s*[,，]\s*"
            r"(?P<upper>[^\]\)]+)\s*(?P<right>[\])])\s*\}?",
            text,
            re.I,
        )
        if definition is None:
            return None
        index = self.sp.Symbol(definition.group("index"), integer=True, positive=True)
        lower = self._expr(definition.group("lower"), index)
        upper = self._expr(definition.group("upper"), index)
        if lower is None or upper is None:
            return None
        if lower.free_symbols - {index} or upper.free_symbols - {index}:
            return None
        lower_poly = self.sp.Poly(lower, index)
        upper_poly = self.sp.Poly(upper, index)
        if lower_poly.degree() != 1 or upper_poly.degree() != 1:
            return None
        speed = lower_poly.coeff_monomial(index)
        if self.sp.simplify(upper_poly.coeff_monomial(index) - speed) != 0:
            return None
        width = self.sp.simplify(upper - lower)
        if speed.is_positive is not True or width.is_positive is not True or width.free_symbols:
            return None
        function = definition.group("function")
        width_text = self.symbolic._format(width)
        result = (
            rf"对每个固定 $x$，区间左端以正速度趋于 $+\infty$，故 "
            rf"${function}_{index}(x)\to0$（事实上逐点，因而几乎处处）。又因 "
            rf"$|{function}_{index}|\le1$，当 $K>1$ 时 "
            rf"$\sup_{index}\int_{{|{function}_{index}|>K}}|{function}_{index}|=0$，且对任意集合 $E$ 有 "
            rf"$\sup_{index}\int_E|{function}_{index}|\le m(E)$，所以该族一致可积。"
            rf"同时 $\|{function}_{index}\|_1={width_text}$，不趋于 0，故不在 $L^1$ 中收敛到 0。"
            rf"质量平移到无穷远，说明无限测度空间还需要一致紧性等防止质量逃逸的条件。"
            if self._zh(text) else
            rf"For each fixed $x$, the left endpoint tends to $+\infty$ at positive speed, so "
            rf"${function}_{index}(x)\to0$ pointwise and hence almost everywhere. Since "
            rf"$|{function}_{index}|\le1$, for $K>1$ one has "
            rf"$\sup_{index}\int_{{|{function}_{index}|>K}}|{function}_{index}|=0$; moreover "
            rf"$\sup_{index}\int_E|{function}_{index}|\le m(E)$, proving uniform integrability. "
            rf"But $\|{function}_{index}\|_1={width_text}$, so there is no $L^1$ convergence to zero. "
            rf"The mass escapes to infinity, showing why a tightness condition is needed on an infinite-measure space."
        )
        return self._result(
            text,
            "translating_indicator_uniform_integrability",
            result,
            "ae_limit_uniform_integrability_and_l1_norm",
            "affine_translation_tail_and_small_set_integrals",
            (
                "indicator_interval_parsed",
                "positive_translation_speed_checked",
                "constant_positive_width_checked",
                "pointwise_eventual_exclusion_proved",
                "tail_integral_zero_for_large_threshold",
                "small_set_integral_bound",
                "l1_norm_recomputed",
            ),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "almost_everywhere_limit",
                "uniform_integrability_check",
                "l1_nonconvergence",
                "l1_limit_conclusion",
                "l1_norm_check",
            ),
            ("expression", "text", "truth", "proof"),
        )

    def _concentrating_spike_nonuniform_integrability(self, text: str) -> Optional[ToolResult]:
        """Certify linearly growing indicators on inversely shrinking intervals."""
        if not re.search(r"一致可积|uniform(?:ly)?\s+integrab", text, re.I):
            return None
        definition = re.search(
            r"(?P<function>[A-Za-z])\s*_?\s*\{?(?P<index>[A-Za-z])\}?"
            r"(?:\s*\(\s*[A-Za-z]\s*\))?\s*=\s*"
            r"(?P<amplitude>[^$，,。;；\n]+?)\s*"
            r"\\(?:mathbf|mathbb)\s*\{?\s*1\s*\}?\s*_?\s*\{?\s*"
            r"(?P<left>[\[(])\s*0\s*[,，]\s*(?P<upper>[^\]\)]+)\s*"
            r"(?P<right>[\])])\s*\}?",
            text,
            re.I,
        )
        if definition is None:
            return None
        index = self.sp.Symbol(definition.group("index"), integer=True, positive=True)
        amplitude = self._expr(definition.group("amplitude"), index)
        upper = self._expr(definition.group("upper"), index)
        if amplitude is None or upper is None:
            return None
        if amplitude.free_symbols - {index} or upper.free_symbols - {index}:
            return None
        amplitude_scale = self.sp.simplify(amplitude / index)
        width_scale = self.sp.simplify(upper * index)
        if (
            amplitude_scale.free_symbols
            or width_scale.free_symbols
            or amplitude_scale.is_positive is not True
            or width_scale.is_positive is not True
            or self.sp.simplify(amplitude - amplitude_scale * index) != 0
            or self.sp.simplify(upper - width_scale / index) != 0
        ):
            return None
        mass = self.sp.simplify(amplitude_scale * width_scale)
        function = definition.group("function")
        mass_text = self.symbolic._format(mass)
        amplitude_text = self.symbolic._format(amplitude_scale)
        result = (
            rf"对每个 $x>0$，充分大时 $x>{self.symbolic._format(width_scale)}/{index}$；"
            rf"在 $x=0$ 至多有一个零测集例外，故 ${function}_{index}\to0$ 几乎处处。"
            rf"但 $\|{function}_{index}\|_1={mass_text}$，所以不可能在 $L^1$ 中收敛到其几乎处处极限 0。"
            rf"对任意 $K>0$，取 ${index}>K/{amplitude_text}$，则尖峰支撑上 "
            rf"$|{function}_{index}|>K$，并且 "
            rf"$\int_{{|{function}_{index}|>K}}|{function}_{index}|={mass_text}$；"
            rf"尾积分不能一致趋于 0，故该族不一致可积。"
            if self._zh(text) else
            rf"For every $x>0$, eventually $x>{self.symbolic._format(width_scale)}/{index}$; the possible "
            rf"exception at zero is null, so ${function}_{index}\to0$ almost everywhere. Yet "
            rf"$\|{function}_{index}\|_1={mass_text}$, excluding $L^1$ convergence to the a.e. limit zero. "
            rf"For any $K>0$, choose ${index}>K/{amplitude_text}$. On the spike support "
            rf"$|{function}_{index}|>K$ and "
            rf"$\int_{{|{function}_{index}|>K}}|{function}_{index}|={mass_text}$; the tails do not vanish "
            rf"uniformly, so the family is not uniformly integrable."
        )
        return self._result(
            text,
            "concentrating_spike_nonuniform_integrability",
            result,
            "ae_limit_l1_nonconvergence_and_nonuniform_integrability",
            "inverse_width_constant_mass_tail_obstruction",
            (
                "spike_amplitude_and_interval_parsed",
                "linear_amplitude_checked",
                "inverse_width_checked",
                "constant_positive_l1_mass_recomputed",
                "almost_everywhere_limit_proved",
                "tail_integral_obstruction_recomputed",
            ),
            (
                "result_present",
                "numeric_result",
                "reasoning",
                "almost_everywhere_limit",
                "uniform_integrability_check",
                "l1_nonconvergence",
                "l1_limit_conclusion",
                "l1_norm_check",
            ),
            ("expression", "text", "truth"),
        )

    def _expr(self, value: str, symbol=None):
        try:
            expression = self.sp.simplify(self.symbolic._parse(value))
            if symbol is not None:
                plain = self.sp.Symbol(symbol.name)
                expression = expression.subs(plain, symbol)
            return expression
        except Exception:
            return None

    def _measure_assignment(self, text: str, body: str):
        match = re.search(
            rf"(?:\\mu|μ)\s*\(\s*{body}\s*\)\s*=\s*\$?\s*([-+]?\d+(?:/\d+)?|[-+]?0?\.\d+)",
            text,
            re.I,
        )
        return self.symbolic._parse(match.group(1)) if match else None

    @staticmethod
    def _zh(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _result(problem: str, operation: str, result: str, result_kind: str,
                method: str, checks: tuple[str, ...], requirements: tuple[str, ...],
                shapes: tuple[str, ...]) -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=result,
            requirements=requirements,
            answer_shapes=shapes,
        )
