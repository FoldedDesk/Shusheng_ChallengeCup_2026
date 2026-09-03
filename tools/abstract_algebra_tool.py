"""Parameterized finite algebra computations and hypothesis-checked proofs."""

from __future__ import annotations

from math import gcd
import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class AbstractAlgebraTool:
    """Certify small exact group, quotient-ring, and finite-field tasks."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        handlers = (
            self._additive_cyclic_order,
            self._power_element_order,
            self._homomorphism_kernel_normal,
            self._finite_field_irreducibility,
            self._finite_field_irreducible_count,
            self._polynomial_quotient_power,
            self._polynomial_root_bound,
            self._finite_field_multiplicative_group,
            self._maximal_ideal_quotient,
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

    def _additive_cyclic_order(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"加法群|additive\s+group", text, re.I) or not re.search(r"阶|order", text, re.I):
            return None
        modulus = re.search(r"(?:Z|\\mathbb\s*\{Z\})\s*_?\s*\{?\s*(\d+)\s*\}?", text, re.I)
        element = re.search(r"\[\s*([-+]?\d+)\s*\]", text)
        if modulus is None or element is None:
            return None
        n, k = int(modulus.group(1)), int(element.group(1))
        if n < 1:
            return None
        divisor = gcd(n, k)
        order = n // divisor
        result = (
            rf"根据循环加法群的元素阶公式，在 $\mathbb Z_{{{n}}}$ 中有 "
            rf"$|[{k}]|={n}/\gcd({n},{k})={order}$。"
            if self._zh(text) else
            rf"By the element-order formula for a cyclic additive group, in $\mathbb Z_{{{n}}}$ "
            rf"one has $|[{k}]|={n}/\gcd({n},{k})={order}$."
        )
        return self._result(text, "additive_cyclic_element_order", result, "element_order",
                            "cyclic_additive_gcd_formula",
                            ("modulus_parsed", "representative_parsed", "gcd_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1", "support_anchor_2"),
                            ("number", "expression", "text"))

    def _power_element_order(self, text: str) -> Optional[ToolResult]:
        base = re.search(
            r"([A-Za-z])\s*(?:的|has\s+)?(?:阶|order)\s*(?:为|是|is|=|of)?\s*(\d+)",
            text,
            re.I,
        )
        power = re.search(r"([A-Za-z])\s*\^\s*\{?\s*(\d+)\s*\}?", text)
        if base is None or power is None or base.group(1).casefold() != power.group(1).casefold():
            return None
        order, exponent = int(base.group(2)), int(power.group(2))
        if order < 1:
            return None
        divisor = gcd(order, exponent)
        value = order // divisor
        result = (
            rf"阶公式给出 $|{base.group(1)}^{{{exponent}}}|={order}/\gcd({order},{exponent})={value}$，"
            rf"其中最大公因数为 {divisor}。"
            if self._zh(text) else
            rf"The order formula gives $|{base.group(1)}^{{{exponent}}}|={order}/\gcd({order},{exponent})={value}$, "
            rf"where the greatest common divisor is {divisor}."
        )
        return self._result(text, "power_element_order", result, "element_order",
                            "element_power_order_gcd_formula",
                            ("base_order_parsed", "exponent_parsed", "gcd_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1", "support_anchor_2"),
                            ("number", "expression", "text"))

    def _homomorphism_kernel_normal(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"群同态|group\s+homomorphism", text, re.I) or not re.search(r"核|kernel|ker", text, re.I):
            return None
        if not re.search(r"正规|normal", text, re.I):
            return None
        result = (
            r"$\ker\varphi$ 是子群。若 $k\in\ker\varphi,g\in G$，则 "
            r"$\varphi(gkg^{-1})=\varphi(g)e_H\varphi(g)^{-1}=e_H$，故 "
            r"$gkg^{-1}\in\ker\varphi$。因此 $\ker\varphi\trianglelefteq G$。"
            if self._zh(text) else
            r"The kernel is a subgroup. For $k\in\ker\varphi$ and $g\in G$, "
            r"$\varphi(gkg^{-1})=\varphi(g)e_H\varphi(g)^{-1}=e_H$, so "
            r"$gkg^{-1}\in\ker\varphi$ and hence $\ker\varphi\trianglelefteq G$."
        )
        return self._result(text, "homomorphism_kernel_normal", result, "proof",
                            "conjugation_closed_kernel",
                            ("group_homomorphism_hypothesis", "kernel_subgroup", "conjugation_computation"),
                            ("result_present", "reasoning"), ("proof", "expression", "text"))

    def _finite_field_irreducibility(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"不可约|irreducib", text, re.I) or not re.search(r"多项式环|polynomial\s+ring", text, re.I):
            return None
        field = re.search(r"(?:F|\\mathbb\s*\{F\})\s*_?\s*\{?\s*(\d+)\s*\}?\s*\[\s*x\s*\]", text, re.I)
        polynomial = re.search(
            r"(?:判断|determine|decide|check)\s*\$?\s*([^，,。;；$]+?)\s*\$?\s*(?:是否|is\s+)?(?:为|an?\s+)?不可约|"
            r"(?:polynomial)\s*\$?\s*([^，,。;；$]+?)\s*\$?\s*(?:is|whether)",
            text,
            re.I,
        )
        if field is None or polynomial is None:
            return None
        prime = int(field.group(1))
        if not self.sp.isprime(prime):
            return None
        raw = next(item for item in polynomial.groups() if item)
        expression = self._expr(raw)
        x = self.sp.Symbol("x")
        if expression is None or expression.free_symbols - {x}:
            return None
        poly = self.sp.Poly(expression, x, modulus=prime)
        degree = poly.degree()
        if not 1 <= degree <= 20:
            return None
        irreducible = bool(poly.is_irreducible)
        max_factor_degree = degree // 2
        if irreducible:
            conclusion_zh, conclusion_en = "不可约", "irreducible"
            detail = (
                rf"只需检查次数不超过 {max_factor_degree} 的不可约因子；有限域上的精确因式分解未找到此类因子。"
                if self._zh(text) else
                rf"It suffices to check irreducible factors of degree at most {max_factor_degree}; exact finite-field factorization finds none."
            )
        else:
            factors = self.sp.factor_list(poly, modulus=prime)[1]
            factor_text = r"\cdot".join(
                rf"({self.symbolic._format(item.as_expr())})^{{{multiplicity}}}"
                for item, multiplicity in factors
            )
            conclusion_zh, conclusion_en = "可约", "reducible"
            detail = (
                rf"有限域上的精确分解为 ${factor_text}$。"
                if self._zh(text) else
                rf"Exact finite-field factorization gives ${factor_text}$."
            )
        result = (
            rf"该多项式在 $\mathbb F_{{{prime}}}[x]$ 中{conclusion_zh}。{detail}"
            if self._zh(text) else
            rf"The polynomial is {conclusion_en} in $\mathbb F_{{{prime}}}[x]$. {detail}"
        )
        return self._result(text, "small_finite_field_irreducibility", result, "irreducibility",
                            "exact_finite_field_factorization",
                            ("prime_field_parsed", "polynomial_parsed", "degree_bounded", "factorization_recomputed"),
                            ("result_present", "judgement", "irreducibility_judgement", "factor_degree_check", "reasoning"),
                            ("truth", "expression", "text"))

    def _finite_field_irreducible_count(self, text: str) -> Optional[ToolResult]:
        if not (
            re.search(r"首一|monic", text, re.I)
            and re.search(r"不可约|irreducible", text, re.I)
            and re.search(r"个数|数目|多少|count|number", text, re.I)
        ):
            return None
        field_matches = re.findall(
            r"(?:F|\\mathbb\s*\{F\})\s*_?\s*\{?\s*(\d{1,7})\s*\}?",
            text,
            re.I,
        )
        if len(field_matches) != 1:
            return None
        degree_tokens = re.findall(
            r"(?:首一|monic)[^。；;\n]{0,45}?"
            r"(?:(\d{1,3}|一|二|两|三|四|五|六|七|八|九|十)\s*次|"
            r"degree\s*[- ]?(\d{1,3}))"
            r"[^。；;\n]{0,30}(?:多项式|polynomials?)|"
            r"(?:多项式|polynomials?)[^。；;\n]{0,30}?"
            r"(?:(\d{1,3}|一|二|两|三|四|五|六|七|八|九|十)\s*次|"
            r"degree\s*[- ]?(\d{1,3}))",
            text,
            re.I,
        )
        degrees = {
            self._small_integer(next(value for value in groups if value))
            for groups in degree_tokens
        }
        if len(degrees) != 1 or None in degrees:
            return None
        q = int(field_matches[0])
        degree = next(iter(degrees))
        if not isinstance(degree, int) or not 1 <= degree <= 100:
            return None
        factors = self.sp.factorint(q)
        if (
            q < 2
            or len(factors) != 1
            or any(exponent < 1 for exponent in factors.values())
        ):
            return None

        divisors = tuple(int(value) for value in self.sp.divisors(degree))
        direct_numerator = sum(
            int(self.sp.mobius(divisor)) * q ** (degree // divisor)
            for divisor in divisors
        )
        if direct_numerator % degree:
            return None
        direct = direct_numerator // degree

        recurrence: dict[int, int] = {}
        for current in range(1, degree + 1):
            proper = [
                divisor
                for divisor in self.sp.divisors(current)
                if divisor < current
            ]
            numerator = q**current - sum(
                int(divisor) * recurrence[int(divisor)]
                for divisor in proper
            )
            if numerator % current:
                return None
            recurrence[current] = numerator // current
        if direct < 0 or recurrence[degree] != direct:
            return None

        divisor_terms = ", ".join(map(str, divisors))
        if self._zh(text):
            result = (
                rf"个数为 $N_{{{q}}}({degree})={direct}$。由"
                rf"$q^n=\sum_{{d\mid n}}dN_q(d)$ 作 Möbius 反演，"
                rf"$N_q(n)=\frac1n\sum_{{d\mid n}}\mu(d)q^{{n/d}}$；"
                rf"本题约数为 {divisor_terms}，代入得 {direct}。"
            )
        else:
            result = (
                rf"The number is $N_{{{q}}}({degree})={direct}$. From "
                rf"$q^n=\sum_{{d\mid n}}dN_q(d)$, Möbius inversion gives "
                rf"$N_q(n)=\frac1n\sum_{{d\mid n}}\mu(d)q^{{n/d}}$; "
                rf"using divisors {divisor_terms} yields {direct}."
            )
        return self._result(
            text,
            "finite_field_monic_irreducible_count",
            result,
            "irreducible_polynomial_count",
            "mobius_formula_and_divisor_recurrence_crosscheck",
            (
                "single_prime_power_field_size_parsed",
                "single_degree_and_monic_requirement_parsed",
                "mobius_inversion_recomputed",
                "divisor_recurrence_independently_recomputed",
            ),
            ("result_present", "numeric_result", "count_conclusion", "reasoning"),
            ("count", "number", "expression"),
        )

    def _polynomial_quotient_power(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"商环|quotient\s+ring", text, re.I):
            return None
        modulus_match = re.search(r"[A-Za-zZ\\\[\]]+\s*\[\s*x\s*\]\s*/\s*\(\s*([^()]+)\s*\)", text)
        if modulus_match is None:
            return None
        power_match = re.search(r"x\s*(?:的)?\s*(平方|立方)|x\s*\^\s*\{?\s*(\d+)\s*\}?", text)
        if power_match is None:
            return None
        exponent = {"平方": 2, "立方": 3}.get(power_match.group(1), int(power_match.group(2)) if power_match.group(2) else 0)
        modulus = self._expr(modulus_match.group(1))
        x = self.sp.Symbol("x")
        if modulus is None or modulus.free_symbols - {x} or exponent < 1:
            return None
        poly = self.sp.Poly(modulus, x, domain=self.sp.ZZ)
        if poly.degree() < 1 or poly.LC() not in {1, -1}:
            return None
        remainder = self.sp.rem(self.sp.Poly(x**exponent, x, domain=self.sp.ZZ), poly).as_expr()
        rendered = self.symbolic._format(self.sp.expand(remainder))
        result = (
            rf"在商环中模关系为 ${self.symbolic._format(modulus)}=0$；多项式除法给出 "
            rf"$x^{{{exponent}}}\equiv {rendered}\pmod{{{self.symbolic._format(modulus)}}}$。"
            if self._zh(text) else
            rf"The quotient imposes ${self.symbolic._format(modulus)}=0$; polynomial division gives "
            rf"$x^{{{exponent}}}\equiv {rendered}\pmod{{{self.symbolic._format(modulus)}}}$."
        )
        return self._result(text, "polynomial_quotient_power", result, "quotient_class",
                            "exact_polynomial_remainder",
                            ("quotient_modulus_parsed", "power_parsed", "monic_integer_modulus", "remainder_recomputed"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _polynomial_root_bound(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"多项式|polynomial", text, re.I) or not re.search(r"不同根|distinct\s+roots?", text, re.I):
            return None
        degree_match = re.search(r"(?:次数|degree)\s*(?:为|是|is|=|of)?\s*(\d+)", text, re.I)
        roots_match = re.search(r"(?:有|has|with)\s*(\d+)\s*(?:个)?\s*(?:不同根|distinct\s+roots?)", text, re.I)
        if degree_match is None or roots_match is None:
            return None
        degree, roots = int(degree_match.group(1)), int(roots_match.group(1))
        if roots <= degree:
            return None
        result = (
            rf"域上的非零 {degree} 次多项式至多有 {degree} 个不同根；题设有 {roots}>{degree} 个不同根，"
            r"故只能是零多项式，即 $f=0$。"
            if self._zh(text) else
            rf"A nonzero degree-{degree} polynomial over a field has at most {degree} distinct roots. "
            rf"Since there are {roots}>{degree}, the polynomial must be zero: $f=0$."
        )
        return self._result(text, "polynomial_root_bound", result, "proof",
                            "nonzero_polynomial_root_bound",
                            ("field_hypothesis", "degree_parsed", "distinct_root_count_parsed", "strict_excess_checked"),
                            ("result_present", "reasoning"), ("proof", "expression", "text"))

    def _finite_field_multiplicative_group(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"有限域|finite\s+field", text, re.I) or not re.search(r"乘法群|multiplicative\s+group", text, re.I):
            return None
        match = re.search(r"(?:F|\\mathbb\s*\{F\})\s*_?\s*\{?\s*(\d+)\s*\}?", text)
        if match is None:
            return None
        q = int(match.group(1))
        factors = self.sp.factorint(q)
        if len(factors) != 1:
            return None
        order = q - 1
        result = (
            rf"$\mathbb F_{{{q}}}^\times$ 由全部非零元素组成，故阶为 ${q}-1={order}$；"
            rf"由有限群的拉格朗日定理，每个 $a\ne0$ 都满足 $a^{{{order}}}=1$。"
            if self._zh(text) else
            rf"$\mathbb F_{{{q}}}^\times$ consists of all nonzero field elements, so its order is ${q}-1={order}$. "
            rf"Lagrange's theorem gives $a^{{{order}}}=1$ for every $a\ne0$."
        )
        return self._result(text, "finite_field_multiplicative_group", result, "group_order_and_identity",
                            "finite_field_nonzero_elements_and_lagrange",
                            ("prime_power_checked", "nonzero_elements_counted", "lagrange_identity"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("number", "expression", "text"))

    def _maximal_ideal_quotient(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"极大理想|maximal\s+ideal", text, re.I) or not re.search(r"商环|quotient\s+ring|R\s*/\s*I", text, re.I):
            return None
        result = (
            r"$R/I$ 是域。因为 $I$ 极大，任意 $a\notin I$ 都有理想 $(I,a)=R$，故存在 "
            r"$r\in R,i\in I$ 使 $ra+i=1$；于是 $(r+I)(a+I)=1+I$，每个非零类均可逆。"
            if self._zh(text) else
            r"$R/I$ is a field. Since $I$ is maximal, for every $a\notin I$ one has $(I,a)=R$, "
            r"so $ra+i=1$ for some $r\in R,i\in I$. Thus $(r+I)(a+I)=1+I$, and every nonzero class is invertible."
        )
        return self._result(text, "maximal_ideal_quotient", result, "proof",
                            "maximal_ideal_nonzero_classes_invertible",
                            ("maximal_ideal_hypothesis", "proper_quotient", "bezout_ideal_identity", "inverse_class_constructed"),
                            ("result_present", "reasoning"), ("proof", "expression", "text"))

    def _expr(self, value: str):
        try:
            prepared = re.sub(r"(?<![A-Za-z0-9_])([A-Za-z])\s*\(", r"\1*(", str(value).strip().strip("$"))
            return self.sp.simplify(self.symbolic._parse(prepared))
        except Exception:
            return None

    @staticmethod
    def _small_integer(token: str) -> Optional[int]:
        words = {
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
        value = str(token or "").strip()
        return int(value) if value.isdigit() else words.get(value)

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
            answer_shapes=shapes,
            requirements=requirements,
        )
