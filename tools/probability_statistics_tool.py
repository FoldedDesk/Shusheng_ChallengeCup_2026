"""Parameterized probability and statistics computations with local checks."""

from __future__ import annotations

import re
from statistics import NormalDist
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class ProbabilityStatisticsTool:
    """Answer only formula-complete requests whose parameters can be parsed."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if not text or self.sp is None:
            return []
        handlers = (
            self._conditional_two_dice,
            self._independent_standard_normal_sum,
            self._geometric_tail,
            self._discrete_moments,
            self._symmetric_walk_moments,
            self._poisson_increment,
            self._renewal_limit,
            self._z_critical,
            self._variance_preference,
            self._full_covariance_gls,
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

    def _conditional_two_dice(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"(?:两|2)次[^。.!?]{0,40}骰子|two\s+(?:fair\s+)?(?:\d+[- ]sided\s+)?dice",
            text,
            re.I,
        ):
            return None
        if not re.search(r"条件概率|given|conditional", text, re.I):
            return None
        sum_match = re.search(
            r"(?:点数)?和(?:为|等于)\s*\$?(\d+)|(?:sum|total)\s+(?:is|equals?|=)\s*\$?(\d+)",
            text,
            re.I,
        )
        first_match = re.search(
            r"第(?:一|1)次(?:为|等于)\s*\$?(\d+)|first\s+(?:die|roll|value)\s+(?:is|equals?|=)\s*\$?(\d+)",
            text,
            re.I,
        )
        if sum_match is None or first_match is None:
            return None
        total = int(self._group(sum_match))
        first = int(self._group(first_match))
        side_match = re.search(r"(\d+)\s*(?:面|[- ]sided)", text, re.I)
        sides = int(side_match.group(1)) if side_match else 6
        if not 2 <= sides <= 10_000:
            return None
        outcomes = [(a, total - a) for a in range(1, sides + 1) if 1 <= total - a <= sides]
        if not outcomes:
            return None
        favourable = int((first, total - first) in outcomes)
        value = self.symbolic._format(self.sp.Rational(favourable, len(outcomes)))
        result = (
            f"条件样本空间有 {len(outcomes)} 个等可能有序结果，其中满足第一次为 {first} 的有 "
            f"{favourable} 个，故条件概率为 ${value}$。"
            if self._zh(text) else
            f"The conditional sample space has {len(outcomes)} equally likely ordered outcomes; "
            f"{favourable} have first value {first}, so the probability is ${value}$."
        )
        return self._result(text, "two_dice_conditional_probability", result, "probability",
                            "finite_conditional_enumeration",
                            ("faces_parsed", "conditioned_outcomes_enumerated", "favourable_counted"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("probability", "number", "expression"))

    def _independent_standard_normal_sum(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"标准正态|standard\s+normal", text, re.I) or not re.search(r"独立|independent", text, re.I):
            return None
        pair = re.search(r"([A-Za-z])\s*[,，、]\s*([A-Za-z])[^。.!?]{0,90}(?:独立|independent)", text, re.I)
        if pair is None:
            return None
        left, right = pair.groups()
        if left.casefold() == right.casefold() or not re.search(
            rf"(?:{re.escape(left)}\s*\+\s*{re.escape(right)}|{re.escape(right)}\s*\+\s*{re.escape(left)})",
            text,
            re.I,
        ):
            return None
        total = f"{left}+{right}"
        result = (
            rf"根据独立性，两个正态变量的均值相加、方差相加，故 "
            rf"${total}\sim N(0,2)$，且 $\operatorname{{Var}}({total})=2$。"
            if self._zh(text) else
            rf"By independence, normal means and variances add, so ${total}\sim N(0,2)$ and "
            rf"$\operatorname{{Var}}({total})=2$."
        )
        return self._result(text, "independent_standard_normal_sum", result, "distribution",
                            "independent_normal_parameter_addition",
                            ("two_variables_parsed", "independence_present", "means_added", "variances_added"),
                            ("result_present", "reasoning", "support_anchor_1", "target_x"),
                            ("expression", "text"))

    def _geometric_tail(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"首次(?:成功|正面)|直到首次|geometric|first\s+(?:success|head)", text, re.I):
            return None
        tail = re.search(r"(?:次数|等待时间|[TN])\s*(?:大于|>)\s*\$?(\d+)|(?:trials|tosses|waiting\s+time|[TN])\s*>\s*\$?(\d+)", text, re.I)
        if tail is None:
            return None
        steps = int(self._group(tail))
        probability = re.search(
            r"(?:成功|正面)(?:概率|的概率)?(?:为|=)\s*\$?([0-9]+(?:\s*/\s*[0-9]+)?|0?\.\d+)|"
            r"(?:success|head)\s+probability\s*(?:(?:is|=)\s*)?\$?([0-9]+(?:\s*/\s*[0-9]+)?|0?\.\d+)",
            text,
            re.I,
        )
        if probability:
            p = self.symbolic._parse(self._group(probability).replace(" ", ""))
        elif re.search(r"公平(?:硬币)?|fair\s+coin", text, re.I):
            p = self.sp.Rational(1, 2)
        else:
            return None
        if p.is_real is not True or not bool(0 < p <= 1):
            return None
        value = self.symbolic._format(self.sp.simplify((1 - p) ** steps))
        result = (
            rf"在几何分布中，$T>{steps}$ 等价于前 {steps} 次均失败，故 "
            rf"$P(T>{steps})=(1-p)^{{{steps}}}={value}$。"
            if self._zh(text) else
            rf"For the geometric distribution, $T>{steps}$ means that the first {steps} trials all fail, so "
            rf"$P(T>{steps})=(1-p)^{{{steps}}}={value}$."
        )
        return self._result(text, "geometric_waiting_tail", result, "probability",
                            "independent_failure_product",
                            ("success_probability_parsed", "strict_tail_index_checked", "power_recomputed"),
                            ("result_present", "numeric_result", "reasoning", "support_anchor_1"),
                            ("probability", "number", "expression"))

    def _discrete_moments(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"取值|values?", text, re.I) or not re.search(r"概率分别|probabilit", text, re.I):
            return None
        if not re.search(r"E\s*\[|Var|期望|方差", text, re.I):
            return None
        matched = re.search(
            r"(?:取值|values?(?:\s+are)?)\s*[:：=]?\s*(.+?)\s*"
            r"(?:且\s*概率分别(?:为)?|with\s+(?:respective\s+)?probabilities?|probabilities\s+(?:are|=))"
            r"\s*[:：=]?\s*(.+?)(?=(?:\.|。|,|，)\s*(?:Find|求)|[。;；\n]|$)",
            text,
            re.I,
        )
        if matched is None:
            return None
        values = self._number_list(matched.group(1))
        weights = self._number_list(matched.group(2))
        if not values or len(values) != len(weights) or len(values) > 100:
            return None
        if any(weight.is_real is not True or bool(weight < 0) for weight in weights) or self.sp.simplify(sum(weights) - 1) != 0:
            return None
        mean = self.sp.simplify(sum(value * weight for value, weight in zip(values, weights)))
        variance = self.sp.simplify(sum((value - mean) ** 2 * weight for value, weight in zip(values, weights)))
        mean_text = self.symbolic._format(mean)
        variance_text = self.symbolic._format(variance)
        result = (
            rf"$E[X]={mean_text}$，$\operatorname{{Var}}(X)={variance_text}$；二者分别由 "
            r"$\sum x_ip_i$ 与 $\sum(x_i-E[X])^2p_i$ 逐项计算。"
            if self._zh(text) else
            rf"$E[X]={mean_text}$ and $\operatorname{{Var}}(X)={variance_text}$, recomputed "
            r"from $\sum x_i p_i$ and $\sum(x_i-E[X])^2p_i$."
        )
        return self._result(text, "finite_discrete_moments", result, "moments",
                            "finite_support_exact_moments",
                            ("support_parsed", "mass_normalized", "first_moment", "centered_second_moment"),
                            ("result_present", "numeric_result", "reasoning"),
                            ("expression", "number", "text"))

    def _symmetric_walk_moments(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"简单对称随机游走|simple\s+symmetric\s+random\s+walk", text, re.I):
            return None
        step = re.search(r"S_?\{?(\d+)\}?|S\s*\(\s*(\d+)\s*\)", text)
        if step is None or not re.search(r"E\s*\[|Var|期望|方差", text, re.I):
            return None
        n = int(self._group(step))
        start_match = re.search(
            r"从\s*(-?\d+)\s*出发|start(?:s|ing)?\s+(?:from|at)\s*(-?\d+)",
            text,
            re.I,
        )
        start = int(self._group(start_match)) if start_match else 0
        result = (
            rf"$E[S_{{{n}}}]={start}$，$\operatorname{{Var}}(S_{{{n}}})={n}$；这是 {n} 个独立、均值0、方差1的增量之和。"
            if self._zh(text) else
            rf"$E[S_{{{n}}}]={start}$ and $\operatorname{{Var}}(S_{{{n}}})={n}$ because the displacement is a sum of "
            f"{n} independent increments of mean 0 and variance 1."
        )
        return self._result(text, "symmetric_random_walk_moments", result, "moments",
                            "independent_increment_moment_addition",
                            ("step_index_parsed", "start_parsed", "mean_added", "variance_added"),
                            ("result_present", "reasoning", "support_anchor_1"),
                            ("expression", "number", "text"))

    def _poisson_increment(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"泊松过程|Poisson\s+process", text, re.I):
            return None
        increment = re.search(r"N\s*\(\s*([^()]+?)\s*\)\s*-\s*N\s*\(\s*([^()]+?)\s*\)", text, re.I)
        rate_match = re.search(
            r"(?:强度|速率|rate|intensity)(?:为|is|=)?\s*\$?"
            r"(\\lambda|λ|[A-Za-z]|[0-9]+(?:/[0-9]+)?|0?\.\d+)",
            text,
            re.I,
        )
        if increment is None or rate_match is None:
            return None
        upper = self.symbolic._parse(increment.group(1))
        lower = self.symbolic._parse(increment.group(2))
        if upper.free_symbols or lower.free_symbols or bool(upper < lower):
            return None
        interval = self.sp.simplify(upper - lower)
        raw_rate = rate_match.group(1)
        if raw_rate in {"λ", r"\lambda"}:
            parameter = (
                r"\lambda"
                if interval == 1
                else self.symbolic._format(interval) + r"\lambda"
            )
        else:
            rate = self.symbolic._parse(raw_rate)
            parameter = self.symbolic._format(self.sp.simplify(rate * interval))
        upper_text = self.symbolic._format(upper)
        lower_text = self.symbolic._format(lower)
        result = (
            rf"由平稳独立增量，$N({upper_text})-N({lower_text})\sim\operatorname{{Poisson}}({parameter})$；给定过去计数不改变该分布。"
            if self._zh(text) else
            rf"By stationary independent increments, $N({upper_text})-N({lower_text})\sim\operatorname{{Poisson}}({parameter})$; "
            "conditioning on the past count does not change this distribution."
        )
        return self._result(text, "poisson_independent_increment", result, "distribution",
                            "poisson_stationary_independent_increment",
                            ("endpoints_parsed", "rate_parsed", "interval_scaled"),
                            ("result_present", "reasoning", "support_anchor_1"),
                            ("expression", "text"))

    def _renewal_limit(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"更新过程|renewal\s+process", text, re.I) or not re.search(r"N\s*\(\s*t\s*\)\s*/\s*t", text, re.I):
            return None
        mean_match = re.search(r"(?:间隔|interarrival|renewal\s+interval)[^。.!?]{0,35}(?:均值|mean)(?:为|is|=)?\s*\$?([0-9]+(?:/[0-9]+)?|0?\.\d+)", text, re.I)
        if mean_match is None:
            return None
        mean = self.symbolic._parse(mean_match.group(1))
        if mean.is_real is not True or not bool(mean > 0):
            return None
        value = self.symbolic._format(self.sp.simplify(1 / mean))
        result = (
            rf"由更新强大数律，$\lim_{{t\to\infty}}N(t)/t=1/E[X_1]={value}$（几乎处处）。"
            if self._zh(text) else
            rf"By the renewal strong law, $\lim_{{t\to\infty}}N(t)/t=1/E[X_1]={value}$ almost surely."
        )
        return self._result(text, "renewal_strong_law", result, "limit",
                            "renewal_rate_reciprocal",
                            ("positive_mean_parsed", "reciprocal_recomputed"),
                            ("result_present", "reasoning"),
                            ("expression", "number", "text"))

    def _z_critical(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"双侧.*Z|two[- ]sided\s+Z", text, re.I):
            return None
        alpha_match = re.search(r"(?:显著性水平|significance\s+level|alpha|α)(?:为|is|=)?\s*\$?(0?\.\d+)", text, re.I)
        if alpha_match is None:
            return None
        alpha = float(alpha_match.group(1))
        if not 0 < alpha < 1:
            return None
        z = NormalDist().inv_cdf(1 - alpha / 2)
        rendered = f"{z:.4f}".rstrip("0").rstrip(".")
        result = (
            rf"根据标准正态分位数，临界值为 "
            rf"$\pm z_{{1-\alpha/2}}\approx\pm {rendered}$，拒绝域为 $|Z|>{rendered}$。"
            if self._zh(text) else
            rf"By the standard normal quantile, the critical values are "
            rf"$\pm z_{{1-\alpha/2}}\approx\pm {rendered}$; reject when $|Z|>{rendered}$."
        )
        return self._result(text, "two_sided_z_critical_value", result, "critical_region",
                            "standard_normal_inverse_cdf",
                            ("alpha_parsed", "tails_split", "quantile_recomputed"),
                            ("result_present", "reasoning", "support_anchor_1"),
                            ("expression", "interval", "text"))

    def _variance_preference(self, text: str) -> Optional[ToolResult]:
        match = re.search(r"Var\s*\(\s*([A-Za-z](?:_\{?\d+\}?)?)\s*\)\s*<\s*Var\s*\(\s*([A-Za-z](?:_\{?\d+\}?)?)\s*\)", text, re.I)
        if match is None or not re.search(r"无偏|unbiased", text, re.I):
            return None
        preferred, other = match.groups()
        result = (
            f"应优先选择 ${preferred}$；两者无偏时，它的方差更小，因而均方误差更小、效率更高。"
            if self._zh(text) else
            f"Prefer ${preferred}$: among unbiased estimators it has smaller variance, hence smaller MSE and higher efficiency than ${other}$."
        )
        return self._result(text, "unbiased_estimator_variance_choice", result, "estimator_choice",
                            "unbiased_mse_equals_variance",
                            ("unbiasedness_present", "variance_order_parsed", "mse_order_inferred"),
                            ("result_present", "reasoning"),
                            ("expression", "text", "choice"))

    def _full_covariance_gls(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"广义最小二乘|\bGLS\b|generalized\s+least\s+squares",
            text,
            re.I,
        ):
            return None
        design = self._labelled_matrix(text, r"X")
        covariance = self._labelled_matrix(text, r"(?:\\Sigma|Σ|Sigma)")
        if design is None or covariance is None:
            return None
        observations = self._labelled_vector(text, "y", design.rows)
        if observations is None:
            return None
        if (
            not 2 <= design.rows <= 12
            or not 1 <= design.cols <= 8
            or covariance.shape != (design.rows, design.rows)
            or covariance != covariance.T
            or design.rank() != design.cols
        ):
            return None
        for size in range(1, covariance.rows + 1):
            determinant = self.sp.simplify(covariance[:size, :size].det())
            if determinant.is_positive is not True:
                return None
        try:
            inverse = self.sp.simplify(covariance.inv())
        except Exception:
            return None
        if (
            self.sp.simplify(inverse * covariance) != self.sp.eye(covariance.rows)
            or inverse != inverse.T
        ):
            return None
        normal_matrix = self.sp.simplify(design.T * inverse * design)
        normal_rhs = self.sp.simplify(design.T * inverse * observations)
        if normal_matrix.det() == 0:
            return None
        estimate = self.sp.simplify(normal_matrix.inv() * normal_rhs)
        if self.sp.simplify(normal_matrix * estimate - normal_rhs) != self.sp.zeros(
            design.cols,
            1,
        ):
            return None
        for size in range(1, normal_matrix.rows + 1):
            if self.sp.simplify(normal_matrix[:size, :size].det()).is_positive is not True:
                return None

        inverse_text = self.symbolic._format(inverse)
        normal_text = self.symbolic._format(normal_matrix)
        rhs_text = self.symbolic._format(normal_rhs)
        estimate_text = self.symbolic._format(estimate)
        if self._zh(text):
            result = (
                rf"$\Sigma^{{-1}}={inverse_text}$，加权正规方程为 "
                rf"$({normal_text})\widehat\beta={rhs_text}$，即 "
                rf"$X^T\Sigma^{{-1}}X={normal_text}$、"
                rf"$X^T\Sigma^{{-1}}y={rhs_text}$；故 "
                rf"$\widehat\beta_{{GLS}}={estimate_text}$。"
            )
            support = (
                r"已精确核对 $\Sigma^{-1}\Sigma=I$、协方差正定、"
                r"$X$ 满列秩，并将估计量代回加权正规方程；"
                r"$X^T\Sigma^{-1}X$ 正定，故该解是唯一极小点。"
            )
        else:
            result = (
                rf"$\Sigma^{{-1}}={inverse_text}$. The weighted normal equation is "
                rf"$({normal_text})\widehat\beta={rhs_text}$, namely "
                rf"$X^T\Sigma^{{-1}}X={normal_text}$ and "
                rf"$X^T\Sigma^{{-1}}y={rhs_text}$; hence "
                rf"$\widehat\beta_{{GLS}}={estimate_text}$."
            )
            support = (
                r"The identities $\Sigma^{-1}\Sigma=I$, positive definiteness, "
                r"and full column rank of $X$ were checked exactly. The estimate "
                r"satisfies the weighted normal equation, whose matrix is positive "
                "definite, so it is the unique minimizer."
            )
        return self._result(
            text,
            "full_covariance_gls_estimate",
            result,
            "weighted_normal_equation_and_estimate",
            "exact_spd_inverse_normal_equation_and_residual",
            (
                "single_design_covariance_and_response_parsed",
                "covariance_symmetric_positive_definite",
                "design_full_column_rank",
                "covariance_inverse_identity_recomputed",
                "weighted_normal_equation_recomputed",
                "estimate_residual_zero",
            ),
            (
                "result_present",
                "reasoning",
                "normal_equation",
                "coefficient_estimate",
            ),
            ("expression", "matrix"),
            support=support,
        )

    def _labelled_matrix(self, text: str, label: str):
        matches = list(re.finditer(
            rf"(?<![A-Za-z]){label}\s*=\s*"
            r"\\begin\{[pbvBV]?matrix\}(.+?)\\end\{[pbvBV]?matrix\}",
            text,
            re.I | re.DOTALL,
        ))
        if len(matches) != 1:
            return None
        rows = [
            row.strip()
            for row in re.split(r"\\\\", matches[0].group(1))
            if row.strip()
        ]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if (
            not cells
            or len(cells) > 12
            or len(cells[0]) > 12
            or any(len(row) != len(cells[0]) for row in cells)
        ):
            return None
        parsed = []
        for row in cells:
            parsed_row = []
            for cell in row:
                try:
                    value = self.sp.simplify(
                        self.symbolic._parse(
                            self.symbolic._latex_to_sympy(cell)
                        )
                    )
                except Exception:
                    return None
                if value.free_symbols or value.is_real is not True:
                    return None
                parsed_row.append(value)
            parsed.append(parsed_row)
        return self.sp.Matrix(parsed)

    def _labelled_vector(self, text: str, label: str, dimension: int):
        match = re.search(
            rf"(?<![A-Za-z]){re.escape(label)}\s*=\s*"
            r"\(\s*([^()\n]+)\s*\)\s*\^?\s*\{?T\}?",
            text,
            re.I,
        )
        if match is None:
            return None
        parts = [part.strip() for part in re.split(r"[,，]", match.group(1))]
        if len(parts) != dimension:
            return None
        parsed = []
        for part in parts:
            try:
                value = self.sp.simplify(
                    self.symbolic._parse(self.symbolic._latex_to_sympy(part))
                )
            except Exception:
                return None
            if value.free_symbols or value.is_real is not True:
                return None
            parsed.append(value)
        return self.sp.Matrix(parsed)

    def _number_list(self, value: str):
        value = re.sub(r"\\(?:frac|dfrac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
        tokens = [item.strip() for item in value.replace("，", ",").replace("、", ",").replace("$", "").split(",") if item.strip()]
        parsed = []
        for token in tokens:
            found = re.search(r"[-+]?(?:\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?|\.\d+)\s*$", token)
            if found is None:
                return []
            parsed.append(self.symbolic._parse(found.group(0).replace(" ", "")))
        return parsed

    @staticmethod
    def _group(match: re.Match) -> str:
        return next(item for item in match.groups() if item is not None)

    @staticmethod
    def _zh(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def _result(problem: str, operation: str, result: str, result_kind: str,
                method: str, checks: tuple[str, ...], requirements: tuple[str, ...],
                shapes: tuple[str, ...], support: str = "") -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=support or result,
            requirements=requirements,
            answer_shapes=shapes,
        )
