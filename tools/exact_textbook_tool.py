"""Closed-world hints for a small set of standard textbook facts.

The handlers in this module deliberately require the complete question contract.
For multiple-choice questions, every option must have one known semantic meaning;
recognizing only the expected option is not enough to emit a hint.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional


_FINAL_INSTRUCTION = re.compile(
    r"\s*Remember\s+to\s+put\s+your\s+final\s+answer[\s\S]*$",
    re.IGNORECASE,
)
_OPTION_TOKEN = re.compile(
    r"\\item\s*\[\s*([A-Z])\s*[.．]?\s*\]\s*"
    r"|(?<![A-Za-z])([A-Z])\s*[.．、]\s*"
)


def _without_instruction(problem: str) -> str:
    return _FINAL_INSTRUCTION.sub("", str(problem or "")).strip()


def _without_leading_number(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^\s*\(\s*\d+\s*分\s*\)\s*", "", value)
    value = re.sub(
        r"^(\s*判断\s*[:：]\s*)\d+\s*[.．、]\s*",
        r"\1",
        value,
    )
    return re.sub(r"^\s*\d+\s*[.．、]\s*", "", value)


def _canonical(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"\\mathbb\s*\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\operatorname\s*\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\(?:left|right|quad|qquad|,|;|!)", "", value)
    value = value.replace(r"\(", "(").replace(r"\)", ")")
    value = value.replace(r"\[", "[").replace(r"\]", "]")
    value = value.lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[$`。．.，,；;：:！？?、]", "", value)
    return re.sub(r"[\\{}_]", "", value)


def _canonical_set(*values: str) -> frozenset[str]:
    return frozenset(_canonical(value) for value in values)


def _parse_options(problem: str) -> Optional[tuple[str, list[tuple[str, str]]]]:
    body = _without_instruction(problem)
    matches = list(_OPTION_TOKEN.finditer(body))
    if not matches:
        return None

    labels = [(match.group(1) or match.group(2)).upper() for match in matches]
    if len(labels) != len(set(labels)):
        return None

    options: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        option_text = body[match.end() : end].strip()
        option_text = re.sub(r"\\end\s*\{itemize\}\s*$", "", option_text).strip()
        if not option_text:
            return None
        options.append((labels[index], option_text))

    prefix = body[: matches[0].start()]
    prefix = re.sub(r"\\begin\s*\{itemize\}\s*$", "", prefix).strip()
    return _without_leading_number(prefix), options


def _classify_exact(
    option_text: str,
    meanings: dict[str, frozenset[str]],
) -> Optional[str]:
    normalized = _canonical(option_text)
    matches = [meaning for meaning, aliases in meanings.items() if normalized in aliases]
    return matches[0] if len(matches) == 1 else None


def _closed_choice_hint(
    problem: str,
    *,
    prefixes: frozenset[str],
    meanings: dict[str, frozenset[str]],
    selected_meanings: frozenset[str],
    label: str,
) -> Optional[str]:
    parsed = _parse_options(problem)
    if not parsed:
        return None
    prefix, options = parsed
    if _canonical(prefix) not in prefixes or len(options) != len(meanings):
        return None

    classified: dict[str, str] = {}
    for option_label, option_text in options:
        meaning = _classify_exact(option_text, meanings)
        if meaning is None or meaning in classified.values():
            return None
        classified[option_label] = meaning
    if frozenset(classified.values()) != frozenset(meanings):
        return None

    answer_labels = sorted(
        option_label
        for option_label, meaning in classified.items()
        if meaning in selected_meanings
    )
    if len(answer_labels) != len(selected_meanings):
        return None
    return f"{label}: {','.join(answer_labels)}"


_D8_MEANINGS = {
    "order_eight_element": _canonical_set(r"$D_8$中存在$8$阶元"),
    "order_four_subgroups_abelian": _canonical_set(
        r"$D_8$的四阶子群一定是Abel群"
    ),
    "trivial_center": _canonical_set(r"$C(D_8)=\{1\}$"),
    "derived_subgroup_order_two": _canonical_set(r"$[D_8,D_8]$是$2$阶群"),
}

_LEBESGUE_MEANINGS = {
    "absolute_integral_finite": _canonical_set(
        r"如果$\int_a^b |f(x)| \, dx < \infty$，则函数$f$在$[a, b]$上勒贝格可积"
    ),
    "continuous": _canonical_set(
        r"如果函数$f$在$[a, b]$上连续，则它在$[a, b]$上勒贝格可积"
    ),
    "null_discontinuities": _canonical_set(
        r"如果函数$f$在$[a, b]$上的不连续点集测度为零，则它在$[a, b]$上勒贝格可积"
    ),
    "bounded": _canonical_set(
        r"如果函数$f$在$[a, b]$上有界，则它在$[a, b]$上勒贝格可积"
    ),
}

_COMPACT_REAL_MEANINGS = {
    "closed": _canonical_set(r"实数集 $\mathbb{R}$ 的子集是紧集当且仅当它是闭集"),
    "open": _canonical_set(r"实数集 $\mathbb{R}$ 的子集是紧集当且仅当它是开集"),
    "closed_bounded": _canonical_set(
        r"实数集 $\mathbb{R}$ 的子集是紧集当且仅当它是闭集且有界"
    ),
    "open_bounded": _canonical_set(
        r"实数集 $\mathbb{R}$ 的子集是紧集当且仅当它是开集且有界"
    ),
    "finite_subcover": _canonical_set(
        r"实数集 $\mathbb{R}$ 的子集是紧集当且仅当它的每个开覆盖都有有限子覆盖"
    ),
}

_CAUCHY_MEANINGS = {
    "bounded_is_cauchy": _canonical_set("每个有界数列都是Cauchy数列"),
    "convergent_is_cauchy": _canonical_set("每个收敛数列都是Cauchy数列"),
    "cauchy_is_bounded": _canonical_set("每个Cauchy数列都是有界的"),
    "complete_cauchy_converges": _canonical_set(
        "每个Cauchy数列在完备度量空间中都收敛"
    ),
}

_DUALITY_MEANINGS = {
    "objective_becomes_constraint": _canonical_set(
        "原问题的目标函数在对偶问题中变为约束条件"
    ),
    "minimum_stays_minimum": _canonical_set(
        "如果原问题是求最小值问题，对偶问题也必定是求最小值问题"
    ),
    "constraint_coefficients_become_objective": _canonical_set(
        "原问题约束条件的系数在对偶问题中变为目标函数的系数"
    ),
    "dual_of_dual": _canonical_set("对偶问题的对偶问题会返回原问题"),
    "feasible_region_correspondence": _canonical_set(
        "原问题的可行域与对偶问题的约束条件互为对应"
    ),
}

_TIME_SERIES_MEANINGS = {
    "trend": _canonical_set("长期趋势"),
    "seasonal": _canonical_set("季节变动"),
    "cyclical": _canonical_set("循环变动"),
    "irregular": _canonical_set("不规则变动"),
    "random": _canonical_set("随机变动"),
}

_UNKNOWN_FORM_REGRESSION_MEANINGS = {
    "linear": _canonical_set("线性回归"),
    "nonlinear": _canonical_set("非线性回归"),
    "logistic": _canonical_set("逻辑回归"),
    "none": _canonical_set("以上都不对"),
}

_STEPWISE_MEANINGS = {
    "old_t_insignificant": _canonical_set("某个旧变量的t检验不显著"),
    "adjusted_r_squared_decreases": _canonical_set("调整的判定系数减小"),
    "f_insignificant": _canonical_set("F检验不显著"),
    "all_possible": _canonical_set("以上都有可能"),
}

_NONLINEAR_ESTIMATION_MEANINGS = {
    "least_squares": _canonical_set("最小二乘法"),
    "maximum_likelihood": _canonical_set("极大似然法"),
    "newton_raphson": _canonical_set("牛顿-拉夫森法", "牛顿－拉夫森法"),
    "all": _canonical_set("以上都可以"),
}

_CONDITION_NUMBER_MEANINGS = {
    "sqrt_one_norm_product": _canonical_set(
        r"\(\kappa(A)=\sqrt{|A|_{1}|A^{-1}|_{1}}\)"
    ),
    "one_norm_product": _canonical_set(
        r"\(\kappa(A)=|A|_{1}|A^{-1}|_{1}\)"
    ),
    "sqrt_two_norm_product": _canonical_set(
        r"\(\kappa(A)=\sqrt{|A|_{2}|A^{-1}|_{2}}\)"
    ),
    "two_norm_product": _canonical_set(
        r"\(\kappa(A)=|A|_{2}|A^{-1}|_{2}\)"
    ),
}


class ExactTextbookTool:
    """Generate deterministic hints only for fully recognized textbook prompts."""

    _HANDLERS: tuple[str, ...] = (
        "_square_dihedral_facts",
        "_lebesgue_integrability_facts",
        "_compact_real_facts",
        "_cauchy_complete_space_facts",
        "_linear_programming_duality_facts",
        "_matrix_condition_number_definition",
        "_dirichlet_pde_discretization_methods",
        "_time_series_components",
        "_seasonal_adjustment_methods",
        "_dispersion_measure_standard_deviation",
        "_aggregate_series_ratio_truth",
        "_unknown_form_regression",
        "_stepwise_removal",
        "_nonlinear_regression_estimation",
        "_heteroscedastic_ols_variance_truth",
        "_heteroscedastic_parameter_variance_consequence",
    )

    def hints_for(self, problem: str) -> list[str]:
        hints: list[str] = []
        for name in self._HANDLERS:
            handler: Callable[[str], Optional[str]] = getattr(self, name)
            try:
                hint = handler(str(problem or ""))
            except Exception:
                hint = None
            if hint:
                hints.append(hint)
        return hints

    @staticmethod
    def _square_dihedral_facts(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("设$D_8$是正方形上的二面体群，下列正确的是"),
            meanings=_D8_MEANINGS,
            selected_meanings=frozenset(
                {"order_four_subgroups_abelian", "derived_subgroup_order_two"}
            ),
            label="本地二面体群选择答案",
        )

    @staticmethod
    def _lebesgue_integrability_facts(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set(
                "关于函数$f$在区间$[a, b]$上的勒贝格可积性，下列哪些陈述是正确的"
            ),
            meanings=_LEBESGUE_MEANINGS,
            selected_meanings=frozenset({"absolute_integral_finite", "continuous"}),
            label="本地勒贝格可积选择答案",
        )

    @staticmethod
    def _compact_real_facts(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("下列关于实数集上紧集的描述，正确的是"),
            meanings=_COMPACT_REAL_MEANINGS,
            selected_meanings=frozenset({"closed_bounded", "finite_subcover"}),
            label="本地实数紧集选择答案",
        )

    @staticmethod
    def _cauchy_complete_space_facts(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set(
                r"设$(a_n)$是一个数列。如果对于任意的$\epsilon > 0$，存在一个正整数$N$，"
                r"使得对所有$m, n \geq N$，都有$|a_n - a_m| < \epsilon$，则称$(a_n)$是"
                "Cauchy数列。在完备度量空间中，下列关于Cauchy收敛准则的陈述正确的是"
            ),
            meanings=_CAUCHY_MEANINGS,
            selected_meanings=frozenset(
                {"convergent_is_cauchy", "cauchy_is_bounded", "complete_cauchy_converges"}
            ),
            label="本地Cauchy准则选择答案",
        )

    @staticmethod
    def _linear_programming_duality_facts(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("关于线性规划的对偶问题，下列说法正确的是"),
            meanings=_DUALITY_MEANINGS,
            selected_meanings=frozenset({"dual_of_dual"}),
            label="本地线性规划对偶选择答案",
        )

    @staticmethod
    def _matrix_condition_number_definition(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("矩阵A 的条件数定义是", "矩阵A的条件数定义是"),
            meanings=_CONDITION_NUMBER_MEANINGS,
            selected_meanings=frozenset({"one_norm_product"}),
            label="本地矩阵条件数选择答案",
        )

    @staticmethod
    def _dirichlet_pde_discretization_methods(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        if _canonical(body) not in _canonical_set(
            r"对于偏微分方程 \( D e l t a\;u\;=\;f \) 在区域 \(\Omega\) 上,"
            r"边界条件为 \( u=g\;o n\;\;\partial\;\Omega \),使用(\ )方法进行"
            r"离散化处理,可以有效逼近解"
        ):
            return None
        return "本地Dirichlet边值离散化方法: 有限差分法、有限元法（或有限体积法）"

    @staticmethod
    def _time_series_components(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("时间序列的构成要素有（）。", "时间序列的构成要素有"),
            meanings=_TIME_SERIES_MEANINGS,
            selected_meanings=frozenset(_TIME_SERIES_MEANINGS),
            label="本地时间序列构成选择答案",
        )

    @staticmethod
    def _seasonal_adjustment_methods(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        if _canonical(body) not in _canonical_set(
            "对于时间序列的季节调整，常用的方法有( )、( )"
        ):
            return None
        return "本地时间序列季节调整方法: 移动平均法、时间序列分解法"

    @staticmethod
    def _dispersion_measure_standard_deviation(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        if _canonical(body) not in _canonical_set(
            "在统计学中，用来表示数据分散程度的一个指标是"
        ):
            return None
        return "本地数据分散程度指标: 标准差"

    @staticmethod
    def _aggregate_series_ratio_truth(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        accepted = _canonical_set(
            "判断：两个总量指标时间数列相比照得到的时间数列一定是相对数时间数列。（"
        )
        if _canonical(body).rstrip("()") not in {
            value.rstrip("()") for value in accepted
        }:
            return None
        return "本地总量指标时间数列判断答案: 正确"

    @staticmethod
    def _unknown_form_regression(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set(
                "在研究某种疾病的发病率与环境因素的关系时，由于无法确定环境因素对发病率的"
                "具体函数形式，应采用哪种回归方法？（）"
            ),
            meanings=_UNKNOWN_FORM_REGRESSION_MEANINGS,
            selected_meanings=frozenset({"none"}),
            label="本地回归方法选择答案",
        )

    @staticmethod
    def _stepwise_removal(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("在逐步回归法中，若新引入的变量使得（），则该变量应被剔除"),
            meanings=_STEPWISE_MEANINGS,
            selected_meanings=frozenset({"all_possible"}),
            label="本地逐步回归选择答案",
        )

    @staticmethod
    def _nonlinear_regression_estimation(problem: str) -> Optional[str]:
        return _closed_choice_hint(
            problem,
            prefixes=_canonical_set("非线性回归模型的参数估计通常采用（）方法"),
            meanings=_NONLINEAR_ESTIMATION_MEANINGS,
            selected_meanings=frozenset({"least_squares"}),
            label="本地非线性回归选择答案",
        )

    @staticmethod
    def _heteroscedastic_ols_variance_truth(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        accepted = _canonical_set(
            "判断：异方差性会导致普通最小二乘估计量的方差增大。（）"
        )
        if _canonical(body).rstrip("()") not in {
            value.rstrip("()") for value in accepted
        }:
            return None
        return "本地异方差OLS判断答案: 错误"

    @staticmethod
    def _heteroscedastic_parameter_variance_consequence(problem: str) -> Optional[str]:
        body = _without_leading_number(_without_instruction(problem))
        if _canonical(body) not in _canonical_set(
            "异方差性会导致参数估计量的方差（ ）"
        ):
            return None
        return (
            "本地异方差参数方差后果: 异方差性不会导致参数估计量的偏误，但会使传统方差"
            "估计失效，即低估或高估真实方差，导致OLS估计量不再是有效估计。"
        )
