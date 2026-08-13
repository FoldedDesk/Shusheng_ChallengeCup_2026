import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


FINAL = r"Remember to put your final answer within \boxed{}."
MOMENTS = (
    "设随机变量X取值1,2,3且概率分别为1/2,1/3,1/6，求E[X]与Var(X)。\n"
    + FINAL
)
Z_TEST = (
    "在显著性水平0.05的双侧Z检验中，写出拒绝域临界值并说明标准正态分位数。\n"
    + FINAL
)
GEOMETRIC_TAIL = "公平硬币持续抛掷直到首次正面，求抛掷次数大于3的概率并说明几何分布。"
POISSON_INCREMENT = "泊松过程N(t)强度为λ，求给定N(2)=3时N(3)-N(2)的条件分布。"
DICE_CONDITIONAL = (
    "连续投掷一枚公平六面骰子两次。已知两次点数之和等于7，"
    "列出条件样本空间，并求第一次点数为2的条件概率。"
)
BERNOULLI_MOMENT = (
    "设X服从Bernoulli(p)分布，计算E[(X-p)^2]并将其识别为方差。"
)
CAUCHY_FISHER = (
    r"设 $X_1,\ldots,X_n$ 独立同分布，密度为 "
    r"$f(x;\theta)=\{\pi[1+(x-\theta)^2]\}^{-1}$。"
    r"求样本关于位置参数 $\theta$ 的 Fisher 信息。"
)
WALD = (
    r"估计量 $\widehat\beta=(2,-1)^{\mathsf T}$ 的协方差矩阵为 "
    r"$\begin{pmatrix}0.25&0.10\\0.10&0.36\end{pmatrix}$。"
    r"检验线性约束 $H_0:\beta_1+\beta_2=0$，求一维 Wald 卡方统计量。"
)
GLS = (
    r"广义最小二乘模型中 $X=\begin{pmatrix}1&0\\1&1\\1&2\end{pmatrix}$、"
    r"$y=(1,2,2)^{\mathsf T}$，误差协方差矩阵与 "
    r"$\Omega=\operatorname{diag}(1,2,1)$ 成比例。求 GLS 估计 $\widehat\beta$。"
)
VARIANCE_CI = (
    r"来自正态总体 $N(\mu,\sigma^2)$ 的样本量为 $10$，且 "
    r"$\sum_{i=1}^{10}(X_i-\bar X)^2=18$。已知 "
    r"$\chi^2_{0.95,9}=16.919$、$\chi^2_{0.05,9}=3.325$，"
    r"求 $\sigma^2$ 的双侧 $90\%$ 置信区间。"
)


def _operation(problem: str, name: str):
    return [result for result in SympyTool().results_for(problem) if result.operation == name]


@pytest.mark.parametrize(
    "problem, operation, expected, method",
    [
        (MOMENTS, "finite_discrete_moments", "E[X]=5/3，Var(X)=5/9", "exact_finite_probability_sum"),
        (Z_TEST, "two_sided_z_rejection", "|Z|>1.96", "standard_normal_quantile"),
        (GEOMETRIC_TAIL, "fair_coin_geometric_tail", r"\frac{1}{8}", "geometric_tail_identity"),
        (
            POISSON_INCREMENT,
            "poisson_process_increment",
            r"N(3)-N(2)\mid N(2)=3\sim\operatorname{Poisson}(\lambda)",
            "poisson_independent_increment_law",
        ),
        (
            DICE_CONDITIONAL,
            "fair_dice_conditional_probability",
            r"\Omega_{S=7}=\{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\},\quad "
            r"P(D_1=2\mid S=7)=\frac{1}{6}",
            "finite_ordered_outcome_enumeration",
        ),
        (
            BERNOULLI_MOMENT,
            "bernoulli_centered_second_moment",
            r"E[(X-p)^2]=\operatorname{Var}(X)=p(1-p)",
            "bernoulli_two_point_exact_expectation",
        ),
        (
            CAUCHY_FISHER,
            "cauchy_location_fisher_information",
            r"I_n(\theta)=\frac{n}{2}",
            "exact_cauchy_score_integral",
        ),
        (
            WALD,
            "one_dimensional_wald_statistic",
            r"\frac{100}{81}",
            "exact_linear_contrast_quadratic_form",
        ),
        (
            GLS,
            "diagonal_gls_estimate",
            r"\begin{pmatrix}\frac{11}{10}\\\frac{1}{2}\end{pmatrix}",
            "exact_weighted_normal_equations",
        ),
        (
            VARIANCE_CI,
            "normal_variance_confidence_interval",
            r"\left[\frac{18}{16.919},\frac{18}{3.325}\right]\approx[1.064,5.414]",
            "exact_chi_square_interval_inversion",
        ),
    ],
)
def test_exact_statistics_results_are_certified_whole_answers(problem, operation, expected, method):
    results = _operation(problem, operation)

    assert len(results) == 1
    result = results[0]
    assert result.result == expected
    assert result.verified
    assert result.whole_answer_eligible
    assert result.certificate.method == method
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


def test_changed_probability_table_is_recomputed_exactly():
    changed = MOMENTS.replace("1,2,3", "0,1,2").replace("1/2,1/3,1/6", "1/4,1/2,1/4")

    assert _operation(changed, "finite_discrete_moments")[0].result == "E[X]=1，Var(X)=1/2"


def test_wald_and_gls_routes_recompute_changed_inputs_exactly():
    changed_wald = WALD.replace("(2,-1)", "(3,-1)").replace("0.36", "0.56")
    changed_gls = GLS.replace("y=(1,2,2)", "y=(1,2,3)")

    assert _operation(changed_wald, "one_dimensional_wald_statistic")[0].result == r"\frac{400}{101}"
    assert _operation(changed_gls, "diagonal_gls_estimate")[0].result == (
        r"\begin{pmatrix}1\\1\end{pmatrix}"
    )


@pytest.mark.parametrize(
    "sample_end, expected",
    [
        ("m", r"I_m(\theta)=\frac{m}{2}"),
        ("5", r"I_5(\theta)=\frac{5}{2}"),
        ("10", r"I_{10}(\theta)=5"),
    ],
)
def test_cauchy_fisher_route_uses_the_explicit_sample_size(sample_end, expected):
    changed = CAUCHY_FISHER.replace("X_n", f"X_{sample_end}")

    assert _operation(changed, "cauchy_location_fisher_information")[0].result == expected


def test_cauchy_fisher_route_rejects_missing_or_ambiguous_sample_size():
    missing = CAUCHY_FISHER.replace(r"X_1,\ldots,X_n", "若干个观测")
    ambiguous = CAUCHY_FISHER.replace(
        r"密度为 ",
        r"且另一组 $X_1,\ldots,X_m$ 也独立同分布，密度为 ",
    )

    assert _operation(missing, "cauchy_location_fisher_information") == []
    assert _operation(ambiguous, "cauchy_location_fisher_information") == []


@pytest.mark.parametrize(
    "changed",
    [
        CAUCHY_FISHER.replace("密度为 ", "另一组 $Y_1,\\ldots,Y_m$ 也独立同分布，密度为 "),
        CAUCHY_FISHER.replace("密度为 ", "截断 Cauchy 密度为 "),
        CAUCHY_FISHER.replace("密度为 ", "条件密度为 "),
    ],
)
def test_cauchy_fisher_route_rejects_multiple_or_modified_samples(changed):
    assert _operation(changed, "cauchy_location_fisher_information") == []


def test_cauchy_fisher_route_parses_an_english_symbolic_sample_size():
    problem = (
        r"Let $X_1,\ldots,X_m$ be i.i.d. with density "
        r"$f(x;\theta)=1/\{\pi(1+(x-\theta)^2)\}$. "
        r"Find the Fisher information in the sample for location parameter $\theta$."
    )

    result = _operation(problem, "cauchy_location_fisher_information")
    assert len(result) == 1
    assert result[0].result == r"I_m(\theta)=\frac{m}{2}"


@pytest.mark.parametrize(
    "changed",
    [
        CAUCHY_FISHER.replace("Fisher 信息", "Fisher 信息的倒数"),
        CAUCHY_FISHER.replace("样本关于", "每个观测关于"),
        CAUCHY_FISHER.replace("求样本", "只观察每个样本的符号，求样本"),
    ],
)
def test_cauchy_fisher_route_rejects_transformed_targets_and_observation_changes(changed):
    assert _operation(changed, "cauchy_location_fisher_information") == []


@pytest.mark.parametrize(
    "factor, expected",
    [
        ("2", r"\frac{50}{81}"),
        (r"\frac{1}{2}", r"\frac{200}{81}"),
    ],
)
def test_wald_route_applies_an_explicit_covariance_scalar(factor, expected):
    changed = WALD.replace("协方差矩阵为 ", f"协方差矩阵为 ${factor}").replace(
        r"$\begin{pmatrix}",
        r"\begin{pmatrix}",
    )

    assert _operation(changed, "one_dimensional_wald_statistic")[0].result == expected


@pytest.mark.parametrize("factor", [r"\sigma^2", "-2", "0"])
def test_wald_route_rejects_unknown_or_nonpositive_covariance_scalars(factor):
    changed = WALD.replace("协方差矩阵为 ", f"协方差矩阵为 ${factor}").replace(
        r"$\begin{pmatrix}",
        r"\begin{pmatrix}",
    )

    assert _operation(changed, "one_dimensional_wald_statistic") == []


@pytest.mark.parametrize(
    "changed",
    [
        WALD.replace("求一维 Wald 卡方统计量", "求带符号的一维 Wald z 统计量"),
        WALD.replace("的协方差矩阵为", r"满足 $\sqrt{n}(\widehat\beta-\beta)$ 的渐近协方差矩阵为"),
    ],
)
def test_wald_route_rejects_unsquared_or_asymptotically_scaled_targets(changed):
    assert _operation(changed, "one_dimensional_wald_statistic") == []


def test_gls_route_rejects_inverse_covariance_parameterization():
    changed = GLS.replace(r"$\Omega=", r"$\Omega^{-1}=")

    assert _operation(changed, "diagonal_gls_estimate") == []


@pytest.mark.parametrize(
    "suffix",
    [
        r" 并要求参数约束 $\beta_1=0$。",
        " 仅求斜率分量。",
        " Find only the intercept estimate.",
    ],
)
def test_gls_route_rejects_parameter_constraints_and_component_only_targets(suffix):
    assert _operation(GLS + suffix, "diagonal_gls_estimate") == []


@pytest.mark.parametrize(
    "suffix",
    [
        " 只给近似值。",
        " 不需要近似值。",
        " 用开区间表示。",
        " Report approximate values only.",
        " Give the interval without an approximation.",
    ],
)
def test_variance_interval_route_rejects_changed_representation_contracts(suffix):
    assert _operation(VARIANCE_CI + suffix, "normal_variance_confidence_interval") == []


@pytest.mark.parametrize(
    "problem, operation",
    [
        (CAUCHY_FISHER.replace("[1+(x-\\theta)^2]", "[4+(x-\\theta)^2]"), "cauchy_location_fisher_information"),
        (CAUCHY_FISHER + " 并求渐近方差。", "cauchy_location_fisher_information"),
        (WALD.replace(r"\beta_1+\beta_2=0", r"\beta_1-\beta_2=0"), "one_dimensional_wald_statistic"),
        (WALD.replace("0.10\\\\0.10", "0.10\\\\0.20"), "one_dimensional_wald_statistic"),
        (GLS.replace(r"\operatorname{diag}(1,2,1)", r"\operatorname{diag}(1,-2,1)"), "diagonal_gls_estimate"),
        (GLS + " 并求残差。", "diagonal_gls_estimate"),
        (VARIANCE_CI.replace("0.95,9", "0.95,8"), "normal_variance_confidence_interval"),
        (VARIANCE_CI.replace("双侧", "单侧"), "normal_variance_confidence_interval"),
        (VARIANCE_CI + " 并检验方差是否为1。", "normal_variance_confidence_interval"),
        (VARIANCE_CI.replace(r"求 $\sigma^2$", r"求 $\mu$"), "normal_variance_confidence_interval"),
        (VARIANCE_CI.replace(r"求 $\sigma^2$", r"求 $\sigma$"), "normal_variance_confidence_interval"),
        (VARIANCE_CI + " 并求区间宽度。", "normal_variance_confidence_interval"),
        (VARIANCE_CI + " 结果保留两位小数。", "normal_variance_confidence_interval"),
    ],
)
def test_new_inference_routes_reject_changed_models_or_extra_targets(problem, operation):
    matching = _operation(problem, operation)
    if matching:
        evidence = SubmissionAgent._tool_evidence(matching, build_problem_spec(problem))
        assert SubmissionAgent._whole_tool_answer(evidence) == ""


@pytest.mark.parametrize(
    "problem",
    [
        MOMENTS.replace("1/2,1/3,1/6", "1/2,1/3,1/3"),
        MOMENTS.replace("1,2,3", "1,2"),
        MOMENTS.replace("求E[X]与Var(X)", "只求E[X]"),
        MOMENTS.replace("求E[X]与Var(X)", "求E[X]、Var(X)与偏度"),
    ],
)
def test_incomplete_or_expanded_moment_contract_is_rejected(problem):
    assert _operation(problem, "finite_discrete_moments") == []


@pytest.mark.parametrize(
    "level, expected",
    [("0.10", "|Z|>1.645"), ("0.02", "|Z|>2.326"), ("0.01", "|Z|>2.576")],
)
def test_supported_z_levels_are_recomputed(level, expected):
    assert _operation(Z_TEST.replace("0.05", level), "two_sided_z_rejection")[0].result == expected


@pytest.mark.parametrize(
    "problem",
    [
        Z_TEST.replace("双侧", "单侧"),
        Z_TEST.replace("Z检验", "t检验"),
        Z_TEST.replace("0.05", "0.025"),
        Z_TEST.replace("拒绝域临界值", "只写临界值"),
    ],
)
def test_changed_z_contract_never_reuses_the_two_sided_region(problem):
    assert _operation(problem, "two_sided_z_rejection") == []


@pytest.mark.parametrize(
    "problem",
    [
        GEOMETRIC_TAIL.replace("公平硬币", "有偏硬币"),
        GEOMETRIC_TAIL.replace("首次正面", "第二次正面"),
        GEOMETRIC_TAIL.replace("大于3", "至少3"),
        GEOMETRIC_TAIL.replace("并说明几何分布", "并求期望"),
    ],
)
def test_geometric_tail_route_rejects_changed_stopping_rules_or_targets(problem):
    assert _operation(problem, "fair_coin_geometric_tail") == []


def test_geometric_tail_recomputes_the_strict_tail_exponent():
    changed = GEOMETRIC_TAIL.replace("大于3", "大于5")

    assert _operation(changed, "fair_coin_geometric_tail")[0].result == r"\frac{1}{32}"


@pytest.mark.parametrize(
    "problem",
    [
        POISSON_INCREMENT.replace("泊松过程", "非齐次泊松过程"),
        POISSON_INCREMENT.replace("N(2)=3", "N(3)=3"),
        POISSON_INCREMENT.replace("条件分布", "条件期望"),
        POISSON_INCREMENT.replace("N(3)-N(2)", "N(2)-N(3)"),
        POISSON_INCREMENT.replace("N(2)=3时", "N(1)=1且N(2)=3时"),
    ],
)
def test_poisson_increment_route_rejects_non_independent_or_expanded_targets(problem):
    assert _operation(problem, "poisson_process_increment") == []


def test_poisson_increment_recomputes_numeric_rate_times_interval_length():
    changed = (
        POISSON_INCREMENT.replace("强度为λ", "强度为2")
        .replace("N(3)-N(2)", "N(5)-N(2)")
    )

    result = _operation(changed, "poisson_process_increment")[0].result
    assert result.endswith(r"\operatorname{Poisson}(6)")


def test_probability_process_contracts_preserve_requested_distribution_content():
    geometric = build_problem_spec(GEOMETRIC_TAIL)
    poisson = build_problem_spec(POISSON_INCREMENT)

    assert geometric.profile.subject == "概率论"
    assert geometric.profile.answer_shape == "number"
    assert any(
        requirement.name == "geometric_distribution_identification"
        and requirement.strict
        and requirement.category == "support"
        for goal in geometric.goals
        for requirement in goal.requirements
    )
    assert poisson.profile.subject == "随机过程"
    assert poisson.profile.answer_shape == "expression"
    assert any(
        requirement.name == "distribution_result" and requirement.strict
        for goal in poisson.goals
        for requirement in goal.requirements
    )


@pytest.mark.parametrize(
    "problem, operation",
    [
        (
            GEOMETRIC_TAIL + " 最终只需给出所求概率加1后的值。",
            "fair_coin_geometric_tail",
        ),
        (
            POISSON_INCREMENT + " 另外要求结果满足一个附加限制。",
            "poisson_process_increment",
        ),
    ],
)
def test_new_probability_routes_do_not_bypass_added_output_obligations(problem, operation):
    results = _operation(problem, operation)
    assert len(results) == 1

    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert evidence[0].scope == "subexpression"
    assert SubmissionAgent._whole_tool_answer(evidence) == ""


class NoCallClient:
    def chat_result(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")


@pytest.mark.parametrize(
    "problem, expected_terms",
    [
        (GEOMETRIC_TAIL, ("Geom", r"\frac{1}{8}")),
        (POISSON_INCREMENT, ("Poisson", r"\lambda")),
        (DICE_CONDITIONAL, (r"\Omega", r"\frac{1}{6}")),
        (BERNOULLI_MOMENT, ("Var", "p(1-p)")),
        (CAUCHY_FISHER, ("I_n", r"\frac{n}{2}")),
        (WALD, (r"\frac{100}{81}",)),
        (GLS, (r"\frac{11}{10}", r"\frac{1}{2}")),
        (VARIANCE_CI, (r"\frac{18}{16.919}", r"\frac{18}{3.325}")),
    ],
)
def test_certified_probability_routes_bypass_model_with_complete_answers(problem, expected_terms):
    solved = SubmissionAgent(NoCallClient()).solve(problem, {})

    assert all(term in solved["final_response"] for term in expected_terms)
    selection = next(step for step in solved["trace"] if step["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"
    finalize = next(step for step in solved["trace"] if step["step"] == "finalize")
    assert finalize["content"]["model_call_count"] == 0


@pytest.mark.parametrize(
    "problem, expected",
    [(MOMENTS, "E[X]=5/3，Var(X)=5/9"), (Z_TEST, "|Z|>1.96")],
)
def test_exact_statistics_routes_bypass_the_model(problem, expected):
    result = SubmissionAgent(NoCallClient()).solve(problem, {})

    assert result["final_response"] == rf"\boxed{{{expected}}}"
    selection = next(step for step in result["trace"] if step["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"
    finalize = next(step for step in result["trace"] if step["step"] == "finalize")
    assert finalize["content"]["model_call_count"] == 0
