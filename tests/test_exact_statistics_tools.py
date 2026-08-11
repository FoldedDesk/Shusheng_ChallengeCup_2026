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
