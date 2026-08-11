import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


UNION = "若A,B独立且P(A)=1/2,P(B)=1/3，求P(A∪B)并说明独立性用在何处。"
NORMAL_SUM = "设X,Y独立且均服从标准正态分布，求X+Y的分布及其方差，并说明独立性的用途。"
BROWNIAN = "设B(t)为标准布朗运动，求Cov(B(s),B(t))在s≤t时的值，并说明独立增量推导。"
SAMPLE_MEAN = "总体方差已知为1，样本量n=4，求样本均值的方差并说明独立同分布假设。"
RENEWAL = "更新过程的更新间隔均值为2，写出强大数律给出的N(t)/t极限。"


def _operation(problem, operation):
    return [result for result in SympyTool().results_for(problem) if result.operation == operation]


@pytest.mark.parametrize(
    "problem, operation, result, method, subject",
    [
        (
            UNION,
            "independent_event_union",
            r"P(A\cup B)=\frac{2}{3}",
            "independent_union_identity",
            "概率论",
        ),
        (
            NORMAL_SUM,
            "independent_standard_normal_sum",
            r"X+Y\sim N(0,2),\quad \operatorname{Var}(X+Y)=2",
            "normal_convolution_parameters",
            "概率论",
        ),
        (
            BROWNIAN,
            "brownian_covariance",
            r"\operatorname{Cov}(B(s),B(t))=s",
            "brownian_independent_increment_identity",
            "随机过程",
        ),
        (
            SAMPLE_MEAN,
            "sample_mean_variance",
            r"\operatorname{Var}(\bar X)=\frac{1}{4}",
            "iid_variance_scaling",
            "概率论",
        ),
        (
            RENEWAL,
            "renewal_rate_limit",
            r"\lim_{t\to\infty}\frac{N(t)}{t}=\frac{1}{2}",
            "renewal_strong_law_rate",
            "随机过程",
        ),
    ],
)
def test_exact_statistics_formula_is_certified_and_contract_complete(
    problem, operation, result, method, subject
):
    spec = build_problem_spec(problem)
    results = _operation(problem, operation)

    assert spec.profile.subject == subject
    assert len(results) == 1
    assert results[0].result == result
    assert results[0].verified
    assert results[0].certificate.method == method
    evidence = SubmissionAgent._tool_evidence(results, spec)
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].support
    assert evidence[0].result == evidence[0].support


def test_independent_union_recomputes_changed_probabilities():
    changed = UNION.replace("1/2", "1/4").replace("1/3", "1/5")

    assert _operation(changed, "independent_event_union")[0].result == r"P(A\cup B)=\frac{2}{5}"


@pytest.mark.parametrize(
    "problem",
    [
        UNION.replace("独立", "互斥"),
        UNION.replace("P(A∪B)", "P(A∩B)"),
        UNION.replace("并说明独立性用在何处", "并求条件概率P(A|B)"),
        UNION.replace("P(A)=1/2", "P(A)=3/2"),
    ],
)
def test_independent_union_rejects_changed_event_contract(problem):
    assert _operation(problem, "independent_event_union") == []


@pytest.mark.parametrize(
    "problem",
    [
        NORMAL_SUM.replace("独立", "相关"),
        NORMAL_SUM.replace("X+Y", "X-Y"),
        NORMAL_SUM.replace("标准正态分布", "均值1方差1的正态分布"),
        NORMAL_SUM.replace("分布及其方差", "分布及其协方差"),
    ],
)
def test_standard_normal_sum_rejects_changed_parameters_or_targets(problem):
    assert _operation(problem, "independent_standard_normal_sum") == []


@pytest.mark.parametrize(
    "problem",
    [
        BROWNIAN.replace("标准布朗运动", "带漂移布朗运动"),
        BROWNIAN.replace("s≤t", "t≤s").replace("Cov(B(s),B(t))", "Cov(B(s),B(u))"),
        BROWNIAN.replace("Cov(B(s),B(t))", "Corr(B(s),B(t))"),
    ],
)
def test_brownian_covariance_rejects_changed_process_or_time_contract(problem):
    assert _operation(problem, "brownian_covariance") == []


def test_sample_mean_variance_recomputes_population_variance_and_size():
    changed = SAMPLE_MEAN.replace("方差已知为1", "方差已知为3/2").replace("n=4", "n=6")

    assert _operation(changed, "sample_mean_variance")[0].result == (
        r"\operatorname{Var}(\bar X)=\frac{1}{4}"
    )


def test_renewal_rate_limit_recomputes_the_reciprocal_mean():
    changed = RENEWAL.replace("均值为2", "均值为5/2")

    assert _operation(changed, "renewal_rate_limit")[0].result.endswith(r"=\frac{2}{5}")


@pytest.mark.parametrize(
    "problem",
    [
        RENEWAL.replace("更新过程", "延迟更新过程"),
        RENEWAL.replace("强大数律", "中心极限定理"),
        RENEWAL.replace("更新间隔均值为2", "更新间隔均值未知"),
        RENEWAL.replace("N(t)/t极限", "N(t)的期望"),
    ],
)
def test_renewal_rate_limit_rejects_changed_process_or_target(problem):
    assert _operation(problem, "renewal_rate_limit") == []


@pytest.mark.parametrize(
    "problem",
    [
        SAMPLE_MEAN.replace("独立同分布", "相关同分布"),
        SAMPLE_MEAN.replace("方差并说明", "标准误并说明"),
        SAMPLE_MEAN.replace("样本量n=4", "不放回抽样，样本量n=4"),
    ],
)
def test_sample_mean_variance_rejects_non_iid_or_different_targets(problem):
    assert _operation(problem, "sample_mean_variance") == []


class NoCallClient:
    def chat_result(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")


@pytest.mark.parametrize(
    "problem, expected_terms",
    [
        (UNION, (r"P(A\cup B)", r"\frac{2}{3}")),
        (NORMAL_SUM, ("N(0,2)", "Var")),
        (BROWNIAN, ("Cov", "=s")),
        (SAMPLE_MEAN, ("Var", r"\frac{1}{4}")),
        (RENEWAL, (r"N(t)/t", r"\frac{1}{2}")),
    ],
)
def test_exact_statistics_formula_routes_bypass_the_model(problem, expected_terms):
    solved = SubmissionAgent(NoCallClient()).solve(problem, {})

    assert all(term in solved["final_response"] for term in expected_terms)
    selection = next(step for step in solved["trace"] if step["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"
