import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool


TREE_PROBLEM = "一棵有20个顶点的树恰有5个叶子，若其余非叶顶点度数都为2或3，求度数为3的顶点个数。"
INVOLUTION_PROBLEM = "求n=7时满足置换σ^2=id且σ恰有3个不动点的置换数，先将其余元素配对。"
TRAPEZOID_PROBLEM = "用复化梯形公式将[0,2]分为两段近似积分∫_0^2 x^2dx，求近似值并与精确值比较。"


def _operation(problem: str, operation: str):
    return [result for result in SympyTool().results_for(problem) if result.operation == operation]


@pytest.mark.parametrize(
    "problem, operation, expected, required_checks",
    [
        (
            TREE_PROBLEM,
            "tree_degree_census",
            "3",
            {"tree_handshake_identity", "nonnegative_integer_degree_counts"},
        ),
        (
            INVOLUTION_PROBLEM,
            "involution_fixed_point_count",
            r"\binom{7}{3}3!!=105",
            {"fixed_points_chosen", "perfect_matching_count", "parity_checked"},
        ),
        (
            TRAPEZOID_PROBLEM,
            "composite_trapezoid",
            "近似值=3，精确值=8/3，误差=1/3（近似值偏大）",
            {"endpoint_and_interior_weight_evaluation", "exact_integral_evaluation", "signed_error_comparison"},
        ),
    ],
)
def test_real_hard112_problem_is_certified_and_whole(problem, operation, expected, required_checks):
    results = _operation(problem, operation)

    assert len(results) == 1
    result = results[0]
    assert result.result == expected
    assert result.verified
    assert required_checks <= set(result.certificate.checks)
    evidence = SubmissionAgent._tool_evidence(results, build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"


def test_tree_degree_census_recomputes_both_allowed_targets():
    degree_three = TREE_PROBLEM.replace("20", "30").replace("5个叶子", "8个叶子")
    degree_two = degree_three.replace("求度数为3", "求度数为2")

    assert _operation(degree_three, "tree_degree_census")[0].result == "6"
    assert _operation(degree_two, "tree_degree_census")[0].result == "16"


def test_tree_degree_census_recomputes_a_different_higher_degree():
    problem = "一棵树有20个顶点且恰有6个叶节点，其余非叶节点的度均为2或4，求度为4的节点个数。"

    assert _operation(problem, "tree_degree_census")[0].result == "2"


@pytest.mark.parametrize(
    "problem",
    [
        TREE_PROBLEM.replace("的树", "的简单图"),
        TREE_PROBLEM.replace("恰有5个", "至少有5个"),
        TREE_PROBLEM.replace("2或3", "2或3或4"),
        TREE_PROBLEM.replace("求度数为3", "求度数为1"),
        TREE_PROBLEM.replace("20个顶点", "5个顶点"),
        TREE_PROBLEM.replace("顶点个数。", "顶点个数，且说明该树是否唯一。"),
    ],
)
def test_tree_degree_census_rejects_semantic_neighbours(problem):
    assert _operation(problem, "tree_degree_census") == []


def test_involution_count_recomputes_size_and_fixed_points():
    changed = INVOLUTION_PROBLEM.replace("n=7", "n=8").replace("3个不动点", "2个不动点")

    assert _operation(changed, "involution_fixed_point_count")[0].result == r"\binom{8}{2}5!!=420"


def test_involution_count_detects_impossible_pairing_parity():
    changed = INVOLUTION_PROBLEM.replace("n=7", "n=8")

    assert _operation(changed, "involution_fixed_point_count")[0].result == "0（其余5个元素不能完全配对）"


@pytest.mark.parametrize(
    "problem",
    [
        INVOLUTION_PROBLEM.replace("σ^2=id", "σ^3=id"),
        INVOLUTION_PROBLEM.replace("恰有3个", "至少有3个"),
        INVOLUTION_PROBLEM.replace("置换数", "不动点数"),
        INVOLUTION_PROBLEM.replace("先将其余元素配对", "先将其余元素任意排列"),
        INVOLUTION_PROBLEM.replace("σ^2=id", "σ^2=σ"),
        INVOLUTION_PROBLEM.replace("n=7", "n=2").replace("3个不动点", "3个不动点"),
    ],
)
def test_involution_count_rejects_changed_contract(problem):
    assert _operation(problem, "involution_fixed_point_count") == []


def test_composite_trapezoid_recomputes_grid_and_integrand():
    changed = "用复合梯形法将[0,1]分为四段近似计算积分∫_0^1 x^2dx，求近似值并与精确值比较。"

    assert _operation(changed, "composite_trapezoid")[0].result == (
        "近似值=11/32，精确值=1/3，误差=1/96（近似值偏大）"
    )


@pytest.mark.parametrize(
    "problem",
    [
        TRAPEZOID_PROBLEM.replace("复化梯形", "辛普森"),
        TRAPEZOID_PROBLEM.replace("分为两段", "取两个节点"),
        TRAPEZOID_PROBLEM.replace("∫_0^2", "∫_0^3"),
        TRAPEZOID_PROBLEM.replace("求近似值并与精确值比较", "只求近似值"),
        TRAPEZOID_PROBLEM.replace("分为两段", "使用非均匀网格分为两段"),
        TRAPEZOID_PROBLEM.replace("x^2dx", "(x^2+1)dx"),
    ],
)
def test_composite_trapezoid_rejects_changed_contract(problem):
    assert _operation(problem, "composite_trapezoid") == []


class NoCallClient:
    def chat_result(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")

    def chat(self, **kwargs):
        raise AssertionError(f"unexpected model call: {kwargs}")


@pytest.mark.parametrize(
    "problem, operation",
    [
        (TREE_PROBLEM, "tree_degree_census"),
        (INVOLUTION_PROBLEM, "involution_fixed_point_count"),
        (TRAPEZOID_PROBLEM, "composite_trapezoid"),
    ],
)
def test_full_semantic_match_bypasses_the_model(problem, operation):
    result = SubmissionAgent(NoCallClient()).solve(problem, {})

    call_plan = next(item for item in result["trace"] if item["step"] == "call_plan")
    evidence = next(item for item in result["trace"] if item["step"] == "tool_evidence")
    assert call_plan["content"]["route"] == "certified_tool"
    assert operation in evidence["content"]["operations"]
