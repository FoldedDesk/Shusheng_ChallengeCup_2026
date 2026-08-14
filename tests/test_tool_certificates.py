import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sympy_tool import SympyTool
from tools.tool_contract import result_from_legacy_hint
from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent


AMBIENT = r"计算函数 f(x,y)=x^2+y^2 在圆周 x^2+y^2=1 上的欧氏环境拉普拉斯算子。"


def _only(problem: str):
    results = SympyTool().results_for(problem)
    assert len(results) == 1
    return results[0]


def test_ambient_circle_laplacian_is_four_with_a_structured_certificate():
    result = _only(AMBIENT)

    assert result.operation == "circle_laplacian"
    assert result.result == "4"
    assert result.verified
    assert result.whole_answer_eligible
    assert result.contract is not None
    assert result.contract.certificate_method == "ambient_second_derivatives"
    assert result.contract.covers(1)
    assert not result.contract.covers(1, problem_facts=())
    assert result.contract.covers(1, problem_facts=result.certificate.checks)
    assert not result.contract.covers(1, ("proof",))
    assert "explicit_intrinsic_operator" in result.contract.forbidden_problem_facts
    assert "second_derivatives_sum_to_4" in result.certificate.checks
    json.dumps(result.trace_content(), ensure_ascii=False)


def test_circle_radius_perturbation_does_not_change_ambient_laplacian():
    result = _only(
        r"Compute the ambient Euclidean Laplacian of f(x,y)=x^2+y^2 on the circle x^2+y^2=R^2."
    )

    assert result.operation == "circle_laplacian"
    assert result.result == "4"


def test_unqualified_circle_laplacian_is_verification_only():
    result = _only(r"计算函数 f(x,y)=x^2+y^2 在圆周 x^2+y^2=1 上的拉普拉斯算子。")

    assert result.operation == "circle_laplacian_ambiguous"
    assert result.verified
    assert not result.whole_answer_eligible
    assert "=2+2=4" in result.result
    assert "值为 \\(0\\)" in result.result


def test_explicit_laplace_beltrami_of_the_restriction_is_zero():
    result = _only(
        r"计算 f(x,y)=x^2+y^2 限制到圆周 x^2+y^2=4 后的 Laplace-Beltrami 算子。"
    )

    assert result.operation == "circle_laplace_beltrami"
    assert result.result == "0"
    assert result.whole_answer_eligible
    assert "restriction_is_constant" in result.certificate.checks


def test_explicit_operator_ambiguity_is_check_evidence_not_a_whole_answer():
    result = _only(
        r"对 f(x,y)=x^2+y^2 和圆周 x^2+y^2=1，题目未说明拉普拉斯是环境还是内蕴。"
    )

    assert result.operation == "circle_laplacian_ambiguous"
    assert result.verified
    assert not result.whole_answer_eligible
    assert "=2+2=4" in result.result
    assert "值为 \\(0\\)" in result.result


def test_missing_circle_condition_and_changed_operator_do_not_match():
    tool = SympyTool()

    assert not tool.results_for(r"计算 f(x,y)=x^2+y^2 的拉普拉斯算子。")
    assert not tool.results_for(
        r"计算 f(x,y)=x^2+y^2 在圆周 x^2+y^2=1 上的双调和算子。"
    )
    assert not tool.results_for(
        r"Compute the bi-Laplacian of f(x,y)=x^2+y^2 on the circle x^2+y^2=1."
    )


def test_extra_proof_obligation_downgrades_to_a_verified_check():
    result = _only(AMBIENT + "并证明所用拉普拉斯公式。")

    assert result.operation == "circle_laplacian_check"
    assert result.verified
    assert not result.whole_answer_eligible
    assert result.result == "4"


def test_legacy_text_cannot_self_assert_a_certificate():
    untrusted = result_from_legacy_hint("本地圆周拉普拉斯: 4")
    unknown = result_from_legacy_hint("模型自称已验证: 4", trusted_source=True)

    assert untrusted is not None and not untrusted.verified
    assert untrusted.certificate.issues == ("untrusted_legacy_text",)
    assert unknown is not None and not unknown.verified
    assert unknown.operation == "local_hint"
    assert unknown.certificate.issues == ("unregistered_operation",)


def test_legacy_hint_api_remains_available():
    assert SympyTool().hints_for(AMBIENT) == ["本地圆周拉普拉斯: 4"]


def test_certificate_is_bound_to_the_problem_that_produced_it():
    result = _only(AMBIENT)
    unrelated = build_problem_spec("Calculate 1+1.")

    evidence = SubmissionAgent._tool_evidence([result], unrelated)

    assert len(evidence) == 1
    assert evidence[0].scope == "subexpression"
    assert SubmissionAgent._whole_tool_answer(evidence) == ""
