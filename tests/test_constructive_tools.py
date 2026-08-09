from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


SPIKE_PROBLEM = "构造[0,1]上非负可测函数列f_n使其逐点趋于0而积分恒为1，并写出一个具体公式。"
BERNOULLI_PROBLEM = "构造两个边缘均为Bernoulli(1/2)但不独立的随机变量，并给出P(X=Y)。"


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified construction unexpectedly called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().hints_for(problem), spec)


def test_unit_mass_spike_is_a_certified_complete_construction():
    hint = SympyTool().hints_for(SPIKE_PROBLEM)[0]
    evidence = _evidence(SPIKE_PROBLEM)

    assert hint.startswith("本地尖峰函数构造答案: ")
    assert r"f_n(x)=n\mathbf{1}_{(0,1/n]}(x)" in hint
    assert r"逐点 f_n(x)\to0\ (\forall x\in[0,1])" in hint
    assert r"积分为 \int_0^1 f_n(x)\,dx=1" in hint
    assert evidence[0].operation == "spike_sequence_construction"
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].verified

    result = ReasoningAgent(_NoModelClient()).solve(SPIKE_PROBLEM, {})
    assert result["final_response"].startswith(r"取 f_n(x)=n\mathbf{1}_{(0,1/n]}")
    assert next(step for step in result["trace"] if step["step"] == "selection")["content"]["source"] == "sympy_verified"


def test_perfectly_dependent_fair_bernoulli_pair_is_certified_complete():
    hint = SympyTool().hints_for(BERNOULLI_PROBLEM)[0]
    evidence = _evidence(BERNOULLI_PROBLEM)

    assert hint.startswith("本地Bernoulli依赖构造答案: ")
    assert r"P((X,Y)=(0,0))=P((X,Y)=(1,1))=1/2" in hint
    assert r"P(X=1,Y=1)=1/2\neq1/4" in hint
    assert "P=1，即 P(X=Y)=1" in hint
    assert evidence[0].operation == "dependent_bernoulli_construction"
    assert evidence[0].scope == "whole_goal"

    result = ReasoningAgent(_NoModelClient()).solve(BERNOULLI_PROBLEM, {})
    assert "Y=X" in result["final_response"]
    assert "P=1，即 P(X=Y)=1" in result["final_response"]
    assert next(step for step in result["trace"] if step["step"] == "selection")["content"]["source"] == "sympy_verified"


def test_extra_proof_obligations_force_both_constructions_to_local_checks():
    spike = SPIKE_PROBLEM + "并证明逐点收敛和积分结论。"
    bernoulli = BERNOULLI_PROBLEM + "请证明它们不独立。"

    for problem, operation in (
        (spike, "spike_sequence_construction_check"),
        (bernoulli, "dependent_bernoulli_construction_check"),
    ):
        evidence = _evidence(problem)
        assert len(evidence) == 1
        assert evidence[0].operation == operation
        assert evidence[0].scope == "subexpression"
        assert evidence[0].verified


def test_extra_calculations_and_probabilities_never_use_the_whole_route():
    spike = SPIKE_PROBLEM + "并计算其上确界。"
    bernoulli = BERNOULLI_PROBLEM + "并给出P(X!=Y)和协方差。"

    spike_evidence = _evidence(spike)
    bernoulli_evidence = _evidence(bernoulli)
    assert spike_evidence[0].operation == "spike_sequence_construction_check"
    assert spike_evidence[0].scope == "subexpression"
    assert bernoulli_evidence[0].operation == "dependent_bernoulli_construction_check"
    assert bernoulli_evidence[0].scope == "subexpression"


def test_nearby_but_different_conditions_are_verification_only():
    near_spikes = (
        SPIKE_PROBLEM.replace("[0,1]", "[0,2]"),
        SPIKE_PROBLEM.replace("积分恒为1", "积分恒为2"),
        SPIKE_PROBLEM.replace("逐点趋于0", "依测度趋于0"),
    )
    near_bernoullis = (
        BERNOULLI_PROBLEM.replace("Bernoulli(1/2)", "Bernoulli(1/3)"),
        BERNOULLI_PROBLEM.replace("不独立", "独立"),
        BERNOULLI_PROBLEM.replace("两个边缘均", "一个边缘为"),
    )

    for problem in near_spikes:
        evidence = _evidence(problem)
        assert evidence[0].operation == "spike_sequence_construction_check"
        assert evidence[0].scope == "subexpression"
    for problem in near_bernoullis:
        evidence = _evidence(problem)
        assert evidence[0].operation == "dependent_bernoulli_construction_check"
        assert evidence[0].scope == "subexpression"


def test_equivalent_precise_english_contracts_are_also_certified():
    spike = (
        "Construct a sequence of non-negative measurable functions f_n on [0,1] that "
        "converges pointwise to 0, has integral equal to 1 for every n, and give an explicit formula."
    )
    bernoulli = (
        "Construct two random variables X and Y whose marginals are both Bernoulli(1/2), "
        "which are not independent, and give P(X=Y)."
    )

    assert _evidence(spike)[0].operation == "spike_sequence_construction"
    assert _evidence(spike)[0].scope == "whole_goal"
    assert _evidence(bernoulli)[0].operation == "dependent_bernoulli_construction"
    assert _evidence(bernoulli)[0].scope == "whole_goal"

    spike_answer = ReasoningAgent(_NoModelClient()).solve(spike, {})["final_response"]
    bernoulli_answer = ReasoningAgent(_NoModelClient()).solve(bernoulli, {})["final_response"]
    assert spike_answer.startswith("Take ")
    assert "pointwise" in spike_answer
    assert bernoulli_answer.startswith("Let ")
    assert "not independent" in bernoulli_answer
    assert not any(term in spike_answer + bernoulli_answer for term in ("取 ", "两边缘", "故 "))


def test_similarly_named_unrelated_problems_do_not_trigger_construction_tools():
    cases = (
        "求Bernoulli方程y'+y=y^2的非零通解。",
        "设f_n逐点收敛到f，陈述控制收敛定理。",
        "若X服从Bernoulli(1/2)，求E[X]。",
    )

    for problem in cases:
        operations = {item.operation for item in _evidence(problem)}
        assert "spike_sequence_construction" not in operations
        assert "spike_sequence_construction_check" not in operations
        assert "dependent_bernoulli_construction" not in operations
        assert "dependent_bernoulli_construction_check" not in operations
