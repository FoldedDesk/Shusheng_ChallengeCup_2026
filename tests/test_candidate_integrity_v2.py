from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from reasoning.candidate_selector import assess_candidate, choose_candidate
from user_agent import ReasoningAgent


class _RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_unlabelled_process_body_is_rejected_and_triggers_rescue():
    client = _RecordingClient([
        "计算过程如下：\n中间值为17。",
        r"FINAL: \boxed{17}",
    ])

    result = ReasoningAgent(client).solve(
        "某项统计量已知为十七，求该统计量的数值。", {}
    )

    assert result["final_response"] == "17"
    assert len(client.calls) == 2
    admission = next(
        item["content"] for item in result["trace"]
        if item["step"] == "review_admission"
    )
    assert admission["mode"] == "rescue"


def test_unlabelled_future_action_is_rejected_but_concise_answer_is_allowed():
    spec = build_problem_spec("某项统计量已知为十七，求该统计量的数值。")
    unfinished = assess_candidate(
        "先计算中间量a=1。\n接下来需要继续分析。",
        "solve",
        spec,
        (),
        extraction_method="whole_response",
    )
    concise = assess_candidate(
        "结果为17。",
        "solve",
        spec,
        (),
        extraction_method="whole_response",
    )

    assert unfinished.validation_tier == "rejected"
    assert "unlabelled_future_action" in unfinished.rejected_reasons
    assert "unlabelled_intermediate_result" in unfinished.rejected_reasons
    assert concise.accepted


def test_proof_like_unlabelled_bodies_require_real_support():
    cases = (
        (
            "说明为什么连续函数在闭区间上可积。",
            "连续函数在闭区间上可积。",
            "由于连续函数在闭区间上一致连续，因此上下和之差可任意小，故可积。",
        ),
        (
            "推导二次方程的求根公式。",
            "x=(-b±sqrt(b^2-4ac))/(2a)。",
            "推导过程如下：\n由配方法可得(2ax+b)^2=b^2-4ac，因此"
            "x=(-b±sqrt(b^2-4ac))/(2a)。",
        ),
    )
    for problem, bare_answer, supported_answer in cases:
        spec = build_problem_spec(problem)
        bare = assess_candidate(
            bare_answer, "solve", spec, (), extraction_method="whole_response"
        )
        supported = assess_candidate(
            supported_answer, "solve", spec, (), extraction_method="whole_response"
        )

        assert not bare.accepted
        assert "missing_required_support" in bare.rejected_reasons
        assert supported.accepted


def test_construction_requires_object_and_condition_verification():
    spec = build_problem_spec("构造一个整数x，使x^2=4且x>0。")
    bare = assess_candidate("x=2", "solve", spec, ())
    object_free = assess_candidate("经验证满足题设条件。", "solve", spec, ())
    verified = assess_candidate(
        "取x=2，代入得2^2=4且2>0，满足题设条件。", "solve", spec, ()
    )

    assert all(bare.result_coverage)
    assert not all(bare.support_coverage)
    assert not bare.accepted
    assert "missing_construction_verification" in bare.rejected_reasons
    assert "missing_construction_object" in object_free.rejected_reasons
    assert verified.accepted


def test_construct_and_prove_requires_both_argument_and_checked_object():
    spec = build_problem_spec(
        "Construct a function and prove that it satisfies the stated condition."
    )
    no_object = assess_candidate(
        "Because the condition is consistent, such a function exists and satisfies it.",
        "solve",
        spec,
        (),
    )
    complete = assess_candidate(
        "Let f(x)=x. Because f(0)=0, direct substitution verifies that f satisfies "
        "the stated condition.",
        "solve",
        spec,
        (),
    )

    assert "missing_construction_object" in no_object.rejected_reasons
    assert not no_object.accepted
    assert complete.accepted


def test_equivalent_candidates_prefer_support_but_conflicts_still_prefer_verify():
    spec = build_problem_spec(
        "求长度为10的二进制串中恰有4个1且不含相邻两个1的串数，"
        "要求先选取1的位置再计算。"
    )
    supported = assess_candidate(
        "35\n因为先选取位置可得35。", "solve", spec, ()
    )
    equivalent_bare = assess_candidate("35", "verify", spec, ())
    conflicting_bare = assess_candidate("36", "verify", spec, ())

    assert choose_candidate([supported, equivalent_bare]).source == "solve"
    assert choose_candidate([supported, conflicting_bare]).source == "verify"


def test_newton_result_requirements_need_actual_assignments_not_headings():
    spec = build_problem_spec(
        "用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。"
    )
    heading_only = assess_candidate(
        "x_1=7/4。牛顿法迭代公式为：",
        "verify",
        spec,
        (),
    )
    complete = assess_candidate(
        "x_{n+1}=(x_n+3/x_n)/2，x_1=7/4。",
        "verify",
        spec,
        (),
    )

    assert not heading_only.complete_goals
    assert not heading_only.accepted
    assert complete.accepted
