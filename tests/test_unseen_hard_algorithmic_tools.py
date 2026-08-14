import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


class _NoModelClient:
    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        raise AssertionError("a certified whole-answer route must not call the model")


def _result(problem: str, operation: str):
    matches = [
        result for result in SympyTool().results_for(problem)
        if result.operation == operation
    ]
    assert len(matches) == 1
    return matches[0]


def _whole_evidence(problem: str, operation: str):
    result = _result(problem, operation)
    evidence = SubmissionAgent._tool_evidence([result], build_problem_spec(problem))
    assert len(evidence) == 1
    assert evidence[0].scope == "whole_goal"
    return result, evidence[0]


def _has_operation(problem: str, operation: str) -> bool:
    return any(
        result.operation == operation
        for result in SympyTool().results_for(problem)
    )


def test_bounded_self_exponential_divisibility_exhausts_composite_solutions():
    problem = (
        r"Determine all positive integers $1\leq n\leq10000$ such that "
        r"$n\mid(2^n+1)$. The justification must certify every integer in the range."
    )
    result, evidence = _whole_evidence(problem, "bounded_self_exponential_divisibility")

    assert result.result == r"\{1,3,9,27,81,171,243,513,729,1539,2187,3249,4617,6561,9747\}"
    assert result.verified
    assert "every_integer_in_range_enumerated" in result.certificate.checks
    assert "pow" in evidence.result
    contract = build_problem_spec(problem).answer_contract
    assert contract.mode in {"proof", "answer_with_support"}
    assert "reasoning" in contract.explicit_support_requirements


def test_bounded_self_exponential_route_transfers_to_chinese_and_new_parameters():
    problem = r"求所有正整数 $1\leq n\leq 500$，使得 $n\mid(3^n+1)$。"
    result, _ = _whole_evidence(problem, "bounded_self_exponential_divisibility")

    assert result.result == r"\{1,2,10,50,250\}"


def test_bounded_self_exponential_route_rejects_an_added_filter():
    problem = (
        r"Determine all positive integers $1\leq n\leq500$ such that "
        r"$n\mid(3^n+1)$ and $n$ is prime."
    )
    assert not [
        result for result in SympyTool().results_for(problem)
        if result.operation == "bounded_self_exponential_divisibility"
    ]


@pytest.mark.parametrize(
    "problem",
    (
        r"Determine all positive integers $1\leq n\leq500$ such that $n\mid(3^{n+1}+1)$.",
        r"Determine all positive integers $1\leq n\leq500$ such that $n\mid(2^n+1+3^n)$.",
        (
            r"Determine all positive integers $1\leq n\leq500$ such that "
            r"$n\mid(3^n+1)$ and $n$ is a perfect square."
        ),
        r"求所有正整数 $1\leq n\leq500$，使得 $n\mid(3^{n+1}+1)$。",
        r"求所有正整数 $1\leq n\leq500$，使得 $n\mid(2^n+1+3^n)$。",
        (
            r"求所有正整数 $1\leq n\leq500$，使得 $n\mid(3^n+1)$，"
            r"且 $n$ 不被 $3$ 整除。"
        ),
    ),
)
def test_bounded_self_exponential_route_rejects_changed_exponent_extra_terms_and_filters(
    problem,
):
    assert not _has_operation(problem, "bounded_self_exponential_divisibility")


def test_competing_coin_patterns_uses_overlap_preserving_exact_recursion():
    problem = (
        "A fair coin is tossed repeatedly until one of HTHH and THTH first appears; "
        "overlapping occurrences are allowed. Determine the probability that HTHH "
        "appears first, and prove it using full prefix-state recursion."
    )
    result, evidence = _whole_evidence(problem, "competing_coin_patterns")

    assert result.result == r"\frac{5}{14}"
    assert result.verified
    assert "overlap_preserving_prefix_automaton" in result.certificate.checks
    assert "complete system" in evidence.result
    assert r"u_{\emptyset}=\frac{5}{14}" in evidence.result


def test_prefix_state_recursion_wording_is_not_an_output_transform():
    problem = (
        "A fair coin is tossed repeatedly until HTHH or THTH first appears, with "
        "overlaps allowed. Determine the probability that HTHH appears first, and prove "
        "it using prefix-state recursion rather than independent blocks."
    )

    _, evidence = _whole_evidence(problem, "competing_coin_patterns")
    assert r"\boxed{\frac{5}{14}}" in evidence.result
    assert not SubmissionAgent._has_uncovered_tool_obligation(
        problem, "competing_coin_patterns"
    )


def test_competing_coin_pattern_route_transfers_to_chinese_patterns():
    problem = (
        "反复抛掷一枚公平硬币，直到 HTH 与 HHT 之一首次出现，允许重叠。"
        "求 HTH 先出现的概率，并用完整的前缀状态递推证明。"
    )
    result, _ = _whole_evidence(problem, "competing_coin_patterns")

    assert result.result == r"\frac{1}{3}"


def test_competing_coin_pattern_route_rejects_biased_or_nonoverlapping_variants():
    changed = (
        "A biased coin is tossed until HTH or HHT appears. Determine the probability "
        "that HTH appears first."
    )
    nonoverlap = (
        "A fair coin is tossed until HTH or HHT appears without overlap. Determine the "
        "probability that HTH appears first."
    )

    for problem in (changed, nonoverlap):
        assert not [
            result for result in SympyTool().results_for(problem)
            if result.operation == "competing_coin_patterns"
        ]


@pytest.mark.parametrize(
    "problem",
    (
        (
            "Given that the first toss is H, a fair coin is tossed repeatedly until HTH "
            "or HHT first appears. Determine the probability that HTH appears first."
        ),
        (
            "反复抛掷一枚公平硬币，直到 HTH 或 HHT 之一首次出现。已知第一次抛掷为 H，"
            "求 HTH 先出现的概率。"
        ),
        (
            "A fair coin is tossed repeatedly until HTH or HHT first appears. Determine "
            "the probability that HTH appears first and the stopping time is at most six tosses."
        ),
        (
            "A fair coin is tossed repeatedly until HTH or HHT first appears, but after "
            "each T the recorded history is reset. Determine the probability that HTH appears first."
        ),
        (
            "A fair coin is tossed repeatedly until HTH or HHT first appears, except that "
            "after TT the next toss is forced to be H. Determine the probability that HTH "
            "appears first."
        ),
    ),
)
def test_competing_coin_pattern_route_rejects_initial_state_stopping_and_process_changes(
    problem,
):
    assert not _has_operation(problem, "competing_coin_patterns")


def test_unboxed_english_coin_race_is_a_certified_no_model_answer_not_fallback_zero():
    problem = (
        "A fair coin is tossed repeatedly until HTH or HHT first appears, with overlapping "
        "occurrences allowed. Determine the probability that HTH appears first."
    )
    client = _NoModelClient()

    solved = ReasoningAgent(client).solve(problem, {})
    answer = solved["final_response"]

    assert client.calls == 0
    assert r"\frac{1}{3}" in answer
    assert answer.strip(" \t\r\n。.") != "0"
    assert "未能生成可验证的数学答案" not in answer


def test_certified_proof_routes_keep_support_outside_the_final_box_end_to_end():
    problems = (
        (
            r"Determine all positive integers $1\leq n\leq500$ such that $n\mid(3^n+1)$. "
            r"The justification must certify every integer. Put the final answer in \boxed{}.",
            r"\boxed{\{1,2,10,50,250\}}",
        ),
        (
            "A fair coin is tossed until HTH or HHT first appears, with overlaps allowed. "
            "Determine the probability that HTH appears first and prove it by prefix-state "
            r"recursion. Put the final answer in \boxed{}.",
            r"\boxed{\frac{1}{3}}",
        ),
    )

    for problem, expected_box in problems:
        client = _NoModelClient()
        result = ReasoningAgent(client).solve(problem, {})
        assert client.calls == 0
        assert result["final_response"].count(r"\boxed{") == 1
        assert result["final_response"].endswith(expected_box)
        assert "exhaust" in result["final_response"].lower()
