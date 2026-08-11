from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate
from reasoning.math_equivalence import equivalent_answers
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


PROBLEMS = (
    (
        "求长度为10的二进制串中恰有4个1且不含相邻两个1的串数，要求先选取1的位置再计算。",
        "nonadjacent_binary_string_count",
        "35",
        (r"\binom{7}{4}=35", "7个空位"),
    ),
    (
        "在5个不同元素的排列中，求元素a在b之前且c不在首位的排列数，使用容斥或条件计数。",
        "precedence_permutation_count",
        "48",
        ("60-12=48", "条件计数"),
    ),
    (
        "求从集合{1,2,3,4}到{a,b,c}的满射个数，要求使用容斥原理而非直接枚举。",
        "surjection_count",
        "36",
        ("=36", "容斥原理"),
    ),
)


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified combinatorial route called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def test_standard_counting_contracts_are_certified_whole_answers():
    for problem, operation, expected, support_terms in PROBLEMS:
        evidence = _evidence(problem)
        result = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert evidence[0].operation == operation
        assert evidence[0].scope == "whole_goal"
        assert evidence[0].verified
        assert expected in evidence[0].result
        assert all(term in evidence[0].support for term in support_terms)
        assert expected in result["final_response"]
        assert all(term in result["final_response"] for term in support_terms)
        assert next(
            step for step in result["trace"] if step["step"] == "selection"
        )["content"]["source"] == "sympy_verified"


def test_equivalent_english_counting_contracts_return_english_answers():
    cases = (
        (
            "Find the number of binary strings of length 10 with exactly 4 ones and no two ones adjacent.",
            "nonadjacent_binary_string_count",
            "35",
        ),
        (
            "Among permutations of 5 distinct elements, how many have a before b and c is not first?",
            "precedence_permutation_count",
            "48",
        ),
        (
            "Find the number of surjections from a four-element set to a three-element set using inclusion-exclusion.",
            "surjection_count",
            "36",
        ),
    )
    for problem, operation, expected in cases:
        evidence = _evidence(problem)
        answer = ReasoningAgent(_NoModelClient()).solve(problem, {})["final_response"]

        assert evidence[0].operation == operation
        assert evidence[0].scope == "whole_goal"
        assert expected in evidence[0].result
        assert evidence[0].support
        assert expected in answer


def test_extra_proof_obligations_downgrade_counts_to_local_checks():
    cases = (
        PROBLEMS[0][0] + "并证明该公式。",
        PROBLEMS[1][0] + "并解释为什么没有重复计数。",
        PROBLEMS[2][0] + "并证明容斥公式。",
    )
    for problem in cases:
        evidence = _evidence(problem)

        assert len(evidence) == 1
        assert evidence[0].operation.endswith("_check")
        assert evidence[0].scope == "subexpression"


def test_nearby_but_different_counting_problems_do_not_trigger_whole_routes():
    cases = (
        "求长度为10且恰有4个1的全部二进制串数。",
        "在5个不同元素的排列中，求a在b之前且a不在首位的排列数。",
        "求从集合{1,1,2,3}到{a,b,c}的满射个数。",
        "求从集合{1,2,3,4}到{a,b,c}的所有映射个数。",
    )
    operations = {
        "nonadjacent_binary_string_count",
        "precedence_permutation_count",
        "surjection_count",
    }
    for problem in cases:
        assert not any(
            item.operation in operations and item.scope == "whole_goal"
            for item in _evidence(problem)
        )


def test_false_closed_binomial_identity_is_hard_rejected():
    problem = PROBLEMS[0][0]
    spec = build_problem_spec(problem)
    wrong = assess_candidate(
        r"结论：35。插空选位置：\(\binom{8}{4}=35\)。",
        "solve",
        spec,
        (),
    )
    correct = assess_candidate(
        r"结论：35。插空选位置：\(\binom{7}{4}=35\)。",
        "solve",
        spec,
        (),
    )

    assert "numeric_identity_conflict" in wrong.rejected_reasons
    assert wrong.validation_tier == "rejected"
    assert correct.accepted
