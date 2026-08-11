from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


def _whole(problem: str) -> bool:
    return build_problem_spec(problem).tool_can_answer_whole


def test_direct_arithmetic_is_the_only_numeric_whole_answer_route():
    assert _whole("Calculate 2 + 3 * 4.")
    assert _whole("计算 2^5-3。")

    assert not _whole("Find the area of a circle of radius 2.")
    assert not _whole("Calculate 2 + 3 and explain the result.")
    assert not _whole("Calculate 2/3 to three decimal places.")


def test_only_unconstrained_one_variable_equations_are_whole_answer_routes():
    assert _whole("Solve the equation x^2-1=0.")
    assert _whole("Solve x^2-1=0.")
    assert _whole("Find x if x+1=3.")
    assert _whole("求解方程 y^2-4=0。")

    assert not _whole("Solve x+y=2.")
    assert not _whole("Solve ax=2 for x.")
    assert not _whole("Solve x^2=2 over the real numbers.")
    assert not _whole("Find the positive solution of x^2=2.")
    assert not _whole("Solve x+y=2 and x-y=0.")


def test_single_direct_calculus_requests_may_use_the_whole_tool_result():
    assert _whole("Find the derivative of f(x)=x^3+2*x.")
    assert _whole(r"Evaluate $\int_0^1 x^2\,dx$.")
    assert _whole(r"Find $\lim_{x\to 0} \sin(x)/x$.")

    assert not _whole("Find the second derivative of f(x)=x^3.")
    assert not _whole(r"Evaluate the indefinite integral $\int x^2\,dx$.")
    assert not _whole(r"Evaluate $\int x^2\,dx$.")
    assert not _whole(r"Evaluate $\int_0^1 x*y\,dx$.")
    assert not _whole(r"Evaluate $\int_0^1 x\,dx$ and $\int_0^2 x\,dx$.")
    assert not _whole("Find the derivative of f(x)=x^2 and compare it with x.")


def test_generic_symbolic_evidence_obeys_the_problem_level_whole_answer_gate():
    cases = (
        ("Find the second derivative of f(x)=x^3.", "derivative"),
        (r"在实数范围内求解方程 $x^2+1=0$。", "solve_equation"),
    )
    for problem, operation in cases:
        spec = build_problem_spec(problem)
        evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)

        assert not spec.tool_can_answer_whole
        assert any(item.operation == operation for item in evidence)
        assert all(item.scope == "subexpression" for item in evidence)

    direct_problem = "Find the derivative of f(x)=x^3."
    direct_spec = build_problem_spec(direct_problem)
    direct = SubmissionAgent._tool_evidence(SympyTool().results_for(direct_problem), direct_spec)
    assert direct_spec.tool_can_answer_whole
    assert any(
        item.operation == "derivative"
        and item.result == "3*x^2"
        and item.scope == "whole_goal"
        for item in direct
    )


def test_indefinite_integrals_are_never_certified_as_complete_answers():
    from tools.tool_contract import result_from_legacy_hint

    result = result_from_legacy_hint("SymPy 不定积分: log(x)", trusted_source=True)

    assert result is not None and result.verified
    assert result.contract is not None
    assert not result.contract.whole_answer_capable
    assert not result.whole_answer_eligible


def test_numerical_methods_and_requested_processes_are_local_evidence_only():
    cases = (
        "Use Newton's method for x^2-2=0 and give x_1 from x_0=1.",
        "Use the bisection method to approximate a root of x^2-2=0.",
        "Use Euler's method to solve y'=y with y(0)=1.",
        "Iterate x_{n+1}=(x_n+2/x_n)/2 until convergence.",
        "Verify that x=2 solves x^2=4.",
        "Prove that the equation x^2=2 has no rational root.",
    )
    assert all(not _whole(problem) for problem in cases)


def test_olympiad_and_long_context_problems_never_bypass_the_solver():
    assert not _whole(
        "Find all positive integers n such that n divides 2^n-1."
    )
    assert not _whole(
        "Let x be a number arising in a mathematical competition problem. "
        + "Assume the surrounding construction has several dependent conditions. " * 4
        + "Solve x^2-1=0."
    )


def test_remember_box_suffix_is_format_only_and_preserves_wrapper():
    spec = build_problem_spec(
        r"Find the number of roots of x^2-1=0. "
        r"Remember to put your final answer within \boxed{}."
    )

    assert spec.goals[0].instruction == "Find the number of roots of x^2-1=0"
    assert spec.answer_contract.wrapper == "boxed"
    assert "final answer" not in spec.goals[0].instruction.lower()


def test_english_count_contract_does_not_invent_a_chinese_unit():
    spec = build_problem_spec("Find the number of roots of x^2-1=0.")

    assert spec.answer_frame.question_kind == "count"
    assert spec.answer_frame.unit == ""
    assert spec.answer_contract.unit == ""
    assert spec.answer_contract.parts[0].unit == ""


def test_explicit_english_unit_is_still_retained():
    spec = build_problem_spec("Find angle A measured in degrees.")

    assert spec.answer_contract.unit == "degrees"


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified local route unexpectedly called the model: {kwargs}")


def test_certified_statistical_and_numerical_routes_bypass_the_model():
    cases = (
        (
            "某城市有10000名市民与两个剧院，每个剧院有 x 个座位，每个市民独立等可能地"
            "选取一个剧院。若无座概率不超过0.01，计算 x 的最小值。"
            r" Remember to put your final answer within \boxed{}.",
            r"\boxed{5129}",
        ),
        (
            r"假设 $(X_n)$ 是 $N$ 个顶点的完全图上的简单随机游动，T为首次遍访所有顶点"
            r"的时间，求ET. Remember to put your final answer within \boxed{}.",
            r"\boxed{(N-1)\sum_{j=1}^{N-1}\frac{1}{j}}",
        ),
        (
            r"计算函数 f(x,y)=x^2+y^2 在圆周 x^2+y^2=1 上的欧氏环境拉普拉斯算子。"
            r"Remember to put your final answer within \boxed{}.",
            r"\boxed{4}",
        ),
    )

    for problem, expected in cases:
        result = ReasoningAgent(_NoModelClient()).solve(problem, {})
        assert result["final_response"] == expected
        assert next(step for step in result["trace"] if step["step"] == "selection")["content"]["source"] == "sympy_verified"


def test_centered_difference_is_a_certified_whole_answer():
    spec = build_problem_spec(
        r"使用中心差分公式计算函数 f(x)=\sin(x) 在 x=\pi/4 处的一阶导数，取 h=0.1。"
    )
    from core.submission_agent import SubmissionAgent
    from tools.sympy_tool import SympyTool

    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(
        r"使用中心差分公式计算函数 f(x)=\sin(x) 在 x=\pi/4 处的一阶导数，取 h=0.1。"
    ), spec)

    assert len(evidence) == 1
    assert evidence[0].operation == "central_difference"
    assert evidence[0].scope == "whole_goal"
    assert "中心差分公式" in evidence[0].result
    assert r"\approx 0.7059" in evidence[0].result

    result = ReasoningAgent(_NoModelClient()).solve(
        r"使用中心差分公式计算函数 f(x)=\sin(x) 在 x=\pi/4 处的一阶导数，取 h=0.1。"
        r"Remember to put your final answer within \boxed{}.",
        {},
    )
    assert "中心差分公式" in result["final_response"]
    assert r"\approx 0.7059" in result["final_response"]


def test_embedded_equations_cannot_replace_the_requested_function_result():
    problem = "求函数f(z)=z^2的实部u(x,y)，并验证u满足拉普拉斯方程。"
    spec = build_problem_spec(problem)
    evidence = SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)

    assert evidence
    assert all(item.scope == "subexpression" for item in evidence)
    assert any(item.operation == "solve_equation" for item in evidence)


NUMBER_WRITING_GAME = (
    r"Two players $A$ and $B$ are playing a game by taking turns writing numbers from the set "
    r"$\{1, \dots, N\}$, where $N$ is a positive integer. Player $A$ starts the game by writing the number $1$. If a player "
    r"writes the number $n$, then the other player can write either $n+1$ or $2n$, provided "
    r"the number does not exceed $N$. The player who writes the number $N$ wins. We say that "
    r"$N$ is of type $A$ if player $A$ has a winning strategy, and of type $B$ if player $B$ "
    r"has a winning strategy. Find the least $N > 400$ such that it is a type B number."
)

PATH_PARTITION = (
    r"Let $P_n$ be a path on $n$ vertices and let $\lambda$ be a positive real number. "
    r"Define $Z_{P_n}(\lambda)=\sum_{I\in\mathcal I(P_n)}\lambda^{|I|}$, where "
    r"$\mathcal I(P_n)$ is the set of independent sets of $P_n$. "
    r"Compute $Z_{P_{15}}(\lambda)$ in terms of $\lambda$."
)

RATIONAL_F2 = (
    r"Suppose that a function $f:\mathbb{Q}\rightarrow\mathbb{F}_2$ satisfies "
    r"$f(r)+f(r')=1$ for all distinct rational numbers $r,r'$ satisfying either "
    r"$r+r'=0$ or $r+r'=1$ or $rr'=1$. Suppose further that $f(11/3)=1$. "
    r"Evaluate $f(7/3)+f(11/4)+f(2024/5)+f(109/11)+f(3/71)+f(5/204)$."
)

DIGIT_SUM_WINDOW = (
    r"Let $S(n)$ be the sum of the digits in the decimal representation of a positive "
    r"integer $n$. Find the smallest positive integer $n$ such that "
    r"$S(n)S(n+1)\cdots S(n+37)$ is not a multiple of $11$."
)


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def test_certified_finite_game_and_path_partition_routes_bypass_the_model():
    cases = (
        (NUMBER_WRITING_GAME, r"\boxed{512}", "number_writing_game"),
        (
            PATH_PARTITION,
            r"\boxed{1+15\lambda+91\lambda^{2}+286\lambda^{3}+495\lambda^{4}+462\lambda^{5}+210\lambda^{6}+36\lambda^{7}+\lambda^{8}}",
            "path_independent_set_partition",
        ),
    )

    for problem, expected, operation in cases:
        result = ReasoningAgent(_NoModelClient()).solve(
            problem + r" Remember to put your final answer within \boxed{}.", {}
        )
        assert result["final_response"] == expected
        selection = next(step for step in result["trace"] if step["step"] == "selection")
        assert selection["content"]["source"] == "sympy_verified"
        tool_step = next(step for step in result["trace"] if step["step"] == "tool_evidence")
        assert operation in tool_step["content"]["operations"]


def test_certified_rational_constraint_and_digit_window_routes_bypass_the_model():
    cases = (
        (RATIONAL_F2, r"\boxed{1}", "rational_f2_constraint"),
        (DIGIT_SUM_WINDOW, r"\boxed{999981}", "digit_sum_window"),
    )
    for problem, expected, operation in cases:
        result = ReasoningAgent(_NoModelClient()).solve(
            problem + r" Remember to put your final answer within \boxed{}.", {}
        )
        assert result["final_response"] == expected
        tool_step = next(step for step in result["trace"] if step["step"] == "tool_evidence")
        assert operation in tool_step["content"]["operations"]


def test_new_certified_routes_refuse_missing_rules_and_extra_proof_obligations():
    missing_relation = RATIONAL_F2.replace(" or $rr'=1$", "")
    assert not _evidence(missing_relation)
    rational_proof = _evidence(RATIONAL_F2 + " Explain why the value is forced.")
    assert rational_proof[0].scope == "subexpression"
    assert rational_proof[0].operation == "rational_f2_constraint_check"

    digit_proof = _evidence(DIGIT_SUM_WINDOW + " Prove that your integer is minimal.")
    assert digit_proof[0].scope == "subexpression"
    assert digit_proof[0].operation == "digit_sum_window_check"


def test_certified_single_result_cannot_bypass_an_unrelated_second_part():
    problem = DIGIT_SUM_WINDOW[:-1] + " and determine the digit sum of that integer."
    evidence = _evidence(problem)

    assert len(build_problem_spec(problem).goals) == 2
    assert evidence[0].operation == "digit_sum_window"
    assert evidence[0].scope == "subexpression"


def test_finite_game_whole_answer_requires_every_game_rule():
    assert _evidence(NUMBER_WRITING_GAME)[0].scope == "whole_goal"

    ambiguous = (
        NUMBER_WRITING_GAME.replace("The player who writes the number $N$ wins. ", ""),
        NUMBER_WRITING_GAME.replace("$n+1$ or $2n$", "$n+2$ or $2n$"),
        NUMBER_WRITING_GAME.replace("starts the game by writing the number $1$", "starts at $2$"),
        NUMBER_WRITING_GAME.replace("does not exceed $N$", "does not exceed $N+1$"),
        NUMBER_WRITING_GAME.replace("a positive integer", "a positive real number"),
    )
    for problem in ambiguous:
        assert not any(item.operation.startswith("number_writing_game") for item in _evidence(problem))


def test_finite_game_requested_strategy_keeps_dp_value_as_subexpression():
    evidence = _evidence(NUMBER_WRITING_GAME + " Also give a winning strategy and justify it.")

    assert len(evidence) == 1
    assert evidence[0].operation == "number_writing_game_state_check"
    assert evidence[0].scope == "subexpression"
    assert evidence[0].result == "512"


def test_path_partition_whole_answer_requires_the_complete_definition():
    evidence = _evidence(PATH_PARTITION)
    assert len(evidence) == 1
    assert evidence[0].operation == "path_independent_set_partition"
    assert evidence[0].scope == "whole_goal"
    small = _evidence(PATH_PARTITION.replace("15", "4"))
    assert small[0].result == r"1+4\lambda+3\lambda^{2}"

    ambiguous = (
        PATH_PARTITION.replace("a path", "a cycle"),
        PATH_PARTITION.replace("a positive real number", "a real number"),
        PATH_PARTITION.replace("independent sets", "vertex subsets"),
        PATH_PARTITION.replace(r"\lambda^{|I|}", r"\lambda^{2|I|}"),
    )
    for problem in ambiguous:
        assert not any("path_" in item.operation for item in _evidence(problem))


def test_path_partition_proof_obligation_is_local_evidence_only():
    problem = PATH_PARTITION + r" Also derive and prove the recurrence $Z_n=Z_{n-1}+\lambda Z_{n-2}$."
    evidence = _evidence(problem)

    assert len(evidence) == 1
    assert evidence[0].operation == "path_partition_recurrence_check"
    assert evidence[0].scope == "subexpression"
