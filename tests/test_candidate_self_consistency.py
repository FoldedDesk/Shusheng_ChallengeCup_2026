from classifier.problem_spec import build_problem_spec
from reasoning.candidate_selector import (
    assess_candidate,
    candidate_consistency_reasons,
)
from user_agent import ReasoningAgent


SIMPLE_SPEC = build_problem_spec("某项统计量已知为17，求该统计量的数值。")


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_false_closed_numeric_identity_is_hard_rejected():
    candidate = (
        r"FINAL: \boxed{3}"
        "\n"
        r"核验：\frac{1}{2}\cdot 6=2。"
    )

    assessment = assess_candidate(candidate, "solve", SIMPLE_SPEC, ())

    assert assessment.validation_tier == "rejected"
    assert "numeric_identity_conflict" in assessment.rejected_reasons


def test_correct_numeric_identities_and_normal_rounding_are_not_rejected():
    candidates = (
        r"FINAL: \boxed{3}" "\n" r"核验：1/2*6=3。",
        r"FINAL: \boxed{3}" "\n" r"核验：(1+2)^2=9。",
        r"FINAL: \boxed{0.333}" "\n" r"核验：1/3=0.333。",
        r"FINAL: \boxed{0.333}" "\n" r"核验：0.3333=0.333。",
        (
            r"C(9,3)=\frac{9!}{3!6!}=84，"
            r"\binom{5}{2}\binom{4}{1}=40，40/84=10/21。"
            "\n" r"FINAL: \boxed{\frac{10}{21}}"
        ),
        (
            r"\sum_{i=1}^{12}\deg(v_i)=2E，12\times5=2E，"
            r"E=\frac{12\times5}{2}=30。"
            "\n" r"FINAL: \boxed{30}"
        ),
        (
            r"\binom{8}{3}+\binom{8}{3}=2\binom{8}{3}，"
            r"\binom{8}{3}=56，2\binom{8}{3}=2\times56=112。"
            "\n" r"FINAL: \boxed{112}"
        ),
    )

    for candidate in candidates:
        assert "numeric_identity_conflict" not in candidate_consistency_reasons(
            candidate, SIMPLE_SPEC
        )


def test_boolean_algebra_idempotence_is_not_treated_as_false_integer_arithmetic():
    spec = build_problem_spec(
        "设布尔代数中x+y=1且xy=0，化简表达式(x+z)(y+z)，并使用分配律说明。"
    )
    candidate = (
        r"利用分配律，(x+z)(y+z)=xy+xz+yz+z^2。"
        r"由xy=0、z^2=z及x+y=1，得z(x+y+1)=z(1+1)=z。"
        "\n" r"FINAL: \boxed{z}"
    )

    assessment = assess_candidate(candidate, "solve", spec, ())

    assert "numeric_identity_conflict" not in assessment.rejected_reasons
    assert assessment.complete_goals, assessment.rejected_reasons


def test_variable_assignments_and_ordinary_intermediate_values_are_not_numeric_identities():
    candidates = (
        r"FINAL: \boxed{2}" "\n" r"x_0=1,\quad x_1=2。",
        r"FINAL: \boxed{2}" "\n" r"中间量 a=1，b=2。",
        r"FINAL: \boxed{2}" "\n" r"先有 y=1，再得 x=2。",
        r"FINAL: \boxed{1}" "\n" r"f'(0)=\frac{1}{1+0}=1。",
        r"FINAL: \boxed{-1}" "\n" r"f''(0)=-\frac{1}{(1+0)^2}=-1。",
        r"FINAL: \boxed{2}" "\n" r"f'''(0)=\frac{2}{(1+0)^3}=2。",
    )

    for candidate in candidates:
        assert "numeric_identity_conflict" not in candidate_consistency_reasons(
            candidate, SIMPLE_SPEC
        )


def test_labelled_or_boxed_final_conflicting_with_terminal_conclusion_is_rejected():
    candidates = (
        r"FINAL: \boxed{3}" "\n" "因此答案为2。",
        r"FINAL: \boxed{x=3}" "\n" "Therefore x=2.",
        r"\boxed{3}" "\n" "故答案为2。",
        r"FINAL: x=3" "\n" r"CONCLUSION: x=2",
        r"FINAL: \boxed{35}" "\n" "因此，满足条件的置换数为105。",
    )

    for candidate in candidates:
        assessment = assess_candidate(candidate, "solve", SIMPLE_SPEC, ())
        assert assessment.validation_tier == "rejected"
        assert "final_conclusion_conflict" in assessment.rejected_reasons


def test_equivalent_or_different_target_conclusions_are_not_conflicts():
    candidates = (
        r"FINAL: \boxed{\frac{1}{2}}" "\n" "因此答案为0.5。",
        r"FINAL: \boxed{x=1}" "\n" "因此 y=2。",
        r"FINAL: \boxed{x=1,y=2}" "\n" "故 y=2。",
        r"FINAL: \boxed{x=3}" "\n" "因此 x^2=9。",
        r"FINAL: \boxed{42}" "\n" r"Check: \boxed{6}",
        (
            "由于每个顶点的度数为5，所以：\n"
            r"\[12\times5=2E\]"
            "\n" r"FINAL: \boxed{30}"
        ),
        (
            r"FINAL: \boxed{\frac{1}{8}}" "\n"
            r"故 P(X>3)=(1-p)^3=(0.5)^3=1/8。"
        ),
        (
            r"FINAL: \boxed{f(x)=x^{-1/2}\in L^1[0,1],\ "
            r"\int_0^1 x^{-1/2}\,dx=2}" "\n"
            r"此处 \alpha=1/2<1，故 f\in L^1[0,1]。"
        ),
        (
            r"FINAL: \boxed{a_n=1}" "\n"
            r"因此 b_n=0，即 a_n=1 对所有 n 成立。"
        ),
        (
            r"FINAL: \boxed{y(0.2)\approx1.21}" "\n"
            r"因此，y(0.2) 的近似值为 \(1.21\)。"
        ),
        (
            r"FINAL: \boxed{y=xe^{-2x}}" "\n"
            r"故显式解为 $y=xe^{-2x}$（定义域 $x\in\mathbb R$）。"
        ),
    )

    for candidate in candidates:
        assert "final_conclusion_conflict" not in candidate_consistency_reasons(
            candidate, SIMPLE_SPEC
        )


def test_numeric_equality_chain_conflicting_with_final_is_rejected():
    candidate = (
        r"FINAL: \boxed{\frac{1}{8}}" "\n"
        r"故 P(X>3)=(1-p)^3=(0.5)^3=1/4。"
    )

    assert "final_conclusion_conflict" in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_conflicting_categorical_conclusions_are_rejected():
    assert "final_conclusion_conflict" in candidate_consistency_reasons(
        "FINAL: A\n因此答案为B。", SIMPLE_SPEC
    )


def test_terminal_self_correction_of_statistical_target_rejects_first_final():
    candidate = (
        r"FINAL: \boxed{E[X]=\frac43,\ Var(X)=\frac59}" "\n"
        r"重算得 E[X]=\frac53，Var(X)=\frac59。因此正确结论为" "\n"
        r"\[\boxed{E[X]=\frac53,\ Var(X)=\frac59}\]"
    )

    assert "final_conclusion_conflict" in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_unrelated_check_box_after_final_is_not_a_concluding_correction():
    candidate = r"FINAL: \boxed{42}" "\n" r"Check: \boxed{6}"

    assert "final_conclusion_conflict" not in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_concluding_intermediate_check_box_does_not_replace_final():
    candidate = (
        r"FINAL: \boxed{5}" "\n"
        r"CHECK: 因为2+2=4，因此中间校验量为 \boxed{4}"
    )

    assert "final_conclusion_conflict" not in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_unboxed_early_final_conflicting_with_explicit_terminal_correction_is_rejected():
    candidate = (
        r"FINAL: 5" "\n"
        r"Rechecking found an error. Corrected answer: \boxed{4}"
    )

    assert "final_conclusion_conflict" in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_unboxed_early_final_with_unlabelled_check_box_is_not_auto_corrected():
    candidate = r"FINAL: 5" "\n" r"Check: \boxed{4}"

    assert "final_conclusion_conflict" not in candidate_consistency_reasons(
        candidate, SIMPLE_SPEC
    )


def test_unlabelled_multiline_nonproof_body_is_rejected_as_scratch_work():
    assessment = assess_candidate(
        "Assume n=10.\n2+3=5", "solve", SIMPLE_SPEC, (),
        extraction_method="whole_response",
    )

    assert assessment.validation_tier == "rejected"
    assert "unlabelled_process_body" in assessment.rejected_reasons


def test_inconsistent_first_candidate_is_rejected_and_triggers_rescue():
    client = RecordingClient([
        r"FINAL: \boxed{17}" "\n" r"核验：1/2*6=2。",
        r"FINAL: \boxed{17}",
    ])

    result = ReasoningAgent(client).solve(SIMPLE_SPEC.problem_text, {})

    assert result["final_response"] == "17"
    assert len(client.calls) == 2
    admission = next(
        item["content"]
        for item in result["trace"]
        if item["step"] == "review_admission"
    )
    assert admission["mode"] == "rescue"
    validation = next(
        item["content"]
        for item in result["trace"]
        if item["step"] == "validation"
    )
    solve_reasons = {
        reason
        for source, item in validation.items()
        if source.startswith("solve")
        for reason in item["rejected_reasons"]
    }
    assert "numeric_identity_conflict" in solve_reasons


def test_conflicting_final_conclusion_also_triggers_rescue():
    client = RecordingClient([
        r"FINAL: \boxed{17}" "\n" "因此答案为16。",
        r"FINAL: \boxed{17}",
    ])

    result = ReasoningAgent(client).solve(SIMPLE_SPEC.problem_text, {})

    assert result["final_response"] == "17"
    assert len(client.calls) == 2
    admission = next(
        item["content"]
        for item in result["trace"]
        if item["step"] == "review_admission"
    )
    assert admission["mode"] == "rescue"
    validation = next(
        item["content"]
        for item in result["trace"]
        if item["step"] == "validation"
    )
    solve_reasons = {
        reason
        for source, item in validation.items()
        if source.startswith("solve")
        for reason in item["rejected_reasons"]
    }
    assert "final_conclusion_conflict" in solve_reasons
