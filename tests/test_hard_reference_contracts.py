import json
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from reasoning.candidate_selector import assess_candidate, choose_candidate
from reasoning.math_equivalence import equivalent_answers


DATASET = Path(__file__).parents[1] / "sample_data" / "judge1_style_112_hard_v1.jsonl"


def _rows():
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("row", _rows(), ids=lambda row: str(row["idx"]))
def test_reference_form_is_not_rejected_by_the_public_answer_contract(row):
    spec = build_problem_spec(row["problem"])
    assessment = assess_candidate(row["answer"], "offline_reference", spec, ())

    assert assessment.accepted, assessment.rejected_reasons


def test_explicit_construction_is_one_result_and_needs_no_narrated_check():
    row = next(row for row in _rows() if row["idx"] == 5034)
    spec = build_problem_spec(row["problem"])

    assert spec.profile.task_kind == "construction"
    assert len(spec.goals) == 1
    assert assess_candidate(row["answer"], "construction", spec, ()).accepted


def test_explicit_verification_request_still_requires_support():
    spec = build_problem_spec(
        "Construct three integers whose sum is 6, and verify their sum explicitly."
    )

    bare = assess_candidate(r"\{1,2,3\}", "bare", spec, ())
    checked = assess_candidate(
        r"Take \{1,2,3\}; their sum is 1+2+3=6, so the condition is verified.",
        "checked",
        spec,
        (),
    )
    assert not bare.accepted
    assert "missing_construction_verification" in bare.rejected_reasons
    assert checked.accepted


@pytest.mark.parametrize(
    "idx",
    [5084, 5086, 5088, 5095, 5096],
)
def test_single_math_object_can_satisfy_a_result_specific_contract(idx):
    row = next(row for row in _rows() if row["idx"] == idx)
    assessment = assess_candidate(
        row["answer"], "single_math_object", build_problem_spec(row["problem"]), ()
    )

    assert assessment.accepted, assessment.rejected_reasons


def test_boxed_bare_operator_norm_formula_covers_the_requested_result():
    row = next(row for row in _rows() if row["idx"] == 5096)
    assessment = assess_candidate(
        r"\boxed{\frac{1}{n!}}",
        "boxed_formula",
        build_problem_spec(row["problem"]),
        (),
    )

    assert assessment.accepted, assessment.rejected_reasons


@pytest.mark.parametrize(
    "notation",
    (
        r"\|L\|=1", r"\lVert L\rVert=1", "||L||=1", "算子范数为1",
        r"\lVert L\rVert_1=1", r"||L||_1=1",
    ),
)
def test_operator_norm_requirement_accepts_common_explicit_notations(notation):
    spec = build_problem_spec(
        "在C[0,1]配备一致范数，证明评价泛函L(f)=f(1/2)有界并求其算子范数。"
    )
    answer = (
        f"有界且{notation}。对任意f，依一致范数定义有"
        "|L(f)|=|f(1/2)|<=sup|f(x)|=||f||，故||L||<=1。"
        "取常函数f(x)=1，则||f||=1且L(f)=1，故||L||>=1，综上取等。"
    )
    assessment = assess_candidate(answer, "operator_norm_notation", spec, ())

    assert assessment.complete_goals, assessment.rejected_reasons


def test_operator_norm_requirement_rejects_boundedness_without_norm_value():
    spec = build_problem_spec(
        "在C[0,1]配备一致范数，证明评价泛函L(f)=f(1/2)有界并求其算子范数。"
    )
    assessment = assess_candidate(
        "对任意f有|L(f)|<=||f||，所以L有界。",
        "missing_norm_value",
        spec,
        (),
    )

    assert not assessment.complete_goals
    assert "missing_required_goal" in assessment.rejected_reasons


def test_point_value_absolute_bar_does_not_supply_operator_norm_value():
    spec = build_problem_spec(
        "在C[0,1]配备一致范数，证明评价泛函L(f)=f(1/2)有界并求其算子范数。"
    )
    assessment = assess_candidate(
        "取f使得|L(f)|=1，因此L有界。",
        "point_value_only",
        spec,
        (),
    )

    assert not assessment.complete_goals
    assert "missing_required_goal" in assessment.rejected_reasons


def test_subscripted_operator_norm_does_not_lose_to_a_wrong_complete_value():
    spec = build_problem_spec("计算算子L的1-范数。")
    correct = assess_candidate(r"\lVert L\rVert_1=1", "solve", spec, ())
    corroboration = assess_candidate(r"||L||_1=1", "rescue", spec, ())
    wrong = assess_candidate("算子范数为7", "verify", spec, ())

    assert correct.complete_goals, correct.rejected_reasons
    selected = choose_candidate([correct, corroboration, wrong])
    assert selected in {correct, corroboration}
    assert selected is not wrong


def test_norm_notation_is_not_equivalent_to_absolute_value():
    assert equivalent_answers(r"\lVert A\rVert=2", r"||A||=2")
    assert not equivalent_answers(r"\lVert A\rVert=2", r"|A|=2")
