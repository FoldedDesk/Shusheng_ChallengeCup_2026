import json
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from reasoning.candidate_selector import assess_candidate


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
