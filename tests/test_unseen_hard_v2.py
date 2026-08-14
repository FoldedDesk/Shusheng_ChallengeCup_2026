from __future__ import annotations

import json

from scripts.audit_unseen_hard_v2 import (
    DATASET,
    _method_errors,
    _string_similarity,
    abstract_template,
    audit,
    formula_skeletons,
)
from scripts.verify_unseen_hard_v2_answers import verify


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v2_answer_key_has_three_way_verification_and_passes():
    report = verify()
    assert report == {
        "rows": 14,
        "verified": 14,
        "passed": True,
        "failed_indices": [],
        "unchecked_indices": [],
        "runtime_answer_exposure": False,
        "agent_executed": False,
    }


def test_v2_isolation_audit_passes_without_whole_goal_routes():
    report = audit()
    assert report["passed"]
    assert report["failure_count"] == 0
    assert report["whole_tool_route_indices"] == []
    assert report["agent_executed"] is False
    assert report["answer_exposed_to_runtime"] is False


def test_v2_rubric_has_three_layers_three_checks_and_balanced_languages():
    rows = _rows()
    assert len(rows) == 14
    assert all(row["difficulty"]["score"] in {3, 4} for row in rows)
    assert all(len(set(row["difficulty"]["reasoning_layers"])) >= 3 for row in rows)
    assert all(len(set(row["difficulty"]["independent_checks"])) >= 3 for row in rows)
    chinese = sum(any("\u4e00" <= char <= "\u9fff" for char in row["problem"]) for row in rows)
    assert chinese == len(rows) - chinese == 7


def test_number_abstracted_template_detects_parameter_only_rewrites():
    first = (
        "Determine all 17 by 17 matrices A satisfying A^3=I and trace(A)=4, "
        "with a rigorous proof that the classification is complete."
    )
    second = (
        "Determine all 19 by 19 matrices B satisfying B^5=I and trace(B)=6, "
        "with a rigorous proof that the classification is complete."
    )
    assert _string_similarity(abstract_template(first), abstract_template(second)) > 0.8


def test_formula_skeleton_ignores_numbers_but_preserves_structure():
    first = r"Prove that $x^4+3x^2+7=0$ has no real roots."
    second = r"Prove that $y^8+9y^4+11=0$ has no real roots."
    left, right = formula_skeletons(first), formula_skeletons(second)
    assert left and right
    assert _string_similarity(left[0], right[0]) > 0.8


def test_forbidden_v1_method_fingerprint_is_rejected():
    row = {
        "method_fingerprint": {
            "domain": "number_theory",
            "object": "positive_integer_pairs",
            "target": "all_solutions",
            "constraints": "binary_quadratic_equation",
            "primary_method": "vieta_jumping_descent",
        }
    }
    errors = _method_errors(row, set())
    assert any("forbidden V1/current-tool method family" in error for error in errors)
