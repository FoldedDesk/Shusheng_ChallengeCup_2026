from __future__ import annotations

import hashlib
import json

from scripts.audit_unseen_hard_v3 import (
    DATASET,
    FROZEN_SHA256,
    FORBIDDEN_V2_FAMILIES,
    METHOD_FIELDS,
    _string_similarity,
    abstract_template,
    audit,
    formula_skeletons,
    nfkc_body,
    normalized_problem,
)
from scripts.verify_unseen_hard_v3_answers import VERIFICATION_METHODS, verify


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v3_answer_certificate_passes_with_three_methods_per_item():
    report = verify()
    assert report == {
        "rows": 14,
        "verified": 14,
        "passed": True,
        "failed_indices": [],
        "unchecked_indices": [],
        "minimum_independent_methods": 3,
        "runtime_answer_exposure": False,
        "agent_executed": False,
    }
    assert set(VERIFICATION_METHODS) == set(range(92001, 92015))
    assert all(len(set(methods)) >= 3 for methods in VERIFICATION_METHODS.values())


def test_v3_full_isolation_and_whole_goal_audit_passes():
    report = audit()
    assert report["passed"]
    assert report["failure_count"] == 0
    assert report["global_errors"] == []
    assert report["whole_tool_route_indices"] == []
    assert report["sympy_exact_whole_goal_count"] == 0
    assert report["maxima"]["nfkc_lexical"] < report["thresholds"]["nfkc_lexical"]
    assert report["maxima"]["number_variable_template"] < report["thresholds"]["number_variable_template"]
    assert report["maxima"]["formula_plus_context"] < report["thresholds"]["formula_plus_context"]
    assert report["maxima"]["prior_method_overlap_fields"] <= 1
    assert report["maxima"]["internal_method_overlap_fields"] <= 1
    assert report["agent_executed"] is False
    assert report["answer_exposed_to_runtime"] is False


def test_v3_dataset_is_frozen_balanced_hard_and_single_target():
    rows = _rows()
    assert len(rows) == 14
    assert {row["idx"] for row in rows} == set(range(92001, 92015))
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == FROZEN_SHA256
    chinese = sum(any("\u4e00" <= char <= "\u9fff" for char in row["problem"]) for row in rows)
    assert chinese == len(rows) - chinese == 7
    assert all(row["source"] == "unseen_hard_holdout_v3" for row in rows)
    assert all(row["difficulty"]["level"] == "hard" for row in rows)
    assert all(row["difficulty"]["score"] in {3, 4} for row in rows)
    assert all(len(set(row["difficulty"]["reasoning_layers"])) >= 3 for row in rows)
    assert all(len(set(row["difficulty"]["common_traps"])) >= 2 for row in rows)
    assert all(len(set(row["difficulty"]["independent_checks"])) >= 3 for row in rows)


def test_nfkc_lexical_normalization_collapses_compatibility_variants():
    ordinary = r"Determine A^2 for a labeled matrix with entries 1, 2, and 3."
    compatibility = r"Ｄｅｔｅｒｍｉｎｅ Ａ^２ for a labeled matrix with entries １, ２, and ３."
    assert nfkc_body(compatibility).startswith("Determine A^2")
    assert normalized_problem(ordinary) == normalized_problem(compatibility)


def test_abstract_template_removes_numbers_and_single_variables():
    first = r"Prove that $x^7\equiv 11\pmod{13}$ has exactly 4 solutions."
    second = r"Prove that $y^9\equiv 17\pmod{19}$ has exactly 6 solutions."
    assert _string_similarity(abstract_template(first), abstract_template(second)) > 0.75


def test_formula_skeleton_abstracts_coefficients_but_keeps_operators():
    first = r"Classify solutions of $x^4+3x^2=7$."
    second = r"Classify solutions of $y^8+5y^6=11$."
    left, right = formula_skeletons(first), formula_skeletons(second)
    assert left and right
    assert _string_similarity(left[0], right[0]) > 0.75


def test_v3_avoids_explicitly_retired_v2_families():
    rows = _rows()
    for row in rows:
        fingerprint = row["method_fingerprint"]
        assert set(fingerprint) >= set(METHOD_FIELDS)
        joined = " ".join(str(fingerprint[field]).lower() for field in METHOD_FIELDS)
        assert not any(family in joined for family in FORBIDDEN_V2_FAMILIES)
