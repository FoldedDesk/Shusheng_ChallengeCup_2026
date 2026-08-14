from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_unseen_hard_v4 import (
    ACTIVE_DEVELOPMENT_FAMILY_PATTERNS,
    DATASET,
    EXPECTED_INDICES,
    FROZEN_SHA256,
    METHOD_FIELDS,
    RETIRED_FAMILY_MARKERS,
    audit,
)
from scripts.verify_unseen_hard_v4_answers import VERIFICATION_METHODS, verify


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v4_answer_certificate_passes_with_three_methods_per_item():
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
    assert set(VERIFICATION_METHODS) == EXPECTED_INDICES
    assert all(len(set(methods)) >= 3 for methods in VERIFICATION_METHODS.values())


def test_v4_full_isolation_and_whole_goal_audit_passes():
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


def test_v4_dataset_is_frozen_balanced_hard_and_single_target():
    rows = _rows()
    assert len(rows) == 14
    assert {row["idx"] for row in rows} == EXPECTED_INDICES
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == FROZEN_SHA256
    chinese = sum(any("\u4e00" <= char <= "\u9fff" for char in row["problem"]) for row in rows)
    assert chinese == len(rows) - chinese == 7
    assert all(row["source"] == "unseen_hard_holdout_v4" for row in rows)
    assert all(row["difficulty"]["level"] == "hard" for row in rows)
    assert all(row["difficulty"]["score"] in {3, 4} for row in rows)
    assert all(len(set(row["difficulty"]["reasoning_layers"])) >= 3 for row in rows)
    assert all(len(set(row["difficulty"]["common_traps"])) >= 2 for row in rows)
    assert all(len(set(row["difficulty"]["independent_checks"])) >= 3 for row in rows)


def test_v4_explicitly_avoids_retired_and_active_development_families():
    for row in _rows():
        fingerprint = row["method_fingerprint"]
        assert set(fingerprint) >= set(METHOD_FIELDS)
        joined = " ".join(str(fingerprint[field]).lower() for field in METHOD_FIELDS)
        assert not any(marker in joined for marker in RETIRED_FAMILY_MARKERS)
        assert not any(pattern.search(row["problem"]) for pattern in ACTIVE_DEVELOPMENT_FAMILY_PATTERNS)


def test_v4_answers_are_absent_from_runtime_modules():
    runtime_paths = [
        ROOT / "user_agent.py",
        *sorted((ROOT / "classifier").glob("*.py")),
        *sorted((ROOT / "core").glob("*.py")),
        *sorted((ROOT / "rag").glob("*.py")),
        *sorted((ROOT / "reasoning").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
    ]
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert "unseen_hard_holdout_v4" not in text
        assert not any(str(index) in text for index in EXPECTED_INDICES)


def test_v4_verifier_is_authoring_only_and_does_not_import_agent():
    verifier = (ROOT / "scripts" / "verify_unseen_hard_v4_answers.py").read_text(encoding="utf-8")
    assert "from user_agent" not in verifier
    assert "ReasoningAgent" not in verifier
    assert "agent.solve(" not in verifier
    assert "main.py" not in verifier
