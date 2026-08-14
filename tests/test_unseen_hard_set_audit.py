from __future__ import annotations

import json

from scripts.audit_unseen_hard_set import audit, normalized_problem, similarity


def _row(idx: int, problem: str) -> dict:
    return {
        "idx": idx,
        "problem": problem,
        "answer": "42",
        "subject": "数论",
        "source": "isolated_test",
        "difficulty": {
            "level": "hard",
            "reasoning_layers": ["derive a congruence obstruction", "construct all equality cases"],
            "common_traps": ["checking necessity but not sufficiency"],
            "independent_checks": ["modular proof", "bounded exhaustive enumeration"],
        },
    }


def test_normalization_removes_only_terminal_answer_instruction():
    base = "Determine all integers n satisfying n^2 = 4 and prove completeness."
    decorated = base + " Remember to put your final answer within \\boxed{}."
    assert normalized_problem(base) == normalized_problem(decorated)


def test_similarity_detects_parameter_only_rewrites():
    left = "Determine all integers n such that n^2+n is divisible by 7, and prove completeness."
    right = "Determine all integers n such that n^2+n is divisible by 11, and prove completeness."
    assert similarity(left, right) > 0.8


def test_audit_rejects_missing_difficulty_evidence(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = _row(
        1,
        "Determine all positive integers n satisfying n^2+1 = 2^k for some integer k, "
        "and prove that your classification is complete. Remember to put your final answer within \\boxed{}.",
    )
    row["difficulty"]["independent_checks"] = ["one check"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = audit(path)
    assert not report["passed"]
    assert "at least two distinct independent_checks" in " ".join(report["failures"][0]["errors"])


def test_audit_accepts_a_well_documented_isolated_item(tmp_path, monkeypatch):
    path = tmp_path / "candidate.jsonl"
    row = _row(
        2,
        "Determine all positive integers n for which n^2+7 is a power of two. "
        "Prove both necessity and sufficiency, and identify the maximum possible n under n<1000. "
        "Remember to put your final answer within \\boxed{}.",
    )
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr("scripts.audit_unseen_hard_set.existing_problems", lambda candidate: [])
    report = audit(path)
    assert report["passed"]
    assert report["rows"] == 1
