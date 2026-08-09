from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_judge_replay import (
    audit_problem_specs,
    evaluate,
    filter_items_by_ids,
    load_judge_items,
    main,
    parse_id_filter,
    write_problem_only,
)


def test_id_filter_supports_comma_separated_ids_and_inclusive_ranges() -> None:
    items = [{"idx": str(idx)} for idx in (2, 86, 87, 100, 111, 112)]

    selected = filter_items_by_ids(items, parse_id_filter("2, 87-111"))

    assert [item["idx"] for item in selected] == ["2", "87", "100", "111"]


@pytest.mark.parametrize("value", ["", "2-", "4-2", "1,,2", "one", "1-2-3"])
def test_id_filter_rejects_malformed_or_descending_segments(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_id_filter(value)


def test_main_applies_id_filter_before_export_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge_dir = tmp_path / "judge"
    output_dir = tmp_path / "outputs"
    judge_dir.mkdir()
    output_dir.mkdir()
    for idx in range(1, 4):
        (judge_dir / f"{idx}.json").write_text(json.dumps({
            "idx": idx,
            "problem": r"Compute 1+1. Put the final answer in \boxed{}.",
            "reward_model": {"ground_truth": "2"},
        }), encoding="utf-8")
    (output_dir / "2.json").write_text(json.dumps({
        "final_response": r"\boxed{2}",
        "trace": [],
    }), encoding="utf-8")
    destination = tmp_path / "questions.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_judge_replay.py",
        "--judge-output-dir", str(judge_dir),
        "--agent-output-dir", str(output_dir),
        "--write-problem-only", str(destination),
        "--ids", "2",
    ])

    assert main() == 0

    exported = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(capsys.readouterr().out)
    assert [item["idx"] for item in exported] == ["2"]
    assert summary["total"] == 1
    assert summary["processed"] == 1


def test_problem_only_export_never_contains_ground_truth(tmp_path: Path) -> None:
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "7.json").write_text(json.dumps({
        "idx": 7,
        "problem": "Compute 1+1. Put the final answer in \\boxed{}.",
        "reward_model": {"ground_truth": "2"},
    }), encoding="utf-8")

    items = load_judge_items(judge_dir)
    destination = tmp_path / "questions.jsonl"
    write_problem_only(items, destination)
    runtime = json.loads(destination.read_text(encoding="utf-8"))

    assert runtime == {"idx": "7", "problem": items[0]["problem"], "source": "judge_replay"}
    assert "answer" not in runtime
    assert "ground_truth" not in runtime


def test_replay_evaluator_reports_boxed_semantic_hit(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "7.json").write_text(json.dumps({
        "final_response": r"\boxed{2}",
        "trace": [{"step": "selection", "content": {"source": "solve"}}],
    }), encoding="utf-8")
    items = [{"idx": "7", "problem": "Compute.", "answer": "2", "source": "judge_replay"}]

    summary = evaluate(items, output_dir)

    assert summary["processed"] == 1
    assert summary["boxed_contract_count"] == 1
    assert summary["semantic_standard_answer_hits"] == 1
    assert summary["structural_invalid_answers"] == 0
    assert summary["missing_subquestion_count"] == 0
    assert summary["offline_validation_tiers"] == {"complete": 1}


def test_replay_evaluator_counts_missing_subquestions(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "8.json").write_text(json.dumps({
        "final_response": r"\boxed{x=2}",
        "trace": [],
    }), encoding="utf-8")
    items = [{
        "idx": "8",
        "problem": "Find x from x+1=3 and determine y from 2y=6. Put your final answer in \\boxed{}.",
        "answer": "x=2, y=3",
        "source": "judge_replay",
    }]

    summary = evaluate(items, output_dir)

    assert summary["missing_subquestion_count"] == 1
    assert summary["items_missing_subquestion_ids"] == ["8"]
    assert summary["offline_validation_tiers"] == {"degraded": 1}


def test_problem_spec_audit_does_not_need_ground_truth() -> None:
    summary = audit_problem_specs([{
        "idx": "9",
        "problem": r"Calculate 2+2. Put your final answer in \boxed{}.",
        "source": "judge_replay",
    }])

    assert summary["total"] == 1
    assert summary["distributions"]["wrapper"] == {"boxed": 1}
    assert summary["empty_or_blank_goal_count"] == 0
    assert summary["effective_whole_tool_route_count"] == 0
    assert summary["deep_reasoning_route_count"] + summary["quick_response_route_count"] == 1
    assert summary["offline_reference_validation_tiers"] == {}
