"""Compare two completed offline replays against saved judge references."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
from scripts.evaluate_judge_replay import load_judge_items


def _selection(record: dict) -> dict:
    return next(
        (
            step.get("content", {})
            for step in record.get("trace", [])
            if step.get("step") == "selection"
        ),
        {},
    )


def _trace_content(record: dict, step_name: str) -> dict:
    return next(
        (
            step.get("content", {})
            for step in record.get("trace", [])
            if step.get("step") == step_name
            and isinstance(step.get("content", {}), dict)
        ),
        {},
    )


def _call_stages(record: dict) -> list[dict]:
    return [
        step.get("content", {})
        for step in record.get("trace", [])
        if str(step.get("step", "")).startswith("model_call")
    ]


def _calls(record: dict) -> tuple[int, int, int]:
    stages = _call_stages(record)
    return (
        len(stages),
        sum(bool(item.get("provider_truncated")) for item in stages),
        sum(
            int(item.get("elapsed_ms", 0))
            for item in stages
            if isinstance(item.get("elapsed_ms", 0), (int, float))
        ),
    )


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _module_name(stage: str) -> str:
    value = str(stage or "unknown").casefold()
    if value == "candidate_audit":
        return "candidate_audit"
    if value in {"primary", "mog_route_a", "route_a"}:
        return "main_solve"
    if "independent" in value or value in {"quick_consensus", "mog_route_b", "route_b"}:
        return "independent_solve"
    if "critic" in value or "audit" in value or "arbit" in value:
        return "targeted_check_or_repair"
    if "recover" in value or "rescue" in value or "continuation" in value or "synthesis" in value:
        return "recovery"
    if "method_search" in value or "plan" in value:
        return "planning"
    return value or "unknown"


def _empty_metrics() -> dict:
    return {
        "items": 0,
        "correct": 0,
        "calls": 0,
        "truncated_calls": 0,
        "elapsed_ms": 0,
        "invalid": 0,
        "candidate_audit": {
            "called_items": set(),
            "changed_final_items": set(),
            "decisions": Counter(),
        },
        "stages": defaultdict(lambda: {
            "calls": 0,
            "triggered_items": set(),
            "triggered_correct_items": set(),
            "truncated_calls": 0,
            "elapsed_ms": 0,
        }),
    }


def _update_metrics(metrics: dict, record: dict, hit: bool, idx: object) -> None:
    metrics["items"] += 1
    metrics["correct"] += int(hit)
    calls, truncated, elapsed_ms = _calls(record)
    metrics["calls"] += calls
    metrics["truncated_calls"] += truncated
    metrics["elapsed_ms"] += elapsed_ms
    answer = str(record.get("final_response", ""))
    metrics["invalid"] += int(bool(Finalizer.validate_structure(answer)))
    selection = _selection(record)
    audit_stages = [
        stage for stage in _call_stages(record)
        if stage.get("stage") == "candidate_audit"
    ]
    if audit_stages:
        metrics["candidate_audit"]["called_items"].add(idx)
        decision = str(selection.get("arbitration_decision", "unknown"))
        metrics["candidate_audit"]["decisions"][decision] += 1
        if str(selection.get("route", "")).casefold() == "candidate_audit_corrected":
            metrics["candidate_audit"]["changed_final_items"].add(idx)
    for stage in _call_stages(record):
        module = _module_name(stage.get("stage", "unknown"))
        bucket = metrics["stages"][module]
        bucket["calls"] += 1
        bucket["triggered_items"].add(idx)
        if hit:
            bucket["triggered_correct_items"].add(idx)
        bucket["truncated_calls"] += int(bool(stage.get("provider_truncated")))
        elapsed = stage.get("elapsed_ms", 0)
        if isinstance(elapsed, (int, float)):
            bucket["elapsed_ms"] += int(elapsed)


def _finalize_metrics(metrics: dict) -> dict:
    item_count = metrics["items"]
    stages = {}
    for name, bucket in sorted(metrics["stages"].items()):
        triggered = len(bucket["triggered_items"])
        triggered_correct = len(bucket["triggered_correct_items"])
        stages[name] = {
            "calls": bucket["calls"],
            "triggered_questions": triggered,
            # Diagnostic correlation only.  Net causal value is measured by
            # the matched policy comparison below.
            "triggered_final_correct": triggered_correct,
            "triggered_final_accuracy": (
                triggered_correct / triggered if triggered else 0.0
            ),
            "truncated_calls": bucket["truncated_calls"],
            "elapsed_seconds": round(bucket["elapsed_ms"] / 1000, 3),
        }
    return {
        "items": item_count,
        "correct": metrics["correct"],
        "accuracy": metrics["correct"] / item_count if item_count else 0.0,
        "calls": metrics["calls"],
        "calls_per_question": metrics["calls"] / item_count if item_count else 0.0,
        "truncated_calls": metrics["truncated_calls"],
        "elapsed_seconds": round(metrics["elapsed_ms"] / 1000, 3),
        "invalid": metrics["invalid"],
        "candidate_audit": {
            "called_questions": len(metrics["candidate_audit"]["called_items"]),
            "changed_final_answer": len(
                metrics["candidate_audit"]["changed_final_items"]
            ),
            "decisions": dict(sorted(
                metrics["candidate_audit"]["decisions"].items()
            )),
        },
        "modules": stages,
    }


def _summarize_stratum(counts: Counter) -> dict:
    before_correct = counts["right_to_wrong"] + counts["right_to_right"]
    after_correct = counts["wrong_to_right"] + counts["right_to_right"]
    return {
        "questions": sum(counts[key] for key in (
            "wrong_to_wrong", "wrong_to_right", "right_to_wrong", "right_to_right"
        )),
        "wrong_to_wrong": counts["wrong_to_wrong"],
        "wrong_to_right": counts["wrong_to_right"],
        "right_to_wrong": counts["right_to_wrong"],
        "right_to_right": counts["right_to_right"],
        "before_correct": before_correct,
        "after_correct": after_correct,
        "net_correct_gain": after_correct - before_correct,
        "before_calls": counts["before_calls"],
        "after_calls": counts["after_calls"],
        "call_delta": counts["after_calls"] - counts["before_calls"],
        "before_elapsed_seconds": round(counts["before_elapsed_ms"] / 1000, 3),
        "after_elapsed_seconds": round(counts["after_elapsed_ms"] / 1000, 3),
    }


def compare(judge_dir: Path, before_dir: Path, after_dir: Path) -> dict:
    changed: list[dict] = []
    answer_changed_count = 0
    strata: dict[str, Counter] = defaultdict(Counter)
    before_metrics = _empty_metrics()
    after_metrics = _empty_metrics()
    skipped_missing: list[object] = []
    for item in load_judge_items(judge_dir):
        before_path = before_dir / f"{item['idx']}.json"
        after_path = after_dir / f"{item['idx']}.json"
        if not before_path.exists() or not after_path.exists():
            skipped_missing.append(item["idx"])
            continue
        before = _record(before_path)
        after = _record(after_path)
        before_hit = equivalent_answers(before.get("final_response", ""), item["answer"])
        after_hit = equivalent_answers(after.get("final_response", ""), item["answer"])
        _update_metrics(before_metrics, before, before_hit, item["idx"])
        _update_metrics(after_metrics, after, after_hit, item["idx"])
        answer_changed_count += int(
            _normalized_text(before.get("final_response", ""))
            != _normalized_text(after.get("final_response", ""))
        )
        spec = build_problem_spec(item["problem"])
        before_calls, before_truncated, before_elapsed = _calls(before)
        after_calls, after_truncated, after_elapsed = _calls(after)
        transition = (
            ("right" if before_hit else "wrong")
            + "_to_"
            + ("right" if after_hit else "wrong")
        )
        for label in (
            "all",
            f"difficulty:{spec.profile.difficulty}",
            f"task:{spec.profile.task_kind}",
        ):
            strata[label][transition] += 1
            strata[label]["before_calls"] += before_calls
            strata[label]["after_calls"] += after_calls
            strata[label]["before_elapsed_ms"] += before_elapsed
            strata[label]["after_elapsed_ms"] += after_elapsed
        if before_hit == after_hit:
            continue
        before_selection = _selection(before)
        after_selection = _selection(after)
        changed.append({
            "idx": item["idx"],
            "change": "gain" if after_hit else "regression",
            "language": spec.profile.language,
            "subject": getattr(spec.profile, "primary_subject", spec.profile.subject),
            "task_kind": getattr(spec.profile, "task_kind", spec.profile.problem_type),
            "answer_shape": spec.profile.answer_shape,
            "difficulty": spec.profile.difficulty,
            "before": {
                "source": before_selection.get("source", "missing"),
                "route": before_selection.get("route", "missing"),
                "calls": before_calls,
                "truncated": before_truncated,
                "elapsed_ms": before_elapsed,
            },
            "after": {
                "source": after_selection.get("source", "missing"),
                "route": after_selection.get("route", "missing"),
                "calls": after_calls,
                "truncated": after_truncated,
                "elapsed_ms": after_elapsed,
            },
        })
    gain_count = sum(item["change"] == "gain" for item in changed)
    regression_count = sum(item["change"] == "regression" for item in changed)
    before_summary = _finalize_metrics(before_metrics)
    after_summary = _finalize_metrics(after_metrics)
    call_delta = after_summary["calls"] - before_summary["calls"]
    elapsed_delta = after_summary["elapsed_seconds"] - before_summary["elapsed_seconds"]
    net_gain = gain_count - regression_count
    return {
        "matched_items": before_summary["items"],
        "skipped_missing": skipped_missing,
        "before": before_summary,
        "after": after_summary,
        "marginal_value": {
            "changed_final_answer": answer_changed_count,
            "wrong_to_right": gain_count,
            "right_to_wrong": regression_count,
            "net_correct_gain": net_gain,
            "call_delta": call_delta,
            "elapsed_seconds_delta": round(elapsed_delta, 3),
            "net_correct_gain_per_added_call": (
                net_gain / call_delta if call_delta > 0 else None
            ),
            "net_correct_change_per_100_calls_removed": (
                net_gain / (-call_delta) * 100 if call_delta < 0 else None
            ),
        },
        "strata": {
            name: _summarize_stratum(counts)
            for name, counts in sorted(strata.items())
        },
        "gain_count": gain_count,
        "regression_count": regression_count,
        "changed": changed,
    }


def paired_candidate_audit(judge_dir: Path, output_dir: Path) -> dict:
    """Measure Audit on C versus Final(C') within the same model trajectory."""

    matrix = Counter({
        "wrong_to_wrong": 0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "right_to_right": 0,
    })
    strata: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []
    missing_diagnostics: list[object] = []
    audit_calls = 0
    audit_elapsed_ms = 0
    audit_truncated = 0
    changed = 0
    for item in load_judge_items(judge_dir):
        path = output_dir / f"{item['idx']}.json"
        if not path.exists():
            continue
        record = _record(path)
        admission = _trace_content(record, "independent_admission")
        if not (
            admission.get("admitted")
            and admission.get("mode") == "candidate_audit"
        ):
            continue
        base = admission.get("base_candidate")
        if not isinstance(base, str) or not base.strip():
            missing_diagnostics.append(item["idx"])
            continue
        final = str(record.get("final_response", ""))
        before_hit = equivalent_answers(base, item["answer"])
        after_hit = equivalent_answers(final, item["answer"])
        key = (
            "right" if before_hit else "wrong"
        ) + "_to_" + ("right" if after_hit else "wrong")
        matrix[key] += 1
        changed_here = _normalized_text(base) != _normalized_text(final)
        changed += int(changed_here)
        spec = build_problem_spec(item["problem"])
        for label in (
            "all",
            f"difficulty:{spec.profile.difficulty}",
            f"task:{spec.profile.task_kind}",
        ):
            strata[label][key] += 1
        stage_calls = [
            stage for stage in _call_stages(record)
            if stage.get("stage") == "candidate_audit"
        ]
        audit_calls += len(stage_calls)
        audit_truncated += sum(
            bool(stage.get("provider_truncated")) for stage in stage_calls
        )
        audit_elapsed_ms += sum(
            int(stage.get("elapsed_ms", 0))
            for stage in stage_calls
            if isinstance(stage.get("elapsed_ms", 0), (int, float))
        )
        records.append({
            "idx": item["idx"],
            "changed_final_answer": changed_here,
            "transition": key,
            "difficulty": spec.profile.difficulty,
            "task_kind": spec.profile.task_kind,
        })

    def summarize(counts: Counter) -> dict:
        total = sum(counts.values())
        before_correct = counts["right_to_wrong"] + counts["right_to_right"]
        after_correct = counts["wrong_to_right"] + counts["right_to_right"]
        return {
            "questions": total,
            **{name: counts[name] for name in (
                "wrong_to_wrong", "wrong_to_right",
                "right_to_wrong", "right_to_right",
            )},
            "before_correct": before_correct,
            "after_correct": after_correct,
            "net_correct_gain": counts["wrong_to_right"] - counts["right_to_wrong"],
        }

    return {
        "mode": "same_trajectory_candidate_audit",
        "audit_called": sum(matrix.values()),
        "audit_changed_final_answer": changed,
        "audit_calls": audit_calls,
        "audit_truncated_calls": audit_truncated,
        "audit_elapsed_seconds": round(audit_elapsed_ms / 1000, 3),
        "matrix": summarize(matrix),
        "strata": {
            name: summarize(counts) for name, counts in sorted(strata.items())
        },
        "missing_base_candidate_diagnostics": missing_diagnostics,
        "items": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-output-dir", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument(
        "--paired-audit",
        type=Path,
        help="One ON replay produced with --trace-candidates.",
    )
    args = parser.parse_args()
    if args.paired_audit is not None:
        report = paired_candidate_audit(args.judge_output_dir, args.paired_audit)
    elif args.before is not None and args.after is not None:
        report = compare(args.judge_output_dir, args.before, args.after)
    else:
        parser.error("provide --paired-audit or both --before and --after")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
