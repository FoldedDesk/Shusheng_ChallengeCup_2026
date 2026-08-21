"""Summarize paired derived-local-certificate A/B outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name in {"experiment_contract.json", "summary.json"}:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local certificate A/B.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    records = load_records(args.output_dir)
    successful = [record for record in records if record.get("status") == "success"]
    summary: dict[str, Any] = {
        "items": len(successful),
        "errors": len(records) - len(successful),
        "budget_mismatches": sum(not record.get("budget_equal", False) for record in successful),
    }
    for branch in ("branch_a", "branch_b"):
        summary[branch] = {
            "correct": sum(bool(record.get(branch, {}).get("offline_correct")) for record in successful),
            "calls": sum(int(record.get(branch, {}).get("calls", 0) or 0) for record in successful),
            "elapsed_ms": sum(int(record.get(branch, {}).get("elapsed_ms", 0) or 0) for record in successful),
            "provider_truncated_calls": sum(int(record.get(branch, {}).get("provider_truncated_calls", 0) or 0) for record in successful),
            "invalid": sum(bool(record.get(branch, {}).get("invalid")) for record in successful),
        }
    wrong_to_right = sum(
        not record.get("branch_a", {}).get("offline_correct", False)
        and record.get("branch_b", {}).get("offline_correct", False)
        for record in successful
    )
    right_to_wrong = sum(
        record.get("branch_a", {}).get("offline_correct", False)
        and not record.get("branch_b", {}).get("offline_correct", False)
        for record in successful
    )
    tool_summaries = [record.get("branch_b", {}).get("tool_summary", {}) for record in successful]
    summary["paired"] = {
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_gain": wrong_to_right - right_to_wrong,
        "both_correct": sum(
            record.get("branch_a", {}).get("offline_correct", False)
            and record.get("branch_b", {}).get("offline_correct", False)
            for record in successful
        ),
        "both_wrong": sum(
            not record.get("branch_a", {}).get("offline_correct", False)
            and not record.get("branch_b", {}).get("offline_correct", False)
            for record in successful
        ),
    }
    summary["tool_funnel"] = {
        "eligible": len(successful),
        "attempted": sum(int(item.get("attempted", 0) or 0) for item in tool_summaries),
        "contract_valid": sum(int(item.get("contract_valid", 0) or 0) for item in tool_summaries),
        "certificate_generated": sum(int(item.get("certificate_generated", 0) or 0) for item in tool_summaries),
        "followup_completed": sum(bool(item.get("followup_completed")) for item in tool_summaries),
        "local_fact_helpful": wrong_to_right,
    }
    summary["operation_counts"] = dict(sorted(Counter(
        operation for record in successful
        for operation in record.get("branch_b", {}).get("tool_operations", [])
    ).items()))
    summary["promotion_gate_passed"] = bool(
        summary["branch_b"]["correct"] > summary["branch_a"]["correct"]
        and summary["paired"]["net_gain"] > 0
        and summary["paired"]["right_to_wrong"] == 0
        and summary["tool_funnel"]["certificate_generated"] > 0
        and summary["budget_mismatches"] == 0
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.write:
        (args.output_dir / "summary.json").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
