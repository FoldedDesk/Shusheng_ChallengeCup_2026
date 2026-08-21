"""Summarize shared-primary explicit local-contract A/B outputs."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    records = []
    for path in sorted(args.output_dir.glob("*.json")):
        if path.name in {"experiment_contract.json", "summary.json"}:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            records.append(record)
    successful = [item for item in records if item.get("status") == "success"]
    summary = {
        "items": len(successful),
        "errors": len(records) - len(successful),
        "budget_mismatches": sum(not item.get("budget_equal", False) for item in successful),
        "shared_primary": {
            "calls": sum(int(item.get("shared_primary", {}).get("calls", 0) or 0) for item in successful),
            "elapsed_ms": sum(int(item.get("shared_primary", {}).get("elapsed_ms", 0) or 0) for item in successful),
        },
    }
    for branch in ("branch_a", "branch_b"):
        summary[branch] = {
            "correct": sum(bool(item.get(branch, {}).get("offline_correct")) for item in successful),
            "calls": sum(int(item.get(branch, {}).get("calls", 0) or 0) for item in successful),
            "elapsed_ms": sum(int(item.get(branch, {}).get("elapsed_ms", 0) or 0) for item in successful),
            "provider_truncated_calls": sum(int(item.get(branch, {}).get("provider_truncated_calls", 0) or 0) for item in successful),
            "invalid": sum(not bool(item.get(branch, {}).get("final_response", "")) for item in successful),
        }
    wrong_to_right_items = [
        item for item in successful
        if not item.get("branch_a", {}).get("offline_correct", False)
        and item.get("branch_b", {}).get("offline_correct", False)
    ]
    right_to_wrong = sum(
        item.get("branch_a", {}).get("offline_correct", False)
        and not item.get("branch_b", {}).get("offline_correct", False)
        for item in successful
    )
    contracts = [item.get("branch_b", {}).get("contract", {}) for item in successful]
    summary["paired"] = {
        "wrong_to_right": len(wrong_to_right_items),
        "right_to_wrong": right_to_wrong,
        "net_gain": len(wrong_to_right_items) - right_to_wrong,
        "both_correct": sum(
            item.get("branch_a", {}).get("offline_correct", False)
            and item.get("branch_b", {}).get("offline_correct", False)
            for item in successful
        ),
        "both_wrong": sum(
            not item.get("branch_a", {}).get("offline_correct", False)
            and not item.get("branch_b", {}).get("offline_correct", False)
            for item in successful
        ),
    }
    summary["tool_funnel"] = {
        "eligible": len(successful),
        "contract_emitted": sum(bool(item.get("emitted")) for item in contracts),
        "contract_valid": sum(bool(item.get("contract_valid")) for item in contracts),
        "certificate_generated": sum(bool(item.get("certificate_generated")) for item in contracts),
        "certificate_used_verbatim": sum(bool(item.get("certificate_used_verbatim")) for item in contracts),
        "followup_usable": sum(bool(item.get("followup_usable")) for item in contracts),
        "local_fact_helpful": sum(
            bool(item.get("branch_b", {}).get("contract", {}).get("contract_valid"))
            for item in wrong_to_right_items
        ),
    }
    summary["contract_reason_counts"] = dict(sorted(Counter(
        str(item.get("reason") or "ok") for item in contracts
    ).items()))
    summary["operation_counts"] = dict(sorted(Counter(
        str(item.get("operation") or "unknown") for item in contracts
    ).items()))
    summary["promotion_gate_passed"] = bool(
        summary["paired"]["net_gain"] > 0
        and summary["paired"]["right_to_wrong"] == 0
        and summary["tool_funnel"]["local_fact_helpful"]
        == summary["paired"]["wrong_to_right"]
        and summary["budget_mismatches"] == 0
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.write:
        (args.output_dir / "summary.json").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
