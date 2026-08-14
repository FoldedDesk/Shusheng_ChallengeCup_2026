"""Prepare and evaluate an offline replay without exposing answer keys at runtime."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate
from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
from tools.latex_parser import normalize_latex
from tools.sympy_tool import SympyTool


FAILURE_SENTINELS = {
    "TRUNCATED_ALL",
    "未能生成可验证的数学答案。",
    "本题未能在限定时间内生成可验证答案。",
}


def parse_id_filter(value: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-separated numeric IDs and inclusive ranges without expansion."""
    text = str(value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("--ids must contain at least one numeric ID")

    ranges: list[tuple[int, int]] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", part)
        if not match:
            raise argparse.ArgumentTypeError(
                f"invalid --ids segment {raw_part!r}; use values such as 7,12-18"
            )
        try:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) is not None else start
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"numeric --ids segment is too large: {raw_part!r}"
            ) from error
        if end < start:
            raise argparse.ArgumentTypeError(
                f"invalid descending --ids range {raw_part!r}"
            )
        ranges.append((start, end))
    return tuple(ranges)


def filter_items_by_ids(
    items: list[dict],
    id_ranges: tuple[tuple[int, int], ...] | None,
) -> list[dict]:
    """Return matching numeric items in their original replay order."""
    if id_ranges is None:
        return items
    selected: list[dict] = []
    for item in items:
        item_id = str(item.get("idx", "")).strip()
        if not re.fullmatch(r"[0-9]+", item_id):
            continue
        try:
            numeric_id = int(item_id)
        except ValueError:
            continue
        if any(start <= numeric_id <= end for start, end in id_ranges):
            selected.append(item)
    return selected


def load_judge_items(judge_output_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(judge_output_dir.glob("*.json"), key=_path_key):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        problem = record.get("problem")
        reward = record.get("reward_model", {})
        if not isinstance(problem, str) or not isinstance(reward, dict):
            continue
        items.append({
            "idx": str(record.get("idx", path.stem)),
            "problem": problem,
            "answer": str(reward.get("ground_truth", "")),
            "source": "judge_replay",
        })
    return items


def write_problem_only(items: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for item in items:
            runtime_item = {
                "idx": item["idx"],
                "problem": item["problem"],
                "source": item["source"],
            }
            stream.write(json.dumps(runtime_item, ensure_ascii=False) + "\n")


def evaluate(items: list[dict], agent_output_dir: Path) -> dict:
    processed = structural_invalid = sentinels = meta = boxed = hits = normalized_hits = 0
    model_calls = provider_truncated = structural_truncated = 0
    missing_goals = contract_box_missing = 0
    invalid_ids: list[str] = []
    missing_ids: list[str] = []
    missing_goal_ids: list[str] = []
    contract_box_missing_ids: list[str] = []
    sources: dict[str, int] = {}
    validation_tiers: dict[str, int] = {}
    certified_route_ids: list[str] = []
    certified_route_miss_ids: list[str] = []
    model_route_ids: list[str] = []
    model_route_miss_ids: list[str] = []
    certified_route_hits = model_route_hits = 0
    sympy = SympyTool()
    for item in items:
        path = agent_output_dir / f"{item['idx']}.json"
        if not path.is_file():
            missing_ids.append(item["idx"])
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        processed += 1
        final = str(record.get("final_response", "")).strip()
        spec = build_problem_spec(item["problem"])
        assessment = assess_candidate(final, "offline", spec, ())
        reasons = Finalizer.validate_structure(final)
        if reasons:
            structural_invalid += 1
            invalid_ids.append(item["idx"])
        sentinels += int(final in FAILURE_SENTINELS)
        meta += int(Finalizer.contains_meta(final))
        complete_box = _is_complete_boxed(final)
        boxed += int(complete_box)
        if getattr(spec.answer_contract, "wrapper", "none") == "boxed" and not complete_box:
            contract_box_missing += 1
            contract_box_missing_ids.append(item["idx"])
        missing_for_item = sum(not covered for covered in assessment.goal_coverage)
        missing_goals += missing_for_item
        if missing_for_item:
            missing_goal_ids.append(item["idx"])
        validation_tiers[assessment.validation_tier] = validation_tiers.get(assessment.validation_tier, 0) + 1
        semantic_hit = equivalent_answers(final, item["answer"])
        hits += int(semantic_hit)
        normalized_hits += int(_normalized(final) == _normalized(item["answer"]))
        evidence = SubmissionAgent._tool_evidence(sympy.results_for(item["problem"]), spec)
        tool_answer = SubmissionAgent._whole_tool_answer(evidence)
        if tool_answer:
            certified_route_ids.append(item["idx"])
            route_hit = equivalent_answers(tool_answer, item["answer"])
            certified_route_hits += int(route_hit)
            if not route_hit:
                certified_route_miss_ids.append(item["idx"])
        else:
            model_route_ids.append(item["idx"])
            model_route_hits += int(semantic_hit)
            if not semantic_hit:
                model_route_miss_ids.append(item["idx"])
        trace = record.get("trace", [])
        selection = next((step.get("content", {}) for step in trace if step.get("step") == "selection"), {})
        source = str(selection.get("source", "missing"))
        sources[source] = sources.get(source, 0) + 1
        for step in trace:
            if not str(step.get("step", "")).startswith("model_call_"):
                continue
            model_calls += 1
            content = step.get("content", {})
            provider_truncated += int(bool(content.get("provider_truncated")))
            structural_truncated += int(content.get("truncation_signal") == "structural")
    return {
        "total": len(items),
        "processed": processed,
        "missing_output_count": len(missing_ids),
        "missing_output_ids": missing_ids,
        "structural_invalid_answers": structural_invalid,
        "structural_invalid_ids": invalid_ids,
        "failure_sentinel_count": sentinels,
        "final_response_contains_meta": meta,
        "boxed_contract_count": boxed,
        "boxed_contract_missing_count": contract_box_missing,
        "boxed_contract_missing_ids": contract_box_missing_ids,
        "missing_subquestion_count": missing_goals,
        "items_missing_subquestions": len(missing_goal_ids),
        "items_missing_subquestion_ids": missing_goal_ids,
        "offline_validation_tiers": validation_tiers,
        "model_call_count": model_calls,
        "provider_truncated_call_count": provider_truncated,
        "structural_truncated_call_count": structural_truncated,
        "selection_sources": sources,
        "normalized_standard_answer_hits": normalized_hits,
        "semantic_standard_answer_hits": hits,
        "current_certified_route_count": len(certified_route_ids),
        "current_certified_route_hits": certified_route_hits,
        "current_certified_route_accuracy": (
            certified_route_hits / len(certified_route_ids)
            if certified_route_ids else None
        ),
        "current_certified_route_ids": certified_route_ids,
        "current_certified_route_miss_ids": certified_route_miss_ids,
        "current_model_route_count": len(model_route_ids),
        "current_model_route_hits_from_saved_outputs": model_route_hits,
        "current_model_route_accuracy_from_saved_outputs": (
            model_route_hits / len(model_route_ids) if model_route_ids else None
        ),
        "current_model_route_ids": model_route_ids,
        "current_model_route_miss_ids": model_route_miss_ids,
    }


def audit_problem_specs(items: list[dict]) -> dict:
    """Summarize routing decisions without reading any answer during solving."""
    distributions: dict[str, dict[str, int]] = {
        "language": {},
        "mode": {},
        "wrapper": {},
        "shape": {},
        "difficulty": {},
        "topic": {},
        "goal_count": {},
    }
    tool_whole = verification = empty_goals = 0
    certified_whole_ids: list[str] = []
    deep_reasoning_ids: list[str] = []
    sympy = SympyTool()
    for item in items:
        spec = build_problem_spec(item["problem"])
        contract = spec.answer_contract
        values = {
            "language": getattr(contract, "language", spec.profile.language),
            "mode": getattr(contract, "mode", "unknown"),
            "wrapper": getattr(contract, "wrapper", "none"),
            "shape": spec.profile.answer_shape,
            "difficulty": spec.profile.difficulty,
            "topic": getattr(spec.profile, "topic", "general"),
            "goal_count": str(len(spec.goals)),
        }
        for name, value in values.items():
            bucket = distributions[name]
            key = str(value)
            bucket[key] = bucket.get(key, 0) + 1
        tool_whole += int(spec.tool_can_answer_whole)
        verification += int(spec.verification_required)
        empty_goals += int(not spec.goals or any(not goal.instruction.strip() for goal in spec.goals))
        evidence = SubmissionAgent._tool_evidence(
            sympy.results_for(item["problem"]),
            spec,
        )
        if any(entry.scope == "whole_goal" and entry.verified for entry in evidence):
            certified_whole_ids.append(item["idx"])
        if SubmissionAgent._use_deep_reasoning(spec, item["problem"]):
            deep_reasoning_ids.append(item["idx"])
    reference_tiers: dict[str, int] = {}
    reference_degraded_ids: list[str] = []
    reference_issues: list[dict] = []
    for item in items:
        reference = str(item.get("answer", ""))
        if not reference:
            continue
        assessment = assess_candidate(
            reference,
            "offline_reference",
            build_problem_spec(item["problem"]),
            (),
        )
        reference_tiers[assessment.validation_tier] = reference_tiers.get(assessment.validation_tier, 0) + 1
        if assessment.validation_tier != "complete":
            issue_spec = build_problem_spec(item["problem"])
            reference_degraded_ids.append(item["idx"])
            reference_issues.append({
                "idx": item["idx"],
                "shape": issue_spec.profile.answer_shape,
                "goal_coverage": list(assessment.goal_coverage),
                "reasons": list(assessment.rejected_reasons),
                "goals": [{
                    "instruction": goal.instruction,
                    "kind": goal.kind,
                    "requirements": [requirement.name for requirement in goal.requirements],
                } for goal in issue_spec.goals],
            })
    deep_reasoning_set = set(deep_reasoning_ids)
    return {
        "total": len(items),
        "distributions": distributions,
        "tool_can_answer_whole_count": tool_whole,
        "effective_whole_tool_route_count": len(certified_whole_ids),
        "effective_whole_tool_route_ids": certified_whole_ids,
        "deep_reasoning_route_count": len(deep_reasoning_ids),
        "quick_response_route_count": len(items) - len(deep_reasoning_ids),
        "quick_response_route_ids": [
            item["idx"] for item in items if item["idx"] not in deep_reasoning_set
        ],
        "verification_required_count": verification,
        "empty_or_blank_goal_count": empty_goals,
        "offline_reference_validation_tiers": reference_tiers,
        "offline_reference_noncomplete_ids": reference_degraded_ids,
        "offline_reference_issues": reference_issues,
    }


def _is_complete_boxed(value: str) -> bool:
    text = str(value or "").strip()
    result = Finalizer.extract_result(text)
    return bool(result.valid and result.explicit_answer and r"\boxed{" in text)


def _normalized(value: str) -> str:
    result = Finalizer.extract_result(str(value or ""))
    text = result.answer if result.answer else str(value or "")
    text = normalize_latex(text).lower().replace("−", "-")
    return re.sub(r"[\s{}\\,，。；;：:`'$]", "", text)


def _path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-output-dir", type=Path, required=True)
    parser.add_argument("--agent-output-dir", type=Path)
    parser.add_argument("--write-problem-only", type=Path)
    parser.add_argument(
        "--ids",
        type=parse_id_filter,
        help="Use comma-separated numeric IDs and inclusive ranges, for example 7,12-18.",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N numeric item IDs (for a smoke replay).")
    args = parser.parse_args()

    items = load_judge_items(args.judge_output_dir)
    items = filter_items_by_ids(items, args.ids)
    if args.limit is not None:
        items = items[:max(0, args.limit)]
    if args.write_problem_only:
        write_problem_only(items, args.write_problem_only)
    if args.agent_output_dir:
        result = evaluate(items, args.agent_output_dir)
        result["problem_spec_audit"] = audit_problem_specs(items)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = audit_problem_specs(items)
        result["problem_only_written"] = bool(args.write_problem_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
