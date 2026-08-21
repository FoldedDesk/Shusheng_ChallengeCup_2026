"""Paired production-pipeline A/B for derived local certificates.

Branch A is the current production configuration. Branch B changes only the
complex-subproblem tool gate. Eligible items are selected from statement-level
contracts; reference answers are loaded only after both branches finish.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.execution_limits import MAX_CONCURRENCY
from core.submission_agent import SubmissionAgent
from llm_client import InternChatClient
from reasoning.local_tool_opportunity import detect_local_tool_opportunity
from reasoning.math_equivalence import equivalent_answers


EXPERIMENT_CONTRACT = {
    "single_variable": "enable_complex_subproblem_tools",
    "same_model_parameters": True,
    "same_stage_budget": True,
    "same_recovery_state_machine": True,
    "same_extraction": True,
    "tool_provider_rounds_consume_existing_max_calls": True,
    "tool_result_scope": "submitted_operation_only",
    "problem_goal_status": "NOT_CERTIFIED",
    "promotion_gate": {
        "branch_b_correct_gt_branch_a": True,
        "net_gain_gt_zero": True,
        "right_to_wrong_eq_zero": True,
        "certificate_generated_gt_zero": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B derived local certificate assistance.")
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--label-file", type=Path)
    parser.add_argument("--label", default="E3")
    parser.add_argument("--indices", default="")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        record.setdefault("idx", line_number)
        records.append(record)
    return records


def selected_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items = load_jsonl(args.input_file)
    selected = {
        value.strip() for value in str(args.indices or "").split(",") if value.strip()
    }
    if selected:
        items = [item for item in items if str(item.get("idx")) in selected]
    if args.label_file is not None:
        labels = json.loads(args.label_file.read_text(encoding="utf-8"))
        items = [
            item for item in items
            if str(labels.get(str(item.get("idx")), {}).get("label", ""))
            == args.label
        ]
    eligible = []
    for item in items:
        problem = str(item.get("problem", ""))
        spec = build_problem_spec(problem)
        opportunity = detect_local_tool_opportunity(
            problem, spec, allow_derived=True
        )
        if opportunity.eligible and opportunity.scope == "derived_subproblem":
            eligible.append(item)
    return eligible


def load_reference(reference_dir: Path | None, idx: str) -> str:
    if reference_dir is None:
        return ""
    path = reference_dir / f"{idx}.json"
    if not path.is_file():
        return ""
    record = json.loads(path.read_text(encoding="utf-8"))
    reward = record.get("reward_model", {})
    return str(reward.get("ground_truth", "")) if isinstance(reward, dict) else ""


def _steps(result: dict) -> list[dict]:
    trace = result.get("trace", [])
    return [item for item in trace if isinstance(item, dict)] if isinstance(trace, list) else []


def _content(steps: list[dict], name: str) -> dict:
    for step in steps:
        value = step.get("content")
        if step.get("step") == name and isinstance(value, dict):
            return value
    return {}


def _run_branch(problem: str, client: InternChatClient, *, enabled: bool) -> dict:
    agent = SubmissionAgent(client)
    agent.enable_complex_subproblem_tools = enabled
    started = monotonic()
    result = agent.solve(problem, {})
    elapsed_ms = round((monotonic() - started) * 1000)
    steps = _steps(result)
    calls = [
        step.get("content", {}) for step in steps
        if step.get("step") == "model_call"
        and isinstance(step.get("content"), dict)
    ]
    tool_steps = [
        step.get("content", {}) for step in steps
        if step.get("step") == "local_math_tool"
        and isinstance(step.get("content"), dict)
    ]
    return {
        "final_response": str(result.get("final_response", "")),
        "calls": len(calls),
        "elapsed_ms": elapsed_ms,
        "model_elapsed_ms": sum(int(item.get("elapsed_ms", 0) or 0) for item in calls),
        "provider_truncated_calls": sum(bool(item.get("provider_truncated")) for item in calls),
        "invalid": not bool(str(result.get("final_response", "")).strip()),
        "budget": _content(steps, "budget"),
        "opportunity": _content(steps, "local_tool_opportunity"),
        "tool_summary": _content(steps, "model_tool_summary"),
        "tool_rounds": len(tool_steps),
        "tool_operations": [
            operation for item in tool_steps
            for operation in item.get("operations", [])
        ],
        "selection": _content(steps, "selection"),
    }


def run_pair(problem: str, idx: str, client: InternChatClient) -> dict[str, Any]:
    spec = build_problem_spec(problem)
    opportunity = detect_local_tool_opportunity(problem, spec, allow_derived=True)
    b_first = int(idx) % 2 == 1 if idx.isdigit() else False
    if b_first:
        branch_b = _run_branch(problem, client, enabled=True)
        branch_a = _run_branch(problem, client, enabled=False)
    else:
        branch_a = _run_branch(problem, client, enabled=False)
        branch_b = _run_branch(problem, client, enabled=True)
    return {
        "status": "success",
        "execution_order": "B_then_A" if b_first else "A_then_B",
        "frozen_opportunity": opportunity.trace_content(),
        "budget_equal": branch_a["budget"] == branch_b["budget"],
        "branch_a": branch_a,
        "branch_b": branch_b,
    }


def annotate(record: dict[str, Any], reference: str) -> None:
    if not reference:
        return
    record["offline_reference_available"] = True
    for key in ("branch_a", "branch_b"):
        response = str(record.get(key, {}).get("final_response", ""))
        record[key]["offline_correct"] = bool(
            response and equivalent_answers(response, reference)
        )


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def process_item(
    item: dict[str, Any],
    *,
    client: InternChatClient,
    output_dir: Path,
    reference_dir: Path | None,
    semaphore: asyncio.Semaphore,
) -> None:
    idx = str(item.get("idx"))
    path = output_dir / f"{idx}.json"
    if path.is_file() and path.stat().st_size:
        print(f"Skip idx={idx}")
        return
    async with semaphore:
        try:
            record = await asyncio.to_thread(
                run_pair, str(item.get("problem", "")), idx, client
            )
            record["idx"] = idx
            annotate(record, load_reference(reference_dir, idx))
        except Exception as exc:  # noqa: BLE001 - one local diagnostic per pair.
            record = {
                "idx": idx,
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        await asyncio.to_thread(write_record, path, record)
        print(f"Finished idx={idx}")


async def run(args: argparse.Namespace) -> None:
    items = selected_items(args)
    concurrency = max(1, min(int(args.concurrency), MAX_CONCURRENCY))
    client = InternChatClient(retry=1)
    if args.preflight:
        reply = client.chat(
            messages=[{"role": "user", "content": "只回复OK"}],
            temperature=0.0,
            max_tokens=8,
            thinking_mode=False,
        )
        if not str(reply or "").strip():
            raise RuntimeError("Client preflight returned an empty response")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_contract.json").write_text(
        json.dumps({
            **EXPERIMENT_CONTRACT,
            "frozen_indices": [str(item.get("idx")) for item in items],
            "item_count": len(items),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Loaded {len(items)} paired items. Max concurrency: {concurrency}.")
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(
        process_item(
            item,
            client=client,
            output_dir=args.output_dir,
            reference_dir=args.reference_dir,
            semaphore=semaphore,
        )
        for item in items
    ))
    print(f"Saved paired outputs to {args.output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
