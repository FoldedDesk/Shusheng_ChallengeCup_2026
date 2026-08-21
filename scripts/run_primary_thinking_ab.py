"""Paired production-pipeline A/B for one primary-generation variable.

Both branches use the same production agent, prompt, token budget, tools,
recovery state machine, and extraction.  Branch B changes only the primary
call's hidden-thinking decision.  Local reference answers are consulted only
after both branches have returned and are never passed to either agent.
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

from core.execution_limits import MAX_CONCURRENCY
from core.submission_agent import SubmissionAgent
from llm_client import InternChatClient
from rag.card_retriever import RetrievalBundle
from reasoning.math_equivalence import equivalent_answers


class NoPrimaryThinkingAgent(SubmissionAgent):
    """Production agent with only the primary hidden-thinking gate disabled."""

    @staticmethod
    def _hidden_thinking(spec) -> bool:
        del spec
        return False


class EmptyRetriever:
    """Preserve public classification metadata while supplying no RAG cards."""

    @staticmethod
    def retrieve(spec, *, subject_override: str = "") -> RetrievalBundle:
        primary = subject_override or getattr(
            spec.profile, "primary_subject", spec.profile.subject
        )
        confidence = getattr(spec.profile, "subject_confidence", "low")
        if subject_override and confidence == "low":
            confidence = "medium"
        return RetrievalBundle(
            (),
            (),
            language=getattr(spec.answer_contract, "language", spec.profile.language),
            primary_subject=primary,
            secondary_subject=getattr(spec.profile, "secondary_subject", ""),
            subject_confidence=confidence,
        )


class NoRagAgent(SubmissionAgent):
    """Production agent with only local card retrieval disabled."""

    def __init__(self, client) -> None:
        super().__init__(client)
        self.retriever = EmptyRetriever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary generation single-variable A/B.")
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--indices", default="")
    parser.add_argument(
        "--experiment",
        choices=("thinking", "rag"),
        default="thinking",
    )
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
    return [item for item in items if str(item.get("idx")) in selected] if selected else items


def _trace_steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = result.get("trace", [])
    return [item for item in trace if isinstance(item, dict)] if isinstance(trace, list) else []


def _branch(problem: str, client: InternChatClient, *, variant: str) -> dict[str, Any]:
    agent_type = {
        "production": SubmissionAgent,
        "no_thinking": NoPrimaryThinkingAgent,
        "no_rag": NoRagAgent,
    }[variant]
    started = monotonic()
    result = agent_type(client).solve(problem, {})
    elapsed_ms = round((monotonic() - started) * 1000)
    steps = _trace_steps(result)
    calls = [
        step.get("content", {})
        for step in steps
        if step.get("step") == "model_call"
        and isinstance(step.get("content"), dict)
    ]
    primary_calls = [call for call in calls if call.get("stage") == "primary"]
    primary_truncation = any(
        step.get("step") == "truncation_state"
        and isinstance(step.get("content"), dict)
        and step["content"].get("stage") == "primary"
        for step in steps
    )
    selection = next((
        step.get("content", {})
        for step in reversed(steps)
        if step.get("step") == "selection" and isinstance(step.get("content"), dict)
    ), {})
    response = str(result.get("final_response", ""))
    return {
        "final_response": response,
        "invalid": not bool(response.strip()),
        "calls": sum(int(call.get("request_count", 1) or 1) for call in calls),
        "elapsed_ms": elapsed_ms,
        "model_elapsed_ms": sum(int(call.get("elapsed_ms", 0) or 0) for call in calls),
        "provider_truncated_calls": sum(bool(call.get("provider_truncated")) for call in calls),
        "primary_calls": len(primary_calls),
        "primary_truncated": primary_truncation,
        "selection_route": str(selection.get("route", "")),
    }


def run_pair(
    problem: str,
    idx: str,
    client: InternChatClient,
    *,
    experiment: str,
) -> dict[str, Any]:
    b_first = sum(idx.encode("utf-8")) % 2 == 1
    branch_b_variant = "no_thinking" if experiment == "thinking" else "no_rag"
    if b_first:
        branch_b = _branch(problem, client, variant=branch_b_variant)
        branch_a = _branch(problem, client, variant="production")
    else:
        branch_a = _branch(problem, client, variant="production")
        branch_b = _branch(problem, client, variant=branch_b_variant)
    return {
        "status": "success",
        "experiment": experiment,
        "execution_order": "B_then_A" if b_first else "A_then_B",
        "branch_a": branch_a,
        "branch_b": branch_b,
    }


def annotate(record: dict[str, Any], reference: str) -> None:
    if not reference.strip():
        return
    record["offline_reference_available"] = True
    for key in ("branch_a", "branch_b"):
        response = str(record[key].get("final_response", ""))
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
    experiment: str,
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
                run_pair,
                str(item.get("problem", "")),
                idx,
                client,
                experiment=experiment,
            )
            record["idx"] = idx
            annotate(record, str(item.get("answer", "")))
        except Exception as exc:  # noqa: BLE001 - one diagnostic per pair.
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
            "experiment": args.experiment,
            "only_variable": (
                "primary_thinking_mode" if args.experiment == "thinking" else "rag_cards"
            ),
            "branch_a": "production_default",
            "branch_b": "primary_thinking_false" if args.experiment == "thinking" else "empty_rag_bundle",
            "same_prompt_temperature_tokens_recovery_extraction": True,
            "reference_loaded_after_both_branches": True,
            "branch_order": "idx_checksum_crossover",
            "frozen_indices": [str(item.get("idx")) for item in items],
            "promotion_gate": {
                "branch_b_net_gain_gt_zero": True,
                "branch_b_right_to_wrong_eq_zero": True,
                "replication_required": True,
            },
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
            experiment=args.experiment,
            semaphore=semaphore,
        )
        for item in items
    ))
    print(f"Saved paired outputs to {args.output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
