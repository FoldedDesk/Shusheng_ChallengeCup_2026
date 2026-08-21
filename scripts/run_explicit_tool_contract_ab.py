"""Shared-primary, equal-budget A/B for explicit local tool contracts.

Branch A spends two calls continuing the primary draft. Branch B spends the
same two calls and output-token budget on (1) one strict JSON operation
contract and (2) one continuation supplied with the locally recomputed fact.
References are read only after all model calls for both branches finish.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.execution_limits import MAX_CONCURRENCY
from core.model_response import ModelCallResult
from core.stage_budget import plan_stage_budget
from core.submission_agent import SubmissionAgent
from llm_client import InternChatClient
from reasoning.candidate_selector import CandidateAssessment, choose_candidate
from reasoning.local_tool_opportunity import detect_local_tool_opportunity
from reasoning.math_equivalence import equivalent_answers
from reasoning.obligation_graph import MathematicalObligationGraph
from tools.tool_contract import CertificateStatus


A_CONTINUE_TOKENS = 4096
A_FINISH_TOKENS = 2048
B_CONTRACT_TOKENS = 1536
B_SOLVE_TOKENS = 4608
BRANCH_MAX_CALLS = 2
BRANCH_TOTAL_TOKENS = 6144


@dataclass(frozen=True)
class Attempt:
    raw: str
    candidate: CandidateAssessment | None
    usable: bool
    truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit local contract A/B.")
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
        opportunity = detect_local_tool_opportunity(problem, spec, allow_derived=True)
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


def _assess(
    agent: SubmissionAgent,
    raw: str,
    result: ModelCallResult,
    *,
    source: str,
    spec,
    evidence,
    independence_group: str,
) -> Attempt:
    truncated = agent._truncated(result, raw)
    candidates = agent._transport_admissible(
        agent._assess_response(
            raw,
            source=source,
            spec=spec,
            evidence=evidence,
            method_id=source,
            independence_group=independence_group,
        ),
        truncated,
    )
    candidate = choose_candidate(candidates)
    usable = agent._complete_after_transport(candidate, truncated, spec, raw)
    return Attempt(raw, candidate, usable, truncated)


def _pick(attempts: tuple[Attempt, ...]) -> CandidateAssessment | None:
    for attempt in reversed(attempts):
        if attempt.usable and attempt.candidate is not None:
            return attempt.candidate
    return None


def _stage_stats(trace: list[dict], stages: set[str]) -> dict[str, int]:
    calls = [
        step.get("content", {}) for step in trace
        if step.get("step") == "model_call"
        and isinstance(step.get("content"), dict)
        and str(step["content"].get("stage", "")) in stages
    ]
    return {
        "calls": len(calls),
        "elapsed_ms": sum(int(item.get("elapsed_ms", 0) or 0) for item in calls),
        "provider_truncated_calls": sum(bool(item.get("provider_truncated")) for item in calls),
    }


def _branch_record(
    agent: SubmissionAgent,
    spec,
    attempts: tuple[Attempt, ...],
    fallback: Attempt,
    trace: list[dict],
    stages: set[str],
) -> dict[str, Any]:
    candidate = _pick(attempts)
    if candidate is None and fallback.usable:
        candidate = fallback.candidate
    answer = agent._render_submission(candidate.answer, spec) if candidate else ""
    return {
        **_stage_stats(trace, stages),
        "max_calls": BRANCH_MAX_CALLS,
        "max_output_tokens": BRANCH_TOTAL_TOKENS,
        "candidate_usable": candidate is not None,
        "final_response": answer,
        "attempt_truncated": [attempt.truncated for attempt in attempts],
    }


def _contract_request(problem: str, spec, opportunity, schemas: list[dict]) -> str:
    schema_text = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "complete requested result"
    return (
        "You are a LOCAL_OPERATION_TRANSLATOR, not a solver or reviewer. Find one "
        "finite deterministic subproblem whose exact value could advance the hardest "
        "obligation. Translate only facts rigorously derivable from the statement. Do "
        "not guess the final answer and do not emit code. The operation result will "
        "certify only the submitted arguments, never their correspondence to the full "
        "problem. If no exact supported operation exists, abstain.\n\n"
        f"Opportunity class: {opportunity.kind.value}\n"
        f"Allowed operation schema: {schema_text}\n"
        f"Target obligations: {obligations}\n\n"
        "Output exactly one JSON object and nothing else. Either:\n"
        '{"status":"CALL","operation":"allowed_name","arguments":{...}}\n'
        "or:\n"
        '{"status":"ABSTAIN","reason":"short reason"}\n\n'
        f"Problem:\n{problem}"
    )


def _certificate_followup(spec, execution) -> str:
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "the complete requested result"
    if execution.ok:
        certificate = (
            f"Operation: {execution.name}\n"
            f"Local certificate status: {execution.local_certificate_status.value}\n"
            f"Certified operation result: {execution.result}\n"
            "Scope: submitted operation only; translation and whole goal NOT_CERTIFIED."
        )
    else:
        certificate = (
            f"No valid local certificate was produced ({execution.reason}). "
            "Do not invent or assume a tool value."
        )
    return (
        "Continue the original draft once. First check whether the local operation, if "
        "present, follows exactly from the statement with the correct domain, endpoints, "
        "labels, repetitions, and quantifiers. Discard it on any mismatch. Use it only as "
        "a local fact, then complete the mathematics; do not restart broad search or call "
        "another tool. The first line must start FINAL: and immediately state the full "
        "gradable conclusion. Follow it with only the shortest necessary proof.\n\n"
        f"{certificate}\nRequired obligations: {obligations}"
    )


def _run_a(agent, primary_request: str, primary: Attempt, spec, evidence, trace):
    a1_raw, a1_result = agent._call(
        primary_request,
        stage="branch_a_continuation",
        max_tokens=A_CONTINUE_TOKENS,
        temperature=0.1,
        thinking_mode=False,
        trace=trace,
        prior_response=primary.raw,
        followup=agent._continuation_instruction(spec),
    )
    a1 = _assess(
        agent, a1_raw, a1_result, source="branch_a_continuation",
        spec=spec, evidence=evidence, independence_group="branch_a",
    )
    a2_raw, a2_result = agent._call(
        primary_request,
        stage="branch_a_finish",
        max_tokens=A_FINISH_TOKENS,
        temperature=0.1,
        thinking_mode=False,
        trace=trace,
        prior_response=a1_raw,
        followup=agent._conclusion_only_instruction(spec),
    )
    a2 = _assess(
        agent, a2_raw, a2_result, source="branch_a_finish",
        spec=spec, evidence=evidence, independence_group="branch_a",
    )
    return (a1, a2)


def _run_b(agent, problem: str, primary_request: str, primary: Attempt, spec, evidence, opportunity, trace):
    schemas = agent.model_math_tools.schemas(opportunity.allowed_tools)
    contract_raw, contract_result = agent._call(
        _contract_request(problem, spec, opportunity, schemas),
        stage="branch_b_tool_contract",
        max_tokens=B_CONTRACT_TOKENS,
        temperature=0.0,
        thinking_mode=False,
        trace=trace,
        system_prompt=(
            "You translate one explicitly bounded mathematical subproblem into "
            "one allowed JSON operation contract. Never solve the whole problem, "
            "never provide an answer, never emit code or prose, and never invent "
            "missing bounds. Output exactly one JSON object matching the user schema."
        ),
    )
    contract_truncated = agent._truncated(contract_result, contract_raw)
    execution = agent.model_math_tools.execute_contract(
        "" if contract_truncated else contract_raw,
        allowed_names=opportunity.allowed_tools,
    )
    tool_evidence = agent._model_tool_evidence(ModelCallResult(
        "",
        usage={"_model_tool_outcomes": ({
            "operation": execution.name,
            "ok": execution.ok,
            "result": execution.result,
        },)},
    )) if execution.ok else ()
    b_evidence = tuple((*evidence, *tool_evidence))
    b_raw, b_result = agent._call(
        primary_request,
        stage="branch_b_certificate_continuation",
        max_tokens=B_SOLVE_TOKENS,
        temperature=0.1,
        thinking_mode=False,
        trace=trace,
        prior_response=primary.raw,
        followup=_certificate_followup(spec, execution),
    )
    b = _assess(
        agent, b_raw, b_result, source="branch_b_certificate_continuation",
        spec=spec, evidence=b_evidence, independence_group="branch_b",
    )
    used = bool(
        execution.ok
        and len(execution.result) <= 200
        and execution.result.strip()
        and execution.result.strip() in b_raw
    )
    contract_meta = {
        "emitted": bool(contract_raw.strip()),
        "truncated": contract_truncated,
        "contract_valid": execution.ok,
        "operation": execution.name,
        "reason": execution.reason,
        "certificate_generated": (
            execution.local_certificate_status is CertificateStatus.CERTIFIED_TRUE
        ),
        "certificate_status": execution.local_certificate_status.value,
        "result_fingerprint": (
            hashlib.sha256(execution.result.encode("utf-8")).hexdigest()[:16]
            if execution.ok else ""
        ),
        "result_length": len(execution.result),
        "certificate_used_verbatim": used,
        "followup_usable": b.usable,
    }
    return (b,), contract_meta


def run_pair(problem: str, idx: str, client: InternChatClient) -> dict[str, Any]:
    agent = SubmissionAgent(client)
    spec = build_problem_spec(problem)
    statement = spec.problem_text or str(problem).strip()
    opportunity = detect_local_tool_opportunity(statement, spec, allow_derived=True)
    cards = agent.retriever.retrieve(spec)
    evidence = ()
    graph = MathematicalObligationGraph.fallback(spec)
    plan = graph.route_plan(spec, 0)
    deep_reasoning = agent._deep_reasoning(spec)
    hidden_thinking = agent._hidden_thinking(spec)
    budget = plan_stage_budget(spec, False, deep_reasoning=deep_reasoning)
    trace: list[dict] = []
    primary_request = agent._primary_request(
        statement,
        spec,
        cards,
        evidence,
        plan=plan,
        deep_reasoning=deep_reasoning,
        hidden_thinking=hidden_thinking,
    )
    primary_raw, primary_result = agent._call(
        primary_request,
        stage="shared_primary",
        max_tokens=budget.solve_tokens,
        temperature=0.2,
        thinking_mode=hidden_thinking,
        trace=trace,
    )
    primary = _assess(
        agent, primary_raw, primary_result, source="shared_primary",
        spec=spec, evidence=evidence, independence_group="shared",
    )

    b_first = int(idx) % 2 == 1 if idx.isdigit() else False
    if b_first:
        b_attempts, contract_meta = _run_b(
            agent, statement, primary_request, primary, spec, evidence, opportunity, trace
        )
        a_attempts = _run_a(agent, primary_request, primary, spec, evidence, trace)
    else:
        a_attempts = _run_a(agent, primary_request, primary, spec, evidence, trace)
        b_attempts, contract_meta = _run_b(
            agent, statement, primary_request, primary, spec, evidence, opportunity, trace
        )

    branch_a = _branch_record(
        agent, spec, a_attempts, primary, trace,
        {"branch_a_continuation", "branch_a_finish"},
    )
    branch_b = _branch_record(
        agent, spec, b_attempts, primary, trace,
        {"branch_b_tool_contract", "branch_b_certificate_continuation"},
    )
    branch_b["contract"] = contract_meta
    shared = _stage_stats(trace, {"shared_primary"})
    return {
        "status": "success",
        "execution_order": "B_then_A" if b_first else "A_then_B",
        "frozen_opportunity": opportunity.trace_content(),
        "shared_primary": shared,
        "budget_equal": bool(
            branch_a["max_calls"] == branch_b["max_calls"]
            and branch_a["max_output_tokens"] == branch_b["max_output_tokens"]
        ),
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


async def process_item(item, *, client, output_dir, reference_dir, semaphore) -> None:
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
            "single_variable": "continuation_budget_reallocated_to_explicit_local_contract",
            "shared_primary": True,
            "branch_max_calls": BRANCH_MAX_CALLS,
            "branch_total_output_tokens": BRANCH_TOTAL_TOKENS,
            "branch_a_tokens": [A_CONTINUE_TOKENS, A_FINISH_TOKENS],
            "branch_b_tokens": [B_CONTRACT_TOKENS, B_SOLVE_TOKENS],
            "reference_loaded_after_all_calls": True,
            "frozen_indices": [str(item.get("idx")) for item in items],
            "item_count": len(items),
            "promotion_gate": {
                "net_gain_gt_zero": True,
                "right_to_wrong_eq_zero": True,
                "helpful_change_has_valid_contract": True,
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Loaded {len(items)} shared-primary pairs. Max concurrency: {concurrency}.")
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(
        process_item(
            item, client=client, output_dir=args.output_dir,
            reference_dir=args.reference_dir, semaphore=semaphore,
        )
        for item in items
    ))
    print(f"Saved paired outputs to {args.output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
