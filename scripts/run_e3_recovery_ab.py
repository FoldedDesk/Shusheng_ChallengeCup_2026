"""Run a shared-primary, equal-budget recovery A/B on a local frozen corpus.

Branch A continues the exhausted primary state. Branch B never sees that state:
it searches for one missing mathematical bridge, then performs a targeted solve.
The optional ``thinking`` experiment instead gives both branches the exact same
continuation context, prompts, temperatures, and output budget; only Branch B
enables provider thinking mode.
Reference answers are consulted only after every model call has returned.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.execution_limits import MAX_CONCURRENCY
from core.stage_budget import plan_stage_budget
from llm_client import InternChatClient
from reasoning.candidate_selector import CandidateAssessment, choose_candidate
from reasoning.math_equivalence import equivalent_answers
from reasoning.obligation_graph import MathematicalObligationGraph
from core.submission_agent import SubmissionAgent


BRANCH_SEARCH_TOKENS = 4096
BRANCH_FINISH_TOKENS = 2048
BRANCH_MAX_CALLS = 2
FOUND = re.compile(r"(?im)^\s*STATUS\s*[:：]\s*FOUND\s*$")


@dataclass(frozen=True)
class Attempt:
    raw: str
    candidate: CandidateAssessment | None
    usable: bool
    truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Equal-budget continuation versus breakthrough-search A/B."
    )
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Optional local JSON object keyed by idx, used only to select a bucket.",
    )
    parser.add_argument("--label", default="E3")
    parser.add_argument("--indices", default="")
    parser.add_argument(
        "--experiment",
        choices=("breakthrough", "thinking"),
        default="breakthrough",
        help="Single variable compared against the production continuation.",
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


def load_reference(reference_dir: Path | None, idx: str) -> str:
    if reference_dir is None:
        return ""
    path = reference_dir / f"{idx}.json"
    if not path.is_file():
        return ""
    record = json.loads(path.read_text(encoding="utf-8"))
    reward = record.get("reward_model", {})
    return str(reward.get("ground_truth", "")) if isinstance(reward, dict) else ""


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
            if str(labels.get(str(item.get("idx")), {}).get("label", "")) == args.label
        ]
    return items


def _stage_stats(trace: list[dict], stages: set[str]) -> dict[str, int]:
    calls = [
        step.get("content", {})
        for step in trace
        if step.get("step") == "model_call"
        and isinstance(step.get("content"), dict)
        and str(step["content"].get("stage", "")) in stages
    ]
    return {
        "calls": sum(int(item.get("request_count", 1) or 1) for item in calls),
        "elapsed_ms": sum(int(item.get("elapsed_ms", 0) or 0) for item in calls),
        "provider_truncated_calls": sum(
            bool(item.get("provider_truncated")) for item in calls
        ),
    }


def _assess(
    agent: SubmissionAgent,
    raw: str,
    result,
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
    return Attempt(raw=raw, candidate=candidate, usable=usable, truncated=truncated)


def _pick(attempts: tuple[Attempt, ...]) -> CandidateAssessment | None:
    for attempt in reversed(attempts):
        if attempt.usable and attempt.candidate is not None:
            return attempt.candidate
    return None


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
    stats = _stage_stats(trace, stages)
    return {
        **stats,
        "max_calls": BRANCH_MAX_CALLS,
        "max_output_tokens": BRANCH_SEARCH_TOKENS + BRANCH_FINISH_TOKENS,
        "candidate_usable": candidate is not None,
        "candidate": str(candidate.answer if candidate else "")[:12_000],
        "final_response": answer,
        "attempt_truncated": [attempt.truncated for attempt in attempts],
    }


def _breakthrough_request(problem: str, spec, attempted_method: str) -> str:
    target = str(getattr(spec.semantics, "target", "") or "").strip()
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "complete requested result"
    subject = str(getattr(spec.profile, "primary_subject", spec.profile.subject))
    if spec.profile.language == "zh":
        return (
            "你正在执行 KEY_BREAKTHROUGH_SEARCH，不是完整解题、答案审查或续写。"
            "不要复述题目，不要猜最终答案，不要沿用失败路线。只寻找一个能跨越最难义务的"
            "决定性桥梁：关键引理、显式构造、关键变换、可复现反例，或证明原路线不可行的障碍。"
            "每个断言必须从题面推出；找不到时必须诚实输出 NOT_FOUND。\n\n"
            f"学科：{subject}\n精确目标：{target or obligations}\n必答义务：{obligations}\n"
            f"先前尝试的方法类别：{attempted_method or '未知'}\n"
            "已确认事实：仅题面假设；先前草稿中的任何中间结论均不可信。\n"
            "未决义务：找到一个足以使完整求解可执行的数学桥梁。\n\n"
            "只输出以下字段；DERIVATION 最多六个紧凑步骤：\n"
            "STATUS: FOUND 或 NOT_FOUND\n"
            "BRIDGE_TYPE: lemma|construction|transformation|counterexample|obstruction\n"
            "BREAKTHROUGH: 精确陈述\nDERIVATION: 从题面验证该桥梁\n"
            "TARGETED_USE: 如何用它完成原目标\nFALSIFIER: 一个可复现核验\n\n"
            f"原题：\n{problem}"
        )
    return (
        "You are in KEY_BREAKTHROUGH_SEARCH, not a full solve, answer audit, or "
        "continuation. Do not restate the problem, guess the final answer, or continue "
        "the failed route. Find exactly one decisive bridge across the hardest missing "
        "obligation: a key lemma, explicit construction, critical transformation, "
        "reproducible counterexample, or an obstruction proving the old route cannot work. "
        "Derive every claim from the statement. If none is found, say NOT_FOUND.\n\n"
        f"Subject: {subject}\nExact target: {target or obligations}\n"
        f"Output obligations: {obligations}\n"
        f"Attempted method class: {attempted_method or 'unknown'}\n"
        "Established: only the original hypotheses; no fact from the prior draft is trusted.\n"
        "Unresolved obligation: find a bridge that makes a complete solution executable.\n\n"
        "Output only these fields; DERIVATION has at most six compact steps:\n"
        "STATUS: FOUND or NOT_FOUND\n"
        "BRIDGE_TYPE: lemma|construction|transformation|counterexample|obstruction\n"
        "BREAKTHROUGH: exact statement\nDERIVATION: verification from the hypotheses\n"
        "TARGETED_USE: how it completes the original target\n"
        "FALSIFIER: one reproducible check\n\n"
        f"Original problem:\n{problem}"
    )


def _targeted_finish_instruction(spec) -> str:
    obligations = "; ".join(
        part.description for part in spec.answer_contract.parts if part.strict
    ) or "the complete requested result"
    if spec.profile.language == "zh":
        return (
            "把上面的突破当作未经认证的候选：先核对其推导和适用条件。若成立，只沿这条桥梁"
            "完成原目标；若不成立，在同一预算内改用一个独立的决定性桥梁。不要评价草稿、"
            "不要列备用路线。第一行必须是 FINAL: 后接完整可判分答案；随后只写最短必要证明。"
            f"必须覆盖：{obligations}。"
        )
    return (
        "Treat the proposed breakthrough as untrusted: first verify its derivation and "
        "hypotheses. If valid, use only that bridge to complete the original target; if "
        "invalid, replace it with one independent decisive bridge within the same budget. "
        "Do not review the draft or list alternatives. The first line must be FINAL: "
        "followed by the complete gradable answer, then only the shortest necessary proof. "
        f"Cover: {obligations}."
    )


def _run_continuation_branch(
    agent: SubmissionAgent,
    *,
    primary_request: str,
    primary_raw: str,
    spec,
    evidence,
    trace: list[dict],
    stage_prefix: str,
    thinking_mode: bool,
) -> tuple[Attempt, Attempt]:
    first_raw, first_result = agent._call(
        primary_request,
        stage=f"{stage_prefix}_continuation",
        max_tokens=BRANCH_SEARCH_TOKENS,
        temperature=0.1,
        thinking_mode=thinking_mode,
        trace=trace,
        prior_response=primary_raw,
        followup=agent._continuation_instruction(spec),
    )
    first = _assess(
        agent,
        first_raw,
        first_result,
        source=f"{stage_prefix}_continuation",
        spec=spec,
        evidence=evidence,
        independence_group=stage_prefix,
    )
    finish_raw, finish_result = agent._call(
        primary_request,
        stage=f"{stage_prefix}_finish",
        max_tokens=BRANCH_FINISH_TOKENS,
        temperature=0.1,
        thinking_mode=thinking_mode,
        trace=trace,
        prior_response=first_raw,
        followup=agent._conclusion_only_instruction(spec),
    )
    finish = _assess(
        agent,
        finish_raw,
        finish_result,
        source=f"{stage_prefix}_finish",
        spec=spec,
        evidence=evidence,
        independence_group=stage_prefix,
    )
    return first, finish


def _run_breakthrough_branch(
    agent: SubmissionAgent,
    *,
    statement: str,
    spec,
    evidence,
    trace: list[dict],
) -> tuple[Attempt, Attempt, str]:
    breakthrough_request = _breakthrough_request(
        statement, spec, str(spec.primary_method or "")
    )
    first_raw, first_result = agent._call(
        breakthrough_request,
        stage="branch_b_breakthrough",
        max_tokens=BRANCH_SEARCH_TOKENS,
        temperature=0.2,
        thinking_mode=False,
        trace=trace,
    )
    first = _assess(
        agent,
        first_raw,
        first_result,
        source="branch_b_breakthrough",
        spec=spec,
        evidence=evidence,
        independence_group="branch_b",
    )
    finish_raw, finish_result = agent._call(
        breakthrough_request,
        stage="branch_b_targeted_finish",
        max_tokens=BRANCH_FINISH_TOKENS,
        temperature=0.1,
        thinking_mode=False,
        trace=trace,
        prior_response=first_raw,
        followup=_targeted_finish_instruction(spec),
    )
    finish = _assess(
        agent,
        finish_raw,
        finish_result,
        source="branch_b_targeted_finish",
        spec=spec,
        evidence=evidence,
        independence_group="branch_b",
    )
    return first, finish, first_raw


def run_pair(
    problem: str,
    idx: str,
    client: InternChatClient,
    *,
    experiment: str,
) -> dict[str, Any]:
    agent = SubmissionAgent(client)
    agent.local_candidate_diagnostics = True
    spec = build_problem_spec(problem)
    statement = spec.problem_text or str(problem).strip()
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
    tool_names = agent._model_tool_names(
        statement,
        spec,
        allow_complex_subproblem=agent.enable_complex_subproblem_tools,
    )
    first_raw, first_result = agent._call(
        primary_request,
        stage="shared_primary",
        max_tokens=budget.solve_tokens,
        temperature=0.2,
        thinking_mode=hidden_thinking,
        trace=trace,
        model_tools=agent.model_math_tools if tool_names else None,
        model_tool_names=tool_names,
        max_tool_rounds=1,
        require_model_tool=agent._model_tool_required(statement, spec),
    )
    model_evidence = agent._model_tool_evidence(first_result)
    if model_evidence:
        evidence = tuple(model_evidence)
    primary = _assess(
        agent,
        first_raw,
        first_result,
        source="shared_primary",
        spec=spec,
        evidence=evidence,
        independence_group="shared",
    )

    b_first = sum(idx.encode("utf-8")) % 2 == 1
    continuation_args = {
        "agent": agent,
        "primary_request": primary_request,
        "primary_raw": first_raw,
        "spec": spec,
        "evidence": evidence,
        "trace": trace,
    }
    breakthrough_raw = ""
    if experiment == "thinking":
        if b_first:
            b1, b2 = _run_continuation_branch(
                **continuation_args,
                stage_prefix="branch_b",
                thinking_mode=True,
            )
            a1, a2 = _run_continuation_branch(
                **continuation_args,
                stage_prefix="branch_a",
                thinking_mode=False,
            )
        else:
            a1, a2 = _run_continuation_branch(
                **continuation_args,
                stage_prefix="branch_a",
                thinking_mode=False,
            )
            b1, b2 = _run_continuation_branch(
                **continuation_args,
                stage_prefix="branch_b",
                thinking_mode=True,
            )
        branch_b_stages = {"branch_b_continuation", "branch_b_finish"}
    else:
        a1, a2 = _run_continuation_branch(
            **continuation_args,
            stage_prefix="branch_a",
            thinking_mode=False,
        )
        b1, b2, breakthrough_raw = _run_breakthrough_branch(
            agent,
            statement=statement,
            spec=spec,
            evidence=evidence,
            trace=trace,
        )
        branch_b_stages = {"branch_b_breakthrough", "branch_b_targeted_finish"}

    primary_candidate = primary.candidate if primary.usable else None
    primary_answer = (
        agent._render_submission(primary_candidate.answer, spec)
        if primary_candidate is not None else ""
    )
    branch_a = _branch_record(
        agent, spec, (a1, a2), primary, trace,
        {"branch_a_continuation", "branch_a_finish"},
    )
    branch_b_attempts = (b1, b2)
    if experiment == "breakthrough":
        # In breakthrough mode B1 is proof-state IR, not a gradable answer.
        branch_b_attempts = (
            Attempt(
                raw=b1.raw,
                candidate=None,
                usable=False,
                truncated=b1.truncated,
            ),
            b2,
        )
    branch_b = _branch_record(
        agent,
        spec,
        branch_b_attempts,
        primary,
        trace,
        branch_b_stages,
    )
    if experiment == "breakthrough":
        branch_b.update({
            "breakthrough_claimed": bool(FOUND.search(breakthrough_raw)),
            "breakthrough_head": breakthrough_raw[:1600],
            "breakthrough_tail": (
                breakthrough_raw[-2400:] if len(breakthrough_raw) > 1600 else ""
            ),
        })
    return {
        "status": "success",
        "experiment": experiment,
        "execution_order": "B_then_A" if experiment == "thinking" and b_first else "A_then_B",
        "budget_contract": {
            "shared_primary_not_charged_to_branches": True,
            "branch_calls": BRANCH_MAX_CALLS,
            "branch_tokens": BRANCH_SEARCH_TOKENS + BRANCH_FINISH_TOKENS,
            "branch_a_thinking": False,
            "branch_b_thinking": experiment == "thinking",
            "same_continuation_context": experiment == "thinking",
            "same_prompts_and_temperatures": experiment == "thinking",
            "request_timeout_seconds": 120,
        },
        "primary": {
            **_stage_stats(trace, {"shared_primary"}),
            "truncated": primary.truncated,
            "candidate_usable": primary_candidate is not None,
            "candidate": str(primary_candidate.answer if primary_candidate else "")[:12_000],
            "final_response": primary_answer,
        },
        "branch_a": branch_a,
        "branch_b": branch_b,
    }


def annotate(record: dict[str, Any], reference: str) -> None:
    if not reference:
        return
    record["offline_reference_available"] = True
    for key in ("primary", "branch_a", "branch_b"):
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
            # This lookup happens only after all model calls and is never passed
            # into run_pair, a prompt, a tool, or SubmissionAgent.
            annotate(record, load_reference(reference_dir, idx))
        except Exception as exc:  # noqa: BLE001 - retain one diagnostic per item.
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
            "shared_primary": True,
            "reference_loaded_after_both_branches": True,
            "branch_order": "idx_checksum_crossover" if args.experiment == "thinking" else "A_then_B",
            "branch_calls": BRANCH_MAX_CALLS,
            "branch_tokens": BRANCH_SEARCH_TOKENS + BRANCH_FINISH_TOKENS,
            "branch_a_thinking": False,
            "branch_b_thinking": args.experiment == "thinking",
            "same_context_prompt_temperature_tokens": args.experiment == "thinking",
            "frozen_indices": [str(item.get("idx")) for item in items],
            "promotion_gate": {
                "branch_b_net_gain_gt_branch_a": True,
                "branch_b_right_to_wrong_eq_zero": True,
                "replicate_on_frozen_holdout": True,
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
            reference_dir=args.reference_dir,
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
