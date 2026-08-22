import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Dict, List

from llm_client import InternChatClient
from user_agent import ReasoningAgent
from core.execution_limits import MAX_CONCURRENCY


LOCAL_MAX_CONCURRENCY = max(
    1,
    min(int(os.environ.get("LOCAL_MAX_CONCURRENCY", str(MAX_CONCURRENCY))), MAX_CONCURRENCY),
)
# Kept for compatibility with local imports. Diagnostics belong in the error
# and trace fields; a failure sentence is never a gradable mathematical answer.
FAILED_ANSWER = "0"


class _NoThinkingClientProxy:
    """Local A/B proxy; the platform injects its client directly."""

    def __init__(self, client) -> None:
        self._client = client

    def chat_result(self, *args, **kwargs):
        kwargs["thinking_mode"] = False
        return self._client.chat_result(*args, **kwargs)

    def chat(self, *args, **kwargs):
        kwargs["thinking_mode"] = False
        return self._client.chat(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("idx", line_number)
            items.append(item)
    return items


def result_path(output_dir: Path, item: Dict) -> Path:
    return output_dir / f"{item['idx']}.json"


def is_processed(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def write_json(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    tmp_path.replace(path)


def build_output_record(item: Dict, agent_result: Dict) -> Dict:
    final_response = agent_result.get("final_response", "")
    if not isinstance(final_response, str) or not final_response.strip():
        raise ValueError("agent.solve must return a non-empty string field: final_response")

    output = {
        "idx": item["idx"],
        "status": "success",
        "final_response": final_response,
        "trace": agent_result.get("trace", []),
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Competition sample reasoning agent.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL.")
    parser.add_argument("--output_dir", required=True, help="Directory for per-problem JSON outputs.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Verify one lightweight Client call before starting the local batch.",
    )
    parser.add_argument(
        "--reasoning-policy",
        choices=(
            "default",
            "no-thinking",
            "visible-deep",
            "quick-consensus",
            "no-rag",
            "long-primary",
            "short-recovery",
            "judge7-model",
            "dual-deep-8k",
            "dual-deep-16-8",
            "compact-answer-first",
            "mog-portfolio",
            "mog-portfolio-thinking",
            "mog-portfolio-thinking-a6k",
            "candidate-audit",
            "no-candidate-audit",
            "single-route",
            "dual-consensus",
            "quick-consensus",
            "compact-primary-prompt",
            "verbose-primary-prompt",
            "primary-plus-1k",
            "primary-minus-1k",
            "complex-subproblem-tools",
        ),
        default="default",
        help="Local A/B override; the platform does not call this runner.",
    )
    parser.add_argument(
        "--trace-candidates",
        action="store_true",
        help="Include bounded candidate text in local traces for post-run selection audits.",
    )
    parser.add_argument(
        "--trace-model-output",
        action="store_true",
        help="Include bounded model-response excerpts in local-only diagnostic traces.",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated local item ids; never used by the platform entry point.",
    )
    return parser.parse_args()


def preflight_client(client: InternChatClient) -> None:
    """Fail fast locally when credentials or network access are unavailable."""
    response = client.chat(
        messages=[{"role": "user", "content": "只回复OK"}],
        temperature=0.0,
        max_tokens=8,
        thinking_mode=False,
    )
    if not str(response or "").strip():
        raise RuntimeError("Client preflight returned an empty response")


def solve_item(agent: ReasoningAgent, item: Dict) -> Dict:
    result = agent.solve(
        problem=item["problem"],
        metadata={"idx": item["idx"], "source": item.get("source", "")},
    )
    return build_output_record(item, result)


async def process_item(
    agent: ReasoningAgent,
    item: Dict,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    path = result_path(output_dir, item)
    if is_processed(path):
        print(f"Skip idx={item['idx']} because {path} already exists.")
        return

    async with semaphore:
        try:
            record = await asyncio.to_thread(solve_item, agent, item)
        except Exception as exc:  # noqa: BLE001 - keep one output file per input item.
            record = {
                "idx": item["idx"],
                "status": "error",
                "final_response": FAILED_ANSWER,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "trace": [],
            }
        await asyncio.to_thread(write_json, path, record)
        print(f"Finished idx={item['idx']}")


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    items = load_jsonl(input_path)
    selected_indices = {
        value.strip() for value in str(args.indices or "").split(",") if value.strip()
    }
    if selected_indices:
        items = [item for item in items if str(item.get("idx")) in selected_indices]

    client = InternChatClient()
    if args.preflight:
        # A health check must fail fast instead of consuming the same two
        # three-minute transport windows reserved for actual mathematics.
        preflight_client(InternChatClient(timeout=30, retry=1))
    runtime_client = client
    if args.reasoning_policy == "visible-deep":
        runtime_client = _NoThinkingClientProxy(client)
    agent = ReasoningAgent(client=runtime_client)
    if args.reasoning_policy in {"candidate-audit", "no-candidate-audit"}:
        agent.agent.enable_candidate_audit = (
            args.reasoning_policy == "candidate-audit"
        )
    if args.reasoning_policy in {"single-route", "dual-consensus"}:
        agent.agent.enable_blind_consensus = (
            args.reasoning_policy == "dual-consensus"
        )
    if args.reasoning_policy == "quick-consensus":
        agent.agent.enable_quick_consensus = True
    if args.reasoning_policy == "compact-primary-prompt":
        agent.agent.compact_primary_prompt = True
    if args.reasoning_policy == "verbose-primary-prompt":
        agent.agent.compact_primary_prompt = False
    if args.reasoning_policy == "complex-subproblem-tools":
        agent.agent.enable_complex_subproblem_tools = True
    if args.reasoning_policy in {"primary-plus-1k", "primary-minus-1k"}:
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget
        token_delta = (
            1024 if args.reasoning_policy == "primary-plus-1k" else -1024
        )

        def adjusted_primary_budget(
            spec, has_whole_tool_answer, *, deep_reasoning=False
        ):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls or not deep_reasoning:
                return budget
            return replace(
                budget,
                solve_tokens=max(3072, budget.solve_tokens + token_delta),
            )

        submission_module.plan_stage_budget = adjusted_primary_budget
    if args.reasoning_policy in {
        "mog-portfolio",
        "mog-portfolio-thinking",
        "mog-portfolio-thinking-a6k",
    }:
        agent.agent.enable_mog = True
        agent.agent.mog_route_thinking = (
            args.reasoning_policy != "mog-portfolio"
        )
        if args.reasoning_policy == "mog-portfolio-thinking-a6k":
            agent.agent.mog_route_token_limits = (6144, 4096)
    if args.trace_candidates:
        agent.agent.local_candidate_diagnostics = True
        original_candidate_trace = getattr(agent.agent, "_candidate_trace", None)

        if callable(original_candidate_trace):
            def local_candidate_trace(candidate, **extra):
                content = original_candidate_trace(candidate, **extra)
                content["candidate"] = str(candidate.answer or "")[:12000]
                return content

            agent.agent._candidate_trace = local_candidate_trace
    if args.trace_model_output:
        traced_original_call = agent.agent._call

        def local_traced_call(request, _call=traced_original_call, **kwargs):
            raw, result = _call(request, **kwargs)
            trace = kwargs.get("trace")
            if isinstance(trace, list):
                text = str(raw or "")
                trace.append({
                    "step": "local_model_output",
                    "content": {
                        "stage": kwargs.get("stage", ""),
                        "chars": len(text),
                        "head": text[:1200],
                        "tail": text[-2400:] if len(text) > 1200 else "",
                    },
                })
            return raw, result

        agent.agent._call = local_traced_call
    if args.reasoning_policy == "no-rag":
        from rag.card_retriever import RetrievalBundle

        agent.agent.retriever.retrieve = lambda spec: RetrievalBundle((), ())
    if args.reasoning_policy == "long-primary":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def long_primary_budget(spec, has_whole_tool_answer, *, deep_reasoning=False):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls or not deep_reasoning:
                return budget
            return replace(budget, solve_tokens=max(16_384, budget.solve_tokens))

        submission_module.plan_stage_budget = long_primary_budget
    if args.reasoning_policy == "short-recovery":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def short_recovery_budget(spec, has_whole_tool_answer, *, deep_reasoning=False):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls:
                return budget
            return replace(
                budget,
                repair_tokens=min(2048, budget.repair_tokens),
                emergency_tokens=min(2048, budget.emergency_tokens),
            )

        submission_module.plan_stage_budget = short_recovery_budget
    if args.reasoning_policy == "judge7-model":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def judge7_model_budget(spec, has_whole_tool_answer, *, deep_reasoning=False):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls or not deep_reasoning:
                return budget
            return replace(
                budget,
                solve_tokens=8192,
                review_tokens=8192,
                repair_tokens=4096,
                emergency_tokens=2048,
                max_calls=4,
            )

        submission_module.plan_stage_budget = judge7_model_budget
        judge7_original_call = agent.agent._call

        def judge7_model_call(request, _call=judge7_original_call, **kwargs):
            if kwargs.get("stage") == "independent_solve":
                kwargs["thinking_mode"] = True
            return _call(request, **kwargs)

        agent.agent._call = judge7_model_call
    if args.reasoning_policy == "dual-deep-8k":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def dual_deep_budget(spec, has_whole_tool_answer, *, deep_reasoning=False):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls or not deep_reasoning:
                return budget
            return replace(
                budget,
                solve_tokens=8192,
                review_tokens=8192,
                repair_tokens=4096,
                emergency_tokens=3072,
                max_calls=3,
            )

        submission_module.plan_stage_budget = dual_deep_budget
        agent.agent._hidden_thinking = agent.agent._deep_reasoning
        dual_deep_original_call = agent.agent._call

        def dual_deep_call(request, _call=dual_deep_original_call, **kwargs):
            if kwargs.get("stage") in {
                "independent_solve", "blind_rescue", "computation_tiebreak"
            }:
                kwargs["thinking_mode"] = True
                kwargs["max_tokens"] = max(8192, int(kwargs.get("max_tokens", 0)))
            return _call(request, **kwargs)

        agent.agent._call = dual_deep_call
    if args.reasoning_policy == "dual-deep-16-8":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def dual_deep_16_8_budget(
            spec, has_whole_tool_answer, *, deep_reasoning=False
        ):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls or not deep_reasoning:
                return budget
            return replace(
                budget,
                solve_tokens=16384,
                review_tokens=8192,
                repair_tokens=4096,
                emergency_tokens=3072,
                max_calls=3,
            )

        submission_module.plan_stage_budget = dual_deep_16_8_budget
        agent.agent._hidden_thinking = agent.agent._deep_reasoning
        dual_deep_16_8_original_call = agent.agent._call

        def dual_deep_16_8_call(
            request, _call=dual_deep_16_8_original_call, **kwargs
        ):
            if kwargs.get("stage") in {
                "independent_solve", "blind_rescue", "computation_tiebreak"
            }:
                kwargs["thinking_mode"] = True
                kwargs["max_tokens"] = max(8192, int(kwargs.get("max_tokens", 0)))
            return _call(request, **kwargs)

        agent.agent._call = dual_deep_16_8_call
    if args.reasoning_policy in {
        "no-thinking", "quick-consensus", "compact-answer-first"
    }:
        agent.agent._deep_reasoning = lambda spec: False
    if args.reasoning_policy == "quick-consensus":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def consensus_budget(spec, has_whole_tool_answer, *, deep_reasoning=False):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if budget.max_calls:
                budget = replace(budget, require_independent_review=True)
            return budget

        submission_module.plan_stage_budget = consensus_budget
    if args.reasoning_policy == "compact-answer-first":
        import core.submission_agent as submission_module

        original_budget = submission_module.plan_stage_budget

        def compact_answer_first_budget(
            spec, has_whole_tool_answer, *, deep_reasoning=False
        ):
            budget = original_budget(
                spec,
                has_whole_tool_answer,
                deep_reasoning=deep_reasoning,
            )
            if not budget.max_calls:
                return budget
            return replace(
                budget,
                solve_tokens=4096,
                review_tokens=4096,
                repair_tokens=2048,
                emergency_tokens=1536,
                require_independent_review=True,
                max_calls=3,
            )

        submission_module.plan_stage_budget = compact_answer_first_budget
    semaphore = asyncio.Semaphore(LOCAL_MAX_CONCURRENCY)

    print(f"Loaded {len(items)} items. Max concurrency: {LOCAL_MAX_CONCURRENCY}.")
    tasks = [process_item(agent, item, output_dir, semaphore) for item in items]
    await asyncio.gather(*tasks)
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
