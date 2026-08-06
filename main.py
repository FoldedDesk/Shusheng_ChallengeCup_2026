import argparse
import asyncio
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
FAILED_ANSWER = "本题未能在限定时间内生成可验证答案。"


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
    return parser.parse_args()


def preflight_client(client: InternChatClient) -> None:
    """Fail fast locally when credentials or network access are unavailable."""
    response = client.chat(
        messages=[{"role": "user", "content": "只回复OK"}],
        temperature=0.0,
        max_tokens=8,
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

    client = InternChatClient()
    if args.preflight:
        preflight_client(client)
    agent = ReasoningAgent(client=client)
    semaphore = asyncio.Semaphore(LOCAL_MAX_CONCURRENCY)

    print(f"Loaded {len(items)} items. Max concurrency: {LOCAL_MAX_CONCURRENCY}.")
    tasks = [process_item(agent, item, output_dir, semaphore) for item in items]
    await asyncio.gather(*tasks)
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
