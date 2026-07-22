"""Run the labeled error set against the configured live model API."""

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import InternChatClient
from user_agent import ReasoningAgent


JUDGE_PROMPT = (
    "你是严格数学评测器。根据题目、标准答案和学生回答，判断学生回答是否在数学上"
    "正确且覆盖题目要求。只输出 PASS 或 FAIL。"
)


def main() -> None:
    input_path = Path("sample_data/error.jsonl")
    items = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items.sort(key=lambda item: item["idx"] == 990)
    only_idx = os.environ.get("EVAL_IDX")
    if only_idx:
        items = [item for item in items if item["idx"] == int(only_idx)]
        if not items:
            raise ValueError(f"Unknown EVAL_IDX: {only_idx}")
    client = InternChatClient(timeout=120, retry=2)
    agent = ReasoningAgent(client)
    records = []

    for position, item in enumerate(items, start=1):
        result = agent.solve(item["problem"], {"idx": item["idx"]})
        verdict = client.chat(
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"题目：\n{item['problem']}\n\n标准答案：\n{item['answer']}"
                        f"\n\n学生回答：\n{result['final_response']}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        ).strip().upper()
        matches = re.findall(r"\b(PASS|FAIL)\b", verdict)
        if not matches:
            raise RuntimeError(f"Judge returned no PASS/FAIL verdict: {verdict!r}")
        passed = matches[-1] == "PASS"
        record = {
            "idx": item["idx"],
            "type": item["type"],
            "passed": passed,
            "verdict": verdict,
            "response_chars": len(result["final_response"]),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    passed = sum(record["passed"] for record in records)
    print(json.dumps({
        "passed": passed,
        "total": len(records),
        "accuracy": passed / len(records),
        "records": records,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
