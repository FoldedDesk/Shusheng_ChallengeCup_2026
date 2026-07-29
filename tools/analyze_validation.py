"""Summarize validation outputs into score-oriented error clusters."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> Dict[int, Dict]:
    return {
        row["idx"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def load_outputs(path: Path) -> Iterable[Dict]:
    for output_path in sorted(path.glob("*.json")):
        yield json.loads(output_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_jsonl(args.input_file)
    clusters: Dict[str, List[int]] = {
        "runtime_error": [],
        "protocol_failure": [],
        "english_reasoning": [],
        "oversized_response": [],
        "reference_text_miss": [],
    }
    subjects: Counter[str] = Counter()
    total = success = 0
    english = re.compile(
        r"thinking\s*process|analyze the request|let.?s |i (?:will|need to|should)|wait,",
        re.IGNORECASE,
    )

    for output in load_outputs(args.output_dir):
        total += 1
        idx = output.get("idx")
        item = source.get(idx, {})
        subjects[item.get("subject", "unknown")] += 1
        answer = output.get("final_response", "")
        if output.get("status") != "success":
            clusters["runtime_error"].append(idx)
            continue
        success += 1
        if answer in {"TRUNCATED_ALL", "ALL_GARBAGE", ""}:
            clusters["protocol_failure"].append(idx)
        if english.search(answer):
            clusters["english_reasoning"].append(idx)
        if len(answer) > 1200:
            clusters["oversized_response"].append(idx)
        expected = item.get("answer", "")
        if expected and normalize(expected) not in normalize(answer) and normalize(answer) not in normalize(expected):
            clusters["reference_text_miss"].append(idx)

    print(json.dumps({
        "total": total,
        "success": success,
        "clusters": clusters,
        "subjects": dict(subjects),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
