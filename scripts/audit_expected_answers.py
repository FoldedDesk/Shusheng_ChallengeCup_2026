"""Audit whether generated answers visibly cover public reference answers.

This is deliberately conservative: algebraically equivalent phrasings are
reported for manual review instead of being marked wrong automatically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("【最终答案】", "")
    text = text.replace("\\", "").replace("{", "").replace("}", "")
    return re.sub(r"[\s，,。；;：:（）()$]", "", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    references = {
        item["idx"]: item["answer"]
        for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }
    missing, mismatches = [], []
    for index, expected in references.items():
        path = Path(args.output_dir) / f"{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        actual = normalize(record.get("final_response"))
        if not actual or normalize(expected) not in actual:
            mismatches.append({"idx": index, "expected": expected, "actual": record.get("final_response", "")})

    print(json.dumps({
        "total": len(references),
        "completed": len(references) - len(missing),
        "missing": missing,
        "manual_review": mismatches,
    }, ensure_ascii=False, indent=2))
    return 0 if not missing and not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
