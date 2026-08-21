"""Audit statement-routed local tool opportunities without solving problems.

The report contains only classification metadata and never reads reference
answers.  A local label file may select an already-audited error bucket.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from reasoning.local_tool_opportunity import detect_local_tool_opportunity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local tool opportunity routing.")
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--label-file", type=Path)
    parser.add_argument("--label", default="E3")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    items = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        item.setdefault("idx", line_number)
        items.append(item)
    return items


def main() -> None:
    args = parse_args()
    items = load_items(args.input_file)
    if args.label_file is not None:
        labels = json.loads(args.label_file.read_text(encoding="utf-8"))
        items = [
            item for item in items
            if str(labels.get(str(item.get("idx")), {}).get("label", ""))
            == args.label
        ]

    rows = []
    for item in items:
        problem = str(item.get("problem", ""))
        spec = build_problem_spec(problem)
        strict = detect_local_tool_opportunity(problem, spec)
        derived = detect_local_tool_opportunity(problem, spec, allow_derived=True)
        rows.append({
            "idx": str(item.get("idx")),
            "strict": strict.trace_content(),
            "derived": derived.trace_content(),
            "primary_subject": str(getattr(spec.profile, "primary_subject", "")),
            "topic": str(getattr(spec.profile, "topic", "")),
        })

    eligible = [row for row in rows if row["derived"]["eligible"]]
    report = {
        "items": len(rows),
        "strict_eligible": sum(row["strict"]["eligible"] for row in rows),
        "derived_eligible": len(eligible),
        "kind_counts": dict(sorted(Counter(
            row["derived"]["kind"] for row in eligible
        ).items())),
        "tool_counts": dict(sorted(Counter(
            tool for row in eligible for tool in row["derived"]["allowed_tools"]
        ).items())),
        "eligible_items": eligible,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "eligible_items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
