"""Split an audited E3 bucket into actionable candidate-generation subtypes.

This is an offline diagnostic.  It uses only problem statements and human
failure notes, never reference answers, and does not affect runtime routing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


CONSTRUCTION = re.compile(
    r"construction|construct|sharp (?:lower|upper|bound|configuration|arrangement)|"
    r"extremal|optimality|optimum|maximal|minimal|achiev|"
    r"构造|极值|最优|达到(?:上界|下界)",
    re.IGNORECASE,
)
GAME = re.compile(
    r"game|winning strategy|pairing strategy|first move|minimax|博弈|策略|先手|后手",
    re.IGNORECASE,
)
RECURRENCE = re.compile(
    r"recurrence|recursive|sequence|periodic|closed form|fibonacci|lucas|"
    r"递推|递归|数列|序列|周期|通项",
    re.IGNORECASE,
)
FINITE_EXECUTION = re.compile(
    r"enumerat|permanent|determinant|computed correctly for small|small cases|"
    r"dynamic search|exhaustive|first long gap|never computed|calculation|"
    r"穷举|枚举|行列式|计算|动态搜索",
    re.IGNORECASE,
)
ABSTRACT = re.compile(
    r"functional|classification|completeness|additive closure|prime divisor|"
    r"homomorphism|group|ring|field|measure|topolog|"
    r"函数方程|分类|完备|闭包|同态|群|环|域|测度|拓扑",
    re.IGNORECASE,
)
GEOMETRY = re.compile(
    r"triangle|circle|hexagon|polar|locus|geometr|三角形|圆|六边形|极线|轨迹|几何",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        item.setdefault("idx", line_number)
        records.append(item)
    return records


def classify(problem: str, note: str) -> tuple[str, str]:
    combined = f"{problem}\n{note}"
    # A missing sharp witness/bound is mathematically different from a finite
    # computation that merely has not been executed, so it takes precedence.
    if CONSTRUCTION.search(note):
        return "CONSTRUCTION_EXTREMAL", "missing sharp bound or attaining construction"
    if GAME.search(combined):
        return "KEY_LEMMA_STRATEGY", "missing invariant, pairing, or minimax strategy"
    if RECURRENCE.search(combined):
        return "RECURRENCE_EXECUTION", "sequence structure or scalable recurrence missing"
    if FINITE_EXECUTION.search(note):
        return "FINITE_COMPUTATION", "finite calculation or exhaustive execution unfinished"
    if ABSTRACT.search(combined):
        return "ABSTRACT_PROOF", "classification or abstract closure lemma missing"
    if GEOMETRY.search(combined):
        return "KEY_LEMMA_GEOMETRY", "decisive geometric relation missing"
    return "KEY_LEMMA_GENERAL", "decisive bridge missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify E3 failure subtypes.")
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--label-file", required=True, type=Path)
    parser.add_argument("--label", default="E3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = json.loads(args.label_file.read_text(encoding="utf-8"))
    rows = []
    for item in load_jsonl(args.input_file):
        idx = str(item.get("idx"))
        label = labels.get(idx, {})
        if str(label.get("label", "")) != args.label:
            continue
        note = str(label.get("note", ""))
        subtype, reason = classify(str(item.get("problem", "")), note)
        rows.append({"idx": idx, "subtype": subtype, "reason": reason})

    counts = Counter(row["subtype"] for row in rows)
    report = {
        "items": len(rows),
        "subtype_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "items_by_subtype": {
            subtype: [row for row in rows if row["subtype"] == subtype]
            for subtype, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "items": report["items"],
        "subtype_counts": report["subtype_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
