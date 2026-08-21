"""Summarize paired recovery outputs without reading problem statements."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.stem) if path.stem.isdigit() else (10**12, path.stem)


def _truth(record: dict[str, Any], branch: str) -> bool:
    return bool(record.get(branch, {}).get("offline_correct"))


def summarize(output_dir: Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("*.json"), key=_path_key)
        if path.stem.isdigit()
    ]
    successful = [record for record in records if record.get("status") == "success"]
    errors = [str(record.get("idx")) for record in records if record.get("status") != "success"]
    pair_counts: Counter[str] = Counter()
    transitions: dict[str, list[str]] = {
        "a_only": [], "b_only": [], "both": [], "neither": [],
        "primary_wrong_a_right": [], "primary_wrong_b_right": [],
        "primary_right_a_wrong": [], "primary_right_b_wrong": [],
    }
    breakthrough: Counter[str] = Counter()
    breakthrough_ids: dict[str, list[str]] = {
        "claimed_and_correct": [],
        "claimed_but_wrong": [],
        "not_claimed_but_correct": [],
        "not_claimed_and_wrong": [],
    }
    for record in successful:
        idx = str(record.get("idx"))
        primary = _truth(record, "primary")
        a = _truth(record, "branch_a")
        b = _truth(record, "branch_b")
        pair = "both" if a and b else "a_only" if a else "b_only" if b else "neither"
        pair_counts[pair] += 1
        transitions[pair].append(idx)
        if not primary and a:
            transitions["primary_wrong_a_right"].append(idx)
        if not primary and b:
            transitions["primary_wrong_b_right"].append(idx)
        if primary and not a:
            transitions["primary_right_a_wrong"].append(idx)
        if primary and not b:
            transitions["primary_right_b_wrong"].append(idx)
        claimed = bool(record.get("branch_b", {}).get("breakthrough_claimed"))
        key = (
            "claimed_and_correct" if claimed and b
            else "claimed_but_wrong" if claimed
            else "not_claimed_but_correct" if b
            else "not_claimed_and_wrong"
        )
        breakthrough[key] += 1
        breakthrough_ids[key].append(idx)

    def branch_stats(name: str) -> dict[str, Any]:
        calls = sum(int(record.get(name, {}).get("calls", 0) or 0) for record in successful)
        elapsed = sum(
            int(record.get(name, {}).get("elapsed_ms", 0) or 0)
            for record in successful
        )
        hits = sum(_truth(record, name) for record in successful)
        primary_wrong_right = len(transitions[f"primary_wrong_{'a' if name == 'branch_a' else 'b'}_right"])
        primary_right_wrong = len(transitions[f"primary_right_{'a' if name == 'branch_a' else 'b'}_wrong"])
        net = primary_wrong_right - primary_right_wrong
        return {
            "hits": hits,
            "accuracy": hits / len(successful) if successful else 0.0,
            "calls": calls,
            "elapsed_ms": elapsed,
            "mean_calls_per_item": calls / len(successful) if successful else 0.0,
            "wrong_to_right": primary_wrong_right,
            "right_to_wrong": primary_right_wrong,
            "net_gain": net,
            "net_gain_per_call": net / calls if calls else 0.0,
            "calls_per_net_gain": calls / net if net > 0 else None,
            "seconds_per_net_gain": elapsed / 1000 / net if net > 0 else None,
            "first_call_truncated": sum(
                bool((record.get(name, {}).get("attempt_truncated") or [False])[0])
                for record in successful
            ),
            "candidate_usable": sum(
                bool(record.get(name, {}).get("candidate_usable"))
                for record in successful
            ),
        }

    primary_hits = sum(_truth(record, "primary") for record in successful)
    return {
        "items": len(records),
        "successful": len(successful),
        "errors": len(errors),
        "error_ids": errors,
        "primary_hits": primary_hits,
        "primary_accuracy": primary_hits / len(successful) if successful else 0.0,
        "branch_a": branch_stats("branch_a"),
        "branch_b": branch_stats("branch_b"),
        "paired": dict(sorted(pair_counts.items())),
        "paired_ids": transitions,
        "breakthrough": dict(sorted(breakthrough.items())),
        "breakthrough_ids": breakthrough_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output_dir)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
