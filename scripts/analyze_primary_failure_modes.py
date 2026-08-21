"""Build an offline evidence matrix for first-primary-solve failures.

This script is development-only.  It reads a reference replay and a local agent
replay after both runs have finished; no reference answer is passed to the
runtime agent or written into production code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
from reasoning.truncation_state import classify_truncated_output


PRIMARY_SOURCES = {"primary", "primary_complete", "primary_direct"}
RECOVERY_SOURCE_MARKERS = ("recovery", "continuation", "rescue")


def _path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.stem) if path.stem.isdigit() else (10**12, path.stem)


def _steps(record: dict) -> list[dict]:
    trace = record.get("trace", [])
    if not isinstance(trace, list):
        return []
    return [step for step in trace if isinstance(step, dict)]


def _stage_call(steps: list[dict], stage: str) -> dict:
    for step in steps:
        content = step.get("content")
        if (
            step.get("step") == "model_call"
            and isinstance(content, dict)
            and content.get("stage") == stage
        ):
            return content
    return {}


def _stage_excerpt(steps: list[dict], stage: str) -> dict:
    for step in steps:
        content = step.get("content")
        if (
            step.get("step") == "local_model_output"
            and isinstance(content, dict)
            and content.get("stage") == stage
        ):
            return content
    return {}


def _truncation_state(steps: list[dict], excerpt: str) -> dict:
    for step in steps:
        content = step.get("content")
        if (
            step.get("step") == "truncation_state"
            and isinstance(content, dict)
            and content.get("stage") == "primary"
        ):
            return {
                "phase": str(content.get("phase", "UNKNOWN")),
                "recoverability": str(content.get("recoverability", "UNKNOWN")),
                "evidence": str(content.get("evidence", "runtime_classifier")),
            }
    return classify_truncated_output(excerpt)


def _candidate_entries(steps: list[dict], final_response: str) -> list[dict]:
    entries: list[dict] = []
    for step in steps:
        content = step.get("content")
        if step.get("step") == "candidate_audit" and isinstance(content, list):
            entries.extend(entry for entry in content if isinstance(entry, dict))
        elif step.get("step") == "candidate_probe" and isinstance(content, dict):
            candidates = content.get("candidates", [])
            if isinstance(candidates, list):
                entries.extend(entry for entry in candidates if isinstance(entry, dict))
        elif step.get("step") == "selection" and isinstance(content, dict):
            candidate = content.get("candidate")
            if isinstance(candidate, str) and candidate.strip():
                entries.append({
                    "source": str(content.get("source", "selection")),
                    "candidate": candidate,
                })

    submitted = str(final_response or "").strip()
    if submitted:
        selection = next((
            step.get("content", {})
            for step in steps
            if step.get("step") == "selection"
            and isinstance(step.get("content"), dict)
        ), {})
        entries.append({
            "source": str(selection.get("source", "final_response")),
            "candidate": submitted,
        })

    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        candidate = str(entry.get("candidate", "")).strip()
        source = str(entry.get("source", "unknown")).strip()
        if not candidate or (source, candidate) in seen:
            continue
        seen.add((source, candidate))
        unique.append({"source": source, "candidate": candidate})
    return unique


def _source_kind(source: str) -> str:
    folded = str(source or "").casefold()
    if folded in PRIMARY_SOURCES or folded.startswith("primary_direct"):
        return "primary"
    if any(marker in folded for marker in RECOVERY_SOURCE_MARKERS):
        return "recovery"
    return "other"


def _automatic_bucket(
    *,
    primary_correct: bool,
    recovery_correct: bool,
    primary_call: dict,
    truncation: dict,
    primary_candidates: list[dict],
    excerpt: str,
) -> tuple[str, str, str]:
    """Return only evidence-backed labels; ambiguous mathematics stays REVIEW."""
    if primary_correct:
        return "OK", "high", "first primary already contained a correct answer"

    call_status = str(primary_call.get("status", "missing")).casefold()
    if call_status not in {"ok", "completed"}:
        return "TRANSPORT", "high", "primary model call failed or was empty"

    truncated = bool(primary_call.get("provider_truncated"))
    phase = str(truncation.get("phase", "UNKNOWN"))
    recoverability = str(truncation.get("recoverability", "UNKNOWN"))
    if truncated and recovery_correct and recoverability == "HIGH":
        return "E5", "high", "high-recoverability primary was rescued to the reference answer"
    if truncated and phase == "BEFORE_METHOD":
        return "E3", "medium", "primary exhausted its budget before committing to a method"
    if truncated and phase == "DURING_METHOD_SEARCH":
        return (
            "REVIEW_E2_E3_E4_E5",
            "manual",
            "search-language heuristics cannot distinguish a late correction from a missing breakthrough",
        )
    if truncated and recoverability == "LOW":
        return (
            "REVIEW_E2_E3_E4_E5",
            "manual",
            "low heuristic recoverability still requires inspection of the mathematical state",
        )

    structural_reasons = Finalizer.validate_structure(excerpt)
    if excerpt and not truncated and structural_reasons and not primary_candidates:
        return "E6", "medium", "complete transport yielded no structurally usable primary candidate"

    if primary_candidates:
        return (
            "REVIEW_E1_E2_E4",
            "manual",
            "a complete but wrong mathematical candidate needs semantic inspection",
        )
    if truncated and recoverability in {"MEDIUM", "HIGH"}:
        return (
            "REVIEW_E2_E3_E4_E5",
            "manual",
            "truncated route showed progress but correctness of its method/lemma is unknown",
        )
    return "REVIEW_E1_E2_E3_E4_E6", "manual", "insufficient first-primary evidence"


def build_matrix(reference_dir: Path, agent_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for reference_path in sorted(reference_dir.glob("*.json"), key=_path_key):
        agent_path = agent_dir / reference_path.name
        if not agent_path.is_file():
            continue
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        reward = reference.get("reward_model", {})
        answer = str(reward.get("ground_truth", "")) if isinstance(reward, dict) else ""
        steps = _steps(agent)
        primary_call = _stage_call(steps, "primary")
        excerpt_entry = _stage_excerpt(steps, "primary")
        head = str(excerpt_entry.get("head", ""))
        tail = str(excerpt_entry.get("tail", ""))
        excerpt = f"{head}\n{tail}".strip()
        entries = _candidate_entries(steps, str(agent.get("final_response", "")))
        primary_candidates = [
            entry for entry in entries if _source_kind(entry["source"]) == "primary"
        ]
        recovery_candidates = [
            entry for entry in entries if _source_kind(entry["source"]) == "recovery"
        ]
        primary_correct = any(
            equivalent_answers(entry["candidate"], answer)
            for entry in primary_candidates
        )
        recovery_correct = any(
            equivalent_answers(entry["candidate"], answer)
            for entry in recovery_candidates
        )
        truncation = (
            _truncation_state(steps, excerpt)
            if primary_call.get("provider_truncated")
            else {"phase": "NOT_TRUNCATED", "recoverability": "N/A", "evidence": "complete"}
        )
        bucket, confidence, reason = _automatic_bucket(
            primary_correct=primary_correct,
            recovery_correct=recovery_correct,
            primary_call=primary_call,
            truncation=truncation,
            primary_candidates=primary_candidates,
            excerpt=excerpt,
        )
        rows.append({
            "idx": str(reference.get("idx", reference_path.stem)),
            "final_correct": equivalent_answers(str(agent.get("final_response", "")), answer),
            "first_primary_correct": primary_correct,
            "recovery_correct": recovery_correct,
            "primary_status": str(primary_call.get("status", "missing")),
            "primary_truncated": bool(primary_call.get("provider_truncated")),
            "primary_finish_reason": str(primary_call.get("finish_reason", "unavailable")),
            "primary_output_chars": int(primary_call.get("output_length", 0) or 0),
            "truncation_phase": truncation["phase"],
            "recoverability": truncation["recoverability"],
            "automatic_bucket": bucket,
            "confidence": confidence,
            "reason": reason,
            "primary_candidates": primary_candidates,
            "recovery_candidates": recovery_candidates,
            "problem": str(reference.get("problem", "")),
            "reference_answer": answer,
            "final_response": str(agent.get("final_response", "")),
            "primary_head": head,
            "primary_tail": tail,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    buckets = Counter(row["automatic_bucket"] for row in rows)
    phases = Counter(
        row["truncation_phase"] for row in rows if row["primary_truncated"]
    )
    recoverability = Counter(
        row["recoverability"] for row in rows if row["primary_truncated"]
    )
    return {
        "items": len(rows),
        "final_correct": sum(bool(row["final_correct"]) for row in rows),
        "first_primary_correct": sum(bool(row["first_primary_correct"]) for row in rows),
        "recovery_correct": sum(bool(row["recovery_correct"]) for row in rows),
        "primary_truncated": sum(bool(row["primary_truncated"]) for row in rows),
        "automatic_buckets": dict(sorted(buckets.items())),
        "truncation_phases": dict(sorted(phases.items())),
        "recoverability": dict(sorted(recoverability.items())),
        "manual_review_required": sum(
            str(row["automatic_bucket"]).startswith("REVIEW") for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify first-primary failures from an offline replay."
    )
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_matrix(args.reference_dir, args.agent_dir)
    report = {"summary": summarize(rows), "items": rows}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
