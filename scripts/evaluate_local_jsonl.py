"""Evaluate completed local outputs after solving, never during Agent runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec  # noqa: E402
from reasoning.candidate_selector import assess_candidate  # noqa: E402
from reasoning.finalizer import Finalizer  # noqa: E402
from reasoning.math_equivalence import equivalent_answers  # noqa: E402


_UNAVAILABLE_REFERENCE = re.compile(
    r"未完成(?:独立)?验证|待(?:独立)?验证|"
    r"\b(?:reference\s+unavailable|unverified\s+reference|placeholder)\b",
    re.IGNORECASE,
)


def _usable_reference(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text and not _UNAVAILABLE_REFERENCE.search(text))


def _records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get("problem"), str):
                records.append(value)
    return records


def _reference_sidecar(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    references: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            idx = str(value.get("idx", ""))
            answer = str(value.get("answer", "")).strip()
            if not idx or not answer:
                raise ValueError(f"{path}:{line_number} has no idx/answer")
            if idx in references:
                raise ValueError(f"{path}:{line_number} duplicates idx {idx}")
            references[idx] = answer
    return references


def _selection(record: dict) -> dict:
    return next(
        (
            item.get("content", {})
            for item in record.get("trace", [])
            if item.get("step") == "selection"
        ),
        {},
    )


def evaluate(
    input_path: Path,
    output_dir: Path,
    answer_path: Path | None = None,
) -> dict:
    processed = hits = invalid = missing_obligations = calls = truncated = 0
    missing_outputs: list[str] = []
    unavailable_references: list[str] = []
    hit_ids: list[str] = []
    miss_ids: list[str] = []
    route_counts: dict[str, int] = {}
    route_hits: dict[str, int] = {}
    by_profile: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in ("subject", "task_kind", "answer_shape", "difficulty", "language")
    }
    elapsed_ms = 0
    sidecar = _reference_sidecar(answer_path)
    covered_references = 0

    for item in _records(input_path):
        idx = str(item.get("idx", ""))
        output_path = output_dir / f"{idx}.json"
        if not output_path.is_file():
            missing_outputs.append(idx)
            continue
        output = json.loads(output_path.read_text(encoding="utf-8"))
        answer = str(output.get("final_response", "")).strip()
        reference = str(
            item.get(
                "answer",
                item.get(
                    "reference_answer",
                    item.get("ground_truth", sidecar.get(idx, "")),
                ),
            )
        ).strip()
        if not _usable_reference(reference):
            reference = ""
            unavailable_references.append(idx)
        covered_references += int(bool(reference))
        spec = build_problem_spec(item["problem"])
        assessment = assess_candidate(answer, "offline", spec, ())
        hit = bool(reference and equivalent_answers(answer, reference))
        processed += 1
        hits += int(hit)
        invalid += int(bool(Finalizer.validate_structure(answer)))
        missing_obligations += int(not assessment.complete_goals)
        if reference:
            (hit_ids if hit else miss_ids).append(idx)

        selection = _selection(output)
        route = str(selection.get("route", "missing"))
        if reference:
            route_counts[route] = route_counts.get(route, 0) + 1
            route_hits[route] = route_hits.get(route, 0) + int(hit)

        profile_values = {
            "subject": str(getattr(spec.profile, "primary_subject", spec.profile.subject)),
            "task_kind": str(getattr(spec.profile, "task_kind", spec.profile.problem_type)),
            "answer_shape": str(spec.profile.answer_shape),
            "difficulty": str(spec.profile.difficulty),
            "language": str(spec.profile.language),
        }
        if reference:
            for dimension, value in profile_values.items():
                bucket = by_profile[dimension].setdefault(value, {"count": 0, "hits": 0})
                bucket["count"] += 1
                bucket["hits"] += int(hit)

        for trace_item in output.get("trace", []):
            if trace_item.get("step") != "model_call":
                continue
            content = trace_item.get("content", {})
            if not isinstance(content, dict):
                continue
            calls += 1
            truncated += int(bool(content.get("provider_truncated")))
            try:
                elapsed_ms += int(content.get("elapsed_ms", 0))
            except (TypeError, ValueError):
                pass

    return {
        "total": len(_records(input_path)),
        "processed": processed,
        "reference_coverage": covered_references,
        "unavailable_reference_count": len(unavailable_references),
        "unavailable_reference_ids": unavailable_references,
        "missing_output_count": len(missing_outputs),
        "missing_output_ids": missing_outputs,
        "hits": hits,
        "accuracy": hits / covered_references if covered_references else None,
        "hit_ids": hit_ids,
        "miss_ids": miss_ids,
        "structural_invalid": invalid,
        "missing_obligation_items": missing_obligations,
        "model_calls": calls,
        "provider_truncated_calls": truncated,
        "provider_truncation_rate": truncated / calls if calls else None,
        "model_elapsed_seconds": round(elapsed_ms / 1000, 3),
        "route_accuracy": {
            route: {
                "count": count,
                "hits": route_hits.get(route, 0),
                "accuracy": route_hits.get(route, 0) / count if count else None,
            }
            for route, count in sorted(route_counts.items())
        },
        "profile_accuracy": {
            dimension: {
                value: {
                    **values,
                    "accuracy": values["hits"] / values["count"] if values["count"] else None,
                }
                for value, values in sorted(buckets.items())
            }
            for dimension, buckets in by_profile.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--answer-jsonl",
        type=Path,
        help="Optional offline idx/answer sidecar; never passed to the Agent.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.input_jsonl, args.output_dir, args.answer_jsonl),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
