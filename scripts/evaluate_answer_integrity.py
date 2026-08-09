"""Offline-only summary for a completed regression run.

The sample ``answer`` field is read here for local statistics only; it is never
passed to the submission agent or included in model prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latex_parser import normalize_latex
from classifier.problem_spec import build_problem_spec
from reasoning.candidate_selector import assess_candidate
from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers


def _normalized(value: str) -> str:
    text = normalize_latex(str(value or "")).lower().replace("−", "-")
    return re.sub(r"[\s{}\\,，。；;：:]", "", text)


def _trace_validation(record: dict) -> dict:
    return next((item.get("content", {}) for item in record.get("trace", []) if item.get("step") == "validation"), {})


def evaluate(input_file: Path, output_dir: Path) -> dict:
    expected = {}
    for line in input_file.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        expected[str(item["idx"])] = {
            "answer": str(item.get("answer", "")),
            "problem": str(item.get("problem", "")),
        }

    invalid = fallback = uncertain = tool_conflicts = hits = semantic_hits = processed = 0
    selected_raw_meta = final_meta = malformed = explicit_selected = 0
    model_calls = rescue_admitted = near_budget = all_empty = finalization_failures = 0
    tail_segment_selected = 0
    verification_admitted = conflicts = arbitration_used = corrected_selected = 0
    missing_required_goals = offline_uncertain = 0
    sources: dict[str, int] = {}
    for idx, item in expected.items():
        path = output_dir / f"{idx}.json"
        if not path.is_file():
            continue
        processed += 1
        record = json.loads(path.read_text(encoding="utf-8"))
        final_response = record.get("final_response", "")
        offline_assessment = assess_candidate(
            final_response,
            "offline",
            build_problem_spec(item["problem"]),
            (),
        )
        missing_required_goals += int("missing_required_goal" in offline_assessment.rejected_reasons)
        offline_uncertain += int(offline_assessment.coverage_uncertain)
        validation = _trace_validation(record)
        candidates = list(validation.values()) if isinstance(validation, dict) else []
        rejected = [reason for candidate in candidates for reason in candidate.get("rejected_reasons", [])]
        invalid += int(bool(Finalizer.validate_structure(final_response)))
        selection = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "selection"), {})
        source = str(selection.get("source", "missing"))
        sources[source] = sources.get(source, 0) + 1
        fallback += int(source == "fallback")
        selected = validation.get(source, {}) if isinstance(validation, dict) else {}
        uncertain += int(bool(selected.get("coverage_uncertain")))
        tool_conflicts += int(any(reason == "tool_conflict" for reason in rejected))
        diagnostics = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "candidate_diagnostics"), {})
        calls = [step for step in record.get("trace", []) if str(step.get("step", "")).startswith("model_call_")]
        model_calls += len(calls)
        near_budget += sum(bool(step.get("content", {}).get("response_near_budget")) for step in calls)
        admission = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "review_admission"), {})
        rescue_admitted += int(bool(admission.get("admitted")))
        verification_admitted += int(admission.get("mode") == "verify")
        equivalence = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "equivalence"), {})
        conflicts += int(bool(equivalence.get("conflict")))
        arbitration_used += int(bool(equivalence.get("arbitration_used")))
        non_empty_stages = sum(bool(stage.get("non_empty")) for stage in diagnostics.values())
        all_empty += int(source == "fallback" and non_empty_stages == 0)
        finalization_failures += int(source == "fallback" and non_empty_stages > 0)
        selected_raw_meta += int(bool(diagnostics.get(source, {}).get("raw_has_meta")))
        explicit_selected += int(bool(selected.get("explicit_answer")))
        tail_segment_selected += int(selected.get("extraction_method") == "tail_segment")
        final_reasons = Finalizer.validate_structure(final_response)
        final_meta += int("meta_text" in final_reasons or Finalizer.contains_meta(final_response))
        malformed += int(bool(set(final_reasons) & {"placeholder", "meaningless_fragment", "markup_fragment"}))
        actual = _normalized(final_response)
        target = _normalized(item["answer"])
        hits += int(bool(actual and target and (actual in target or target in actual)))
        semantic_hits += int(equivalent_answers(final_response, item["answer"]))
        corrected_selected += int(selected.get("verification_verdict") == "corrected")
    return {
        "processed": processed,
        "structural_invalid_answers": invalid,
        "fallback_count": fallback,
        "coverage_uncertain_count": uncertain,
        "offline_coverage_uncertain_count": offline_uncertain,
        "missing_required_goal_count": missing_required_goals,
        "tool_conflict_count": tool_conflicts,
        "selected_candidate_contains_meta": selected_raw_meta,
        "final_response_contains_meta": final_meta,
        "malformed_final_response_count": malformed,
        "explicit_marker_selection_count": explicit_selected,
        "tail_segment_selection_count": tail_segment_selected,
        "model_call_count": model_calls,
        "rescue_admission_count": rescue_admitted,
        "verification_admission_count": verification_admitted,
        "candidate_conflict_count": conflicts,
        "arbitration_used_count": arbitration_used,
        "corrected_candidate_selection_count": corrected_selected,
        "response_near_budget_call_count": near_budget,
        "all_model_responses_empty_count": all_empty,
        "nonempty_finalization_failure_count": finalization_failures,
        "selection_sources": sources,
        "normalized_standard_answer_hits": hits,
        "semantic_standard_answer_hits": semantic_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize answer-integrity regression outputs.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input_file, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
