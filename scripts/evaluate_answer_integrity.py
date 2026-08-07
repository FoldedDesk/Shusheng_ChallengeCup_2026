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
from reasoning.finalizer import Finalizer


def _normalized(value: str) -> str:
    text = normalize_latex(str(value or "")).lower().replace("−", "-")
    return re.sub(r"[\s{}\\,，。；;：:]", "", text)


def _trace_validation(record: dict) -> dict:
    return next((item.get("content", {}) for item in record.get("trace", []) if item.get("step") == "validation"), {})


def evaluate(input_file: Path, output_dir: Path) -> dict:
    expected = {}
    for line in input_file.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        expected[str(item["idx"])] = str(item.get("answer", ""))

    invalid = fallback = uncertain = tool_conflicts = hits = processed = 0
    selected_raw_meta = final_meta = malformed = explicit_selected = 0
    sources: dict[str, int] = {}
    for idx, answer in expected.items():
        path = output_dir / f"{idx}.json"
        if not path.is_file():
            continue
        processed += 1
        record = json.loads(path.read_text(encoding="utf-8"))
        validation = _trace_validation(record)
        candidates = list(validation.values()) if isinstance(validation, dict) else []
        rejected = [reason for candidate in candidates for reason in candidate.get("rejected_reasons", [])]
        invalid += int(bool(Finalizer.validate_structure(record.get("final_response", ""))))
        selection = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "selection"), {})
        source = str(selection.get("source", "missing"))
        sources[source] = sources.get(source, 0) + 1
        fallback += int(source == "fallback")
        selected = validation.get(source, {}) if isinstance(validation, dict) else {}
        uncertain += int(bool(selected.get("coverage_uncertain")))
        tool_conflicts += int(any(reason == "tool_conflict" for reason in rejected))
        diagnostics = next((step.get("content", {}) for step in record.get("trace", []) if step.get("step") == "candidate_diagnostics"), {})
        selected_raw_meta += int(bool(diagnostics.get(source, {}).get("raw_has_meta")))
        explicit_selected += int(bool(selected.get("explicit_answer")))
        final_reasons = Finalizer.validate_structure(record.get("final_response", ""))
        final_meta += int("meta_text" in final_reasons or Finalizer.contains_meta(record.get("final_response", "")))
        malformed += int(bool(set(final_reasons) & {"placeholder", "meaningless_fragment", "markup_fragment"}))
        actual = _normalized(record.get("final_response", ""))
        target = _normalized(answer)
        hits += int(bool(actual and target and (actual in target or target in actual)))
    return {
        "processed": processed,
        "structural_invalid_answers": invalid,
        "fallback_count": fallback,
        "coverage_uncertain_count": uncertain,
        "tool_conflict_count": tool_conflicts,
        "selected_candidate_contains_meta": selected_raw_meta,
        "final_response_contains_meta": final_meta,
        "malformed_final_response_count": malformed,
        "explicit_marker_selection_count": explicit_selected,
        "selection_sources": sources,
        "normalized_standard_answer_hits": hits,
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
