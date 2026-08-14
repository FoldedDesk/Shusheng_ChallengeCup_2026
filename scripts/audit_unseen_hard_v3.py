"""Isolation and contract audit for the sealed V3 unseen-hard benchmark.

The audit never calls ``ReasoningAgent.solve``.  It NFKC-normalizes every
statement, compares it with all other local JSONL problems at lexical,
number/variable-template, contextual-formula, and five-field method levels,
and rejects any problem currently answerable as a whole by SymPy/exact tools.
The dataset hash makes post-run tuning visible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from scripts.audit_unseen_hard_set import hardness_signals, load_jsonl, semantic_body
from scripts.audit_unseen_hard_v2 import (
    FORBIDDEN_METHOD_TERMS as EARLIER_FORBIDDEN_METHOD_TERMS,
    METHOD_FIELDS,
    V1_METHOD_FINGERPRINTS,
)
from tools.sympy_tool import SympyTool


DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v3.jsonl"
FROZEN_SHA256 = "8cba2a7a962b2f88ce43f9d2cd79a65c7d882e21c463cd3dbebd57daac92d66a"
EXPECTED_INDICES = set(range(92001, 92015))
ALLOWED_OBLIGATIONS = {"classification", "construction", "proof", "complex_count"}
FORBIDDEN_V2_FAMILIES = {
    "symbol_normalization_and_case_tree",
    "cycle_space_with_nonzero_hyperplane_exclusion",
    "cellular_boundary_and_smith_reduction",
    "affine_chebyshev_minimax_normalization",
}
FORBIDDEN_METHOD_TERMS = set(EARLIER_FORBIDDEN_METHOD_TERMS) | FORBIDDEN_V2_FAMILIES
SUBJECT_BUCKETS = {
    "Combinatorics": "combinatorics",
    "组合数学": "combinatorics",
    "Number Theory": "number_theory",
    "数论": "number_theory",
    "Abstract Algebra": "algebra",
    "微分几何": "geometry",
    "Hyperbolic Geometry": "geometry",
    "概率论": "probability",
    "Statistics": "statistics",
    "实分析": "analysis",
    "Complex Analysis": "analysis",
    "数值分析": "numerical_analysis",
    "Dynamical Systems": "dynamics_topology",
    "偏微分方程": "partial_differential_equations",
}
TOKEN = re.compile(r"[a-z]+|[\u4e00-\u9fff]|\\[a-z]+|\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Match:
    path: str
    idx: str
    score: float


def nfkc_body(problem: str) -> str:
    """Return the semantic statement in Unicode compatibility-normalized form."""
    return unicodedata.normalize("NFKC", semantic_body(problem))


def normalized_problem(problem: str) -> str:
    return " ".join(TOKEN.findall(nfkc_body(problem).lower()))


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _ngrams(value: str, size: int = 4) -> set[str]:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def _string_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right, autojunk=False).ratio()
    left_grams, right_grams = _ngrams(left), _ngrams(right)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(sequence, jaccard)


def abstract_template(problem: str) -> str:
    value = nfkc_body(problem).lower()
    value = re.sub(r"(?<![a-z])\d+(?:\.\d+)?(?![a-z])", " N ", value)
    value = re.sub(r"(?<![a-z\\])[a-z](?:_\{?[a-z0-9]+\}?)?(?![a-z])", " V ", value)
    value = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", value)
    return _compact(value)


def formula_skeletons(problem: str) -> tuple[str, ...]:
    formulas = re.findall(r"\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", nfkc_body(problem), re.DOTALL)
    skeletons: list[str] = []
    for groups in formulas:
        formula = next((group for group in groups if group), "")
        formula = re.sub(r"\\operatorname\s*\{[^{}]+\}", r"\\operatorname{OP}", formula)
        formula = re.sub(r"\d+(?:\.\d+)?", "N", formula)
        formula = re.sub(r"(?<![A-Za-z\\])[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?![A-Za-z])", "V", formula)
        formula = re.sub(r"\s+", "", formula)
        if len(formula) >= 10 and re.search(r"[=<>+\-*/^]|\\(?:sum|int|prod|to|equiv|begin)", formula):
            skeletons.append(formula)
    return tuple(skeletons)


def _local_corpus(candidate: Path) -> list[tuple[Path, str, str]]:
    corpus: list[tuple[Path, str, str]] = []
    for path in sorted((ROOT / "sample_data").glob("*.jsonl")):
        if path.resolve() == candidate.resolve():
            continue
        for row in load_jsonl(path):
            problem = row.get("problem")
            if isinstance(problem, str) and problem.strip():
                corpus.append((path, str(row.get("idx", "")), problem))
    return corpus


def _closest(problem: str, corpus: list[tuple[Path, str, str]], transform) -> Match:
    candidate = transform(problem)
    best = Match("", "", 0.0)
    for path, idx, other in corpus:
        score = _string_similarity(candidate, transform(other))
        if score > best.score:
            best = Match(str(path.relative_to(ROOT)), idx, score)
    return best


def _closest_formula(problem: str, corpus: list[tuple[Path, str, str]]) -> Match:
    candidate = formula_skeletons(problem)
    best = Match("", "", 0.0)
    for path, idx, other in corpus:
        old = formula_skeletons(other)
        score = max(
            (_string_similarity(left, right) for left in candidate for right in old),
            default=0.0,
        )
        if score > best.score:
            best = Match(str(path.relative_to(ROOT)), idx, score)
    return best


def _fingerprint_values(fingerprint: dict) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFKC", str(fingerprint.get(field, "")))
        .replace("_", " ")
        .lower()
        .strip()
        for field in METHOD_FIELDS
    )


def _prior_method_fingerprints(candidate: Path) -> list[tuple[str, tuple[str, ...]]]:
    references = [
        (label, tuple(value.replace("_", " ") for value in values))
        for label, values in V1_METHOD_FINGERPRINTS
    ]
    known_labels = {label for label, _ in references}
    for path in sorted((ROOT / "sample_data").glob("*.jsonl")):
        if path.resolve() == candidate.resolve():
            continue
        for row in load_jsonl(path):
            fingerprint = row.get("method_fingerprint")
            if not isinstance(fingerprint, dict):
                continue
            label = f"{path.name}#{row.get('idx', '')}"
            if label not in known_labels:
                references.append((label, _fingerprint_values(fingerprint)))
                known_labels.add(label)
    return references


def _method_overlap(
    fingerprint: dict,
    references: list[tuple[str, tuple[str, ...]]],
) -> tuple[str, int]:
    values = _fingerprint_values(fingerprint)
    best_label, best_count = "", 0
    for label, old_values in references:
        count = sum(
            _string_similarity(value, old) >= 0.68
            for value, old in zip(values, old_values)
        )
        if count > best_count:
            best_label, best_count = label, count
    return best_label, best_count


def _difficulty_errors(row: dict) -> list[str]:
    difficulty = row.get("difficulty")
    if not isinstance(difficulty, dict):
        return ["difficulty must be an object"]
    errors: list[str] = []
    if difficulty.get("level") != "hard" or difficulty.get("score") not in {3, 4}:
        errors.append("difficulty must be hard with score 3 or 4")
    if difficulty.get("obligation") not in ALLOWED_OBLIGATIONS:
        errors.append("invalid high-difficulty obligation")
    layers = {str(value).strip() for value in difficulty.get("reasoning_layers", []) if str(value).strip()}
    traps = {str(value).strip() for value in difficulty.get("common_traps", []) if str(value).strip()}
    checks = {str(value).strip() for value in difficulty.get("independent_checks", []) if str(value).strip()}
    if len(layers) < 3:
        errors.append("at least three distinct dependent reasoning layers are required")
    if len(traps) < 2:
        errors.append("at least two answer-changing traps are required")
    if len(checks) < 3:
        errors.append("at least three independent checks are required")
    if len(hardness_signals(str(row.get("problem", "")))) < 3:
        errors.append("fewer than three independent textual hardness signals")
    return errors


def _method_errors(row: dict, seen_primary: set[str]) -> list[str]:
    fingerprint = row.get("method_fingerprint")
    if not isinstance(fingerprint, dict):
        return ["method_fingerprint must be an object"]
    missing = [field for field in METHOD_FIELDS if not str(fingerprint.get(field, "")).strip()]
    if missing:
        return ["missing method fingerprint fields: " + ", ".join(missing)]
    primary = str(fingerprint["primary_method"]).strip().lower()
    errors: list[str] = []
    if primary in seen_primary:
        errors.append("duplicate primary_method inside V3")
    seen_primary.add(primary)
    joined = " ".join(str(fingerprint[field]).lower() for field in METHOD_FIELDS)
    forbidden = sorted(term for term in FORBIDDEN_METHOD_TERMS if term in joined)
    if forbidden:
        errors.append("forbidden V1/V2/current exact method family: " + ", ".join(forbidden))
    return errors


def audit(
    path: Path = DATASET,
    *,
    lexical_threshold: float = 0.70,
    template_threshold: float = 0.76,
    contextual_formula_threshold: float = 0.50,
    check_tools: bool = True,
    enforce_frozen_hash: bool | None = None,
) -> dict:
    path = path.resolve()
    rows = load_jsonl(path)
    corpus = _local_corpus(path)
    prior_methods = _prior_method_fingerprints(path)
    failures: list[dict] = []
    nearest: list[dict] = []
    seen_problems: set[str] = set()
    seen_primary: set[str] = set()
    candidate_methods: list[tuple[str, tuple[str, ...]]] = []
    languages = {"zh": 0, "en": 0}
    buckets: dict[str, int] = {}
    whole_tool_routes: list[str] = []
    sympy = SympyTool() if check_tools else None

    for row in rows:
        idx = str(row.get("idx", ""))
        problem = str(row.get("problem", ""))
        errors = _difficulty_errors(row)
        errors.extend(_method_errors(row, seen_primary))
        if row.get("source") != "unseen_hard_holdout_v3":
            errors.append("source must identify the V3 sealed holdout")
        if not str(row.get("answer", "")).strip():
            errors.append("offline answer must be non-empty")

        normalized = normalized_problem(problem)
        if not normalized or normalized in seen_problems:
            errors.append("empty or duplicate NFKC-normalized problem inside V3")
        seen_problems.add(normalized)

        lexical = _closest(problem, corpus, normalized_problem)
        template = _closest(problem, corpus, abstract_template)
        raw_formula = _closest_formula(problem, corpus)
        formula_context = raw_formula.score * max(lexical.score, template.score)
        if lexical.score >= lexical_threshold:
            errors.append(f"lexical duplicate {lexical.score:.3f} against {lexical.path}#{lexical.idx}")
        if template.score >= template_threshold:
            errors.append(f"abstract-template duplicate {template.score:.3f} against {template.path}#{template.idx}")
        if formula_context >= contextual_formula_threshold:
            errors.append(
                f"formula-plus-context duplicate {formula_context:.3f} "
                f"against {raw_formula.path}#{raw_formula.idx}"
            )

        fingerprint = row.get("method_fingerprint", {})
        method_against, method_overlap = _method_overlap(fingerprint, prior_methods)
        internal_against, internal_overlap = _method_overlap(fingerprint, candidate_methods)
        if method_overlap > 1:
            errors.append(f"method fingerprint overlaps {method_overlap}/5 with {method_against}")
        if internal_overlap > 1:
            errors.append(f"internal method fingerprint overlaps {internal_overlap}/5 with {internal_against}")
        candidate_methods.append((f"v3#{idx}", _fingerprint_values(fingerprint)))

        spec = build_problem_spec(problem)
        if len(spec.goals) != 1:
            errors.append(f"expected one external goal, classifier found {len(spec.goals)}")
        if check_tools and sympy is not None:
            results = sympy.results_for(problem)
            evidence = SubmissionAgent._tool_evidence(results, spec)
            if SubmissionAgent._whole_tool_answer(evidence):
                whole_tool_routes.append(idx)
                errors.append("current SymPy/exact tools can answer the whole goal")

        language = "zh" if re.search(r"[\u4e00-\u9fff]", nfkc_body(problem)) else "en"
        languages[language] += 1
        bucket = SUBJECT_BUCKETS.get(str(row.get("subject", "")), "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if bucket == "unknown":
            errors.append(f"unmapped subject: {row.get('subject', '')}")

        if errors:
            failures.append({"idx": idx, "errors": errors})
        nearest.append({
            "idx": idx,
            "lexical": round(lexical.score, 4),
            "lexical_against": f"{lexical.path}#{lexical.idx}",
            "template": round(template.score, 4),
            "template_against": f"{template.path}#{template.idx}",
            "formula_plus_context": round(formula_context, 4),
            "formula_against": f"{raw_formula.path}#{raw_formula.idx}",
            "raw_formula_skeleton": round(raw_formula.score, 4),
            "method_overlap_fields": method_overlap,
            "method_against": method_against,
            "internal_method_overlap_fields": internal_overlap,
            "difficulty_score": row.get("difficulty", {}).get("score"),
            "reasoning_layers": len(row.get("difficulty", {}).get("reasoning_layers", [])),
        })

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if enforce_frozen_hash is None:
        enforce_frozen_hash = path == DATASET.resolve()
    global_errors: list[str] = []
    indices = {int(row.get("idx", -1)) for row in rows}
    if len(rows) != 14 or indices != EXPECTED_INDICES:
        global_errors.append("V3 must contain exactly indices 92001 through 92014")
    if languages != {"zh": 7, "en": 7}:
        global_errors.append("V3 must contain exactly seven Chinese and seven English items")
    known_buckets = {bucket for bucket in buckets if bucket != "unknown"}
    if len(known_buckets) < 8:
        global_errors.append("at least eight domain buckets are required")
    if max((count for bucket, count in buckets.items() if bucket != "unknown"), default=0) > 2:
        global_errors.append("no domain bucket may contain more than two items")
    if whole_tool_routes:
        global_errors.append("SymPy/exact whole-goal route count must be zero")
    if enforce_frozen_hash and digest != FROZEN_SHA256:
        global_errors.append(f"frozen SHA-256 mismatch: expected {FROZEN_SHA256}, got {digest}")

    return {
        "path": str(path),
        "rows": len(rows),
        "passed": not failures and not global_errors,
        "failure_count": len(failures),
        "failures": failures,
        "global_errors": global_errors,
        "thresholds": {
            "nfkc_lexical": lexical_threshold,
            "number_variable_template": template_threshold,
            "formula_plus_context": contextual_formula_threshold,
            "method_fingerprint_max_overlap_fields": 1,
        },
        "maxima": {
            "nfkc_lexical": max((item["lexical"] for item in nearest), default=0.0),
            "number_variable_template": max((item["template"] for item in nearest), default=0.0),
            "formula_plus_context": max((item["formula_plus_context"] for item in nearest), default=0.0),
            "raw_formula_skeleton_diagnostic": max((item["raw_formula_skeleton"] for item in nearest), default=0.0),
            "prior_method_overlap_fields": max((item["method_overlap_fields"] for item in nearest), default=0),
            "internal_method_overlap_fields": max((item["internal_method_overlap_fields"] for item in nearest), default=0),
        },
        "nearest": nearest,
        "languages": languages,
        "domain_buckets": buckets,
        "whole_tool_route_indices": whole_tool_routes,
        "sympy_exact_whole_goal_count": len(whole_tool_routes),
        "dataset_sha256": digest,
        "expected_sha256": FROZEN_SHA256,
        "hash_matches": digest == FROZEN_SHA256,
        "agent_executed": False,
        "answer_exposed_to_runtime": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", nargs="?", type=Path, default=DATASET)
    parser.add_argument("--skip-tool-preflight", action="store_true")
    parser.add_argument("--skip-frozen-hash", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.input_file,
        check_tools=not args.skip_tool_preflight,
        enforce_frozen_hash=False if args.skip_frozen_hash else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
