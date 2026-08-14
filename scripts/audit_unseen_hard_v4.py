"""Isolation and contract audit for the sealed V4 unseen-hard benchmark.

This authoring-only audit never calls ``ReasoningAgent.solve``.  It compares
NFKC-normalized statements, abstract number/variable templates, contextual
formula skeletons, and five-field method fingerprints against every other
local JSONL.  It also rejects retired benchmark families and any current
SymPy/exact-tool route that can answer a whole external goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from scripts.audit_unseen_hard_set import hardness_signals, load_jsonl
from scripts.audit_unseen_hard_v2 import METHOD_FIELDS
from scripts.audit_unseen_hard_v3 import (
    _closest,
    _closest_formula,
    _fingerprint_values,
    _local_corpus,
    _method_overlap,
    _prior_method_fingerprints,
    abstract_template,
    nfkc_body,
    normalized_problem,
)
from tools.sympy_tool import SympyTool


DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v4.jsonl"
FROZEN_SHA256 = "8156fe7ee4c8180cb0989623866cead30a470c336788b68481fccf7f36a9ea8a"
EXPECTED_INDICES = set(range(93001, 93015))
ALLOWED_OBLIGATIONS = {"classification", "construction", "proof", "complex_count"}
SUBJECT_BUCKETS = {
    "Representation Theory": "algebra",
    "交换代数": "algebra",
    "Analytic Number Theory": "number_theory",
    "代数数论": "number_theory",
    "球面几何": "geometry",
    "Convex Geometry": "geometry",
    "Stochastic Processes": "probability",
    "变分法": "analysis",
    "Functional Analysis": "analysis",
    "Information Theory": "information_theory",
    "动力系统": "dynamics",
    "统计决策": "statistics",
    "复分析": "complex_analysis",
    "Partial Differential Equations": "partial_differential_equations",
}

# These are family markers, not generic mathematical words.  Their explicit
# exclusion prevents a benchmark from disguising a retired task by changing
# only constants or language.
RETIRED_FAMILY_MARKERS = {
    "bipartite_2_regular_cycle_decomposition",
    "rooted_forest_unicyclic_count",
    "vieta_jumping",
    "bounded_modular_exponent_search",
    "multiquadratic_minimal_polynomial",
    "crt_idempotent_ring_homomorphism",
    "jordan_centralizer",
    "competing_pattern_prefix_automaton",
    "polya_beta_binomial",
    "barycentric_centroid",
    "ellipse_polar_chord",
    "beta_parameter_differentiation",
    "caratheodory_schwarz",
    "scaled_lp_endpoint_analysis",
    "symbol_normalization_and_case_tree",
    "cycle_space_with_nonzero_hyperplane_exclusion",
    "cellular_boundary_and_smith_reduction",
    "affine_chebyshev_minimax_normalization",
    "best_theorem_with_directed_matrix_tree",
    "lukasiewicz_words_and_cycle_lemma",
    "dense_root_of_unity_radial_blowup",
    "order_conditions_plus_rational_stability_analysis",
}
ACTIVE_DEVELOPMENT_FAMILY_PATTERNS = (
    re.compile(r"\b(?:euler circuits?|euler tours?)\b", re.IGNORECASE),
    re.compile(r"有向欧拉|欧拉回路|欧拉巡回"),
    re.compile(r"\b(?:lukasiewicz|ordered plane rooted tree|prescribed[- ]degree plane tree)\b", re.IGNORECASE),
    re.compile(r"有序平面根树|给定出度.*平面树"),
    re.compile(r"\b(?:lacunary series|natural boundary|fabry gap)\b", re.IGNORECASE),
    re.compile(r"自然边界|稀疏幂级数|法布里间隙"),
    re.compile(r"\b(?:runge[- ]kutta|sdirk|dirk|l-stability)\b", re.IGNORECASE),
    re.compile(r"龙格.*库塔|对角隐式.*稳定"),
    re.compile(r"\b(?:latin squares?|nowhere[- ]zero flows?)\b", re.IGNORECASE),
    re.compile(r"拉丁方|无处为零流"),
    re.compile(r"\b(?:cellular homology|chebyshev minimax)\b", re.IGNORECASE),
    re.compile(r"胞腔同调|切比雪夫.*极小极大"),
)


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
    errors: list[str] = []
    primary = str(fingerprint["primary_method"]).strip().lower()
    if primary in seen_primary:
        errors.append("duplicate primary_method inside V4")
    seen_primary.add(primary)
    joined = " ".join(str(fingerprint[field]).lower() for field in METHOD_FIELDS)
    retired = sorted(marker for marker in RETIRED_FAMILY_MARKERS if marker in joined)
    if retired:
        errors.append("retired benchmark method family: " + ", ".join(retired))
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
        if row.get("source") != "unseen_hard_holdout_v4":
            errors.append("source must identify the V4 sealed holdout")
        if not str(row.get("answer", "")).strip():
            errors.append("offline answer must be non-empty")
        if any(pattern.search(problem) for pattern in ACTIVE_DEVELOPMENT_FAMILY_PATTERNS):
            errors.append("problem reuses a retired or active-development question family")

        normalized = normalized_problem(problem)
        if not normalized or normalized in seen_problems:
            errors.append("empty or duplicate NFKC-normalized problem inside V4")
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
        candidate_methods.append((f"v4#{idx}", _fingerprint_values(fingerprint)))

        spec = build_problem_spec(problem)
        if len(spec.goals) != 1:
            errors.append(f"expected one external goal, classifier found {len(spec.goals)}")
        if check_tools and sympy is not None:
            evidence = SubmissionAgent._tool_evidence(sympy.results_for(problem), spec)
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
            "internal_method_against": internal_against,
            "difficulty_score": row.get("difficulty", {}).get("score"),
            "reasoning_layers": len(row.get("difficulty", {}).get("reasoning_layers", [])),
        })

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if enforce_frozen_hash is None:
        enforce_frozen_hash = path == DATASET.resolve()
    global_errors: list[str] = []
    indices = {int(row.get("idx", -1)) for row in rows}
    if len(rows) != 14 or indices != EXPECTED_INDICES:
        global_errors.append("V4 must contain exactly indices 93001 through 93014")
    if languages != {"zh": 7, "en": 7}:
        global_errors.append("V4 must contain exactly seven Chinese and seven English items")
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
