"""Audit the sealed V2 benchmark without executing the reasoning agent.

The audit checks lexical, number-abstracted, formula-skeleton, and declared
method isolation against every earlier local sample, including V1.  It also
requires a three-layer difficulty certificate and rejects any current exact
whole-goal route.  Reference answers remain offline-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from scripts.audit_unseen_hard_set import (
    existing_problems,
    hardness_signals,
    load_jsonl,
    normalized_problem,
    semantic_body,
    similarity,
)
from tools.sympy_tool import SympyTool


DATASET = ROOT / "sample_data" / "unseen_hard_holdout_v2.jsonl"
ALLOWED_OBLIGATIONS = {"classification", "construction", "proof", "complex_count"}
FORBIDDEN_METHOD_TERMS = {
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
    "bounded_digit_set",
    "prime_floor_quotient",
    "n_good_function",
    "colored_cube_slices",
    "tiling_invariant_profile",
    "finite_game_minimax",
    "splitting_field",
    "fourier_transform",
}
METHOD_FIELDS = ("domain", "object", "target", "constraints", "primary_method")
V1_METHOD_FINGERPRINTS = (
    ("v1_90001", ("enumerative_combinatorics", "two_regular_bipartite_graph", "matrix_count", "row_and_column_sum_two", "cycle_component_decomposition")),
    ("v1_90002", ("enumerative_graph_theory", "connected_unicyclic_labeled_graph", "graph_count", "eight_vertices_and_eight_edges", "unique_cycle_plus_rooted_forests")),
    ("v1_90003", ("diophantine_number_theory", "positive_integer_quadratic_pairs", "complete_solution_family", "symmetric_binary_quadratic_equation", "vieta_jumping_descent")),
    ("v1_90004", ("computational_number_theory", "bounded_divisibility_search", "complete_integer_set", "n_divides_two_power_n_plus_one", "modular_exponent_enumeration")),
    ("v1_90005", ("algebraic_number_theory", "sum_of_three_square_roots", "minimal_polynomial", "multiquadratic_degree_eight", "sign_conjugate_elimination")),
    ("v1_90006", ("ring_theory", "nonunital_ring_homomorphism", "homomorphism_count", "cyclic_quotient_rings", "crt_idempotent_classification")),
    ("v1_90007", ("linear_algebra", "nilpotent_jordan_operator", "centralizer_dimension", "block_sizes_four_two_two_one", "sum_of_block_intertwiner_dimensions")),
    ("v1_90008", ("discrete_probability", "competing_coin_patterns", "first_hit_probability", "overlapping_length_four_words", "joint_prefix_automaton")),
    ("v1_90009", ("urn_probability", "polya_reinforcement", "equal_final_color_counts_probability", "two_red_three_blue_seven_draws", "beta_binomial_distribution")),
    ("v1_90010", ("triangle_geometry", "interior_point_from_side_distances", "vertex_distance", "weighted_distances_equal", "barycentric_centroid_identification")),
    ("v1_90011", ("conic_geometry", "tangents_from_external_point_to_ellipse", "contact_triangle_area", "axis_aligned_ellipse", "polar_chord_and_discriminant")),
    ("v1_90012", ("real_analysis", "improper_logarithmic_integral", "exact_integral_constant", "square_log_over_sqrt_x_one_plus_x", "beta_parameter_second_derivative")),
    ("v1_90013", ("complex_analysis", "positive_real_part_holomorphic_function", "sharp_derivative_bound", "unit_disk_normalized_at_zero", "cayley_transform_and_schwarz_lemma")),
    ("v1_90014", ("real_analysis", "scaled_rational_function_sequence", "lp_convergence_exponent_set", "half_line_with_endpoint_cases", "scaled_norm_and_tail_analysis")),
)
SUBJECT_BUCKETS = {
    "Combinatorics": "combinatorics",
    "组合数学": "combinatorics",
    "Number Theory": "number_theory",
    "数论": "number_theory",
    "Linear Algebra": "algebra",
    "抽象代数": "algebra",
    "Geometry": "geometry",
    "射影几何": "geometry",
    "Probability": "probability",
    "概率论": "probability",
    "Complex Analysis": "analysis",
    "泛函分析": "analysis",
    "Algebraic Topology": "topology",
    "数值分析": "numerical_analysis",
}


@dataclass(frozen=True)
class Match:
    path: str
    idx: str
    score: float


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
    value = semantic_body(problem).lower()
    value = re.sub(r"(?<![a-z])\d+(?:\.\d+)?(?![a-z])", " N ", value)
    value = re.sub(r"(?<![a-z\\])[a-z](?:_\{?[a-z0-9]+\}?)?(?![a-z])", " V ", value)
    value = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", value)
    return _compact(value)


def formula_skeletons(problem: str) -> tuple[str, ...]:
    formulas = re.findall(r"\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", semantic_body(problem), re.DOTALL)
    skeletons: list[str] = []
    for groups in formulas:
        formula = next((group for group in groups if group), "")
        formula = re.sub(r"\\operatorname\s*\{[^{}]+\}", r"\\operatorname{OP}", formula)
        formula = re.sub(r"\d+(?:\.\d+)?", "N", formula)
        formula = re.sub(r"(?<![A-Za-z\\])[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?![A-Za-z])", "V", formula)
        formula = re.sub(r"\s+", "", formula)
        if len(formula) >= 12 and len(re.findall(r"[=<>+\-*/^]", formula)) >= 1:
            skeletons.append(formula)
    return tuple(skeletons)


def _closest(
    problem: str,
    corpus: list[tuple[Path, str, str]],
    transform,
) -> Match:
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


def _method_errors(row: dict, seen_methods: set[str]) -> list[str]:
    errors: list[str] = []
    fingerprint = row.get("method_fingerprint")
    if not isinstance(fingerprint, dict):
        return ["method_fingerprint must be an object"]
    missing = [field for field in METHOD_FIELDS if not str(fingerprint.get(field, "")).strip()]
    if missing:
        errors.append("missing method fingerprint fields: " + ", ".join(missing))
        return errors
    primary = str(fingerprint["primary_method"]).strip().lower()
    if primary in seen_methods:
        errors.append("duplicate primary_method inside V2")
    seen_methods.add(primary)
    joined = " ".join(str(fingerprint[field]).lower() for field in METHOD_FIELDS)
    forbidden = sorted(term for term in FORBIDDEN_METHOD_TERMS if term in joined)
    if forbidden:
        errors.append("forbidden V1/current-tool method family: " + ", ".join(forbidden))
    return errors


def _method_overlap(fingerprint: dict) -> tuple[str, int]:
    values = tuple(str(fingerprint.get(field, "")).replace("_", " ").lower() for field in METHOD_FIELDS)
    best_label, best_count = "", 0
    for label, reference in V1_METHOD_FINGERPRINTS:
        count = sum(
            _string_similarity(value, old.replace("_", " ")) >= 0.68
            for value, old in zip(values, reference)
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
        errors.append("difficulty obligation must be classification/construction/proof/complex_count")
    layers = [str(value).strip() for value in difficulty.get("reasoning_layers", []) if str(value).strip()]
    if len(layers) < 3 or len(set(layers)) < 3:
        errors.append("at least three distinct nontrivial reasoning layers are required")
    traps = [str(value).strip() for value in difficulty.get("common_traps", []) if str(value).strip()]
    if len(traps) < 2:
        errors.append("at least two answer-changing traps are required")
    checks = [str(value).strip() for value in difficulty.get("independent_checks", []) if str(value).strip()]
    if len(set(checks)) < 3:
        errors.append("at least three checks are required for executable V2 items")
    if len(hardness_signals(str(row.get("problem", "")))) < 3:
        errors.append("fewer than three independent textual hardness signals")
    return errors


def audit(
    path: Path = DATASET,
    *,
    lexical_threshold: float = 0.70,
    template_threshold: float = 0.76,
    formula_threshold: float = 0.90,
    check_tools: bool = True,
) -> dict:
    rows = load_jsonl(path)
    corpus = existing_problems(path)
    failures: list[dict] = []
    nearest: list[dict] = []
    seen_problems: set[str] = set()
    seen_methods: set[str] = set()
    languages = {"zh": 0, "en": 0}
    buckets: dict[str, int] = {}
    whole_tool_routes: list[str] = []
    sympy = SympyTool() if check_tools else None

    for row in rows:
        idx = str(row.get("idx", ""))
        problem = str(row.get("problem", ""))
        errors = _difficulty_errors(row)
        errors.extend(_method_errors(row, seen_methods))
        if not str(row.get("answer", "")).strip():
            errors.append("answer must be non-empty and offline-only")
        normalized = normalized_problem(problem)
        if normalized in seen_problems:
            errors.append("duplicate normalized problem inside V2")
        seen_problems.add(normalized)

        lexical = _closest(problem, corpus, normalized_problem)
        template = _closest(problem, corpus, abstract_template)
        formula = _closest_formula(problem, corpus)
        formula_context = formula.score * max(lexical.score, template.score)
        if lexical.score >= lexical_threshold:
            errors.append(f"lexical near-duplicate {lexical.score:.3f} against {lexical.path}#{lexical.idx}")
        if template.score >= template_threshold:
            errors.append(f"number/variable-template duplicate {template.score:.3f} against {template.path}#{template.idx}")
        if formula.score >= formula_threshold and max(lexical.score, template.score) >= 0.50:
            errors.append(f"formula-skeleton duplicate {formula.score:.3f} against {formula.path}#{formula.idx}")
        method_against, method_overlap = _method_overlap(row.get("method_fingerprint", {}))
        if method_overlap >= 4:
            errors.append(f"method fingerprint overlaps {method_overlap}/5 fields with {method_against}")

        spec = build_problem_spec(problem)
        if len(spec.goals) != 1:
            errors.append(f"expected one external goal, classifier found {len(spec.goals)}")
        if check_tools and sympy is not None:
            evidence = SubmissionAgent._tool_evidence(sympy.results_for(problem), spec)
            if SubmissionAgent._whole_tool_answer(evidence):
                whole_tool_routes.append(idx)
                errors.append("current deterministic tool can answer the whole goal")

        language = "zh" if re.search(r"[\u4e00-\u9fff]", semantic_body(problem)) else "en"
        languages[language] += 1
        subject = str(row.get("subject", ""))
        bucket = SUBJECT_BUCKETS.get(subject, "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if bucket == "unknown":
            errors.append(f"unmapped subject bucket: {subject}")

        if errors:
            failures.append({"idx": idx, "errors": errors})
        nearest.append({
            "idx": idx,
            "lexical": round(lexical.score, 4),
            "lexical_against": f"{lexical.path}#{lexical.idx}",
            "template": round(template.score, 4),
            "template_against": f"{template.path}#{template.idx}",
            "formula": round(formula.score, 4),
            "formula_against": f"{formula.path}#{formula.idx}",
            "formula_context": round(formula_context, 4),
            "method_overlap_fields": method_overlap,
            "method_against": method_against,
            "difficulty_score": row.get("difficulty", {}).get("score"),
            "reasoning_layers": len(row.get("difficulty", {}).get("reasoning_layers", [])),
        })

    global_errors: list[str] = []
    if len(rows) < 12:
        global_errors.append("V2 must contain at least 12 items")
    if languages["zh"] != languages["en"]:
        global_errors.append("Chinese and English quotas must be equal")
    if len([bucket for bucket in buckets if bucket != "unknown"]) < 7:
        global_errors.append("at least seven domain buckets are required")
    if max(buckets.values(), default=0) > 2:
        global_errors.append("no domain bucket may contain more than two items")

    return {
        "path": str(path),
        "rows": len(rows),
        "passed": not failures and not global_errors,
        "failure_count": len(failures),
        "failures": failures,
        "global_errors": global_errors,
        "thresholds": {
            "lexical": lexical_threshold,
            "number_variable_template": template_threshold,
            "formula_skeleton": formula_threshold,
            "formula_with_statement_context": 0.50,
            "method_fingerprint_fields": 4,
        },
        "maxima": {
            "lexical": max((item["lexical"] for item in nearest), default=0.0),
            "number_variable_template": max((item["template"] for item in nearest), default=0.0),
            "formula_skeleton": max((item["formula"] for item in nearest), default=0.0),
            "formula_with_statement_context": max((item["formula_context"] for item in nearest), default=0.0),
            "method_fingerprint_overlap_fields": max((item["method_overlap_fields"] for item in nearest), default=0),
        },
        "nearest": nearest,
        "languages": languages,
        "domain_buckets": buckets,
        "whole_tool_route_indices": whole_tool_routes,
        "agent_executed": False,
        "answer_exposed_to_runtime": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", nargs="?", type=Path, default=DATASET)
    parser.add_argument("--skip-tool-preflight", action="store_true")
    args = parser.parse_args()
    report = audit(args.input_file, check_tools=not args.skip_tool_preflight)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
