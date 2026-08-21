"""Audit only production-reachable deterministic tool routes offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.math_equivalence import equivalent_answers
from tools.tool_contract import GENERIC_PRESENTATION_REQUIREMENTS, problem_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--judge-output-dir", type=Path)
    sources.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    agent = SubmissionAgent(client=None)
    abstract_algebra = agent.abstract_algebra
    symbolic = agent.sympy
    textbook = agent.core_textbook
    complex_analysis = agent.complex_analysis
    differential_geometry = agent.differential_geometry
    numerical = agent.numerical_methods
    ode_pde = agent.ode_pde
    structures = agent.finite_structures
    parameterized_discrete = agent.parameterized_discrete
    stochastic = agent.stochastic_matrices
    probability_statistics = agent.probability_statistics
    measure_integrals = agent.measure_integrals
    total = hits = 0
    input_count = 0
    items_with_results = 0
    raw_result_count = 0
    certified_result_count = 0
    direct_result_count = 0
    goal_result_count = 0
    supported_result_count = 0
    rejection_reasons: dict[str, int] = {}
    raw_operations: dict[str, dict[str, int]] = {}
    operations: dict[str, dict[str, int]] = {}
    details: list[dict] = []
    raw_details: list[dict] = []
    route_details: list[dict] = []
    records = (
        _jsonl_records(args.input_jsonl)
        if args.input_jsonl is not None
        else _directory_records(args.judge_output_dir)
    )
    for record in records:
        problem = record.get("problem")
        if not isinstance(problem, str):
            continue
        input_count += 1
        spec = build_problem_spec(problem)
        statement = spec.problem_text or problem
        results = tuple((
            *agent._with_tool_assurance(
                abstract_algebra.results_for(statement), "symbolic"
            ),
            *agent._with_tool_assurance(symbolic.results_for(statement), "symbolic"),
            *agent._with_tool_assurance(textbook.results_for(statement), "symbolic"),
            *agent._with_tool_assurance(
                complex_analysis.results_for(statement), "symbolic"
            ),
            *agent._with_tool_assurance(
                differential_geometry.results_for(statement), "symbolic"
            ),
            *agent._with_tool_assurance(numerical.results_for(statement), "symbolic"),
            *agent._with_tool_assurance(ode_pde.results_for(statement), "symbolic"),
            *agent._with_tool_assurance(structures.results_for(statement), "exhaustive"),
            *agent._with_tool_assurance(
                parameterized_discrete.results_for(statement), "exhaustive"
            ),
            *agent._with_tool_assurance(stochastic.results_for(statement), "exhaustive"),
            *agent._with_tool_assurance(
                probability_statistics.results_for(statement), "symbolic"
            ),
            *agent._with_tool_assurance(
                measure_integrals.results_for(statement), "symbolic"
            ),
        ))
        if results:
            items_with_results += 1
        raw_result_count += len(results)
        certified_result_count += sum(item.verified for item in results)
        direct_result_count += sum(item.direct_submission_eligible for item in results)
        goal_result_count += sum(
            SubmissionAgent._certifies_goal_result(item, spec) for item in results
        )
        supported_result_count += sum(
            item.supported_submission_eligible for item in results
        )
        for item in results:
            bucket = raw_operations.setdefault(
                item.operation,
                {"count": 0, "certified": 0, "direct": 0, "goal": 0},
            )
            bucket["count"] += 1
            bucket["certified"] += int(item.verified)
            bucket["direct"] += int(item.direct_submission_eligible)
            bucket["goal"] += int(
                SubmissionAgent._certifies_goal_result(item, spec)
            )
            item_rejections = _rejection_reasons(
                item,
                spec,
                whole_allowed=(
                    spec.tool_can_answer_whole
                    or item.operation in {
                        "parameterized_factorial_ratio_valuation",
                        "parameterized_lattice_polygon_interior",
                        "parameterized_modular_power_sum",
                        "parameterized_permutation_cycle_inventory",
                        "parameterized_subtraction_game",
                    }
                ),
            )
            for reason in item_rejections:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if args.details:
                raw_details.append({
                    "idx": str(record.get("idx", "")),
                    "operation": item.operation,
                    "task_kind": spec.profile.task_kind,
                    "answer_shape": spec.profile.answer_shape,
                    "tool_can_answer_whole": spec.tool_can_answer_whole,
                    "strict_requirements": [
                        requirement.name
                        for goal in spec.goals
                        for requirement in goal.requirements
                        if requirement.strict
                    ],
                    "rejection_reasons": list(item_rejections),
                })
        whole = SubmissionAgent._whole_tool_result(results, spec)
        evidence = SubmissionAgent._tool_evidence(results, spec, whole)
        supported = agent._supported_tool_candidate(results, spec, evidence)
        if args.details and results:
            route_details.append({
                "idx": str(record.get("idx", "")),
                "whole_operation": whole.operation if whole is not None else "",
                "supported_operation": (
                    supported.method_id if supported is not None else ""
                ),
                "supported_tier": (
                    supported.validation_tier if supported is not None else ""
                ),
                "supported_tool_status": (
                    supported.tool_status if supported is not None else ""
                ),
            })
        answer = whole.result if whole is not None else (
            supported.answer if supported is not None else ""
        )
        if not answer:
            continue
        operation = whole.operation if whole is not None else supported.method_id
        # The reference is read only after the production tools have returned;
        # it is never passed to ProblemSpec, SubmissionAgent, or a tool.
        reference = _reference(record)
        hit = equivalent_answers(answer, reference) if reference else False
        total += 1
        hits += int(hit)
        bucket = operations.setdefault(operation, {"count": 0, "hits": 0})
        bucket["count"] += 1
        bucket["hits"] += int(hit)
        if args.details:
            details.append({
                "idx": str(record.get("idx", "")),
                "operation": operation,
                "semantic_hit": hit,
                "answer": answer,
                "reference": reference,
            })
    output = {
        "input_count": input_count,
        "items_with_results": items_with_results,
        "raw_result_count": raw_result_count,
        "certified_result_count": certified_result_count,
        "direct_result_count": direct_result_count,
        "goal_result_count": goal_result_count,
        "supported_result_count": supported_result_count,
        "route_count": total,
        "semantic_hits": hits,
        "semantic_accuracy": hits / total if total else None,
        "operations": {
            name: {
                **values,
                "accuracy": values["hits"] / values["count"],
            }
            for name, values in sorted(operations.items())
        },
        "raw_operations": dict(sorted(raw_operations.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    if args.details:
        output["details"] = details
        output["raw_details"] = raw_details
        output["route_details"] = route_details
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _rejection_reasons(result, spec, *, whole_allowed: bool) -> tuple[str, ...]:
    """Explain why a produced certificate cannot answer the whole problem."""
    reasons: list[str] = []
    contract = result.contract
    if not result.verified:
        reasons.append("certificate_failed")
    if contract is None:
        reasons.append("missing_contract")
        return tuple(reasons)
    if not result.direct_submission_eligible:
        reasons.append("not_direct_submission_capable")
    if result.certificate.source_fingerprint != problem_fingerprint(spec.problem_text):
        reasons.append("source_fingerprint_mismatch")
    if not whole_allowed:
        reasons.append("problem_spec_disallows_whole_tool")
    if len(spec.goals) != 1:
        reasons.append("goal_count")
    strict = {
        requirement.name
        for goal in spec.goals
        for requirement in goal.requirements
        if requirement.strict
    }
    allowed = set(contract.allowed_requirements) | set(GENERIC_PRESENTATION_REQUIREMENTS)
    if not set(contract.required_requirements) <= strict:
        reasons.append("missing_contract_required_requirement")
    if not strict <= allowed:
        reasons.append("unsupported_strict_requirement")
    if spec.profile.task_kind not in contract.allowed_task_kinds:
        reasons.append("task_kind")
    if (
        contract.allowed_answer_shapes
        and spec.profile.answer_shape not in contract.allowed_answer_shapes
    ):
        reasons.append("answer_shape")
    if any(
        requirement.strict and requirement.category == "support"
        for goal in spec.goals
        for requirement in goal.requirements
    ) and not result.supported_submission_eligible:
        reasons.append("support_required_but_unavailable")
    if SubmissionAgent._certifies_goal_result(result, spec):
        reasons.append("goal_certified")
    return tuple(dict.fromkeys(reasons))


def _path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def _directory_records(directory: Path):
    for path in sorted(directory.glob("*.json"), key=_path_key):
        yield json.loads(path.read_text(encoding="utf-8"))


def _jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _reference(record: dict) -> str:
    reward = record.get("reward_model")
    if isinstance(reward, dict):
        return str(reward.get("ground_truth", "") or "")
    return str(record.get("answer", "") or "")


if __name__ == "__main__":
    raise SystemExit(main())
