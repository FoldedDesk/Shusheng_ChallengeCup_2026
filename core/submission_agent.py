"""Accuracy-first, bounded orchestration for unseen mathematics problems."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import re
from time import monotonic
from typing import Iterable, Mapping

from classifier.choice import canonical_choice_answer
from classifier.problem_spec import ProblemSpec, build_problem_spec
from core.execution_limits import SUBMISSION_SOFT_ITEM_SECONDS
from core.model_response import ModelCallResult, coerce_model_response
from core.runtime_failure import is_recoverable_runtime_failure
from core.stage_budget import StageBudget, plan_stage_budget
from rag.card_retriever import CardRetriever, RetrievalBundle
from reasoning.candidate_selector import (
    CandidateAssessment,
    CheckResult,
    ToolEvidence,
    assess_candidate,
    candidate_consistency_reasons,
    choose_candidate,
)
from reasoning.finalizer import ExtractionResult, Finalizer
from reasoning.local_tool_opportunity import (
    LocalToolOpportunity,
    detect_local_tool_opportunity,
)
from reasoning.math_equivalence import equivalent_answers
from reasoning.obligation_graph import (
    MathematicalObligationGraph,
    planning_request as obligation_planning_request,
    planning_system_prompt as obligation_planning_system_prompt,
)
from reasoning.solve_plan import SolvePlan
from reasoning.subject_protocols import subject_protocol
from reasoning.truncation_state import classify_truncated_output
from tools.abstract_algebra_tool import AbstractAlgebraTool
from tools.core_textbook_tool import CoreTextbookTool
from tools.complex_analysis_tool import ComplexAnalysisTool
from tools.differential_geometry_tool import DifferentialGeometryTool
from tools.finite_structure_tool import FiniteStructureTool
from tools.measure_integral_tool import MeasureIntegralTool
from tools.model_math_tools import ModelMathTools
from tools.numerical_method_tool import NumericalMethodTool
from tools.ode_pde_tool import OdePdeTool
from tools.operation_locator import OperationLocator
from tools.parameterized_discrete_tool import ParameterizedDiscreteTool
from tools.probability_statistics_tool import ProbabilityStatisticsTool
from tools.stochastic_matrix_tool import StochasticMatrixTool
from tools.sympy_tool import SympyTool
from tools.structured_verification import StructuredVerificationTool
from tools.tool_contract import (
    GENERIC_PRESENTATION_REQUIREMENTS,
    ToolResult,
    problem_fingerprint,
)


_AUTO_EXACT_MODEL_TOOLS = frozenset({
    "factorial_ratio_prime_valuation",
    "finite_state_walk_count",
    "lattice_polygon_interior",
    "subtraction_game_outcome",
})

_PARAMETERIZED_STATEMENT_OPERATIONS = frozenset({
    "parameterized_factorial_ratio_valuation",
    "parameterized_lattice_polygon_interior",
    "parameterized_modular_power_sum",
    "parameterized_permutation_cycle_inventory",
    "parameterized_subtraction_game",
})


class SubmissionAgent:
    """Solve one statement without relying on metadata, order, or answer keys."""

    def __init__(self, client) -> None:
        self.client = client
        self.sympy = SympyTool()
        self.abstract_algebra = AbstractAlgebraTool()
        self.structured_verifier = StructuredVerificationTool(self.sympy)
        self.operation_locator = OperationLocator(self.sympy)
        self.core_textbook = CoreTextbookTool()
        self.complex_analysis = ComplexAnalysisTool()
        self.differential_geometry = DifferentialGeometryTool()
        self.numerical_methods = NumericalMethodTool()
        self.ode_pde = OdePdeTool()
        self.finite_structures = FiniteStructureTool()
        self.stochastic_matrices = StochasticMatrixTool()
        self.probability_statistics = ProbabilityStatisticsTool()
        self.measure_integrals = MeasureIntegralTool()
        self.model_math_tools = ModelMathTools(self.sympy)
        self.parameterized_discrete = ParameterizedDiscreteTool(self.model_math_tools)
        self.retriever = CardRetriever()
        self.prompt = self._load_prompt()
        # Kept behind a local A/B gate until a frozen replay demonstrates
        # that the extra method-search call improves accuracy rather than
        # merely adding plausible planning prose.
        self.enable_mog = False
        # Local production A/B gate.  Disabling this removes only the
        # candidate-visible whole-answer audit; it must not silently replace
        # that call with another generic review stage.
        # Frozen same-trajectory A/B: four audits changed no accepted
        # mathematical conclusion (net gain 0).  Keep the stage off in
        # production until a task-specific replay demonstrates positive gain.
        self.enable_candidate_audit = False
        # Frozen A/B: disabling blind second solves preserved 6/16 while
        # removing 18 calls and about 1073 model-seconds.  Necessary recovery
        # and the proof-only targeted audit remain available.
        self.enable_blind_consensus = False
        # Short choice/truth/factual questions have a different error profile
        # from deep derivations.  Keep their cheap blind-consensus experiment
        # independently gated so a calculation-only ablation cannot disable or
        # enable it by accident.
        self.enable_quick_consensus = False
        # Local replay diagnostics only.  Production traces intentionally omit
        # candidate text under PLATFORM_CONSTRAINTS.md.
        self.local_candidate_diagnostics = False
        # Development replays show a positive net accuracy change and lower
        # elapsed time from the compact obligation-first protocol. Local
        # runners may still disable it for matched ablations.
        self.compact_primary_prompt = True
        # Local ablation only.  The production path keeps MOG disabled until
        # a matched replay shows that hidden thinking improves route quality.
        self.mog_route_thinking = False
        self.mog_route_token_limits: tuple[int, int] | None = None
        # Local A/B gate. Complex semantic problems may use a very small set
        # of deterministic functions only as subproblem calculators. Their
        # results remain NOT_CERTIFIED for the original goal and can never be
        # submitted directly. Keep disabled until a frozen replay shows net
        # accuracy gain after accounting for the extra provider round.
        self.enable_complex_subproblem_tools = False

    def solve(self, problem: str, metadata: dict) -> dict:
        del metadata
        started_at = monotonic()
        statement = str(problem or "").strip()
        spec = build_problem_spec(statement)
        statement = spec.problem_text or statement
        # ProblemSpec removes only trailing presentation instructions such as
        # "put the answer in boxed".  Tools and certificates must use that
        # same canonical statement or their source fingerprints disagree.
        tool_statement = spec.problem_text or statement
        cards = self.retriever.retrieve(spec)
        raw_tool_results = tuple((
            *self._with_tool_assurance(
                self.abstract_algebra.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.sympy.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.core_textbook.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.complex_analysis.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.differential_geometry.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.numerical_methods.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.ode_pde.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.finite_structures.results_for(tool_statement), "exhaustive"
            ),
            *self._with_tool_assurance(
                self.parameterized_discrete.results_for(tool_statement), "exhaustive"
            ),
            *self._with_tool_assurance(
                self.stochastic_matrices.results_for(tool_statement), "exhaustive"
            ),
            *self._with_tool_assurance(
                self.probability_statistics.results_for(tool_statement), "symbolic"
            ),
            *self._with_tool_assurance(
                self.measure_integrals.results_for(tool_statement), "symbolic"
            ),
        ))
        whole_tool = self._whole_tool_result(raw_tool_results, spec)
        evidence = self._tool_evidence(raw_tool_results, spec, whole_tool)
        deep_reasoning = self._deep_reasoning(spec)
        hidden_thinking = self._hidden_thinking(spec)
        budget = plan_stage_budget(
            spec,
            has_whole_tool_answer=whole_tool is not None,
            deep_reasoning=deep_reasoning,
        )
        trace: list[dict] = [
            {"step": "blueprint", "content": self._blueprint_trace(spec)},
            {"step": "retrieval", "content": cards.trace_content()},
            {"step": "tools", "content": self._tool_trace(raw_tool_results, whole_tool)},
            {"step": "budget", "content": budget.trace_content()},
        ]

        if whole_tool is not None:
            direct = self._assess_value(
                whole_tool.result,
                source="sympy_verified",
                spec=spec,
                evidence=evidence,
                extraction_method="certified_tool",
                explicit=True,
                method_id=whole_tool.operation,
                independence_group="local_tool",
                certified_tool=whole_tool,
            )
            if direct.validation_tier == "complete" and direct.tool_status == "pass":
                answer = self._render_submission(direct.answer, spec)
                trace.append({
                    "step": "selection",
                    "content": self._candidate_trace(direct, route="certified_tool"),
                })
                return {"final_response": answer, "trace": trace}
            trace.append({
                "step": "tool_fallback",
                "content": {"reason": "candidate_contract_rejected"},
            })
            evidence = tuple(
                ToolEvidence(
                    result=item.result,
                    scope="goal_result" if item.scope == "whole_goal" else item.scope,
                    operation=item.operation,
                    verified=item.verified,
                    certificate_method=item.certificate_method,
                    certificate_checks=item.certificate_checks,
                    certificate_issues=tuple((*item.certificate_issues, "whole_route_rejected")),
                    support=item.support,
                    assurance=item.assurance,
                )
                for item in evidence
            )
            budget = plan_stage_budget(spec, False, deep_reasoning=deep_reasoning)

        supported_tool = self._supported_tool_candidate(
            raw_tool_results,
            spec,
            evidence,
        )
        if supported_tool is not None:
            answer = self._render_submission(supported_tool.answer, spec)
            trace.append({
                "step": "selection",
                "content": self._candidate_trace(
                    supported_tool,
                    route="certified_tool_with_support",
                    model_calls=0,
                ),
            })
            return {"final_response": answer, "trace": trace}

        call_count = 0
        # A separate locator call had zero certificate yield.  The solver may
        # instead emit one grounded declarative block at no extra call cost;
        # local code recomputes it and ignores malformed/unsupported blocks.
        trace.append({
            "step": "tool_locator",
            "content": {
                "mode": "piggyback_declarative_whitelist",
                "separate_model_call": False,
                "model_supplies_result": False,
            },
        })

        obligation_graph = MathematicalObligationGraph.fallback(spec)
        method_search_admitted = bool(
            self.enable_mog
            and budget.allow_plan
            and budget.plan_tokens > 0
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= budget.review_min_remaining_seconds + 60
        )
        method_search_truncated = False
        if method_search_admitted:
            method_search_raw, method_search_result = self._call(
                obligation_planning_request(statement, spec),
                stage="method_search",
                max_tokens=budget.plan_tokens,
                temperature=0.15,
                thinking_mode=False,
                trace=trace,
                system_prompt=obligation_planning_system_prompt(spec),
            )
            call_count += self._request_count(method_search_result)
            method_search_truncated = self._truncated(
                method_search_result, method_search_raw
            )
            if not method_search_truncated:
                obligation_graph = MathematicalObligationGraph.parse(
                    method_search_raw, spec
                )
        primary_plan = obligation_graph.route_plan(spec, 0)
        independent_plan = obligation_graph.route_plan(spec, 1)
        plan = primary_plan
        trace.append({
            "step": "solve_plan",
            "content": {
                "admitted": method_search_admitted,
                "mode": (
                    "mog_method_portfolio"
                    if obligation_graph.valid
                    else "deterministic_problem_spec"
                ),
                "separate_model_call": method_search_admitted,
                "model_plan_admitted": obligation_graph.valid,
                "provider_truncated": method_search_truncated,
            },
        })
        trace.append({
            "step": "obligation_graph",
            "content": obligation_graph.trace_content(),
        })

        if obligation_graph.valid:
            return self._solve_mog_portfolio(
                statement,
                spec,
                cards,
                evidence,
                obligation_graph,
                budget,
                trace,
                call_count,
                started_at,
            )

        candidates: list[CandidateAssessment] = []
        primary_request = self._primary_request(
            statement,
            spec,
            cards,
            evidence,
            plan=primary_plan,
            deep_reasoning=deep_reasoning,
            hidden_thinking=hidden_thinking,
        )
        local_tool_opportunity = detect_local_tool_opportunity(
            statement,
            spec,
            allow_derived=self.enable_complex_subproblem_tools,
        )
        model_tool_names = self._model_tool_names(
            statement,
            spec,
            allow_complex_subproblem=self.enable_complex_subproblem_tools,
            complex_opportunity=local_tool_opportunity,
        )
        trace.append({
            "step": "local_tool_opportunity",
            "content": {
                **local_tool_opportunity.trace_content(),
                "experimental_enabled": self.enable_complex_subproblem_tools,
                "statement_exact_enabled": bool(
                    local_tool_opportunity.eligible
                    and local_tool_opportunity.scope == "statement_exact"
                    and set(local_tool_opportunity.allowed_tools)
                    <= _AUTO_EXACT_MODEL_TOOLS
                ),
                "offered": bool(model_tool_names),
            },
        })
        model_tools = self.model_math_tools if model_tool_names else None
        if model_tools is not None:
            primary_request += self._model_tool_instruction(
                spec,
                model_tool_names,
                opportunity=local_tool_opportunity,
            )
        first_raw, first_result = self._call(
            primary_request,
            stage="primary",
            max_tokens=budget.solve_tokens,
            temperature=0.2,
            thinking_mode=hidden_thinking,
            trace=trace,
            model_tools=model_tools,
            model_tool_names=model_tool_names,
            max_tool_rounds=1,
            require_model_tool=bool(
                self._model_tool_required(statement, spec)
                or (
                    self.enable_complex_subproblem_tools
                    and local_tool_opportunity.eligible
                )
                or (
                    local_tool_opportunity.eligible
                    and local_tool_opportunity.scope == "statement_exact"
                    and set(local_tool_opportunity.allowed_tools)
                    <= _AUTO_EXACT_MODEL_TOOLS
                    and spec.answer_contract.mode == "answer_only"
                )
            ),
            tool_followup_mode=(
                "reasoned_solution"
                if spec.answer_contract.mode != "answer_only"
                or spec.profile.task_kind in {
                    "proof", "derivation", "explanation", "construction"
                }
                else "compact_answer"
            ),
        )
        call_count += self._request_count(first_result)
        model_tool_evidence = self._model_tool_evidence(first_result)
        if model_tool_evidence:
            evidence = tuple((*evidence, *model_tool_evidence))
        first_truncated = self._truncated(first_result, first_raw)
        if first_truncated:
            trace.append({
                "step": "truncation_state",
                "content": {
                    "stage": "primary",
                    **classify_truncated_output(first_raw),
                },
            })
        first_candidates = self._transport_admissible(
            self._assess_response(
                first_raw,
                source="primary",
                spec=spec,
                evidence=evidence,
                method_id=(
                    "mog_route_a" if obligation_graph.valid else spec.primary_method
                ),
                independence_group="model_a",
            ),
            first_truncated,
        )
        candidates.extend(first_candidates)
        first_best = choose_candidate(first_candidates)
        first_usable = self._complete_after_transport(
            first_best, first_truncated, spec, first_raw
        )

        recovery_stage = ""
        recovery_raw = ""
        recovery_result = ModelCallResult("")
        recovery_candidates: list[CandidateAssessment] = []
        if not first_usable:
            if first_truncated and first_raw:
                recovery_stage = "primary_continuation"
                recovery_request = primary_request
                prior_response = first_raw
                followup = self._continuation_instruction(spec)
                recovery_tokens = min(
                    budget.repair_tokens,
                    budget.review_tokens,
                    4096,
                )
            elif model_tool_evidence:
                recovery_stage = "tool_synthesis"
                recovery_request = self._tool_synthesis_request(
                    statement, spec, model_tool_evidence
                )
                prior_response = ""
                followup = ""
                recovery_tokens = min(2048, budget.emergency_tokens)
            else:
                recovery_stage = "primary_rescue"
                recovery_request = self._recovery_request(
                    statement, spec, cards, evidence, plan=primary_plan
                )
                prior_response = ""
                followup = ""
                recovery_tokens = budget.emergency_tokens

        recovery_admitted = bool(
            recovery_stage
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= budget.repair_min_remaining_seconds
        )
        trace.append({
            "step": "primary_recovery_admission",
            "content": {
                "admitted": recovery_admitted,
                "mode": recovery_stage or "none",
                "first_usable": first_usable,
                "first_truncated": first_truncated,
            },
        })
        if recovery_admitted:
            recovery_raw, recovery_result = self._call(
                recovery_request,
                stage=recovery_stage,
                max_tokens=recovery_tokens,
                temperature=0.1,
                # This stage completes an existing draft or emits a short
                # clean answer after transport failure.  A paired replay found
                # that starting hidden reasoning here added truncations and
                # regressed answers, so every finish recovery stays compact.
                thinking_mode=False,
                trace=trace,
                prior_response=prior_response,
                followup=followup,
                system_prompt=(
                    self._critic_system_prompt(spec)
                    if recovery_stage == "tool_synthesis"
                    else ""
                ),
            )
            call_count += self._request_count(recovery_result)
            recovery_truncated = self._truncated(recovery_result, recovery_raw)
            if recovery_truncated:
                trace.append({
                    "step": "truncation_state",
                    "content": {
                        "stage": recovery_stage,
                        **classify_truncated_output(recovery_raw),
                    },
                })
            recovery_candidates = self._transport_admissible(
                self._assess_response(
                    recovery_raw,
                    source="primary_recovery",
                    spec=spec,
                    evidence=evidence,
                    method_id=recovery_stage,
                    independence_group="model_a",
                ),
                recovery_truncated,
            )
            candidates.extend(recovery_candidates)
        else:
            recovery_truncated = False

        recovery_best = choose_candidate(recovery_candidates)
        recovery_usable = self._complete_after_transport(
            recovery_best, recovery_truncated, spec, recovery_raw
        )
        base_candidate = first_best if first_usable else (
            recovery_best if recovery_usable else None
        )

        # Arbitrary model-generated Python is not part of the production
        # pipeline.  Candidate-level deterministic checks are supplied by the
        # whitelisted VERIFY_JSON protocol during response assessment.
        computation_candidate: CandidateAssessment | None = None
        trace.append({
            "step": "structured_verification",
            "content": {
                "protocol": "verify_json_whitelist",
                "arbitrary_code_execution": False,
            },
        })

        # A fresh solve is the only model-only check that adds independent
        # mathematical evidence.  It cannot see the primary draft.  This is
        # deliberately different from a candidate-visible critique, which can
        # inherit the first solution's error and then merely restate it.
        independent_candidates: list[CandidateAssessment] = []
        independent_raw = ""
        completed_raw = ""
        independent_result = ModelCallResult("")
        proof_or_expository = bool(
            spec.answer_contract.mode != "answer_only"
            or spec.profile.task_kind in {
                "proof", "derivation", "explanation", "construction"
            }
        )
        hard_independent_pair_mode = bool(
            self.enable_blind_consensus
            and
            base_candidate is not None
            and deep_reasoning
            and budget.require_independent_review
            and not self._objectively_checked(base_candidate)
            and (
                proof_or_expository
                or spec.profile.difficulty == "hard"
                or spec.profile.topic.startswith("olympiad_")
                or spec.risk_score >= 4
            )
        )
        # Candidate-visible audits are useful for bounded checks but inherit
        # the first derivation's framing. High-risk results instead receive a
        # blind second derivation so agreement represents independent evidence.
        audit_eligible = bool(
            base_candidate is not None
            and not hard_independent_pair_mode
            and not self._objectively_checked(base_candidate)
        )
        audit_mode = bool(self.enable_candidate_audit and audit_eligible)
        audit_suppressed = bool(audit_eligible and not self.enable_candidate_audit)
        conclusion_recovery_mode = bool(
            base_candidate is None
            and recovery_stage == "primary_continuation"
            and recovery_truncated
            and str(recovery_raw or "").strip()
        )
        draft_recovery_mode = bool(
            base_candidate is None
            and not conclusion_recovery_mode
            and (str(first_raw or "").strip() or str(recovery_raw or "").strip())
        )
        blind_rescue_mode = bool(base_candidate is None and not draft_recovery_mode)
        computation_conflict_mode = bool(
            base_candidate is not None
            and computation_candidate is not None
            and self._conflict(base_candidate, computation_candidate, spec)
        )
        quick_consensus_mode = bool(
            self.enable_quick_consensus
            and not audit_mode
            and
            base_candidate is not None
            and computation_candidate is None
            and not deep_reasoning
            and budget.require_independent_review
            and not self._objectively_checked(base_candidate)
        )
        deep_consensus_mode = bool(
            self.enable_blind_consensus
            and
            base_candidate is not None
            and computation_candidate is None
            and deep_reasoning
            and budget.require_independent_review
            and not self._objectively_checked(base_candidate)
            and not audit_suppressed
        )
        independent_mode = (
            "candidate_audit" if audit_mode else
            "conclusion_recovery" if conclusion_recovery_mode else
            "draft_recovery" if draft_recovery_mode else
            "blind_rescue" if blind_rescue_mode else
            "computation_tiebreak" if computation_conflict_mode else
            "independent_solve" if deep_consensus_mode else
            "quick_consensus" if quick_consensus_mode else
            "none"
        )
        independent_min_seconds = (
            90 if blind_rescue_mode and deep_reasoning else
            budget.review_min_remaining_seconds
            if deep_consensus_mode else
            budget.repair_min_remaining_seconds
        )
        independent_needed = bool(
            audit_mode or conclusion_recovery_mode
            or draft_recovery_mode or blind_rescue_mode
            or computation_conflict_mode or deep_consensus_mode
            or quick_consensus_mode
        )
        independent_admitted = bool(
            budget.allow_review
            and independent_needed
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= independent_min_seconds
        )
        admission_content = {
            "admitted": independent_admitted,
            "required": independent_needed,
            "mode": independent_mode,
            "minimum_remaining_seconds": independent_min_seconds,
            "base_source": base_candidate.source if base_candidate else "none",
            "candidate_audit_enabled": self.enable_candidate_audit,
            "candidate_audit_eligible": audit_eligible,
            "candidate_audit_suppressed": audit_suppressed,
            "blind_consensus_enabled": self.enable_blind_consensus,
            "quick_consensus_enabled": self.enable_quick_consensus,
        }
        if self.local_candidate_diagnostics and base_candidate is not None:
            admission_content["base_candidate"] = base_candidate.answer[:12_000]
        trace.append({
            "step": "independent_admission",
            "content": admission_content,
        })
        audit_decision = "not_run"
        audit_selected: CandidateAssessment | None = None
        pair_critic_ran = False
        pair_critic_decision = "not_run"
        pair_critic_selected: CandidateAssessment | None = None
        if independent_admitted:
            independent_request = (
                self._candidate_audit_request(
                    statement, spec, base_candidate, evidence, plan=primary_plan
                )
                if audit_mode and base_candidate is not None
                else primary_request
                if conclusion_recovery_mode
                else self._recovery_request(
                    statement,
                    spec,
                    cards,
                    evidence,
                    drafts=(first_raw, recovery_raw),
                    plan=primary_plan,
                )
                if draft_recovery_mode
                else self._independent_request(
                    statement, spec, cards, evidence, plan=independent_plan
                )
            )
            independent_prior = recovery_raw if conclusion_recovery_mode else ""
            independent_followup = (
                self._conclusion_only_instruction(spec)
                if conclusion_recovery_mode else ""
            )
            independent_raw, independent_result = self._call(
                independent_request,
                stage=independent_mode,
                max_tokens=(
                    budget.repair_tokens if audit_mode
                    else min(1536, budget.emergency_tokens)
                    if conclusion_recovery_mode
                    else budget.emergency_tokens if draft_recovery_mode
                    else min(budget.review_tokens, 8192)
                ),
                temperature=(
                    0.0 if conclusion_recovery_mode
                    else 0.1 if audit_mode or draft_recovery_mode
                    else 0.25
                ),
                # Draft recovery receives bounded, explicitly untrusted excerpts
                # instead of restarting a third long derivation. Other modes are
                # candidate-blind and preserve an independent error profile.
                thinking_mode=bool(
                    deep_reasoning
                    and not audit_mode
                    and not conclusion_recovery_mode
                    and not draft_recovery_mode
                ),
                trace=trace,
                prior_response=independent_prior,
                followup=independent_followup,
                system_prompt=(
                    self._critic_system_prompt(spec) if audit_mode else ""
                ),
            )
            call_count += self._request_count(independent_result)
            independent_truncated = self._truncated(
                independent_result, independent_raw
            )
            independent_candidates.extend(self._transport_admissible(
                self._assess_response(
                    independent_raw,
                    source=independent_mode,
                    spec=spec,
                    evidence=evidence,
                    method_id=(
                        "candidate_audit" if audit_mode
                        else "conclusion_recovery"
                        if conclusion_recovery_mode
                        else "primary_route_recovery"
                        if draft_recovery_mode
                        else "mog_route_b"
                        if obligation_graph.valid
                        else spec.alternative_method
                    ),
                    independence_group=(
                        "model_a" if conclusion_recovery_mode else "model_b"
                    ),
                ),
                independent_truncated,
            ))
            independent_best = choose_candidate(independent_candidates)
            independent_usable = self._complete_after_transport(
                independent_best, independent_truncated, spec, independent_raw
            )

            if audit_mode:
                audit_decision, audit_selected = self._apply_candidate_audit(
                    independent_raw,
                    independent_result,
                    independent_candidates,
                    base_candidate,
                    spec,
                )
                # An unresolved candidate-visible audit is not an independent
                # vote and must never displace the audited solution.
                independent_usable = False

            independent_recovery_stage = ""
            if (
                not audit_mode
                and not conclusion_recovery_mode
                and not independent_usable
            ):
                if base_candidate is not None and independent_raw:
                    # When A already has a complete answer, spending the last
                    # window merely formatting B leaves no independent audit.
                    # Use B's blind work as evidence for a focused Critic that
                    # checks A and emits a complete repaired answer if needed.
                    independent_recovery_stage = "pair_critic_repair"
                    independent_recovery_request = self._pair_critic_repair_request(
                        statement,
                        spec,
                        evidence,
                        base_candidate,
                        independent_raw,
                    )
                    independent_prior = ""
                    independent_followup = ""
                elif independent_truncated and independent_raw:
                    independent_recovery_stage = "independent_continuation"
                    independent_recovery_request = independent_request
                    independent_prior = independent_raw
                    independent_followup = self._continuation_instruction(spec)
                else:
                    independent_recovery_stage = "independent_rescue"
                    independent_recovery_request = self._recovery_request(
                        statement, spec, cards, evidence
                    )
                    independent_prior = ""
                    independent_followup = ""

            independent_recovery_admitted = bool(
                independent_recovery_stage
                and call_count < budget.max_calls
                and self._remaining_seconds(started_at)
                >= budget.repair_min_remaining_seconds
            )
            if independent_recovery_admitted:
                pair_critic_ran = independent_recovery_stage == "pair_critic_repair"
                completed_raw, completed_result = self._call(
                    independent_recovery_request,
                    stage=independent_recovery_stage,
                    max_tokens=budget.repair_tokens,
                    temperature=0.1,
                    thinking_mode=False,
                    trace=trace,
                    prior_response=independent_prior,
                    followup=independent_followup,
                    system_prompt=(
                        self._critic_system_prompt(spec)
                        if pair_critic_ran
                        else ""
                    ),
                )
                call_count += self._request_count(completed_result)
                completed_truncated = self._truncated(
                    completed_result, completed_raw
                )
                completed_candidates = self._transport_admissible(
                    self._assess_response(
                        completed_raw,
                        source=(
                            "pair_critic_repair"
                            if pair_critic_ran
                            else independent_mode
                        ),
                        spec=spec,
                        evidence=evidence,
                        method_id=independent_recovery_stage,
                        independence_group=(
                            "model_c" if pair_critic_ran else "model_b"
                        ),
                    ),
                    completed_truncated,
                )
                independent_candidates.extend(completed_candidates)
                if pair_critic_ran:
                    pair_critic_decision, pair_critic_selected = (
                        self._apply_pair_critic_repair(
                            completed_raw,
                            completed_result,
                            completed_candidates,
                            base_candidate,
                            spec,
                        )
                    )
                    # The Critic is a decision stage, not another independent
                    # vote. Its raw FINAL cannot enter majority selection.
                    independent_usable = False
                else:
                    independent_best = choose_candidate(independent_candidates)
                    independent_usable = self._complete_after_transport(
                        independent_best, completed_truncated, spec, completed_raw
                    )
            else:
                independent_recovery_stage = ""
        else:
            independent_best = None
            independent_usable = False
            independent_truncated = False
            independent_recovery_stage = ""
            independent_recovery_admitted = False

        candidates.extend(independent_candidates)
        selected: CandidateAssessment | None = None
        selection_route = ""
        arbitration_decision = audit_decision
        candidates_conflict = bool(
            not audit_mode
            and base_candidate is not None
            and independent_usable
            and independent_best is not None
            and self._conflict(base_candidate, independent_best, spec)
        )

        if pair_critic_ran and base_candidate is not None:
            selected = pair_critic_selected or base_candidate
            arbitration_decision = pair_critic_decision
            selection_route = (
                "pair_critic_" + pair_critic_decision.lower()
                if pair_critic_selected is not None
                else "primary_retained_after_unresolved_pair_critic"
            )
        elif audit_mode and base_candidate is not None:
            selected = audit_selected or base_candidate
            selection_route = (
                f"candidate_audit_{audit_decision.lower()}"
                if audit_selected is not None
                else "primary_retained_after_unresolved_audit"
            )
        elif base_candidate is not None and computation_candidate is not None:
            if not computation_conflict_mode:
                selected = choose_candidate([
                    base_candidate, computation_candidate
                ]) or base_candidate
                selection_route = "model_executable_consensus"
            elif independent_usable and independent_best is not None:
                independent_matches_base = not self._conflict(
                    independent_best, base_candidate, spec
                )
                independent_matches_computation = not self._conflict(
                    independent_best, computation_candidate, spec
                )
                if independent_matches_base or independent_matches_computation:
                    majority = (
                        [base_candidate, independent_best]
                        if independent_matches_base
                        else [computation_candidate, independent_best]
                    )
                    selected = choose_candidate(majority) or majority[0]
                    selection_route = (
                        "model_majority_after_executable_disagreement"
                        if independent_matches_base
                        else "executable_majority_after_model_disagreement"
                    )
                    arbitration_decision = "independent_tiebreak"
            if selected is None:
                # A successfully executed program may still mistranslate the
                # statement.  With no independent agreement, retain the full
                # mathematical solve rather than treating execution as truth.
                selected = base_candidate
                selection_route = "primary_retained_after_executable_disagreement"
                arbitration_decision = "unresolved_translation"
        elif base_candidate is not None and independent_usable and independent_best is not None:
            objective_winner = self._objective_winner(
                base_candidate, independent_best
            )
            if objective_winner is not None:
                selected = objective_winner
                selection_route = "independent_objective_certificate"
            elif not candidates_conflict:
                consensus = (
                    choose_candidate([base_candidate, independent_best])
                    or base_candidate
                )
                consensus_audit_admitted = bool(
                    hard_independent_pair_mode
                    and proof_or_expository
                    and budget.allow_repair
                    and call_count < budget.max_calls
                    and self._remaining_seconds(started_at)
                    >= budget.repair_min_remaining_seconds
                )
                if consensus_audit_admitted:
                    consensus_raw, consensus_result = self._call(
                        self._consensus_audit_request(
                            statement,
                            spec,
                            evidence,
                            base_candidate,
                            independent_best,
                        ),
                        stage="hard_proof_consensus_audit",
                        max_tokens=budget.repair_tokens,
                        temperature=0.1,
                        thinking_mode=False,
                        trace=trace,
                        system_prompt=self._critic_system_prompt(spec),
                    )
                    call_count += self._request_count(consensus_result)
                    consensus_truncated = self._truncated(
                        consensus_result, consensus_raw
                    )
                    consensus_candidates = self._transport_admissible(
                        self._assess_response(
                            consensus_raw,
                            source="hard_proof_consensus_audit",
                            spec=spec,
                            evidence=evidence,
                            method_id="consensus_audit",
                            independence_group="model_c",
                        ),
                        consensus_truncated,
                    )
                    candidates.extend(consensus_candidates)
                    arbitration_decision, audited = self._apply_consensus_audit(
                        consensus_raw,
                        consensus_result,
                        consensus_candidates,
                        base_candidate,
                        independent_best,
                        spec,
                    )
                    if audited is not None:
                        selected = audited
                        selection_route = (
                            "hard_independent_consensus_"
                            + arbitration_decision.lower()
                        )
                if selected is None:
                    selected = consensus
                    selection_route = (
                        "hard_independent_consensus_retained"
                        if hard_independent_pair_mode
                        else "quick_independent_consensus"
                    )
            else:
                # Deep disagreements benefit from a focused comparison of the
                # two derivations; short factual disagreements remain safer as
                # a blind third vote that cannot inherit either answer.
                tiebreak_admitted = bool(
                    budget.allow_repair
                    and call_count < budget.max_calls
                    and self._remaining_seconds(started_at)
                    >= budget.repair_min_remaining_seconds
                )
                if tiebreak_admitted:
                    tiebreak_request = (
                        self._arbitration_request(
                            statement,
                            spec,
                            evidence,
                            base_candidate,
                            independent_best,
                            first_context=self._candidate_work_context(
                                base_candidate,
                                primary_raw=first_raw,
                                primary_recovery_raw=recovery_raw,
                                independent_source=independent_mode,
                                independent_raw=independent_raw,
                                independent_recovery_raw=completed_raw,
                            ),
                            second_context=self._candidate_work_context(
                                independent_best,
                                primary_raw=first_raw,
                                primary_recovery_raw=recovery_raw,
                                independent_source=independent_mode,
                                independent_raw=independent_raw,
                                independent_recovery_raw=completed_raw,
                            ),
                        )
                        if deep_reasoning
                        else self._blind_tiebreak_request(
                            statement, spec, cards, evidence, plan=plan
                        )
                    )
                    tiebreak_raw, tiebreak_result = self._call(
                        tiebreak_request,
                        stage=("deep_arbitration" if deep_reasoning else "blind_tiebreak"),
                        max_tokens=(
                            budget.repair_tokens
                            if deep_reasoning
                            else budget.review_tokens
                        ),
                        temperature=(0.1 if deep_reasoning else 0.3),
                        # Arbitration compares two completed derivations.  A
                        # third hidden chain repeatedly exhausted the response
                        # budget before emitting its verdict; a short explicit
                        # falsification check is more reliable here.
                        thinking_mode=False,
                        trace=trace,
                        system_prompt=(
                            self._critic_system_prompt(spec)
                            if deep_reasoning
                            else ""
                        ),
                    )
                    call_count += self._request_count(tiebreak_result)
                    tiebreak_truncated = self._truncated(
                        tiebreak_result, tiebreak_raw
                    )
                    tiebreak_candidates = self._transport_admissible(
                        self._assess_response(
                            tiebreak_raw,
                            source=(
                                "deep_arbitration"
                                if deep_reasoning
                                else "blind_tiebreak"
                            ),
                            spec=spec,
                            evidence=evidence,
                            method_id="blind_tiebreak",
                            independence_group="model_c",
                        ),
                        tiebreak_truncated,
                    )
                    candidates.extend(tiebreak_candidates)
                    tiebreak_best = choose_candidate(tiebreak_candidates)
                    if deep_reasoning:
                        arbitration_decision, arbitration_selected = (
                            self._apply_arbitration(
                                tiebreak_raw,
                                tiebreak_result,
                                tiebreak_candidates,
                                base_candidate,
                                independent_best,
                                spec,
                                evidence,
                            )
                        )
                        if arbitration_selected is not None:
                            selected = arbitration_selected
                            selection_route = (
                                f"deep_arbitration_{arbitration_decision.lower()}"
                            )
                    else:
                        tiebreak_usable = self._complete_after_transport(
                            tiebreak_best, tiebreak_truncated, spec, tiebreak_raw
                        )
                        tiebreak_matches_base = bool(
                            tiebreak_usable
                            and not self._conflict(
                                tiebreak_best, base_candidate, spec
                            )
                        )
                        tiebreak_matches_independent = bool(
                            tiebreak_usable
                            and not self._conflict(
                                tiebreak_best, independent_best, spec
                            )
                        )
                        if tiebreak_matches_base or tiebreak_matches_independent:
                            majority = (
                                [base_candidate, tiebreak_best]
                                if tiebreak_matches_base
                                else [independent_best, tiebreak_best]
                            )
                            selected = choose_candidate(majority) or majority[0]
                            arbitration_decision = "blind_majority"
                            selection_route = "blind_majority"
                if selected is None:
                    # Transport quality and stage provenance establish that a
                    # candidate is complete, not that its mathematics is
                    # correct.  Once both conflicting candidates are usable,
                    # only an objective local check or a resolved arbitration
                    # may replace the first complete solution.
                    selected = base_candidate
                    if arbitration_decision == "not_run":
                        arbitration_decision = "no_majority"
                    selection_route = (
                        "primary_retained_after_unresolved_arbitration"
                        if deep_reasoning and tiebreak_admitted
                        else "primary_retained_without_majority"
                    )
        elif base_candidate is not None:
            selected = base_candidate
            selection_route = "primary_only_complete"
        elif independent_usable and independent_best is not None:
            selected = independent_best
            selection_route = "independent_only_complete"

        last_chance_admitted = bool(
            selected is None
            and budget.allow_repair
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= budget.repair_min_remaining_seconds
        )
        if last_chance_admitted:
            last_raw, last_result = self._call(
                self._recovery_request(
                    statement,
                    spec,
                    cards,
                    evidence,
                    drafts=(first_raw, recovery_raw, independent_raw),
                ),
                stage="last_chance",
                max_tokens=budget.emergency_tokens,
                temperature=0.1,
                thinking_mode=False,
                trace=trace,
            )
            call_count += self._request_count(last_result)
            last_truncated = self._truncated(last_result, last_raw)
            last_candidates = self._transport_admissible(
                self._assess_response(
                    last_raw,
                    source="last_chance",
                    spec=spec,
                    evidence=evidence,
                    method_id="last_chance",
                    independence_group="model_c",
                ),
                last_truncated,
            )
            candidates.extend(last_candidates)
            last_best = choose_candidate(last_candidates)
            if self._complete_after_transport(
                last_best, last_truncated, spec, last_raw
            ):
                selected = last_best
                selection_route = "last_chance_complete"

        if selected is None:
            selected = self._objective_winner(*candidates)
            if selected is not None:
                selection_route = "objective_certificate"

        proof_final_audit_decision = "not_run"
        proof_final_audit_admitted = bool(
            self.enable_candidate_audit
            and selected is not None
            and proof_or_expository
            and not self._objectively_checked(selected)
            and not pair_critic_ran
            and not audit_mode
            and not selection_route.startswith("hard_independent_consensus_")
            and not selection_route.startswith("deep_arbitration_")
            and budget.allow_repair
            and call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= budget.repair_min_remaining_seconds
        )
        if proof_final_audit_admitted:
            proof_audit_raw, proof_audit_result = self._call(
                self._candidate_audit_request(
                    statement,
                    spec,
                    selected,
                    evidence,
                    plan=plan,
                ),
                stage="final_proof_audit",
                max_tokens=budget.repair_tokens,
                temperature=0.1,
                thinking_mode=False,
                trace=trace,
                system_prompt=self._critic_system_prompt(spec),
            )
            call_count += self._request_count(proof_audit_result)
            proof_audit_truncated = self._truncated(
                proof_audit_result, proof_audit_raw
            )
            proof_audit_candidates = self._transport_admissible(
                self._assess_response(
                    proof_audit_raw,
                    source="final_proof_audit",
                    spec=spec,
                    evidence=evidence,
                    method_id="final_proof_audit",
                    independence_group="model_c",
                ),
                proof_audit_truncated,
            )
            candidates.extend(proof_audit_candidates)
            proof_audit_decision, proof_audited = self._apply_candidate_audit(
                proof_audit_raw,
                proof_audit_result,
                proof_audit_candidates,
                selected,
                spec,
            )
            proof_final_audit_decision = proof_audit_decision
            if proof_audited is not None:
                selected = proof_audited
                selection_route = (
                    "final_proof_audit_" + proof_audit_decision.lower()
                )
                arbitration_decision = proof_audit_decision

        trace.append({
            "step": "cross_check",
            "content": {
                "primary_recovery_stage": recovery_stage or "none",
                "base_candidate": base_candidate.source if base_candidate else "none",
                "independent_admitted": independent_admitted,
                "independent_recovery_stage": independent_recovery_stage or "none",
                "independent_usable": independent_usable,
                "candidate_conflict": candidates_conflict,
                "arbitration_decision": arbitration_decision,
                "proof_final_audit_admitted": proof_final_audit_admitted,
                "proof_final_audit_decision": proof_final_audit_decision,
            },
        })
        trace.append({
            "step": "candidate_audit",
            "content": [self._candidate_trace(item) for item in candidates[:12]],
        })
        if selected is None:
            selected = self._best_degraded(candidates, spec)
            if selected is not None:
                selection_route = "degraded_fallback"
        certified_fallback = False
        if selected is None:
            selected = self._certified_goal_fallback(raw_tool_results, spec, evidence)
            certified_fallback = selected is not None
            if certified_fallback:
                selection_route = "certified_goal_fallback"
        explicit_consensus_fallback = False
        if selected is None:
            selected = self._rejected_explicit_consensus(candidates, spec)
            explicit_consensus_fallback = selected is not None
            if explicit_consensus_fallback:
                selection_route = "rejected_explicit_consensus"
        if selected is None:
            # Do not turn four unusable model responses into a fabricated
            # numeric/choice/truth answer.  The public entry point treats an
            # empty internal result as a request for its one clean emergency
            # solve; only total provider failure reaches the shape sentinel.
            answer = ""
            trace.append({
                "step": "selection",
                "content": {
                    "route": "entrypoint_emergency_required",
                    "model_calls": call_count,
                    "reason": "no_structurally_usable_candidate",
                },
            })
        else:
            answer = self._render_submission(selected.answer, spec)
            trace.append({
                "step": "selection",
                "content": self._candidate_trace(
                    selected,
                    route=(
                        "certified_goal_fallback" if certified_fallback else selection_route
                    ),
                    model_calls=call_count,
                    primary_recovery_stage=recovery_stage or "none",
                    independent_admitted=independent_admitted,
                    independent_recovery_stage=independent_recovery_stage or "none",
                    arbitration_decision=arbitration_decision,
                ),
            })
        if answer.strip() and Finalizer.validate_structure(answer):
            answer = ""
            trace.append({
                "step": "final_guard",
                "content": {"status": "entrypoint_emergency_after_render_validation"},
            })
        return {"final_response": answer, "trace": trace}

    def emergency_solve(self, problem: str) -> dict:
        """One answer-first attempt for entry-point failures before a valid result."""
        statement = str(problem or "").strip()
        spec = build_problem_spec(statement)
        prompt = self._recovery_request(statement, spec, RetrievalBundle((), ()), ())
        raw, result = self._call(
            prompt,
            stage="entrypoint_recovery",
            max_tokens=2048,
            temperature=0.1,
            thinking_mode=False,
            trace=[],
        )
        candidates = self._assess_response(
            raw,
            source="entrypoint_recovery",
            spec=spec,
            evidence=(),
            method_id="emergency",
            independence_group="emergency",
        )
        complete_candidates = [
            item for item in candidates if item.validation_tier == "complete"
        ]
        selected = (
            choose_candidate(complete_candidates)
            or self._best_degraded(candidates, spec)
            or self._emergency_explicit_intent(candidates, spec)
        )
        answer = self._render_submission(selected.answer, spec) if selected else ""
        return {
            "final_response": answer,
            "trace": [{
                "step": "entrypoint_recovery",
                "content": {
                    "finish_reason": result.finish_reason or "unavailable",
                    "candidate_found": selected is not None,
                },
            }],
        }

    def _solve_mog_portfolio(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        graph: MathematicalObligationGraph,
        budget: StageBudget,
        trace: list[dict],
        call_count: int,
        started_at: float,
    ) -> dict:
        """Reserve one solve window for each method before any repair."""
        route_records: list[dict] = []
        route_raws: list[str] = []
        all_candidates: list[CandidateAssessment] = []
        token_limits = self.mog_route_token_limits or (
            min(3584, max(256, budget.solve_tokens)),
            min(4096, max(256, budget.review_tokens)),
        )
        for index, route_name in enumerate(("A", "B")):
            admitted = bool(
                call_count < budget.max_calls
                and self._remaining_seconds(started_at)
                >= budget.repair_min_remaining_seconds + 60
            )
            if not admitted:
                route_raws.append("")
                route_records.append({
                    "route": route_name,
                    "executed": False,
                    "truncated": False,
                    "key_lemma_visible": False,
                    "remaining_declared": False,
                    "candidate_complete": False,
                })
                continue
            route_plan = graph.route_plan(spec, index)
            raw, result = self._call(
                self._mog_route_request(
                    problem,
                    spec,
                    cards,
                    evidence,
                    route_plan,
                    route_name,
                ),
                stage=f"mog_route_{route_name.lower()}",
                max_tokens=token_limits[index],
                temperature=0.2 if index == 0 else 0.3,
                thinking_mode=self.mog_route_thinking,
                trace=trace,
                system_prompt=self._mog_route_system_prompt(spec),
            )
            call_count += self._request_count(result)
            route_raws.append(raw)
            truncated = self._truncated(result, raw)
            candidates = self._transport_admissible(
                self._assess_response(
                    raw,
                    source=f"mog_route_{route_name.lower()}",
                    spec=spec,
                    evidence=evidence,
                    method_id=f"mog_route_{route_name.lower()}",
                    independence_group=f"model_{route_name.lower()}",
                ),
                truncated,
            )
            all_candidates.extend(candidates)
            best = choose_candidate(candidates)
            route_records.append({
                "route": route_name,
                "executed": True,
                "truncated": truncated,
                "key_lemma_visible": bool(re.search(
                    r"(?im)^\s*KEY_LEMMA\s*[:：]\s*\S", raw
                )),
                "remaining_declared": bool(re.search(
                    r"(?im)^\s*REMAINING\s*[:：]\s*\S", raw
                )),
                "candidate_complete": self._is_complete(best),
                "candidate_explicit": bool(best and best.explicit_answer),
            })

        trace.append({
            "step": "mog_route_search",
            "content": route_records,
        })
        first_best = choose_candidate([
            item for item in all_candidates
            if item.independence_group == "model_a"
        ])
        second_best = choose_candidate([
            item for item in all_candidates
            if item.independence_group == "model_b"
        ])
        base = first_best if self._is_result_usable(first_best) else (
            second_best if self._is_result_usable(second_best) else None
        )
        other = second_best if base is first_best else first_best

        synthesis_candidates: list[CandidateAssessment] = []
        synthesis_raw = ""
        synthesis_result = ModelCallResult("")
        synthesis_admitted = bool(
            call_count < budget.max_calls
            and self._remaining_seconds(started_at)
            >= budget.repair_min_remaining_seconds
        )
        synthesis_decision = "not_run"
        synthesis_selected: CandidateAssessment | None = None
        if synthesis_admitted:
            synthesis_request = self._mog_falsifier_request(
                problem,
                spec,
                evidence,
                base,
                other,
                route_records,
                route_raws,
            )
            synthesis_raw, synthesis_result = self._call(
                synthesis_request,
                stage="mog_falsifier_synthesis",
                max_tokens=min(3072, max(1536, budget.repair_tokens)),
                temperature=0.1,
                thinking_mode=False,
                trace=trace,
                system_prompt=self._critic_system_prompt(spec),
            )
            call_count += self._request_count(synthesis_result)
            synthesis_truncated = self._truncated(
                synthesis_result, synthesis_raw
            )
            synthesis_candidates = self._transport_admissible(
                self._assess_response(
                    synthesis_raw,
                    source="mog_falsifier_synthesis",
                    spec=spec,
                    evidence=evidence,
                    method_id="certificate_first_falsifier",
                    independence_group="model_c",
                ),
                synthesis_truncated,
            )
            all_candidates.extend(synthesis_candidates)
            if base is not None:
                synthesis_check = self._audit_check_section(synthesis_raw)
                if self._mog_reproducible_check(synthesis_check):
                    synthesis_decision, synthesis_selected = (
                        self._apply_pair_critic_repair(
                            synthesis_raw,
                            synthesis_result,
                            synthesis_candidates,
                            base,
                            spec,
                        )
                    )
                else:
                    synthesis_decision = "non_reproducible_check"
            else:
                proposed = choose_candidate(synthesis_candidates)
                if self._complete_after_transport(
                    proposed, synthesis_truncated, spec, synthesis_raw
                ):
                    synthesis_selected = proposed
                    synthesis_decision = "recovered_without_route_candidate"

        if synthesis_selected is not None:
            selected = synthesis_selected
            selection_route = "mog_falsifier_" + synthesis_decision.lower()
        elif (
            self._is_result_usable(first_best)
            and self._is_result_usable(second_best)
            and not self._conflict(first_best, second_best, spec)
        ):
            selected = choose_candidate([first_best, second_best]) or first_best
            selection_route = "mog_independent_consensus"
        else:
            selected = self._objective_winner(first_best, second_best)
            if selected is not None:
                selection_route = "mog_objective_certificate"
            else:
                selected = base or self._best_degraded(all_candidates, spec)
                selection_route = (
                    "mog_best_complete_after_unresolved_falsifier"
                    if base is not None
                    else "mog_degraded_fallback"
                )

        trace.append({
            "step": "mog_falsifier",
            "content": {
                "admitted": synthesis_admitted,
                "decision": synthesis_decision,
                "route_a_executed": bool(route_records and route_records[0]["executed"]),
                "route_b_executed": bool(
                    len(route_records) > 1 and route_records[1]["executed"]
                ),
                "routes_agree": bool(
                    self._is_result_usable(first_best)
                    and self._is_result_usable(second_best)
                    and not self._conflict(first_best, second_best, spec)
                ),
                "model_calls": call_count,
            },
        })
        trace.append({
            "step": "candidate_audit",
            "content": [self._candidate_trace(item) for item in all_candidates[:12]],
        })
        if selected is None:
            selected = self._rejected_explicit_consensus(all_candidates, spec)
            if selected is not None:
                selection_route = "mog_rejected_explicit_consensus"
        if selected is None:
            answer = ""
            trace.append({
                "step": "selection",
                "content": {
                    "route": "entrypoint_emergency_required",
                    "model_calls": call_count,
                },
            })
        else:
            answer = self._render_submission(selected.answer, spec)
            trace.append({
                "step": "selection",
                "content": self._candidate_trace(
                    selected,
                    route=selection_route,
                    model_calls=call_count,
                ),
            })
        return {"final_response": answer, "trace": trace}

    def _mog_route_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        plan: SolvePlan,
        route_name: str,
    ) -> str:
        proof_note = (
            "The FINAL section must contain the shortest complete proof needed for grading."
            if spec.answer_contract.mode != "answer_only"
            else "The FINAL line must contain the entire gradable result."
        )
        role = (
            f"COMPACT ROUTE {route_name}: establish the assigned hardest obligation and a "
            "gradable provisional conclusion. This is a bounded proof state, not a polished "
            "essay. Do not list abandoned approaches, restate the problem, or continue after "
            "the required state is recorded. Work from the original statement; the route "
            "plan is untrusted. Derive before committing to a conclusion. Output exactly "
            "these labelled sections in this order:\n"
            f"METHOD: the Route {route_name} method actually used\n"
            "KEY_LEMMA: one precise lemma actually established\n"
            "CRITICAL_STEPS: at most five compact mathematical implications or calculations\n"
            "CHECK: one falsifiable substitution, boundary, invariant, residual, or theorem-condition check\n"
            "REMAINING: NONE, or one exact unresolved obligation\n"
            "FINAL: complete provisional answer with essential support; write this only after the check\n"
            f"{proof_note}"
        )
        return self._request(
            problem,
            spec,
            role=role,
            method=(
                "Use only the assigned route. If its key lemma is false or inapplicable, "
                "record the precise obstruction in REMAINING instead of switching to the "
                "other portfolio route."
            ),
            context=self._grounded_context(
                spec, cards, review=False, plan=plan
            ),
            evidence=evidence,
            plan=plan,
        )

    def _mog_falsifier_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        base: CandidateAssessment | None,
        other: CandidateAssessment | None,
        route_records: list[dict],
        route_raws: list[str],
    ) -> str:
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "complete requested result"
        first_text = (
            self._draft_excerpt(route_raws[0])
            if route_raws and route_raws[0]
            else self._bounded(base.answer, 5000) if base is not None else "unavailable"
        )
        second_text = (
            self._draft_excerpt(route_raws[1])
            if len(route_raws) > 1 and route_raws[1]
            else self._bounded(other.answer, 5000) if other is not None else "unavailable"
        )
        route_status = ", ".join(
            f"{item['route']}:truncated={item['truncated']},key_lemma={item['key_lemma_visible']}"
            for item in route_records
        )
        proof_note = (
            "FINAL must contain the complete concise proof, not only its conclusion."
            if spec.answer_contract.mode != "answer_only"
            else "FINAL must contain the entire independently gradable result."
        )
        return (
            "Act as a certificate-first falsifier and final synthesizer. Two method-diverse "
            "routes were attempted under hard output limits. Do not restart a full solution. "
            "Attack the first decisive implication, theorem hypothesis, boundary case, sign, "
            "normalization, or finite-domain condition where the candidates could fail. "
            "A failed attack is not proof of correctness. Preserve candidate A unless a "
            "reproducible CHECK establishes an error; use B or a local correction only when "
            "that check identifies the error. Words such as plausible, consistent, standard, "
            "or closer to the bound are not checks. A valid CHECK must contain a recomputable "
            "equality/inequality, an explicit counterexample, a residual, an exhaustive finite "
            "domain argument, or a named theorem with each relevant hypothesis verified. "
            "If neither route has such support, use UNRESOLVED. Output exactly these sections, "
            "with FINAL first:\n"
            "FINAL: the complete checked answer\n"
            "DECISION: KEEP_A, USE_B, CORRECTED, or UNRESOLVED\n"
            "CHECK: one concrete reproducible falsification attempt or certificate\n"
            f"{proof_note}\n\n"
            f"Required content: {obligations}.\n"
            f"Route transport state: {route_status}.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Candidate A:\n{first_text}\n\n"
            f"Candidate B:\n{second_text}\n\n"
            f"Certified local evidence:\n{self._evidence_prompt(evidence) or 'none'}"
        )

    @staticmethod
    def _mog_route_system_prompt(spec: ProblemSpec) -> str:
        """Give route search one output protocol without global FINAL conflicts."""
        language = "Chinese" if spec.profile.language == "zh" else "English"
        return (
            "You are a bounded mathematical route solver. Work through the assigned route "
            "privately, then follow the exact labelled-section protocol and section order "
            "in the user message. That protocol is the only output-format authority for "
            "this call. Do not restate the problem, discuss prompts or formatting, list "
            "abandoned approaches, or repeat a section. Commit to FINAL only after the "
            "labelled CHECK. If the route is blocked, identify one exact unresolved "
            "mathematical obligation in REMAINING. Use " + language + "."
        )

    @staticmethod
    def _mog_reproducible_check(value: str) -> bool:
        text = str(value or "").strip()
        if len(text) < 20 or re.search(
            r"\b(?:plausible|consistent with|given the constraints|nature of the problem|"
            r"without further (?:specific )?evidence|appears? correct|seems? correct|"
            r"strategically placed|standard result)\b|"
            r"看起来|似乎正确|符合题意|根据题目性质|显然正确|没有进一步证据",
            text,
            re.IGNORECASE,
        ):
            return False
        executable_relation = bool(re.search(
            r"(?:[-+]?\d|[A-Za-z]_[A-Za-z0-9{}]+|\\(?:frac|sum|int|lim|det))"
            r"[^\n。；;]{0,180}(?:=|<=|>=|<|>|≤|≥|\\equiv|\\mid)"
            r"[^\n。；;]{1,180}(?:[-+]?\d|[A-Za-z]|\\)",
            text,
        ))
        explicit_attack = bool(re.search(
            r"反例|取\s*[^，。;]{1,60}(?:时|则)|代入|余数|模\s*\d|残差|枚举|"
            r"边界|端点|不满足[^。；;]{0,80}(?:条件|假设)|"
            r"\b(?:counterexample|substitut(?:e|ion)|remainder|modulo|residual|"
            r"enumerat(?:e|ion)|boundary|endpoint|fails?\s+(?:the\s+)?hypothesis)\b",
            text,
            re.IGNORECASE,
        ))
        theorem_audit = bool(re.search(
            r"(?:定理|theorem)[^。；;\n]{0,120}(?:条件|假设|hypothes|requires?)"
            r"[^。；;\n]{0,160}(?:满足|成立|不满足|holds?|fails?|verified)",
            text,
            re.IGNORECASE,
        ))
        return executable_relation or explicit_attack or theorem_audit

    def _call(
        self,
        request: str,
        *,
        stage: str,
        max_tokens: int,
        temperature: float,
        thinking_mode: bool,
        trace: list[dict],
        prior_response: str = "",
        followup: str = "",
        system_prompt: str = "",
        model_tools: ModelMathTools | None = None,
        model_tool_names: tuple[str, ...] = (),
        max_tool_rounds: int = 0,
        require_model_tool: bool = False,
        tool_followup_mode: str = "compact_answer",
    ) -> tuple[str, ModelCallResult]:
        messages = [
            {"role": "system", "content": system_prompt or self.prompt},
            {"role": "user", "content": request},
        ]
        if str(prior_response or "").strip():
            messages.extend((
                {"role": "assistant", "content": str(prior_response)},
                {"role": "user", "content": str(followup)},
            ))
        chat_result = getattr(self.client, "chat_result", None)
        result_call = chat_result if callable(chat_result) else self.client.chat
        message_call = getattr(self.client, "chat", result_call)
        tool_enabled = bool(
            model_tools is not None
            and max_tool_rounds > 0
            and self._supports_keyword(message_call, "tools")
        )
        request_count = 0
        tool_round = 0
        tool_outcomes: list[dict] = []
        final_result = ModelCallResult("")

        while True:
            # The documented ``chat`` API preserves assistant tool-call
            # messages. Some ``chat_result`` adapters intentionally expose
            # only text and transport metadata, which can discard a tool call
            # when the provider also emits nonempty reasoning content.
            call = message_call if tool_enabled and tool_round == 0 else result_call
            current_tokens = (
                max(256, min(int(max_tokens), 4096))
                if tool_round
                else max(256, int(max_tokens))
            )
            kwargs = {
                "messages": messages,
                "temperature": 0.1 if tool_round else temperature,
                "max_tokens": current_tokens,
            }
            if self._supports_keyword(call, "thinking_mode"):
                kwargs["thinking_mode"] = bool(thinking_mode and not tool_round)
            # Offer functions only on the first request. Once a local result
            # is present, the follow-up must synthesize FINAL instead of
            # repeatedly requesting more computations and exhausting the
            # stage budget without an answer.
            if tool_enabled and tool_round == 0:
                kwargs["tools"] = model_tools.schemas(model_tool_names or None)
                if require_model_tool and self._supports_keyword(call, "tool_choice"):
                    kwargs["tool_choice"] = "required"

            started = monotonic()
            failure_type = ""
            raw_response = None
            try:
                raw_response = call(**kwargs)
                result = coerce_model_response(raw_response)
                status = "ok" if result.content.strip() else "empty"
            except BaseException as error:
                if not is_recoverable_runtime_failure(error):
                    raise
                result = ModelCallResult("")
                status = "failed"
                failure_type = type(error).__name__
            request_count += 1

            tool_message = (
                self._tool_call_message(raw_response, result)
                if tool_enabled and status != "failed"
                else None
            )
            has_tool_calls = bool(
                tool_message
                and isinstance(tool_message.get("tool_calls"), list)
                and tool_message["tool_calls"]
            )
            if has_tool_calls:
                status = (
                    "tool_call"
                    if tool_round < max_tool_rounds
                    else "tool_round_limit"
                )

            content = {
                "stage": stage if not tool_round else f"{stage}_tool_followup",
                "status": status,
                "max_tokens": current_tokens,
                "thinking_mode": kwargs.get("thinking_mode", "client_default"),
                "finish_reason": result.finish_reason or "unavailable",
                "provider_truncated": result.provider_truncated,
                "truncated": self._truncated(result, result.content),
                "output_length": len(result.content),
                "usage": {
                    key: value
                    for key, value in result.usage.items()
                    if key in {
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "input_tokens", "output_tokens",
                    }
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                },
                "elapsed_ms": round((monotonic() - started) * 1000),
            }
            if failure_type:
                content["failure_type"] = failure_type
            trace.append({"step": "model_call", "content": content})

            if not has_tool_calls or tool_round >= max_tool_rounds:
                final_result = result if not has_tool_calls else ModelCallResult("")
                break

            assistant_message = dict(tool_message)
            assistant_message.setdefault("role", "assistant")
            messages.append(assistant_message)
            executions = []
            for call_index, tool_call in enumerate(tool_message["tool_calls"][:3]):
                if not isinstance(tool_call, Mapping):
                    continue
                execution = model_tools.execute_call(tool_call)
                executions.append(execution)
                tool_outcomes.append({
                    "operation": execution.name,
                    "ok": execution.ok,
                    "contract_valid": execution.ok,
                    "result": execution.result if execution.ok else "",
                    "local_certificate_status": (
                        execution.local_certificate_status.value
                    ),
                    "certificate_generated": bool(
                        execution.local_certificate_status.value
                        == "CERTIFIED_TRUE"
                    ),
                    "problem_goal_status": "NOT_CERTIFIED",
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": str(
                        tool_call.get("id", f"tool-{tool_round}-{call_index}")
                    ),
                    "name": execution.name,
                    "content": execution.message_content(),
                })
            trace.append({
                "step": "local_math_tool",
                "content": {
                    "round": tool_round + 1,
                    "requested": len(tool_message["tool_calls"][:3]),
                    "executed": len(executions),
                    "operations": [item.name for item in executions],
                    "successful": sum(item.ok for item in executions),
                    "local_certificate_statuses": [
                        item.local_certificate_status.value for item in executions
                    ],
                    "problem_goal_status": "NOT_CERTIFIED",
                },
            })
            if not executions:
                final_result = ModelCallResult("")
                break
            reasoned_followup = (
                "Treat the returned value only as a certified local fact about the submitted "
                "operation. Verify that you derived that exact operation from the statement "
                "and that its domain includes every endpoint, zero case, label, repetition, "
                "and quantifier. Discard it if the correspondence fails. Do not call another "
                "tool or restart broad method search. Then write a complete gradable solution: "
                "the first line starts with FINAL:, followed by the conclusion, and the "
                "remaining lines give the shortest necessary proof covering every obligation. "
                "The local fact is not a proof of the original goal.\n"
                "本地结果只认证已提交的局部运算，不认证题意翻译或全题。先逐项核对定义域、"
                "端点、零情形、标签、重复与量词；不一致就丢弃。禁止再次调用工具或重启广泛"
                "搜索。第一行以 FINAL: 给出结论，随后写覆盖全部义务的最短完整论证。"
            )
            compact_followup = (
                "Treat the returned value only as the result of the submitted operation. "
                "Do not call another tool, restart the derivation, or narrate analysis. "
                "Verify that you derived that exact operation from the statement and that "
                "its domain includes the correct zero cases, endpoints, leading zeros, "
                "labels, repetition, and exact/at-most/at-least conditions. Discard the "
                "tool value if this correspondence fails. Then "
                "respond with exactly two lines:\n"
                "FINAL: the complete gradable answer\n"
                "CHECK: one concise domain or substitution check\n"
                "本地值只证明已提交的运算；不得再次调用工具或重启推导。先核对该"
                "运算确由题意推出且定义域完全一致，不一致就丢弃，再严格输出上述两行。"
            )
            messages.append({
                "role": "user",
                "content": (
                    reasoned_followup
                    if tool_followup_mode == "reasoned_solution"
                    else compact_followup
                ),
            })
            tool_round += 1

        usage = dict(final_result.usage)
        usage["_agent_request_count"] = request_count
        usage["_model_tool_outcomes"] = tuple(tool_outcomes)
        if tool_outcomes:
            trace.append({
                "step": "model_tool_summary",
                "content": {
                    "attempted": len(tool_outcomes),
                    "contract_valid": sum(
                        bool(item.get("contract_valid")) for item in tool_outcomes
                    ),
                    "certificate_generated": sum(
                        bool(item.get("certificate_generated"))
                        for item in tool_outcomes
                    ),
                    "certificate_supplied_to_followup": True,
                    "followup_completed": bool(final_result.content.strip()),
                    "certificate_used": "not_observable_without_answer_text",
                    "problem_goal_status": "NOT_CERTIFIED",
                },
            })
        final_result = ModelCallResult(
            final_result.content,
            final_result.finish_reason,
            usage,
        )
        return final_result.content.strip(), final_result

    @staticmethod
    def _tool_call_message(raw_response, result: ModelCallResult) -> dict | None:
        candidates = []
        if isinstance(raw_response, Mapping):
            candidates.append(raw_response)
            choices = raw_response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    message = choice.get("message")
                    if isinstance(message, Mapping):
                        candidates.append(message)
        text = str(result.content or "").strip()
        if text.startswith("{") and len(text) <= 20_000:
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, dict):
                candidates.append(decoded)
        for candidate in candidates:
            calls = candidate.get("tool_calls")
            if isinstance(calls, list) and calls:
                return dict(candidate)
        return None

    @staticmethod
    def _request_count(result: ModelCallResult) -> int:
        value = result.usage.get("_agent_request_count", 1)
        return value if isinstance(value, int) and 1 <= value <= 4 else 1

    @staticmethod
    def _model_tool_evidence(result: ModelCallResult) -> tuple[ToolEvidence, ...]:
        outcomes = result.usage.get("_model_tool_outcomes", ())
        if not isinstance(outcomes, (tuple, list)):
            return ()
        evidence = []
        for outcome in outcomes[:3]:
            if not isinstance(outcome, Mapping) or not outcome.get("ok"):
                continue
            operation = str(outcome.get("operation", ""))[:80]
            value = str(outcome.get("result", ""))[:4_000]
            if not operation or not value:
                continue
            evidence.append(ToolEvidence(
                result=value,
                scope="advisory",
                operation=operation,
                verified=False,
                certificate_method="model_translated_local_recomputation",
                certificate_checks=(
                    "submitted_operation_locally_certified",
                ),
                certificate_issues=(
                    "problem_translation_not_certified",
                    "whole_goal_not_certified",
                ),
                support=(
                    "Local recomputation of the model-submitted operation returned: "
                    + value
                ),
                assurance="model_translated",
            ))
        return tuple(evidence)

    def _assess_response(
        self,
        raw: str,
        *,
        source: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        method_id: str,
        independence_group: str,
    ) -> list[CandidateAssessment]:
        raw_text = str(raw or "").strip()
        located_tool = self.operation_locator.result_from_response(
            spec.problem_text,
            raw_text,
            spec,
        )
        effective_evidence = evidence
        if located_tool is not None:
            located_tool = self._with_tool_assurance(
                (located_tool,), "symbolic"
            )[0]
            if (
                located_tool.certificate.source_fingerprint
                == problem_fingerprint(spec.problem_text)
                and self._certifies_goal_result(located_tool, spec)
            ):
                effective_evidence = tuple((
                    *evidence,
                    *self._tool_evidence((located_tool,), spec),
                ))
            else:
                located_tool = None
        text = self.operation_locator.strip_blocks(raw_text)
        text = self.structured_verifier.strip_certificates(text)
        raw_consistency_reasons = candidate_consistency_reasons(text, spec)
        assessments: list[CandidateAssessment] = []
        seen: set[str] = set()
        if located_tool is not None:
            local_candidate = self._assess_value(
                located_tool.result,
                source=f"{source}_local_certificate",
                spec=spec,
                evidence=effective_evidence,
                extraction_method="declarative_tool_certificate",
                explicit=True,
                method_id=located_tool.operation,
                independence_group="local_tool",
                certified_tool=located_tool,
            )
            assessments.append(local_candidate)
            seen.add(local_candidate.answer)
        if not text:
            return assessments
        support_mode = spec.answer_contract.mode != "answer_only"
        extracted: list[ExtractionResult] = []
        if support_mode:
            # Proof/derivation grading needs the argument that may appear
            # before a trailing answer label.  Always retain a structurally
            # complete whole response; labelled conclusions remain separate
            # degraded fallbacks for a truncated or contaminated body.
            if not Finalizer.contains_meta(text):
                cleaned = Finalizer.extract_solution(text)
                reasons = Finalizer.validate_structure(cleaned)
                if cleaned:
                    extracted.append(ExtractionResult(
                        cleaned,
                        "whole_solution",
                        not reasons,
                        reasons,
                        False,
                        False,
                    ))
            for block in Finalizer.extract_tagged_submissions(text):
                extracted.append(ExtractionResult(
                    block,
                    "tagged_solution",
                    True,
                    (),
                    Finalizer.contains_meta(text),
                    True,
                ))
            if Finalizer.contains_meta(text):
                for block in Finalizer.extract_terminal_supported_submissions(text):
                    extracted.append(ExtractionResult(
                        block,
                        "terminal_supported_solution",
                        True,
                        (),
                        True,
                        True,
                    ))
            # A complete explicit conclusion remains a useful last resort if
            # the surrounding proof is truncated or contaminated by meta text.
            # It is assessed separately and can never outrank a complete proof.
            extracted.extend(Finalizer.extract_explicit_results(text))
        else:
            extracted.extend(Finalizer.extract_explicit_results(text))
            if not extracted:
                extracted.append(Finalizer.extract_result(text))

        for item in extracted:
            value = self._normalize_candidate(item.answer, spec)
            if not value or value in seen:
                continue
            seen.add(value)
            symbolic_checks = tuple(
                CheckResult(
                    check.name,
                    check.status,
                    "sympy",
                    check.detail,
                    check.decisive,
                )
                for check in self.sympy.verify_candidate(spec.problem_text, value, spec)
            )
            structured_checks = tuple(
                CheckResult(
                    check.name,
                    check.status,
                    "structured_local",
                    check.detail,
                    check.decisive,
                )
                for check in self.structured_verifier.verify_response(
                    spec.problem_text,
                    raw_text,
                    value,
                )
            )
            checks = (*symbolic_checks, *structured_checks)
            assessments.append(assess_candidate(
                value,
                source,
                spec,
                effective_evidence,
                extraction_method=item.method,
                extraction_reasons=tuple(dict.fromkeys((
                    *item.rejected_reasons,
                    *raw_consistency_reasons,
                ))),
                raw_has_meta=item.raw_has_meta,
                explicit_answer=item.explicit_answer,
                method_id=method_id,
                independence_group=independence_group,
                math_checks=checks,
            ))
        return assessments

    def _assess_value(
        self,
        value: str,
        *,
        source: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        extraction_method: str,
        explicit: bool,
        method_id: str,
        independence_group: str,
        extra_checks: tuple[CheckResult, ...] = (),
        certified_tool: ToolResult | None = None,
    ) -> CandidateAssessment:
        normalized = self._normalize_candidate(value, spec)
        if certified_tool is not None and certified_tool.verified:
            checks = (
                CheckResult(
                    f"certificate_{certified_tool.operation}",
                    "pass",
                    "local_tool",
                    certified_tool.certificate.method,
                    certified_tool.direct_submission_eligible,
                ),
            )
        else:
            checks = tuple(
                CheckResult(
                    check.name,
                    check.status,
                    "sympy",
                    check.detail,
                    check.decisive,
                )
                for check in self.sympy.verify_candidate(spec.problem_text, normalized, spec)
            )
        checks += tuple(extra_checks)
        return assess_candidate(
            normalized,
            source,
            spec,
            evidence,
            extraction_method=extraction_method,
            explicit_answer=explicit,
            method_id=method_id,
            independence_group=independence_group,
            math_checks=checks,
        )

    @staticmethod
    def _with_tool_assurance(
        results: Iterable[ToolResult],
        assurance: str,
    ) -> tuple[ToolResult, ...]:
        """Attach producer-level trust without letting a matcher self-certify it."""
        annotated: list[ToolResult] = []
        for result in results:
            contract = result.contract
            if contract is None:
                annotated.append(result)
                continue
            annotated.append(replace(
                result,
                contract=replace(contract, assurance=assurance),
            ))
        return tuple(annotated)

    @staticmethod
    def _whole_tool_result(
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
    ) -> ToolResult | None:
        statement_parameterized = any(
            result.operation in _PARAMETERIZED_STATEMENT_OPERATIONS
            for result in results
        )
        if not spec.tool_can_answer_whole and not statement_parameterized:
            return None
        if any(
            requirement.strict and requirement.category == "support"
            for goal in spec.goals
            for requirement in goal.requirements
        ):
            return None
        strict_requirements = tuple(
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
            if requirement.strict
        )
        eligible = []
        for result in results:
            contract = result.contract
            if (
                not spec.tool_can_answer_whole
                and result.operation not in _PARAMETERIZED_STATEMENT_OPERATIONS
            ):
                continue
            if not result.direct_submission_eligible or contract is None:
                continue
            if result.certificate.source_fingerprint != problem_fingerprint(spec.problem_text):
                continue
            if contract.covers(
                len(spec.goals),
                strict_requirements,
                task_kind=spec.profile.task_kind,
                answer_shape=spec.profile.answer_shape,
                problem_facts=(),
            ):
                eligible.append(result)
        return SubmissionAgent._coalesce_equivalent_tools(eligible)

    @staticmethod
    def _coalesce_equivalent_tools(
        results: Iterable[ToolResult],
    ) -> ToolResult | None:
        eligible = list(results)
        if not eligible:
            return None
        first = eligible[0].result
        if any(not equivalent_answers(first, item.result) for item in eligible[1:]):
            return None
        return max(
            eligible,
            key=lambda item: (
                item.supported_submission_eligible,
                len(item.certificate.checks),
                len(item.support),
                len(item.result),
            ),
        )

    def _supported_tool_candidate(
        self,
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> CandidateAssessment | None:
        fingerprint = problem_fingerprint(spec.problem_text)
        eligible = [
            result
            for result in results
            if result.supported_submission_eligible
            and result.direct_submission_eligible
            and result.certificate.source_fingerprint == fingerprint
            and self._certifies_goal_result(result, spec)
        ]
        result = self._coalesce_equivalent_tools(eligible)
        if result is None:
            return None
        conclusion = result.result
        if spec.answer_contract.wrapper == "boxed":
            conclusion = rf"\boxed{{{conclusion}}}"
        candidate_value = (
            conclusion
            if spec.answer_contract.mode == "answer_only"
            else (
                f"FINAL: {conclusion}"
                + (
                    f"\n{result.support}"
                    if result.support.strip() != result.result.strip()
                    else ""
                )
            )
        )
        candidate = self._assess_value(
            candidate_value,
            source="certified_goal_result",
            spec=spec,
            evidence=evidence,
            extraction_method="certified_tool_with_support",
            explicit=True,
            method_id=result.operation,
            independence_group="local_tool",
            certified_tool=result,
        )
        if candidate.validation_tier != "complete" or candidate.tool_status != "pass":
            return None
        return candidate

    def _certified_goal_fallback(
        self,
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> CandidateAssessment | None:
        """Prefer a certified conclusion with partial support over an unrelated shape sentinel."""
        fingerprint = problem_fingerprint(spec.problem_text)
        eligible = [
            result for result in results
            if result.verified
            and result.goal_result_eligible
            and result.certificate.source_fingerprint == fingerprint
            and self._certifies_goal_result(result, spec)
            and result.support
        ]
        result = self._coalesce_equivalent_tools(eligible)
        if result is None:
            return None
        conclusion = result.result
        if spec.answer_contract.wrapper == "boxed":
            conclusion = rf"\boxed{{{conclusion}}}"
        candidate_value = (
            conclusion
            if spec.answer_contract.mode == "answer_only"
            else (
                f"FINAL: {conclusion}"
                + (
                    f"\n{result.support}"
                    if result.support.strip() != result.result.strip()
                    else ""
                )
            )
        )
        candidate = self._assess_value(
            candidate_value,
            source="certified_goal_fallback",
            spec=spec,
            evidence=evidence,
            extraction_method="certified_goal_fallback",
            explicit=True,
            method_id=result.operation,
            independence_group="local_tool",
            certified_tool=result,
        )
        direct = result.direct_submission_eligible
        if (
            candidate.validation_tier not in {"complete", "degraded"}
            or (
                direct and candidate.tool_status != "pass"
            )
            or (
                not direct
                and candidate.tool_status not in {"unknown", "partial_pass", "pass"}
            )
            or candidate.failed_check
            or not candidate.result_coverage
            or not all(candidate.result_coverage)
        ):
            return None
        return candidate

    @staticmethod
    def _tool_evidence(
        results: tuple[ToolResult, ...],
        spec: ProblemSpec,
        whole: ToolResult | None = None,
    ) -> tuple[ToolEvidence, ...]:
        fingerprint = problem_fingerprint(spec.problem_text)
        return tuple(
            ToolEvidence(
                result=result.result,
                scope=(
                    "whole_goal"
                    if result is whole and result.direct_submission_eligible
                    else "goal_result"
                    if (
                        result.direct_submission_eligible
                        and SubmissionAgent._certifies_goal_result(result, spec)
                    )
                    else "advisory"
                    if SubmissionAgent._certifies_goal_result(result, spec)
                    else "subexpression"
                ),
                operation=result.operation,
                verified=bool(
                    result.verified
                    and result.certificate.source_fingerprint == fingerprint
                ),
                certificate_method=result.certificate.method,
                certificate_checks=result.certificate.checks,
                certificate_issues=result.certificate.issues,
                support=result.support,
                assurance=(
                    result.contract.assurance if result.contract else "schema"
                ),
            )
            for result in results
        )

    @staticmethod
    def _certifies_goal_result(result: ToolResult, spec: ProblemSpec) -> bool:
        contract = result.contract
        if not result.goal_result_eligible or contract is None:
            return False
        requested = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.result_requirements
            if requirement.strict
        }
        return bool(
            len(spec.goals) == 1
            and "residual_output_contract" not in spec.risk_flags
            and len(spec.goals) <= contract.max_goals
            and (
                spec.profile.task_kind in contract.allowed_task_kinds
                or result.supported_submission_eligible
            )
            and set(contract.required_requirements) <= requested
            and requested <= (
                set(contract.allowed_requirements)
                | set(GENERIC_PRESENTATION_REQUIREMENTS)
            )
            and (
                not contract.allowed_answer_shapes
                or spec.profile.answer_shape in contract.allowed_answer_shapes
            )
        )

    @staticmethod
    def _planning_request(
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "complete requested result"
        verified_operations = ", ".join(
            item.operation for item in evidence if item.verified
        ) or "none"
        return (
            f"Problem:\n{problem}\n\n"
            f"Locally extracted task kind: {spec.profile.task_kind}.\n"
            f"Locally extracted answer obligations: {obligations}.\n"
            f"Available certified local operation types: {verified_operations}.\n"
            "Plan the solution route only. Do not calculate or state the final answer."
        )

    @staticmethod
    def _tool_location_request(problem: str, spec: ProblemSpec) -> str:
        return (
            f"Problem:\n{problem}\n\n"
            f"Locally classified task kind: {spec.profile.task_kind}.\n"
            f"Locally classified answer shape: {spec.profile.answer_shape}.\n"
            "Locate a supported whole-result operation only under the system protocol."
        )

    @staticmethod
    def _should_locate_tool(problem: str, spec: ProblemSpec) -> bool:
        if len(spec.goals) != 1:
            return False
        if str(getattr(spec.profile, "topic", "")).startswith("olympiad_"):
            return False
        if len(str(problem or "")) > 900:
            return False
        if spec.profile.task_kind not in {"calculation", "fill_blank"}:
            return False
        if spec.profile.answer_shape not in {
            "number", "expression", "roots", "count", "matrix"
        }:
            return False
        if "residual_output_contract" in spec.risk_flags:
            return False
        return bool(re.search(
            r"\$|\\\(|\\\[|\\(?:frac|sqrt|sum|int|lim|begin\{(?:cases|[pbvBV]?matrix)\})|"
            r"(?<![<>!])=(?!=)",
            str(problem or ""),
        ))

    @staticmethod
    def _should_request_structured_verification(
        problem: str,
        spec: ProblemSpec,
    ) -> bool:
        """Request a declarative check only for bounded algebra/calculus claims."""
        if (
            spec.answer_contract.mode != "answer_only"
            or spec.profile.task_kind not in {"calculation", "fill_blank"}
            or spec.profile.answer_shape not in {
                "number", "expression", "roots", "probability"
            }
            or len(str(problem or "")) > 1_200
        ):
            return False
        return bool(re.search(
            r"(?<![<>!])=(?!=)|\\(?:frac|sqrt|sum|int|lim)|"
            r"(?:\d|\})\s*(?:[+*/^]|-(?!>))\s*(?:\d|\{|\\?[A-Za-z])|"
            r"\b(?:derivative|differentiat|integral|antiderivative|roots?|"
            r"substitut|identity)\b|导数|求导|积分|方程|恒等式|代入",
            str(problem or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def _model_tools_allowed(problem: str, spec: ProblemSpec) -> bool:
        """Whether at least one focused local operation applies to this task."""
        return bool(SubmissionAgent._model_tool_names(problem, spec))

    @staticmethod
    def _model_tool_names(
        problem: str,
        spec: ProblemSpec,
        *,
        allow_complex_subproblem: bool = False,
        complex_opportunity: LocalToolOpportunity | None = None,
    ) -> tuple[str, ...]:
        """Expose a small subject-focused whitelist, never the whole tool catalog."""
        task = spec.profile.task_kind
        shape = spec.profile.answer_shape
        text = str(problem or "")
        topic = str(getattr(spec.profile, "topic", ""))
        complex_semantics = bool(
            topic.startswith("olympiad_")
            or spec.risk_score > 2
            or len(text) > 1_200
            or re.search(
                r"证明|推导|说明|解释|比较|构造|最大|最小|所有|"
                r"\b(?:prove|derive|explain|justify|compare|construct|"
                r"maximum|minimum|largest|smallest|find\s+all|determine\s+all)\b",
                text,
                re.IGNORECASE,
            )
        )
        opportunity = complex_opportunity or detect_local_tool_opportunity(
            text, spec
        )
        if (
            opportunity.eligible
            and opportunity.scope == "statement_exact"
            and set(opportunity.allowed_tools) <= _AUTO_EXACT_MODEL_TOOLS
        ):
            return opportunity.allowed_tools
        if complex_semantics:
            if not allow_complex_subproblem:
                return ()
            # A complex statement may see only the exact operation family
            # approved by the conservative local-opportunity contract. The
            # result remains advisory and cannot become a whole-goal route.
            return opportunity.allowed_tools if opportunity.eligible else ()
        if (
            spec.answer_contract.mode != "answer_only"
            or task not in {"calculation", "fill_blank"}
            or shape not in {
                "number", "expression", "roots", "count", "probability", "matrix"
            }
            or not 1 <= len(text) <= 5_000
        ):
            return ()
        names: list[str] = []

        def add(*items: str) -> None:
            for item in items:
                if item not in names:
                    names.append(item)

        equation = bool(re.search(
            r"(?<![<>!])=(?!=)|方程|方程组|根|零点|"
            r"\b(?:equations?|systems?\s+of\s+equations?|roots?|zeros?)\b",
            text,
            re.IGNORECASE,
        ))
        polynomial = bool(re.search(
            r"多项式|因式分解|展开|化简|恒等式|"
            r"\b(?:polynomial|factor(?:ize|ise|ization|isation)?|expand|simplify|identity)\b",
            text,
            re.IGNORECASE,
        ))
        coefficient = bool(re.search(
            r"系数|生成函数|母函数|\[x\^|"
            r"\b(?:coefficient|generating\s+function)\b",
            text,
            re.IGNORECASE,
        ))
        substitution = bool(re.search(
            r"代入|代换|在.+处(?:的值|取值)|"
            r"\b(?:substitut|evaluate\s+at)\w*\b",
            text,
            re.IGNORECASE,
        ))
        derivative = bool(re.search(
            r"导数|求导|微分|\\frac\s*\{d|"
            r"\b(?:derivative|differentiat)\w*\b",
            text,
            re.IGNORECASE,
        ))
        integral = bool(re.search(
            r"积分|\\int|\b(?:integral|antiderivative)\b",
            text,
            re.IGNORECASE,
        ))
        quadrature = bool(re.search(
            r"数值积分|求积|复化(?:中点|梯形|辛普森)|"
            r"\b(?:quadrature|composite\s+(?:midpoint|trapezoid|simpson)|"
            r"gauss[- ]legendre)\b",
            text,
            re.IGNORECASE,
        ))
        limit = bool(re.search(
            r"极限|\\lim|\blimit\b",
            text,
            re.IGNORECASE,
        ))
        finite_sum = bool(re.search(
            r"求和|\\sum|\b(?:summation|finite\s+sum)\b",
            text,
            re.IGNORECASE,
        ))
        matrix = bool(re.search(
            r"矩阵|行列式|秩|特征值|逆矩阵|线性方程组|"
            r"\b(?:matrix|determinant|rank|eigenvalues?|inverse|linear\s+system)\b",
            text,
            re.IGNORECASE,
        ))
        recurrence = bool(re.search(
            r"递推|数列|序列|通项|(?<![A-Za-z])a\s*_\s*\{?n\}?(?![A-Za-z])|"
            r"\b(?:recurrence|sequence|initial\s+values?|nth\s+term)\b",
            text,
            re.IGNORECASE,
        ))
        finite_integer = bool(re.search(
            r"整数|自然数|非负整数|正整数|有序(?:对|组)|"
            r"\b(?:integers?|natural\s+numbers?|ordered\s+(?:pairs?|tuples?)|"
            r"nonnegative\s+integers?|positive\s+integers?)\b",
            text,
            re.IGNORECASE,
        ))
        digit_count = bool(
            re.search(r"\b(?:digits?|decimal|base[- ]?10)\b|十进制|数位|数字", text, re.IGNORECASE)
            and re.search(r"\b(?:at\s+most|exactly|length|digits?\s+long|divisible)\b|至多|恰好|位数|位|整除", text, re.IGNORECASE)
        )
        modular_count = bool(
            re.search(r"\b(?:residue(?:s|\s+classes)?|modulo|congruence)\b|剩余类|模|同余", text, re.IGNORECASE)
            and re.search(r"\b(?:count|number\s+of|how\s+many|find\s+all)\b|个数|数目|多少|所有", text, re.IGNORECASE)
        )
        explicit_arithmetic = bool(re.search(
            r"\b(?:calculate|compute|evaluate|numerical\s+value)\b|计算|求值|数值|"
            r"(?:\d|\})\s*(?:[+*/^]|-(?!>))\s*(?:\d|\{)",
            text,
            re.IGNORECASE,
        ))

        # Put domain-specific operations first.  The provider is more likely
        # to select a correct function when it does not have to search through
        # a broad, mostly irrelevant catalog.
        if matrix:
            add("matrix_operation")
            if re.search(r"线性方程组|\blinear\s+system\b", text, re.IGNORECASE):
                add("solve_polynomial_system")
        if digit_count:
            add("count_digit_strings")
        if modular_count:
            add("count_modular_solutions")
        if finite_integer:
            add("bounded_integer_search")
        if recurrence:
            add("linear_recurrence_term")
        if integral and not quadrature:
            add("definite_integral")
        if limit:
            add("limit_expression")
        if derivative:
            add("differentiate_expression")
        if finite_sum:
            add("finite_sum")
        if coefficient:
            add("polynomial_coefficient")
        if equation:
            add("solve_equation")
            if re.search(
                r"方程组|联立|\b(?:system\s+of\s+equations?|simultaneous)\b",
                text,
                re.IGNORECASE,
            ):
                add("solve_polynomial_system")
        if polynomial:
            add("simplify_expression")
        if substitution:
            add("substitute_values")
        if not complex_semantics and explicit_arithmetic:
            add("calculate_expression")
        # Too many irrelevant schemas reduce tool-selection reliability.  The
        # order above keeps the most direct operations first.
        return tuple(names[:8])

    @staticmethod
    def _model_tool_required(problem: str, spec: ProblemSpec) -> bool:
        """Force a tool only for a short, direct, low-risk computation."""
        names = SubmissionAgent._model_tool_names(problem, spec)
        text = str(problem or "")
        if not names or len(text) > 800 or spec.risk_score > 2:
            return False
        if str(getattr(spec.profile, "topic", "")).startswith("olympiad_"):
            return False
        if re.search(
            r"证明|推导|说明|解释|比较|构造|最大|最小|所有|使用.+法|利用.+法|"
            r"\b(?:prove|derive|explain|justify|compare|using?\s+the\s+.+method)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        return bool(re.search(
            r"^(?:\s*\d+[.、)]\s*)?(?:计算|求值|求解|解方程|求导|求积分|求极限)|"
            r"\b(?:calculate|compute|evaluate|solve\s+(?:the\s+)?equation|"
            r"differentiate|find\s+(?:the\s+)?(?:determinant|rank|inverse|limit))\b",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _model_tool_instruction(
        spec: ProblemSpec,
        tool_names: tuple[str, ...] = (),
        *,
        opportunity: LocalToolOpportunity | None = None,
    ) -> str:
        offered = ", ".join(tool_names)
        local_kind = (
            opportunity.kind.value
            if opportunity is not None and opportunity.eligible
            else "DIRECT_COMPUTATION"
        )
        if spec.profile.language == "zh":
            return (
                "\n\n本题只开放与已检测局部机会相符的受限数学函数。只有当你能先"
                "独立写出变量、有限定义域、全部约束和精确查询时，才调用最关键的一个"
                "函数。复杂题只把函数用于你已经推导出的子问题；工具只"
                "核验提交的运算，不核验题意翻译、公式推导或穷尽性。收到结果后必须"
                "重新核对变量域、边界和计数是否一一对应，再输出 FINAL；若不对应就"
                "丢弃工具结果。若没有适用函数，直接求解，禁止为了调用而改写题目。"
                f"局部机会类别：{local_kind}。当前函数：{offered}。"
            )
        return (
            "\n\nOnly restricted mathematics functions matching a detected local "
            "opportunity are available. Call at most one only after independently specifying "
            "the variables, finite domain, every constraint, and exact query. In a complex "
            "problem, use it only for a "
            "subproblem you have already derived. A tool certifies only the submitted "
            "operation, not the translation, derivation, or exhaustiveness. Recheck the exact "
            "domain, boundaries, and counting correspondence before emitting FINAL, and "
            "discard a result whose submitted operation does not match. If no function "
            "applies, solve directly and never distort the problem merely to make a call. "
            f"Local opportunity class: {local_kind}. Available functions: {offered}."
        )

    @staticmethod
    def _primary_comprehension_check(spec: ProblemSpec) -> str:
        """Use a private second-language pass to catch statement-reading errors."""
        if spec.profile.language == "en":
            return (
                "Before solving, privately restate the mathematical content in concise "
                "Chinese and compare it clause by clause with the original English. "
                "Preserve every coefficient, quantifier, domain, endpoint, and game rule. "
                "The English statement remains authoritative. Do not print or discuss the "
                "translation; use it only as a comprehension check."
            )
        return (
            "Before solving, privately map only ambiguous specialized terms and named "
            "theorems to their standard English names, then verify that mapping against "
            "the original Chinese statement. Preserve every coefficient, quantifier, "
            "domain, endpoint, and rule. Do not print or discuss the translation."
        )

    @staticmethod
    def _critic_system_prompt(spec: ProblemSpec) -> str:
        """Keep audit stages short and separate from the full-solver role."""
        language = "Chinese" if spec.profile.language == "zh" else "English"
        return (
            "You are a strict mathematical Critic, not a fresh long-form solver. "
            "Follow the exact labelled output protocol in the user message immediately. "
            "Do not emit hidden reasoning, a plan, a restatement of the problem, draft "
            "commentary, formatting commentary, or abandoned approaches. Perform only one "
            "decisive reproducible check. If a correction is required, give the complete "
            "gradable replacement (including a concise proof when requested); otherwise "
            "preserve the audited candidate. Use " + language + "."
        )

    def _primary_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        *,
        plan: SolvePlan | None = None,
        deep_reasoning: bool = False,
        hidden_thinking: bool | None = None,
    ) -> str:
        hidden_thinking = deep_reasoning if hidden_thinking is None else hidden_thinking
        if deep_reasoning and hidden_thinking:
            role = (
                "DEEP_SOLVE stage: solve from first principles. Complete the "
                "derivation and a falsification check in hidden reasoning before "
                "emitting text. The first visible line must be exactly one FINAL "
                "line containing the checked complete answer; then give only the "
                "essential support required for grading."
            )
        elif deep_reasoning:
            role = (
                "COMPACT_DEEP_SOLVE stage: solve from first principles using one "
                "coherent route and one falsification check. Put the complete checked "
                "answer on the first FINAL line, then give only the shortest decisive "
                "support. Do not restart or enumerate abandoned approaches."
            )
        else:
            role = (
                "QUICK_SOLVE stage: solve from first principles and put the complete "
                "checked answer on the first FINAL line."
            )
        compact_directive = ""
        comprehension_check = self._primary_comprehension_check(spec)
        if self.compact_primary_prompt and deep_reasoning:
            comprehension_check = (
                "Check coefficients, domains, endpoints, and quantifiers directly against "
                "the original statement. Do not restate or translate the problem."
            )
            compact_directive = (
                " HARD BUDGET: Commit to one viable route early and spend the response on "
                "its hardest unresolved lemma. Do not narrate failed approaches, repeat "
                "definitions, restate the problem, or polish prose. Switch route at most "
                "once and do not describe the abandoned route. Aim to complete the key "
                "lemma within 1800 reasoning tokens, then emit FINAL immediately. Preserve "
                "only steps indispensable for correctness or exhaustiveness."
            )
        request = self._request(
            problem,
            spec,
            role=role + compact_directive,
            method=(
                "Follow the bounded route plan below when it is mathematically applicable; "
                "correct it from the original statement if a theorem hypothesis fails."
                if plan and plan.valid
                else f"Suggested route, only if applicable: {spec.primary_method}."
            ),
            context=self._grounded_context(
                spec, cards, review=False, plan=plan
            ),
            evidence=evidence,
            plan=plan,
        )
        return f"{comprehension_check}\n\n{request}"

    def _independent_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        *,
        plan: SolvePlan | None = None,
    ) -> str:
        return self._request(
            problem,
            spec,
            role=(
                "Solve independently as a skeptical second mathematician. Re-derive the "
                "requested conclusion from the original statement and actively search "
                "for counterexamples, missing cases, sign errors, and failed theorem "
                "hypotheses. Use a genuinely different checkpoint from the primary route. "
                "Put the complete checked answer on the first FINAL line before concise "
                "support. Never identify a contest source or substitute a recalled answer "
                "for a derivation from the statement. For an all-solutions or attainable-"
                "values claim, verify an arbitrary listed member and exclude the nearest "
                "outside boundary; endpoint examples alone are not exhaustive proof."
            ),
            method=(
                f"Preferred independent route or falsifier, only when applicable: "
                f"{spec.alternative_method}."
            ),
            context=self._join_context(
                self._grounded_context(spec, cards, review=False, plan=plan),
                self._grounded_context(spec, cards, review=True, plan=plan),
                plan.independent_context(spec) if plan is not None and plan.valid else "",
            ),
            evidence=self._independent_evidence(evidence),
            plan=plan,
        )

    def _blind_tiebreak_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        *,
        plan: SolvePlan | None = None,
    ) -> str:
        """Obtain a third solution without exposing either disputed draft."""
        return self._request(
            problem,
            spec,
            role=(
                "Solve this problem as a third independent mathematician. Recompute "
                "the requested conclusion from the original statement only. Use a "
                "direct derivation, invariant, substitution, or boundary check that "
                "would expose a plausible competing answer."
            ),
            method="Choose the most reliable route from the statement; do not infer any earlier answer.",
            context=self._join_context(
                self._grounded_context(spec, cards, review=False, plan=plan),
                self._grounded_context(spec, cards, review=True, plan=plan),
                (
                    plan.independent_context(spec)
                    if plan is not None and plan.valid
                    else ""
                ),
            ),
            evidence=self._independent_evidence(evidence),
            plan=None,
        )

    def _recovery_request(
        self,
        problem: str,
        spec: ProblemSpec,
        cards: RetrievalBundle,
        evidence: tuple[ToolEvidence, ...],
        drafts: tuple[str, ...] = (),
        *,
        plan: SolvePlan | None = None,
    ) -> str:
        support_on_final = (
            " Put the complete conclusion on the first line beginning FINAL:. On the next "
            "line begin PROOF: and give the shortest necessary justification. A result-only "
            "response is incomplete; do not put planning or analysis before FINAL:."
            if spec.answer_contract.mode != "answer_only"
            else ""
        )
        request = self._request(
            problem,
            spec,
            role=(
                "Previous attempts did not produce a complete parseable answer. "
                "Recover the mathematics compactly and put a checked complete FINAL line first."
                + support_on_final
            ),
            method="Prefer a short direct derivation and one decisive check.",
            context=self._join_context(
                self._grounded_context(spec, cards, review=False, plan=plan),
                self._grounded_context(spec, cards, review=True, plan=plan),
            ),
            evidence=evidence,
            plan=plan,
        )
        excerpts = [self._draft_excerpt(item) for item in drafts if str(item or "").strip()]
        excerpts = [item for item in excerpts if item]
        if excerpts:
            request += (
                "\n\nThe following model drafts are untrusted and may be truncated or wrong. "
                "Reuse only calculations you independently verify; resolve disagreements from "
                "the original problem and do not continue their prose. Never use a recalled "
                "contest source or a claimed known answer as evidence.\n"
                + "\n\n".join(
                    f"Draft {index}:\n{excerpt}"
                    for index, excerpt in enumerate(excerpts, start=1)
                )
            )
        return request

    def _tool_synthesis_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
    ) -> str:
        """Turn a locally recomputed operation into a domain-audited answer."""
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "the complete requested result"
        local_results = self._evidence_prompt(evidence) or "none"
        if spec.profile.language == "zh":
            return (
                "模型提交的一个运算已由本地程序精确重算，但运算对题意的翻译尚未认证。"
                "只检查该运算是否与题面目标和定义域完全一致；特别核对 0、端点、前导零、"
                "标号/不标号、是否允许重复，以及‘恰好/至多/至少’。若定义域少算或多算，"
                "在最终值中明确修正。不要从头展开长推导。输出恰好两行：\n"
                "FINAL: 完整可判分答案\n"
                "CHECK: 一条可复现的题意域核验或修正\n\n"
                f"必答内容：{obligations}\n\n题目：\n{problem}\n\n"
                f"模型翻译后的本地重算：\n{local_results}"
            )
        return (
            "A model-submitted operation was recomputed exactly by a local program, but "
            "its translation of the problem is not certified. Check only whether the "
            "operation matches the exact target and domain. Explicitly audit zero, endpoints, "
            "leading zeros, labelled versus unlabelled objects, repetition, and every "
            "'exactly', 'at most', or 'at least' qualifier. Correct the value if the submitted "
            "domain omitted or added cases. Do not restart a long derivation. Output exactly "
            "two lines:\nFINAL: the complete gradable answer\n"
            "CHECK: one reproducible domain check or correction\n\n"
            f"Required content: {obligations}\n\nProblem:\n{problem}\n\n"
            f"Locally recomputed model translation:\n{local_results}"
        )

    @staticmethod
    def _continuation_instruction(spec: ProblemSpec) -> str:
        """Turn an exhausted deep draft into an answer without restarting it."""
        proof_like = spec.answer_contract.mode != "answer_only"
        obligations = "; ".join(
            part.description
            for part in spec.answer_contract.parts
            if part.strict
        )
        if spec.profile.language == "zh":
            support = (
                "不要只在 FINAL 行写数值；把完整结论和最短必要依据都写在该行内"
                f"（必须覆盖：{obligations}），随后只补充仍不可缺少的论证。"
                if proof_like
                else "该行后至多补一行决定性核验。"
            )
            final_template = (
                r"FINAL: \boxed{完整且具体的答案}；依据：最短必要核验。"
                if spec.answer_contract.wrapper == "boxed"
                else "FINAL: 完整且具体的答案；依据：最短必要核验。"
            )
            return (
                "上一条深度推导因输出上限被截断。沿用其中已经完成的数学状态，"
                "不要从头重算、不要复述草稿，也不要讨论格式；若还差一步，只完成决定性的一步。"
                "不得凭题目出处或记忆中的答案补全；全解或取值范围必须核验一般成员及边界外值。"
                f"现在第一行立即按此形式写：{final_template}"
                + support
            )
        support = (
            "Do not put only the result on the FINAL line. Put the complete conclusion and "
            f"shortest required support on that same line (cover: {obligations}); then add "
            "only indispensable remaining details."
            if proof_like
            else "After that line, add at most one decisive check."
        )
        final_template = (
            r"FINAL: \boxed{the complete concrete answer}; CHECK: the shortest required justification."
            if spec.answer_contract.wrapper == "boxed"
            else "FINAL: the complete concrete answer; CHECK: the shortest required justification."
        )
        return (
            "The preceding deep derivation hit its output limit. Use its completed "
            "mathematical state; do not restart, repeat the draft, or discuss formatting. "
            "If one step remains, finish only that decisive step. Never fill a gap from a "
            "recalled contest source or a claimed known answer. For all-solutions or "
            "attainable-value claims, check a general member and an excluded boundary. "
            "Now make the first line "
            f"exactly follow this form: {final_template} " + support
        )

    @staticmethod
    def _conclusion_only_instruction(spec: ProblemSpec) -> str:
        """Second-level recovery after an incremental continuation also truncates."""
        obligations = "; ".join(
            part.description
            for part in spec.answer_contract.parts
            if part.strict
        ) or "the complete requested conclusion"
        wrapper = (
            r"\boxed{...}"
            if spec.answer_contract.wrapper == "boxed" else "the requested form"
        )
        if spec.profile.language == "zh":
            return (
                "续写也已达到输出上限。停止证明、停止继续推导，不要重复任何前文。"
                "只根据刚才已经完成的数学状态提取结论。第一行且仅第一段输出 "
                f"FINAL: {wrapper}，必须覆盖：{obligations}。"
                "答案型题目其后不得增加内容；证明型题目只允许再加一句包含关键依据的完整闭合句。"
            )
        return (
            "The incremental continuation also reached its output limit. Stop proving, "
            "stop deriving, and repeat none of the preceding text. Extract only the "
            "conclusion from the mathematical state you just completed. The first and "
            f"only answer block must begin FINAL: {wrapper} and cover: {obligations}. "
            "For an answer-only task add nothing else; for a proof-bearing task add at "
            "most one complete closed sentence containing the decisive justification."
        )

    def _candidate_audit_request(
        self,
        problem: str,
        spec: ProblemSpec,
        candidate: CandidateAssessment,
        evidence: tuple[ToolEvidence, ...],
        *,
        plan: SolvePlan | None = None,
    ) -> str:
        """Ask for one falsifiable audit instead of another long blind solve."""
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "complete requested result"
        planned_subject = (
            plan.effective_subject(spec)
            if plan is not None and plan.valid
            else ""
        )
        protocol = subject_protocol(
            spec,
            review=True,
            subject_override=planned_subject,
        ) or "direct recomputation"
        planned_check = (
            "Planner-proposed falsification target (untrusted; use it only when "
            f"applicable): {plan.check}. Likely failure mode: {plan.risks}."
            if plan is not None and plan.valid
            else ""
        )
        candidate_text = self._bounded(candidate.answer, 6000)
        evidence_text = self._evidence_prompt(evidence) or "none"
        proof_note = (
            "If CORRECTED, the FINAL section must include a complete concise proof."
            if spec.answer_contract.mode != "answer_only"
            else "The FINAL line must contain the entire gradable result."
        )
        request = (
            "Audit the candidate below against the original problem. Treat it as untrusted: "
            "recompute the smallest decisive quantity, theorem hypothesis, boundary case, "
            "normalization, or counterexample that can confirm or falsify it. Do not restart "
            "a long solution and do not confirm by paraphrasing. Change the answer only when "
            "the CHECK explicitly demonstrates the error. A contest name, remembered answer, "
            "or authority is never a check. For an all-solutions, interval, or attainable-values "
            "claim, endpoint examples cannot confirm it: test the construction for an arbitrary "
            "parameter and test the nearest excluded boundary. For a finite count, explicitly "
            "audit zero and endpoint inclusion, leading zeros, labelled versus unlabelled "
            "objects, repetition, and every 'exactly', 'at most', or 'at least' qualifier; "
            "an exact local computation can still encode the wrong domain.\n\n"
            "Return in this exact order:\n"
            "CHECK: one concrete reproducible mathematical check\n"
            "DECISION: CONFIRMED, CORRECTED, or UNRESOLVED\n"
            "FINAL: \\boxed{the complete answer}\n"
            f"{proof_note}\n\n"
            f"Required content: {obligations}.\n\n"
            f"Audit protocol: {protocol}\n\n"
        )
        if planned_check:
            request += planned_check + "\n\n"
        request += (
            f"Problem:\n{problem}\n\n"
            f"Candidate:\n{candidate_text}\n\n"
            f"Certified local evidence:\n{evidence_text}"
        )
        if self._should_request_structured_verification(problem, spec):
            request += (
                "\n\nA CORRECTED numerical/algebraic conclusion is admissible only when "
                "you also provide a locally executable certificate in the following "
                "whitelisted data-only format. A CONFIRMED or UNRESOLVED decision may "
                "omit it.\n"
                + StructuredVerificationTool.prompt_instruction(spec.profile.language)
            )
        return request

    def _apply_candidate_audit(
        self,
        raw: str,
        result: ModelCallResult,
        audit_candidates: list[CandidateAssessment],
        base: CandidateAssessment | None,
        spec: ProblemSpec,
    ) -> tuple[str, CandidateAssessment | None]:
        """Admit an audit only when its decision and check agree with FINAL."""
        if base is None or self._truncated(result, raw):
            return "truncated_or_missing_base", None
        decision_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?DECISION"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(CONFIRMED|CORRECTED|UNRESOLVED)\b",
            str(raw or ""),
        )
        check = self._audit_check_section(raw)
        if not decision_match or not check:
            return "invalid_protocol", None
        decision = decision_match.group(1).upper()
        if decision == "UNRESOLVED":
            return decision, None
        if self._critic_is_self_uncertain(raw):
            return "self_uncertain", None
        if not self._decisive_check(check):
            return "non_reproducible_check", None
        if decision == "CONFIRMED" and self._small_case_only(check):
            return "non_general_small_case", None
        submitted = choose_candidate(audit_candidates)
        if submitted is None or not self._is_result_usable(submitted):
            return "unusable_final", None
        same = self._same_conclusion(submitted, base, spec)
        if decision == "CONFIRMED":
            if not same:
                return "confirmed_final_mismatch", None
            if base.transport_truncated and self._is_complete(submitted):
                return decision, submitted
            return decision, base
        if same:
            return "corrected_without_change", None
        if not self._is_complete(submitted):
            return "corrected_incomplete", None
        proof_like = spec.profile.task_kind in {
            "proof", "derivation", "explanation", "construction"
        } or spec.answer_contract.mode != "answer_only"
        if not self._objectively_checked(submitted):
            # General proofs rarely admit a SymPy certificate. A correction is
            # still admissible when the reviewer names a concrete fatal flaw
            # and supplies a complete replacement proof. Numeric/result tasks
            # keep the stronger local-certificate requirement.
            if not (
                proof_like
                and self._correction_check_identifies_error(check)
            ):
                return "corrected_without_local_certificate", None
        return decision, submitted

    @staticmethod
    def _audit_check_section(value: str) -> str:
        """Read a bounded CHECK section, including multiline calculations."""
        match = re.search(
            r"(?ims)^\s*(?:[*_`]{0,3}\s*)?CHECK"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(.+?)(?=^\s*(?:[*_`]{0,3}\s*)?DECISION"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]|\Z)",
            str(value or ""),
        )
        if not match:
            return ""
        return match.group(1).strip()[:12_000]

    @staticmethod
    def _correction_check_identifies_error(value: str) -> bool:
        text = str(value or "")
        return bool(
            SubmissionAgent._decisive_check(text)
            and re.search(
                r"错误|有误|不成立|不满足|遗漏|漏掉|反例|矛盾|非法|不能推出|"
                r"缺少(?:条件|情形)|偷换|"
                r"\b(?:error|incorrect(?:ly)?|invalid|fails?|failure|missing|omitted|"
                r"counterexample|contradiction|does\s+not\s+(?:follow|hold)|"
                r"hypothesis\s+is\s+not\s+satisfied|overestimat(?:e|es|ed|ing)|"
                r"underestimat(?:e|es|ed|ing)|off\s+by|too\s+(?:large|small)|"
                r"differs?\s+by\s+(?:a\s+)?factor)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _request(
        self,
        problem: str,
        spec: ProblemSpec,
        *,
        role: str,
        method: str,
        context: str,
        evidence: tuple[ToolEvidence, ...],
        plan: SolvePlan | None = None,
    ) -> str:
        language = "Chinese" if spec.profile.language == "zh" else "English"
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        ) or "an explicit complete result"
        support = (
            "Include essential complete support in final_response."
            if spec.answer_contract.mode != "answer_only"
            else "Return the complete gradable result without a long exposition."
        )
        sections = [
            "Problem:\n" + problem,
            role,
            f"Answer language: {language}.",
            f"Required answer content: {obligations}.",
            support,
            method,
        ]
        if plan is not None and plan.valid:
            sections.append(
                "Untrusted route plan (use it to avoid wandering, but verify it against "
                "the original statement):\n" + plan.prompt_context(spec)
            )
        semantic_context = spec.semantics.prompt_context(spec.profile.language)
        if semantic_context:
            sections.append(
                "Explicit statement checklist (copied from the problem; use only to avoid "
                "dropping a condition):\n" + semantic_context
            )
        if context:
            sections.append("General reference facts (verify applicability):\n" + context)
        tool_context = self._evidence_prompt(evidence)
        if tool_context:
            sections.append(
                "Local mathematical evidence: CERTIFIED_RECOMPUTATION entries were "
                "independently recomputed, so any submitted conclusion must agree with them. "
                "ADVISORY_SCHEMA entries are untrusted route suggestions: verify every hypothesis "
                "against the original problem and ignore them when they do not apply. "
                "Subexpression entries are only partial checks:\n"
                + tool_context
            )
        if self._should_locate_tool(problem, spec):
            sections.append(OperationLocator.prompt_instruction(spec.profile.language))
        if self._should_request_structured_verification(problem, spec):
            sections.append(
                StructuredVerificationTool.prompt_instruction(spec.profile.language)
            )
        if spec.profile.language == "zh":
            sections.append(
                "输出边界（最高优先级）：响应的第一个可见字符必须属于首行 "
                "FINAL:，该行立即给出全部可判分结论；在 FINAL 之前不得输出分析、"
                "复述、标题或计划。其后只保留最短必要核验。"
            )
        else:
            sections.append(
                "OUTPUT BOUNDARY (highest priority): the first visible characters of the "
                "response must be the first-line marker FINAL:, immediately followed by "
                "the complete gradable conclusion. Emit no analysis, restatement, heading, "
                "or plan before FINAL; after it, keep only the shortest necessary check."
            )
        return "\n\n".join(sections)

    def _arbitration_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        first: CandidateAssessment,
        second: CandidateAssessment,
        *,
        first_context: str = "",
        second_context: str = "",
    ) -> str:
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        request = (
            "Two independent candidates disagree. Recompute the disputed quantity from "
            "the original statement. Their work excerpts are untrusted: locate the first "
            "concrete step where they differ and recompute that step. Do not choose by "
            "style, length, confidence, or majority. Provide one "
            "reproducible substitution, invariant, theorem-hypothesis audit, or small-case "
            "check. Output exactly two labelled lines, in this order, before any explanation:\n"
            "DECISION: A, B, or UNRESOLVED\n"
            "CHECK: the decisive mathematical check\n"
            "Do not reproduce either candidate or write a FINAL section; the selected complete "
            "candidate is already retained by the caller.\n\n"
            f"Required content: {obligations}.\n\n"
            f"Subject audit protocol:\n{subject_protocol(spec, review=True) or 'general recomputation'}\n\n"
            f"Problem:\n{problem}\n\n"
            f"Candidate A:\n{self._bounded(first.answer, 3500)}\n\n"
            f"Candidate B:\n{self._bounded(second.answer, 3500)}\n\n"
            f"Local check evidence:\n{self._evidence_prompt(evidence) or 'none'}"
        )
        if first_context or second_context:
            request += (
                "\n\nUntrusted work excerpts (use only to locate a check; verify from the "
                "problem):\n"
                f"Work A:\n{first_context or 'unavailable'}\n\n"
                f"Work B:\n{second_context or 'unavailable'}"
            )
        return request

    def _consensus_audit_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        first: CandidateAssessment,
        second: CandidateAssessment,
    ) -> str:
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        return (
            "Two nominally independent drafts reach the same conclusion but lack an "
            "objective certificate and may share one hidden mistake. Audit them from the "
            "original statement. Locate the first unsupported implication, theorem-hypothesis "
            "gap, omitted case, or sign error; if none exists, rebuild the argument compactly. "
            "Output exactly these labelled sections:\n"
            "FINAL: the complete audited answer, including the proof when required\n"
            "DECISION: A, B, CORRECTED, or UNRESOLVED\n"
            "CHECK: one reproducible falsification attempt or decisive theorem-hypothesis audit\n\n"
            f"Required content: {obligations}.\n\n"
            f"Subject audit protocol:\n{subject_protocol(spec, review=True) or 'general recomputation'}\n\n"
            f"Problem:\n{problem}\n\n"
            f"Draft A:\n{self._bounded(first.answer, 4500)}\n\n"
            f"Draft B:\n{self._bounded(second.answer, 4500)}\n\n"
            f"Local check evidence:\n{self._evidence_prompt(evidence) or 'none'}"
        )

    def _pair_critic_repair_request(
        self,
        problem: str,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...],
        first: CandidateAssessment,
        second_raw: str,
    ) -> str:
        """Audit a complete A using the independent but unfinished work of B."""
        obligations = "; ".join(
            part.description for part in spec.answer_contract.parts if part.strict
        )
        proof_note = (
            "The FINAL section must contain the complete concise proof or construction."
            if spec.answer_contract.mode != "answer_only"
            else "The FINAL line must contain the entire independently gradable result."
        )
        return (
            "Act as a strict mathematical Critic. Solver A produced a complete candidate; "
            "Solver B worked independently but its response may be unfinished or truncated. "
            "Use B only to locate a disagreement or missing case. Recompute the first decisive "
            "step from the original problem; do not choose by confidence, prose, or majority, "
            "and do not merely continue B. Repair the answer only when CHECK demonstrates a "
            "specific error. A remembered source or a single changed-size example cannot "
            "confirm a general count. For all-solutions or attainable-value claims, check an "
            "arbitrary member and the nearest excluded boundary. Output exactly these labelled "
            "sections, with FINAL first:\n"
            "FINAL: the complete checked answer\n"
            "DECISION: KEEP_A, USE_B, CORRECTED, or UNRESOLVED\n"
            "CHECK: one concrete reproducible substitution, invariant, boundary case, "
            "counterexample, or theorem-hypothesis audit\n"
            f"{proof_note}\n\n"
            f"Required content: {obligations}.\n\n"
            f"Subject audit protocol:\n{subject_protocol(spec, review=True) or 'general recomputation'}\n\n"
            f"Problem:\n{problem}\n\n"
            f"Complete candidate A:\n{self._bounded(first.answer, 4500)}\n\n"
            "Independent work B (untrusted and possibly cut off):\n"
            f"{self._draft_excerpt(second_raw)}\n\n"
            f"Local check evidence:\n{self._evidence_prompt(evidence) or 'none'}"
        )

    def _apply_pair_critic_repair(
        self,
        raw: str,
        result: ModelCallResult,
        critic_candidates: list[CandidateAssessment],
        first: CandidateAssessment,
        spec: ProblemSpec,
    ) -> tuple[str, CandidateAssessment | None]:
        """Admit a pair Critic only when its check supports its decision."""
        if self._truncated(result, raw):
            return "truncated", None
        decision_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:DECISION|裁决|判定)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(KEEP_A|USE_B|CORRECTED|UNRESOLVED)\b",
            str(raw or ""),
        )
        check_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:CHECK|核验|检查)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(\S[^\n]{12,})",
            str(raw or ""),
        )
        if not decision_match or not check_match:
            return "invalid_protocol", None
        decision = decision_match.group(1).upper()
        check = check_match.group(1)
        if decision != "UNRESOLVED" and self._critic_is_self_uncertain(raw):
            return "self_uncertain", None
        if not self._decisive_check(check):
            return "non_reproducible_check", None
        if decision == "UNRESOLVED":
            return decision, None
        if decision == "KEEP_A" and self._small_case_only(check):
            return "non_general_small_case", None

        submitted = choose_candidate(critic_candidates)
        if submitted is None or not self._is_complete(submitted):
            return "unusable_final", None
        same_as_first = self._same_conclusion(submitted, first, spec)
        if decision == "KEEP_A":
            return (decision, first) if same_as_first else (
                "keep_a_final_mismatch", None
            )
        if decision == "CORRECTED" and same_as_first:
            return "corrected_without_change", None
        if not self._correction_check_identifies_error(check):
            return "replacement_without_identified_error", None
        return decision, submitted

    def _apply_consensus_audit(
        self,
        raw: str,
        result: ModelCallResult,
        audit_candidates: list[CandidateAssessment],
        first: CandidateAssessment,
        second: CandidateAssessment,
        spec: ProblemSpec,
    ) -> tuple[str, CandidateAssessment | None]:
        """Apply a short attack on two agreeing high-risk proof drafts."""
        if self._truncated(result, raw):
            return "truncated", None
        decision_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:DECISION|裁决|判定)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(A|B|CORRECTED|UNRESOLVED)\b",
            str(raw or ""),
        )
        check_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:CHECK|核验|检查)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(\S[^\n]{12,})",
            str(raw or ""),
        )
        if not decision_match or not check_match:
            return "invalid_protocol", None
        decision = decision_match.group(1).upper()
        check = check_match.group(1)
        if decision != "UNRESOLVED" and self._critic_is_self_uncertain(raw):
            return "self_uncertain", None
        if not self._decisive_check(check):
            return "non_reproducible_check", None
        if decision == "A":
            return decision, first
        if decision == "B":
            return decision, second
        if decision == "UNRESOLVED":
            return decision, None

        submitted = choose_candidate(audit_candidates)
        if submitted is None or not self._is_complete(submitted):
            return "corrected_incomplete", None
        if self._same_conclusion(submitted, first, spec) or self._same_conclusion(
            submitted, second, spec
        ):
            return "corrected_without_change", None
        if not self._correction_check_identifies_error(check):
            return "corrected_without_identified_error", None
        return decision, submitted

    def _apply_arbitration(
        self,
        raw: str,
        result: ModelCallResult,
        _third_candidates: list[CandidateAssessment],
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        spec: ProblemSpec,
        evidence: tuple[ToolEvidence, ...] = (),
    ) -> tuple[str, CandidateAssessment | None]:
        if self._truncated(result, raw):
            return "truncated", None
        decision_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:DECISION|裁决|判定)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(A|B|UNRESOLVED)\b",
            raw,
        )
        check_match = re.search(
            r"(?im)^\s*(?:[*_`]{0,3}\s*)?(?:CHECK|核验|检查)"
            r"\s*(?:[*_`]{0,3}\s*)?[:：]\s*(?:[*_`]{0,3}\s*)?"
            r"(\S[^\n]{12,})",
            raw,
        )
        if not decision_match or not check_match:
            return "invalid_protocol", None
        if (
            decision_match.group(1).upper() != "UNRESOLVED"
            and self._critic_is_self_uncertain(raw)
        ):
            return "self_uncertain", None
        if not self._decisive_check(check_match.group(1)):
            return "non_reproducible_check", None
        decision = decision_match.group(1).upper()
        if decision == "A" and self._is_result_usable(first):
            return decision, first
        if decision == "B" and self._is_result_usable(second):
            return decision, second
        return decision, None

    @staticmethod
    def _conflict(
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        spec: ProblemSpec,
    ) -> bool:
        if not SubmissionAgent._is_result_usable(first) or not SubmissionAgent._is_result_usable(second):
            return False
        left = SubmissionAgent._comparison_value(first.answer)
        right = SubmissionAgent._comparison_value(second.answer)
        if equivalent_answers(left, right):
            return False
        if spec.profile.answer_shape == "proof":
            left_result = Finalizer.extract_result(first.answer)
            right_result = Finalizer.extract_result(second.answer)
            if (
                left_result.valid
                and right_result.valid
                and left_result.explicit_answer
                and right_result.explicit_answer
                and left_result.answer
                and right_result.answer
            ):
                # A proof's body may use different methods, but two explicit
                # FINAL claims are the objects being graded.  Do not collapse
                # contradictory textual conclusions merely because both use
                # positive grammatical forms (for example finite/infinite).
                if equivalent_answers(left_result.answer, right_result.answer):
                    return False
                return not SubmissionAgent._proof_claims_agree(
                    left_result.answer,
                    right_result.answer,
                    spec.problem_text,
                )
        left_claim = SubmissionAgent._explicit_conclusion_claim(left)
        right_claim = SubmissionAgent._explicit_conclusion_claim(right)
        if left_claim and right_claim:
            return not equivalent_answers(left_claim, right_claim)
        if left_claim and equivalent_answers(left_claim, right):
            return False
        if right_claim and equivalent_answers(left, right_claim):
            return False
        if spec.profile.answer_shape == "choice":
            from classifier.choice import answer_choice_labels

            return set(answer_choice_labels(left)) != set(answer_choice_labels(right))
        if spec.profile.answer_shape in {"truth", "proof"}:
            left_polarity = SubmissionAgent._conclusion_polarity(left)
            right_polarity = SubmissionAgent._conclusion_polarity(right)
            if left_polarity and right_polarity:
                return left_polarity != right_polarity
            left_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", left)
            right_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", right)
            return bool(left_numbers and right_numbers and left_numbers != right_numbers)
        return True

    @staticmethod
    def _proof_claims_agree(left: str, right: str, target: str) -> bool:
        """Recognize conservative paraphrases of the same proof conclusion."""
        left_properties = SubmissionAgent._proof_property_signature(left)
        right_properties = SubmissionAgent._proof_property_signature(right)
        opposed = {
            frozenset(("finite", "infinite")),
            frozenset(("exists", "not_exists")),
            frozenset(("converges", "diverges")),
            frozenset(("zero", "nonzero")),
            frozenset(("true", "false")),
        }
        if any(
            frozenset((left_item, right_item)) in opposed
            for left_item in left_properties
            for right_item in right_properties
        ):
            return False
        left_polarity = SubmissionAgent._conclusion_polarity(left)
        right_polarity = SubmissionAgent._conclusion_polarity(right)
        if left_polarity and right_polarity and left_polarity != right_polarity:
            return False

        left_anchors = SubmissionAgent._proof_claim_anchors(left)
        right_anchors = SubmissionAgent._proof_claim_anchors(right)
        target_anchors = SubmissionAgent._proof_claim_anchors(target)
        shared = left_anchors & right_anchors
        if not shared:
            return False
        if target_anchors:
            return bool(
                left_anchors & target_anchors
                and right_anchors & target_anchors
            )
        return True

    @staticmethod
    def _proof_property_signature(value: str) -> set[str]:
        text = str(value or "").casefold()
        patterns = {
            "infinite": r"无穷|无限|infinit(?:e|ely)",
            "finite": r"有限|\bfinite\b",
            "not_exists": r"不存在|无解|does\s+not\s+exist|no\s+solutions?",
            "exists": r"存在|\bexists?\b",
            "diverges": r"发散|\bdiverg(?:e|es|ent)\b",
            "converges": r"收敛|\bconverg(?:e|es|ent)\b",
            "nonzero": r"非零|不为\s*0|\bnonzero\b|not\s+zero",
            "zero": r"为\s*0|等于\s*0|\bzero\b",
            "false": r"不成立|错误|为假|\bfalse\b",
            "true": r"成立|正确|为真|\btrue\b",
        }
        found = {
            name for name, pattern in patterns.items()
            if re.search(pattern, text, re.IGNORECASE)
        }
        # A negative form is more specific than the positive token contained
        # inside it (for example 不存在 contains 存在).
        if "not_exists" in found:
            found.discard("exists")
        if "nonzero" in found:
            found.discard("zero")
        if "false" in found:
            found.discard("true")
        return found

    @staticmethod
    def _proof_claim_anchors(value: str) -> set[str]:
        text = re.sub(r"\\[A-Za-z]+|[^A-Za-z0-9_\u4e00-\u9fff]+", " ", str(value or "").casefold())
        english_stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "being",
            "that", "this", "there", "has", "have", "holds", "true", "false",
            "finite", "infinite", "exists", "does", "not", "prove", "show",
            "conclusion", "set", "collection",
        }
        anchors = {
            token for token in re.findall(r"[a-z][a-z0-9_]{2,}", text)
            if token not in english_stop
        }
        cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
        for stop in (
            "最终答案", "结论", "证明", "断言", "成立", "正确", "错误",
            "集合", "所有", "任意", "一个", "多个", "有", "是", "的",
            "有限", "无限", "无穷", "存在", "不存在", "收敛", "发散",
            "非零", "为零",
        ):
            cjk = cjk.replace(stop, " ")
        for segment in cjk.split():
            if len(segment) == 2:
                anchors.add(segment)
            elif len(segment) > 2:
                anchors.update(
                    segment[index:index + 2]
                    for index in range(len(segment) - 1)
                )
        anchors.update(re.findall(r"(?<![A-Za-z])[A-Za-z](?:_\w+)?", str(value or "")))
        return anchors

    @staticmethod
    def _without_uncorroborated_corrections(
        candidates: list[CandidateAssessment],
        baseline: CandidateAssessment | None,
        *,
        spec: ProblemSpec,
    ) -> list[CandidateAssessment]:
        """Keep value changes only when objective or independent evidence agrees.

        A reviewer labelling its own rewrite ``confirmed`` is not independent
        evidence.  This guard is intentionally phrased in terms of candidate
        groups and mathematical checks, so it applies equally to recovery,
        audit, and future solver stages.
        """
        if not SubmissionAgent._is_result_usable(baseline):
            return list(candidates)

        kept: list[CandidateAssessment] = []
        for candidate in candidates:
            if candidate is baseline or not SubmissionAgent._conflict(
                candidate, baseline, spec
            ):
                kept.append(candidate)
                continue
            if SubmissionAgent._objectively_checked(candidate):
                kept.append(candidate)
                continue
            corroborated = any(
                other is not candidate
                and other is not baseline
                and SubmissionAgent._is_result_usable(other)
                and other.independence_group != candidate.independence_group
                and other.independence_group != baseline.independence_group
                and not SubmissionAgent._conflict(candidate, other, spec)
                for other in candidates
            )
            if corroborated:
                kept.append(candidate)
        return kept

    @staticmethod
    def _weak_consensus(
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        spec: ProblemSpec,
    ) -> bool:
        if not SubmissionAgent._is_complete(first) or not SubmissionAgent._is_complete(second):
            return False
        if SubmissionAgent._conflict(first, second, spec):
            return False
        if SubmissionAgent._objectively_checked(first) or SubmissionAgent._objectively_checked(second):
            return False
        task = spec.profile.task_kind
        if task not in {"proof", "derivation", "explanation", "construction"}:
            return False
        sensitive = bool(
            task == "construction"
            or spec.risk_score >= 6
            or spec.profile.subject_confidence == "low"
        )
        if not sensitive:
            return False
        left_methods = SubmissionAgent._reasoning_families(first.answer)
        right_methods = SubmissionAgent._reasoning_families(second.answer)
        return not (left_methods and right_methods and left_methods.isdisjoint(right_methods))

    @staticmethod
    def _reasoning_families(value: str) -> set[str]:
        patterns = {
            "contradiction": r"反设|矛盾|contradiction|suppose not",
            "induction": r"归纳|induction|inductive",
            "counting": r"双计数|容斥|生成函数|bijection|double count|inclusion|generating function",
            "linear_algebra": r"矩阵|行列式|特征值|秩|matrix|determinant|eigen|rank",
            "calculus": r"求导|积分|极限|Taylor|differentiat|integrat|limit|series expansion",
            "probabilistic": r"条件概率|期望|鞅|conditioning|expectation|martingale",
            "algebraic": r"同态|商|理想|多项式|homomorphism|quotient|ideal|polynomial",
            "topological": r"紧致|连通|开覆盖|同伦|compact|connected|open cover|homotopy",
            "energy": r"能量|变分|energy|variational",
            "direct": r"代入|直接计算|由定义|substitut|direct calculation|from the definition",
        }
        text = str(value or "")
        return {
            family for family, pattern in patterns.items()
            if re.search(pattern, text, re.IGNORECASE)
        }

    @staticmethod
    def _conclusion_polarity(value: str) -> str:
        text = str(value or "")
        if re.search(
            r"不成立|错误|为假|不是|不能|不可|不存在|无解|"
            r"\b(?:false|not|cannot|can't|does not|do not|no solutions?|impossible)\b",
            text,
            re.IGNORECASE,
        ):
            return "negative"
        if re.search(
            r"成立|正确|为真|是|能|可|存在|"
            r"\b(?:true|yes|holds?|exists?|is|are|can)\b",
            text,
            re.IGNORECASE,
        ):
            return "positive"
        return ""

    @staticmethod
    def _explicit_conclusion_claim(value: str) -> str:
        """Extract one asserted formula value before using coarse truth polarity."""
        text = str(value or "").strip()
        equalities = list(re.finditer(r"(?<![<>=])=(?!=)", text))
        if len(equalities) == 1:
            rhs = text[equalities[0].end():].strip().strip("$ ")
            rhs = re.split(r"[。；;\n]", rhs, maxsplit=1)[0].strip()
            if rhs:
                return rhs
        value_phrase = re.search(
            r"(?:等于|其值为|结果为)\s*\$?([^$。；;\n]+)",
            text,
            re.IGNORECASE,
        )
        if value_phrase:
            claim = value_phrase.group(1).strip()
            if re.search(r"\d|\\|[+\-*/^(){}]", claim):
                return claim
        return ""

    @staticmethod
    def _comparison_value(value: str) -> str:
        result = Finalizer.extract_result(value)
        return result.answer if result.valid and result.answer else str(value or "")

    @staticmethod
    def _is_usable(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier in {"complete", "degraded"}
            and not candidate.failed_check
        )

    @staticmethod
    def _is_complete(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier == "complete"
            and not candidate.failed_check
        )

    @staticmethod
    def _is_result_usable(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and candidate.validation_tier in {"complete", "degraded"}
            and candidate.shape_valid
            and candidate.formatting_valid
            and bool(candidate.result_coverage)
            and all(candidate.result_coverage)
            and not candidate.failed_check
        )

    @staticmethod
    def _complete_after_transport(
        candidate: CandidateAssessment | None,
        truncated: bool,
        spec: ProblemSpec | None = None,
        raw_response: str = "",
    ) -> bool:
        if not SubmissionAgent._is_complete(candidate):
            return False
        if not truncated:
            return True
        if candidate is None or not candidate.explicit_answer:
            return False
        if spec is None:
            return False
        if spec.answer_contract.mode == "answer_only":
            # Thinking responses often contain labelled intermediate claims.
            # After a provider cut, only a candidate emitted at the promised
            # first visible boundary is trustworthy enough to skip recovery.
            return SubmissionAgent._has_leading_final_boundary(raw_response)
        # A provider-cut proof body cannot become complete merely because a
        # FINAL label appeared before the cut.  A genuinely self-contained
        # one-line FINAL remains admissible; multiline tagged proof bodies
        # must go through continuation or a clean audit replacement.
        return bool(
            candidate.extraction_method != "tagged_solution"
            and "\n" not in candidate.answer
            and candidate.support_coverage
            and all(candidate.support_coverage)
        )

    @staticmethod
    def _has_leading_final_boundary(raw_response: str) -> bool:
        text = str(raw_response or "").lstrip()
        if not text:
            return False
        text = re.sub(r"^(?:```(?:latex|tex|text)?\s*)", "", text, flags=re.IGNORECASE)
        return bool(re.match(
            r"^(?:[*_`]{0,3}\s*)?"
            r"(?:(?:FINAL(?:\s+ANSWER)?|最终答案|答案)\s*[:：]|\\boxed\s*\{)",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _transport_admissible(
        candidates: list[CandidateAssessment],
        truncated: bool,
    ) -> list[CandidateAssessment]:
        if not truncated:
            return candidates
        explicit = [
            replace(item, transport_truncated=True)
            for item in candidates
            if item.explicit_answer
        ]
        if not explicit:
            return []
        labelled_methods = {
            "tagged_solution", "label_boxed", "label", "label_next_line_boxed",
            "label_next_line", "bracket_label",
        }
        labelled = [
            item for item in explicit if item.extraction_method in labelled_methods
        ]
        pool = labelled or explicit
        representatives: list[CandidateAssessment] = []
        for item in pool:
            if not any(
                equivalent_answers(
                    SubmissionAgent._comparison_value(item.answer),
                    SubmissionAgent._comparison_value(other.answer),
                )
                for other in representatives
            ):
                representatives.append(item)
        # A provider-cut response containing several contradictory answer
        # labels/boxes has no trustworthy final boundary. Do not select the
        # last fragment by length or extraction order; force clean recovery.
        return pool if len(representatives) == 1 else []

    @staticmethod
    def _same_conclusion(
        left: CandidateAssessment,
        right: CandidateAssessment,
        spec: ProblemSpec,
    ) -> bool:
        left_value = SubmissionAgent._comparison_value(left.answer)
        right_value = SubmissionAgent._comparison_value(right.answer)
        if equivalent_answers(left_value, right_value):
            return True
        left_claim = SubmissionAgent._explicit_conclusion_claim(left_value)
        right_claim = SubmissionAgent._explicit_conclusion_claim(right_value)
        if left_claim and right_claim:
            return equivalent_answers(left_claim, right_claim)
        if left_claim and equivalent_answers(left_claim, right_value):
            return True
        if right_claim and equivalent_answers(left_value, right_claim):
            return True
        if spec.profile.answer_shape in {"truth", "proof"}:
            left_polarity = SubmissionAgent._conclusion_polarity(left_value)
            right_polarity = SubmissionAgent._conclusion_polarity(right_value)
            return bool(left_polarity and left_polarity == right_polarity)
        return False

    @staticmethod
    def _decisive_check(value: str) -> bool:
        return bool(re.search(
            r"[=<>≤≥]|代入|验证|边界|端点|反例|矛盾|定义|定理|不变量|归一化|余数|"
            r"\b(?:substitut|verify|boundary|endpoint|counterexample|contradiction|"
            r"definition|theorem|invariant|normalization|residual|remainder|"
            r"divis|compact|differentiat|integrat)\w*\b",
            str(value or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def _critic_is_self_uncertain(value: str) -> bool:
        """Reject a Critic that announces doubt but still selects a candidate."""
        return bool(re.search(
            r"\b(?:seems?\s+plausible|probably|perhaps|maybe|might\s+be|"
            r"cannot\s+(?:confirm|verify)|unable\s+to\s+(?:confirm|verify)|"
            r"not\s+(?:sure|directly\s+applicable)|challenging\s+to\s+confirm|"
            r"would\s+be\s+tedious|i\s+will\s+(?:proceed|assume)|"
            r"known\s+(?:answer|solution|result)|standard\s+result|"
            r"contest\s+(?:problem|source)|aime\b)\b|"
            r"似乎|大概|也许|可能正确|无法(?:确认|验证)|难以(?:确认|验证)|"
            r"凭记忆|已知答案|竞赛题来源",
            str(value or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def _small_case_only(value: str) -> bool:
        """A single changed-size example cannot confirm a general formula."""
        text = str(value or "")
        small_case = bool(re.search(
            r"\b(?:for|at)\s+n\s*=\s*[0-9]+|"
            r"\bsmall(?:er)?\s+case\b|小规模|小例子|令\s*n\s*=\s*[0-9]+",
            text,
            re.IGNORECASE,
        ))
        general_bridge = bool(re.search(
            r"递推|归纳|不变量|对任意|一般情形|"
            r"\b(?:recurren|induct|invariant|for\s+all|arbitrary|general)\w*\b",
            text,
            re.IGNORECASE,
        ))
        return small_case and not general_bridge

    @staticmethod
    def _objectively_checked(candidate: CandidateAssessment | None) -> bool:
        return bool(
            candidate is not None
            and (candidate.tool_status == "pass" or candidate.passed_check_count)
        )

    @staticmethod
    def _objective_winner(
        *items: CandidateAssessment | None,
    ) -> CandidateAssessment | None:
        """Return a uniquely stronger locally checked candidate, if one exists."""
        usable = [
            item for item in items
            if SubmissionAgent._is_result_usable(item)
        ]
        if not usable:
            return None

        def strength(item: CandidateAssessment) -> tuple[int, int]:
            return (item.tool_status == "pass", item.passed_check_count)

        ranked = sorted(usable, key=strength, reverse=True)
        best_strength = strength(ranked[0])
        if best_strength == (0, 0):
            return None
        if len(ranked) > 1 and strength(ranked[1]) == best_strength:
            return None
        return ranked[0]

    @classmethod
    def _select_final_candidate(
        cls,
        candidates: list[CandidateAssessment],
        spec: ProblemSpec,
        first: CandidateAssessment | None,
        second: CandidateAssessment | None,
        third: CandidateAssessment | None,
        objective_winner: CandidateAssessment | None,
    ) -> tuple[CandidateAssessment | None, str]:
        """Select by certificates and independent agreement, never stage prestige."""
        all_objective_winner = cls._objective_winner(first, second, third)
        certified = all_objective_winner or objective_winner
        if certified is not None and cls._is_result_usable(certified):
            route = (
                "recovery_objective_certificate"
                if certified.source == "recovery"
                else "objective_certificate"
            )
            return certified, route

        # Recovery sees excerpts from earlier drafts, so it may supply a clean
        # proof but is not an independent mathematical vote.  A blind
        # tiebreak is independent and may participate in a 2-of-3 majority.
        representatives = [
            item for item in (first, second, third)
            if cls._is_result_usable(item) and item.source != "recovery"
        ]
        agreement: dict[int, int] = {}
        for item in representatives:
            agreement[id(item)] = len({
                other.independence_group
                for other in representatives
                if other is not item
                and other.independence_group != item.independence_group
                and cls._same_conclusion(item, other, spec)
            })
        best_agreement = max(agreement.values(), default=0)
        if best_agreement:
            anchors = [
                item for item in representatives
                if agreement.get(id(item), 0) == best_agreement
            ]
            anchor = max(anchors, key=lambda item: cls._representative_quality(item, spec))
            selected = cls._best_matching_conclusion(candidates, anchor, spec)
            route = (
                "blind_majority"
                if third is not None and third.source == "blind_tiebreak"
                else "independent_consensus"
            )
            if selected is not None and selected.source == "recovery":
                route += "_recovered"
            return selected or anchor, route

        # When all independent conclusions differ, an unverified later stage
        # supplies no reason to overwrite the original deep solve.
        if cls._is_complete(first):
            return first, "primary_retained"
        for item, route in (
            (second, "independent_complete"),
            (
                third,
                "recovery_complete"
                if third is not None and third.source == "recovery"
                else "blind_tiebreak_complete",
            ),
        ):
            if cls._is_complete(item):
                return item, route
        if cls._is_result_usable(first):
            return first, "primary_degraded_retained"
        if cls._is_result_usable(second):
            return second, "independent_degraded"
        if cls._is_result_usable(third):
            return third, (
                "recovery_degraded"
                if third.source == "recovery"
                else "blind_tiebreak_degraded"
            )
        complete = [item for item in candidates if item.validation_tier == "complete"]
        return choose_candidate(complete), "ranked_candidates"

    @staticmethod
    def _representative_quality(
        candidate: CandidateAssessment,
        spec: ProblemSpec,
    ) -> tuple[int, ...]:
        proof_like = spec.profile.task_kind in {"proof", "derivation", "explanation"}
        source_priority = {
            "primary": 3,
            "independent": 2,
            "blind_tiebreak": 1,
            "recovery": 0,
        }
        return (
            candidate.tool_status == "pass",
            candidate.passed_check_count,
            candidate.validation_tier == "complete",
            bool(candidate.support_coverage) and all(candidate.support_coverage),
            candidate.formatting_valid,
            source_priority.get(candidate.source, 0),
            min(len(candidate.answer), 6000) if proof_like else -len(candidate.answer),
        )

    @classmethod
    def _best_matching_conclusion(
        cls,
        candidates: Iterable[CandidateAssessment],
        anchor: CandidateAssessment,
        spec: ProblemSpec,
    ) -> CandidateAssessment | None:
        matching = [
            item for item in candidates
            if cls._is_usable(item)
            and cls._same_conclusion(item, anchor, spec)
        ]
        return max(
            matching,
            key=lambda item: cls._representative_quality(item, spec),
        ) if matching else None

    @staticmethod
    def _best_degraded(
        candidates: Iterable[CandidateAssessment],
        spec: ProblemSpec,
    ) -> CandidateAssessment | None:
        usable = [
            item for item in candidates
            if item.answer
            and item.validation_tier == "degraded"
            and not item.failed_check
            and item.formatting_valid
            and (
                item.shape_valid
                or (
                    spec.answer_contract.mode != "answer_only"
                    and bool(item.result_coverage)
                    and all(item.result_coverage)
                    and item.explicit_answer
                )
            )
        ]
        method_priority = {
            "tagged_solution": 4,
            "label_boxed": 3,
            "label": 3,
            "label_next_line_boxed": 3,
            "label_next_line": 3,
            "bracket_label": 3,
            "whole_solution": 2,
            "boxed": 1,
        }
        agreement: dict[int, int] = {}
        for candidate in usable:
            groups = {
                other.independence_group
                for other in usable
                if other is not candidate
                and other.independence_group != candidate.independence_group
                and equivalent_answers(
                    SubmissionAgent._comparison_value(candidate.answer),
                    SubmissionAgent._comparison_value(other.answer),
                )
            }
            agreement[id(candidate)] = len(groups)
        source_priority = {
            "primary": 4,
            "independent": 3,
            "blind_tiebreak": 2,
            "recovery": 1,
        }
        return max(usable, key=lambda item: (
            agreement[id(item)] > 0,
            agreement[id(item)],
            item.tool_status == "pass",
            item.passed_check_count,
            bool(item.result_coverage) and all(item.result_coverage),
            item.shape_valid,
            source_priority.get(item.source, 0),
            method_priority.get(item.extraction_method, 0),
            item.score,
        )) if usable else None

    @staticmethod
    def _rejected_explicit_consensus(
        candidates: Iterable[CandidateAssessment],
        spec: ProblemSpec,
    ) -> CandidateAssessment | None:
        """Recover only an independently repeated answer-only conclusion.

        A false arithmetic identity in a response body is a valid reason to
        reject that whole response.  It should not, however, force a random
        shape sentinel when two independent calls emit the same clean FINAL
        value.  This narrow fallback accepts only that case; it never weakens
        proof/support obligations or a decisive local mathematical failure.
        """
        if spec.answer_contract.mode != "answer_only":
            return None
        eligible = [
            item
            for item in candidates
            if item.answer
            and item.validation_tier == "rejected"
            and item.explicit_answer
            and item.complete_goals
            and item.shape_valid
            and item.formatting_valid
            and not item.failed_check
            and set(item.rejected_reasons) == {"numeric_identity_conflict"}
        ]
        if len(eligible) < 2:
            return None

        agreeing: list[tuple[CandidateAssessment, int]] = []
        for candidate in eligible:
            groups = {
                other.independence_group
                for other in eligible
                if other is not candidate
                and other.independence_group != candidate.independence_group
                and equivalent_answers(
                    SubmissionAgent._comparison_value(candidate.answer),
                    SubmissionAgent._comparison_value(other.answer),
                )
            }
            if groups:
                agreeing.append((candidate, len(groups)))
        if not agreeing:
            return None
        return max(
            agreeing,
            key=lambda pair: (
                pair[1],
                pair[0].passed_check_count,
                pair[0].score,
                pair[0].source == "primary_recovery",
            ),
        )[0]

    @staticmethod
    def _emergency_explicit_intent(
        candidates: Iterable[CandidateAssessment],
        spec: ProblemSpec,
    ) -> CandidateAssessment | None:
        """Keep one unambiguous FINAL value after the clean emergency call.

        This is deliberately weaker than ordinary admission and is used only
        after the bounded production pipeline has returned no answer at all.
        A body-level arithmetic inconsistency may reject the response, but a
        single explicit, structurally complete intended value is still more
        grounded than inventing a value from the expected answer shape.
        """
        if spec.answer_contract.mode != "answer_only":
            return None
        eligible = [
            item
            for item in candidates
            if item.answer
            and item.validation_tier == "rejected"
            and item.explicit_answer
            and item.complete_goals
            and item.shape_valid
            and item.formatting_valid
            and not item.failed_check
            and set(item.rejected_reasons) == {"numeric_identity_conflict"}
        ]
        if not eligible:
            return None
        anchor = eligible[-1]
        if any(
            not equivalent_answers(
                SubmissionAgent._comparison_value(anchor.answer),
                SubmissionAgent._comparison_value(item.answer),
            )
            for item in eligible[:-1]
        ):
            return None
        return anchor

    @staticmethod
    def _normalize_candidate(value: str, spec: ProblemSpec) -> str:
        answer = str(value or "").strip()
        if spec.profile.answer_shape == "choice":
            answer = canonical_choice_answer(answer, spec.problem_text)
        if spec.profile.answer_shape == "truth" and re.fullmatch(
            r"(?:是|否|正确|错误|成立|不成立|true|false|yes|no)[。.!]?",
            answer,
            re.IGNORECASE,
        ):
            subject = spec.answer_frame.subject.strip()
            if subject:
                separator = "：" if spec.profile.language == "zh" else ": "
                answer = f"{subject}{separator}{answer}"
        return answer

    @staticmethod
    def _render_submission(answer: str, spec: ProblemSpec) -> str:
        value = str(answer or "").strip()
        if spec.answer_contract.mode != "answer_only":
            if (
                spec.answer_contract.wrapper == "boxed"
                and "\n" not in value
                and not re.search(r"\\boxed\s*\{", value)
                and not re.search(r"^(?:结论|Conclusion)\s*[:：]", value, re.IGNORECASE)
            ):
                return rf"\boxed{{{value}}}"
            replacement = "结论：" if spec.profile.language == "zh" else "Conclusion: "
            value = re.sub(
                r"(?i)^\s*(?:FINAL(?:\s+ANSWER)?|【\s*最终答案\s*】|最终答案|答案|CONCLUSION)\s*[:：=]?\s*",
                replacement,
                value,
                count=1,
            )
            value = re.sub(
                r"(?i)^(?:结论\s*[:：]\s*){2,}",
                "结论：",
                value,
                count=1,
            )
            value = re.sub(
                r"(?i)^(?:Conclusion\s*:\s*){2,}",
                "Conclusion: ",
                value,
                count=1,
            )
            value = re.sub(
                r"(?im)^\s*(?:DECISION|CHECK)\s*[:：].*$",
                "",
                value,
            )
            if (
                spec.answer_contract.wrapper == "boxed"
                and not re.search(r"\\boxed\s*\{", value)
            ):
                lines = value.splitlines()
                if lines:
                    labelled = re.match(
                        r"^\s*(结论\s*[:：]|Conclusion\s*:)\s*(\S.*)$",
                        lines[0],
                        re.IGNORECASE,
                    )
                    if labelled:
                        lines[0] = (
                            f"{labelled.group(1)} "
                            rf"\boxed{{{labelled.group(2).strip()}}}"
                        )
                        value = "\n".join(lines)
            if spec.answer_contract.wrapper != "boxed":
                value = re.sub(
                    r"(?m)^\s*\\boxed\s*\{(.+)\}\s*[。.]?\s*$",
                    lambda match: replacement + match.group(1).strip(),
                    value,
                )
            value = re.sub(r"\n{3,}", "\n\n", value).strip()
            return value
        extracted = Finalizer.extract_result(value)
        if extracted.valid and extracted.explicit_answer and extracted.answer:
            value = extracted.answer.strip()
        if spec.answer_contract.wrapper == "boxed" and not re.fullmatch(
            r"\\boxed\s*\{.*\}", value, re.DOTALL
        ):
            return rf"\boxed{{{value}}}"
        return value

    @staticmethod
    def _shape_fallback(spec: ProblemSpec) -> str:
        shape = spec.profile.answer_shape
        if shape == "choice":
            from classifier.choice import option_labels

            labels = option_labels(spec.problem_text)
            value = labels[0] if labels else "A"
            return (
                rf"\boxed{{{value}}}"
                if spec.answer_contract.wrapper == "boxed"
                else value
            )
        if shape == "truth":
            subject = spec.answer_frame.subject or ("该命题" if spec.profile.language == "zh" else "The statement")
            value = (
                f"{subject}不成立。"
                if spec.profile.language == "zh"
                else f"{subject} is false."
            )
            return (
                rf"\boxed{{{value}}}"
                if spec.answer_contract.wrapper == "boxed"
                else value
            )
        if shape in {"roots", "interval"}:
            return r"\varnothing"
        if shape == "matrix":
            return r"\begin{pmatrix}0\end{pmatrix}"
        return r"\boxed{0}" if spec.answer_contract.wrapper == "boxed" else "0"

    @staticmethod
    def _truncated(result: ModelCallResult, raw: str) -> bool:
        if result.provider_truncated:
            return True
        reasons = set(Finalizer.validate_structure(raw)) if raw else {"empty"}
        return bool(reasons.intersection({
            "unclosed_code_fence", "unclosed_inline_math", "unclosed_inline_latex",
            "unclosed_display_latex", "unclosed_latex_environment",
            "unclosed_latex_brace", "unclosed_group_delimiter",
            "trailing_fragment", "truncated_sentence",
        }))

    @staticmethod
    def _deep_reasoning(spec: ProblemSpec) -> bool:
        if spec.answer_contract.mode != "answer_only":
            return True
        # Short factual choice and truth questions benefit from two compact,
        # independent checks. Hard variants still need conflict arbitration.
        if (
            spec.profile.answer_shape in {"choice", "truth"}
            and spec.profile.difficulty != "hard"
            and len(spec.goals) == 1
        ):
            return False
        strict_requirements = {
            requirement.name
            for goal in spec.goals
            for requirement in goal.requirements
            if requirement.strict
        }
        if strict_requirements.intersection({
            "all_solutions",
            "construction_object",
            "construction_check",
            "method_formula",
            "first_iteration",
        }):
            return True
        if spec.profile.task_kind in {
            "proof", "derivation", "explanation", "construction"
        }:
            return True
        if spec.profile.topic.startswith("olympiad_"):
            return True
        if spec.profile.difficulty == "hard":
            return True
        if spec.profile.answer_shape == "count":
            return True
        if any(
            signal.partition(":")[2] == "combinatorics"
            for signal in spec.profile.matched_signals
        ):
            return True
        text = str(spec.problem_text or "")
        if spec.risk_score >= 2 and len(text) >= 48:
            return True
        advanced_subjects = {
            "离散数学", "数值分析", "测度积分", "微分几何", "概率论", "抽象代数",
            "随机过程", "复分析", "常微分方程", "统计推断", "泛函分析", "线性回归",
            "偏微分方程", "非基础及进阶课程", "高等代数", "运筹学", "数学分析",
            "拓扑学", "欧氏几何", "数论",
        }
        if spec.profile.primary_subject in advanced_subjects and len(text) >= 40:
            return True
        if spec.verification_required or spec.risk_score >= 3:
            return True
        if re.search(
            r"\b(?:find|determine|classify)\s+all\s+"
            r"(?:functions?|polynomials?|sequences?|sets?)\b|"
            r"\bnumber\s+of\s+(?:ordered\s+)?(?:real\s+)?"
            r"(?:triples?|tuples?|solutions?)\b|"
            r"fourier\s*(?:transform|transformation|变换)|傅里叶变换|"
            r"(?:求|确定|找出|分类)(?:出)?所有[^。！？!?\n]{0,80}"
            r"(?:函数|多项式|数列|序列|集合)|"
            r"(?:有序)?(?:三元组|多元组|解)的(?:个数|数目)",
            text,
            re.IGNORECASE,
        ):
            return True
        return bool(
            len(text) >= 180
            and spec.profile.answer_shape not in {"choice", "truth"}
        )

    @staticmethod
    def _hidden_thinking(spec: ProblemSpec) -> bool:
        """Use hidden reasoning whenever the risk router selects a deep solve."""
        return SubmissionAgent._deep_reasoning(spec)

    @staticmethod
    def _remaining_seconds(started_at: float) -> float:
        return max(0.0, SUBMISSION_SOFT_ITEM_SECONDS - (monotonic() - started_at))

    @staticmethod
    def _supports_keyword(call, keyword: str) -> bool:
        try:
            signature = inspect.signature(call)
        except (TypeError, ValueError):
            return False
        return keyword in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    @staticmethod
    def _review_reason(budget: StageBudget, usable: bool, truncated: bool) -> str:
        if budget.require_independent_review:
            return "risk_requires_independent_solution"
        if not usable:
            return "no_complete_primary_candidate"
        if truncated:
            return "unverified_truncated_primary"
        return "primary_candidate_sufficient"

    @staticmethod
    def _evidence_prompt(evidence: tuple[ToolEvidence, ...]) -> str:
        lines = []
        for item in evidence:
            if not item.verified:
                continue
            role = (
                "CERTIFIED_RECOMPUTATION"
                if item.assurance in {"symbolic", "exhaustive"}
                else "ADVISORY_SCHEMA"
            )
            lines.append(
                f"- [{role}] {item.operation} ({item.scope}): "
                f"{item.support or item.result}"
            )
        return "\n".join(lines)

    @staticmethod
    def _independent_evidence(
        evidence: tuple[ToolEvidence, ...],
    ) -> tuple[ToolEvidence, ...]:
        """Keep independent solves blind to theorem-template conclusions."""
        return tuple(
            item
            for item in evidence
            if item.assurance in {"symbolic", "exhaustive"}
        )

    @staticmethod
    def _tool_trace(results: tuple[ToolResult, ...], whole: ToolResult | None) -> dict:
        return {
            "result_count": len(results),
            "direct_operation": whole.operation if whole else "none",
            "results": [{
                "operation": result.operation,
                "result_kind": (
                    result.contract.result_kind if result.contract else "unknown"
                ),
                "assurance": (
                    result.contract.assurance if result.contract else "unknown"
                ),
                "whole_answer_eligible": result.whole_answer_eligible,
                "direct_submission_eligible": result.direct_submission_eligible,
                "certificate_passed": result.certificate.passed,
                "certificate_status": result.certificate.status.value,
                "certificate_phases_complete": result.certificate.phases_complete,
                "certificate_method": result.certificate.method,
                "certificate_check_count": len(result.certificate.checks),
                "certificate_issue_count": len(result.certificate.issues),
            } for result in results],
        }

    @staticmethod
    def _candidate_trace(candidate: CandidateAssessment, **extra) -> dict:
        content = {
            "source": candidate.source,
            "extraction": candidate.extraction_method,
            "validation_tier": candidate.validation_tier,
            "correctness_tier": candidate.correctness_tier,
            "complete_goals": candidate.complete_goals,
            "tool_status": candidate.tool_status,
            "checks": [{
                "name": item.name,
                "status": item.status,
                "source": item.source,
                "decisive": item.decisive,
            } for item in candidate.math_checks],
            "rejected_reasons": list(candidate.rejected_reasons),
            "transport_truncated": candidate.transport_truncated,
        }
        content.update(extra)
        return content

    @staticmethod
    def _blueprint_trace(spec: ProblemSpec) -> dict:
        """Return routing metadata without echoing the submitted problem."""
        profile = spec.profile
        return {
            "subject": profile.subject,
            "primary_subject": profile.primary_subject,
            "secondary_subject": profile.secondary_subject,
            "subject_confidence": profile.subject_confidence,
            "problem_type": profile.problem_type,
            "task_kind": profile.task_kind,
            "difficulty": profile.difficulty,
            "answer_shape": profile.answer_shape,
            "result_kind": profile.result_kind,
            "language": profile.language,
            "topic": profile.topic,
            "goal_count": len(spec.goals),
            "requirement_names": [
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            ],
            "risk_flags": list(spec.risk_flags),
            "risk_score": spec.risk_score,
            "verification_required": spec.verification_required,
            "primary_method": spec.primary_method,
            "alternative_method": spec.alternative_method,
            "answer_mode": spec.answer_contract.mode,
            "answer_wrapper": spec.answer_contract.wrapper or "none",
            "tool_can_answer_whole": spec.tool_can_answer_whole,
        }

    @staticmethod
    def _plan_trace(plan: SolvePlan, spec: ProblemSpec) -> dict:
        return {
            "source": plan.source,
            "valid": plan.valid,
            "subject": plan.effective_subject(spec),
            "local_subject": spec.profile.primary_subject,
            "method": plan.method,
            "risk_flag_count": len(spec.risk_flags),
        }

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        text = str(value or "")
        return text if len(text) <= limit else text[:limit] + "\n[content shortened]"

    @staticmethod
    def _grounded_context(
        spec: ProblemSpec,
        cards: RetrievalBundle,
        *,
        review: bool,
        plan: SolvePlan | None = None,
    ) -> str:
        """Inject optional guidance only when statement routing is trustworthy."""
        confidence = getattr(spec.profile, "subject_confidence", "low")
        signals = tuple(getattr(spec.profile, "matched_signals", ()))
        semantics = getattr(spec, "semantics", None)
        explicit_route = bool(
            getattr(semantics, "requested_methods", ())
            or getattr(semantics, "named_theorems", ())
        )
        planned_subject = plan.effective_subject(spec) if plan is not None else ""
        protocol_allowed = bool(
            confidence == "high"
            or explicit_route
            or (confidence == "medium" and len(signals) >= 2)
            or str(getattr(spec.profile, "topic", "")).startswith("olympiad_")
            or (plan is not None and plan.valid and bool(planned_subject))
            or planned_subject == "进阶数学"
        )
        relevant_scores = cards.review_scores if review else cards.solve_scores
        precise_card_match = max(relevant_scores, default=0) >= 18
        cards_allowed = bool(
            confidence in {"high", "medium"}
            or explicit_route
            or precise_card_match
        )
        return SubmissionAgent._join_context(
            subject_protocol(
                spec,
                review=review,
                subject_override=planned_subject,
            ) if protocol_allowed else "",
            (
                cards.review_context() if review else cards.solve_context()
            ) if cards_allowed else "",
        )

    @staticmethod
    def _join_context(*parts: str) -> str:
        return "\n".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def _draft_excerpt(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        explicit = [
            item.answer
            for item in Finalizer.extract_explicit_results(text)
            if item.answer and item.valid
        ]
        labelled = "\n".join(f"Explicit candidate: {item}" for item in explicit[-2:])
        if len(text) <= 6500:
            body = text
        else:
            body = text[:900] + "\n[earlier draft shortened]\n" + text[-5600:]
        return (labelled + "\n" + body).strip() if labelled else body

    @staticmethod
    def _candidate_work_context(
        candidate: CandidateAssessment,
        *,
        primary_raw: str,
        primary_recovery_raw: str,
        independent_source: str,
        independent_raw: str,
        independent_recovery_raw: str,
    ) -> str:
        """Return bounded derivation evidence for a conflict-only audit."""
        if candidate.source == "primary":
            drafts = (primary_raw,)
        elif candidate.source == "primary_recovery":
            drafts = (primary_raw, primary_recovery_raw)
        elif candidate.source == independent_source:
            drafts = (independent_raw, independent_recovery_raw)
        else:
            drafts = ()
        text = "\n\n[continuation]\n\n".join(
            str(item or "").strip() for item in drafts if str(item or "").strip()
        )
        if not text:
            return ""
        explicit = [
            item.answer
            for item in Finalizer.extract_explicit_results(text)
            if item.valid and item.answer
        ]
        labelled = "\n".join(
            f"Explicit conclusion in draft: {item}" for item in explicit[-2:]
        )
        if len(text) > 3600:
            text = text[:700] + "\n[work shortened]\n" + text[-2700:]
        return (labelled + "\n" + text).strip() if labelled else text

    @staticmethod
    def _load_prompt() -> str:
        path = Path(__file__).resolve().parent.parent / "prompts" / "submission.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return (
                "Solve the mathematics. Put a checked complete answer on the first "
                "line as FINAL: <answer>; include essential proof when required."
            )
