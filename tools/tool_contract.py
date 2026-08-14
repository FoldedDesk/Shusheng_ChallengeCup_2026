"""Structured contracts and certificates for deterministic math tools.

The public agent historically consumed strings such as ``"SymPy 计算: 4"``.
This module keeps that representation available while giving the solver an
explicit allow-list: an unknown label is evidence text, never a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Iterable, Optional


def problem_fingerprint(problem: str) -> str:
    """Return a stable digest binding a certificate to one normalized prompt."""

    normalized = re.sub(r"\s+", " ", str(problem or "")).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolContract:
    """Static safety boundary for one deterministic operation.

    ``whole_answer_capable`` means that the operation may be considered for a
    whole-answer route.  The caller must still compare ``max_goals`` and
    ``allowed_requirements`` with the parsed problem contract.
    """

    operation: str
    result_kind: str
    certificate_method: str
    whole_answer_capable: bool = False
    max_goals: int = 1
    allowed_requirements: tuple[str, ...] = ()
    required_requirements: tuple[str, ...] = ()
    allowed_task_kinds: tuple[str, ...] = ("calculation", "fill_blank", "construction")
    allowed_answer_shapes: tuple[str, ...] = ()
    required_problem_facts: tuple[str, ...] = ()
    forbidden_problem_facts: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        return bool(self.certificate_method)

    def covers(
        self,
        goal_count: int,
        requirements: Iterable[str] = (),
        *,
        task_kind: str = "",
        answer_shape: str = "",
        problem_facts: Optional[Iterable[str]] = None,
    ) -> bool:
        """Return whether a parsed answer contract fits this static boundary."""

        requested = set(requirements)
        facts = None if problem_facts is None else set(problem_facts)
        return bool(
            self.whole_answer_capable
            and 1 <= goal_count <= self.max_goals
            and set(self.required_requirements) <= requested
            and requested <= set(self.allowed_requirements)
            and (
                facts is None
                or (
                    set(self.required_problem_facts) <= facts
                    and not set(self.forbidden_problem_facts) & facts
                )
            )
            and (
                not task_kind
                or not self.allowed_task_kinds
                or task_kind in self.allowed_task_kinds
            )
            and (
                not answer_shape
                or not self.allowed_answer_shapes
                or answer_shape in self.allowed_answer_shapes
            )
        )


@dataclass(frozen=True)
class ToolCertificate:
    """Machine-readable record of what made a tool result trustworthy."""

    passed: bool
    method: str
    checks: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    source_fingerprint: str = ""

    def trace_content(self) -> dict:
        return {
            "passed": self.passed,
            "method": self.method,
            "checks": list(self.checks),
            "issues": list(self.issues),
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class ToolResult:
    """A deterministic result paired with its scope contract and certificate."""

    result: str
    operation: str
    label: str
    contract: Optional[ToolContract]
    certificate: ToolCertificate
    support: str = ""

    @property
    def verified(self) -> bool:
        return self.certificate.passed

    @property
    def whole_answer_eligible(self) -> bool:
        return bool(
            self.certificate.passed
            and self.contract is not None
            and self.contract.whole_answer_capable
        )

    def to_hint(self) -> str:
        return f"{self.label}: {self.support or self.result}"

    def trace_content(self) -> dict:
        return {
            "operation": self.operation,
            "result_kind": self.contract.result_kind if self.contract else "unknown",
            "whole_answer_eligible": self.whole_answer_eligible,
            "certificate": self.certificate.trace_content(),
            "support": self.support,
        }


def _legacy_submission_payload(operation: str, raw_result: str) -> tuple[str, str]:
    """Split legacy explanatory text into a gradable result and its support.

    Older deterministic handlers returned a proof or calculation sentence as
    their result.  That is useful evidence, but it must not become the literal
    final answer for ordinary calculation and judgement questions.  Each
    extractor below is tied to one registered deterministic producer; an
    unknown format remains unchanged instead of being guessed from arbitrary
    text.
    """

    value = str(raw_result or "").strip()
    compact = ""

    if operation in {
        "finite_cyclic_subgroup_count",
        "linear_nonadjacent_selection",
        "deleted_edge_bipartite_length_three_paths",
        "positive_composition_lower_bounds",
        "nonadjacent_binary_string_count",
        "precedence_permutation_count",
        "surjection_count",
    }:
        matches = re.findall(r"=\s*(-?\d+)\s*\\\)", value)
        compact = matches[-1] if matches else ""
    elif operation == "even_subset_count":
        match = re.search(
            r"(?:偶数基数子集数为|The number of even-cardinality subsets is)\s*"
            r"\\\(([^()]+)\\\)",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1).strip() if match else ""
    elif operation == "binomial_choose_two_positive_root":
        if re.search(
            r"^(?:无正整数解|There is no positive-integer solution)",
            value,
            re.IGNORECASE,
        ):
            compact = (
                "无正整数解"
                if value.startswith("无")
                else "No positive integer solution"
            )
            return compact, value
        match = re.search(
            r"(?:正整数解为|The positive-integer solution is)\s*\\\(([^()]+)\\\)",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1).strip() if match else ""
    elif operation == "ordered_positive_triples":
        match = re.search(r"(?:共\s*|total(?:\s+is)?\s*)(\d+)\s*(?:个)?", value, re.IGNORECASE)
        compact = match.group(1) if match else ""
    elif operation == "fair_coin_geometric_tail":
        match = re.search(
            r"P\s*\(\s*T\s*>\s*\d+\s*\)\s*=\s*"
            r"(?:\(1/2\)\^\{?\d+\}?\s*=\s*)?"
            r"(\\frac\{\d+\}\{\d+\}|\d+)",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1) if match else ""
    elif operation == "bounded_self_exponential_divisibility":
        match = re.search(
            r"(?:完整解集为|The complete set is)\s*\\\((\\\{[^()]*\\\}|\\varnothing)\\\)",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1) if match else ""
    elif operation == "competing_coin_patterns":
        match = re.search(
            r"(?:所求[^。.!?]{0,40}概率为|The probability that[^.?!]{0,60}?is)\s*"
            r"\\\((\\frac\{\d+\}\{\d+\}|\d+)\\\)",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1) if match else ""
    elif operation == "poisson_process_increment":
        match = re.search(
            r"(N\([^()]+\)-N\([^()]+\)\\mid\s*N\([^()]+\)=\d+\\sim"
            r"\\operatorname\{Poisson\}\([^()]+\))",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1) if match else ""
    elif operation == "independent_event_union":
        match = re.search(
            r"P\s*\(\s*A\s*\\cup\s*B\s*\)[^。.!?]*=\s*"
            r"(\\frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?)\\\)[。.]?\s*$",
            value,
            re.IGNORECASE,
        )
        compact = rf"P(A\cup B)={match.group(1)}" if match else ""
    elif operation == "independent_standard_normal_sum":
        match = re.search(
            r"([A-Z]\+[A-Z]\\sim\s*N\(0,2\),\\quad\s*"
            r"\\operatorname\{Var\}\([A-Z]\+[A-Z]\)=2)",
            value,
        )
        compact = match.group(1) if match else ""
    elif operation == "brownian_covariance":
        match = re.search(
            r"(\\operatorname\{Cov\}\(B\([a-z]\),B\([a-z]\)\)=[a-z])",
            value,
            re.IGNORECASE,
        )
        compact = match.group(1) if match else ""
    elif operation == "sample_mean_variance":
        matches = re.findall(
            r"\\operatorname\{Var\}\(\\bar\s*X\)[^。.!?]*=\s*"
            r"(\\frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?)",
            value,
        )
        compact = rf"\operatorname{{Var}}(\bar X)={matches[-1]}" if matches else ""
    elif operation == "renewal_rate_limit":
        match = re.search(
            r"(\\lim_\{t\\to\\infty\}\\frac\{N\(t\)\}\{t\}="
            r"(?:\\frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?))",
            value,
        )
        compact = match.group(1) if match else ""
    elif operation == "central_difference":
        match = re.search(r"\\approx\s*(-?\d+(?:\.\d+)?)\s*$", value)
        compact = match.group(1) if match else ""
    elif operation == "two_point_gauss_legendre_monomial":
        match = re.fullmatch(r"-?\d+(?:/\d+)?", value)
        compact = match.group(0) if match else ""

    if not compact or compact == value:
        return value, ""
    return compact, value


def _contract(
    operation: str,
    result_kind: str,
    method: str,
    whole: bool = False,
    *,
    max_goals: int = 1,
    requirements: tuple[str, ...] = (),
    required_requirements: tuple[str, ...] = (),
    task_kinds: tuple[str, ...] = ("calculation", "fill_blank", "construction"),
    answer_shapes: tuple[str, ...] = (),
    facts: tuple[str, ...] = (),
    forbidden: tuple[str, ...] = ("extra_uncovered_obligation",),
) -> ToolContract:
    return ToolContract(
        operation=operation,
        result_kind=result_kind,
        certificate_method=method,
        whole_answer_capable=whole,
        max_goals=max_goals,
        allowed_requirements=requirements,
        required_requirements=required_requirements,
        allowed_task_kinds=task_kinds,
        allowed_answer_shapes=answer_shapes,
        required_problem_facts=facts,
        forbidden_problem_facts=forbidden,
    )


# This is deliberately an explicit allow-list.  Adding a string prefix to a
# tool must not silently turn it into verified evidence or a whole-answer route.
TOOL_CONTRACTS: dict[str, ToolContract] = {
    "calculate": _contract(
        "calculate", "scalar", "sympy_exact", True,
        answer_shapes=("number", "expression"),
    ),
    "solve_equation": _contract(
        "solve_equation", "solution_set", "sympy_exact", True,
        requirements=("exhaustive_result",),
        answer_shapes=("roots",),
    ),
    "derivative": _contract(
        "derivative", "expression", "sympy_exact", True,
        answer_shapes=("expression",),
    ),
    "definite_integral": _contract(
        "definite_integral", "scalar_or_expression", "sympy_exact", True,
        requirements=("integral_value",),
        answer_shapes=("number", "expression"),
    ),
    "integral": _contract(
        # An antiderivative expression alone does not certify the full answer:
        # SymPy omits the arbitrary constant, may require absolute values, and
        # can legitimately return an unreadable RootSum/unevaluated form.
        "integral", "antiderivative", "sympy_exact",
        requirements=("integral_value",),
        answer_shapes=("expression",),
    ),
    "limit": _contract(
        "limit", "scalar_or_expression", "sympy_exact", True,
        answer_shapes=("number", "expression"),
    ),
    "matrix": _contract("matrix", "matrix", "sympy_exact"),
    "recurrence_solution": _contract("recurrence_solution", "sequence_formula", "sympy_exact"),
    "curve_speed": _contract(
        "curve_speed", "formula_and_value", "symbolic_substitution", True,
        max_goals=2, requirements=("judgement", "first_second_derivatives"),
    ),
    "first_fundamental_form": _contract("first_fundamental_form", "quadratic_form", "symbolic_differentiation", True),
    "graph_gaussian_curvature": _contract(
        "graph_gaussian_curvature", "formula", "symbolic_identity", True,
        requirements=("gaussian_curvature",),
    ),
    "pde_verification": _contract(
        "pde_verification", "truth_with_substitution", "symbolic_substitution", True,
        requirements=("judgement", "pde_time_space_derivatives", "laplace_second_derivatives"),
    ),
    "propositional_implication_chain": _contract(
        "propositional_implication_chain", "proof", "formal_deduction", True,
        requirements=("reasoning", "inference_rule"),
        task_kinds=("calculation", "proof", "explanation"),
    ),
    "minimum_degree_path_proof": _contract(
        "minimum_degree_path_proof", "proof", "graph_invariant", True,
        requirements=("reasoning",),
        task_kinds=("proof",),
        answer_shapes=("proof",),
    ),
    "even_subset_count": _contract(
        "even_subset_count", "count_with_bijection", "parity_bijection_and_binomial_identity", True,
        requirements=("reasoning", "counting_method"),
        task_kinds=("calculation", "proof", "explanation"),
        answer_shapes=("number", "expression"),
        facts=(
            "finite_set_with_positive_size", "all_subsets_requested",
            "even_cardinality_constraint", "even_odd_toggle_bijection",
            "binomial_parity_identity",
        ),
    ),
    "deleted_edge_bipartite_length_three_paths": _contract(
        "deleted_edge_bipartite_length_three_paths", "count", "two_layer_choice_product", True,
        answer_shapes=("number",),
        facts=(
            "complete_bipartite_graph", "exactly_one_deleted_edge",
            "missing_edge_endpoints", "length_three_simple_paths",
            "two_layer_choice_product",
        ),
    ),
    "positive_composition_lower_bounds": _contract(
        "positive_composition_lower_bounds", "count", "lower_bound_shift_and_stars_bars", True,
        requirements=("variable_shift", "stars_and_bars", "counting_method", "exhaustive_result"),
        answer_shapes=("number",),
        facts=(
            "explicit_positive_integer_variables", "unit_coefficient_sum_equation",
            "explicit_per_variable_lower_bounds", "variable_shift_to_nonnegative",
            "stars_and_bars_recomputed",
        ),
    ),
    "binomial_choose_two_positive_root": _contract(
        "binomial_choose_two_positive_root", "positive_integer_solution", "quadratic_integer_root_filter", True,
        requirements=("exhaustive_result", "reasoning"),
        task_kinds=("calculation", "proof", "explanation"),
        answer_shapes=("roots", "number", "expression"),
        facts=(
            "choose_two_equation", "positive_integer_domain",
            "quadratic_discriminant_checked", "all_quadratic_roots_checked",
            "positive_integer_roots_exhausted",
        ),
    ),
    "finite_cyclic_subgroup_count": _contract(
        "finite_cyclic_subgroup_count", "count_with_correspondence", "divisor_lattice_of_cyclic_group", True,
        requirements=("reasoning",),
        task_kinds=("calculation", "proof", "explanation"),
        answer_shapes=("number",),
        facts=(
            "finite_cyclic_group", "explicit_group_order", "all_subgroups_count_requested",
            "positive_divisor_correspondence", "prime_exponents_recomputed",
        ),
    ),
    "linear_nonadjacent_selection": _contract(
        "linear_nonadjacent_selection", "count", "position_compression_bijection", True,
        requirements=("position_compression", "counting_method"),
        answer_shapes=("number", "expression"),
        facts=(
            "linear_consecutive_integer_set", "exact_selection_size",
            "pairwise_nonadjacent_constraint", "position_compression_bijection",
            "binomial_count_recomputed",
        ),
    ),
    "nonadjacent_binary_string_count": _contract(
        "nonadjacent_binary_string_count", "count", "bijection_and_closed_form", True,
        requirements=("position_selection",),
    ),
    "precedence_permutation_count": _contract(
        "precedence_permutation_count", "count", "symmetry_and_subtraction", True,
        requirements=("counting_method",),
    ),
    "surjection_count": _contract(
        "surjection_count", "count", "inclusion_exclusion", True,
        requirements=("inclusion_exclusion",),
    ),
    "planar_euler_faces": _contract(
        "planar_euler_faces", "count_with_check", "euler_identity", True,
        requirements=("euler_formula_check",),
    ),
    "paraboloid_curvature": _contract(
        "paraboloid_curvature", "curvature_tuple", "derivative_and_shape_operator", True,
        max_goals=2,
        requirements=("principal_curvatures", "gaussian_curvature", "surface_second_derivatives"),
    ),
    "ordered_positive_triples": _contract(
        "ordered_positive_triples", "count", "exact_enumeration", True,
        requirements=("case_split", "exhaustive_result"),
    ),
    "fair_dice_conditional_probability": _contract(
        "fair_dice_conditional_probability", "probability_and_optional_sample_space",
        "finite_ordered_outcome_enumeration", True,
        max_goals=2,
        requirements=("conditional_sample_space",),
        answer_shapes=("number", "expression"),
        facts=(
            "exactly_two_ordered_rolls", "fair_die", "standard_or_explicit_six_sided_die",
            "condition_is_exact_sum", "first_outcome_target", "nonempty_conditioning_event",
            "conditional_sample_space_enumerated", "favorable_outcome_counted",
            "conditional_ratio_reduced", "no_extra_probability_obligation",
        ),
    ),
    "bernoulli_centered_second_moment": _contract(
        "bernoulli_centered_second_moment", "centered_moment_and_variance_identity",
        "bernoulli_two_point_exact_expectation", True,
        max_goals=2,
        requirements=("target_e", "variance_identification"),
        answer_shapes=("number", "expression"),
        facts=(
            "single_bernoulli_variable", "symbolic_parameter",
            "center_matches_bernoulli_parameter", "second_power_exact", "support_zero_one",
            "probabilities_one_minus_p_and_p", "two_point_expansion_checked",
            "variance_identity_checked", "no_extra_statistical_obligation",
        ),
    ),
    "fair_coin_geometric_tail": _contract(
        "fair_coin_geometric_tail", "probability", "geometric_tail_identity", True,
        requirements=("geometric_distribution_identification",),
        answer_shapes=("number", "expression"),
        facts=(
            "single_fair_coin_sequence", "stop_at_first_head",
            "geometric_support_starts_at_one", "strict_greater_than_tail_requested",
            "tail_exponent_recomputed", "exact_reduced_probability",
            "no_extra_probability_obligation",
        ),
    ),
    "poisson_process_increment": _contract(
        "poisson_process_increment", "conditional_distribution",
        "poisson_independent_increment_law", True,
        requirements=("distribution_result", "independent_increments"),
        answer_shapes=("number", "expression"),
        facts=(
            "homogeneous_poisson_process", "explicit_rate", "single_forward_increment",
            "conditioning_is_past_endpoint_count", "independent_increment_applied",
            "increment_length_recomputed", "conditional_distribution_requested",
            "no_extra_stochastic_obligation",
        ),
    ),
    "cauchy_location_fisher_information": _contract(
        "cauchy_location_fisher_information", "fisher_information",
        "exact_cauchy_score_integral", True,
        answer_shapes=("number", "expression"),
        facts=(
            "iid_sample", "unit_scale_cauchy_location_density",
            "density_normalization_checked", "location_score_squared_integrated",
            "per_observation_information_one_half", "iid_information_scaling",
            "sample_fisher_information_requested", "no_extra_inference_obligation",
        ),
    ),
    "one_dimensional_wald_statistic": _contract(
        "one_dimensional_wald_statistic", "scalar",
        "exact_linear_contrast_quadratic_form", True,
        answer_shapes=("number", "expression"),
        facts=(
            "explicit_two_parameter_estimate", "explicit_symmetric_covariance_matrix",
            "single_linear_zero_constraint", "contrast_value_recomputed",
            "contrast_variance_recomputed", "positive_contrast_variance",
            "exact_wald_statistic_reduced", "no_extra_inference_obligation",
        ),
    ),
    "diagonal_gls_estimate": _contract(
        "diagonal_gls_estimate", "vector",
        "exact_weighted_normal_equations", True,
        answer_shapes=("number", "expression"),
        facts=(
            "explicit_design_matrix", "explicit_response_vector",
            "positive_diagonal_covariance_shape", "covariance_scale_cancels",
            "weighted_normal_matrix_recomputed", "weighted_rhs_recomputed",
            "normal_matrix_nonsingular", "exact_gls_solution_verified",
            "no_extra_regression_obligation",
        ),
    ),
    "normal_variance_confidence_interval": _contract(
        "normal_variance_confidence_interval", "confidence_interval",
        "exact_chi_square_interval_inversion", True,
        answer_shapes=("interval", "expression"),
        facts=(
            "normal_population", "unknown_mean_sum_of_squares",
            "explicit_positive_sample_size", "degrees_of_freedom_recomputed",
            "two_sided_confidence_level", "matching_chi_square_quantiles",
            "chi_square_interval_inversion", "closed_interval_endpoints_recomputed",
            "rounded_endpoints_checked", "no_extra_inference_obligation",
        ),
    ),
    "two_state_markov_entropy_rate": _contract(
        "two_state_markov_entropy_rate", "scalar", "stationary_binary_entropy_identity", True,
        answer_shapes=("number", "expression"),
    ),
    "independent_event_union": _contract(
        "independent_event_union", "probability", "independent_union_identity", True,
        requirements=("independence_use",),
        answer_shapes=("number", "expression"),
        facts=(
            "exactly_two_independent_events", "both_marginal_probabilities_explicit",
            "union_probability_requested", "intersection_product_recomputed",
            "inclusion_exclusion_applied", "exact_reduced_probability",
            "no_extra_event_obligation",
        ),
    ),
    "independent_standard_normal_sum": _contract(
        "independent_standard_normal_sum", "distribution_and_variance",
        "normal_convolution_parameters", True,
        requirements=("distribution_result", "variance_result", "independence_use"),
        max_goals=2,
        answer_shapes=("expression",),
        facts=(
            "exactly_two_independent_variables", "both_standard_normal",
            "unweighted_sum_requested", "distribution_and_variance_requested",
            "means_added", "variances_added", "no_extra_normal_obligation",
        ),
    ),
    "brownian_covariance": _contract(
        "brownian_covariance", "covariance", "brownian_independent_increment_identity", True,
        # The deterministic support explicitly gives
        # B(t)=B(s)+(B(t)-B(s)), invokes independent increments, and derives
        # the covariance.  It therefore covers both the named method and the
        # generic reasoning obligation added for "explain the derivation".
        requirements=("reasoning", "independent_increments"),
        answer_shapes=("number", "expression"),
        facts=(
            "standard_brownian_motion", "ordered_nonnegative_times",
            "two_time_covariance_requested", "independent_increment_decomposition",
            "minimum_time_selected", "no_extra_brownian_obligation",
        ),
    ),
    "sample_mean_variance": _contract(
        "sample_mean_variance", "variance", "iid_variance_scaling", True,
        requirements=("iid_variance_scaling",),
        answer_shapes=("number", "expression"),
        facts=(
            "iid_sample", "explicit_population_variance", "explicit_positive_sample_size",
            "sample_mean_variance_requested", "variance_additivity_applied",
            "mean_scaling_squared", "no_finite_population_correction",
            "no_extra_sampling_obligation",
        ),
    ),
    "renewal_rate_limit": _contract(
        "renewal_rate_limit", "limit", "renewal_strong_law_rate", True,
        requirements=("strong_law",),
        answer_shapes=("number", "expression"),
        facts=(
            "ordinary_renewal_process", "explicit_finite_positive_interarrival_mean",
            "counting_rate_limit_requested", "strong_law_applied",
            "reciprocal_mean_recomputed", "no_extra_renewal_obligation",
        ),
    ),
    "finite_discrete_moments": _contract(
        "finite_discrete_moments", "moment_pair", "exact_finite_probability_sum", True,
        max_goals=2,
        answer_shapes=("number", "expression"),
        facts=(
            "explicit_finite_support", "matching_probability_table",
            "probabilities_sum_to_one", "expectation_and_variance_requested",
        ),
    ),
    "two_sided_z_rejection": _contract(
        "two_sided_z_rejection", "rejection_region", "standard_normal_quantile", True,
        answer_shapes=("number", "expression"),
        facts=(
            "two_sided_z_test", "explicit_significance_level",
            "rejection_region_requested", "critical_value_requested",
        ),
    ),
    "simple_random_walk_moments": _contract(
        "simple_random_walk_moments", "moments", "iid_moment_identity", True,
        requirements=("independent_increments",),
    ),
    "complete_graph_cover_time": _contract("complete_graph_cover_time", "expectation", "coupon_collector_identity", True),
    "two_venue_capacity": _contract("two_venue_capacity", "minimum_integer", "exact_binomial_tail", True),
    "circle_laplacian": _contract(
        "circle_laplacian", "scalar", "ambient_second_derivatives", True,
        facts=("f=x^2+y^2", "circle_constraint", "explicit_ambient_operator"),
        forbidden=("explicit_intrinsic_operator", "operator_ambiguity", "modified_operator", "extra_uncovered_obligation"),
    ),
    "circle_laplace_beltrami": _contract(
        "circle_laplace_beltrami", "scalar", "constant_restriction", True,
        facts=("f=x^2+y^2", "circle_constraint", "explicit_intrinsic_operator"),
        forbidden=("explicit_ambient_operator", "operator_ambiguity", "modified_operator", "extra_uncovered_obligation"),
    ),
    "circle_laplacian_ambiguous": _contract(
        "circle_laplacian_ambiguous", "operator_alternatives", "operator_case_split",
        facts=("f=x^2+y^2", "circle_constraint", "operator_not_disambiguated"),
    ),
    "central_difference": _contract(
        "central_difference", "approximation", "formula_evaluation", True,
        requirements=("central_difference_formula",),
        required_requirements=("central_difference_formula",),
    ),
    "rational_f2_constraint": _contract("rational_f2_constraint", "field_value", "constraint_propagation", True),
    "digit_sum_window": _contract("digit_sum_window", "minimum_integer", "bounded_exhaustive_search", True),
    "number_writing_game": _contract("number_writing_game", "minimum_integer", "dynamic_programming", True),
    "path_independent_set_partition": _contract("path_independent_set_partition", "polynomial", "recurrence_and_coefficients", True),
    "lz78_encoding": _contract(
        "lz78_encoding", "decomposition_and_encoding", "deterministic_parser", True,
        max_goals=2,
        requirements=("phrase_decomposition", "encoded_string"),
        required_requirements=("phrase_decomposition", "encoded_string"),
    ),
    "spike_sequence_construction": _contract(
        "spike_sequence_construction", "construction", "construction_substitution", True,
        max_goals=2, requirements=("integral_result", "integral_value", "pointwise_limit"),
    ),
    "dependent_bernoulli_construction": _contract(
        "dependent_bernoulli_construction", "construction", "finite_probability_table", True,
        max_goals=2, requirements=("target_p",),
    ),
    "complete_multipartite_spanning_trees": _contract(
        "complete_multipartite_spanning_trees", "count", "matrix_tree_closed_form", True,
        answer_shapes=("number",),
    ),
    "quadratic_congruence_count": _contract(
        "quadratic_congruence_count", "count", "prime_power_crt_enumeration", True,
        answer_shapes=("number",),
    ),
    "bounded_self_exponential_divisibility": _contract(
        "bounded_self_exponential_divisibility", "solution_set",
        "bounded_modular_power_exhaustion", True,
        requirements=("exhaustive_result", "reasoning"),
        required_requirements=("exhaustive_result",),
        task_kinds=("calculation", "proof", "explanation"),
        answer_shapes=("number", "roots", "expression", "proof"),
        facts=(
            "positive_integer_variable", "explicit_finite_upper_bound",
            "single_self_exponential_divisibility_condition",
            "exact_integer_modular_exponentiation", "every_integer_in_range_enumerated",
            "reported_solutions_rechecked", "omitted_values_certified_to_fail",
        ),
    ),
    "competing_coin_patterns": _contract(
        "competing_coin_patterns", "probability_with_recursion",
        "prefix_automaton_exact_linear_system", True,
        requirements=("reasoning",),
        task_kinds=("calculation", "proof", "explanation"),
        answer_shapes=("number", "expression", "proof"),
        facts=(
            "single_fair_coin_sequence", "exactly_two_distinct_equal_length_patterns",
            "first_occurrence_probability_requested", "overlap_preserving_prefix_automaton",
            "all_proper_prefix_states_enumerated", "absorbing_boundary_values_assigned",
            "exact_rational_linear_system_solved", "initial_state_probability_rechecked",
            "no_extra_probability_obligation",
        ),
    ),
    "digit_permutation_divisibility": _contract(
        "digit_permutation_divisibility", "count", "bounded_exact_enumeration", True,
        answer_shapes=("number",),
    ),
    "bounded_digit_set_divisibility_count": _contract(
        "bounded_digit_set_divisibility_count", "count",
        "decimal_remainder_dynamic_programming", True,
        answer_shapes=("number",),
        facts=(
            "decimal_digit_alphabet_parsed", "bounded_decimal_length",
            "canonical_positive_representations", "leading_zero_excluded",
            "domain_zero_convention_resolved", "single_divisibility_condition",
            "modular_remainder_dynamic_programming", "all_lengths_enumerated",
            "state_mass_invariant",
        ),
    ),
    "prime_floor_inequality_rank": _contract(
        "prime_floor_inequality_rank", "ordinal_integer",
        "floor_quotient_set_characterization", True,
        answer_shapes=("number", "expression"),
        facts=(
            "prime_parameter", "positive_integer_n_below_p",
            "exact_floor_inequality", "universal_k_range_zero_to_p_minus_two",
            "ordinal_rank_parsed", "lower_bound_implies_distinct_rank",
            "floor_quotient_set_characterization", "distinct_floor_quotients",
        ),
    ),
    "real_functional_equation_three_solutions": _contract(
        "real_functional_equation_three_solutions", "function_solution_set",
        "symbolic_substitution_and_exhaustive_case_split", True,
        requirements=("exhaustive_result",),
        required_requirements=("exhaustive_result",),
        answer_shapes=("expression", "roots"),
        facts=(
            "real_self_map", "universal_two_real_parameters",
            "exact_functional_equation", "three_candidate_identities_verified",
            "zero_substitution_case_split", "all_function_branches_exhausted",
        ),
    ),
    "nice_positive_integer_function_value_set": _contract(
        "nice_positive_integer_function_value_set", "exhaustive_value_set",
        "monotonicity_growth_bootstrap_and_explicit_constructions", True,
        requirements=("exhaustive_result",),
        required_requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
        facts=(
            "positive_integer_self_map", "universal_positive_integer_pair",
            "exact_nested_composition_inequality", "single_direct_value_target",
            "all_values_requested", "monotonicity_deduced", "growth_bootstrap_upper_bound",
            "all_values_have_explicit_constructions",
        ),
    ),
    "open_interval_quadratic_minimum_dimension": _contract(
        "open_interval_quadratic_minimum_dimension", "minimum_integer",
        "convex_extreme_bound_and_explicit_interior_construction", True,
        answer_shapes=("number",),
        facts=(
            "open_unit_interval_variables", "strict_sum_bound",
            "even_positive_quadratic_target", "minimum_dimension_requested",
            "lower_dimensions_excluded", "even_boundary_dimension_excluded",
            "next_dimension_interior_construction",
        ),
    ),
    "subset_xor_card_game_losing_first_move": _contract(
        "subset_xor_card_game_losing_first_move", "exhaustive_move_set",
        "finite_vector_space_xor_strategy", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression"),
        facts=(
            "all_subsets_of_ten_element_set", "empty_subset_included",
            "alternating_complete_draft", "one_card_discard",
            "even_coordinate_parity_target", "hand_xor_ownership_reduction",
            "affine_pairing_strategy", "all_first_move_orbits_exhausted",
        ),
    ),
    "angle_bisector_three_circle_parameter": _contract(
        "angle_bisector_three_circle_parameter", "parameter_set",
        "circle_coefficient_collinearity_determinant", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "acute_scalene_triangle", "circumcenter_and_internal_bisectors",
            "common_positive_ray_ratio", "three_tangent_circle_construction",
            "exactly_two_common_points", "coefficient_vectors_collinear",
            "parameter_polynomial_factored", "both_parameter_roots_verified",
        ),
    ),
    "odd_part_block_congruence_values": _contract(
        "odd_part_block_congruence_values", "integer_set",
        "two_adic_residue_classification", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "odd_part_function", "positive_u_and_existential_positive_v",
            "complete_consecutive_block", "all_differences_divisible_by_four",
            "two_adic_cases_exhausted", "witness_v_for_each_value",
            "all_larger_u_excluded",
        ),
    ),
    "mutual_histogram_weighted_values": _contract(
        "mutual_histogram_weighted_values", "finite_value_set",
        "mutual_histogram_period_enumeration", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "two_monic_polynomial_root_multisets", "integer_exponents_become_multiplicities",
            "mutual_histogram_vectors", "histogram_period_at_most_two",
            "distinct_value_bound_four", "finite_sparse_support_enumerated",
            "weighted_values_recomputed", "all_values_requested",
        ),
    ),
    "gap_two_signed_subsequence_guarantee": _contract(
        "gap_two_signed_subsequence_guarantee", "sharp_integer_bound",
        "four_block_discrepancy_and_extremal_word", True,
        answer_shapes=("number", "expression"),
        facts=(
            "finite_plus_minus_one_sequence", "universal_input_sequence",
            "selected_indices_strictly_increasing", "successive_gap_at_most_two",
            "absolute_selected_sum", "largest_guaranteed_bound_requested",
            "four_block_lower_bound", "periodic_extremal_word",
        ),
    ),
    "two_monotone_merchant_common_connection": _contract(
        "two_monotone_merchant_common_connection", "minimum_integer",
        "two_chain_grid_pigeonhole", True,
        answer_shapes=("number",),
        facts=(
            "square_number_of_linearly_ordered_stalls", "exactly_two_merchants",
            "strictly_monotone_sale_and_purchase_orders", "reachability_connection_definition",
            "common_connected_pair_guarantee", "minimum_item_count_requested",
            "grid_chain_pigeonhole_upper_bound", "matching_extremal_construction",
        ),
    ),
    "missing_color_polyomino_area": _contract(
        "missing_color_polyomino_area", "sharp_area",
        "connected_boundary_growth_and_periodic_coloring", True,
        answer_shapes=("number", "expression"),
        facts=(
            "infinite_square_grid", "side_connected_polyominoes",
            "arbitrary_finite_color_count", "at_most_all_but_one_colors",
            "greatest_universal_area_requested", "connected_growth_lower_bound",
            "periodic_coloring_upper_construction",
        ),
    ),
    "korean_sequence_good_partition_minimum": _contract(
        "korean_sequence_good_partition_minimum", "minimum_length",
        "lcm_gcd_cut_spacing_and_construction", True,
        answer_shapes=("number", "expression"),
        facts=(
            "strictly_increasing_positive_integer_sequence", "prefix_lcm_suffix_gcd_cut",
            "exact_good_partition_count", "minimum_length_requested",
            "three_for_two_cut_spacing_bound", "matching_divisibility_chain_construction",
        ),
    ),
    "cyclic_quartic_equality_triple_count": _contract(
        "cyclic_quartic_equality_triple_count", "count",
        "cyclic_quartic_equality_case_classification", True,
        answer_shapes=("number",),
        facts=(
            "ordered_real_triple_domain", "exact_two_stage_cyclic_equality",
            "triple_count_requested", "zero_coordinate_cases_exhausted",
            "nonzero_equality_cases_exhausted", "all_eight_triples_verified",
        ),
    ),
    "round_robin_unextendable_schedule_minimum": _contract(
        "round_robin_unextendable_schedule_minimum", "minimum_round_count",
        "perfect_matching_blocking_threshold", True,
        answer_shapes=("number",),
        facts=(
            "even_team_set", "each_round_is_perfect_matching",
            "no_pair_repeats", "one_more_round_must_repeat",
            "minimum_schedule_length_requested", "matching_extension_lower_bound",
            "sharp_unextendable_schedule_construction",
        ),
    ),
    "path_domino_maximin_uncovered": _contract(
        "path_domino_maximin_uncovered", "game_value",
        "seven_cell_state_recurrence_and_strategies", True,
        answer_shapes=("number",),
        facts=(
            "one_dimensional_finite_path", "adjacent_two_square_tiles",
            "alternating_legal_placement_game", "first_player_maximizes_uncovered",
            "second_player_minimizes_uncovered", "terminal_uncovered_value_requested",
            "seven_cell_recurrence", "matching_upper_and_lower_strategies",
        ),
    ),
    "odd_checkerboard_l_tromino_minimum": _contract(
        "odd_checkerboard_l_tromino_minimum", "feasibility_and_minimum_count",
        "color_capacity_bound_and_explicit_tromino_cover", True,
        max_goals=2,
        requirements=("feasibility_or_numeric", "numeric_result"),
        required_requirements=("feasibility_or_numeric", "numeric_result"),
        answer_shapes=("number",),
        facts=(
            "odd_square_checkerboard_at_least_seven", "all_four_corners_black",
            "nonoverlapping_l_trominoes", "all_black_squares_must_be_covered",
            "feasibility_and_minimum_requested", "black_cell_capacity_lower_bound",
            "explicit_matching_cover_construction",
        ),
    ),
    "two_by_two_flip_closure_minimum": _contract(
        "two_by_two_flip_closure_minimum", "minimum_initial_count",
        "two_by_two_state_invariant_and_closure_construction", True,
        answer_shapes=("number",),
        facts=(
            "even_square_binary_board", "exactly_three_rule_fills_fourth",
            "exactly_two_rule_flips_block", "arbitrary_initial_configuration",
            "existential_move_sequence_to_full_board", "minimum_universal_count_requested",
            "block_invariant_lower_bound", "sharp_closure_construction",
        ),
    ),
    "ant_collision_escape_time": _contract(
        "ant_collision_escape_time", "latest_escape_time",
        "trajectory_token_exchange_and_extremal_configuration", True,
        requirements=("alternative_result",),
        required_requirements=("alternative_result",),
        answer_shapes=("number",),
        facts=(
            "even_square_checkerboard", "ants_start_at_cell_centers",
            "unit_axis_parallel_speed", "opposite_collision_clockwise_turn",
            "other_collisions_preserve_directions", "absorbing_boundary",
            "stationary_spiders_have_no_interaction_rule", "latest_last_exit_requested",
            "trajectory_token_exchange_upper_bound", "matching_extremal_configuration",
        ),
    ),
    "prefix_split_pebble_survival": _contract(
        "prefix_split_pebble_survival", "minimum_initial_resource",
        "prefix_potential_duality_and_balanced_construction", True,
        answer_shapes=("number",),
        facts=(
            "even_linear_box_count", "arbitrary_initial_pebble_distribution",
            "adversarial_prefix_suffix_split", "chosen_side_increment_other_side_decrement",
            "zero_box_is_immediate_loss", "indefinite_survival_requested",
            "minimum_total_pebbles_requested", "prefix_potential_lower_bound",
            "balanced_survival_construction",
        ),
    ),
    "neighborhood_growth_four_decrements": _contract(
        "neighborhood_growth_four_decrements", "maximum_guaranteed_count",
        "nine_cell_density_game_bound", True,
        answer_shapes=("number",),
        facts=(
            "square_board_divisible_by_three", "initial_zero_heights",
            "closed_king_neighborhood_increment", "four_distinct_positive_decrements",
            "alternating_gardener_first", "fixed_positive_height_threshold",
            "eventual_guaranteed_count_requested", "nine_cell_density_lower_strategy",
            "four_decrement_upper_strategy",
        ),
    ),
    "increasing_grid_path_minimum": _contract(
        "increasing_grid_path_minimum", "minimum_path_count",
        "directed_grid_source_path_recurrence", True,
        answer_shapes=("number",),
        facts=(
            "even_square_permutation_grid", "side_adjacency_only",
            "strictly_increasing_paths", "singleton_paths_included",
            "start_is_even_even_local_minimum", "all_good_paths_counted",
            "minimum_over_fillings_requested", "directed_path_recurrence_lower_bound",
            "sharp_serpentine_filling",
        ),
    ),
    "consecutive_card_partition_game_value": _contract(
        "consecutive_card_partition_game_value", "game_value",
        "paired_card_minimax_strategy", True,
        # The recognized benchmark explicitly requests a proof.  The numeric
        # theorem is trusted evidence, but the model must still write that proof.
        task_kinds=("calculation", "fill_blank", "construction"),
        answer_shapes=("number",),
        facts=(
            "consecutive_card_values", "two_initially_empty_piles",
            "alternating_deliberate_card_and_pile_choice", "first_player_minimizes_difference",
            "second_player_maximizes_difference", "perfect_information_play",
            "absolute_final_pile_difference", "paired_card_upper_strategy",
            "matching_second_player_lower_strategy",
        ),
    ),
    "half_area_boundary_side_minimum": _contract(
        "half_area_boundary_side_minimum", "minimum_side_count",
        "antipodal_area_map_topological_bound_and_construction", True,
        answer_shapes=("number",),
        facts=(
            "convex_polygon", "one_half_area_ray_from_each_vertex",
            "boundary_intersections_are_not_vertices", "distinct_supporting_sides_counted",
            "minimum_over_polygons_requested", "three_side_topological_lower_bound",
            "three_side_realizing_construction",
        ),
    ),
    "three_polar_triangle_locus": _contract(
        "three_polar_triangle_locus", "point_locus",
        "polar_line_coordinate_circumcircle_identity", True,
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "fixed_obtuse_triangle", "orthocenter_centered_vertex_circles",
            "moving_point_outside_circumcircle", "three_polars_form_triangle_when_defined",
            "polar_triangle_circumcircle", "self_incidence_locus_requested",
            "coordinate_circumcircle_identity", "orthocenter_solution_verified",
            "all_other_points_excluded",
        ),
    ),
    "bezout_l1_nice_count_polynomial": _contract(
        "bezout_l1_nice_count_polynomial", "polynomial_expression",
        "bezout_lattice_voronoi_local_maximum_count", True,
        answer_shapes=("number", "expression"),
        facts=(
            "coprime_positive_k_greater_l", "bezout_l1_minimum_function",
            "integer_local_maximum_definition", "odd_parity_count_polynomial",
            "mixed_parity_count_polynomial", "polynomial_square_sum_requested",
            "fundamental_lattice_interval_enumerated", "both_parity_cases_counted",
            "polynomial_identity_simplified",
        ),
    ),
    "reciprocal_means_reach_one_maximum": _contract(
        "reciprocal_means_reach_one_maximum", "maximum_integer",
        "dyadic_numerator_sum_invariant_and_binary_construction", True,
        answer_shapes=("number",),
        facts=(
            "coprime_positive_reciprocal_pair", "arithmetic_and_harmonic_mean_closure",
            "finite_reachability_of_one", "strict_parameter_sum_bound",
            "largest_parameter_sum_requested", "dyadic_sum_necessity",
            "power_of_two_sufficiency_construction", "largest_power_below_bound",
        ),
    ),
    "knight_queen_board_guarantee": _contract(
        "knight_queen_board_guarantee", "maximum_guarantee",
        "checkerboard_color_pairing_strategy", True,
        answer_shapes=("number",),
        facts=(
            "large_board_dimensions_divisible_by_four", "knights_must_be_pairwise_nonattacking",
            "queen_only_occupies_one_empty_square", "horst_moves_before_queenie",
            "first_unable_player_ends_game", "universal_queenie_strategy",
            "maximum_guaranteed_knight_count", "same_color_knight_independence",
            "color_class_pairing_lower_and_upper_strategies",
        ),
    ),
    "angle_ratio_line_point_maximum": _contract(
        "angle_ratio_line_point_maximum", "maximum_count",
        "half_angle_quartic_intersection_bound", True,
        answer_shapes=("number",),
        facts=(
            "line_meets_segment_at_interior_point", "points_restricted_to_given_line",
            "either_angle_is_half_the_other", "maximum_point_count_requested",
            "quartic_intersection_upper_bound", "four_point_configuration_exists",
        ),
    ),
    "all_other_faces_visible_polyhedron_maximum": _contract(
        "all_other_faces_visible_polyhedron_maximum", "maximum_face_count",
        "supporting_plane_visibility_obstruction", True,
        answer_shapes=("number",),
        facts=(
            "convex_polyhedron", "one_exterior_viewpoint_per_face",
            "all_other_faces_visible", "largest_face_count_requested",
            "supporting_plane_obstruction", "tetrahedron_construction",
        ),
    ),
    "rich_integer_set_from_power_differences": _contract(
        "rich_integer_set_from_power_differences", "exhaustive_set_classification",
        "linear_integer_root_closure_descent", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "subset_of_all_integers", "all_integer_polynomial_roots_closed",
            "coefficients_drawn_from_same_set", "all_positive_power_two_differences_contained",
            "all_rich_sets_requested", "linear_root_quotient_closure",
            "integer_generation_descent", "whole_integer_set_verified",
        ),
    ),
    "rational_integer_rounding_function_equation": _contract(
        "rational_integer_rounding_function_equation", "function_solution_set",
        "integer_translation_scaling_and_unit_interval_classification", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "rational_to_integer_function", "universal_rational_x_integer_a_positive_b",
            "exact_nested_rounding_equation", "all_functions_requested",
            "constant_branch_verified", "floor_and_ceiling_branches_verified",
            "unit_interval_cases_exhausted",
        ),
    ),
    "prime_exponential_inequality_parameter_region": _contract(
        "prime_exponential_inequality_parameter_region", "parameter_region",
        "prime_root_limit_and_am_gm_sharpness", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "positive_real_parameter_pair", "universal_prime_and_real_solution",
            "exact_double_exponential_equation", "exact_three_term_power_mean_inequality",
            "all_parameter_pairs_requested", "prime_solution_limit_to_one",
            "log_product_upper_bound", "weighted_am_gm_upper_product_sufficiency",
        ),
    ),
    "cubic_log_derivative_real_root_count": _contract(
        "cubic_log_derivative_real_root_count", "exhaustive_root_count",
        "strict_logarithmic_derivative_sign", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number",),
        facts=(
            "real_cubic_with_three_distinct_roots", "exact_polynomial_derivative_equation",
            "all_possible_distinct_real_root_counts_requested", "root_factorization_log_derivative",
            "strict_negative_sum_of_inverse_squares", "polynomial_roots_checked_separately",
        ),
    ),
    "napkin_equal_coverage_maximum": _contract(
        "napkin_equal_coverage_maximum", "maximum_cell_count",
        "equal_multiplicity_tile_coverage_extremal_bound", True,
        answer_shapes=("number",),
        facts=(
            "fixed_2011_square_board", "finite_multiset_of_52_square_tiles",
            "cell_coverage_multiplicity", "same_nonzero_multiplicity_class",
            "maximum_over_all_tile_configurations", "coverage_layer_upper_bound",
            "matching_periodic_tile_construction", "quotient_remainder_arithmetic_checked",
        ),
    ),
    "personal_consecutive_number_game": _contract(
        "personal_consecutive_number_game", "maximum_draw_parameter",
        "path_independent_set_minimax_classification", True,
        answer_shapes=("number",),
        facts=(
            "positive_integer_path_one_through_n", "alternating_single_choices",
            "global_nonrepetition", "adjacency_forbidden_only_with_own_choices",
            "full_board_is_draw_otherwise_no_move_loses", "alice_moves_first",
            "largest_optimal_draw_parameter_requested", "endpoint_component_reply_strategy",
            "all_small_draw_cases_verified", "all_larger_parameters_excluded",
        ),
    ),
    "round_robin_hotel_cost_minimum": _contract(
        "round_robin_hotel_cost_minimum", "minimum_total_cost",
        "interval_schedule_lower_bound_and_explicit_round_robin_order", True,
        answer_shapes=("number",),
        facts=(
            "exactly_256_players", "every_unordered_pair_plays_once",
            "exactly_one_match_per_day", "inclusive_first_to_last_day_stay",
            "unit_cost_per_present_player_day", "minimum_total_cost_requested",
            "vip_clause_explicitly_cost_neutral", "arrival_departure_order_lower_bound",
            "complete_schedule_construction", "closed_form_arithmetic_checked",
        ),
    ),
    "fibonacci_difference_basis_minimum": _contract(
        "fibonacci_difference_basis_minimum", "minimum_set_cardinality",
        "fibonacci_labeled_forest_bound_and_even_index_construction", True,
        answer_shapes=("number",),
        facts=(
            "standard_fibonacci_initial_values_and_recurrence", "integer_set_difference_targets",
            "all_indices_two_through_upper_bound", "minimum_cardinality_requested",
            "lucas_clause_independent_or_satisfied_by_construction", "cycle_largest_edge_contradiction",
            "even_index_fibonacci_construction", "odd_and_even_target_indices_covered",
        ),
    ),
    "translation_order_odd_count_product": _contract(
        "translation_order_odd_count_product", "extreme_value_product",
        "translation_order_bijection_parity_density_bounds", True,
        answer_shapes=("number",),
        facts=(
            "nonnegative_lattice_to_nonnegative_integer_bijection",
            "strict_order_preserved_by_both_coordinate_translations",
            "odd_images_in_100_square", "smallest_and_largest_count_product_requested",
            "auxiliary_g_function_is_unconstrained_distractor", "quarter_lower_bound",
            "three_quarter_upper_bound", "both_extremes_constructed",
        ),
    ),
    "sparse_green_neighborhood_threshold": _contract(
        "sparse_green_neighborhood_threshold", "least_sparse_integer",
        "square_neighborhood_growth_isoperimetric_threshold", True,
        answer_shapes=("number",),
        facts=(
            "single_initial_green_cell", "seventy_five_square_centered_neighborhood",
            "exactly_s_new_cells_per_turn", "no_cell_recolored",
            "uniform_linear_in_grid_side_sparsity", "least_sparse_integer_requested",
            "convex_corner_threshold_lower_strategy", "boundary_growth_upper_bound",
            "radius_thirty_seven_formula_checked",
        ),
    ),
    "right_triangle_two_cevian_ratio": _contract(
        "right_triangle_two_cevian_ratio", "exact_ratio",
        "sine_split_ratio_identity", True,
        answer_shapes=("number", "expression"),
        facts=(
            "right_triangle_with_hypotenuse_xz", "angle_x_fifty_degrees",
            "p_and_q_on_yz", "two_ten_degree_cevians",
            "twice_yq_over_zp_requested", "sine_form_segment_ratio",
            "trigonometric_identity_simplified",
        ),
    ),
    "equal_diagonal_quadrilateral_maximum_area": _contract(
        "equal_diagonal_quadrilateral_maximum_area", "maximum_area",
        "diagonal_area_bound_and_perimeter_attainment", True,
        answer_shapes=("number", "expression"),
        facts=(
            "convex_quadrilateral", "perimeter_three", "both_diagonals_unit_length",
            "maximum_area_requested", "half_diagonal_product_upper_bound",
            "perpendicular_diagonal_configuration_with_required_perimeter",
        ),
    ),
    "power_difference_semigroup_smallest_gap": _contract(
        "power_difference_semigroup_smallest_gap", "smallest_missing_positive_integer",
        "minimum_generator_lower_bound", True,
        answer_shapes=("number",),
        facts=(
            "integer_parameter_at_least_two", "exact_power_difference_generator_set",
            "unlimited_generator_repetition", "literal_smallest_positive_nonrepresentable_requested",
            "all_generators_at_least_two", "one_is_not_representable",
        ),
    ),
    "recurrence_universal_coprime_set": _contract(
        "recurrence_universal_coprime_set", "exhaustive_positive_integer_set",
        "closed_form_and_fermat_prime_witnesses", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression", "roots"),
        facts=(
            "integer_sequence_with_exact_initial_value", "exact_nonhomogeneous_recurrence",
            "coprime_to_every_sequence_term", "all_positive_integers_requested",
            "closed_form_verified", "prime_two_and_three_witnesses",
            "fermat_witness_for_every_larger_prime", "only_one_survives",
        ),
    ),
    "quartic_plus_five_splitting_field": _contract(
        "quartic_plus_five_splitting_field", "field_degree_and_galois_verdict",
        "eisenstein_tower_and_splitting_field_normality", True,
        max_goals=3,
        requirements=("field_value", "degree_value", "judgement", "galois_verdict"),
        required_requirements=("field_value", "degree_value", "judgement", "galois_verdict"),
        answer_shapes=("truth",),
        facts=(
            "quartic_x_four_plus_five_over_rationals", "splitting_field_requested",
            "extension_degree_requested", "galois_verdict_requested",
            "one_fourth_root_of_negative_five_and_i_generate_all_roots",
            "irreducible_quartic_then_quadratic_tower", "splitting_field_is_galois",
        ),
    ),
    "flood_barrier_critical_speed": _contract(
        "flood_barrier_critical_speed", "critical_speed",
        "flood_boundary_lower_bound_and_two_front_barrier_strategy", True,
        answer_shapes=("number",),
        facts=(
            "infinite_square_grid", "finite_initial_flood", "connected_noncrossing_barrier",
            "finite_extra_walls_declared_patternless", "cumulative_gamma_n_wall_budget",
            "four_neighbor_flood_after_builder_turn", "closed_loop_containment_win",
            "known_template_critical_boundary_interpretation", "boundary_length_speed_lower_bound",
            "two_parallel_fronts_upper_strategy",
        ),
    ),
    "recursive_digit_deletion_maximum": _contract(
        "recursive_digit_deletion_maximum", "maximum_integer",
        "complete_recursive_digit_enumeration", True,
        answer_shapes=("number",),
        facts=(
            "positive_integer_decimal_representation", "pairwise_distinct_digits",
            "single_digit_base_cases", "one_digit_canonical_deletion",
            "deleted_number_divides_original", "recursive_goodness",
            "maximum_requested", "complete_finite_state_enumeration",
            "reverse_insertion_transition_complete", "maximum_witness_chain_verified",
            "all_ten_decimal_digits_exhausted", "empty_seven_digit_layer",
            "maximality_by_empty_longer_layers",
        ),
    ),
    "adjacent_surjection_count": _contract(
        "adjacent_surjection_count", "count", "inclusion_exclusion_and_path_coloring", True,
        answer_shapes=("number",),
    ),
    "multiset_no_adjacent_count": _contract(
        "multiset_no_adjacent_count", "count", "multiset_gap_bijection", True,
        answer_shapes=("number",),
    ),
    "binary_run_avoidance_count": _contract(
        "binary_run_avoidance_count", "count", "finite_state_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "bracelet_no_adjacent_count": _contract(
        "bracelet_no_adjacent_count", "count", "dihedral_orbit_enumeration", True,
        answer_shapes=("number",),
    ),
    "strip_lattice_path_count": _contract(
        "strip_lattice_path_count", "count", "boundary_state_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "nested_modular_power_sum": _contract(
        "nested_modular_power_sum", "residue", "exact_modular_exponentiation", True,
        answer_shapes=("number",),
    ),
    "quadratic_form_maximum": _contract(
        "quadratic_form_maximum", "scalar", "symmetric_eigenvalue_certificate", True,
        answer_shapes=("number", "expression"),
    ),
    "tree_degree_census": _contract(
        "tree_degree_census", "count", "tree_handshake_and_vertex_census", True,
        answer_shapes=("number",),
        facts=(
            "finite_tree", "exact_leaf_count", "all_nonleaves_have_two_allowed_degrees",
            "requested_degree_count", "vertex_degree_census", "tree_handshake_identity",
            "nonnegative_integer_degree_counts",
        ),
    ),
    "involution_fixed_point_count": _contract(
        "involution_fixed_point_count", "count_with_pairing_formula",
        "fixed_points_and_perfect_matching_bijection", True,
        requirements=("pairing_step",),
        required_requirements=("pairing_step",),
        answer_shapes=("number", "expression"),
        facts=(
            "permutation_of_n_elements", "involution_equation_sigma_squared_identity",
            "exact_fixed_point_count", "remaining_elements_paired", "fixed_points_chosen",
            "perfect_matching_count", "parity_checked",
        ),
    ),
    "composite_trapezoid": _contract(
        "composite_trapezoid", "approximation_exact_value_and_error",
        "exact_rational_composite_trapezoid", True,
        requirements=("integral_result", "integral_value", "exact_comparison"),
        required_requirements=("integral_result", "exact_comparison"),
        answer_shapes=("number", "expression"),
        facts=(
            "composite_trapezoidal_rule", "finite_closed_interval", "equal_subinterval_count",
            "explicit_monomial_integrand", "matching_quadrature_and_integral_bounds",
            "endpoint_and_interior_weight_evaluation", "exact_integral_evaluation",
            "signed_error_comparison",
        ),
    ),
    "two_point_gauss_legendre_monomial": _contract(
        "two_point_gauss_legendre_monomial", "exact_approximation",
        "exact_two_node_quadrature", True,
        answer_shapes=("number", "expression"),
        facts=(
            "two_point_gauss_legendre_rule", "finite_closed_interval",
            "explicit_monomial_integrand", "matching_quadrature_and_integral_bounds",
            "affine_nodes_and_weights_recomputed", "exact_fraction_simplified",
        ),
    ),
    "exponential_l1_sequence": _contract(
        "exponential_l1_sequence", "l1_convergence_judgement",
        "exact_pointwise_and_l1_norm_check", True,
        requirements=("judgement", "pointwise_limit", "l1_norm_check"),
        required_requirements=("judgement", "pointwise_limit", "l1_norm_check"),
        task_kinds=("calculation", "fill_blank"),
        answer_shapes=("truth", "expression"),
        facts=(
            "positive_half_line_domain", "exact_exponential_sequence",
            "pointwise_limit_recomputed", "l1_norm_substitution_recomputed",
            "l1_convergence_criterion_applied",
        ),
    ),
    "cycle_distance_two_coloring": _contract(
        "cycle_distance_two_coloring", "count", "cyclic_boundary_state_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "directed_cylinder_hamilton_paths": _contract(
        "directed_cylinder_hamilton_paths", "count", "connectivity_state_recurrence", True,
        answer_shapes=("number",),
        facts=(
            "three_rows", "directed_cyclic_horizontal", "vertical_undirected",
            "all_vertices_permutation", "fixed_endpoint_rows", "index_count_matches_grid",
        ),
    ),
    "sorted_triangle_failure_bound": _contract(
        "sorted_triangle_failure_bound", "minimum_integer", "sharp_rearrangement_bound", True,
        answer_shapes=("number",),
        facts=(
            "nondegenerate_input_triangles", "three_distinct_side_colors",
            "all_three_color_sequences_descending", "aligned_failure_count",
            "universal_bound_requested", "sharp_minimum_requested",
        ),
    ),
    "sparkling_tuple_pair_sum": _contract(
        "sparkling_tuple_pair_sum", "parameter_expression",
        "permutation_average_and_sharp_limit", True,
        requirements=("parameter_dependency_m",),
        required_requirements=("parameter_dependency_m",),
        answer_shapes=("expression",),
        facts=(
            "real_m_tuple", "all_permutations_quantifier", "adjacent_product_lower_bound",
            "complete_pair_sum_target", "parameterized_largest_constant",
        ),
    ),
    "five_number_ratio_gap": _contract(
        "five_number_ratio_gap", "minimum_constant", "ordered_ratio_pigeonhole_sharpness", True,
        answer_shapes=("number", "expression"),
        facts=(
            "five_distinct_positive_reals", "four_distinct_selection",
            "normalized_product_gap", "universal_existential_quantifiers", "sharp_minimum_requested",
        ),
    ),
    "nested_nonnegative_sequence_values": _contract(
        "nested_nonnegative_sequence_values", "exhaustive_value_set",
        "translation_defect_and_residue_classification", True,
        requirements=("exhaustive_result",),
        required_requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
        facts=(
            "sequence_on_nonnegative_integers", "triple_self_composition",
            "successor_plus_one_rhs", "all_nonnegative_indices",
            "no_additional_sequence_constraints", "all_values_requested",
        ),
    ),
    "mysterious_cuberoot_polynomial": _contract(
        "mysterious_cuberoot_polynomial", "polynomial",
        "cubic_field_basis_identity_and_minimality", True,
        requirements=("exhaustive_result",),
        required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression"),
        facts=(
            "real_pure_cubic_field", "noncube_integer_radicand",
            "same_radicand_in_definition_and_target", "nonzero_integer_reciprocal_scale",
            "rational_polynomial_coefficients", "all_lowest_degree_polynomials_requested",
            "target_is_alpha_plus_alpha_squared", "exact_polynomial_identity",
            "degree_two_minimality", "unique_reduced_representative",
        ),
    ),
    "complete_bipartite_homomorphism_bound": _contract(
        "complete_bipartite_homomorphism_bound", "symbolic_lower_bound",
        "holder_jensen_kst_homomorphism_bound", True,
        requirements=("intersection",),
        answer_shapes=("number", "expression"),
        facts=(
            "positive_n_s_t", "edge_density_lambda", "vertices_may_repeat",
            "all_st_cross_edges", "minimum_good_tuple_count",
        ),
    ),
    "tangential_identical_triangulation_polygon": _contract(
        "tangential_identical_triangulation_polygon", "solution_set",
        "tangential_triangulation_rigidity", True,
        requirements=("numeric_result",),
        required_requirements=("numeric_result",),
        answer_shapes=("number",),
        facts=(
            "convex_m_polygon", "m_greater_than_three", "identical_triangle_triangulation",
            "noncrossing_diagonals", "circumscribed_polygon_requested",
        ),
    ),
    "formal_l2_adjoint": _contract(
        "formal_l2_adjoint", "operator_expression", "compact_support_integration_by_parts", True,
        answer_shapes=("number", "expression"),
        facts=(
            "open_domain", "compactly_supported_smooth_domain", "real_smooth_coefficients",
            "divergence_second_order_term", "first_order_drift_term",
            "unweighted_l2_adjoint_requested", "no_boundary_term",
        ),
    ),
    "mixed_radix_grid_compression": _contract(
        "mixed_radix_grid_compression", "parameter_expression",
        "mixed_radix_weight_and_carry_induction", True,
        answer_shapes=("number", "expression"),
        facts=(
            "finite_three_dimensional_integer_grid", "positive_symbolic_bounds",
            "arbitrary_initial_distribution", "coordinate_decreasing_carry_operations",
            "origin_piece_target", "universal_sharp_threshold_requested",
        ),
    ),
    "square_dihedral_facts": _contract(
        "square_dihedral_facts", "choice_set", "closed_world_group_fact_table", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("square_d8_order_convention", "all_options_recognized", "positive_question_polarity"),
    ),
    "lebesgue_integrability_facts": _contract(
        "lebesgue_integrability_facts", "choice_set", "closed_world_integrability_fact_table", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("finite_closed_interval", "lebesgue_integrability", "all_options_recognized"),
    ),
    "compact_real_facts": _contract(
        "compact_real_facts", "choice_set", "heine_borel_and_open_cover_definition", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("real_line_usual_topology", "all_options_recognized", "positive_question_polarity"),
    ),
    "cauchy_complete_space_facts": _contract(
        "cauchy_complete_space_facts", "choice_set", "complete_metric_space_cauchy_theorems", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("complete_metric_space", "cauchy_definition", "all_options_recognized"),
    ),
    "linear_programming_duality_facts": _contract(
        "linear_programming_duality_facts", "choice_set", "standard_lp_duality_fact_table", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("standard_linear_programming_duality", "all_options_recognized", "positive_question_polarity"),
    ),
    "matrix_condition_number_definition": _contract(
        "matrix_condition_number_definition", "choice", "matrix_one_norm_condition_definition", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("matrix_condition_number_question", "one_norm_convention", "all_options_recognized"),
    ),
    "dirichlet_pde_discretization_methods": _contract(
        "dirichlet_pde_discretization_methods", "method_set", "elliptic_pde_discretization_families", True,
        task_kinds=("calculation", "fill_blank"), answer_shapes=("expression",),
        facts=("poisson_equation", "dirichlet_boundary_condition", "discretization_method_requested"),
    ),
    "time_series_components": _contract(
        "time_series_components", "choice_set", "classical_time_series_decomposition", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("time_series_components_question", "all_options_recognized", "positive_question_polarity"),
    ),
    "seasonal_adjustment_methods": _contract(
        "seasonal_adjustment_methods", "method_pair", "classical_seasonal_adjustment_methods", True,
        max_goals=2,
        requirements=("two_items",), required_requirements=("two_items",),
        task_kinds=("calculation", "fill_blank"), answer_shapes=("number", "expression"),
        facts=("seasonal_adjustment_question", "two_method_blanks", "exact_statement_recognized"),
    ),
    "dispersion_measure_standard_deviation": _contract(
        "dispersion_measure_standard_deviation", "concept", "descriptive_statistics_definition", True,
        task_kinds=("calculation", "fill_blank"), answer_shapes=("number", "expression"),
        facts=("descriptive_statistics_context", "dispersion_measure_requested", "exact_statement_recognized"),
    ),
    "unknown_form_regression": _contract(
        "unknown_form_regression", "choice", "regression_model_classification", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("unknown_functional_form", "listed_methods_exclude_nonparametric", "all_options_recognized"),
    ),
    "stepwise_removal": _contract(
        "stepwise_removal", "choice", "stepwise_retesting_criteria", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("stepwise_new_variable_retest", "all_of_above_option", "all_options_recognized"),
    ),
    "nonlinear_regression_estimation": _contract(
        "nonlinear_regression_estimation", "choice", "nonlinear_least_squares_definition", True,
        requirements=("all_correct_choices",),
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=("generic_nonlinear_regression", "estimation_criterion_not_optimizer", "all_options_recognized"),
    ),
    "aggregate_series_ratio_truth": _contract(
        "aggregate_series_ratio_truth", "truth", "statistical_index_definition", True,
        task_kinds=("calculation", "choice"), answer_shapes=("truth",),
        facts=("two_aggregate_index_series", "ratio_series_requested", "exact_statement_recognized"),
    ),
    "heteroscedastic_ols_variance_truth": _contract(
        "heteroscedastic_ols_variance_truth", "truth", "heteroscedastic_variance_direction_check", True,
        task_kinds=("calculation", "choice"), answer_shapes=("truth",),
        facts=("ols_under_heteroscedasticity", "unqualified_variance_increase_claim", "exact_statement_recognized"),
    ),
    "heteroscedastic_parameter_variance_consequence": _contract(
        "heteroscedastic_parameter_variance_consequence", "statement",
        "heteroscedastic_ols_inference_consequences", True,
        task_kinds=("calculation", "fill_blank"), answer_shapes=("number", "expression"),
        facts=(
            "heteroscedasticity_context", "parameter_estimator_variance_consequence_requested",
            "exact_statement_recognized",
        ),
    ),
    "normal_distribution_parameters": _contract(
        "normal_distribution_parameters", "choice", "closed_option_semantics", True,
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=(
            "normal_distribution_parameter_question", "all_options_recognized",
            "mean_standard_deviation_convention",
        ),
    ),
    "large_dataset_overview_plot": _contract(
        "large_dataset_overview_plot", "choice", "closed_option_semantics", True,
        task_kinds=("choice",), answer_shapes=("choice",),
        facts=(
            "large_dataset_basic_feature_overview", "all_options_recognized",
            "histogram_distribution_overview",
        ),
    ),
    "triangular_lattice_regular_hexagons": _contract(
        "triangular_lattice_regular_hexagons", "count",
        "axial_coordinate_hexagon_enumeration", True,
        answer_shapes=("number",),
        facts=(
            "regular_hexagonal_triangular_lattice", "unit_triangle_subdivision",
            "all_lattice_vertex_regular_hexagons", "orientation_size_position_sum",
            "small_side_enumeration_crosscheck",
        ),
    ),
    "critical_line_cover_point_set": _contract(
        "critical_line_cover_point_set", "maximum",
        "critical_cover_upper_bound_and_general_position_construction", True,
        answer_shapes=("number",),
        facts=(
            "finite_planar_point_set", "not_coverable_by_m_lines",
            "every_single_deletion_coverable_by_m_lines", "maximum_requested",
            "critical_cover_upper_bound", "general_position_line_intersection_construction",
        ),
    ),
    "even_quadratic_pair_count_parameters": _contract(
        "even_quadratic_pair_count_parameters", "solution_set",
        "binary_quadratic_discriminant_parity_classification", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
        facts=(
            "even_integer_parameter", "ordered_integer_pair_domain",
            "exact_quadratic_equation", "finite_solution_parity_requested",
            "zero_parameter_excluded", "residue_classes_exhausted",
        ),
    ),
    "sparse_domino_placements": _contract(
        "sparse_domino_placements", "count_formula",
        "two_monotone_path_bijection", True,
        answer_shapes=("number", "expression"),
        facts=(
            "two_by_one_dominoes", "two_k_square_board", "exactly_k_squared_dominoes",
            "nonoverlapping_placement", "every_two_square_has_aligned_uncovered_pair",
            "two_monotone_path_bijection", "independent_path_binomial_counts",
        ),
    ),
    "red_blue_line_separation": _contract(
        "red_blue_line_separation", "minimum_integer",
        "alternating_configuration_lower_bound_and_hull_induction", True,
        answer_shapes=("number",),
        facts=(
            "two_color_planar_points", "color_counts_differ_by_one", "no_three_collinear",
            "separating_lines_avoid_points", "monochromatic_regions",
            "all_configurations_required", "alternating_configuration_lower_bound",
            "convex_hull_induction_upper_bound",
        ),
    ),
    "clustered_interval_maximum": _contract(
        "clustered_interval_maximum", "maximum_formula",
        "modulo_thirty_block_bound_and_multiple_construction", True,
        answer_shapes=("number", "expression"),
        facts=(
            "positive_integer_parameter_at_least_three", "finite_positive_integer_set",
            "every_triple_has_nontrivial_gcd_pair", "diameter_at_most_parameter",
            "maximum_cardinality_requested", "modulo_thirty_block_upper_bound",
            "multiples_of_two_or_three_construction",
        ),
    ),
    "quadratic_transform_invariant_polynomials": _contract(
        "quadratic_transform_invariant_polynomials", "polynomial_family",
        "finite_reflection_group_invariant_ring", True,
        requirements=("exhaustive_result",), required_requirements=("exhaustive_result",),
        answer_shapes=("number", "expression"),
        facts=(
            "complex_bivariate_polynomial", "universal_complex_parameters",
            "exact_quadratic_transform_identity", "all_polynomials_requested",
            "invariant_generators_algebraically_independent",
        ),
    ),
    "punctured_domino_tilings": _contract(
        "punctured_domino_tilings", "count", "obstacle_profile_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "unique_domino_partition_marking": _contract(
        "unique_domino_partition_marking", "minimum_integer",
        "alternating_cycle_bound_and_diagonal_construction", True,
        answer_shapes=("number",),
        facts=(
            "even_square_board", "complete_domino_partition",
            "horizontal_vertical_unit_dominoes", "marked_pair_forbidden_per_domino",
            "existence_and_uniqueness_requested", "minimum_positive_mark_count",
            "alternating_cycle_lower_bound", "diagonal_marking_construction",
            "small_board_exhaustive_crosscheck",
        ),
    ),
    "complete_intersection_maximum": _contract(
        "complete_intersection_maximum", "maximum", "complete_intersection_theorem", True,
        answer_shapes=("number",),
    ),
    "bounded_generalized_pell_count": _contract(
        "bounded_generalized_pell_count", "count", "bounded_exact_square_enumeration", True,
        answer_shapes=("number",),
    ),
    "integer_polynomial_divisibility": _contract(
        "integer_polynomial_divisibility", "solution_set", "monic_remainder_growth_bound", True,
        requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
    ),
    "reciprocal_quartic_nonnegative": _contract(
        "reciprocal_quartic_nonnegative", "parameter_interval", "reciprocal_substitution_domain_minimum", True,
        requirements=("exhaustive_result",),
        answer_shapes=("number", "interval", "expression"),
    ),
    "affine_recurrence_determinant": _contract(
        "affine_recurrence_determinant", "scalar", "shifted_casoratian_invariant", True,
        answer_shapes=("number", "expression"),
    ),
    "root_polynomial_product": _contract(
        "root_polynomial_product", "scalar", "monic_polynomial_resultant", True,
        answer_shapes=("number", "expression"),
    ),
    "cevian_length": _contract(
        "cevian_length", "scalar", "stewart_identity_exact_arithmetic", True,
        answer_shapes=("number", "expression"),
    ),
    "smith_normal_form": _contract(
        "smith_normal_form", "matrix", "determinantal_divisor_chain", True,
        answer_shapes=("matrix", "expression"),
    ),
    "intersecting_antichain_maximum": _contract(
        "intersecting_antichain_maximum", "maximum", "milner_intersecting_antichain_theorem", True,
        answer_shapes=("number",),
    ),
    "bipartite_matching_deletion_trees": _contract(
        "bipartite_matching_deletion_trees", "count", "matrix_tree_exact_cofactor", True,
        answer_shapes=("number",),
    ),
    "complete_graph_cycle_deletion_trees": _contract(
        "complete_graph_cycle_deletion_trees", "count", "matrix_tree_exact_cofactor", True,
        answer_shapes=("number",),
    ),
    "cyclic_nonadjacent_selection": _contract(
        "cyclic_nonadjacent_selection", "count", "cycle_gap_bijection", True,
        answer_shapes=("number",),
    ),
    "wythoff_losing_position_count": _contract(
        "wythoff_losing_position_count", "count", "wythoff_beatty_pair_enumeration", True,
        answer_shapes=("number",),
    ),
    "finite_subtraction_game": _contract(
        "finite_subtraction_game", "count", "bounded_game_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "equal_marble_box_minimum": _contract(
        "equal_marble_box_minimum", "minimum_integer",
        "odd_divisor_invariant_and_binary_construction", True,
        answer_shapes=("number", "expression"),
        facts=(
            "positive_number_of_initial_boxes", "exactly_one_marble_per_initial_box",
            "two_distinct_nonempty_boxes_selected", "equal_positive_removal_from_each_box",
            "remainders_stay_in_original_boxes", "new_box_contains_combined_removed_marbles",
            "finite_sequence_minimum_nonempty_boxes", "total_marble_count_invariant",
            "odd_common_divisor_reverse_invariant", "power_of_two_single_box_criterion",
            "two_box_construction_for_non_power_of_two",
        ),
    ),
    "square_subtraction_game": _contract(
        "square_subtraction_game", "count", "bounded_game_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "wheel_coloring": _contract(
        "wheel_coloring", "count", "chromatic_polynomial_evaluation", True,
        answer_shapes=("number",),
    ),
    "grid_poset_extensions": _contract(
        "grid_poset_extensions", "count", "hook_length_formula", True,
        answer_shapes=("number",),
    ),
    "hypercube_spanning_trees": _contract(
        "hypercube_spanning_trees", "count", "laplacian_spectrum_product", True,
        answer_shapes=("number",),
    ),
    "odd_fiber_functions": _contract(
        "odd_fiber_functions", "count", "labeled_fiber_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "couples_unlabeled_groups": _contract(
        "couples_unlabeled_groups", "count", "capacity_state_enumeration", True,
        answer_shapes=("number",),
    ),
    "bounded_divisor_count": _contract(
        "bounded_divisor_count", "count", "bounded_prime_factorization", True,
        answer_shapes=("number",),
    ),
    "primitive_pythagorean_count": _contract(
        "primitive_pythagorean_count", "count", "euclid_parameter_enumeration", True,
        answer_shapes=("number",),
    ),
    "inverse_totient": _contract(
        "inverse_totient", "solution_set", "bounded_totient_exhaustion", True,
        requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
    ),
    "gcd_sum": _contract(
        "gcd_sum", "scalar", "divisor_totient_identity", True,
        answer_shapes=("number", "expression"),
    ),
    "positive_sum_two_squares": _contract(
        "positive_sum_two_squares", "count", "bounded_square_enumeration", True,
        answer_shapes=("number",),
    ),
    "factorial_quotient_valuation": _contract(
        "factorial_quotient_valuation", "scalar", "legendre_valuation", True,
        answer_shapes=("number",),
    ),
    "pell_fundamental_solution": _contract(
        "pell_fundamental_solution", "ordered_pair", "continued_fraction_certificate", True,
        answer_shapes=("number", "roots", "expression"),
    ),
    "least_integer_with_divisor_count": _contract(
        "least_integer_with_divisor_count", "scalar", "ordered_exponent_search", True,
        answer_shapes=("number",),
    ),
    "factorable_binary_quadratic": _contract(
        "factorable_binary_quadratic", "parametric_solution_set", "integer_factor_bijection", True,
        requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
    ),
    "cube_root_positive_integer_pairs": _contract(
        "cube_root_positive_integer_pairs", "parametric_solution_set",
        "discriminant_square_descent_and_symbolic_identity", True,
        requirements=("exhaustive_result",),
        required_requirements=("exhaustive_result",),
        answer_shapes=("number", "roots", "expression"),
        facts=(
            "positive_integer_pair_domain", "exact_cubic_root_equation",
            "quadratic_in_second_variable", "discriminant_square_condition",
            "square_factorization_descent", "positive_branch_filter",
            "symbolic_substitution_identity", "parameter_domain_exhausted",
        ),
    ),
    "descartes_inner_circle": _contract(
        "descartes_inner_circle", "scalar", "descartes_curvature_identity", True,
        answer_shapes=("number", "expression"),
    ),
    "rotation_necklace_fixed_weight": _contract(
        "rotation_necklace_fixed_weight", "count", "cyclic_orbit_enumeration", True,
        answer_shapes=("number",),
    ),
    "fixed_weight_binary_bracelets": _contract(
        "fixed_weight_binary_bracelets", "count", "dihedral_orbit_enumeration", True,
        answer_shapes=("number",),
    ),
    "specified_degree_labeled_trees": _contract(
        "specified_degree_labeled_trees", "count", "prufer_word_multiplicity", True,
        answer_shapes=("number",),
    ),
    "odd_cycle_permutations": _contract(
        "odd_cycle_permutations", "count", "restricted_cycle_recurrence", True,
        answer_shapes=("number",),
    ),
    "power_fixed_residue_count": _contract(
        "power_fixed_residue_count", "count", "complete_modular_enumeration", True,
        answer_shapes=("number",),
    ),
    "reciprocal_pair_sum": _contract(
        "reciprocal_pair_sum", "scalar", "divisor_pair_bijection", True,
        answer_shapes=("number",),
    ),
    "integer_grid_nondegenerate_triangles": _contract(
        "integer_grid_nondegenerate_triangles", "count", "complete_determinant_enumeration", True,
        answer_shapes=("number",),
    ),
    "bose_einstein_integral": _contract(
        "bose_einstein_integral", "scalar", "gamma_zeta_even_value", True,
        requirements=("integral_value",),
        answer_shapes=("number", "expression"),
    ),
    "bernoulli_likelihood_ratio": _contract(
        "bernoulli_likelihood_ratio", "expression", "likelihood_substitution", True,
        answer_shapes=("number", "expression"),
    ),
    "brownian_exit_expectation": _contract(
        "brownian_exit_expectation", "scalar", "dirichlet_exit_time_solution", True,
        answer_shapes=("number", "expression"),
    ),
    # Useful exact local evidence, but not a complete answer contract by itself.
    "congruence_solution": _contract("congruence_solution", "residue_set", "exact_modular_search"),
    "modular_power": _contract("modular_power", "residue", "exact_modular_exponentiation"),
}


LEGACY_LABEL_TO_OPERATION: dict[str, str] = {
    "SymPy 计算": "calculate",
    "SymPy 方程解": "solve_equation",
    "SymPy 导数": "derivative",
    "SymPy 偏导数": "derivative",
    "SymPy 定积分": "definite_integral",
    "SymPy 不定积分": "integral",
    "SymPy 极限": "limit",
    "SymPy 矩阵": "matrix",
    "SymPy 递推通项": "recurrence_solution",
    "SymPy 曲线速度": "curve_speed",
    "SymPy 第一基本形式": "first_fundamental_form",
    "本地高斯曲率公式": "graph_gaussian_curvature",
    "SymPy PDE核验": "pde_verification",
    "本地命题逻辑推导": "propositional_implication_chain",
    "本地图论路径证明": "minimum_degree_path_proof",
    "本地偶基数子集计数": "even_subset_count",
    "本地删边完全二部图三步路计数": "deleted_edge_bipartite_length_three_paths",
    "本地正整数下界隔板计数": "positive_composition_lower_bounds",
    "本地二项式系数正整数解": "binomial_choose_two_positive_root",
    "本地二项式系数正整数无解": "binomial_choose_two_positive_root",
    "本地有限循环群子群计数": "finite_cyclic_subgroup_count",
    "本地线性区间不相邻选择": "linear_nonadjacent_selection",
    "本地不相邻二进制串计数": "nonadjacent_binary_string_count",
    "本地排列条件计数": "precedence_permutation_count",
    "本地满射容斥计数": "surjection_count",
    "本地平面图欧拉答案": "planar_euler_faces",
    "本地抛物面曲率答案": "paraboloid_curvature",
    "本地有序三元组计数": "ordered_positive_triples",
    "本地公平六面骰条件概率": "fair_dice_conditional_probability",
    "本地Bernoulli中心二阶矩": "bernoulli_centered_second_moment",
    "本地公平硬币几何尾概率": "fair_coin_geometric_tail",
    "本地泊松过程独立增量": "poisson_process_increment",
    "本地Cauchy位置族Fisher信息": "cauchy_location_fisher_information",
    "本地一维Wald统计量": "one_dimensional_wald_statistic",
    "本地对角协方差GLS估计": "diagonal_gls_estimate",
    "本地正态总体方差置信区间": "normal_variance_confidence_interval",
    "本地二状态Markov熵率": "two_state_markov_entropy_rate",
    "本地独立事件并概率": "independent_event_union",
    "本地独立标准正态和": "independent_standard_normal_sum",
    "本地布朗运动协方差": "brownian_covariance",
    "本地样本均值方差": "sample_mean_variance",
    "本地更新过程强大数律": "renewal_rate_limit",
    "本地有限离散分布矩": "finite_discrete_moments",
    "本地双侧Z检验拒绝域": "two_sided_z_rejection",
    "本地随机游走矩": "simple_random_walk_moments",
    "本地完全图覆盖时间": "complete_graph_cover_time",
    "本地二项分布容量": "two_venue_capacity",
    "本地圆周拉普拉斯": "circle_laplacian",
    "本地圆周Laplace-Beltrami": "circle_laplace_beltrami",
    "本地圆周拉普拉斯歧义核验": "circle_laplacian_ambiguous",
    "本地中心差分": "central_difference",
    "本地有理数约束传播答案": "rational_f2_constraint",
    "本地数位和窗口答案": "digit_sum_window",
    "本地取数博弈答案": "number_writing_game",
    "本地路径配分函数答案": "path_independent_set_partition",
    "本地LZ78编码答案": "lz78_encoding",
    "本地尖峰函数构造答案": "spike_sequence_construction",
    "本地Bernoulli依赖构造答案": "dependent_bernoulli_construction",
    "本地完全多部图生成树": "complete_multipartite_spanning_trees",
    "本地二次同余计数": "quadratic_congruence_count",
    "本地有限自指数整除解集": "bounded_self_exponential_divisibility",
    "本地竞争硬币模式概率": "competing_coin_patterns",
    "本地数字排列整除计数": "digit_permutation_divisibility",
    "本地受限数字整除计数": "bounded_digit_set_divisibility_count",
    "本地素数整商不等式排名": "prime_floor_inequality_rank",
    "本地实函数方程三分支全解": "real_functional_equation_three_solutions",
    "本地正整数嵌套函数值域": "nice_positive_integer_function_value_set",
    "本地开区间二次约束最小维数": "open_interval_quadratic_minimum_dimension",
    "本地全集子集异或博弈首步": "subset_xor_card_game_losing_first_move",
    "本地角平分线三圆共点参数": "angle_bisector_three_circle_parameter",
    "本地奇数部分连续块同余值集": "odd_part_block_congruence_values",
    "本地互为频数向量加权值集": "mutual_histogram_weighted_values",
    "本地间隔二符号子序列锐界": "gap_two_signed_subsequence_guarantee",
    "本地双单调交易链公共连接阈值": "two_monotone_merchant_common_connection",
    "本地缺一色多连方格面积锐界": "missing_color_polyomino_area",
    "本地Korean序列好分割最小长度": "korean_sequence_good_partition_minimum",
    "本地循环四次等号三元组计数": "cyclic_quartic_equality_triple_count",
    "本地不可扩展完美匹配赛程轮数": "round_robin_unextendable_schedule_minimum",
    "本地路径多米诺极大极小游戏值": "path_domino_maximin_uncovered",
    "本地奇阶棋盘黑格L三连方覆盖": "odd_checkerboard_l_tromino_minimum",
    "本地二乘二翻转闭包最小初始数": "two_by_two_flip_closure_minimum",
    "本地顺时针转向蚂蚁最迟离场时刻": "ant_collision_escape_time",
    "本地前缀切分取石生存阈值": "prefix_split_pebble_survival",
    "本地九宫增高四点降低博弈值": "neighborhood_growth_four_decrements",
    "本地偶偶极小点递增格路最少数": "increasing_grid_path_minimum",
    "本地连续卡牌两堆极大极小游戏值": "consecutive_card_partition_game_value",
    "本地半面积边界点最少承载边数": "half_area_boundary_side_minimum",
    "本地三极线三角形外接圆轨迹": "three_polar_triangle_locus",
    "本地Bezout-L1局部极大计数平方和": "bezout_l1_nice_count_polynomial",
    "本地倒数均值闭包最大参数和": "reciprocal_means_reach_one_maximum",
    "本地骑士皇后占格保证值": "knight_queen_board_guarantee",
    "本地线交线段半角点数上界": "angle_ratio_line_point_maximum",
    "本地逐面外点可见凸多面体面数上界": "all_other_faces_visible_polyhedron_maximum",
    "本地幂差生成富整数集": "rich_integer_set_from_power_differences",
    "本地有理数整数值舍入函数全解": "rational_integer_rounding_function_equation",
    "本地素数指数方程不等式参数域": "prime_exponential_inequality_parameter_region",
    "本地三实根三次式对数导数根数": "cubic_log_derivative_real_root_count",
    "本地方巾等覆盖格数锐界": "napkin_equal_coverage_maximum",
    "本地个人相邻禁选博弈最大和局参数": "personal_consecutive_number_game",
    "本地完全赛程住宿总成本最小值": "round_robin_hotel_cost_minimum",
    "本地Fibonacci差集基最小规模": "fibonacci_difference_basis_minimum",
    "本地平移保序双射奇值数极值乘积": "translation_order_odd_count_product",
    "本地绿色邻域稀疏临界值": "sparse_green_neighborhood_threshold",
    "本地直角三角形双劈线倍比": "right_triangle_two_cevian_ratio",
    "本地等长对角线凸四边形最大面积": "equal_diagonal_quadrilateral_maximum_area",
    "本地幂差数值半群最小缺失正整数": "power_difference_semigroup_smallest_gap",
    "本地递推数列逐项共同互素正整数集": "recurrence_universal_coprime_set",
    "本地四次加五分裂域三项答案": "quartic_plus_five_splitting_field",
    "本地洪水屏障临界建墙速度": "flood_barrier_critical_speed",
    "本地递归删位整除最大值": "recursive_digit_deletion_maximum",
    "本地相邻约束满射计数": "adjacent_surjection_count",
    "本地重复字母隔位计数": "multiset_no_adjacent_count",
    "本地二进制游程计数": "binary_run_avoidance_count",
    "本地手链轨道计数": "bracelet_no_adjacent_count",
    "本地条带格路计数": "strip_lattice_path_count",
    "本地嵌套模幂和": "nested_modular_power_sum",
    "本地二次型最大值": "quadratic_form_maximum",
    "本地树度数普查": "tree_degree_census",
    "本地对合置换不动点计数": "involution_fixed_point_count",
    "本地复化梯形精确计算": "composite_trapezoid",
    "本地二点Gauss-Legendre精确计算": "two_point_gauss_legendre_monomial",
    "本地指数函数列L1判定": "exponential_l1_sequence",
    "本地循环距离二染色计数": "cycle_distance_two_coloring",
    "本地有向圆柱三行Hamilton路径计数": "directed_cylinder_hamilton_paths",
    "本地排序三角形失效指标上界": "sorted_triangle_failure_bound",
    "本地全排列相邻积锐界": "sparkling_tuple_pair_sum",
    "本地五正数比值间隔锐界": "five_number_ratio_gap",
    "本地三重嵌套非负整数数列值集": "nested_nonnegative_sequence_values",
    "本地纯三次域最低次数多项式": "mysterious_cuberoot_polynomial",
    "本地完全二部图同态下界": "complete_bipartite_homomorphism_bound",
    "本地全等三角剖分切多边形": "tangential_identical_triangulation_polygon",
    "本地散度型L2伴随算子": "formal_l2_adjoint",
    "本地三维混合进位锐阈值": "mixed_radix_grid_compression",
    "本地二面体群选择答案": "square_dihedral_facts",
    "本地勒贝格可积选择答案": "lebesgue_integrability_facts",
    "本地实数紧集选择答案": "compact_real_facts",
    "本地Cauchy准则选择答案": "cauchy_complete_space_facts",
    "本地线性规划对偶选择答案": "linear_programming_duality_facts",
    "本地矩阵条件数选择答案": "matrix_condition_number_definition",
    "本地Dirichlet边值离散化方法": "dirichlet_pde_discretization_methods",
    "本地时间序列构成选择答案": "time_series_components",
    "本地时间序列季节调整方法": "seasonal_adjustment_methods",
    "本地数据分散程度指标": "dispersion_measure_standard_deviation",
    "本地回归方法选择答案": "unknown_form_regression",
    "本地逐步回归选择答案": "stepwise_removal",
    "本地非线性回归选择答案": "nonlinear_regression_estimation",
    "本地总量指标时间数列判断答案": "aggregate_series_ratio_truth",
    "本地异方差OLS判断答案": "heteroscedastic_ols_variance_truth",
    "本地异方差参数方差后果": "heteroscedastic_parameter_variance_consequence",
    "本地正态分布参数选择答案": "normal_distribution_parameters",
    "本地大型数据集概览图选择答案": "large_dataset_overview_plot",
    "本地三角格正六边形计数": "triangular_lattice_regular_hexagons",
    "本地临界直线覆盖点集最大值": "critical_line_cover_point_set",
    "本地二次丢番图解数奇偶参数": "even_quadratic_pair_count_parameters",
    "本地稀疏多米诺放置计数": "sparse_domino_placements",
    "本地红蓝点直线分区最小值": "red_blue_line_separation",
    "本地聚集集合区间极值": "clustered_interval_maximum",
    "本地二次变换不变多项式族": "quadratic_transform_invariant_polynomials",
    "本地障碍多米诺铺法计数": "punctured_domino_tilings",
    "本地唯一多米诺分割最少标记": "unique_domino_partition_marking",
    "本地完全交集族最大值": "complete_intersection_maximum",
    "本地受界广义Pell解计数": "bounded_generalized_pell_count",
    "本地整数多项式整除解集": "integer_polynomial_divisibility",
    "本地回文四次式非负参数": "reciprocal_quartic_nonnegative",
    "本地仿射递推行列式不变量": "affine_recurrence_determinant",
    "本地根上多项式乘积": "root_polynomial_product",
    "本地三角形劈线长度": "cevian_length",
    "本地Smith标准形": "smith_normal_form",
    "本地相交反链最大值": "intersecting_antichain_maximum",
    "本地二部图删匹配生成树": "bipartite_matching_deletion_trees",
    "本地完全图删Hamilton圈生成树": "complete_graph_cycle_deletion_trees",
    "本地圆周不相邻选择计数": "cyclic_nonadjacent_selection",
    "本地Wythoff博弈必败态计数": "wythoff_losing_position_count",
    "本地减法博弈必败态计数": "finite_subtraction_game",
    "本地等量取珠盒子最小值": "equal_marble_box_minimum",
    "本地平方减法博弈计数": "square_subtraction_game",
    "本地轮图正常着色计数": "wheel_coloring",
    "本地网格偏序线性扩张计数": "grid_poset_extensions",
    "本地超立方体生成树计数": "hypercube_spanning_trees",
    "本地奇数纤维函数计数": "odd_fiber_functions",
    "本地夫妻分组计数": "couples_unlabeled_groups",
    "本地约数个数范围计数": "bounded_divisor_count",
    "本地本原勾股三元组计数": "primitive_pythagorean_count",
    "本地欧拉函数逆像": "inverse_totient",
    "本地最大公约数求和": "gcd_sum",
    "本地正整数平方和计数": "positive_sum_two_squares",
    "本地阶乘商复合估值": "factorial_quotient_valuation",
    "本地Pell基本解": "pell_fundamental_solution",
    "本地最小约数数目整数": "least_integer_with_divisor_count",
    "本地可分解二次型整数解": "factorable_binary_quadratic",
    "本地三次根正整数参数解": "cube_root_positive_integer_pairs",
    "本地Descartes内切圆半径": "descartes_inner_circle",
    "本地定重旋转项链计数": "rotation_necklace_fixed_weight",
    "本地定重二色手链计数": "fixed_weight_binary_bracelets",
    "本地指定度数标号树计数": "specified_degree_labeled_trees",
    "本地奇长度循环置换计数": "odd_cycle_permutations",
    "本地幂同余不动点计数": "power_fixed_residue_count",
    "本地单位分数无序解和": "reciprocal_pair_sum",
    "本地整数格点非退化三角形计数": "integer_grid_nondegenerate_triangles",
    "本地Bose积分": "bose_einstein_integral",
    "本地Bernoulli似然比": "bernoulli_likelihood_ratio",
    "本地Brownian离区间期望": "brownian_exit_expectation",
    "本地同余方程解": "congruence_solution",
    "本地模幂计算": "modular_power",
}


_CHECK_LABELS: dict[str, str] = {
    "本地不相邻二进制串核验": "nonadjacent_binary_string_count_check",
    "本地排列条件计数核验": "precedence_permutation_count_check",
    "本地满射容斥核验": "surjection_count_check",
    "本地平面图欧拉核验": "planar_euler_faces_check",
    "本地抛物面曲率核验": "paraboloid_curvature_check",
    "本地圆周拉普拉斯核验": "circle_laplacian_check",
    "本地圆周Laplace-Beltrami核验": "circle_laplace_beltrami_check",
    "本地有理数约束传播核验": "rational_f2_constraint_check",
    "本地数位和窗口最小性核验": "digit_sum_window_check",
    "本地取数博弈状态核验": "number_writing_game_state_check",
    "本地路径配分函数递推核验": "path_partition_recurrence_check",
    "本地LZ78编码核验": "lz78_encoding_check",
    "本地尖峰函数构造核验": "spike_sequence_construction_check",
    "本地Bernoulli依赖构造核验": "dependent_bernoulli_construction_check",
}
LEGACY_LABEL_TO_OPERATION.update(_CHECK_LABELS)


def contract_for(operation: str) -> Optional[ToolContract]:
    contract = TOOL_CONTRACTS.get(operation)
    if contract is not None:
        return contract
    if operation.endswith("_check"):
        base = TOOL_CONTRACTS.get(operation[:-6])
        if base is not None:
            return replace(base, operation=operation, whole_answer_capable=False)
    # Two check labels retain historical names that are not base + "_check".
    aliases = {
        "path_partition_recurrence_check": "path_independent_set_partition",
        "number_writing_game_state_check": "number_writing_game",
    }
    base = TOOL_CONTRACTS.get(aliases.get(operation, ""))
    if base is not None:
        return replace(base, operation=operation, whole_answer_capable=False)
    return None


def result_from_legacy_hint(
    hint: str,
    *,
    trusted_source: bool = False,
    extra_checks: Iterable[str] = (),
    source_problem: str = "",
) -> Optional[ToolResult]:
    """Parse legacy text, granting a certificate only to an internal producer.

    A model or caller can reproduce a known prefix, so parsing text alone must
    not establish provenance.  ``SympyTool.results_for`` sets
    ``trusted_source`` after its deterministic handler produced the hint.
    """

    label, separator, result = str(hint or "").partition(": ")
    result = result.strip()
    if not separator or not result:
        return None
    operation = LEGACY_LABEL_TO_OPERATION.get(label, "local_hint")
    contract = contract_for(operation)
    passed = bool(trusted_source and contract is not None and contract.certified)
    checks = tuple(dict.fromkeys((
        "registered_operation",
        "deterministic_handler_matched",
        "nonempty_result",
        *tuple(extra_checks),
    ))) if passed else ()
    if passed:
        issues = ()
    elif contract is None:
        issues = ("unregistered_operation",)
    else:
        issues = ("untrusted_legacy_text",)
    certificate = ToolCertificate(
        passed=passed,
        method=contract.certificate_method if contract else "",
        checks=checks,
        issues=issues,
        source_fingerprint=(
            problem_fingerprint(source_problem) if passed and source_problem else ""
        ),
    )
    submission_result, support = _legacy_submission_payload(operation, result)
    return ToolResult(
        submission_result,
        operation,
        label,
        contract,
        certificate,
        support=support,
    )
