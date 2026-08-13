"""Structured contracts and certificates for deterministic math tools.

The public agent historically consumed strings such as ``"SymPy 计算: 4"``.
This module keeps that representation available while giving the solver an
explicit allow-list: an unknown label is evidence text, never a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, Optional


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

    def trace_content(self) -> dict:
        return {
            "passed": self.passed,
            "method": self.method,
            "checks": list(self.checks),
            "issues": list(self.issues),
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
        requirements=("independent_increments",),
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
    "digit_permutation_divisibility": _contract(
        "digit_permutation_divisibility", "count", "bounded_exact_enumeration", True,
        answer_shapes=("number",),
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
            "all_ten_decimal_digits_exhausted", "maximality_by_empty_longer_layers",
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
    "本地数字排列整除计数": "digit_permutation_divisibility",
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
