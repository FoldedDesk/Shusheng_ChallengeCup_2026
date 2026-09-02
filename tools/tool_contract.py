"""Contracts and reproducible certificates for general local math tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable, Optional


GENERIC_OPERATIONS = frozenset({
    "calculate",
    "solve_equation",
    "derivative",
    "laplacian",
    "central_difference_first_derivative",
    "definite_integral",
    "iterated_definite_integral",
    "limit",
    "matrix_determinant",
    "matrix_rank",
    "matrix_inverse",
    "matrix_eigenvalues",
    "explicit_modular_congruence_count",
    "explicit_polynomial_coefficient",
    "explicit_linear_recurrence_term",
    "function_evaluation",
    "finite_sum",
    "solve_linear_system",
    "graph_spanning_trees",
    "hypercube_spanning_trees",
    "wheel_chromatic_polynomial",
    "complete_multipartite_spanning_trees",
    "modular_square_root_count",
    "pandigital_divisibility_count",
    "surjective_adjacent_function_count",
    "multiset_no_adjacent_count",
    "binary_pattern_avoidance_count",
    "binary_fixed_weight_no_adjacent_count",
    "binary_bracelet_count",
    "lattice_path_band_count",
    "modular_power_tower_residue",
    "quadratic_form_maximum",
    "path_independence_polynomial",
    "bounded_digit_language_count",
    "digit_sum_window_nondivisibility",
    "monotone_affine_reachability_game",
    "lz78_phrase_encoding",
    "distinct_digit_deletion_divisor_maximum",
    "directed_three_row_cylinder_hamilton_paths",
    "quartic_binomial_splitting_field",
    "complete_graph_cover_time",
    "formal_divergence_adjoint",
    "two_bin_capacity_exact",
    "two_bin_capacity_normal_approximation",
    "cauchy_sequence_facts",
    "heine_borel_facts",
    "lebesgue_integrability_facts",
    "heteroscedasticity_facts",
    "dihedral_group_facts",
    "linear_program_duality_facts",
    "time_series_component_facts",
    "time_series_ratio_fact",
    "seasonal_adjustment_methods",
    "dispersion_statistic",
    "unknown_form_regression_method",
    "stepwise_regression_removal",
    "nonlinear_regression_estimation",
    "heteroscedastic_variance_direction",
    "poisson_dirichlet_discretization",
    "circle_intrinsic_laplacian",
    "even_cardinality_subsets",
    "deleted_bipartite_length_three_paths",
    "minimum_degree_path",
    "positive_composition_with_lower_bounds",
    "implication_chain",
    "cyclic_additive_generators",
    "divisor_poset_extrema",
    "affine_first_order_recurrence",
    "labelled_tree_exact_leaf_count",
    "tree_degree_distribution",
    "permutation_precedence_exclusion",
    "congruence_equivalence_class",
    "binomial_choose_two_equation",
    "connected_planar_face_count",
    "surjection_without_singleton_fibers",
    "surjection_count",
    "cyclic_group_subgroup_count",
    "nonadjacent_subset_count",
    "square_even_contrapositive",
    "adjacency_matrix_walk_interpretation",
    "involution_fixed_point_count",
    "index_two_normal_subgroup",
    "ordered_positive_triples",
    "boolean_complement_simplification",
    "tournament_hamilton_path",
    "linear_exact_quadrature_weights",
    "one_step_bisection",
    "bisection_approximation",
    "newton_iteration",
    "newton_approximation",
    "polynomial_interpolation",
    "composite_trapezoid",
    "composite_simpson",
    "jacobi_norm_convergence",
    "fixed_point_contraction",
    "forward_euler",
    "improved_euler",
    "runge_kutta_4",
    "explicit_runge_kutta_stability",
    "dirichlet_wave_energy",
    "linear_transport_pde_ivp",
    "taylor_polynomial",
    "secant_iteration",
    "secant_approximation",
    "condition_number_sensitivity",
    "chebyshev_nodes",
    "strict_diagonal_dominance_iteration",
    "shrinking_interval_measure",
    "power_sequence_integral_limit",
    "simple_indicator_integral",
    "nonnegative_zero_integral",
    "measure_union",
    "concentrating_spike_nonuniform_integrability",
    "moving_spike_sequence",
    "translating_indicator_uniform_integrability",
    "l1_absolute_continuity",
    "counting_measure_geometric_integral",
    "power_singularity_integrability",
    "monotone_convergence_integral",
    "plane_curve_curvature",
    "circle_arclength_curvature",
    "parametric_curve_speed",
    "graph_surface_principal_curvatures",
    "sphere_curvatures",
    "unit_tangent_orthogonality",
    "zero_curvature_line",
    "first_fundamental_form",
    "graph_gaussian_at_critical_point",
    "planar_conformal_gaussian_curvature",
    "parametric_surface_gaussian_curvature",
    "two_dice_conditional_probability",
    "bernoulli_variance_identity",
    "two_color_hypergeometric",
    "independent_event_union",
    "independent_standard_normal_sum",
    "coupon_collector_expectation",
    "conditional_order_statistic_given_maximum",
    "geometric_waiting_tail",
    "finite_discrete_moments",
    "dependent_bernoulli_construction",
    "additive_cyclic_element_order",
    "finite_abelian_exact_order_count",
    "power_element_order",
    "homomorphism_kernel_normal",
    "small_finite_field_irreducibility",
    "polynomial_quotient_power",
    "polynomial_root_bound",
    "finite_field_multiplicative_group",
    "maximal_ideal_quotient",
    "symmetric_random_walk_moments",
    "poisson_independent_increment",
    "finite_markov_absorption_entry",
    "continuous_birth_death_absorption_time",
    "brownian_covariance",
    "brownian_point_hitting_laplace",
    "renewal_strong_law",
    "stationary_covariance_lag",
    "period_two_markov_construction",
    "rational_contour_integral",
    "rouche_dominant_monomial_zero_count",
    "holomorphic_power_real_part",
    "rational_residue_at_point",
    "geometric_power_series",
    "liouville_bounded_entire",
    "conjugate_not_complex_differentiable",
    "linear_first_order_ivp",
    "constant_second_order_ivp",
    "constant_second_order_forced_ivp",
    "autonomous_power_blowup_ivp",
    "diagonal_linear_system_stability",
    "bernoulli_constant_coeff_general",
    "sample_mean_variance",
    "sample_mean_unbiased",
    "two_sided_z_critical_value",
    "unbiased_estimator_variance_choice",
    "poisson_exponential_umvu",
    "evaluation_functional_norm",
    "affine_multiplication_operator_spectrum",
    "right_shift_isometry",
    "hilbert_nearest_point_uniqueness",
    "finite_dimensional_subspace_closed",
    "regression_predictor_translation",
    "intercept_normal_equation",
    "ridge_from_gram_statistics",
    "simple_regression_r_squared",
    "uniform_wasserstein_squared",
    "heat_equation_solution_check",
    "wave_equation_solution_check",
    "transport_growth_pde_ivp",
    "harmonic_function_check",
    "dalembert_general_solution",
    "compact_extreme_value",
    "circle_point_fundamental_group",
    "matrix_trace_determinant_from_eigenvalues",
    "linear_program_two_variables",
    "continuous_nonnegative_zero_integral",
    "closed_subset_of_compact",
    "logarithmic_power_series_uniform_convergence",
    "real_projective_space_cellular_homology",
    "scaled_cauchy_kernel_limit",
    "surface_gaussian_curvature",
    "gauss_legendre_error",
    "gauss_legendre_quadrature",
    "implicit_two_step_stability",
    "jacobi_exact_iterations",
    "uniform_maximum_spacing_expectation",
    "conditional_order_statistic_given_extreme",
    "birth_death_hitting_probability",
    "birth_death_stationary_distribution",
    "contour_residue_integral",
    "linear_ode_ivp",
    "intercept_gls",
    "diagonal_gls_estimate",
    "full_covariance_gls_estimate",
    "integer_matrix_smith_cokernel",
    "mapping_torus_first_homology",
    "normal_wasserstein_squared_map",
    "exponential_survival_umvu",
    "poisson_disk_harmonic_measure",
    "abelian_presentation_snf",
    "nilpotent_jordan_partition",
    "uniform_scale_umvu",
    "sobolev_energy_minimization",
    "linear_program_2d_vertices",
    "covering_linear_program_primal_dual",
    "packing_linear_program_primal_dual",
    "binary_pattern_race",
    "plane_tree_degree_count",
    "complete_digraph_euler_circuits",
    "finite_group_nowhere_zero_flow",
    "finite_modular_divisibility_scan",
    "finite_field_irreducible_count",
    "finite_field_monic_irreducible_count",
    "quadratic_endpoint_energy",
    "sine_product_specialization",
    "periodic_unilateral_shift_spectrum",
    "volterra_operator_norm",
    "chebyshev_minimax_polynomial",
    "endpoint_concentration_limit",
    "candidate_symbolic_check",
    "candidate_range_check",
    "propositional_formula_classification",
    "finite_set_union",
    "finite_set_intersection",
    "finite_set_difference",
    "finite_set_symmetric_difference",
    "finite_set_cartesian_product",
    "explicit_graph_connected",
    "explicit_graph_tree",
    "explicit_graph_bipartite",
    "explicit_graph_degree_sequence",
    "explicit_graph_shortest_path",
    "explicit_graph_euler_path",
    "explicit_graph_euler_circuit",
    "finite_markov_transition_power",
    "finite_markov_stationary_distribution",
    "finite_markov_absorbing_states",
    "parameterized_subtraction_game",
    "parameterized_factorial_ratio_valuation",
    "parameterized_modular_power_sum",
    "parameterized_lattice_polygon_interior",
    "parameterized_permutation_cycle_inventory",
})

GENERIC_PRESENTATION_REQUIREMENTS = frozenset({
    "count_conclusion",
    "numeric_result",
})

DIRECT_ASSURANCE_LEVELS = frozenset({
    "symbolic",
    "exhaustive",
})
ASSURANCE_LEVELS = frozenset({
    *DIRECT_ASSURANCE_LEVELS,
    "schema",
})


class CertificateStatus(str, Enum):
    """Three-valued outcome of a local mathematical certificate.

    ``CERTIFIED_FALSE`` means that a fully checked local proposition was
    deterministically refuted.  A failed parser, missing assumption, runtime
    error, or incomplete check is always ``NOT_CERTIFIED``; it must never be
    promoted to either truth value.
    """

    CERTIFIED_TRUE = "CERTIFIED_TRUE"
    CERTIFIED_FALSE = "CERTIFIED_FALSE"
    NOT_CERTIFIED = "NOT_CERTIFIED"


def problem_fingerprint(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", str(problem or "")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


@dataclass(frozen=True)
class ToolContract:
    operation: str
    result_kind: str
    certificate_method: str
    assurance: str = "schema"
    whole_answer_capable: bool = False
    written_support_capable: bool = False
    max_goals: int = 1
    allowed_requirements: tuple[str, ...] = ()
    required_requirements: tuple[str, ...] = ()
    allowed_task_kinds: tuple[str, ...] = ("calculation", "fill_blank")
    allowed_answer_shapes: tuple[str, ...] = ()
    required_problem_facts: tuple[str, ...] = ()
    forbidden_problem_facts: tuple[str, ...] = ("unparsed_suffix", "extra_obligation")

    @property
    def certified(self) -> bool:
        return bool(
            self.operation in GENERIC_OPERATIONS
            and self.certificate_method
            and self.assurance in ASSURANCE_LEVELS
        )

    @property
    def direct_submission_capable(self) -> bool:
        """Whether the result was independently recomputed, not template-matched."""
        return bool(
            self.certified
            and self.assurance in DIRECT_ASSURANCE_LEVELS
            and self.whole_answer_capable
        )

    def covers(
        self,
        goal_count: int,
        requirements: Iterable[str] = (),
        *,
        task_kind: str = "",
        answer_shape: str = "",
        problem_facts: Optional[Iterable[str]] = None,
    ) -> bool:
        requested = set(requirements)
        facts = set(problem_facts or ())
        return bool(
            self.certified
            and self.whole_answer_capable
            and goal_count == 1
            and goal_count <= self.max_goals
            and set(self.required_requirements) <= requested
            and requested <= (
                set(self.allowed_requirements)
                | set(GENERIC_PRESENTATION_REQUIREMENTS)
            )
            and (not task_kind or task_kind in self.allowed_task_kinds)
            and (not answer_shape or not self.allowed_answer_shapes or answer_shape in self.allowed_answer_shapes)
            and set(self.required_problem_facts) <= facts
            and not set(self.forbidden_problem_facts).intersection(facts)
        )


@dataclass(frozen=True)
class ToolCertificate:
    passed: bool
    method: str
    checks: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    source_fingerprint: str = ""
    preconditions: tuple[str, ...] = ()
    execution_checks: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    certified_value: Optional[bool] = None

    @property
    def phases_complete(self) -> bool:
        return bool(
            self.preconditions
            and self.execution_checks
            and self.postconditions
        )

    @property
    def status(self) -> CertificateStatus:
        if (
            not self.passed
            or self.issues
            or not self.method
            or not self.source_fingerprint
            or not self.phases_complete
            or self.certified_value is None
        ):
            return CertificateStatus.NOT_CERTIFIED
        return (
            CertificateStatus.CERTIFIED_TRUE
            if self.certified_value
            else CertificateStatus.CERTIFIED_FALSE
        )

    def trace_content(self) -> dict:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "certified_value": self.certified_value,
            "method": self.method,
            "checks": list(self.checks),
            "issues": list(self.issues),
            "source_fingerprint": self.source_fingerprint,
            "phases": {
                "precondition": list(self.preconditions),
                "execution": list(self.execution_checks),
                "postcondition": list(self.postconditions),
                "complete": self.phases_complete,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    result: str
    operation: str
    label: str
    contract: Optional[ToolContract]
    certificate: ToolCertificate
    support: str = ""

    @property
    def verified(self) -> bool:
        return bool(
            self.certificate.status is CertificateStatus.CERTIFIED_TRUE
            and self.contract
            and self.contract.certified
        )

    @property
    def refuted(self) -> bool:
        return bool(
            self.certificate.status is CertificateStatus.CERTIFIED_FALSE
            and self.contract
            and self.contract.certified
        )

    @property
    def whole_answer_eligible(self) -> bool:
        return bool(self.verified and self.contract and self.contract.whole_answer_capable)

    @property
    def direct_submission_eligible(self) -> bool:
        return bool(
            self.verified
            and self.contract
            and self.contract.direct_submission_capable
        )

    @property
    def goal_result_eligible(self) -> bool:
        """Whether the result certifies the requested conclusion, excluding prose obligations."""
        return self.whole_answer_eligible

    @property
    def supported_submission_eligible(self) -> bool:
        return bool(
            self.goal_result_eligible
            and self.contract
            and self.contract.written_support_capable
            and self.support
        )

    def to_hint(self) -> str:
        return f"{self.label}: {self.support or self.result}"

    def trace_content(self) -> dict:
        return {
            "operation": self.operation,
            "result_kind": self.contract.result_kind if self.contract else "unknown",
            "assurance": self.contract.assurance if self.contract else "unknown",
            "whole_answer_eligible": self.whole_answer_eligible,
            "direct_submission_eligible": self.direct_submission_eligible,
            "goal_result_eligible": self.goal_result_eligible,
            "supported_submission_eligible": self.supported_submission_eligible,
            "certificate": self.certificate.trace_content(),
            "support": self.support,
        }


def make_tool_result(
    *,
    problem: str,
    operation: str,
    result: str,
    result_kind: str,
    method: str,
    whole: bool,
    written_support: bool = False,
    checks: Iterable[str],
    issues: Iterable[str] = (),
    preconditions: Iterable[str] = (),
    execution_checks: Iterable[str] = (),
    postconditions: Iterable[str] = (),
    certified_value: Optional[bool] = None,
    support: str = "",
    answer_shapes: tuple[str, ...] = (),
    requirements: tuple[str, ...] = ("result_present",),
    assurance: str = "schema",
) -> ToolResult:
    clean_result = str(result or "").strip()
    check_tuple = tuple(dict.fromkeys(str(item) for item in checks if str(item)))
    issue_tuple = tuple(dict.fromkeys(str(item) for item in issues if str(item)))
    source_fingerprint = problem_fingerprint(problem)
    precondition_tuple = tuple(dict.fromkeys(
        str(item) for item in preconditions if str(item)
    ))
    execution_tuple = tuple(dict.fromkeys(
        str(item) for item in execution_checks if str(item)
    ))
    postcondition_tuple = tuple(dict.fromkeys(
        str(item) for item in postconditions if str(item)
    ))
    contract = ToolContract(
        operation=operation,
        result_kind=result_kind,
        certificate_method=method,
        assurance=str(assurance or "schema"),
        whole_answer_capable=whole,
        written_support_capable=written_support,
        allowed_requirements=requirements,
        allowed_answer_shapes=answer_shapes,
    )
    passed = bool(
        operation in GENERIC_OPERATIONS
        and contract.assurance in ASSURANCE_LEVELS
        and clean_result
        and check_tuple
        and precondition_tuple
        and execution_tuple
        and postcondition_tuple
        and certified_value is not None
        and source_fingerprint
        and not issue_tuple
    )
    return ToolResult(
        result=clean_result,
        operation=operation,
        label=f"local.{operation}",
        contract=contract,
        certificate=ToolCertificate(
            passed=passed,
            method=method,
            checks=check_tuple,
            issues=issue_tuple,
            source_fingerprint=source_fingerprint,
            preconditions=precondition_tuple,
            execution_checks=execution_tuple,
            postconditions=postcondition_tuple,
            certified_value=certified_value,
        ),
        support=str(support or "").strip(),
    )


def make_parameterized_tool_result(**kwargs) -> ToolResult:
    """Certify a result produced by a guarded deterministic tool compiler.

    Domain tools call this only after their operation-specific recognizer has
    matched the current statement, parsed every required datum/hypothesis, run
    a fixed local implementation, and assembled explicit postcondition checks.
    Keeping this separate from ``make_tool_result`` preserves fail-closed
    behavior for every caller that has not established that three-phase
    contract.
    """
    operation = str(kwargs.get("operation", "") or "").strip()
    checks = tuple(kwargs.get("checks", ()))
    kwargs.setdefault("preconditions", (
        "current_statement_operation_anchor_matched",
        "required_problem_data_and_hypotheses_checked",
    ))
    kwargs.setdefault("execution_checks", (
        f"deterministic_{operation or 'whitelist'}_implementation_completed",
    ))
    kwargs.setdefault("postconditions", checks)
    kwargs.setdefault("certified_value", True)
    return make_tool_result(**kwargs)


_LEGACY_PREFIXES = {
    "SymPy 计算": ("calculate", "scalar", "sympy_exact", ("number", "expression")),
    "SymPy 方程解": ("solve_equation", "solution_set", "sympy_substitution", ("roots",)),
    "SymPy 导数": ("derivative", "expression", "sympy_differentiate", ("expression",)),
    "SymPy 定积分": ("definite_integral", "scalar", "sympy_exact", ("number", "expression")),
    "SymPy 极限": ("limit", "expression", "sympy_limit", ("number", "expression")),
}


def result_from_legacy_hint(
    hint: str,
    *,
    trusted_source: bool = False,
    extra_checks: Iterable[str] = (),
    source_problem: str = "",
) -> Optional[ToolResult]:
    """Compatibility parser limited to the generic operations above."""
    if not trusted_source:
        return None
    label, separator, value = str(hint or "").partition(":")
    spec = _LEGACY_PREFIXES.get(label.strip())
    if not separator or spec is None or not value.strip():
        return None
    operation, result_kind, method, shapes = spec
    return make_tool_result(
        problem=source_problem,
        operation=operation,
        result=value.strip(),
        result_kind=result_kind,
        method=method,
        whole=False,
        checks=("trusted_generic_producer", *tuple(extra_checks)),
        answer_shapes=shapes,
    )
