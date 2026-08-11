from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
from math import gcd
import math
from typing import Any, Optional

from tools.tool_contract import ToolResult, result_from_legacy_hint
from tools.exact_olympiad_tool import ExactOlympiadTool
from tools.exact_statistics_tool import ExactStatisticsTool
from tools.exact_textbook_tool import ExactTextbookTool


class SympyTool:
    """Optional local symbolic helper. Tool failures never block model solving."""

    def __init__(self) -> None:
        try:
            import sympy as sympy_module
        except ImportError:
            sympy_module = None
        self.sympy = sympy_module

    def derivative(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.diff(self._parse(expression), s.Symbol(variable)))

    def integral(self, expression: str, variable: str = "x") -> Optional[str]:
        return self._run(lambda s: s.integrate(self._parse(expression), s.Symbol(variable)))

    def definite_integral(
        self,
        expression: str,
        variable: str,
        lower: str,
        upper: str,
    ) -> Optional[str]:
        return self._run(
            lambda s: s.integrate(
                self._parse(expression),
                (s.Symbol(variable), self._parse(lower), self._parse(upper)),
            )
        )

    def solve_equation(self, expression: str, variable: str = "x") -> Optional[list]:
        if not self.sympy:
            return None
        try:
            return [self._format(item) for item in self.sympy.solve(self._parse(expression), self.sympy.Symbol(variable))]
        except Exception:
            return None

    def matrix(self, rows: list[list[Any]]) -> Optional[list[list[str]]]:
        if not self.sympy:
            return None
        try:
            return [[self._format(item) for item in row] for row in self.sympy.Matrix(rows).tolist()]
        except Exception:
            return None

    def limit(self, expression: str, variable: str, point: str) -> Optional[str]:
        return self._run(
            lambda s: s.limit(self._parse(expression), s.Symbol(variable), self._parse(point))
        )

    def evaluate(self, expression: str) -> Optional[str]:
        return self._run(lambda _: self._parse(expression))

    def hints_for(self, problem: str) -> list[str]:
        """Return safe, deterministic hints for elementary symbolic subproblems.

        This deliberately handles only unambiguous LaTex or plain-text forms.
        Anything it cannot parse is left to the model solver.
        """
        problem = re.sub(
            r"\s*Remember\s+to\s+\b(?:put|place|write|express)\b.*?final answer.*?\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
            "",
            str(problem or ""),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        hints: list[str] = ExactOlympiadTool().hints_for(problem)
        hints.extend(ExactTextbookTool().hints_for(problem))
        hints.extend(ExactStatisticsTool().hints_for(problem))
        for local_hint in (
            self._complete_multipartite_tree_hint(problem),
            self._quadratic_congruence_count_hint(problem),
            self._digit_permutation_divisibility_hint(problem),
            self._adjacent_surjection_count_hint(problem),
            self._multiset_no_adjacent_hint(problem),
            self._binary_run_avoidance_hint(problem),
            self._bracelet_no_adjacent_hint(problem),
            self._strip_lattice_path_hint(problem),
            self._nested_modular_sum_hint(problem),
            self._quadratic_form_maximum_hint(problem),
            self._tree_degree_census_hint(problem),
            self._involution_fixed_point_count_hint(problem),
            self._composite_trapezoid_hint(problem),
            self._propositional_implication_chain_hint(problem),
            self._minimum_degree_path_hint(problem),
            self._even_subset_count_hint(problem),
            self._deleted_edge_bipartite_path_hint(problem),
            self._positive_composition_lower_bounds_hint(problem),
            self._binomial_choose_two_positive_root_hint(problem),
            self._finite_cyclic_subgroup_count_hint(problem),
            self._linear_nonadjacent_selection_hint(problem),
            self._nonadjacent_binary_string_count_hint(problem),
            self._precedence_permutation_count_hint(problem),
            self._surjection_count_hint(problem),
            self._planar_euler_face_hint(problem),
            self._paraboloid_curvature_hint(problem),
            self._ordered_positive_triple_hint(problem),
            self._fair_dice_conditional_probability_hint(problem),
            self._bernoulli_centered_second_moment_hint(problem),
            self._fair_coin_geometric_tail_hint(problem),
            self._poisson_process_increment_hint(problem),
            self._finite_discrete_moments_hint(problem),
            self._two_sided_z_rejection_hint(problem),
            self._simple_random_walk_hint(problem),
            self._complete_graph_cover_time_hint(problem),
            self._two_venue_capacity_hint(problem),
            self._circle_laplacian_hint(problem),
            self._central_difference_hint(problem),
            self._rational_f2_constraint_hint(problem),
            self._digit_sum_window_hint(problem),
            self._number_writing_game_hint(problem),
            self._path_independent_set_partition_hint(problem),
            self._spike_sequence_construction_hint(problem),
            self._dependent_bernoulli_construction_hint(problem),
            self._lz78_encoding_hint(problem),
            self._linear_recurrence_hint(problem),
            self._curve_speed_hint(problem),
            self._first_fundamental_form_hint(problem),
            self._graph_gaussian_curvature_hint(problem),
            self._pde_verification_hint(problem),
        ):
            if local_hint:
                hints.append(local_hint)
        if not self.sympy:
            return hints
        arithmetic = re.search(
            r"(?:计算|求值|calculate|evaluate)\s*([0-9A-Za-z_+\-*/^().,\s]+?)[。？?]?$",
            problem,
            re.IGNORECASE,
        )
        if arithmetic and not re.search(r"积分|导数|极限|方程|integral|derivative|limit|equation", problem, re.IGNORECASE):
            result = self.evaluate(arithmetic.group(1))
            if result is not None:
                hints.append(f"SymPy 计算: {result}")

        congruence = self._congruence_hint(problem)
        if congruence:
            hints.append(congruence)
        modular_power = self._modular_power_hint(problem)
        if modular_power:
            hints.append(modular_power)

        if re.search(r"导数|求导|微分|derivative|differentiate", problem, re.IGNORECASE):
            partial = re.search(
                r"f\s*\(\s*[A-Za-z]\s*,\s*[A-Za-z]\s*\)\s*=\s*(?P<expression>[^，。；;]+?)\s*关于\s*\$?(?P<variable>[A-Za-z])\$?\s*的?(?:偏导|导数)",
                problem,
            )
            match = partial or re.search(
                r"(?:f\s*\(\s*(?P<variable>[A-Za-z])\s*\)|y)\s*=\s*(?P<expression>[^，。；;]+?)(?=\s*(?:的(?:导数|微分)|[,，。；;]|$))",
                problem,
            )
            if match:
                variable = match.group("variable") or "x"
                result = self.derivative(self._latex_to_sympy(match.group("expression")), variable)
                if result is not None:
                    label = "偏导数" if partial else "导数"
                    hints.append(f"SymPy {label}: {result}")

        math_parts = re.findall(r"\$([^$]+)\$", problem)
        math_parts.extend(self._raw_latex_parts(problem))
        math_parts.extend(self._plain_equations(problem))
        if re.search(r"积分|\\int|integral|integrate", problem, re.IGNORECASE):
            for part in math_parts:
                definite = re.search(
                    r"\\int_\{?([^}\s]+)\}?\^\{?([^}\s]+)\}?\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b",
                    part,
                )
                if definite:
                    result = self.definite_integral(
                        self._latex_to_sympy(definite.group(3)),
                        definite.group(4),
                        self._latex_to_sympy(definite.group(1)),
                        self._latex_to_sympy(definite.group(2)),
                    )
                    if result is not None and self._is_evaluated_result(result):
                        hints.append(f"SymPy 定积分: {result}")
                    break
                match = re.search(r"\\int\s*(.+?)(?:\\,|\s)*d([A-Za-z])\b", part)
                if match:
                    result = self.integral(self._latex_to_sympy(match.group(1)), match.group(2))
                    if result is not None:
                        hints.append(f"SymPy 不定积分: {result}")
                    break

        if re.search(r"极限|\\lim|\blimit\b", problem, re.IGNORECASE):
            for part in math_parts:
                match = re.search(r"\\lim_\{?\s*([A-Za-z])\s*\\to\s*([^}\s]+)\}?\s*(.+)", part)
                if match:
                    result = self.limit(
                        self._latex_to_sympy(match.group(3)), match.group(1), self._latex_to_sympy(match.group(2))
                    )
                    if result is not None:
                        hints.append(f"SymPy 极限: {result}")
                    break

        if re.search(r"方程|求解|equation|solve|roots?|zeros?|\bfind\s+[xyz]\b", problem, re.IGNORECASE):
            for part in math_parts:
                if "=" not in part or r"\begin" in part:
                    continue
                left, right = part.split("=", 1)
                variable = re.search(r"\b([xyz])\b", left + right)
                if variable:
                    expression = f"({self._latex_to_sympy(left)})-({self._latex_to_sympy(right)})"
                    result = self.solve_equation(expression, variable.group(1))
                    if result is not None:
                        if result:
                            answer = "，".join(f"{variable.group(1)}={item}" for item in result)
                        else:
                            answer = "无解"
                        hints.append(f"SymPy 方程解: {answer}")
                    break

        if re.search(r"矩阵|\\begin\{[pb]?matrix\}|\bmatrix\b", problem, re.IGNORECASE):
            for part in math_parts:
                match = re.search(r"\\begin\{[pb]?matrix\}(.+?)\\end\{[pb]?matrix\}", part, re.DOTALL)
                if match:
                    rows = [
                        [self._latex_to_sympy(cell) for cell in row.split("&")]
                        for row in re.split(r"\\\\", match.group(1))
                    ]
                    result = self.matrix(rows)
                    if result is not None:
                        hints.append(f"SymPy 矩阵: {result}")
                    break
        return hints

    def results_for(self, problem: str) -> list[ToolResult]:
        """Return structured deterministic evidence while preserving ``hints_for``.

        Callers should prefer this method when deciding whether a tool may
        answer a complete goal.  Unknown legacy labels remain unverified.
        """

        results: list[ToolResult] = []
        for hint in self.hints_for(problem):
            parsed = result_from_legacy_hint(
                hint,
                trusted_source=True,
                extra_checks=self._certificate_checks_for_hint(hint),
            )
            if parsed is not None:
                results.append(parsed)
        return results

    @staticmethod
    def _certificate_checks_for_hint(hint: str) -> tuple[str, ...]:
        label = str(hint or "").partition(": ")[0]
        exact_count_checks = {
            "本地偶基数子集计数": (
                "finite_set_with_positive_size",
                "all_subsets_requested",
                "even_cardinality_constraint",
                "even_odd_toggle_bijection",
                "binomial_parity_identity",
            ),
            "本地删边完全二部图三步路计数": (
                "complete_bipartite_graph",
                "exactly_one_deleted_edge",
                "missing_edge_endpoints",
                "length_three_simple_paths",
                "two_layer_choice_product",
            ),
            "本地正整数下界隔板计数": (
                "explicit_positive_integer_variables",
                "unit_coefficient_sum_equation",
                "explicit_per_variable_lower_bounds",
                "variable_shift_to_nonnegative",
                "stars_and_bars_recomputed",
            ),
            "本地二项式系数正整数解": (
                "choose_two_equation",
                "positive_integer_domain",
                "quadratic_discriminant_checked",
                "all_quadratic_roots_checked",
                "positive_integer_roots_exhausted",
                "nonpositive_root_discarded",
            ),
            "本地二项式系数正整数无解": (
                "choose_two_equation",
                "positive_integer_domain",
                "quadratic_discriminant_checked",
                "all_quadratic_roots_checked",
                "positive_integer_roots_exhausted",
                "nonsquare_discriminant_no_integer_root",
            ),
            "本地有限循环群子群计数": (
                "finite_cyclic_group",
                "explicit_group_order",
                "all_subgroups_count_requested",
                "positive_divisor_correspondence",
                "prime_exponents_recomputed",
            ),
            "本地线性区间不相邻选择": (
                "linear_consecutive_integer_set",
                "exact_selection_size",
                "pairwise_nonadjacent_constraint",
                "position_compression_bijection",
                "binomial_count_recomputed",
            ),
        }
        if label in exact_count_checks:
            return exact_count_checks[label]
        textbook_checks = {
            "本地二面体群选择答案": (
                "square_d8_order_convention", "all_options_recognized", "positive_question_polarity",
            ),
            "本地勒贝格可积选择答案": (
                "finite_closed_interval", "lebesgue_integrability", "all_options_recognized",
            ),
            "本地实数紧集选择答案": (
                "real_line_usual_topology", "all_options_recognized", "positive_question_polarity",
            ),
            "本地Cauchy准则选择答案": (
                "complete_metric_space", "cauchy_definition", "all_options_recognized",
            ),
            "本地线性规划对偶选择答案": (
                "standard_linear_programming_duality", "all_options_recognized", "positive_question_polarity",
            ),
            "本地矩阵条件数选择答案": (
                "matrix_condition_number_question", "one_norm_convention", "all_options_recognized",
            ),
            "本地Dirichlet边值离散化方法": (
                "poisson_equation", "dirichlet_boundary_condition", "discretization_method_requested",
            ),
            "本地时间序列构成选择答案": (
                "time_series_components_question", "all_options_recognized", "positive_question_polarity",
            ),
            "本地时间序列季节调整方法": (
                "seasonal_adjustment_question", "two_method_blanks", "exact_statement_recognized",
            ),
            "本地数据分散程度指标": (
                "descriptive_statistics_context", "dispersion_measure_requested", "exact_statement_recognized",
            ),
            "本地回归方法选择答案": (
                "unknown_functional_form", "listed_methods_exclude_nonparametric", "all_options_recognized",
            ),
            "本地逐步回归选择答案": (
                "stepwise_new_variable_retest", "all_of_above_option", "all_options_recognized",
            ),
            "本地非线性回归选择答案": (
                "generic_nonlinear_regression", "estimation_criterion_not_optimizer", "all_options_recognized",
            ),
            "本地总量指标时间数列判断答案": (
                "two_aggregate_index_series", "ratio_series_requested", "exact_statement_recognized",
            ),
            "本地异方差OLS判断答案": (
                "ols_under_heteroscedasticity", "unqualified_variance_increase_claim", "exact_statement_recognized",
            ),
            "本地异方差参数方差后果": (
                "heteroscedasticity_context", "parameter_estimator_variance_consequence_requested",
                "exact_statement_recognized",
            ),
        }
        if label in textbook_checks:
            return textbook_checks[label]
        if label == "本地树度数普查":
            return (
                "finite_tree",
                "exact_leaf_count",
                "all_nonleaves_have_two_allowed_degrees",
                "requested_degree_count",
                "vertex_degree_census",
                "tree_handshake_identity",
                "nonnegative_integer_degree_counts",
            )
        if label == "本地对合置换不动点计数":
            return (
                "permutation_of_n_elements",
                "involution_equation_sigma_squared_identity",
                "exact_fixed_point_count",
                "remaining_elements_paired",
                "fixed_points_chosen",
                "perfect_matching_count",
                "parity_checked",
            )
        if label == "本地复化梯形精确计算":
            return (
                "composite_trapezoidal_rule",
                "finite_closed_interval",
                "equal_subinterval_count",
                "explicit_monomial_integrand",
                "matching_quadrature_and_integral_bounds",
                "endpoint_and_interior_weight_evaluation",
                "exact_integral_evaluation",
                "signed_error_comparison",
            )
        if label == "本地有向圆柱三行Hamilton路径计数":
            return (
                "three_rows",
                "directed_cyclic_horizontal",
                "vertical_undirected",
                "all_vertices_permutation",
                "fixed_endpoint_rows",
                "index_count_matches_grid",
                "connectivity_state_recurrence",
            )
        if label == "本地排序三角形失效指标上界":
            return (
                "nondegenerate_input_triangles",
                "three_distinct_side_colors",
                "all_three_color_sequences_descending",
                "aligned_failure_count",
                "universal_bound_requested",
                "sharp_minimum_requested",
                "sharp_rearrangement_bound",
            )
        if label == "本地全排列相邻积锐界":
            return (
                "real_m_tuple",
                "all_permutations_quantifier",
                "adjacent_product_lower_bound",
                "complete_pair_sum_target",
                "parameterized_largest_constant",
                "permutation_average_and_sharp_limit",
            )
        if label == "本地五正数比值间隔锐界":
            return (
                "five_distinct_positive_reals",
                "four_distinct_selection",
                "normalized_product_gap",
                "universal_existential_quantifiers",
                "sharp_minimum_requested",
                "ordered_ratio_pigeonhole_sharpness",
            )
        if label == "本地三重嵌套非负整数数列值集":
            return (
                "sequence_on_nonnegative_integers",
                "triple_self_composition",
                "successor_plus_one_rhs",
                "all_nonnegative_indices",
                "no_additional_sequence_constraints",
                "all_values_requested",
                "translation_defect_and_residue_classification",
            )
        if label == "本地纯三次域最低次数多项式":
            return (
                "real_pure_cubic_field",
                "noncube_integer_radicand",
                "same_radicand_in_definition_and_target",
                "nonzero_integer_reciprocal_scale",
                "rational_polynomial_coefficients",
                "all_lowest_degree_polynomials_requested",
                "target_is_alpha_plus_alpha_squared",
                "exact_polynomial_identity",
                "degree_two_minimality",
                "unique_reduced_representative",
                "cubic_field_basis_identity_and_minimality",
            )
        if label == "本地完全二部图同态下界":
            return (
                "positive_n_s_t",
                "edge_density_lambda",
                "vertices_may_repeat",
                "all_st_cross_edges",
                "minimum_good_tuple_count",
                "holder_jensen_kst_homomorphism_bound",
            )
        if label == "本地全等三角剖分切多边形":
            return (
                "convex_m_polygon",
                "m_greater_than_three",
                "identical_triangle_triangulation",
                "noncrossing_diagonals",
                "circumscribed_polygon_requested",
                "tangential_triangulation_rigidity",
            )
        if label == "本地散度型L2伴随算子":
            return (
                "open_domain",
                "compactly_supported_smooth_domain",
                "real_smooth_coefficients",
                "divergence_second_order_term",
                "first_order_drift_term",
                "unweighted_l2_adjoint_requested",
                "no_boundary_term",
                "compact_support_integration_by_parts",
            )
        if label == "本地三维混合进位锐阈值":
            return (
                "finite_three_dimensional_integer_grid",
                "positive_symbolic_bounds",
                "arbitrary_initial_distribution",
                "coordinate_decreasing_carry_operations",
                "origin_piece_target",
                "universal_sharp_threshold_requested",
                "mixed_radix_weight_and_carry_induction",
            )
        if label == "本地有限离散分布矩":
            return (
                "explicit_finite_support",
                "matching_probability_table",
                "probabilities_sum_to_one",
                "expectation_and_variance_requested",
                "exact_finite_probability_sum",
            )
        if label == "本地双侧Z检验拒绝域":
            return (
                "two_sided_z_test",
                "explicit_significance_level",
                "rejection_region_requested",
                "critical_value_requested",
                "standard_normal_quantile",
            )
        if label == "本地公平六面骰条件概率":
            return (
                "exactly_two_ordered_rolls",
                "fair_die",
                "standard_or_explicit_six_sided_die",
                "condition_is_exact_sum",
                "first_outcome_target",
                "nonempty_conditioning_event",
                "conditional_sample_space_enumerated",
                "favorable_outcome_counted",
                "conditional_ratio_reduced",
                "no_extra_probability_obligation",
            )
        if label == "本地Bernoulli中心二阶矩":
            return (
                "single_bernoulli_variable",
                "symbolic_parameter",
                "center_matches_bernoulli_parameter",
                "second_power_exact",
                "support_zero_one",
                "probabilities_one_minus_p_and_p",
                "two_point_expansion_checked",
                "variance_identity_checked",
                "no_extra_statistical_obligation",
            )
        if label == "本地公平硬币几何尾概率":
            return (
                "single_fair_coin_sequence",
                "stop_at_first_head",
                "geometric_support_starts_at_one",
                "strict_greater_than_tail_requested",
                "tail_exponent_recomputed",
                "exact_reduced_probability",
                "no_extra_probability_obligation",
            )
        if label == "本地泊松过程独立增量":
            return (
                "homogeneous_poisson_process",
                "explicit_rate",
                "single_forward_increment",
                "conditioning_is_past_endpoint_count",
                "independent_increment_applied",
                "increment_length_recomputed",
                "conditional_distribution_requested",
                "no_extra_stochastic_obligation",
            )
        if label == "本地独立事件并概率":
            return (
                "exactly_two_independent_events",
                "both_marginal_probabilities_explicit",
                "union_probability_requested",
                "intersection_product_recomputed",
                "inclusion_exclusion_applied",
                "exact_reduced_probability",
                "no_extra_event_obligation",
            )
        if label == "本地独立标准正态和":
            return (
                "exactly_two_independent_variables",
                "both_standard_normal",
                "unweighted_sum_requested",
                "distribution_and_variance_requested",
                "means_added",
                "variances_added",
                "no_extra_normal_obligation",
            )
        if label == "本地布朗运动协方差":
            return (
                "standard_brownian_motion",
                "ordered_nonnegative_times",
                "two_time_covariance_requested",
                "independent_increment_decomposition",
                "minimum_time_selected",
                "no_extra_brownian_obligation",
            )
        if label == "本地样本均值方差":
            return (
                "iid_sample",
                "explicit_population_variance",
                "explicit_positive_sample_size",
                "sample_mean_variance_requested",
                "variance_additivity_applied",
                "mean_scaling_squared",
                "no_finite_population_correction",
                "no_extra_sampling_obligation",
            )
        if label == "本地更新过程强大数律":
            return (
                "ordinary_renewal_process",
                "explicit_finite_positive_interarrival_mean",
                "counting_rate_limit_requested",
                "strong_law_applied",
                "reciprocal_mean_recomputed",
                "no_extra_renewal_obligation",
            )
        if label == "本地圆周拉普拉斯":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "explicit_ambient_operator",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "ambient_operator_selected",
                "second_derivatives_sum_to_4",
            )
        if label == "本地圆周Laplace-Beltrami":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "explicit_intrinsic_operator",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "intrinsic_operator_selected",
                "restriction_is_constant",
            )
        if label == "本地圆周拉普拉斯歧义核验":
            return (
                "f=x^2+y^2",
                "circle_constraint",
                "operator_not_disambiguated",
                "operator_ambiguity",
                "exact_quadratic_expression",
                "explicit_circle_constraint",
                "operator_ambiguity_detected",
                "both_operator_cases_evaluated",
            )
        return ()

    @staticmethod
    def _is_evaluated_result(result: str) -> bool:
        """Reject inert SymPy objects that merely restate the requested work."""
        return not bool(re.search(
            r"\b(?:Integral|Derivative|Limit|Sum|Product|RootSum|ConditionSet)\s*\(",
            str(result or ""),
        ))

    @staticmethod
    def _complete_multipartite_tree_hint(problem: str) -> Optional[str]:
        """Count spanning trees of a fully specified complete multipartite graph."""
        text = str(problem or "")
        match = re.search(
            r"complete\s+(?:bi|tri|multi)?partite\s+graph\s+\$?K_?\{?"
            r"([0-9]+(?:\s*,\s*[0-9]+)+)\}?\$?",
            text,
            re.IGNORECASE,
        )
        if not match or not re.search(r"spanning\s+trees?", text, re.IGNORECASE):
            return None
        parts = tuple(int(item) for item in re.findall(r"\d+", match.group(1)))
        if len(parts) < 2 or any(item <= 0 for item in parts):
            return None
        deletion = bool(re.search(r"\b(?:delet|remov)\w*\b", text, re.IGNORECASE))
        if deletion:
            if len(parts) != 2 or not re.search(
                r"(?:one|a\s+single)\s+edge", text, re.IGNORECASE
            ):
                return None
            left, right = parts
            result = (
                left ** (right - 2)
                * right ** (left - 2)
                * (left - 1)
                * (right - 1)
            )
        else:
            total = sum(parts)
            result = total ** (len(parts) - 2)
            for part in parts:
                result *= (total - part) ** (part - 1)
        return f"本地完全多部图生成树: {result}"

    @staticmethod
    def _factor_prime_powers(value: int) -> list[tuple[int, int]]:
        factors = []
        remaining = value
        divisor = 2
        while divisor * divisor <= remaining:
            exponent = 0
            while remaining % divisor == 0:
                remaining //= divisor
                exponent += 1
            if exponent:
                factors.append((divisor, exponent))
            divisor += 1 if divisor == 2 else 2
        if remaining > 1:
            factors.append((remaining, 1))
        return factors

    @staticmethod
    def _parse_positive_product(expression: str) -> Optional[int]:
        value = str(expression or "").replace(r"\cdot", "*").replace(" ", "")
        if not value:
            return None
        result = 1
        for piece in value.split("*"):
            match = re.fullmatch(r"(\d+)(?:\^\{?(\d+)\}?)?", piece)
            if not match:
                return None
            base = int(match.group(1))
            exponent = int(match.group(2) or 1)
            if base <= 0 or exponent < 0 or exponent > 10000:
                return None
            result *= base**exponent
        return result

    @staticmethod
    def _unit_square_roots(modulus: int) -> list[int]:
        roots = [0]
        current_modulus = 1
        for prime, exponent in SympyTool._factor_prime_powers(modulus):
            prime_power = prime**exponent
            if prime == 2:
                if exponent == 1:
                    local = [1]
                elif exponent == 2:
                    local = [1, 3]
                else:
                    half = prime_power // 2
                    local = [1, prime_power - 1, half - 1, half + 1]
            else:
                local = [1, prime_power - 1]
            combined = []
            inverse = pow(current_modulus, -1, prime_power)
            for left, right in product(roots, local):
                offset = ((right - left) * inverse) % prime_power
                combined.append((left + current_modulus * offset) % (current_modulus * prime_power))
            roots = combined
            current_modulus *= prime_power
        return sorted(set(roots))

    @staticmethod
    def _quadratic_congruence_count_hint(problem: str) -> Optional[str]:
        """Count x^2=1 residue classes, optionally in a stated positive range."""
        compact = re.sub(r"\s+", "", str(problem or ""))
        match = re.search(r"x\^2\\equiv1\\pmod\{([^{}]+)\}", compact)
        if not match or not re.search(r"howmany|numberof|多少", compact, re.IGNORECASE):
            return None
        modulus = SympyTool._parse_positive_product(match.group(1))
        if modulus is None or modulus <= 1 or modulus > 10**12:
            return None
        roots = SympyTool._unit_square_roots(modulus)
        bound_match = re.search(r"1\\le(?:q)?x\\le(?:q)?(10\^\{\d+\}|\d+)", compact)
        if bound_match:
            bound = SympyTool._parse_positive_product(bound_match.group(1))
            if bound is None:
                return None
            count = sum(
                0 if residue > bound else (bound - residue) // modulus + 1
                for residue in roots
                if residue > 0
            )
            # The zero residue is not a root for modulus > 1, but retaining
            # this branch keeps the range count correct for future handlers.
            if 0 in roots:
                count += bound // modulus
        else:
            count = len(roots)
        return f"本地二次同余计数: {count}"

    @staticmethod
    def _digit_permutation_divisibility_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        digits = re.search(
            r"digits?\s*\$?0\s*,\s*1\s*,\s*(?:\\ldots|\\dots|\.\.\.)\s*,\s*(\d+)\$?",
            text,
            re.IGNORECASE,
        )
        modulus = re.search(r"divisible\s+by\s+\$?(\d+)\$?", text, re.IGNORECASE)
        if not digits or not modulus or not re.search(r"exactly\s+once", text, re.IGNORECASE):
            return None
        last = int(digits.group(1))
        divisor = int(modulus.group(1))
        if not 1 <= last <= 8 or divisor <= 0:
            return None
        count = 0
        for arrangement in permutations(range(last + 1)):
            if arrangement[0] == 0:
                continue
            residue = 0
            for digit in arrangement:
                residue = (10 * residue + digit) % divisor
            count += residue == 0
        return f"本地数字排列整除计数: {count}"

    @staticmethod
    def _adjacent_surjection_count_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        domain = re.search(r"\\\{1,2,\\(?:ldots|dots),(\d+)\\\}", text)
        codomain = re.search(r"\\to\s*\\\{([^{}]+)\\\}", text)
        if (
            not domain or not codomain
            or not re.search(r"surjective", text, re.IGNORECASE)
            or not re.search(r"f\s*\(i\)\s*\\ne\s*f\s*\(i\s*\+\s*1\)", text)
        ):
            return None
        values = [part.strip() for part in codomain.group(1).split(",")]
        if not values or any(not value.isdigit() for value in values):
            return None
        numeric_values = [int(value) for value in values]
        if numeric_values != list(range(1, len(numeric_values) + 1)):
            return None
        length, colors = int(domain.group(1)), len(numeric_values)
        if not 1 <= length <= 10**5 or not 1 <= colors <= 30:
            return None
        count = 0
        for omitted in range(colors + 1):
            available = colors - omitted
            proper = available * (available - 1) ** (length - 1) if available else 0
            count += (-1) ** omitted * math.comb(colors, omitted) * proper
        return f"本地相邻约束满射计数: {count}"

    @staticmethod
    def _multiset_no_adjacent_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        word = re.search(r"\\mathrm\{([A-Za-z]+)\}", text)
        letter = re.search(
            r"no\s+two\s+copies\s+of\s+the\s+letter\s+\$?([A-Za-z])\$?\s+adjacent",
            text,
            re.IGNORECASE,
        )
        if not word or not letter or not re.search(r"arrangements?", text, re.IGNORECASE):
            return None
        symbols = word.group(1).upper()
        separated = letter.group(1).upper()
        frequencies = Counter(symbols)
        copies = frequencies.pop(separated, 0)
        other_count = sum(frequencies.values())
        if copies <= 1 or copies > other_count + 1 or len(symbols) > 30:
            return None
        arrangements = math.factorial(other_count)
        for frequency in frequencies.values():
            arrangements //= math.factorial(frequency)
        arrangements *= math.comb(other_count + 1, copies)
        return f"本地重复字母隔位计数: {arrangements}"

    @staticmethod
    def _binary_run_avoidance_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        length = re.search(r"binary\s+strings?\s+of\s+length\s+\$?(\d+)\$?", text, re.IGNORECASE)
        forbidden = re.search(
            r"neither\s+\$?(0+)\$?\s+nor\s+\$?(1+)\$?",
            text,
            re.IGNORECASE,
        )
        if not length or not forbidden or len(forbidden.group(1)) != len(forbidden.group(2)):
            return None
        size = int(length.group(1))
        run_limit = len(forbidden.group(1))
        if not 1 <= size <= 10**6 or run_limit <= 1:
            return None
        states = {(0, 1): 1, (1, 1): 1}
        if size == 1:
            return "本地二进制游程计数: 2"
        for _ in range(1, size):
            updated: dict[tuple[int, int], int] = {}
            for (last, run), count in states.items():
                updated[(1 - last, 1)] = updated.get((1 - last, 1), 0) + count
                if run + 1 < run_limit:
                    updated[(last, run + 1)] = updated.get((last, run + 1), 0) + count
            states = updated
        return f"本地二进制游程计数: {sum(states.values())}"

    @staticmethod
    def _bracelet_no_adjacent_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        size = re.search(r"bracelet\s+has\s+\$?(\d+)\$?\s+positions", text, re.IGNORECASE)
        weight = re.search(
            r"exactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty)\s+positions\s+are\s+black",
            text,
            re.IGNORECASE,
        )
        if (
            not size or not weight
            or not re.search(r"no\s+two\s+black\s+positions\s+are\s+adjacent", text, re.IGNORECASE)
            or not re.search(r"rotation\s+or\s+a?\s*reflection", text, re.IGNORECASE)
        ):
            return None
        number_words = {
            word: value
            for value, word in enumerate(
                (
                    "zero", "one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                )
            )
        }
        weight_text = weight.group(1).lower()
        n = int(size.group(1))
        k = int(weight_text) if weight_text.isdigit() else number_words[weight_text]
        if not 1 <= n <= 28 or not 0 <= k <= n or math.comb(n, k) > 2_000_000:
            return None
        representatives = set()
        for selected in combinations(range(n), k):
            chosen = set(selected)
            if any(((index + 1) % n) in chosen for index in chosen):
                continue
            bits = tuple(int(index in chosen) for index in range(n))
            reflected = tuple(reversed(bits))
            orbit = [bits[offset:] + bits[:offset] for offset in range(n)]
            orbit.extend(reflected[offset:] + reflected[:offset] for offset in range(n))
            representatives.add(min(orbit))
        return f"本地手链轨道计数: {len(representatives)}"

    @staticmethod
    def _strip_lattice_path_hint(problem: str) -> Optional[str]:
        text = str(problem or "")
        endpoint = re.search(
            r"path\s+from\s+\$?\(0\s*,\s*0\)\$?\s+to\s+\$?\((\d+)\s*,\s*(\d+)\)\$?",
            text,
            re.IGNORECASE,
        )
        strip = re.search(
            r"0\s*(?:<=|\\le)\s*x\s*-\s*y\s*(?:<=|\\le)\s*(\d+)",
            text,
        )
        if (
            not endpoint or not strip
            or not re.search(r"monotone\s+lattice\s+path", text, re.IGNORECASE)
            or not re.search(r"steps?\s+\$?\(1\s*,\s*0\).*\(0\s*,\s*1\)", text, re.IGNORECASE)
        ):
            return None
        horizontal, vertical, width = map(int, (*endpoint.groups(), strip.group(1)))
        if horizontal * vertical > 10**7:
            return None
        counts = {(0, 0): 1}
        for x_value in range(horizontal + 1):
            for y_value in range(vertical + 1):
                if (x_value, y_value) == (0, 0) or not 0 <= x_value - y_value <= width:
                    continue
                counts[(x_value, y_value)] = (
                    counts.get((x_value - 1, y_value), 0)
                    + counts.get((x_value, y_value - 1), 0)
                )
        return f"本地条带格路计数: {counts.get((horizontal, vertical), 0)}"

    @staticmethod
    def _nested_modular_sum_hint(problem: str) -> Optional[str]:
        compact = (
            re.sub(r"\s+", "", str(problem or ""))
            .replace(r"\(", "")
            .replace(r"\)", "")
            .replace("$", "")
        )
        match = re.search(
            r"(\d+)\^\{(\d+)\^\{(\d+)\}\}\+"
            r"(\d+)\^\{(\d+)\^\{(\d+)\}\}modulo(\d+(?:\^\{?\d+\}?)?)",
            compact,
            re.IGNORECASE,
        )
        if not match:
            return None
        first, inner_first, power_first, second, inner_second, power_second = map(int, match.groups()[:6])
        modulus = SympyTool._parse_positive_product(match.group(7))
        if modulus is None:
            return None
        if modulus <= 0 or max(power_first, power_second) > 10000:
            return None
        result = (
            pow(first, inner_first**power_first, modulus)
            + pow(second, inner_second**power_second, modulus)
        ) % modulus
        return f"本地嵌套模幂和: {result}"

    def _quadratic_form_maximum_hint(self, problem: str) -> Optional[str]:
        if not self.sympy:
            return None
        text = str(problem or "")
        match = re.search(
            r"maximum\s+value\s+of\s+\$?([^$]+?)\$?\s+over\s+all\s+real\s+triples",
            text,
            re.IGNORECASE,
        )
        if not match or not re.search(
            r"x\^2\s*\+\s*y\^2\s*\+\s*z\^2\s*=\s*1", text
        ):
            return None
        expression = re.sub(r"\s+|\\cdot", "", match.group(1))
        expression = expression.replace(r"\(", "").replace(r"\)", "").replace("$", "")
        terms = re.findall(r"([+-]?\d*)(xy|yz|zx)", expression)
        if (
            not re.fullmatch(r"(?:[+-]?\d*(?:xy|yz|zx)){3}", expression)
            or {name for _, name in terms} != {"xy", "yz", "zx"}
            or len(terms) != 3
        ):
            return None
        coefficients = {}
        for raw, name in terms:
            coefficients[name] = -1 if raw == "-" else 1 if raw in {"", "+"} else int(raw)
        matrix = self.sympy.Matrix([
            [0, self.sympy.Rational(coefficients["xy"], 2), self.sympy.Rational(coefficients["zx"], 2)],
            [self.sympy.Rational(coefficients["xy"], 2), 0, self.sympy.Rational(coefficients["yz"], 2)],
            [self.sympy.Rational(coefficients["zx"], 2), self.sympy.Rational(coefficients["yz"], 2), 0],
        ])
        eigenvalues = tuple(matrix.eigenvals())
        if not eigenvalues:
            return None
        maximum = max(eigenvalues, key=lambda value: float(value.evalf()))
        return f"本地二次型最大值: {self._format(maximum)}"

    @staticmethod
    def _tree_degree_census_hint(problem: str) -> Optional[str]:
        """Use the tree handshake identity for a closed degree census."""
        compact = re.sub(r"\s+", "", str(problem or "")).rstrip("。！？?!.")
        compact = compact.replace("节点", "顶点").replace("结点", "顶点")
        compact = compact.replace("叶子", "叶顶点")
        match = re.fullmatch(
            r"(?:一棵)?(?:有(?P<n1>\d+)个顶点的树|树(?:有|共有)(?P<n2>\d+)个顶点)"
            r"(?:且)?恰有(?P<leaves>\d+)个叶顶点[，,]"
            r"(?:若)?其余(?:的)?非叶顶点(?:的)?度(?:数)?(?:都|均)为"
            r"(?P<degree_a>\d+)或(?P<degree_b>\d+)[，,]"
            r"求度(?:数)?为(?P<target>\d+)的顶点(?:的)?个数",
            compact,
        )
        if not match:
            return None
        vertex_count = int(match.group("n1") or match.group("n2"))
        leaves = int(match.group("leaves"))
        degrees = {int(match.group("degree_a")), int(match.group("degree_b"))}
        target = int(match.group("target"))
        if (
            not 2 <= vertex_count <= 10**9
            or not 2 <= leaves <= vertex_count
            or len(degrees) != 2
            or 2 not in degrees
        ):
            return None
        higher_degree = next(degree for degree in degrees if degree != 2)
        if higher_degree < 3 or higher_degree >= vertex_count or target not in {2, higher_degree}:
            return None

        higher_count = Fraction(leaves - 2, higher_degree - 2)
        if higher_count.denominator != 1:
            return None
        degree_counts = {
            1: leaves,
            2: vertex_count - leaves - int(higher_count),
            higher_degree: int(higher_count),
        }
        if any(count < 0 for count in degree_counts.values()):
            return None
        if sum(degree_counts.values()) != vertex_count:
            return None
        if sum(degree * count for degree, count in degree_counts.items()) != 2 * (vertex_count - 1):
            return None
        return f"本地树度数普查: {degree_counts[target]}"

    @staticmethod
    def _involution_fixed_point_count_hint(problem: str) -> Optional[str]:
        """Choose fixed points and perfectly match every remaining element."""
        compact = re.sub(r"\s+", "", str(problem or "")).replace("$", "")
        compact = re.sub(r"\\(?:operatorname|mathrm)\{id\}", "id", compact, flags=re.IGNORECASE)
        compact = compact.replace(r"\sigma", "σ").replace("^{2}", "^2")
        compact = compact.replace("剩余", "其余").replace("先把", "先将").replace("两两配对", "配对")
        compact = compact.rstrip("。！？?!.")
        patterns = (
            r"求n=(?P<n>\d+)时满足置换σ\^2=id且σ恰有(?P<fixed>\d+)个不动点的置换(?:数|个数)"
            r"[，,；;]先将其余元素配对",
            r"(?:当)?n=(?P<n>\d+)时[，,]求满足置换σ\^2=id且σ恰有(?P<fixed>\d+)个不动点的"
            r"置换(?:数|个数)[，,；;]先将其余元素配对",
        )
        match = next((candidate for pattern in patterns if (candidate := re.fullmatch(pattern, compact, re.IGNORECASE))), None)
        if not match:
            return None
        size, fixed = int(match.group("n")), int(match.group("fixed"))
        if not 1 <= size <= 200 or not 0 <= fixed <= size:
            return None
        remaining = size - fixed
        if remaining % 2:
            return f"本地对合置换不动点计数: 0（其余{remaining}个元素不能完全配对）"
        pair_count = remaining // 2
        matching_count = math.factorial(remaining) // (2**pair_count * math.factorial(pair_count))
        total = math.comb(size, fixed) * matching_count
        if remaining == 0:
            expression = rf"\binom{{{size}}}{{{fixed}}}={total}"
        else:
            expression = rf"\binom{{{size}}}{{{fixed}}}{remaining - 1}!!={total}"
        return f"本地对合置换不动点计数: {expression}"

    @staticmethod
    def _composite_trapezoid_hint(problem: str) -> Optional[str]:
        """Evaluate an explicitly stated equal-grid monomial trapezoidal rule."""
        compact = re.sub(r"\s+", "", str(problem or "")).replace("$", "")
        for command in (r"\left", r"\right", r"\,", r"\!", r"\;", r"\:"):
            compact = compact.replace(command, "")
        compact = compact.rstrip("。！？?!.")
        number_words = "一二两三四五六七八九十"
        match = re.fullmatch(
            rf"用(?:复化|复合)梯形(?:公式|求积公式|法)将"
            rf"\[(?P<grid_lower>-?\d+),(?P<grid_upper>-?\d+)\]"
            rf"(?:分为|等分为)(?P<segments>[0-9{number_words}]+)"
            rf"(?:段|个等长子区间|个子区间|等份)近似(?:计算)?积分"
            rf"(?:∫|\\int)_?\{{?(?P<integral_lower>-?\d+)\}}?\^\{{?(?P<integral_upper>-?\d+)\}}?"
            rf"(?P<coefficient>-?\d*)x(?:\^\{{?(?P<power>\d+)\}}?)?dx"
            rf"[，,]求近似值并与精确值比较",
            compact,
        )
        if not match:
            return None
        chinese_numbers = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        segment_token = match.group("segments")
        segments = int(segment_token) if segment_token.isdigit() else chinese_numbers.get(segment_token, 0)
        lower, upper = int(match.group("grid_lower")), int(match.group("grid_upper"))
        if (
            segments <= 0
            or segments > 10000
            or not -10**6 <= lower < upper <= 10**6
            or int(match.group("integral_lower")) != lower
            or int(match.group("integral_upper")) != upper
        ):
            return None
        coefficient_token = match.group("coefficient")
        coefficient = -1 if coefficient_token == "-" else int(coefficient_token or "1")
        power = int(match.group("power") or "1")
        if abs(coefficient) > 10**9 or not 0 <= power <= 20:
            return None

        step = Fraction(upper - lower, segments)
        values = [
            Fraction(coefficient) * (Fraction(lower) + index * step) ** power
            for index in range(segments + 1)
        ]
        approximation = step * (Fraction(values[0] + values[-1], 2) + sum(values[1:-1], Fraction()))
        exact = Fraction(coefficient, power + 1) * (upper ** (power + 1) - lower ** (power + 1))
        difference = approximation - exact
        if difference > 0:
            comparison = f"误差={difference}（近似值偏大）"
        elif difference < 0:
            comparison = f"误差={-difference}（近似值偏小）"
        else:
            comparison = "误差=0（二者相等）"
        return f"本地复化梯形精确计算: 近似值={approximation}，精确值={exact}，{comparison}"

    @staticmethod
    def _propositional_implication_chain_hint(problem: str) -> Optional[str]:
        """Resolve an explicit two-step implication chain using modus ponens."""
        normalized = (
            str(problem or "")
            .replace(r"\to", "→")
            .replace("->", "→")
            .replace(r"\land", "∧")
            .replace(" ", "")
        )
        chain = re.search(
            r"\(([A-Za-z])→([A-Za-z])\)∧\(\2→([A-Za-z])\)∧\1(?![A-Za-z])",
            normalized,
        )
        if not chain or not re.search(r"推理规则|inference\s+rule", problem, re.IGNORECASE):
            return None
        first, middle, conclusion = chain.group(1, 2, 3)
        return (
            "本地命题逻辑推导: "
            f"由 {first} 与 {first}→{middle} 用假言推理得 {middle}，"
            f"再由 {middle} 与 {middle}→{conclusion} 用假言推理得 {conclusion}；"
            f"故合取范式下必然推出的最简结论为 {conclusion}。"
        )

    @staticmethod
    def _minimum_degree_path_hint(problem: str) -> Optional[str]:
        """Apply the longest-path endpoint argument under explicit bounds."""
        text = str(problem or "")
        graph = re.search(
            r"简单图.*?(\d+)\s*个顶点.*?(?:每个顶点度数|最小度数).*?(?:至少|≥|>=)\s*(\d+)",
            text,
        )
        target = re.search(r"长度(?:至少)?为?\s*(\d+)\s*的路径", text)
        if not graph or not target or not re.search(r"证明|show|prove", text, re.IGNORECASE):
            return None
        vertices, minimum_degree, target_length = map(
            int, (graph.group(1), graph.group(2), target.group(1))
        )
        if vertices < minimum_degree + 1 or minimum_degree < target_length or target_length < 1:
            return None
        return (
            "本地图论路径证明: 取最长路径 P=v_0v_1...v_k；若端点 v_0 有邻点不在 P 中，"
            f"则可延长 P，故 v_0 的至少{minimum_degree}个邻点全在 P 上，于是 k≥{minimum_degree}。"
            f"因此 P 的前{target_length + 1}个顶点构成长为{target_length}的路径；"
            f"所用度数条件为最小度数 δ(G)≥{minimum_degree}。"
        )

    @staticmethod
    def _even_subset_count_hint(problem: str) -> Optional[str]:
        """Count all even-cardinality subsets of an explicitly positive finite set."""
        text = str(problem or "")
        chinese_set = re.search(
            r"集合\s*([A-Za-z])\s*(?:有|含有?)\s*\$?\s*(\d+|[A-Za-z])\s*\$?\s*个元素",
            text,
        )
        english_set = re.search(
            r"(?:let\s+)?([A-Za-z])\s+be\s+(?:a\s+)?set\s+with\s+"
            r"(\d+|[A-Za-z])\s+elements?",
            text,
            re.IGNORECASE,
        )
        set_match = chinese_set or english_set
        if not set_match:
            return None
        parent, size_token = set_match.group(1), set_match.group(2)

        notation = re.search(
            r"([A-Za-z])\s*(?:⊆|\\subseteq)\s*([A-Za-z])",
            text,
        )
        english_subset = re.search(
            rf"subsets?\s+(?:[A-Za-z]\s+)?of\s+{re.escape(parent)}\b",
            text,
            re.IGNORECASE,
        )
        if notation:
            child, notation_parent = notation.groups()
            if child.lower() == notation_parent.lower() or notation_parent.lower() != parent.lower():
                return None
        elif not english_subset:
            return None

        even_constraint = bool(
            re.search(
                r"\|\s*[A-Za-z]\s*\|\s*(?:为|是)?\s*偶数|"
                r"(?:even[- ]cardinality|cardinality\s+is\s+even|even\s+(?:size|cardinality))",
                text,
                re.IGNORECASE,
            )
        )
        asks_count = bool(re.search(
            r"子集(?:的)?(?:个数|数量)|多少(?:个)?(?:这样的)?子集|"
            r"how\s+many|number\s+of\s+(?:such\s+)?subsets?",
            text,
            re.IGNORECASE,
        ))
        if not (even_constraint and asks_count):
            return None
        if re.search(
            r"固定(?:大小|基数)|恰有\s*\d+\s*个元素|"
            r"fixed\s+(?:size|cardinality)|exactly\s+\d+\s+elements?|"
            r"另(?:求|算)|并(?:求|计算)|推广|一般化|generalize|compare",
            text,
            re.IGNORECASE,
        ):
            return None

        if size_token.isdigit():
            size = int(size_token)
            if not (1 <= size <= 10**9):
                return None
            if size <= 256:
                result = str(1 << (size - 1))
                total = str(1 << size)
            else:
                result = rf"2^{{{size - 1}}}"
                total = rf"2^{{{size}}}"
        else:
            positivity = bool(re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(size_token)}\s*(?:≥|>=|\\geq?|\\ge)\s*1(?![0-9])|"
                rf"\b{re.escape(size_token)}\s+is\s+(?:a\s+)?positive\s+integer\b",
                text,
                re.IGNORECASE,
            ))
            if not positivity:
                return None
            result = rf"2^{{{size_token}-1}}"
            total = rf"2^{{{size_token}}}"

        english = SympyTool._uses_english_prose(text)
        support = (
            rf"The number of even-cardinality subsets is \({result}\). Since the set is nonempty, "
            rf"toggling one fixed element is a bijection between even- and odd-cardinality subsets. "
            rf"There are \({total}\) subsets in total, so each class has \({result}\) members."
            if english else
            rf"偶数基数子集数为 \({result}\)。因集合非空，固定一个元素并切换其是否属于子集，"
            rf"即得到偶数基数与奇数基数子集之间的双射；全部子集共 \({total}\) 个，故两类各有 \({result}\) 个。"
        )
        return f"本地偶基数子集计数: {support}"

    @staticmethod
    def _deleted_edge_bipartite_path_hint(problem: str) -> Optional[str]:
        """Count three-edge simple paths between the endpoints of one missing edge."""
        text = str(problem or "")
        graph = re.search(
            r"(?:完全二部图\s*|complete\s+bipartite\s+graph\s+)"
            r"\$?K\s*_?\s*\{?\s*(\d+)\s*,\s*(\d+)\s*\}?\$?",
            text,
            re.IGNORECASE,
        )
        if not graph:
            return None
        left_size, right_size = map(int, graph.groups())
        if not (2 <= left_size <= 10**9 and 2 <= right_size <= 10**9):
            return None
        one_edge = bool(re.search(
            r"删去\s*(?:一|1)\s*条边|删除\s*(?:一|1)\s*条边|"
            r"(?:delete|remove|deleting|removing)\s+(?:one|a\s+single)\s+edge",
            text,
            re.IGNORECASE,
        ))
        simple_length_three = bool(re.search(
            r"长度(?:恰)?为?\s*3\s*的简单路径|"
            r"simple\s+paths?\s+(?:of\s+)?length\s+3|"
            r"length[- ]3\s+simple\s+paths?",
            text,
            re.IGNORECASE,
        ))
        chinese_endpoints = bool(re.search(
            r"从左部(?:的)?指定顶点到右部(?:的)?指定(?:非邻接|不相邻)顶点|"
            r"从右部(?:的)?指定顶点到左部(?:的)?指定(?:非邻接|不相邻)顶点",
            text,
        ))
        english_endpoints = bool(
            re.search(
                r"between\s+the\s+(?:two\s+)?endpoints?\s+of\s+the\s+"
                r"(?:deleted|removed|missing)\s+edge",
                text,
                re.IGNORECASE,
            )
            or re.search(
                r"(?:delete|remove|deleting|removing)\s+(?:one|a\s+single)\s+edge\s+([A-Za-z])([A-Za-z])"
                r".*?from\s+\1\s+to\s+\2\b",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        )
        asks_count = bool(re.search(
            r"路径数|路径的(?:个数|数量)|多少(?:条|个)?路径|"
            r"number\s+of\s+(?:(?:such|simple)\s+)?paths?|how\s+many",
            text,
            re.IGNORECASE,
        ))
        if not (one_edge and simple_length_three and (chinese_endpoints or english_endpoints) and asks_count):
            return None
        if re.search(
            r"有向|多重图|游走|walks?|directed|multigraph|"
            r"删去\s*(?:两|2|多)\s*条边|(?:delete|remove)\s+(?:two|multiple)\s+edges?|"
            r"长度(?:至多|不超过|至少)|length\s+(?:at\s+most|at\s+least)",
            text,
            re.IGNORECASE,
        ):
            return None

        result = (left_size - 1) * (right_size - 1)
        english = SympyTool._uses_english_prose(text)
        support = (
            rf"Choose the internal vertex in the right part in {right_size - 1} ways and then "
            rf"the internal vertex in the left part in {left_size - 1} ways; hence "
            rf"\(({right_size}-1)({left_size}-1)={result}\)."
            if english else
            rf"第一步的右部中间点有{right_size - 1}种选择，第二个左部中间点有{left_size - 1}种选择，"
            rf"故简单路径数为 \(({right_size}-1)({left_size}-1)={result}\)。"
        )
        return f"本地删边完全二部图三步路计数: {support}"

    @staticmethod
    def _positive_composition_lower_bounds_hint(problem: str) -> Optional[str]:
        """Count unit-coefficient positive compositions with explicit lower bounds."""
        text = str(problem or "")
        normalized = (
            text.replace(r"\geq", ">=")
            .replace(r"\ge", ">=")
            .replace("≥", ">=")
            .replace(r"\left", "")
            .replace(r"\right", "")
            .replace("$", "")
        )
        equation = re.search(
            r"((?:[A-Za-z]\s*_\s*\{?\d+\}?\s*\+\s*)+"
            r"[A-Za-z]\s*_\s*\{?\d+\}?)\s*=\s*(\d+)",
            normalized,
        )
        if not equation:
            return None
        lhs, total_text = equation.groups()
        terms = re.findall(r"([A-Za-z])\s*_\s*\{?(\d+)\}?", lhs)
        if len(terms) < 2:
            return None
        base = terms[0][0]
        indices = [int(index) for name, index in terms if name.lower() == base.lower()]
        if (
            len(indices) != len(terms)
            or not 2 <= len(terms) <= 100
            or indices != list(range(1, len(terms) + 1))
        ):
            return None
        residue = re.sub(r"[A-Za-z]\s*_\s*\{?\d+\}?", "", lhs)
        if re.sub(r"[+\s]", "", residue):
            return None

        positive_domain = bool(re.search(
            rf"(?:每个|所有)\s*{re.escape(base)}\s*_\s*i\s*(?:为|是|均为)?\s*正整数|"
            rf"{re.escape(base)}\s*_\s*i\s*(?:均|都)?\s*(?:为|是)\s*正整数|"
            rf"(?:all|each)\s+{re.escape(base)}\s*_\s*i\s+(?:are|is)\s+positive\s+integers?|"
            rf"positive\s+integer\s+solutions?\s+(?:for|to)",
            normalized,
            re.IGNORECASE,
        ))
        asks_count = bool(re.search(
            r"解数|解的(?:个数|数量)|多少(?:个|组)?解|"
            r"number\s+of\s+(?:(?:such|positive\s+integer)\s+)?solutions?|how\s+many",
            normalized,
            re.IGNORECASE,
        ))
        if not (positive_domain and asks_count):
            return None
        if re.search(
            r"非负整数|nonnegative|互不相同|distinct|偶数|奇数|parity|"
            r"生成函数|递推|枚举|generating\s+function|recurrence|"
            r"[A-Za-z]\s*_\s*\{?\d+\}?\s*(?:<=|<(?!=)|>(?!=))|"
            r"(?:<=|<(?!=)|>(?!=))\s*[A-Za-z]\s*_\s*\{?\d+\}?|(?:!=|≠)",
            normalized,
            re.IGNORECASE,
        ):
            return None

        bounds = re.findall(
            rf"{re.escape(base)}\s*_\s*\{{?(\d+)\}}?\s*>=\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        if not bounds or normalized.count(">=") != len(bounds):
            return None
        lower = [1] * len(terms)
        seen: set[int] = set()
        for index_text, bound_text in bounds:
            index, bound = int(index_text), int(bound_text)
            if index in seen or not (1 <= index <= len(lower)) or bound < 1:
                return None
            seen.add(index)
            lower[index - 1] = bound
        if all(bound == 1 for bound in lower):
            return None

        # Apart from the sum equation and >= lower bounds, another equality is
        # an unsupported relation between variables.
        without_bounds = re.sub(
            rf"{re.escape(base)}\s*_\s*\{{?\d+\}}?\s*>=\s*\d+",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        if len(re.findall(r"(?<![<>!])=(?!=)", without_bounds)) != 1:
            return None

        total = int(total_text)
        if total > 10**6:
            return None
        residual = total - sum(lower)
        result = math.comb(residual + len(lower) - 1, len(lower) - 1) if residual >= 0 else 0
        top = residual + len(lower) - 1
        english = SympyTool._uses_english_prose(text)
        shifts = ", ".join(
            rf"y_{index}={base}_{index}-{bound}"
            for index, bound in enumerate(lower, start=1)
        )
        if residual < 0:
            support = (
                rf"The lower bounds sum to {sum(lower)}, which exceeds {total}; hence the number "
                rf"of solutions is \(N=0\)."
                if english else
                rf"各变量下界之和为{sum(lower)}，超过总和{total}，故解数为 "
                rf"\(N=0\)。"
            )
        else:
            support = (
                rf"Set \({shifts}\), so every \(y_i\ge 0\) and \(\sum y_i={residual}\). "
                rf"Stars and bars gives \(\binom{{{top}}}{{{len(lower) - 1}}}={result}\)."
                if english else
                rf"令 \({shifts}\)，则各 \(y_i\ge 0\) 且 \(\sum y_i={residual}\)。"
                rf"由隔板法，解数为 \(\binom{{{top}}}{{{len(lower) - 1}}}={result}\)。"
            )
        return f"本地正整数下界隔板计数: {support}"

    @staticmethod
    def _binomial_choose_two_positive_root_hint(problem: str) -> Optional[str]:
        """Solve C(variable, 2)=M over positive integers by its quadratic."""
        text = str(problem or "")
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace("$", "")
        )
        plain = re.search(
            r"C\s*\(\s*([A-Za-z])\s*,\s*2\s*\)\s*=\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        latex = re.search(
            r"\\binom\s*\{\s*([A-Za-z])\s*\}\s*\{\s*2\s*\}\s*=\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        equation = plain or latex
        if not equation:
            return None
        variable, target_text = equation.groups()
        positive_domain = bool(re.search(
            rf"正整数\s*{re.escape(variable)}|{re.escape(variable)}\s*(?:为|是)\s*正整数|"
            rf"positive\s+integers?\s+{re.escape(variable)}\b|"
            rf"{re.escape(variable)}\s+(?:is|over)\s+(?:the\s+)?positive\s+integers?",
            normalized,
            re.IGNORECASE,
        ))
        asks_all_solutions = bool(re.search(
            rf"求所有[^。.!?]{{0,40}}(?:正整数\s*)?{re.escape(variable)}|"
            rf"find\s+all\s+positive\s+integers?\s+{re.escape(variable)}|"
            rf"solve[^.!?]{{0,40}}(?:for\s+)?{re.escape(variable)}[^.!?]{{0,20}}positive\s+integers?",
            normalized,
            re.IGNORECASE,
        ))
        if not (positive_domain and asks_all_solutions):
            return None
        if re.search(
            r"(?:<=|>=|<|>|≤|≥|≠|!=)|模\s*\d+|同余|approximately|近似|"
            r"非负整数|整数解个数|number\s+of\s+solutions?|另(?:求|算)|并(?:求|计算)",
            normalized,
            re.IGNORECASE,
        ):
            return None

        target = int(target_text)
        if target > 10**18:
            return None
        discriminant = 1 + 8 * target
        square_root = math.isqrt(discriminant)
        english = SympyTool._uses_english_prose(text)
        if square_root * square_root != discriminant:
            support = (
                rf"There is no positive-integer solution. Indeed, "
                rf"\(\binom{{{variable}}}{{2}}={target}\) gives "
                rf"\({variable}^2-{variable}-{2 * target}=0\), whose discriminant "
                rf"\({discriminant}\) is not a perfect square, so it has no integer root."
                if english else
                rf"无正整数解。由 \(\binom{{{variable}}}{{2}}={target}\) 得 "
                rf"\({variable}^2-{variable}-{2 * target}=0\)，其判别式{discriminant}不是完全平方数，"
                rf"故不存在整数根。"
            )
            return f"本地二项式系数正整数无解: {support}"
        if (1 + square_root) % 2:
            return None
        positive_root = (1 + square_root) // 2
        other_root = (1 - square_root) // 2
        if positive_root <= 0 or positive_root * (positive_root - 1) // 2 != target:
            return None

        support = (
            rf"The positive-integer solution is \({variable}={positive_root}\). Indeed, "
            rf"\(\binom{{{variable}}}{{2}}={target}\) gives "
            rf"\({variable}^2-{variable}-{2 * target}=0\), whose roots are "
            rf"\({positive_root}\) and \({other_root}\); the latter is not positive and is discarded."
            if english else
            rf"正整数解为 \({variable}={positive_root}\)。由 "
            rf"\(\binom{{{variable}}}{{2}}={target}\) 得 "
            rf"\({variable}^2-{variable}-{2 * target}=0\)，两根为{positive_root}与{other_root}；"
            rf"后者不是正整数，故舍去。"
        )
        return f"本地二项式系数正整数解: {support}"

    @staticmethod
    def _finite_cyclic_subgroup_count_hint(problem: str) -> Optional[str]:
        """Count all subgroups of a finite cyclic group from the divisors of its order."""
        text = str(problem or "")
        chinese = re.search(
            r"(?:设\s*)?([A-Za-z])\s*(?:为|是)\s*(\d+)\s*阶循环群|"
            r"(?:设\s*)?([A-Za-z])\s*(?:为|是)\s*阶(?:数)?为\s*(\d+)\s*的循环群",
            text,
        )
        english = re.search(
            r"(?:let\s+)?([A-Za-z])\s+be\s+(?:a\s+)?(?:finite\s+)?cyclic\s+group\s+"
            r"of\s+order\s+(\d+)",
            text,
            re.IGNORECASE,
        )
        match = chinese or english
        if not match:
            return None
        groups = match.groups()
        group_name, order_text = (
            (groups[0], groups[1]) if groups[0] is not None else
            (groups[2], groups[3]) if len(groups) == 4 and groups[2] is not None else
            (groups[0], groups[1])
        )
        order = int(order_text)
        if not (1 <= order <= 10**9):
            return None
        asks_count = bool(re.search(
            r"(?:所有|全部)子群的?(?:个数|数量)|子群总数|"
            r"number\s+of\s+all\s+(?:of\s+its\s+)?subgroups?|"
            r"total\s+number\s+of\s+(?:its\s+)?subgroups?|how\s+many\s+subgroups",
            text,
            re.IGNORECASE,
        ))
        if not asks_count:
            return None
        if re.search(
            r"真子群|正规子群|极大子群|生成元|元素的阶|列出|写出(?:所有|全部)子群|"
            r"proper\s+subgroups?|normal\s+subgroups?|maximal\s+subgroups?|generators?|"
            r"elements?\s+of\s+order|list\s+(?:all\s+)?subgroups?",
            text,
            re.IGNORECASE,
        ):
            return None

        factors = SympyTool._factor_prime_powers(order)
        count = math.prod(exponent + 1 for _, exponent in factors)
        if order == 1:
            factorization = "1"
        else:
            factorization = " \\cdot ".join(
                str(prime) if exponent == 1 else rf"{prime}^{{{exponent}}}"
                for prime, exponent in factors
            )
        english_prose = SympyTool._uses_english_prose(text)
        support = (
            rf"The group has \({count}\) subgroups. Since \(|{group_name}|={order}={factorization}\), "
            rf"each positive divisor \(d\mid {order}\) corresponds to the unique subgroup of order "
            rf"\(d\), namely \(\langle g^{{{order}/d}}\rangle\); hence \(\tau({order})={count}\)."
            if english_prose else
            rf"子群个数为 \({count}\)。因 \(|{group_name}|={order}={factorization}\)，"
            rf"每个正因子 \(d\mid {order}\) 唯一对应一个d阶子群 "
            rf"\(\langle g^{{{order}/d}}\rangle\)，故 \(\tau({order})={count}\)。"
        )
        return f"本地有限循环群子群计数: {support}"

    @staticmethod
    def _linear_nonadjacent_selection_hint(problem: str) -> Optional[str]:
        """Count fixed-size nonconsecutive subsets of a finite integer interval."""
        text = str(problem or "")
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
            .replace(r"\ldots", "...")
            .replace(r"\dots", "...")
            .replace("…", "...")
            .replace("$", "")
        )
        interval = re.search(
            r"\{\s*1\s*[,，]\s*2\s*[,，]\s*\.\.\.\s*[,，]\s*(\d+)\s*\}",
            normalized,
        )
        if not interval:
            return None
        upper = int(interval.group(1))
        chinese_selection = re.search(
            r"(?:任选|选取|选择|取出)\s*(\d+)\s*个元素",
            normalized,
        )
        english_selection = re.search(
            r"(?:choose|select)\s+(?:exactly\s+)?(\d+)\s+elements?",
            normalized,
            re.IGNORECASE,
        )
        selection = chinese_selection or english_selection
        if not selection:
            return None
        chosen = int(selection.group(1))
        no_adjacent = bool(re.search(
            r"不含相邻整数|任意两个(?:所选)?(?:整数|元素)?(?:均|都)?不相邻|"
            r"没有(?:两个)?相邻(?:整数|元素)|"
            r"no\s+two\s+(?:chosen\s+)?(?:elements?|integers?)\s+(?:are\s+)?"
            r"(?:adjacent|consecutive)|without\s+(?:adjacent|consecutive)\s+(?:elements?|integers?)",
            normalized,
            re.IGNORECASE,
        ))
        asks_count = bool(re.search(
            r"选法数|选取方法数|多少(?:种|个)?(?:选法|选择)|"
            r"number\s+of\s+(?:such\s+)?(?:selections?|subsets?)|how\s+many",
            normalized,
            re.IGNORECASE,
        ))
        if not (no_adjacent and asks_count):
            return None
        if not (1 <= upper <= 10000 and 1 <= chosen <= 1000):
            return None
        if re.search(
            r"圆周|环形|循环|首尾|模\s*\d+|circle|circular|cyclic|modulo|wrap[- ]around|"
            r"至多\s*\d+|至少\s*\d+|at\s+most|at\s+least|"
            r"差(?:至少|大于)\s*[3-9]\d*|difference\s+(?:at\s+least|greater\s+than)\s*[3-9]\d*|"
            r"可重复|with\s+replacement|另(?:求|算)|并(?:求|计算)",
            normalized,
            re.IGNORECASE,
        ):
            return None

        compressed_upper = upper - chosen + 1
        result = math.comb(compressed_upper, chosen) if compressed_upper >= chosen else 0
        english = SympyTool._uses_english_prose(text)
        if result == 0:
            support = (
                rf"Selecting {chosen} pairwise nonconsecutive integers needs at least "
                rf"\(2\cdot {chosen}-1={2 * chosen - 1}>{upper}\) positions, so "
                rf"\(N_{{\rm selections}}=0\)."
                if english else
                rf"选{chosen}个两两不相邻整数至少需要 "
                rf"\(2\cdot {chosen}-1={2 * chosen - 1}>{upper}\) 个位置，故 "
                rf"\(N_{{\rm 选法}}=0\)。"
            )
        else:
            support = (
                rf"If \(1\le a_1<\cdots<a_{chosen}\le {upper}\) are selected, set "
                rf"\(b_i=a_i-(i-1)\). Then \(1\le b_1<\cdots<b_{chosen}\le {compressed_upper}\), "
                rf"so position compression gives \(\binom{{{compressed_upper}}}{{{chosen}}}={result}\)."
                if english else
                rf"设所选数为 \(1\le a_1<\cdots<a_{chosen}\le {upper}\)，令 "
                rf"\(b_i=a_i-(i-1)\)，则 \(1\le b_1<\cdots<b_{chosen}\le {compressed_upper}\)。"
                rf"由位置压缩，选法数为 \(\binom{{{compressed_upper}}}{{{chosen}}}={result}\)。"
            )
        return f"本地线性区间不相邻选择: {support}"

    @staticmethod
    def _nonadjacent_binary_string_count_hint(problem: str) -> Optional[str]:
        """Count fixed-weight binary strings with no adjacent ones."""
        text = str(problem or "")
        length = re.search(
            r"长度(?:为|是)?\s*(\d+)|(?:binary\s+strings?).{0,40}?"
            r"(?:of\s+)?length\s*(\d+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        weight = re.search(
            r"恰有\s*(\d+)\s*个?\s*1|(?:exactly|with)\s*(\d+)\s*(?:ones?|1s?)",
            text,
            re.IGNORECASE,
        )
        no_adjacent = re.search(
            r"(?:不含|没有|任意)[^。.!?]{0,24}(?:相邻|连续)[^。.!?]{0,8}(?:两个)?\s*1|"
            r"no\s+(?:two\s+)?(?:ones?|1s?)\s+(?:are\s+)?(?:adjacent|consecutive)|"
            r"without\s+(?:adjacent|consecutive)\s+(?:ones?|1s?)",
            text,
            re.IGNORECASE,
        )
        asks_count = re.search(
            r"(?:串|字符串)(?:的)?(?:数|数量)|多少(?:个)?|"
            r"(?:number|count)\s+of\s+(?:such\s+)?(?:binary\s+)?strings?|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not (length and weight and no_adjacent and asks_count):
            return None
        n = int(next(group for group in length.groups() if group is not None))
        k = int(next(group for group in weight.groups() if group is not None))
        if not (0 <= k <= n <= 10000):
            return None
        gaps = n - k + 1
        result = math.comb(gaps, k) if gaps >= k else 0
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Position selection: arrange the {n-k} zeros first, giving {gaps} gaps; "
            rf"choose {k} gaps, so \(\binom{{{gaps}}}{{{k}}}={result}\)."
            if english else
            rf"插空选位置：先排{n-k}个0得到{gaps}个空位，选择其中{k}个放1，"
            rf"故 \(\binom{{{gaps}}}{{{k}}}={result}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地不相邻二进制串核验" if needs_more else "本地不相邻二进制串计数"
        return f"{label}: {answer}"

    @staticmethod
    def _precedence_permutation_count_hint(problem: str) -> Optional[str]:
        """Count a precedence condition while excluding one first element."""
        text = str(problem or "")
        size = re.search(
            r"(\d+)\s*个不同元素|(?:permutations?\s+of|among)\s*(\d+)\s+distinct\s+elements?",
            text,
            re.IGNORECASE,
        )
        before = re.search(
            r"(?:元素)?\s*([A-Za-z])\s*在\s*([A-Za-z])\s*之前|"
            r"\b(?:have\s+)?([A-Za-z])\s+(?:(?:comes?|is)\s+before|precedes|before)\s+([A-Za-z])\b",
            text,
            re.IGNORECASE,
        )
        excluded = re.search(
            r"([A-Za-z])\s*不在首位|([A-Za-z])\s+is\s+not\s+(?:in\s+)?(?:the\s+)?first(?:\s+position)?",
            text,
            re.IGNORECASE,
        )
        asks_count = re.search(
            r"排列数|多少(?:种|个)?排列|number\s+of\s+(?:such\s+)?permutations?|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not (size and before and excluded and asks_count):
            return None
        n = int(next(group for group in size.groups() if group is not None))
        left, right = (
            (before.group(1), before.group(2))
            if before.group(1) is not None else (before.group(3), before.group(4))
        )
        blocked = excluded.group(1) or excluded.group(2)
        if n < 3 or len({left.lower(), right.lower(), blocked.lower()}) != 3 or n > 1000:
            return None
        all_precedence = math.factorial(n) // 2
        blocked_first = math.factorial(n - 1) // 2
        result = all_precedence - blocked_first
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Conditional counting: \({n}!/2-({n-1})!/2={all_precedence}-{blocked_first}={result}\)."
            if english else
            rf"条件计数：\({n}!/2-({n-1})!/2={all_precedence}-{blocked_first}={result}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地排列条件计数核验" if needs_more else "本地排列条件计数"
        return f"{label}: {answer}"

    @staticmethod
    def _surjection_count_hint(problem: str) -> Optional[str]:
        """Count onto maps between two explicitly finite sets."""
        text = (
            str(problem or "")
            .replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        if not re.search(r"满射|surjections?|onto\s+(?:maps?|functions?)", text, re.IGNORECASE):
            return None
        explicit_sets = re.search(
            r"(?:从|from)\s*(?:集合\s*)?\{([^{}]+)\}\s*(?:到|to)\s*"
            r"(?:集合\s*)?\{([^{}]+)\}",
            text,
            re.IGNORECASE,
        )
        if explicit_sets:
            source_items = [item.strip() for item in re.split(r"[,，]", explicit_sets.group(1)) if item.strip()]
            target_items = [item.strip() for item in re.split(r"[,，]", explicit_sets.group(2)) if item.strip()]
            if (
                len(set(source_items)) != len(source_items)
                or len(set(target_items)) != len(target_items)
            ):
                return None
            n, m = len(source_items), len(target_items)
        else:
            size_match = re.search(
                r"从\s*([一二三四五六七八九十\d]+)\s*元素集合\s*到\s*"
                r"([一二三四五六七八九十\d]+)\s*元素集合|"
                r"from\s+(?:a\s+)?([a-z\d-]+)[ -]element\s+set\s+to\s+"
                r"(?:a\s+)?([a-z\d-]+)[ -]element\s+set",
                text,
                re.IGNORECASE,
            )
            if not size_match:
                return None
            first, second = (
                (size_match.group(1), size_match.group(2))
                if size_match.group(1) is not None else (size_match.group(3), size_match.group(4))
            )
            n = SympyTool._small_integer_word(first)
            m = SympyTool._small_integer_word(second)
        if not n or not m or not (1 <= n <= 1000 and 1 <= m <= 50):
            return None
        asks_count = re.search(
            r"(?:满射)(?:的)?(?:个数|数量)|求[^。.!?]{0,40}满射[^。.!?]{0,12}(?:个数|数量)|"
            r"number\s+of\s+(?:such\s+)?(?:surjections?|onto\s+(?:maps?|functions?))|how\s+many",
            text,
            re.IGNORECASE,
        )
        if not asks_count:
            return None
        result = sum(
            (-1) ** omitted * math.comb(m, omitted) * (m - omitted) ** n
            for omitted in range(m + 1)
        )
        english = SympyTool._uses_english_prose(text)
        formula = rf"\sum_{{j=0}}^{{{m}}}(-1)^j\binom{{{m}}}{{j}}({m}-j)^{{{n}}}={result}"
        answer = (
            rf"Inclusion-exclusion gives \({formula}\)."
            if english else rf"由容斥原理，\({formula}\)。"
        )
        needs_more = bool(re.search(
            r"证明|解释|推广|比较|另求|并求|并计算|"
            r"\b(?:prove|justify|explain|generalize|compare|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地满射容斥核验" if needs_more else "本地满射容斥计数"
        return f"{label}: {answer}"

    @staticmethod
    def _small_integer_word(value: str) -> Optional[int]:
        token = str(value or "").strip().lower()
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        if token.isdigit():
            return int(token)
        return words.get(token)

    @staticmethod
    def _planar_euler_face_hint(problem: str) -> Optional[str]:
        """Compute the face count of a connected planar graph and verify Euler."""
        text = str(problem or "")
        chinese = re.search(
            r"连通平面(?:简单)?图.*?(\d+)\s*个顶点.*?(\d+)\s*条边",
            text,
            re.DOTALL,
        )
        english_match = re.search(
            r"connected\s+(?:(?:simple\s+)?planar|planar(?:\s+simple)?)\s+graph"
            r".*?(?:has|with)\s*(\d+)\s+vertices?"
            r".*?(?:and|with)\s*(\d+)\s+edges?",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        match = chinese or english_match
        if not match or not re.search(r"面数|number\s+of\s+faces?|how\s+many\s+faces?", text, re.IGNORECASE):
            return None
        if not re.search(
            r"验证[^。.!?]{0,40}欧拉公式|verify[^.!?]{0,60}euler(?:'s)?\s+formula",
            text,
            re.IGNORECASE,
        ):
            return None
        vertices, edges = map(int, match.groups())
        if vertices < 1 or edges < 0 or vertices > 10**9 or edges > 10**12:
            return None
        faces = edges - vertices + 2
        if faces < 1:
            return None
        english = SympyTool._uses_english_prose(text)
        answer = (
            rf"Euler's formula gives \(F=E-V+2={edges}-{vertices}+2={faces}\), and "
            rf"\({vertices}-{edges}+{faces}=2\)."
            if english else
            rf"由欧拉公式，\(F=E-V+2={edges}-{vertices}+2={faces}\)，且 "
            rf"\({vertices}-{edges}+{faces}=2\)。"
        )
        extra = bool(re.search(
            r"证明|推广|另求|并求|并计算|"
            r"\b(?:prove|generalize|also\s+(?:find|compute|determine))\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地平面图欧拉核验" if extra else "本地平面图欧拉答案"
        return f"{label}: {answer}"

    @staticmethod
    def _paraboloid_curvature_hint(problem: str) -> Optional[str]:
        """Return the exact curvatures of z=x^2+y^2 at the origin."""
        text = re.sub(r"\s+", "", str(problem or ""))
        surface = re.search(
            r"(?:曲面|surface)(?:z=)?x\^\{?2\}?\+y\^\{?2\}?|"
            r"z=x\^\{?2\}?\+y\^\{?2\}?",
            text,
            re.IGNORECASE,
        )
        origin = re.search(r"原点|at(?:the)?origin|\(0,0(?:,0)?\)", text, re.IGNORECASE)
        principal = re.search(r"主曲率|principalcurvatures?", text, re.IGNORECASE)
        gaussian = re.search(r"高斯曲率|gaussiancurvature", text, re.IGNORECASE)
        derivatives = re.search(r"二阶导数|second(?:order)?(?:partial)?derivatives?", text, re.IGNORECASE)
        if not (surface and origin and principal and gaussian and derivatives):
            return None
        english = SympyTool._uses_english_prose(problem)
        answer = (
            r"At the origin, \(f_x=f_y=0\) and \(f_{xx}=2,f_{xy}=0,f_{yy}=2\). "
            r"Thus the shape operator is \(\operatorname{diag}(2,2)\), so "
            r"\(\kappa_1=\kappa_2=2\) and \(K=\kappa_1\kappa_2=4\)."
            if english else
            r"原点处 \(f_x=f_y=0\)，二阶导数为 \(f_{xx}=2,f_{xy}=0,f_{yy}=2\)。"
            r"形算子为 \(\operatorname{diag}(2,2)\)，故 \(\kappa_1=\kappa_2=2\)，"
            r"\(K=\kappa_1\kappa_2=4\)。"
        )
        extra = bool(re.search(
            r"证明|推广|另求|并求|并计算|"
            r"\b(?:prove|generalize|also\s+(?:find|compute|determine))\b",
            str(problem or ""),
            re.IGNORECASE,
        ))
        label = "本地抛物面曲率核验" if extra else "本地抛物面曲率答案"
        return f"{label}: {answer}"

    @staticmethod
    def _ordered_positive_triple_hint(problem: str) -> Optional[str]:
        """Exactly count an explicitly ordered positive-integer triple."""
        normalized = (
            str(problem or "")
            .replace(r"\leq", "≤")
            .replace(r"\le", "≤")
            .replace("<=", "≤")
        )
        total_match = re.search(
            r"([a-z])\s*\+\s*([a-z])\s*\+\s*([a-z])\s*=\s*(\d+)(?!\d)",
            normalized,
            re.IGNORECASE,
        )
        if not total_match or not re.search(r"正整数|positive\s+integers?", normalized, re.IGNORECASE):
            return None
        first, second, third = total_match.group(1, 2, 3)
        if len({first.lower(), second.lower(), third.lower()}) != 3:
            return None
        ordered = re.compile(
            rf"{re.escape(first)}\s*≤\s*{re.escape(second)}\s*≤\s*{re.escape(third)}",
            re.IGNORECASE,
        )
        if not ordered.search(normalized):
            return None
        total = int(total_match.group(4))
        if not 3 <= total <= 10000:
            return None
        counts: list[tuple[int, int]] = []
        for first_value in range(1, total + 1):
            count = sum(
                1
                for second_value in range(first_value, total + 1)
                if total - first_value - second_value >= second_value
            )
            if count:
                counts.append((first_value, count))
        if not counts:
            return None
        cases = "，".join(f"{value}时{count}个" for value, count in counts)
        return f"本地有序三元组计数: 按{first}分类，{first}={cases}，共{sum(count for _, count in counts)}个"

    @staticmethod
    def _fair_dice_conditional_probability_hint(problem: str) -> Optional[str]:
        """Enumerate an exact conditional event for two fair standard dice."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        normalized = re.sub(
            r"\\(?:operatorname|mathrm|text)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
        for token in (r"\left", r"\right", r"\,", r"\;", r"\!", "$", r"\(", r"\)"):
            normalized = normalized.replace(token, "")
        normalized = normalized.replace("−", "-").replace("–", "-")

        if not re.search(r"骰|\b(?:die|dice)\b", normalized, re.IGNORECASE):
            return None
        if re.search(
            r"不公平|非公平|有偏|加权|各面概率不(?:同|等)|"
            r"\b(?:unfair|biased|loaded|weighted|non[- ]uniform)\b|"
            r"\bnot\s+(?:an?\s+)?fair\b",
            normalized,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"公平|均匀(?:的)?(?:骰|六面)|各面(?:出现)?(?:等可能|概率相等)|"
            r"\b(?:fair|unbiased)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None

        explicit_side_counts = [
            int(value)
            for value in re.findall(
                r"(?<!\d)(\d+)\s*(?:面(?:的)?(?:骰子|骰)?|[- ]sided\b)",
                normalized,
                re.IGNORECASE,
            )
        ]
        explicit_side_counts.extend(
            int(value)
            for value in re.findall(r"\b[dD](\d+)\b", normalized)
        )
        side_words = re.findall(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty)"
            r"[- ]sided\b",
            normalized,
            re.IGNORECASE,
        )
        if any(value != 6 for value in explicit_side_counts):
            return None
        if any(word.lower() != "six" for word in side_words):
            return None
        chinese_side_words = re.findall(
            r"([一二三四五六七八九十]+)\s*面(?:的)?(?:骰子|骰)?",
            normalized,
        )
        if any(word != "六" for word in chinese_side_words):
            return None

        chinese_other_roll_count = re.search(
            r"(?:掷|投掷|抛掷|抛|扔)[^，。；;,.!?]{0,20}"
            r"(?:一|三|四|五|六|七|八|九|十|[013-9]|[1-9]\d+)\s*次",
            normalized,
        )
        english_other_roll_count = re.search(
            r"\b(?:roll|toss)(?:ed|ing|es|s)?\b[^.!?]{0,35}"
            r"(?:once|three|four|five|six|seven|eight|nine|ten|[013-9]|[1-9]\d+)\s+times?\b",
            normalized,
            re.IGNORECASE,
        )
        if chinese_other_roll_count or english_other_roll_count:
            return None
        exactly_two = bool(re.search(
            r"(?:连续)?(?:掷|投掷|抛掷|抛|扔)[^，。；;,.!?]{0,28}(?:两|2)\s*次|"
            r"(?:掷|投掷|抛掷|抛|扔)[^，。；;,.!?]{0,12}(?:两|2)\s*(?:枚|个|颗)"
            r"[^，。；;,.!?]{0,16}骰|"
            r"\b(?:roll|toss)(?:ed|ing|es|s)?\b[^.!?]{0,38}\btwice\b|"
            r"\btwo\s+(?:successive\s+)?(?:rolls?|tosses?)\b|"
            r"\b(?:roll|toss)(?:ed|ing|es|s)?\b[^.!?]{0,24}\btwo\s+"
            r"(?:fair\s+)?(?:six[- ]sided\s+)?dice\b",
            normalized,
            re.IGNORECASE,
        ))
        if not exactly_two:
            return None

        chinese_sum_patterns = (
            r"(?:已知|给定)[^，。；;,.!?]{0,35}(?:总点数|点数(?:之)?和|两次(?:结果|点数)(?:之)?和)"
            r"\s*(?:为|是|等于|=)\s*(-?\d+)",
            r"(?:总点数|点数(?:之)?和|两次(?:结果|点数)(?:之)?和)"
            r"\s*(?:为|是|等于|=)\s*(-?\d+)\s*(?:的)?条件下",
            r"在[^，。；;,.!?]{0,28}(?:总点数|点数(?:之)?和|两次(?:结果|点数)(?:之)?和)"
            r"\s*(?:为|是|等于|=)\s*(-?\d+)\s*(?:时|的情况下)",
        )
        english_sum_patterns = (
            r"(?:given(?:\s+that)?|conditioned\s+on|conditional\s+on|provided\s+that|"
            r"under\s+the\s+condition\s+that)[^.!?]{0,55}"
            r"(?:sum|total)(?:\s+of\s+(?:the\s+)?(?:two\s+)?(?:rolls?|dice))?"
            r"\s*(?:is|equals?|=)\s*(-?\d+)",
            r"(?:given(?:\s+that)?|conditioned\s+on|conditional\s+on|provided\s+that)"
            r"[^.!?]{0,55}(?:rolls?|dice)\s+(?:sum|add\s+up)\s+to\s*(-?\d+)",
        )
        sum_value: Optional[int] = None
        for pattern in (*chinese_sum_patterns, *english_sum_patterns):
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                sum_value = int(match.group(1))
                break
        if sum_value is None:
            return None

        chinese_first = re.findall(
            r"(?:第一次|第一\s*(?:枚|个|颗)(?:骰子|骰)?)"
            r"(?:\s*(?:掷(?:骰子)?|投掷|抛掷))?(?:\s*(?:所得|掷出|出现|的))?"
            r"\s*(?:点数|结果)?\s*(?:为|是|等于|=)\s*(-?\d+)",
            normalized,
        )
        english_first = re.findall(
            r"\bfirst\s+(?:roll|die|dice)(?:'s)?(?:\s+(?:result|outcome|value|face))?"
            r"\s*(?:is|equals?|=|shows?|being|lands?\s+on)\s*(-?\d+)",
            normalized,
            re.IGNORECASE,
        )
        queried_values = [int(value) for value in (*chinese_first, *english_first)]
        if len(queried_values) != 1:
            return None
        first_value = queried_values[0]
        if not 1 <= first_value <= 6:
            return None
        if not re.search(r"条件概率|概率|\b(?:conditional\s+)?probability\b|\bchance\b", normalized, re.IGNORECASE):
            return None
        if re.search(
            r"期望|均值|方差|协方差|标准差|分布函数|熵|证明|说明理由|推导|"
            r"乘积|积为|差为|最大值|最小值|至少|至多|奇数|偶数|第二次[^，。；;,.!?]{0,12}(?:为|是|=)|"
            r"\b(?:expectation|expected\s+value|mean|variance|covariance|standard\s+deviation|"
            r"distribution\s+function|entropy|prove|justify|derive|product|difference|maximum|minimum|"
            r"at\s+least|at\s+most|odd|even)\b|"
            r"\bsecond\s+(?:roll|die|dice)[^.!?]{0,16}(?:is|equals?|=)",
            normalized,
            re.IGNORECASE,
        ):
            return None

        outcomes = tuple(
            (first, sum_value - first)
            for first in range(1, 7)
            if 1 <= sum_value - first <= 6
        )
        if not outcomes:
            return None
        favorable = int((first_value, sum_value - first_value) in outcomes)
        probability = Fraction(favorable, len(outcomes))

        def render_fraction(value: Fraction) -> str:
            if value.denominator == 1:
                return str(value.numerator)
            return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"

        wants_sample_space = bool(re.search(
            r"(?:列出|写出|给出|枚举|求)[^，。；;,.!?]{0,18}条件样本空间|"
            r"条件样本空间[^，。；;,.!?]{0,12}(?:是什么|为多少)|"
            r"\b(?:list|write|give|state|enumerate|find)\b[^.!?]{0,45}"
            r"\bconditional\s+sample\s+space\b|"
            r"\bwhat\s+is\s+the\s+conditional\s+sample\s+space\b",
            normalized,
            re.IGNORECASE,
        ))
        probability_text = render_fraction(probability)
        if wants_sample_space:
            sample_space = ",".join(f"({first},{second})" for first, second in outcomes)
            result = (
                rf"\Omega_{{S={sum_value}}}=\{{{sample_space}\}},\quad "
                rf"P(D_1={first_value}\mid S={sum_value})={probability_text}"
            )
        else:
            result = rf"P(D_1={first_value}\mid D_1+D_2={sum_value})={probability_text}"
        return f"本地公平六面骰条件概率: {result}"

    @staticmethod
    def _bernoulli_centered_second_moment_hint(problem: str) -> Optional[str]:
        """Evaluate the centered second moment of one Bernoulli variable."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        normalized = re.sub(
            r"\\(?:operatorname|mathrm|text)\s*\{([^{}]*)\}",
            r"\1",
            text,
        )
        normalized = re.sub(r"\\mathbb\s*\{?\s*E\s*\}?", "E", normalized)
        normalized = re.sub(r"\\(?:mathbf|mathsf|mathcal)\s*\{?\s*E\s*\}?", "E", normalized)
        normalized = normalized.replace(r"\sim", "~").replace("∼", "~")
        normalized = normalized.replace("−", "-").replace("–", "-")
        normalized = re.sub(r"\^\s*\{\s*2\s*\}", "^2", normalized)
        for token in (
            r"\left", r"\right", r"\bigl", r"\bigr", r"\big", r"\,", r"\;", r"\!", "$", r"\(", r"\)",
        ):
            normalized = normalized.replace(token, "")

        if len(re.findall(r"Bernoulli|伯努利", normalized, re.IGNORECASE)) != 1:
            return None
        if re.search(
            r"微分方程|导数|似然|估计|样本|构造|独立性|"
            r"\b(?:differential\s+equation|derivative|likelihood|estimat|sample|construct|independen)\w*\b|"
            r"[A-Za-z]\s*'",
            normalized,
            re.IGNORECASE,
        ):
            return None

        distribution_patterns = (
            r"(?P<variable>[A-Za-z])\s*~\s*(?:Bernoulli|伯努利)\s*\(\s*(?P<parameter>[a-z])\s*\)",
            r"(?P<variable>[A-Za-z])\s*服从\s*参数\s*(?:为|=)?\s*(?P<parameter>[a-z])\s*的?\s*"
            r"(?:Bernoulli|伯努利)(?:分布)?",
            r"(?P<variable>[A-Za-z])\s*服从\s*(?:Bernoulli|伯努利)\s*\(\s*(?P<parameter>[a-z])\s*\)\s*(?:分布)?",
            r"(?:let\s+)?(?P<variable>[A-Za-z])\s+(?:be|is|follows?|obeys?|has)\s+(?:a\s+)?"
            r"Bernoulli(?:\s+(?:random\s+variable|distribution))?\s*\(\s*(?P<parameter>[a-z])\s*\)",
            r"(?:let\s+)?(?P<variable>[A-Za-z])\s+(?:be|is|follows?|obeys?|has)\s+(?:a\s+)?"
            r"Bernoulli(?:\s+(?:random\s+variable|distribution))?\s+with\s+(?:the\s+)?parameter\s+"
            r"(?P<parameter>[a-z])\b",
        )
        distribution = next(
            (
                candidate
                for pattern in distribution_patterns
                if (candidate := re.search(pattern, normalized, re.IGNORECASE))
            ),
            None,
        )
        if distribution is None:
            return None
        variable = distribution.group("variable")
        parameter = distribution.group("parameter")
        if not variable.isupper() or not parameter.islower():
            return None

        moment_pattern = (
            rf"(?:E|expectation\s+of)\s*[\[(]\s*\(\s*{re.escape(variable)}\s*-\s*"
            rf"{re.escape(parameter)}\s*\)\s*(?:\^\s*2|\*\*\s*2)\s*[\])]"
        )
        moments = re.findall(moment_pattern, normalized, re.IGNORECASE)
        if len(moments) != 1:
            return None
        expectation_count = len(re.findall(
            r"(?<![A-Za-z])E\s*[\[(]|\bexpectation\s+of\b",
            normalized,
            re.IGNORECASE,
        ))
        if expectation_count != 1:
            return None
        if not re.search(
            r"求|计算|确定|识别|认出|指出|"
            r"\b(?:find|compute|calculate|evaluate|determine|identify|recognize|what\s+is)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"条件期望|条件方差|协方差|标准差|偏度|峰度|特征函数|矩母函数|分布函数|熵|"
            r"最大(?:化|值)|最小(?:化|值)|证明|推导|说明理由|解释|"
            r"\b(?:conditional\s+(?:expectation|variance)|covariance|standard\s+deviation|skewness|"
            r"kurtosis|characteristic\s+function|moment[- ]generating\s+function|distribution\s+function|"
            r"entropy|maximi[sz]e|minimi[sz]e|maximum|minimum|prove|derive|justify|explain|demonstrate)\b|"
            r"\b(?:binomial|poisson|normal|geometric)\s+(?:random\s+variable|distribution)\b|"
            r"二项分布|泊松分布|正态分布|几何分布|"
            r"\bP\s*\(",
            normalized,
            re.IGNORECASE,
        ):
            return None

        result = (
            rf"E[({variable}-{parameter})^2]="
            rf"\operatorname{{Var}}({variable})={parameter}(1-{parameter})"
        )
        return f"本地Bernoulli中心二阶矩: {result}"

    @staticmethod
    def _fair_coin_geometric_tail_hint(problem: str) -> Optional[str]:
        """Compute a strict tail for tosses until the first head of one fair coin."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\,", "")
            .replace(r"\;", "")
            .replace(r"\gt", ">")
            .replace("−", "-")
        )
        if not re.search(r"公平(?:的)?硬币|均匀(?:的)?硬币|\bfair\s+coin\b", normalized, re.IGNORECASE):
            return None
        if re.search(
            r"不公平|有偏|加权|两枚|多个硬币|"
            r"\b(?:unfair|biased|loaded|weighted|two|multiple)\s+coins?\b",
            normalized,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"(?:持续|反复|不断)?(?:抛掷|投掷|掷|抛)[^。！？!?]{0,60}"
            r"直到[^。！？!?]{0,24}(?:首次|第一次)[^。！？!?]{0,12}正面|"
            r"\b(?:toss|flip)(?:ed|ing|s)?\b[^.!?]{0,70}\buntil\b"
            r"[^.!?]{0,24}\b(?:the\s+)?first\s+head\b",
            normalized,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"概率|\bprobability\b|P\s*\(", normalized, re.IGNORECASE):
            return None
        if re.search(
            r"大于等于|不少于|至少|至多|不超过|恰好|正好|无记忆|条件概率|"
            r"\b(?:at\s+least|at\s+most|no\s+more\s+than|exactly|memoryless|"
            r"conditional\s+probability)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None

        thresholds: list[int] = []
        patterns = (
            r"(?:抛掷|投掷|掷|抛)(?:的)?次数\s*(?:T\s*)?(?:大于|超过|>)\s*(\d+)",
            r"P\s*\(\s*T\s*>\s*(\d+)\s*\)",
            r"(?:number\s+of\s+(?:tosses|flips)|T)\s*(?:is\s+)?"
            r"(?:greater\s+than|exceeds?|>)\s*(\d+)",
        )
        for pattern in patterns:
            thresholds.extend(
                int(value)
                for value in re.findall(pattern, normalized, re.IGNORECASE)
            )
        unique_thresholds = set(thresholds)
        if len(unique_thresholds) != 1:
            return None
        threshold = unique_thresholds.pop()
        if not 0 <= threshold <= 1000:
            return None
        if re.search(
            r"期望|均值|方差|标准差|生成函数|分布函数|熵|证明|推导|"
            r"\b(?:expectation|expected\s+value|mean|variance|standard\s+deviation|"
            r"generating\s+function|distribution\s+function|entropy|prove|derive)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None

        probability = Fraction(1, 2 ** threshold)
        probability_text = (
            str(probability.numerator)
            if probability.denominator == 1
            else rf"\frac{{{probability.numerator}}}{{{probability.denominator}}}"
        )
        if SympyTool._uses_english_prose(problem):
            support = (
                rf"Let \(T\) be the number of tosses through the first head. Then "
                rf"\(T\sim\operatorname{{Geom}}(1/2)\) on \(1,2,\ldots\), so "
                rf"\(P(T>{threshold})=(1/2)^{{{threshold}}}={probability_text}\)."
            )
        else:
            support = (
                rf"令 \(T\) 为首次出现正面所需的抛掷次数，则 "
                rf"\(T\sim\operatorname{{Geom}}(1/2)\)，其取值从1开始，故 "
                rf"\(P(T>{threshold})=(1/2)^{{{threshold}}}={probability_text}\)。"
            )
        return f"本地公平硬币几何尾概率: {support}"

    @staticmethod
    def _poisson_process_increment_hint(problem: str) -> Optional[str]:
        """Return a conditional law certified by homogeneous independent increments."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\,", "")
            .replace(r"\;", "")
            .replace(r"\lambda", "λ")
            .replace("−", "-")
        )
        if not re.search(r"泊松过程|\bPoisson\s+process\b", normalized, re.IGNORECASE):
            return None
        if re.search(
            r"非齐次|非均匀|复合泊松|Cox过程|"
            r"\b(?:nonhomogeneous|non-homogeneous|inhomogeneous|compound|Cox)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None

        rate_match = re.search(
            r"(?:强度|速率|率)\s*(?:为|是|等于|=)?\s*(λ|lambda|\d+(?:\.\d+|/\d+)?)|"
            r"\b(?:intensity|rate)\s*(?:is|equals?|=|of)?\s*(λ|lambda|\d+(?:\.\d+|/\d+)?)",
            normalized,
            re.IGNORECASE,
        )
        if not rate_match:
            return None
        rate_token = next(group for group in rate_match.groups() if group is not None)
        symbolic_rate = rate_token.lower() in {"λ", "lambda"}
        if symbolic_rate:
            rate_value: Optional[Fraction] = None
        else:
            try:
                rate_value = Fraction(rate_token)
            except (ValueError, ZeroDivisionError):
                return None
            if rate_value <= 0:
                return None

        number = r"(?:\d+(?:\.\d+)?|\d+/\d+)"
        increment_matches = re.findall(
            rf"N\s*\(\s*({number})\s*\)\s*-\s*N\s*\(\s*({number})\s*\)",
            normalized,
            re.IGNORECASE,
        )
        if len(increment_matches) != 1:
            return None
        upper_raw, lower_raw = increment_matches[0]
        try:
            upper = Fraction(upper_raw)
            lower = Fraction(lower_raw)
        except (ValueError, ZeroDivisionError):
            return None
        duration = upper - lower
        if duration <= 0:
            return None

        conditioning = re.findall(
            rf"N\s*\(\s*({number})\s*\)\s*=\s*(\d+)",
            normalized,
            re.IGNORECASE,
        )
        if len(conditioning) != 1:
            return None
        condition_time_raw, condition_count_raw = conditioning[0]
        try:
            condition_time = Fraction(condition_time_raw)
        except (ValueError, ZeroDivisionError):
            return None
        if condition_time != lower:
            return None
        condition_count = int(condition_count_raw)
        if condition_count < 0:
            return None
        if not re.search(
            r"条件分布|给定[^。！？!?]{0,120}(?:时|条件下)[^。！？!?]{0,120}分布|"
            r"\bconditional\s+distribution\b|\bdistribution\b[^.!?]{0,120}\bgiven\b|"
            r"\bgiven\b[^.!?]{0,160}\bdistribution\b",
            normalized,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"期望|均值|方差|协方差|概率|到达时间|等待时间|矩母函数|证明|推导|"
            r"\b(?:expectation|expected\s+value|mean|variance|covariance|probability|"
            r"arrival\s+time|waiting\s+time|moment[- ]generating\s+function|prove|derive)\b",
            normalized,
            re.IGNORECASE,
        ):
            return None

        if symbolic_rate:
            if duration == 1:
                parameter = r"\lambda"
            elif duration.denominator == 1:
                parameter = rf"{duration.numerator}\lambda"
            else:
                parameter = rf"\frac{{{duration.numerator}}}{{{duration.denominator}}}\lambda"
        else:
            parameter_value = duration * rate_value
            parameter = (
                str(parameter_value.numerator)
                if parameter_value.denominator == 1
                else rf"\frac{{{parameter_value.numerator}}}{{{parameter_value.denominator}}}"
            )
        increment = rf"N({upper_raw})-N({lower_raw})"
        condition = rf"N({condition_time_raw})={condition_count}"
        result = (
            rf"{increment}\mid {condition}\sim"
            rf"\operatorname{{Poisson}}({parameter})"
        )
        if SympyTool._uses_english_prose(problem):
            support = (
                rf"By independent increments, \({increment}\) is independent of "
                rf"\({condition}\) and has parameter rate times interval length. Hence "
                rf"\({result}\)."
            )
        else:
            support = (
                rf"由泊松过程的独立增量性，\({increment}\) 与过去事件 \({condition}\) 独立，"
                rf"其参数为强度乘区间长度，故 \({result}\)。"
            )
        return f"本地泊松过程独立增量: {support}"

    @staticmethod
    def _finite_discrete_moments_hint(problem: str) -> Optional[str]:
        """Compute exact mean and variance from a complete finite probability table."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        chinese = re.search(
            r"随机变量\s*([A-Za-z])\s*取值\s*([^，。；;]+?)\s*(?:且|，)?\s*"
            r"概率分别为\s*([^，。；;]+?)(?=\s*[，,]?\s*(?:求|计算))",
            text,
            re.IGNORECASE,
        )
        english = re.search(
            r"random\s+variable\s+([A-Za-z])\s+takes?\s+(?:the\s+)?values?\s*"
            r"([^.;]+?)\s+with\s+(?:respective\s+)?probabilities\s*"
            r"([^.;]+?)(?=\s*[.;,]?\s*(?:find|compute|calculate))",
            text,
            re.IGNORECASE,
        )
        match = chinese or english
        if not match:
            return None
        variable = match.group(1)
        if not (
            re.search(rf"E\s*\[\s*{re.escape(variable)}\s*\]", text, re.IGNORECASE)
            and re.search(rf"Var\s*\(\s*{re.escape(variable)}\s*\)", text, re.IGNORECASE)
        ):
            return None
        if re.search(
            r"协方差|分布函数|特征函数|矩母函数|偏度|峰度|条件期望|"
            r"\b(?:covariance|distribution\s+function|characteristic\s+function|"
            r"moment[- ]generating|skewness|kurtosis|conditional\s+expectation)\b",
            text,
            re.IGNORECASE,
        ):
            return None

        def parse_list(raw: str) -> Optional[list[Fraction]]:
            tokens = [token.strip() for token in re.split(r"[,，、]", raw) if token.strip()]
            try:
                return [Fraction(token) for token in tokens]
            except (ValueError, ZeroDivisionError):
                return None

        values = parse_list(match.group(2))
        probabilities = parse_list(match.group(3))
        if (
            not values
            or not probabilities
            or len(values) != len(probabilities)
            or not 2 <= len(values) <= 20
            or len(set(values)) != len(values)
            or any(probability < 0 for probability in probabilities)
            or sum(probabilities, Fraction()) != 1
        ):
            return None
        expectation = sum(
            (value * probability for value, probability in zip(values, probabilities)),
            Fraction(),
        )
        second_moment = sum(
            (value * value * probability for value, probability in zip(values, probabilities)),
            Fraction(),
        )
        variance = second_moment - expectation * expectation

        def render(value: Fraction) -> str:
            return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

        return (
            f"本地有限离散分布矩: E[{variable}]={render(expectation)}，"
            f"Var({variable})={render(variance)}"
        )

    @staticmethod
    def _two_sided_z_rejection_hint(problem: str) -> Optional[str]:
        """Return the standard two-sided Z rejection region at common levels."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        if not (
            re.search(r"双侧\s*Z\s*检验|two[- ]sided\s+Z[- ]?test", text, re.IGNORECASE)
            and re.search(r"拒绝域|rejection\s+region", text, re.IGNORECASE)
            and re.search(r"临界值|critical\s+values?", text, re.IGNORECASE)
        ):
            return None
        if re.search(r"单侧|one[- ]sided|t\s*检验|t[- ]?test", text, re.IGNORECASE):
            return None
        level = re.search(
            r"(?:显著性水平|significance\s+level)\s*(?:为|=|of|is)?\s*"
            r"(0\.10|0\.05|0\.02|0\.01)(?!\d)",
            text,
            re.IGNORECASE,
        )
        if not level:
            return None
        critical_values = {
            "0.10": "1.645",
            "0.05": "1.96",
            "0.02": "2.326",
            "0.01": "2.576",
        }
        critical = critical_values[level.group(1)]
        return f"本地双侧Z检验拒绝域: |Z|>{critical}"

    @staticmethod
    def _simple_random_walk_hint(problem: str) -> Optional[str]:
        """Return exact first and second moments for a named simple symmetric walk."""
        text = str(problem or "")
        if not re.search(r"简单对称随机游走|simple\s+symmetric\s+random\s+walk", text, re.IGNORECASE):
            return None
        if not re.search(r"从\s*0\s*出发|S_?\{?0\}?\s*=\s*0|starts?\s+(?:at|from)\s+0", text, re.IGNORECASE):
            return None
        expectation = re.search(r"E\s*\[\s*S_?\{?(\d+)\}?\s*\]", text, re.IGNORECASE)
        variance = re.search(r"Var\s*\(\s*S_?\{?(\d+)\}?\s*\)", text, re.IGNORECASE)
        if not expectation or not variance or expectation.group(1) != variance.group(1):
            return None
        step = int(expectation.group(1))
        return (
            f"本地随机游走矩: E[S_{step}]=0，Var(S_{step})={step}；由独立增量，"
            f"E[S_{step}]={step}E[X_1]，Var(S_{step})={step}Var(X_1)"
        )

    @staticmethod
    def _complete_graph_cover_time_hint(problem: str) -> Optional[str]:
        """Exact coupon-collector expectation for a complete-graph walk."""
        text = str(problem or "")
        numeric_graph = re.search(r"完全图\s*\$?K_?\{?(\d+)\}?\$?", text)
        if numeric_graph:
            size = int(numeric_graph.group(1))
            words = {
                "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            }
            all_vertices = re.search(r"首次访问全部([一二两三四五六七八九十\d]+)个顶点", text)
            alternatives = re.search(r"每一步等概率走向另外([一二两三四五六七八九十\d]+)个顶点", text)

            def parse_count(value: str) -> Optional[int]:
                return int(value) if value.isdigit() else words.get(value)

            if (
                2 <= size <= 10**6
                and all_vertices and parse_count(all_vertices.group(1)) == size
                and alternatives and parse_count(alternatives.group(1)) == size - 1
                and re.search(r"简单随机游走", text)
                and re.search(r"期望", text)
                and not re.search(r"懒惰|加权|自环", text)
            ):
                expectation = (size - 1) * sum(
                    (Fraction(1, index) for index in range(1, size)),
                    Fraction(),
                )
                answer = (
                    str(expectation.numerator)
                    if expectation.denominator == 1
                    else f"{expectation.numerator}/{expectation.denominator}"
                )
                return f"本地完全图覆盖时间: {answer}"
        if not re.search(
            r"(?:\$\s*)?N(?:\s*\$|\\\))?\s*个顶点的完全图|"
            r"complete\s+graph\s+(?:on|with)\s+(?:\$\s*)?N(?:\s*\$)?\s+vertices",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"首次遍访所有顶点|cover\s+time|visited?\s+all\s+vertices", text, re.IGNORECASE):
            return None
        if not re.search(r"简单随机游动|simple\s+random\s+walk", text, re.IGNORECASE):
            return None
        if re.search(r"lazy|weighted|加权|懒惰|自环|self[- ]?loop", text, re.IGNORECASE):
            return None
        if not re.search(r"(?:求|find|compute)\s*E\s*T|期望", text, re.IGNORECASE):
            return None
        return r"本地完全图覆盖时间: (N-1)\sum_{j=1}^{N-1}\frac{1}{j}"

    @staticmethod
    def _two_venue_capacity_hint(problem: str) -> Optional[str]:
        """Invert an exact symmetric binomial tail for two equiprobable venues."""
        text = str(problem or "")
        population = re.search(r"(\d+)\s*名(?:市民|观众|顾客)", text)
        threshold = re.search(r"概率不超过\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not population or not threshold:
            return None
        if not re.search(r"两个(?:剧院|场馆).*(?:独立)?等可能", text, re.DOTALL):
            return None
        if not re.search(
            r"(?:每个|各).*?有\s*(?:\\\(\s*)?[xX](?:\s*\\\))?\s*个座位|"
            r"(?:\\\(\s*)?[xX](?:\s*\\\))?\s*的最小值",
            text,
            re.DOTALL,
        ):
            return None
        count = int(population.group(1))
        if count < 1 or count > 20000:
            return None
        probability = Fraction(threshold.group(1))
        total_outcomes = 1 << count
        tail = 0
        largest_tail_index = -1
        for index in range(0, count // 2):
            proposed = tail + math.comb(count, index)
            if 2 * proposed * probability.denominator > probability.numerator * total_outcomes:
                break
            tail = proposed
            largest_tail_index = index
        capacity = count - largest_tail_index - 1
        return f"本地二项分布容量: {capacity}"

    @staticmethod
    def _circle_laplacian_hint(problem: str) -> Optional[str]:
        """Separate the ambient Laplacian from the circle Laplace--Beltrami operator."""
        text = str(problem or "")
        compact = re.sub(r"\s+", "", text).lower()
        if re.search(
            r"bi\s*[-–—]?\s*laplacian|biharmonic|双拉普拉斯|双调和|"
            r"weighted\s+laplacian|加权拉普拉斯|p\s*[-–—]?\s*laplacian|p\s*[-–—]?\s*拉普拉斯",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"拉普拉斯|laplacian|laplace\s*[-–—]?\s*beltrami", text, re.IGNORECASE):
            return None
        expression = bool(re.search(
            r"f\(x,y\)=x(?:\^\{?2\}?|²)\+y(?:\^\{?2\}?|²)", compact,
        ))
        circle = bool(
            re.search(r"圆周|circle|s\^?1", text, re.IGNORECASE)
            and re.search(
                r"x(?:\^\{?2\}?|²)\+y(?:\^\{?2\}?|²)="
                r"(?:1|[1-9]\d*(?:\^\{?2\}?|²)?|[a-z](?:\^\{?2\}?|²))",
                compact,
                re.IGNORECASE,
            )
        )
        if not (expression and circle):
            return None

        explicit_intrinsic = bool(re.search(
            r"laplace\s*[-–—]?\s*beltrami|laplacebeltrami|"
            r"拉普拉斯\s*[-–—]?\s*贝尔特拉米|拉普拉斯贝尔特拉米|"
            r"内蕴拉普拉斯|intrinsic\s+laplacian|"
            r"(?:限制函数|限制到|restriction\s+of|restricted\s+to|f\s*\|)"
            r"[^。.!?]{0,40}(?:拉普拉斯|laplacian)",
            text,
            re.IGNORECASE,
        ))
        explicit_ambient = bool(re.search(
            r"环境拉普拉斯|欧氏拉普拉斯|ambient\s+laplacian|euclidean\s+laplacian|"
            r"\\Delta_?\{?\\mathbb\s*\{?R\}?\^?2\}?",
            text,
            re.IGNORECASE,
        ))
        explicit_ambiguity = bool(
            re.search(
                r"(?:环境|欧氏|ambient|euclidean).{0,30}(?:还是|或|或者|or|versus|vs\.?)"
                r".{0,30}(?:内蕴|贝尔特拉米|intrinsic|beltrami)|"
                r"(?:内蕴|贝尔特拉米|intrinsic|beltrami).{0,30}"
                r"(?:还是|或|或者|or|versus|vs\.?).{0,30}(?:环境|欧氏|ambient|euclidean)|"
                r"(?:未说明|不明确|有歧义|unspecified|ambiguous).{0,20}"
                r"(?:拉普拉斯|laplacian)",
                text,
                re.IGNORECASE,
            )
            or (explicit_intrinsic and explicit_ambient)
        )
        needs_support = bool(re.search(
            r"证明|推导|解释|说明理由|"
            r"\b(?:prove|derive|justify|explain|show\s+why)\b",
            text,
            re.IGNORECASE,
        ))

        if explicit_ambiguity:
            return (
                "本地圆周拉普拉斯歧义核验: "
                r"若指环境算子，则 \(\Delta_{\mathbb R^2}f=f_{xx}+f_{yy}=2+2=4\)；"
                r"若指限制函数的 Laplace--Beltrami 算子，则 \(f|_{S^1}\) 为常数，故值为 \(0\)"
            )
        if explicit_intrinsic:
            label = "本地圆周Laplace-Beltrami核验" if needs_support else "本地圆周Laplace-Beltrami"
            return f"{label}: 0"
        if explicit_ambient:
            label = "本地圆周拉普拉斯核验" if needs_support else "本地圆周拉普拉斯"
            return f"{label}: 4"
        return (
            "本地圆周拉普拉斯歧义核验: "
            r"若指环境算子，则 \(\Delta_{\mathbb R^2}f=f_{xx}+f_{yy}=2+2=4\)；"
            r"若指限制函数的 Laplace--Beltrami 算子，则 \(f|_{S^1}\) 为常数，故值为 \(0\)"
        )

    @staticmethod
    def _central_difference_hint(problem: str) -> Optional[str]:
        """Compute the named centered first-difference formula for sin."""
        text = str(problem or "")
        if not re.search(r"中心差分|central\s+difference", text, re.IGNORECASE):
            return None
        if not re.search(r"f\s*\(\s*x\s*\)\s*=\s*(?:\\sin|sin)\s*\(\s*x\s*\)", text, re.IGNORECASE):
            return None
        point = re.search(r"x\s*=\s*(?:\\pi|π)\s*/\s*(\d+)", text, re.IGNORECASE)
        step = re.search(r"h\s*=\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not point or not step:
            return None
        denominator = int(point.group(1))
        h_value = float(step.group(1))
        if denominator <= 0 or not (0 < h_value <= 1):
            return None
        x_value = math.pi / denominator
        approximation = (
            math.sin(x_value + h_value) - math.sin(x_value - h_value)
        ) / (2 * h_value)
        point_text = rf"\pi/{denominator}"
        return (
            "本地中心差分: 中心差分公式 "
            rf"\frac{{\sin({point_text}+{h_value:g})-\sin({point_text}-{h_value:g})}}"
            rf"{{2\times {h_value:g}}}\approx {approximation:.4f}"
        )

    @staticmethod
    def _rational_f2_constraint_hint(problem: str) -> Optional[str]:
        """Propagate the exact parity invariant generated by three rational involutions."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        function = re.search(
            r"([A-Za-z])\s*:\s*\\mathbb\s*\{Q\}\s*\\rightarrow\s*"
            r"\\mathbb\s*\{F\}\s*_?\s*\{?2\}?",
            text,
        )
        if not function:
            return None
        name = re.escape(function.group(1))
        if not re.search(
            rf"{name}\s*\(\s*r\s*\)\s*\+\s*{name}\s*\(\s*r['’]?\s*\)\s*=\s*1",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"distinct\s+rational\s+numbers", text, re.IGNORECASE):
            return None
        required_relations = (
            r"r\s*\+\s*r['’]?\s*=\s*0",
            r"r\s*\+\s*r['’]?\s*=\s*1",
            r"r\s*r['’]?\s*=\s*1",
        )
        if any(not re.search(pattern, text, re.IGNORECASE) for pattern in required_relations):
            return None
        seed = re.search(
            rf"{name}\s*\(\s*([+-]?\d+)\s*/\s*(\d+)\s*\)\s*=\s*([01])",
            text,
        )
        request = re.search(r"\b(?:evaluate|compute|find)\b", text, re.IGNORECASE)
        if not seed or not request:
            return None
        denominator = int(seed.group(2))
        if denominator == 0:
            return None
        seed_value = Fraction(int(seed.group(1)), denominator)
        seed_bit = int(seed.group(3))
        query = text[request.end():]
        terms = re.findall(
            rf"{name}\s*\(\s*([+-]?\d+)(?:\s*/\s*(\d+))?\s*\)",
            query,
        )
        if not terms or len(terms) > 100:
            return None
        rationals = []
        for numerator_text, denominator_text in terms:
            term_denominator = int(denominator_text or "1")
            if term_denominator == 0:
                return None
            rationals.append(Fraction(int(numerator_text), term_denominator))

        seed_colour = SympyTool._rational_involution_colour(seed_value)
        result = 0
        for value in rationals:
            result ^= seed_bit ^ seed_colour ^ SympyTool._rational_involution_colour(value)
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+why)\b",
            query,
            re.IGNORECASE,
        ))
        label = "本地有理数约束传播核验" if needs_support else "本地有理数约束传播答案"
        return f"{label}: {result}"

    @staticmethod
    def _rational_involution_colour(value: Fraction) -> int:
        """Two-colour Q for edges r~-r, r~1-r and r~1/r."""
        if value == 0:
            return 1
        flips = int(value < 0)
        numerator = abs(value.numerator)
        denominator = value.denominator
        while numerator != denominator:
            if numerator > denominator:
                # Repeated integer translations use two involutions and keep colour.
                numerator = (numerator - 1) % denominator + 1
            else:
                numerator, denominator = denominator, numerator
                flips ^= 1
        return flips

    @staticmethod
    def _digit_sum_window_hint(problem: str) -> Optional[str]:
        """Find the first window whose digit sums all avoid a requested divisor."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        if not re.search(
            r"S\s*\(\s*n\s*\)\s*\$?\s+(?:be|is).*?sum\s+of\s+the\s+digits.*?"
            r"decimal\s+representation.*?positive\s+integer",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"smallest\s+positive\s+integer\s+\$?n\$?", text, re.IGNORECASE):
            return None
        offset = re.search(
            r"S\s*\(\s*n\s*\)\s*S\s*\(\s*n\s*\+\s*1\s*\)\s*"
            r"(?:\\cdots|\\dots|\.\.\.).*?S\s*\(\s*n\s*\+\s*(\d+)\s*\)",
            text,
            re.IGNORECASE,
        )
        modulus = re.search(r"not\s+a\s+multiple\s+of\s+\$?(\d+)\$?", text, re.IGNORECASE)
        if not offset or not modulus:
            return None
        window_offset = int(offset.group(1))
        divisor = int(modulus.group(1))
        if not 1 <= window_offset <= 200 or not 2 <= divisor <= 100:
            return None
        window_size = window_offset + 1
        search_limit = 2_000_000

        def divisible_digit_sum(number: int) -> int:
            total = 0
            value = number
            while value:
                total += value % 10
                value //= 10
            return int(total % divisor == 0)

        bad = sum(divisible_digit_sum(number) for number in range(1, window_size + 1))
        answer = None
        for start in range(1, search_limit + 1):
            if bad == 0:
                answer = start
                break
            bad -= divisible_digit_sum(start)
            bad += divisible_digit_sum(start + window_size)
        if answer is None:
            return None
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+that)\b",
            text,
            re.IGNORECASE,
        ))
        label = "本地数位和窗口最小性核验" if needs_support else "本地数位和窗口答案"
        return f"{label}: {answer}"

    @staticmethod
    def _number_writing_game_hint(problem: str) -> Optional[str]:
        """Exhaust a fully specified ``n+1``/``2n`` normal-play game.

        The matcher intentionally verifies every rule needed to identify a game
        position.  A computed value is only labelled as a complete answer when
        the prompt asks for that value alone; requested proofs or strategies
        still receive the value as local evidence for the model solver.
        """
        text = str(problem or "")
        compact = re.sub(r"\s+", " ", text).strip()
        if not re.search(
            r"two\s+players\s+[\$({]*A[\$)}]*\s+and\s+[\$({]*B[\$)}]*.*?"
            r"taking\s+turns\s+writing\s+numbers?",
            compact,
            re.IGNORECASE,
        ):
            return None
        required_rules = (
            r"set\s*\$?\\?\{\s*1\s*,\s*(?:\\dots|\\ldots|\.\.\.)\s*,\s*N\s*\\?\}\$?",
            r"\$?N\$?\s+is\s+a\s+positive\s+integer",
            r"player\s*\$?A\$?\s+starts?\s+(?:the\s+game\s+)?by\s+writing\s+(?:the\s+number\s+)?\$?1\$?",
            r"if\s+a\s+player\s+writes?\s+(?:the\s+number\s+)?\$?n\$?.*?"
            r"other\s+player\s+can\s+write\s+either\s+\$?n\s*\+\s*1\$?\s+or\s+\$?2\s*n\$?",
            r"provided\s+(?:that\s+)?the\s+number\s+does\s+not\s+exceed\s+\$?N(?!\s*[+\-*/])\$?",
            r"player\s+who\s+writes?\s+(?:the\s+number\s+)?\$?N(?!\s*[+\-*/])\$?\s+wins?",
            r"\$?N\$?\s+is\s+of\s+type\s+\$?A\$?.*?player\s+\$?A\$?\s+has\s+a\s+winning\s+strategy",
            r"(?:\$?N\$?\s+is\s+|and\s+)of\s+type\s+\$?B\$?.*?"
            r"player\s+\$?B\$?\s+has\s+a\s+winning\s+strategy",
        )
        if any(not re.search(rule, compact, re.IGNORECASE) for rule in required_rules):
            return None
        if re.search(
            r"player\s+who\s+(?:cannot|can\s+not)\s+move|no\s+legal\s+move|"
            r"may\s+also\s+write|instead\s+of",
            compact,
            re.IGNORECASE,
        ):
            return None
        request = re.search(
            r"find\s+the\s+least\s+\$?N\s*>\s*(\d+)\$?\s+such\s+that\s+"
            r"it\s+is\s+a\s+type\s+\$?([AB])\$?\s+number",
            compact,
            re.IGNORECASE,
        )
        if not request:
            return None
        threshold = int(request.group(1))
        requested_type = request.group(2).upper()
        # Keep exhaustive search predictably cheap.  Refusing a larger instance
        # is safer than presenting an unchecked heuristic as certified output.
        if not 1 <= threshold <= 1000:
            return None
        search_limit = max(2048, 2 * threshold + 1024)
        candidate = next(
            (
                limit
                for limit in range(threshold + 1, search_limit + 1)
                if SympyTool._number_game_type(limit) == requested_type
            ),
            None,
        )
        if candidate is None:
            return None
        request_tail = compact[request.end():]
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+that|give|describe)\b.*?"
            r"\b(?:strategy|reason|proof|derivation)\b",
            compact,
            re.IGNORECASE,
        ) or re.search(r"\b(?:also|in\s+addition|and\s+then)\b", request_tail, re.IGNORECASE))
        label = "本地取数博弈状态核验" if needs_support else "本地取数博弈答案"
        return f"{label}: {candidate}"

    @staticmethod
    def _number_game_type(limit: int) -> str:
        """Return the winning player's type after A has written the initial 1."""
        winning = bytearray(limit + 1)
        for current in range(limit - 1, 0, -1):
            plus_one_loses = not winning[current + 1]
            doubled_loses = 2 * current <= limit and not winning[2 * current]
            winning[current] = plus_one_loses or doubled_loses
        # At position 1 it is B's turn.  A losing position for that player means
        # the initial writer A has the winning strategy.
        return "B" if winning[1] else "A"

    @staticmethod
    def _path_independent_set_partition_hint(problem: str) -> Optional[str]:
        """Compute the hard-core partition polynomial of an explicitly named path."""
        text = str(problem or "")
        compact = re.sub(r"\s+", " ", text).strip()
        if not re.search(
            r"(?:let\s+)?\$?P_?\{?n\}?\$?\s+(?:be|is)\s+a\s+path\s+on\s+"
            r"\$?n\$?\s+vertices",
            compact,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"\$?\\lambda\$?\s+(?:be|is)\s+a\s+positive\s+real\s+number",
            compact,
            re.IGNORECASE,
        ):
            return None
        if not (
            re.search(r"define\s+\$?Z_?\{?P_?\{?n\}?\}?\s*\(\s*\\lambda\s*\)", compact, re.IGNORECASE)
            and re.search(r"=\s*\\sum_?\{?\s*I\s*\\in", compact, re.IGNORECASE)
            and re.search(r"\\lambda\s*\^\s*\{?\s*\|\s*I\s*\|\s*\}?", compact, re.IGNORECASE)
            and re.search(r"independent\s+sets?\s+of\s+\$?P_?\{?n\}?", compact, re.IGNORECASE)
        ):
            return None
        request = re.search(
            r"(?:compute|find|determine)\s+(?:the\s+value\s+of\s+)?\$?"
            r"[zZ]_?\{?(?:P_?\{?)?(\d+)\}?\}?"
            r"(?:\s*\(\s*\\lambda\s*\))?\$?\s+in\s+terms\s+of\s+\$?\\lambda\$?",
            compact,
            re.IGNORECASE,
        )
        if not request:
            return None
        vertices = int(request.group(1))
        if not 1 <= vertices <= 100:
            return None
        polynomial = SympyTool._path_partition_polynomial(vertices)
        request_tail = compact[request.end():]
        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show|establish)\b|"
            r"(?:also|and)\s+(?:give|find|derive|show).*?\brecurrence\b",
            compact,
            re.IGNORECASE,
        ) or re.search(r"\b(?:also|in\s+addition|and\s+then)\b", request_tail, re.IGNORECASE))
        label = "本地路径配分函数递推核验" if needs_support else "本地路径配分函数答案"
        return f"{label}: {polynomial}"

    @staticmethod
    def _path_partition_polynomial(vertices: int) -> str:
        """Apply ``Z_n=Z_{n-1}+lambda*Z_{n-2}`` from ``Z_0,Z_1`` exactly."""
        previous_two = [1]
        previous_one = [1, 1]
        for _ in range(2, vertices + 1):
            coefficients = previous_one.copy()
            if len(coefficients) < len(previous_two) + 1:
                coefficients.append(0)
            for size, coefficient in enumerate(previous_two, start=1):
                coefficients[size] += coefficient
            previous_two, previous_one = previous_one, coefficients
        result = previous_one
        terms = [str(result[0])]
        for size, coefficient in enumerate(result[1:], start=1):
            variable = r"\lambda" if size == 1 else rf"\lambda^{{{size}}}"
            terms.append(variable if coefficient == 1 else f"{coefficient}{variable}")
        return "+".join(terms)

    @staticmethod
    def _spike_sequence_construction_hint(problem: str) -> Optional[str]:
        """Provide the canonical unit-mass spike only for its exact contract."""
        text = str(problem or "")
        english = SympyTool._uses_english_prose(text)
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace("，", ",")
        )
        construct = bool(re.search(
            r"构造|写出.*(?:函数列|例子)|\b(?:construct|exhibit|give|find)\b.*?"
            r"(?:sequence|example)",
            normalized,
            re.IGNORECASE | re.DOTALL,
        ))
        function_sequence = bool(re.search(
            r"函数列|f\s*_?\s*\{?n\}?|sequence\s+of\s+(?:nonnegative\s+)?(?:measurable\s+)?functions?",
            normalized,
            re.IGNORECASE,
        ))
        convergence_context = bool(re.search(
            r"逐点|收敛|趋于|极限|pointwise|converge|tend|limit",
            normalized,
            re.IGNORECASE,
        ))
        integral_context = bool(re.search(r"积分|\\int|\bintegrals?\b", normalized, re.IGNORECASE))
        if not (construct and function_sequence and convergence_context and integral_context):
            return None

        exact_conditions = {
            "domain": bool(re.search(r"\[\s*0\s*,\s*1\s*\]", normalized)),
            "nonnegative": bool(re.search(r"非负|non[- ]?negative", normalized, re.IGNORECASE)),
            "measurable": bool(re.search(r"可测|measurable", normalized, re.IGNORECASE)),
            "pointwise_zero": bool(re.search(
                r"逐点\s*(?:收敛|趋(?:于|向)|极限(?:为|是)?)\s*(?:到|至)?\s*0|"
                r"(?:converges?|tends?)\s+pointwise\s+to\s+0|"
                r"pointwise.{0,24}(?:converges?|tends?|limit).{0,12}0",
                normalized,
                re.IGNORECASE,
            )),
            "unit_integral": bool(re.search(
                r"积分\s*(?:恒|始终)?\s*(?:等于|为|=)\s*1(?![\d/.])|"
                r"(?:integral|\\int).{0,40}(?:equals?|equal\s+to|is|=)\s*1(?![\d/.])",
                normalized,
                re.IGNORECASE,
            )),
            "formula": bool(re.search(
                r"具体公式|显式公式|写出.{0,12}公式|explicit\s+formula|"
                r"(?:give|write|state).{0,16}(?:formula|expression)",
                normalized,
                re.IGNORECASE,
            )),
        }
        extra_obligation = bool(re.search(
            r"证明|说明(?:理由|为什么|为何)|解释|验证|推广|比较|讨论|"
            r"并\s*(?:求|计算|证明|说明|验证|比较|讨论)|"
            r"范数|上确界|下确界|一致收敛|依测度收敛|控制收敛|"
            r"\b(?:prove|justify|explain|verify|generalize|compare|discuss|also|"
            r"supremum|infimum|norm|uniform\s+convergence|convergence\s+in\s+measure|"
            r"dominated\s+convergence)\b",
            normalized,
            re.IGNORECASE,
        ))
        result = (
            (
                r"Take \(f_n(x)=n\mathbf{1}_{(0,1/n]}(x)\) for \(x\in[0,1]\). "
                r"It is nonnegative and measurable, \(f_n(x)\to0\) pointwise on \([0,1]\), "
                r"and \(\int_0^1 f_n(x)\,dx=1\)."
            )
            if english else (
                r"取 f_n(x)=n\mathbf{1}_{(0,1/n]}(x)\ (x\in[0,1])；"
                r"则 f_n\geq0 且可测，逐点 f_n(x)\to0\ (\forall x\in[0,1])，"
                r"积分为 \int_0^1 f_n(x)\,dx=1。"
            )
        )
        if not all(exact_conditions.values()) or extra_obligation:
            missing = ",".join(name for name, present in exact_conditions.items() if not present)
            reason = "存在额外证明或计算义务" if extra_obligation else f"标准条件未完整匹配({missing})"
            return f"本地尖峰函数构造核验: {result} 仅核验上述标准构造；{reason}。"
        return f"本地尖峰函数构造答案: {result}"

    @staticmethod
    def _dependent_bernoulli_construction_hint(problem: str) -> Optional[str]:
        """Construct perfectly dependent fair Bernoulli marginals when asked exactly."""
        text = str(problem or "")
        english = SympyTool._uses_english_prose(text)
        normalized = re.sub(r"\s+", " ", text).strip()
        construct = bool(re.search(r"构造|\b(?:construct|exhibit|give)\b", normalized, re.IGNORECASE))
        random_variables = bool(re.search(
            r"随机变量|random\s+variables?",
            normalized,
            re.IGNORECASE,
        ))
        bernoulli = bool(re.search(r"Bernoulli|伯努利", normalized, re.IGNORECASE))
        dependence_context = bool(re.search(
            r"独立|不(?:相互)?独立|非独立|not\s+independent|\bdependent\b|\bindependent\b",
            normalized,
            re.IGNORECASE,
        ))
        if not (construct and random_variables and bernoulli and dependence_context):
            return None

        fair = r"(?:Bernoulli|伯努利)\s*[（(]\s*(?:1\s*/\s*2|0\.5)\s*[）)]"
        two_variables = bool(re.search(
            r"两个.{0,60}随机变量|random\s+variables?\s+X\s+and\s+Y|"
            r"two\s+(?:Bernoulli\s+)?random\s+variables?",
            normalized,
            re.IGNORECASE,
        ))
        fair_marginals = bool(
            re.search(
                rf"(?:两个)?边缘(?:分布)?.{{0,12}}(?:均|都).{{0,12}}{fair}|"
                rf"both\s+(?:marginals?|marginal\s+distributions?).{{0,12}}{fair}|"
                rf"(?:marginals?|marginal\s+distributions?).{{0,12}}(?:are\s+)?both.{{0,12}}{fair}|"
                rf"X\s+and\s+Y.{{0,20}}(?:both|each).{{0,16}}{fair}",
                normalized,
                re.IGNORECASE,
            )
        )
        not_independent = bool(re.search(
            r"不(?:相互)?独立|非独立|not\s+independent|\bdependent\b",
            normalized,
            re.IGNORECASE,
        ))
        equality_probability = bool(re.search(
            r"(?:P|\\mathbb\s*\{P\})\s*[（(]\s*X\s*=\s*Y\s*[）)]",
            normalized,
            re.IGNORECASE,
        ))

        probability_events = [
            re.sub(r"\s+", "", event).upper()
            for event in re.findall(
                r"(?:P|\\mathbb\s*\{P\})\s*[（(]\s*([^）)]+)\s*[）)]",
                normalized,
                re.IGNORECASE,
            )
        ]
        extra_probability = any(event not in {"X=Y", "Y=X"} for event in probability_events)
        extra_obligation = extra_probability or bool(re.search(
            r"证明|说明(?:理由|为什么|为何)|解释|验证|协方差|相关系数|相关性|"
            r"联合分布(?:表)?|条件概率|期望|方差|熵|互信息|不相关|"
            r"\b(?:prove|justify|explain|verify|covariance|correlation|uncorrelated|"
            r"joint\s+distribution|conditional\s+probability|expectation|variance|"
            r"entropy|mutual\s+information)\b",
            normalized,
            re.IGNORECASE,
        ))
        result = (
            (
                r"Let \(P((X,Y)=(0,0))=P((X,Y)=(1,1))=1/2\), with probability zero "
                r"otherwise (equivalently, \(X\sim\operatorname{Bernoulli}(1/2)\) and \(Y=X\)). "
                r"Both marginals are \(\operatorname{Bernoulli}(1/2)\), and "
                r"\(P(X=1,Y=1)=1/2\neq1/4=P(X=1)P(Y=1)\), so they are not independent; "
                r"\(P(X=Y)=1\)."
            )
            if english else (
                r"取 P((X,Y)=(0,0))=P((X,Y)=(1,1))=1/2，其余情形概率为0（即 "
                r"X\sim\operatorname{Bernoulli}(1/2),\ Y=X）。两边缘均为 "
                r"\operatorname{Bernoulli}(1/2)，且 P(X=1,Y=1)=1/2\neq1/4="
                r"P(X=1)P(Y=1)，故 X,Y 不独立；所求概率 P=1，即 P(X=Y)=1。"
            )
        )
        exact = two_variables and fair_marginals and not_independent and equality_probability
        if not exact or extra_obligation:
            reason = "存在额外证明或计算义务" if extra_obligation else "公平Bernoulli边缘、非独立或目标概率条件未完整匹配"
            return f"本地Bernoulli依赖构造核验: {result} 仅核验上述标准构造；{reason}。"
        return f"本地Bernoulli依赖构造答案: {result}"

    @staticmethod
    def _uses_english_prose(problem: str) -> bool:
        """Choose tool-answer prose without counting one-letter math variables."""
        value = re.sub(
            r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
            " ",
            str(problem or ""),
            flags=re.DOTALL,
        )
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", value))
        english_words = len(re.findall(r"\b[A-Za-z]{2,}\b", value))
        return english_words >= 2 and english_words > chinese_chars

    @staticmethod
    def _lz78_encoding_hint(problem: str) -> Optional[str]:
        """Encode a fully specified, standard fixed-width LZ78 exercise.

        A complete answer is certified only for the empty-dictionary LZ78
        convention (empty string at index 0), an explicit one-character binary
        alphabet map, and an input that ends when a new phrase is emitted.  The
        final dictionary has ``m`` phrases, so every prefix index that an output
        pair may contain lies in ``0,...,m-1`` and needs ``ceil(log2(m))`` bits.
        Variant or incomplete statements still receive a deterministic phrase
        check, but are never presented as a complete encoded answer.
        """
        text = str(problem or "")
        explicit_lz78 = bool(re.search(r"\bLZ[- ]?\s*78\b", text, re.IGNORECASE))
        generic_lempel_ziv = bool(re.search(r"\bLempel[- ]Ziv\b", text, re.IGNORECASE))
        if not (explicit_lz78 or generic_lempel_ziv):
            return None
        if re.search(r"\bLZ\s*77\b|sliding\s+window|滑动窗口", text, re.IGNORECASE):
            return None
        asks_phrases = bool(re.search(
            r"(?:decomposition\s+into\s+phrases|phrase\s+decomposition|"
            r"decompos(?:e|ition).*?phrases?|短语分解|分解.*?短语)",
            text,
            re.IGNORECASE | re.DOTALL,
        ))
        asks_encoding = bool(re.search(
            r"(?:encoded\s+string|encode(?:d|s|ing)?\s+(?:the\s+)?(?:message|string)|"
            r"编码(?:串|结果|该?(?:消息|字符串)))",
            text,
            re.IGNORECASE,
        ))
        if not (asks_phrases and asks_encoding):
            return None

        message = SympyTool._lz78_message(text)
        if not message or len(message) > 10000:
            return None
        pairs, phrases, terminal_prefix = SympyTool._lz78_parse(message)
        if not pairs:
            return None

        pair_text = ", ".join(f"({index},{symbol})" for index, symbol in pairs)
        phrase_text = ", ".join(phrases)
        base = f"Phrases: {phrase_text}; pairs: {pair_text}"
        issues: list[str] = []
        if not explicit_lz78:
            issues.append(
                "the Lempel-Ziv variant is unspecified; these values use standard empty-dictionary LZ78"
            )
        if terminal_prefix:
            issues.append(
                "the input ends in an existing dictionary phrase "
                f"{terminal_prefix!r}, so an EOF convention is required"
            )

        mapping, mapping_issue = SympyTool._lz78_letter_mapping(text)
        if mapping_issue:
            issues.append(mapping_issue)
        missing_symbols = sorted(set(message) - set(mapping))
        if missing_symbols:
            issues.append("the explicit bit mapping omits " + ", ".join(missing_symbols))

        phrase_count = len(pairs)
        derived_width = max(1, (phrase_count - 1).bit_length())
        explicit_widths = SympyTool._lz78_explicit_index_widths(text)
        if len(explicit_widths) > 1:
            issues.append("conflicting index widths are stated")
            index_width = derived_width
        elif explicit_widths:
            index_width = next(iter(explicit_widths))
            if index_width < derived_width:
                issues.append(
                    f"the stated {index_width}-bit index cannot represent all prefix indices 0,...,{phrase_count - 1}"
                )
            elif index_width > 64:
                issues.append("the stated index width is outside the supported deterministic range")
        else:
            index_width = derived_width

        if re.search(
            r"(?:index|indices|dictionary\s+entries?).{0,24}(?:start|begin)(?:s|ning)?\s+(?:at|from)\s+1|"
            r"(?:索引|下标).{0,12}从\s*1\s*开始|"
            r"(?:preloaded|initial(?:ly)?\s+contains|initial\s+dictionary\s+(?:is\s+)?(?:not\s+empty|contains)|"
            r"预置字典|初始字典.{0,8}(?:非空|包含))|"
            r"(?:variable|dynamic|adaptive)[- ](?:width|length)\s+(?:index|code)|"
            r"(?:变长|动态|自适应).{0,8}(?:索引|编码)",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            issues.append("the problem specifies a nonstandard dictionary or index convention")

        needs_support = bool(re.search(
            r"\b(?:prove|justify|explain|derive|show\s+why)\b|证明|说明理由|解释|推导",
            text,
            re.IGNORECASE,
        ))
        if needs_support:
            issues.append("the requested justification is not covered by the deterministic encoding result")

        can_encode = (
            not missing_symbols
            and len(explicit_widths) <= 1
            and index_width >= derived_width
            and index_width <= 64
        )
        encoded_chunks = (
            [f"{index:0{index_width}b}{mapping[symbol]}" for index, symbol in pairs]
            if can_encode else []
        )
        encoded_check = (
            f"; candidate encoded string: {' '.join(encoded_chunks)}"
            if encoded_chunks else ""
        )
        if issues:
            return (
                f"本地LZ78编码核验: {base}{encoded_check}; "
                f"verification only: {'; '.join(issues)}"
            )

        width_reason = (
            f"stated index width: {index_width} bits"
            if explicit_widths
            else (
                f"fixed index width: ceil(log2({phrase_count}))={index_width} bits "
                f"for prefix indices 0,...,{phrase_count - 1}"
            )
        )
        return (
            f"本地LZ78编码答案: {base}; {width_reason}; "
            f"encoded string: {' '.join(encoded_chunks)}"
        )

    @staticmethod
    def _lz78_message(text: str) -> str:
        """Extract one explicit alphanumeric message token from public text."""
        patterns = (
            r"(?:consider\s+the\s+)?message\s*(?:is|=|:|：)?\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
            r"(?:encode|compress)\s+(?:the\s+)?(?:message|string)\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
            r"(?:消息|报文|字符串)\s*(?:为|是|=|:|：)\s*"
            r"(?:\\texttt\s*\{|[`\"'$])?\s*([A-Za-z0-9]+)\s*(?:\}|[`\"'$])?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1).lower() not in {
                "is", "the", "a", "an", "using", "obtained", "string",
            }:
                return match.group(1)
        return ""

    @staticmethod
    def _lz78_parse(message: str) -> tuple[list[tuple[int, str]], list[str], str]:
        """Return standard LZ78 output pairs and any ambiguous terminal prefix."""
        dictionary = {"": 0}
        pairs: list[tuple[int, str]] = []
        phrases: list[str] = []
        position = 0
        while position < len(message):
            cursor = position
            prefix = ""
            while cursor < len(message) and prefix + message[cursor] in dictionary:
                prefix += message[cursor]
                cursor += 1
            if cursor == len(message):
                return pairs, phrases, prefix
            symbol = message[cursor]
            phrase = prefix + symbol
            pairs.append((dictionary[prefix], symbol))
            phrases.append(phrase)
            dictionary[phrase] = len(dictionary)
            position = cursor + 1
        return pairs, phrases, ""

    @staticmethod
    def _lz78_letter_mapping(text: str) -> tuple[dict[str, str], str]:
        normalized = (
            text.replace(r"\rightarrow", "->")
            .replace(r"\longrightarrow", "->")
            .replace(r"\to", "->")
            .replace("→", "->")
        )
        entries = re.findall(
            r"(?<![A-Za-z0-9])([A-Za-z0-9])\s*->\s*([01]+)(?![01])",
            normalized,
        )
        mapping: dict[str, str] = {}
        for symbol, bits in entries:
            if symbol in mapping and mapping[symbol] != bits:
                return mapping, f"conflicting bit codes are given for {symbol}"
            mapping[symbol] = bits
        if not mapping:
            return {}, "no explicit letter-to-bit mapping was found"
        widths = {len(bits) for bits in mapping.values()}
        if len(widths) != 1:
            return mapping, "the explicit letter codes do not have one fixed width"
        if len(set(mapping.values())) != len(mapping):
            return mapping, "the explicit letter-to-bit mapping is not one-to-one"
        return mapping, ""

    @staticmethod
    def _lz78_explicit_index_widths(text: str) -> set[int]:
        widths: set[int] = set()
        patterns = (
            r"(?:dictionary\s+)?(?:index|pointer)\s*(?:field)?\s*(?:uses?|is|has|:|=)?\s*(\d+)\s*[- ]?bits?",
            r"(\d+)\s*[- ]?bit\s+(?:dictionary\s+)?(?:index|pointer)",
            r"(?:索引|下标)(?:字段)?.{0,12}?(\d+)\s*位",
        )
        for pattern in patterns:
            widths.update(int(value) for value in re.findall(pattern, text, re.IGNORECASE))
        return widths

    def _linear_recurrence_hint(self, problem: str) -> Optional[str]:
        """Solve a first-order affine recurrence only when every coefficient is explicit."""
        match = re.search(
            r"a_n\s*=\s*([+-]?\d*)\s*\*?\s*a_\{?n-1\}?\s*([+-]\s*\d+)?",
            problem,
            re.IGNORECASE,
        )
        initial = re.search(r"a_1\s*=\s*([+-]?\d+(?:/\d+)?)", problem, re.IGNORECASE)
        if not match or not initial or not self.sympy:
            return None
        coefficient_text = match.group(1).replace(" ", "")
        coefficient_text = "1" if coefficient_text in {"", "+"} else ("-1" if coefficient_text == "-" else coefficient_text)
        offset_text = (match.group(2) or "0").replace(" ", "")
        try:
            coefficient = self.sympy.Rational(coefficient_text)
            offset = self.sympy.Rational(offset_text)
            first = self.sympy.Rational(initial.group(1))
            n = self.sympy.Symbol("n", integer=True, positive=True)
            if coefficient == 1:
                expression = first + (n - 1) * offset
            else:
                fixed_point = offset / (1 - coefficient)
                expression = fixed_point + (first - fixed_point) * coefficient ** (n - 1)
            return f"SymPy 递推通项: a_n={self._format(self.sympy.simplify(expression))}"
        except Exception:
            return None

    def _curve_speed_hint(self, problem: str) -> Optional[str]:
        if not self.sympy or not re.search(r"速度长度|弧长参数", problem):
            return None
        match = re.search(
            r"(?:γ|gamma)\s*\(\s*([A-Za-z])\s*\)\s*=\s*\(([^()]+(?:\([^()]*\)[^()]*)*)\)",
            problem,
            re.IGNORECASE,
        )
        if not match:
            return None
        components = [item.strip() for item in match.group(2).split(",")]
        if len(components) not in {2, 3}:
            return None
        try:
            variable = self.sympy.Symbol(match.group(1))
            vector = [self._parse(self._latex_to_sympy(item)) for item in components]
            speed = self.sympy.simplify(self.sympy.sqrt(sum(self.sympy.diff(item, variable) ** 2 for item in vector)))
            judgement = "是弧长参数" if self.sympy.simplify(speed - 1) == 0 else "不是弧长参数"
            return f"SymPy 曲线速度: 速度长度为{self._format(speed)}，{judgement}"
        except Exception:
            return None

    def _first_fundamental_form_hint(self, problem: str) -> Optional[str]:
        if not self.sympy or not re.search(r"第一基本形式.*E\s*[,，]\s*F\s*[,，]\s*G", problem, re.IGNORECASE):
            return None
        match = re.search(
            r"X\s*\(\s*([A-Za-z])\s*,\s*([A-Za-z])\s*\)\s*=\s*\(([^()]+)\)",
            problem,
        )
        if not match:
            return None
        components = [item.strip() for item in match.group(3).split(",")]
        if len(components) != 3:
            return None
        try:
            u, v = self.sympy.Symbol(match.group(1)), self.sympy.Symbol(match.group(2))
            vector = [self._parse(self._latex_to_sympy(item)) for item in components]
            xu = [self.sympy.diff(item, u) for item in vector]
            xv = [self.sympy.diff(item, v) for item in vector]
            e_value = self.sympy.simplify(sum(item * item for item in xu))
            f_value = self.sympy.simplify(sum(left * right for left, right in zip(xu, xv)))
            g_value = self.sympy.simplify(sum(item * item for item in xv))
            return (
                "SymPy 第一基本形式: "
                f"E={self._format(e_value)}，F={self._format(f_value)}，G={self._format(g_value)}"
            )
        except Exception:
            return None

    @staticmethod
    def _graph_gaussian_curvature_hint(problem: str) -> Optional[str]:
        if re.search(r"曲面.*z\s*=\s*f\s*\(\s*x\s*,\s*y\s*\).*∇f\s*=\s*0", problem, re.IGNORECASE) and re.search(
            r"高斯曲率.*Hessian|Hessian.*高斯曲率", problem, re.IGNORECASE
        ):
            return "本地高斯曲率公式: K=f_{xx}f_{yy}-f_{xy}^2"
        return None

    def _pde_verification_hint(self, problem: str) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            x, y, t = self.sympy.symbols("x y t")
            if re.search(r"热方程.*u_t\s*=\s*u_\{?xx\}?", problem, re.IGNORECASE):
                match = re.search(r"u\s*\(\s*x\s*,\s*t\s*\)\s*=\s*(.+?)(?=是否|，|。|；|;|$)", problem)
                if match:
                    expression = self._parse(self._latex_to_sympy(match.group(1)))
                    time_derivative = self.sympy.simplify(self.sympy.diff(expression, t))
                    space_derivative = self.sympy.simplify(self.sympy.diff(expression, x, 2))
                    judgement = "是解" if self.sympy.simplify(time_derivative - space_derivative) == 0 else "不是解"
                    return (
                        "SymPy PDE核验: "
                        f"u_t={self._format(time_derivative)}，u_{{xx}}={self._format(space_derivative)}，{judgement}"
                    )
            if re.search(r"拉普拉斯方程|u_\{?xx\}?\s*\+\s*u_\{?yy\}?\s*=\s*0", problem, re.IGNORECASE):
                match = re.search(r"函数\s*u\s*=\s*(.+?)(?=是否|调和|，|。|；|;|$)", problem)
                if match:
                    expression = self._parse(self._latex_to_sympy(match.group(1)))
                    u_xx = self.sympy.simplify(self.sympy.diff(expression, x, 2))
                    u_yy = self.sympy.simplify(self.sympy.diff(expression, y, 2))
                    total = self.sympy.simplify(u_xx + u_yy)
                    judgement = "是调和函数" if total == 0 else "不是调和函数"
                    return (
                        "SymPy PDE核验: "
                        f"u_{{xx}}={self._format(u_xx)}，u_{{yy}}={self._format(u_yy)}，二者之和为{self._format(total)}，{judgement}"
                    )
        except Exception:
            return None
        return None

    @staticmethod
    def _plain_equations(problem: str) -> list[str]:
        """Extract only short, ASCII-style equations outside LaTex delimiters."""
        if "$" in problem or not re.search(r"方程|求解|equation|solve|roots?|zeros?", problem, re.IGNORECASE):
            return []
        matches = re.findall(
            r"([0-9xyzXYZ(][0-9A-Za-z_+\-*/^().,\s]{0,120}=[0-9A-Za-z_+\-*/^().,\s]{1,120})",
            problem,
        )
        return [match.strip() for match in matches]

    @staticmethod
    def _congruence_hint(problem: str) -> Optional[str]:
        normalized = (
            problem.replace(r"\equiv", "≡")
            .replace(r"\pmod{", " mod ")
            .replace(r"\pmod", " mod ")
            .replace("}", "")
        )
        match = re.search(
            r"(-?\d+)\s*\*?\s*x\s*≡\s*(-?\d+)\s*(?:mod\s*|\b)(\d+)",
            normalized,
        )
        if not match:
            return None
        coefficient, constant, modulus = map(int, match.groups())
        divisor = gcd(coefficient, modulus)
        if constant % divisor:
            return "本地同余方程：无解"
        if divisor != 1:
            return None
        solution = (pow(coefficient % modulus, -1, modulus) * constant) % modulus
        return f"本地同余方程解: x={solution} (mod {modulus})"

    @staticmethod
    def _modular_power_hint(problem: str) -> Optional[str]:
        normalized = problem.replace(r"\bmod", "mod").replace("{", "").replace("}", "")
        match = re.search(r"(-?\d+)\s*\^\s*(\d+)\s*mod\s*(\d+)", normalized)
        if not match:
            return None
        base, exponent, modulus = map(int, match.groups())
        if modulus == 0:
            return None
        return f"本地模幂计算: {pow(base, exponent, modulus)}"

    @staticmethod
    def _raw_latex_parts(problem: str) -> list[str]:
        """Find standalone raw LaTex limits and integrals without `$...$`."""
        return re.findall(r"(\\(?:lim|int)[^。？?\n]+)", problem)

    def _parse(self, expression: str):
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", expression):
            raise ValueError("unsupported symbolic expression")
        identifiers = set(re.findall(r"[A-Za-z]+", expression))
        allowed = {"sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt", "pi", "oo"}
        if any(identifier not in allowed and len(identifier) != 1 for identifier in identifiers):
            raise ValueError("unsupported symbolic identifier")
        return parse_expr(
            expression,
            transformations=standard_transformations + (implicit_multiplication_application,),
        )

    @staticmethod
    def _latex_to_sympy(expression: str) -> str:
        # English prose extractors may include the sentence-final period.  It
        # would turn an integer exponent such as ``x^3.`` into SymPy's ``3.0``.
        value = expression.strip().replace("$", "").rstrip("。；;，,.!?？")
        value = SympyTool._replace_fractions(value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        function_names = {
            "arctan": "atan",
            "arcsin": "asin",
            "arccos": "acos",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "ln": "log",
            "exp": "exp",
        }
        value = re.sub(
            r"\\(arctan|arcsin|arccos|sin|cos|tan|log|ln|exp)",
            lambda m: f" {function_names[m.group(1)]}",
            value,
        )
        value = (
            value.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\!", "")
            .replace(r"\pi", "pi")
            .replace(r"\infty", "oo")
            .replace(r"\,", "")
        )
        value = value.replace("^", "**").replace("{", "(").replace("}", ")")
        value = re.sub(r"(?<=[xyzXYZ])(?=[xyzXYZ])", "*", value)
        return re.sub(r"(?<![A-Za-z])e(?=\s*\*\*)", "E", value)

    @staticmethod
    def _replace_fractions(value: str) -> str:
        """Convert nested LaTex fractions without relying on a full TeX parser."""
        marker = r"\frac"
        while marker in value:
            start = value.find(marker)
            numerator = SympyTool._braced_group(value, start + len(marker))
            if numerator is None:
                break
            numerator_text, after_numerator = numerator
            denominator = SympyTool._braced_group(value, after_numerator)
            if denominator is None:
                break
            denominator_text, after_denominator = denominator
            replacement = f"({numerator_text})/({denominator_text})"
            value = value[:start] + replacement + value[after_denominator:]
        return value

    @staticmethod
    def _braced_group(value: str, start: int) -> Optional[tuple[str, int]]:
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value) or value[start] != "{":
            return None
        depth = 0
        for index in range(start, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    return value[start + 1:index], index + 1
        return None

    @staticmethod
    def _format(value: Any) -> str:
        text = str(value).replace("**", "^")
        text = re.sub(r"\blog\(", "ln(", text)
        text = re.sub(r"\batan\(", "arctan(", text)
        text = re.sub(r"\basin\(", "arcsin(", text)
        text = re.sub(r"\bacos\(", "arccos(", text)
        text = re.sub(r"\bexp\(x\)", "e^x", text)
        text = re.sub(r"\bexp\(([^()]+)\)", r"e^(\1)", text)
        return re.sub(r"(?<![A-Za-z])oo(?![A-Za-z])", "∞", text)

    def _run(self, operation) -> Optional[str]:
        if not self.sympy:
            return None
        try:
            return self._format(operation(self.sympy))
        except Exception:
            return None
