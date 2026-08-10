"""Structured contracts and certificates for deterministic math tools.

The public agent historically consumed strings such as ``"SymPy 计算: 4"``.
This module keeps that representation available while giving the solver an
explicit allow-list: an unknown label is evidence text, never a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
        return f"{self.label}: {self.result}"

    def trace_content(self) -> dict:
        return {
            "operation": self.operation,
            "result_kind": self.contract.result_kind if self.contract else "unknown",
            "whole_answer_eligible": self.whole_answer_eligible,
            "certificate": self.certificate.trace_content(),
        }


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
    "simple_random_walk_moments": _contract(
        "simple_random_walk_moments", "moments", "iid_moment_identity", True,
        requirements=("independent_increments",),
    ),
    "complete_graph_cover_time": _contract("complete_graph_cover_time", "expectation", "coupon_collector_identity", True),
    "two_venue_capacity": _contract("two_venue_capacity", "minimum_integer", "exact_binomial_tail", True),
    "circle_laplacian": _contract(
        "circle_laplacian", "scalar", "ambient_second_derivatives", True,
        facts=("f=x^2+y^2", "circle_constraint", "ambient_or_unqualified_laplacian"),
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
    "central_difference": _contract("central_difference", "approximation", "formula_evaluation", True),
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
    "cycle_distance_two_coloring": _contract(
        "cycle_distance_two_coloring", "count", "cyclic_boundary_state_dynamic_programming", True,
        answer_shapes=("number",),
    ),
    "punctured_domino_tilings": _contract(
        "punctured_domino_tilings", "count", "obstacle_profile_dynamic_programming", True,
        answer_shapes=("number",),
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
    "cyclic_nonadjacent_selection": _contract(
        "cyclic_nonadjacent_selection", "count", "cycle_gap_bijection", True,
        answer_shapes=("number",),
    ),
    "finite_subtraction_game": _contract(
        "finite_subtraction_game", "count", "bounded_game_dynamic_programming", True,
        answer_shapes=("number",),
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
    "descartes_inner_circle": _contract(
        "descartes_inner_circle", "scalar", "descartes_curvature_identity", True,
        answer_shapes=("number", "expression"),
    ),
    "rotation_necklace_fixed_weight": _contract(
        "rotation_necklace_fixed_weight", "count", "cyclic_orbit_enumeration", True,
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
    "本地不相邻二进制串计数": "nonadjacent_binary_string_count",
    "本地排列条件计数": "precedence_permutation_count",
    "本地满射容斥计数": "surjection_count",
    "本地平面图欧拉答案": "planar_euler_faces",
    "本地抛物面曲率答案": "paraboloid_curvature",
    "本地有序三元组计数": "ordered_positive_triples",
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
    "本地相邻约束满射计数": "adjacent_surjection_count",
    "本地重复字母隔位计数": "multiset_no_adjacent_count",
    "本地二进制游程计数": "binary_run_avoidance_count",
    "本地手链轨道计数": "bracelet_no_adjacent_count",
    "本地条带格路计数": "strip_lattice_path_count",
    "本地嵌套模幂和": "nested_modular_power_sum",
    "本地二次型最大值": "quadratic_form_maximum",
    "本地循环距离二染色计数": "cycle_distance_two_coloring",
    "本地障碍多米诺铺法计数": "punctured_domino_tilings",
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
    "本地圆周不相邻选择计数": "cyclic_nonadjacent_selection",
    "本地减法博弈必败态计数": "finite_subtraction_game",
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
    "本地Descartes内切圆半径": "descartes_inner_circle",
    "本地定重旋转项链计数": "rotation_necklace_fixed_weight",
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
    return ToolResult(result, operation, label, contract, certificate)
