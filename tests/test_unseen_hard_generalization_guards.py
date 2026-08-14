from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate
from rag.card_retriever import CardRetriever


def test_chinese_classification_problem_with_required_proof_is_a_proof_task():
    problem = (
        r"确定方程 x^2+y^2+1=3xy 的全部正整数有序解。"
        "要求给出无遗漏的分类证明，并说明交换两个坐标后的情形。"
    )
    spec = build_problem_spec(problem)
    assert spec.profile.problem_type == "proof"
    assert spec.profile.subject == "数论"
    assert spec.primary_method == "vieta_jumping_descent"
    assert spec.answer_contract.mode == "proof"


def test_state_recursion_is_not_misread_as_an_output_state_command():
    problem = (
        "A fair coin is tossed until either HHTH or THHT first appears. "
        "Determine the probability that HHTH appears first, and prove it using full "
        "prefix-state recursion rather than independent blocks."
    )
    spec = build_problem_spec(problem)
    requirements = {
        requirement.name
        for goal in spec.goals
        for requirement in goal.requirements
    }
    assert "output_object_transform" not in requirements


def test_endpoint_proof_is_support_not_a_second_external_goal():
    problem = (
        r"For n\ge1 let g_n(x)=n/(1+n^2x^2) on (0,\infty). "
        r"Determine exactly the p\in[1,\infty] for which g_n converges in L^p, "
        "and give a rigorous proof treating both endpoints separately."
    )
    spec = build_problem_spec(problem)
    assert len(spec.goals) == 1
    assert spec.answer_contract.mode == "proof"


def test_one_changed_degraded_audit_cannot_replace_a_usable_value():
    spec = build_problem_spec(
        "Determine a probability and prove it using a complete state recursion."
    )
    baseline = assess_candidate(
        r"Conclusion: \frac{5}{14}. Since the prefix-state equations form a complete recursion, the probability is \frac{5}{14}.",
        "continue",
        spec,
        (),
    )
    changed = assess_candidate(
        r"Since a different recursion gives \frac{5}{12}, the probability is \frac{5}{12}.",
        "verify_recovered",
        spec,
        (),
        verification_verdict="confirmed",
    )
    filtered = SubmissionAgent._without_uncorroborated_corrections(
        [baseline, changed], baseline, spec=spec
    )
    assert baseline in filtered
    assert changed not in filtered


def test_cellular_homology_result_and_mandatory_proof_remain_one_goal():
    problem = (
        r"Let Y be a finite CW complex with one vertex and several oriented cells. "
        r"Determine H_1(Y;\mathbb Z). The proof must construct the cellular boundary "
        "matrix from the attaching words and reduce it to Smith normal form."
    )
    spec = build_problem_spec(problem)
    bundle = CardRetriever().retrieve(spec)

    assert spec.profile.subject == "拓扑学"
    assert spec.profile.subject_confidence == "high"
    assert spec.profile.problem_type == "proof"
    assert len(spec.goals) == 1
    assert "H_1" in spec.goals[0].instruction
    assert "cellular boundary" in spec.goals[0].instruction
    assert spec.answer_contract.mode == "proof"
    assert spec.primary_method == "cellular_chain_complex_then_smith_normal_form"
    assert spec.alternative_method == "fundamental_group_abelianization_check"
    assert bundle.solve_cards[0].id == "method.topology.cellular_homology"
    assert "Applicability gate" in bundle.solve_context()


def test_chebyshev_polynomial_domain_is_not_an_interval_answer():
    chinese = build_problem_spec(
        "在一般区间[a,b]上，确定满足给定首项系数的四次极小极大多项式。"
    )
    english = build_problem_spec(
        "Find a monic minimax polynomial of degree five on [2,7] and justify "
        "optimality by the equioscillation theorem."
    )

    for spec in (chinese, english):
        bundle = CardRetriever().retrieve(spec)
        assert spec.profile.subject == "数值分析"
        assert spec.profile.subject_confidence == "high"
        assert spec.profile.answer_shape != "interval"
        assert spec.primary_method == "chebyshev_affine_map_and_normalized_alternation"
        assert spec.alternative_method == "equioscillation_linear_system_check"
        assert bundle.solve_cards[0].id == "method.numerical.chebyshev_minimax"
    assert "适用门槛" in CardRetriever().retrieve(chinese).solve_context()
    assert "Applicability gate" in CardRetriever().retrieve(english).solve_context()


def test_requested_polynomial_does_not_override_a_direct_root_request():
    assert build_problem_spec("求多项式 x^3-1 的全部根。").profile.answer_shape == "roots"
    assert build_problem_spec(
        "Find the zeros of the polynomial p(x)=x^4+1."
    ).profile.answer_shape == "roots"


def test_nonproof_homology_group_is_an_expression_result():
    spec = build_problem_spec(
        "用附着映射写出有限 CW 复形的胞腔链复形，并确定其同调群。"
    )
    assert spec.profile.subject == "拓扑学"
    assert spec.profile.answer_shape == "expression"
    assert spec.primary_method == "cellular_chain_complex_then_smith_normal_form"


def test_latin_square_routes_to_symmetry_normalization_not_generic_counting():
    for problem in (
        "计数所有行列均为符号集排列、且两条对角线还满足指定限制的拉丁方。",
        "Count Latin squares whose rows and columns are permutations, subject to "
        "separate constraints on the two diagonals.",
        "An n by n labeled array uses n symbols. Each symbol occurs exactly once "
        "in every row and every column, while both main diagonals have distinct "
        "entries. Determine the count by normalizing a row and restoring labels.",
    ):
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)
        assert spec.profile.subject == "离散数学"
        assert spec.profile.subject_confidence == "high"
        assert spec.primary_method == "normalize_symmetry_then_exhaust_structural_cases"
        assert spec.alternative_method == "exact_enumeration_with_orbit_size_check"
        assert bundle.solve_cards[0].id == "method.combinatorics.latin_square"


def test_nowhere_zero_flow_routes_to_cycle_space_edgewise_inclusion_exclusion():
    for problem in (
        "对一个含若干连通分支的有限图，求有限域上的无处零流数量，并用圈空间核验。",
        "For a finite graph, count nowhere-zero flows over GF(q) by working in "
        "the cycle space and enforcing conservation at every vertex.",
    ):
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)
        assert spec.profile.subject == "离散数学"
        assert spec.profile.subject_confidence == "high"
        assert spec.primary_method == "cycle_space_coordinate_inclusion_exclusion"
        assert spec.alternative_method == "tutte_flow_polynomial_or_exact_edge_enumeration"
        assert bundle.solve_cards[0].id == "method.graph.nowhere_zero_flow"
        rendered = bundle.solve_context().lower()
        assert "每一条边" in rendered or "every edge" in rendered
