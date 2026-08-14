from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from classifier.subject import classify_subjects
from rag.card_retriever import CardRetriever


class DirectedEulerCircuitRoutingTest(unittest.TestCase):
    def test_cyclic_rotation_of_a_directed_tour_is_not_geometry(self):
        problem = (
            "A strongly connected directed multigraph has equal indegree and outdegree "
            "at every vertex. Count its Eulerian circuits up to cyclic rotation when a "
            "specified first arc is fixed."
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "离散数学")
        self.assertEqual(spec.profile.topic, "directed_euler_circuits")
        self.assertEqual(spec.profile.difficulty, "hard")
        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(spec.primary_method, "best_theorem_with_fixed_arc_normalization")
        self.assertEqual(
            spec.alternative_method,
            "directed_matrix_tree_and_exit_ordering_check",
        )
        self.assertIn(
            "method.graph.directed_euler_circuits",
            {card.id for card in bundle.solve_cards},
        )

    def test_chinese_directed_euler_vocabulary_has_high_subject_confidence(self):
        route = classify_subjects(
            "给定一个有向多重图，求固定首弧的欧拉闭迹数量，并核对有向生成树。"
        )

        self.assertEqual(route.primary, "离散数学")
        self.assertEqual(route.confidence, "high")
        self.assertIn("离散数学:directed-euler-circuits", route.matched_signals)


class PlaneRootedTreeRoutingTest(unittest.TestCase):
    def test_outdegree_profile_routes_to_lukasiewicz_cycle_lemma(self):
        problem = (
            "Count unlabeled ordered plane rooted trees with a prescribed out-degree "
            "profile, and justify the prefix constraint through Lukasiewicz words."
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "离散数学")
        self.assertEqual(spec.profile.topic, "plane_rooted_tree_enumeration")
        self.assertEqual(spec.primary_method, "lukasiewicz_words_and_cycle_lemma")
        self.assertEqual(
            spec.alternative_method,
            "rooted_plane_tree_degree_sequence_formula",
        )
        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(bundle.solve_cards[0].id, "method.combinatorics.plane_rooted_trees")
        self.assertIn("sum i*n_i=n-1", bundle.solve_context())


class LacunaryNaturalBoundaryRoutingTest(unittest.TestCase):
    def test_radius_and_natural_boundary_remain_one_complete_proof_goal(self):
        problem = (
            r"For the lacunary power series f(z)=\sum_{k\geq0}z^{m_k}, "
            "determine its radius of convergence and prove that the unit circle "
            "is a natural boundary under the stated gap condition."
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)
        requirements = {item.name for item in spec.goals[0].requirements}

        self.assertEqual(spec.profile.subject, "复分析")
        self.assertNotEqual(spec.profile.subject, "初等几何")
        self.assertEqual(spec.profile.topic, "lacunary_natural_boundary")
        self.assertEqual(len(spec.goals), 1)
        self.assertEqual(spec.profile.answer_shape, "proof")
        self.assertTrue({
            "convergence_radius",
            "natural_boundary_classification",
            "reasoning",
        }.issubset(requirements))
        by_name = {item.name: item for item in spec.goals[0].requirements}
        explicit_conclusion = r"R=1; |z|=1 is a natural boundary."
        self.assertTrue(by_name["convergence_radius"].matches(explicit_conclusion))
        self.assertTrue(by_name["natural_boundary_classification"].matches(explicit_conclusion))
        self.assertEqual(spec.primary_method, "radius_then_dense_boundary_singularities")
        self.assertEqual(spec.alternative_method, "fabry_or_hadamard_gap_theorem")
        self.assertEqual(bundle.solve_cards[0].id, "method.complex.lacunary_natural_boundary")

    def test_implicit_no_continuation_through_any_arc_is_the_same_family(self):
        problem = (
            r"For the power series g(z)=\sum_{j\geq1} z^{q_j}, classify its circle "
            "of convergence. First find the radius, exhibit dense boundary singularities, "
            "and exclude analytic continuation through every boundary arc."
        )

        spec = build_problem_spec(problem)
        requirements = {item.name for item in spec.goals[0].requirements}

        self.assertEqual(spec.profile.subject, "复分析")
        self.assertEqual(spec.profile.topic, "lacunary_natural_boundary")
        self.assertEqual(spec.primary_method, "radius_then_dense_boundary_singularities")
        self.assertIn("convergence_radius", requirements)
        self.assertIn("natural_boundary_classification", requirements)

    def test_explicit_convergence_domain_is_a_result_obligation(self):
        spec = build_problem_spec(
            "确定该稀疏幂级数的收敛域，并证明其收敛圆周是自然边界。"
        )
        requirements = {item.name for item in spec.goals[0].requirements}

        self.assertIn("convergence_domain", requirements)
        self.assertIn("natural_boundary_classification", requirements)


class RungeKuttaStabilityRoutingTest(unittest.TestCase):
    def test_sdirk_order_and_l_stability_get_a_numerical_analysis_route(self):
        problem = (
            "给定一个两级 SDIRK 方法的 Butcher 表，推导稳定函数 R(z)，"
            "利用阶条件判定二阶参数，并检查 L-稳定所需的无穷远极限。"
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)
        requirements = {item.name for item in spec.goals[0].requirements}

        self.assertEqual(spec.profile.subject, "数值分析")
        self.assertEqual(spec.profile.topic, "runge_kutta_stability")
        self.assertEqual(spec.profile.difficulty, "hard")
        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(spec.primary_method, "order_conditions_then_stability_function")
        self.assertEqual(
            spec.alternative_method,
            "imaginary_axis_modulus_and_infinity_limit",
        )
        self.assertIn("stability_function", requirements)
        self.assertIn("stability_infinity_limit", requirements)
        by_name = {item.name: item for item in spec.goals[0].requirements}
        explicit_result = r"R(z)=1+zb^T(I-zA)^{-1}e,\quad \lim_{z\to\infty}R(z)=0."
        self.assertTrue(by_name["stability_function"].matches(explicit_result))
        self.assertTrue(by_name["stability_infinity_limit"].matches(explicit_result))
        self.assertEqual(bundle.solve_cards[0].id, "method.numerical.runge_kutta_stability")
        self.assertIn("R(z)=1+z", bundle.solve_context())


class SphericalTriangleAreaRoutingTest(unittest.TestCase):
    def test_result_with_mandatory_derivation_stays_one_goal(self):
        problem = (
            "On a sphere of radius R, a spherical triangle has geodesic side "
            "lengths a,b,c. Compute its area. The derivation must use the spherical "
            "law of cosines and justify the radius normalization."
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "微分几何")
        self.assertEqual(spec.profile.topic, "spherical_triangle_area")
        self.assertEqual(spec.profile.answer_shape, "number")
        self.assertEqual(len(spec.goals), 1)
        self.assertIn("radius normalization", spec.goals[0].instruction)
        self.assertEqual(spec.answer_contract.mode, "answer_with_support")
        self.assertEqual(
            spec.primary_method,
            "spherical_cosine_law_then_girard_excess",
        )
        self.assertEqual(
            spec.alternative_method,
            "gram_matrix_or_vector_angle_area_check",
        )
        self.assertEqual(bundle.solve_cards[0].id, "method.geometry.spherical_triangle_area")
        self.assertIn("R^2", bundle.solve_context())

    def test_chinese_girard_area_route_is_differential_geometry(self):
        route = classify_subjects(
            "求半径为R的球面上一个球面三角形的面积，并用球面余弦定理和Girard定理说明。"
        )

        self.assertEqual(route.primary, "微分几何")
        self.assertEqual(route.confidence, "high")
        self.assertIn("微分几何:spherical-triangle-area", route.matched_signals)


class WeierstrassSineProductRoutingTest(unittest.TestCase):
    def test_chinese_imaginary_substitution_routes_to_sine_product(self):
        problem = (
            "利用正弦函数的 Weierstrass 无穷乘积，通过纯虚代换推导双曲正弦的乘积公式。"
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "复分析")
        self.assertEqual(spec.profile.topic, "weierstrass_sine_product")
        self.assertEqual(
            spec.primary_method,
            "weierstrass_sine_product_then_imaginary_substitution",
        )
        self.assertEqual(
            spec.alternative_method,
            "zero_set_normalization_and_log_derivative_check",
        )
        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(bundle.solve_cards[0].id, "method.complex.weierstrass_sine_product")
        self.assertIn("sin(iu)=i sinh(u)", bundle.solve_context())

    def test_english_hyperbolic_sine_product_has_high_confidence(self):
        route = classify_subjects(
            "Derive the hyperbolic sine infinite product from the Weierstrass sine product."
        )

        self.assertEqual(route.primary, "复分析")
        self.assertEqual(route.confidence, "high")
        self.assertIn("复分析:weierstrass-sine-product", route.matched_signals)


class PolyharmonicFundamentalSolutionRoutingTest(unittest.TestCase):
    def test_mathbb_r2_is_pde_not_regression_and_shape_is_expression(self):
        problem = (
            r"Find a radial fundamental solution of \Delta^2 on \mathbb R^2. "
            "The proof must determine the distributional normalization by a flux calculation."
        )
        spec = build_problem_spec(problem)
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "偏微分方程")
        self.assertNotEqual(spec.profile.subject, "线性回归")
        self.assertNotIn("线性回归:regression-diagnostics", spec.profile.matched_signals)
        self.assertEqual(
            spec.profile.topic,
            "two_dimensional_polyharmonic_fundamental_solution",
        )
        self.assertEqual(spec.profile.answer_shape, "expression")
        self.assertEqual(len(spec.goals), 1)
        self.assertIn("distributional normalization", spec.goals[0].instruction)
        self.assertEqual(
            spec.primary_method,
            "radial_laplacian_recurrence_and_flux_normalization",
        )
        self.assertEqual(
            spec.alternative_method,
            "fourier_symbol_and_distributional_constant_check",
        )
        self.assertEqual(
            bundle.solve_cards[0].id,
            "method.pde.polyharmonic_fundamental_solution",
        )

    def test_chinese_required_normalization_stays_attached(self):
        problem = (
            r"在二维全空间中求多调和算子 \Delta^m 的径向基本解。"
            "必须通过小圆通量说明分布意义下的归一化系数。"
        )
        spec = build_problem_spec(problem)

        self.assertEqual(spec.profile.subject, "偏微分方程")
        self.assertEqual(spec.profile.answer_shape, "expression")
        self.assertEqual(len(spec.goals), 1)
        self.assertIn("归一化系数", spec.goals[0].instruction)
        self.assertIn("reasoning", spec.answer_contract.support_requirements)


class AdvancedFamilyNegativeRoutingTest(unittest.TestCase):
    def test_ordinary_geometry_and_euler_ode_do_not_cross_route(self):
        geometry = build_problem_spec(
            "Prove that a cyclic quadrilateral inscribed in a circle has supplementary opposite angles."
        )
        ode = build_problem_spec(
            "Use Euler's method with step size h to approximate the initial value problem y'=y."
        )

        self.assertEqual(geometry.profile.subject, "初等几何")
        self.assertEqual(geometry.profile.topic, "olympiad_geometry")
        self.assertNotEqual(geometry.primary_method, "best_theorem_with_fixed_arc_normalization")
        self.assertNotEqual(ode.profile.topic, "directed_euler_circuits")
        self.assertNotEqual(ode.primary_method, "best_theorem_with_fixed_arc_normalization")

    def test_unit_circle_alone_and_generic_rooted_tree_do_not_get_special_cards(self):
        contour = build_problem_spec(
            r"Evaluate \int_{|z|=1} z^2\,dz around the unit circle."
        )
        tree = build_problem_spec(
            "A rooted tree is drawn in the plane. Determine its height from the displayed levels."
        )
        retriever = CardRetriever()
        contour_ids = {card.id for card in retriever.retrieve(contour).solve_cards}
        tree_ids = {card.id for card in retriever.retrieve(tree).solve_cards}

        self.assertNotEqual(contour.profile.topic, "lacunary_natural_boundary")
        self.assertNotIn("method.complex.lacunary_natural_boundary", contour_ids)
        self.assertNotEqual(tree.profile.topic, "plane_rooted_tree_enumeration")
        self.assertNotIn("method.combinatorics.plane_rooted_trees", tree_ids)

    def test_runge_kutta_without_stability_does_not_claim_the_stability_card(self):
        spec = build_problem_spec(
            "Apply the classical Runge-Kutta method for one step to y'=t+y."
        )
        bundle = CardRetriever().retrieve(spec)

        self.assertNotEqual(spec.profile.topic, "runge_kutta_stability")
        self.assertNotIn(
            "method.numerical.runge_kutta_stability",
            {card.id for card in bundle.solve_cards},
        )

    def test_planar_triangle_does_not_get_spherical_area_route(self):
        spec = build_problem_spec(
            "Use the ordinary law of cosines to find the area of a planar triangle."
        )
        bundle = CardRetriever().retrieve(spec)

        self.assertNotEqual(spec.profile.topic, "spherical_triangle_area")
        self.assertNotIn(
            "method.geometry.spherical_triangle_area",
            {card.id for card in bundle.solve_cards},
        )

    def test_weierstrass_approximation_and_generic_product_do_not_cross_route(self):
        approximation = build_problem_spec(
            "State the Weierstrass approximation theorem for continuous functions."
        )
        product = build_problem_spec(
            "Determine whether the infinite product of (1+1/n^2) converges."
        )
        retriever = CardRetriever()

        for spec in (approximation, product):
            with self.subTest(problem=spec.problem_text):
                ids = {card.id for card in retriever.retrieve(spec).solve_cards}
                self.assertNotEqual(spec.profile.topic, "weierstrass_sine_product")
                self.assertNotIn("method.complex.weierstrass_sine_product", ids)

    def test_mathbb_r2_alone_is_not_regression_and_poisson_is_not_polyharmonic(self):
        vector_space = classify_subjects(r"Let x belong to \mathbb R^2 and compute its norm.")
        poisson = build_problem_spec(
            r"Find a fundamental solution of the Poisson equation on \mathbb R^2."
        )

        self.assertNotEqual(vector_space.primary, "线性回归")
        self.assertFalse(any(
            signal == "线性回归:regression-diagnostics"
            for signal in vector_space.matched_signals
        ))
        self.assertNotEqual(
            poisson.profile.topic,
            "two_dimensional_polyharmonic_fundamental_solution",
        )


if __name__ == "__main__":
    unittest.main()
