from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from classifier.choice import answer_choice_labels
from core.submission_agent import SubmissionAgent
from rag.card_retriever import CardRetriever, KnowledgeCard, RetrievalBundle
from reasoning.candidate_selector import assess_candidate


class AnswerContractTest(unittest.TestCase):
    def test_truncation_continuation_preserves_explicit_support_contracts(self):
        chinese = build_problem_spec(
            "求从四元素集合到三元素集合的满射个数，要求使用容斥原理。"
        )
        english = build_problem_spec(
            "Construct two dependent Bernoulli(1/2) random variables and give P(X=Y)."
        )

        chinese_instruction = SubmissionAgent._continuation_instruction(chinese, True)
        english_instruction = SubmissionAgent._continuation_instruction(english, True)

        self.assertIn("方法、公式或核验", chinese_instruction)
        self.assertNotIn("该行之后不要输出推导", chinese_instruction)
        self.assertIn("every requested result or construction", english_instruction)
        self.assertNotIn("Output no derivation", english_instruction)

    def test_truncation_continuation_keeps_answer_only_results_short(self):
        chinese = build_problem_spec("计算2+3。")
        english = build_problem_spec("Calculate 2+3.")

        self.assertIn(
            "该行之后不要输出推导",
            SubmissionAgent._continuation_instruction(chinese, True),
        )
        self.assertIn(
            "Output no derivation",
            SubmissionAgent._continuation_instruction(english, True),
        )

    def test_boxed_suffix_is_not_a_semantic_goal(self):
        spec = build_problem_spec(
            r"Find all positive integers n such that n divides 2^n-1. "
            r"Put your final answer in \boxed{}."
        )

        self.assertEqual(spec.profile.subject, "数论")
        self.assertEqual(spec.profile.topic, "olympiad_number_theory")
        self.assertEqual(spec.profile.difficulty, "hard")
        self.assertFalse(spec.profile.tool_eligible)
        self.assertEqual(spec.answer_contract.language, "en")
        self.assertEqual(spec.answer_contract.mode, "answer_only")
        self.assertEqual(spec.answer_contract.wrapper, "boxed")
        self.assertNotIn("boxed", spec.goals[0].instruction.lower())
        self.assertNotIn("final answer", spec.goals[0].instruction.lower())
        self.assertIn("exhaustiveness_required", spec.risk_flags)
        self.assertIn("integer_constraints", spec.risk_flags)

    def test_show_final_answer_suffix_does_not_change_the_problem_type(self):
        spec = build_problem_spec(
            r"Find x if x+1=3. Please show your final answer in \boxed{}."
        )

        self.assertEqual(spec.profile.problem_type, "calculation")
        self.assertEqual(spec.answer_contract.mode, "answer_only")
        self.assertEqual(spec.goals[0].instruction, "Find x if x+1=3")

    def test_chinese_boxed_suffix_is_not_a_semantic_goal(self):
        spec = build_problem_spec(r"求 x+1=3 的解。请在\boxed{}中作答。")

        self.assertEqual(spec.profile.problem_type, "calculation")
        self.assertEqual(spec.answer_contract.language, "zh")
        self.assertEqual(spec.answer_contract.wrapper, "boxed")
        self.assertEqual(spec.goals[0].instruction, "求 x+1=3 的解")

    def test_justification_stays_attached_to_the_answer_part(self):
        spec = build_problem_spec(
            "Determine all real numbers x satisfying x^2=1 and justify your answer."
        )

        self.assertEqual(len(spec.goals), 1)
        self.assertIn("justify", spec.goals[0].instruction.lower())
        self.assertEqual(spec.answer_contract.mode, "answer_with_support")
        self.assertIn("reasoning", spec.answer_contract.explicit_support_requirements)
        self.assertIn("reasoning", spec.answer_contract.parts[0].support_requirements)

    def test_only_independent_asks_are_split(self):
        two_parts = build_problem_spec(
            "Find x from x+1=3 and determine y from 2y=6."
        )
        one_part = build_problem_spec(
            "Find all pairs (x,y) such that x>0 and y>0."
        )
        numbered = build_problem_spec(
            "Complete both parts: (a) Find x from x+1=3. (b) Prove that x is unique."
        )

        self.assertEqual(len(two_parts.goals), 2)
        self.assertEqual(len(one_part.goals), 1)
        self.assertEqual(len(numbered.goals), 2)
        self.assertEqual([part.id for part in numbered.answer_contract.parts], ["g1", "g2"])

    def test_parallel_named_results_add_requirements_without_splitting(self):
        cases = (
            (
                "总成本C(Q)=100+2Q+0.01Q^2，需求P=50-0.02Q。"
                "求利润最大化的产量Q、价格P以及最大利润。",
                {"production_quantity", "price_value", "profit_value"},
                "Q=800，P=34，最大利润为19100。",
                "最大利润为19100。",
            ),
            (
                "物体以初速度40m/s竖直上抛，求最大高度和到达最高点所需时间。",
                {"maximum_height", "time_to_peak"},
                "最大高度为80m，到达最高点所需时间为4s。",
                "最大高度为80m。",
            ),
            (
                "求线性规划max(3x+2y)在给定约束下的最优解与最优值。",
                {"optimal_solution", "optimal_value"},
                "最优解(x,y)=(2,3)，最优值为12。",
                "最优值为12。",
            ),
        )

        for problem, expected, complete, incomplete in cases:
            with self.subTest(problem=problem):
                spec = build_problem_spec(problem)
                self.assertEqual(len(spec.goals), 1)
                requirements = {
                    item.name: item for item in spec.goals[0].requirements
                }
                self.assertTrue(expected.issubset(requirements))
                self.assertTrue(all(
                    requirements[name].category == "result" for name in expected
                ))
                self.assertTrue(assess_candidate(
                    complete, "complete", spec, ()
                ).accepted)
                self.assertFalse(assess_candidate(
                    incomplete, "incomplete", spec, ()
                ).accepted)

    def test_parallel_result_requirements_accept_compact_equivalent_forms(self):
        normal = build_problem_spec(
            "设X,Y独立且均服从标准正态分布，求X+Y的分布及其方差。"
        )
        series = build_problem_spec(
            "展开1/(1-z)在z=0邻域的幂级数，并给出其收敛半径。"
        )
        equilibrium = build_problem_spec(
            "对线性系统x'=x,y'=-y，给出平衡点及其稳定性类型。"
        )
        production = build_problem_spec(
            "工厂生产A、B两种产品，求最大利润及对应的产量。"
        )
        radius_only = build_problem_spec(
            r"求幂级数\sum_{n=1}^{\infty}\frac{n}{2^n}x^n的收敛半径。"
        )

        self.assertTrue(assess_candidate("N(0,2)", "normal", normal, ()).accepted)
        self.assertTrue(assess_candidate(
            "Σ_{n=0}^∞z^n，收敛半径1", "series", series, ()
        ).accepted)
        self.assertTrue(assess_candidate(
            "原点为鞍点，不稳定。", "equilibrium", equilibrium, ()
        ).accepted)
        self.assertTrue(assess_candidate(
            "生产A 28件、B 36件，最大利润2200元。",
            "production",
            production,
            (),
        ).accepted)
        self.assertNotIn(
            "series_expansion",
            {item.name for item in radius_only.goals[0].requirements},
        )
        self.assertTrue(assess_candidate("2", "radius", radius_only, ()).accepted)

    def test_shared_stem_is_preserved_in_each_split_part(self):
        spec = build_problem_spec(
            "Let a and b be positive real numbers; find x from x+a=3; determine y from y+b=4."
        )

        self.assertEqual(len(spec.goals), 2)
        self.assertTrue(all("positive real numbers" in goal.instruction for goal in spec.goals))

    def test_supporting_clause_is_not_treated_as_an_independent_part(self):
        support = build_problem_spec(
            "Let A be strictly diagonally dominant; determine whether the iteration converges; "
            "give the supporting criterion."
        )
        independent = build_problem_spec(
            "Determine whether x is positive; calculate the value of x."
        )

        self.assertEqual(len(support.goals), 1)
        self.assertIn("supporting criterion", support.goals[0].instruction)
        self.assertEqual(len(independent.goals), 2)

    def test_geometry_contract_records_topic_and_unit(self):
        spec = build_problem_spec(
            r"In triangle ABC, its circumcircle is tangent to line l. "
            r"Find angle A in degrees. Express your answer in \boxed{}."
        )

        self.assertEqual(spec.profile.subject, "初等几何")
        self.assertEqual(spec.profile.topic, "olympiad_geometry")
        self.assertEqual(spec.answer_contract.unit, "degrees")
        self.assertEqual(spec.answer_contract.parts[0].unit, "degrees")
        self.assertNotIn("Express your answer", spec.goals[0].instruction)

    def test_elementary_calculation_is_not_promoted_to_olympiad(self):
        circle = build_problem_spec("Find the area of a circle of radius 2.")
        inequality = build_problem_spec("Solve the inequality x^2 < 1.")

        self.assertEqual(circle.profile.subject, "初等几何")
        self.assertEqual(circle.profile.topic, "general")
        self.assertTrue(circle.profile.tool_eligible)
        self.assertNotIn(
            "method.olympiad.geometry",
            {card.id for card in CardRetriever().retrieve(circle).solve_cards},
        )
        self.assertEqual(inequality.profile.topic, "general")
        self.assertEqual(inequality.profile.difficulty, "medium")

    def test_formal_proof_contract_requires_support(self):
        spec = build_problem_spec(
            r"Prove that there are infinitely many primes. "
            r"Your final answer should be written in \boxed{}."
        )

        self.assertEqual(spec.answer_contract.mode, "proof")
        self.assertEqual(spec.answer_contract.wrapper, "boxed")
        self.assertIn("reasoning", spec.answer_contract.support_requirements)
        self.assertNotIn("final answer", spec.goals[0].instruction.lower())

    def test_zero_integral_level_set_proof_requires_ae_zero_conclusion(self):
        cases = (
            (
                "若f≥0可测且∫f dμ=0，证明对任意ε>0集合{f≥ε}为零测集，"
                "并写出结论。",
                "由积分单调性可得 μ({f≥ε})=0。",
                "由积分单调性可得 μ({f≥ε})=0，从而 f=0 几乎处处。",
            ),
            (
                "Let f be a nonnegative measurable function with ∫ f dμ=0. "
                "Prove that every level set {f≥ε}, ε>0, has measure zero and "
                "state the conclusion.",
                "Because ε μ({f≥ε})≤∫f dμ=0, every such level set has measure zero.",
                "Because ε μ({f≥ε})≤∫f dμ=0, every such level set has measure zero; "
                "therefore f=0 almost everywhere.",
            ),
        )

        for problem, missing_conclusion, complete_answer in cases:
            with self.subTest(problem=problem):
                spec = build_problem_spec(problem)
                requirements = {
                    requirement.name
                    for goal in spec.goals
                    for requirement in goal.requirements
                }
                incomplete = assess_candidate(missing_conclusion, "solve", spec, ())
                complete = assess_candidate(complete_answer, "verify", spec, ())

                self.assertIn("almost_everywhere_zero", requirements)
                self.assertFalse(incomplete.accepted)
                self.assertIn("missing_required_goal", incomplete.rejected_reasons)
                self.assertTrue(complete.accepted)

    def test_conceptual_fill_blanks_accept_textual_results(self):
        cases = (
            ("在统计学中，用来表示数据分散程度的一个指标是", "标准差"),
            ("异方差性会导致参数估计量的方差（ ）", "可能被低估或高估"),
        )

        for problem, answer in cases:
            with self.subTest(problem=problem):
                spec = build_problem_spec(problem)
                assessment = assess_candidate(answer, "solve", spec, ())

                self.assertEqual(spec.profile.answer_shape, "expression")
                self.assertEqual(assessment.validation_tier, "complete")

        numeric_variance = build_problem_spec(
            "总体方差为1，独立样本量n=4，求样本均值的方差。"
        )
        self.assertEqual(numeric_variance.profile.answer_shape, "number")

    def test_euler_formula_verification_requires_a_checkable_equality(self):
        cases = (
            (
                "一个连通平面简单图有10个顶点和16条边，"
                "求其面数并验证欧拉公式。",
                "F=8，且 V-E+F=10-16+8=2。",
            ),
            (
                "A connected planar graph has 10 vertices and 16 edges. "
                "Find its number of faces and verify Euler formula.",
                "F=8; V-E+F=10-16+8=2.",
            ),
        )

        for problem, complete_answer in cases:
            spec = build_problem_spec(problem)
            names = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            self.assertIn("euler_formula_check", names)
            self.assertIn(
                "euler_formula_check", spec.answer_contract.parts[0].result_requirements
            )
            self.assertEqual(spec.answer_contract.mode, "answer_only")
            self.assertFalse(assess_candidate("F=8", "incomplete", spec, ()).accepted)
            self.assertTrue(assess_candidate(complete_answer, "complete", spec, ()).accepted)

    def test_exhaustive_count_accepts_explicit_total_phrasing(self):
        chinese = build_problem_spec(
            "求所有满足a+b+c=15、a≤b≤c且a,b,c为正整数的三元组个数，"
            "需要按a分类讨论。"
        )
        english = build_problem_spec(
            "Find all positive integer triples satisfying the constraints and report the total number."
        )

        for spec, accepted_phrases in (
            (chinese, ("共19个", "合计为19个", "总计19个")),
            (english, ("A total of 19 tuples.", "There are 19 tuples in total.")),
        ):
            exhaustive = next(
                requirement
                for goal in spec.goals
                for requirement in goal.requirements
                if requirement.name == "exhaustive_result"
            )
            self.assertFalse(exhaustive.matches("19"))
            for phrase in accepted_phrases:
                self.assertTrue(exhaustive.matches(phrase), phrase)

        self.assertTrue(assess_candidate(
            "按a分类讨论，合计19个。", "complete", chinese, ()
        ).accepted)
        self.assertTrue(assess_candidate(
            "A total of 19 tuples.", "complete", english, ()
        ).accepted)

    def test_curvature_contract_accepts_symbolic_labels_and_derivatives(self):
        chinese = build_problem_spec(
            "设曲面z=x^2+y^2，求原点处两条主曲率及高斯曲率，"
            "并写出二阶导数依据。"
        )
        english = build_problem_spec(
            "For the surface z=x^2+y^2, find both principal curvatures and the "
            "Gaussian curvature at the origin, and give the second partial derivatives used."
        )

        for spec in (chinese, english):
            names = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            self.assertTrue({
                "principal_curvatures", "gaussian_curvature", "surface_second_derivatives",
            }.issubset(names))
            self.assertFalse(any(
                term in {"主曲率", "高斯曲率"}
                for goal in spec.goals
                for term in goal.required_terms
            ))
            self.assertIn(
                "surface_second_derivatives",
                {
                    name
                    for part in spec.answer_contract.parts
                    for name in part.result_requirements
                },
            )

        symbolic = r"\kappa_1=\kappa_2=2,\ K=4;\ z_{xx}=z_{yy}=2,\ z_{xy}=0"
        named = (
            "The principal curvatures are 2 and 2; Gaussian curvature K=4; "
            "Hessian=diag(2,2)."
        )
        self.assertTrue(assess_candidate(symbolic, "complete", chinese, ()).accepted)
        self.assertTrue(assess_candidate(named, "complete", english, ()).accepted)
        self.assertFalse(assess_candidate(
            r"\kappa_1=\kappa_2=2,\ K=4", "missing_derivatives", chinese, ()
        ).accepted)
        self.assertFalse(assess_candidate(
            r"\kappa_1=\kappa_2=2;\ f_{xx}=f_{yy}=2,\ f_{xy}=0",
            "missing_gaussian", english, (),
        ).accepted)
        requirements = {
            requirement.name: requirement
            for goal in english.goals
            for requirement in goal.requirements
        }
        self.assertFalse(requirements["gaussian_curvature"].matches("Gaussian curvature"))
        self.assertFalse(requirements["surface_second_derivatives"].matches(
            "f_xx, f_xy, f_yy"
        ))

    def test_planar_curve_derivatives_do_not_require_surface_partials(self):
        chinese = build_problem_spec(
            "设平面曲线γ(t)=(t,t^2)，求t=0处曲率，需先计算一阶和二阶导数。"
        )
        english = build_problem_spec(
            "For the plane curve gamma(t)=(t,t^2), find the curvature at t=0 "
            "after computing the first and second derivatives."
        )

        for spec in (chinese, english):
            names = {
                requirement.name
                for goal in spec.goals
                for requirement in goal.requirements
            }
            self.assertIn("first_second_derivatives", names)
            self.assertNotIn("surface_second_derivatives", names)

        self.assertTrue(assess_candidate(
            "γ′(0)=(1,0)，γ′′(0)=(0,2)，κ(0)=2",
            "complete",
            chinese,
            (),
        ).accepted)
        self.assertFalse(assess_candidate(
            "κ(0)=2", "missing_derivatives", chinese, ()
        ).accepted)

    def test_irreducibility_assertions_are_valid_truth_results(self):
        chinese = build_problem_spec(
            "在多项式环F_2[x]中判断x^3+x+1是否不可约，说明只需检查何种次数的因子。"
        )
        english = build_problem_spec(
            "Determine whether x^3+x+1 is irreducible over F_2."
        )

        for spec, answer in (
            (chinese, r"x^3+x+1\text{ 在 }F_2[x]\text{ 中不可约，只需检查一次因子}"),
            (english, "x^3+x+1 is irreducible over F_2."),
        ):
            self.assertEqual(spec.profile.answer_shape, "truth")
            self.assertEqual(spec.answer_frame.style, "math")
            self.assertTrue(assess_candidate(answer, "complete", spec, ()).accepted)

    def test_conditional_sample_space_is_a_strict_enumerated_result(self):
        spec = build_problem_spec(
            "连续掷两次公平骰子，求点数和为7条件下第一次为3的条件概率，列出条件样本空间。"
        )
        english = build_problem_spec(
            "Roll two fair dice. Given that their sum is 7, find the probability that "
            "the first die is 3 and list the conditional sample space."
        )
        requirement = next(
            requirement
            for goal in spec.goals
            for requirement in goal.requirements
            if requirement.name == "conditional_sample_space"
        )

        self.assertTrue(requirement.strict)
        self.assertEqual(requirement.category, "result")
        self.assertIn(
            "conditional_sample_space",
            spec.answer_contract.parts[0].result_requirements,
        )
        self.assertFalse(requirement.matches("条件样本空间共有6个结果"))
        self.assertTrue(requirement.matches(
            r"\{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\}"
        ))
        english_requirement = next(
            requirement
            for goal in english.goals
            for requirement in goal.requirements
            if requirement.name == "conditional_sample_space"
        )
        self.assertTrue(english_requirement.matches(
            "Conditional sample space: {(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)}"
        ))
        self.assertFalse(assess_candidate(
            r"\frac{1}{6}", "probability_only", spec, ()
        ).accepted)
        self.assertTrue(assess_candidate(
            r"条件样本空间为\{(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)\}，"
            r"条件概率为\frac{1}{6}",
            "complete",
            spec,
            (),
        ).accepted)

    def test_variance_identification_requires_both_named_quantities(self):
        chinese = build_problem_spec(
            "设X服从参数p的伯努利分布，求E[(X-p)^2]并由定义识别其方差。"
        )
        english = build_problem_spec(
            "Let X be Bernoulli(p). Compute E[(X-p)^2] and identify it as the variance."
        )

        for spec in (chinese, english):
            requirement = next(
                requirement
                for goal in spec.goals
                for requirement in goal.requirements
                if requirement.name == "variance_identification"
            )
            self.assertTrue(requirement.strict)
            self.assertEqual(requirement.category, "result")
            self.assertFalse(assess_candidate(
                "p(1-p)", "value_only", spec, ()
            ).accepted)
            self.assertTrue(assess_candidate(
                r"E[(X-p)^2]=\operatorname{Var}(X)=p(1-p)",
                "complete",
                spec,
                (),
            ).accepted)

        requirement = next(
            requirement
            for goal in chinese.goals
            for requirement in goal.requirements
            if requirement.name == "variance_identification"
        )
        self.assertTrue(requirement.matches(
            "E[(X-p)^2]=p(1-p)，该值即为X的方差"
        ))
        self.assertFalse(requirement.matches("Var(X)=p(1-p)"))
        ordinary = build_problem_spec("若X服从Bernoulli(p)，求Var(X)。")
        self.assertNotIn(
            "variance_identification",
            {
                requirement.name
                for goal in ordinary.goals
                for requirement in goal.requirements
            },
        )

    def test_target_after_latex_period_does_not_inherit_definition_terms(self):
        spec = build_problem_spec(
            r"We call $(x,y)$ a good intersection if an edge exists $(1\leq x\leq n).$ "
            r"Find the minimum number of good insertions."
        )

        self.assertEqual(spec.goals[0].instruction, "Find the minimum number of good insertions")
        self.assertNotIn("intersection", {item.name for item in spec.goals[0].requirements})

    def test_for_which_values_is_not_reduced_to_a_yes_no_contract(self):
        spec = build_problem_spec(
            "For which values of m is it possible for the polygon to be circumscribed?"
        )

        self.assertEqual(spec.profile.answer_shape, "number")
        self.assertNotIn("judgement", {item.name for item in spec.goals[0].requirements})
        self.assertTrue(assess_candidate("m=6", "reference", spec, ()).accepted)

    def test_choice_answer_accepts_nested_math_wrappers_and_multiple_labels(self):
        self.assertEqual(answer_choice_labels(r"\(\boxed{B}\)"), ("B",))
        self.assertEqual(answer_choice_labels(r"\boxed{\text{BCD}}"), ("B", "C", "D"))

    def test_set_and_probability_arguments_are_not_choice_options(self):
        spec = build_problem_spec(
            "设A,B可测且μ(A)=3,μ(B)=5,μ(A∩B)=2，求μ(A∪B)并使用容斥公式。"
        )

        self.assertNotEqual(spec.profile.answer_shape, "choice")
        self.assertNotEqual(spec.profile.problem_type, "choice")

    def test_potential_multi_select_requires_every_correct_label(self):
        chinese = build_problem_spec(
            "下列关于平稳时间序列的说法正确的是："
            "A. 均值不随时间变化 B. 方差有限 C. 协方差只与时差有关 D. 必为独立序列"
        )
        english = build_problem_spec(
            "Select all that apply. A. 0 is even. B. 1 is even. C. 2 is even."
        )

        for spec in (chinese, english):
            requirements = {item.name: item for item in spec.goals[0].requirements}
            self.assertIn("all_correct_choices", requirements)
            self.assertTrue(requirements["all_correct_choices"].strict)
            self.assertTrue(requirements["all_correct_choices"].matches("AC"))
        self.assertIn("返回全部正确选项标签", chinese.goals[0].instruction)
        self.assertIn("不得按单选只保留一个标签", chinese.goals[0].instruction)
        self.assertIn("return every correct option label", english.goals[0].instruction)
        self.assertIn("do not assume single-choice", english.goals[0].instruction)
        self.assertIn("返回全部正确选项标签", SubmissionAgent._goal_context(chinese))
        self.assertIn("return every correct option label", SubmissionAgent._goal_context(english))

    def test_explicit_single_choice_does_not_get_select_all_contract(self):
        spec = build_problem_spec(
            "单项选择题：下列说法正确的是：A. 1=1 B. 1=2 C. 2=3 D. 3=4"
        )

        names = {item.name for item in spec.goals[0].requirements}
        self.assertNotIn("all_correct_choices", names)
        self.assertNotIn("返回全部正确选项标签", spec.goals[0].instruction)

    def test_three_fill_in_lines_become_three_verifiable_parts(self):
        problem = (
            r"$x^4+5\in\mathbb{Q}[x]$在$\mathbb{Q}$上的分裂域(记为$E$)是$(\quad)$." "\n"
            r"$[E:\mathbb{Q}]=(\quad)$." "\n"
            r"$E/\mathbb{Q}$ $(\quad)$(填“是”或“否”.)为Galois扩张."
        )
        spec = build_problem_spec(problem)
        complete = assess_candidate(
            r"\mathbb{Q}(5^{1/4},\zeta_8),\ 16,\ \text{是}",
            "reference",
            spec,
            (),
        )
        missing_degree = assess_candidate(
            r"\mathbb{Q}(5^{1/4},\zeta_8),\ \text{是}",
            "incomplete",
            spec,
            (),
        )

        self.assertEqual(len(spec.goals), 3)
        self.assertEqual([part.id for part in spec.answer_contract.parts], ["g1", "g2", "g3"])
        self.assertTrue(complete.accepted)
        self.assertFalse(missing_degree.accepted)
        self.assertEqual(missing_degree.goal_coverage, (True, False, True))

    def test_lz_contract_rejects_an_answer_missing_the_encoded_string(self):
        spec = build_problem_spec(
            "Describe the decomposition into phrases used by Lempel-Ziv, "
            "and give the encoded string obtained using Lempel-Ziv."
        )

        incomplete = assess_candidate(
            "The phrases are (0,a), (1,b), (2,a).",
            "incomplete",
            spec,
            (),
        )
        complete = assess_candidate(
            "The phrases are (0,a), (1,b), (2,a); encoded string: 000000 001001.",
            "complete",
            spec,
            (),
        )

        requirement_names = {
            item.name for goal in spec.goals for item in goal.requirements
        }
        self.assertIn("encoded_string", requirement_names)
        self.assertFalse(incomplete.accepted)
        self.assertTrue(complete.accepted)

    def test_lz_contract_accepts_a_labelled_phrase_list(self):
        spec = build_problem_spec(
            "Describe the decomposition into phrases used by Lempel-Ziv, "
            "and give the encoded string obtained using Lempel-Ziv."
        )
        labelled = assess_candidate(
            "Phrases: a, ab, aba, b, c; encoded string: 000000 001001.",
            "labelled",
            spec,
            (),
        )

        phrase_requirement = next(
            item
            for goal in spec.goals
            for item in goal.requirements
            if item.name == "phrase_decomposition"
        )
        self.assertTrue(phrase_requirement.matches("短语：a，ab，aba，b，c"))
        self.assertTrue(labelled.accepted)

    def test_two_blanks_require_two_answer_items(self):
        spec = build_problem_spec("对于时间序列的季节调整，常用的方法有( )、( )")

        self.assertFalse(assess_candidate("移动平均法", "one", spec, ()).accepted)
        self.assertTrue(assess_candidate("移动平均法、时间序列分解法", "two", spec, ()).accepted)
        context = CardRetriever().retrieve(spec).solve_context()
        self.assertIn("优先填“移动平均法、时间序列分解法”", context)

    def test_numbered_truth_question_keeps_the_judged_object(self):
        spec = build_problem_spec(
            "判断：5. 两个总量指标时间数列相比照得到的时间数列一定是相对数时间数列。"
        )

        self.assertIn("两个总量指标", spec.goals[0].instruction)
        self.assertEqual(spec.profile.answer_shape, "truth")


class HighConfidenceRagTest(unittest.TestCase):
    def test_verifier_is_candidate_blind_but_arbitration_can_use_curated_fact(self):
        card = KnowledgeCard(
            "fact.test", "theorem", "抽象代数", "换位子群为2阶。", ("换位子群",),
        )
        cards = RetrievalBundle((card,), (), (20,), (), "zh")
        spec = build_problem_spec("判断换位子群的阶数。")

        verification = SubmissionAgent._verification_request(
            "判断换位子群的阶数。", spec, cards, (), [],
        )
        arbitration = SubmissionAgent._arbitration_request(
            "判断换位子群的阶数。", spec, cards, [], (),
        )

        self.assertNotIn("换位子群为2阶", verification)
        self.assertIn("换位子群为2阶", arbitration)
        self.assertIn("必须使用", arbitration)

    def test_english_lz78_problem_gets_an_english_fact_card(self):
        bundle = CardRetriever().retrieve(build_problem_spec(
            "Describe the phrase decomposition used by LZ78 and give the encoded string."
        ))

        self.assertEqual(bundle.solve_cards[0].id, "fact.lz78.encoding")
        self.assertIn("longest dictionary prefix", bundle.solve_context())

    def test_spaced_pde_prompt_retrieves_discretization_facts(self):
        bundle = CardRetriever().retrieve(build_problem_spec(
            r"对于偏微分方程 D e l t a\;u=f，使用什么方法离散化？"
        ))

        self.assertTrue(bundle.solve_cards)
        self.assertIn("有限差分", bundle.solve_context())

    def test_olympiad_geometry_gets_one_specific_method_card(self):
        spec = build_problem_spec(
            "In triangle ABC, the incircle touches BC at D. Prove that two stated angles are equal."
        )
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(len(bundle.solve_cards), 1)
        self.assertEqual(bundle.solve_cards[0].id, "method.olympiad.geometry")
        self.assertGreaterEqual(bundle.solve_scores[0], 9)
        self.assertLessEqual(len(bundle.review_cards), 1)
        self.assertNotIn("method.olympiad.geometry", {card.id for card in bundle.review_cards})
        self.assertNotIn("method.olympiad.geometry", bundle.solve_context())

    def test_low_confidence_query_does_not_force_a_card(self):
        bundle = CardRetriever().retrieve(build_problem_spec("Calculate 2+2."))

        self.assertEqual(bundle.solve_cards, ())
        self.assertEqual(bundle.review_cards, ())

    def test_finite_field_note_remains_the_top_specific_card(self):
        bundle = CardRetriever().retrieve(
            build_problem_spec("设 F_81 为 81 元有限域，求生成整个扩张的元素个数。")
        )

        self.assertEqual(len(bundle.solve_cards), 1)
        self.assertIn("有限域", bundle.solve_cards[0].id)


if __name__ == "__main__":
    unittest.main()
