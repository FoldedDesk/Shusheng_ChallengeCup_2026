from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import Requirement, build_problem_spec
from classifier.subject import classify_subjects
from rag.card_retriever import CardRetriever


class PrimaryTaskContractTest(unittest.TestCase):
    def test_result_request_stays_primary_when_reason_is_also_requested(self):
        cases = (
            "计算 2+3，并说明理由。",
            "Compute 2+3 and justify your answer.",
            "求 x^2=1 的全部实根并证明没有其他根。",
            "Find all real roots of x^2=1 and prove there are no others.",
        )
        for problem in cases:
            with self.subTest(problem=problem):
                spec = build_problem_spec(problem)
                self.assertEqual(spec.profile.task_kind, "calculation")
                self.assertEqual(spec.profile.problem_type, "calculation")
                self.assertEqual(spec.profile.result_kind, spec.profile.answer_shape)
                self.assertEqual(spec.answer_contract.mode, "answer_with_support")
                reasoning = [
                    item
                    for goal in spec.goals
                    for item in goal.requirements
                    if item.name == "reasoning"
                ]
                self.assertTrue(reasoning)
                self.assertTrue(all(item.category == "support" for item in reasoning))

    def test_context_before_a_verdict_does_not_turn_it_into_an_explanation(self):
        for problem in (
            "设Jacobi迭代矩阵B的无穷范数为0.6，说明迭代是否收敛并给出所用充分条件。",
            "在多项式环F_2[x]中判断x^3+x+1是否不可约，并说明理由。",
        ):
            with self.subTest(problem=problem):
                profile = build_problem_spec(problem).profile
                self.assertEqual(profile.task_kind, "calculation")
                self.assertEqual(profile.result_kind, "truth")

    def test_value_plus_explanation_keeps_the_value_as_primary_result(self):
        integral = build_problem_spec(
            "计算围道|z|=2上积分∮ dz/(z-1)，说明极点是否位于围道内。"
        )
        operator = build_problem_spec(
            "设T为右移算子，求||T||并说明是否为等距。"
        )

        self.assertEqual(integral.profile.result_kind, "expression")
        self.assertIn("pole_location", integral.answer_contract.support_requirements)
        self.assertEqual(operator.profile.result_kind, "expression")
        self.assertTrue({"operator_norm", "judgement"}.issubset(
            set(operator.answer_contract.parts[0].result_requirements)
        ))

    def test_pure_proof_and_explanation_keep_their_task_kind(self):
        cases = (
            ("证明素数有无穷多个。", "proof"),
            ("给出素数有无穷多个的证明。", "proof"),
            ("Prove that there are infinitely many primes.", "proof"),
            ("Give a proof that there are infinitely many primes.", "proof"),
            ("说明为什么连续函数在闭区间上可积。", "explanation"),
            ("Explain why every polynomial is continuous.", "explanation"),
        )
        for problem, expected in cases:
            with self.subTest(problem=problem):
                profile = build_problem_spec(problem).profile
                self.assertEqual(profile.task_kind, expected)
                self.assertEqual(profile.problem_type, expected)

    def test_result_kind_is_explicit_and_answer_shape_remains_compatible(self):
        number = build_problem_spec(
            "Find the number of functions from a 3-element set to a 2-element set."
        ).profile
        roots = build_problem_spec("Find the roots of the polynomial x^2-1.").profile

        self.assertEqual((number.result_kind, number.answer_shape), ("number", "number"))
        self.assertEqual((roots.result_kind, roots.answer_shape), ("roots", "roots"))

    def test_construction_is_a_first_class_task_kind(self):
        for problem in (
            "构造两个边缘均为Bernoulli(1/2)但不独立的随机变量。",
            "Construct an example of two dependent Bernoulli random variables.",
        ):
            with self.subTest(problem=problem):
                self.assertEqual(build_problem_spec(problem).profile.task_kind, "construction")

    def test_requirements_are_partitioned_without_breaking_legacy_tuple(self):
        spec = build_problem_spec(
            "从初值 x_0=1 出发，用牛顿法求 x^2-2=0 的迭代公式和第一次迭代值，并说明理由。"
        )
        goal = spec.goals[0]
        categories = {item.name: item.category for item in goal.requirements}

        self.assertEqual(categories["iteration_formula"], "result")
        self.assertEqual(categories["first_iteration"], "result")
        self.assertEqual(categories["reasoning"], "support")
        self.assertIn("iteration_formula", spec.answer_contract.parts[0].result_requirements)
        self.assertIn("reasoning", spec.answer_contract.parts[0].support_requirements)
        self.assertEqual(
            Requirement("boxed_wrapper", (("boxed",),)).category,
            "format",
        )
        boxed = build_problem_spec(
            r"Compute 2+3. Put the final answer in \boxed{}."
        )
        self.assertIn("boxed_wrapper", boxed.answer_contract.parts[0].format_requirements)


class WeightedSubjectRoutingTest(unittest.TestCase):
    def test_profile_exposes_ranked_bilingual_subject_evidence(self):
        cases = (
            "Use Newton method to find the roots of a monic polynomial.",
            "使用牛顿法求多项式的根。",
        )
        for problem in cases:
            with self.subTest(problem=problem):
                profile = build_problem_spec(problem).profile
                self.assertEqual(profile.subject, "数值分析")
                self.assertEqual(profile.primary, "数值分析")
                self.assertEqual(profile.secondary, "高等代数")
                self.assertEqual(profile.subject_confidence, "medium")
                self.assertTrue(profile.matched_signals)
                self.assertIn("数值分析:root-method", profile.matched_signals)

    def test_function_value_and_function_count_are_not_functional_equations(self):
        value = build_problem_spec("Given f(x)=x^2, compute f(3).").profile
        count = build_problem_spec(
            "Find the number of functions from a 3-element set to a 2-element set."
        ).profile
        equation = build_problem_spec(
            "Find all functions f such that f(x+y)=f(x)+f(y)."
        ).profile

        self.assertEqual(value.topic, "general")
        self.assertNotEqual(value.subject, "高等代数")
        self.assertEqual(count.subject, "离散数学")
        self.assertEqual(count.topic, "general")
        self.assertEqual(equation.topic, "olympiad_functional_equation")
        self.assertEqual(equation.subject, "高等代数")

    def test_weighted_classifier_returns_scores_and_secondary_subject(self):
        route = classify_subjects(
            "Use a finite-difference method for the heat equation."
        )

        self.assertIn(route.primary, {"偏微分方程", "数值分析"})
        self.assertIn(route.secondary, {"偏微分方程", "数值分析"})
        self.assertNotEqual(route.primary, route.secondary)
        self.assertEqual(route.confidence, "low")
        self.assertGreaterEqual(len(route.scores), 2)


class MultiDomainRagRoutingTest(unittest.TestCase):
    def test_compound_knowledge_file_keeps_every_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "数论与代数进阶.txt"
            path.write_text(
                "Finite field multiplicative groups are cyclic.\n",
                encoding="utf-8",
            )
            retriever = CardRetriever(Path(directory))
            card = next(item for item in retriever.cards if item.id.startswith("note."))

        self.assertEqual(card.domain, "数论")
        self.assertEqual(card.effective_domains, ("数论", "抽象代数"))

    def test_low_confidence_domain_guess_alone_does_not_inject_a_card(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "数值分析.txt"
            path.write_text("A domain-specific theorem with no query terms.\n", encoding="utf-8")
            retriever = CardRetriever(Path(directory))
            spec = build_problem_spec("Compute q.")
            low_profile = replace(
                spec.profile,
                subject="数值分析",
                primary_subject="数值分析",
                subject_confidence="low",
            )
            high_profile = replace(low_profile, subject_confidence="high")

            low = retriever.retrieve(replace(spec, profile=low_profile))
            high = retriever.retrieve(replace(spec, profile=high_profile))

        self.assertFalse(any(card.id.startswith("note.") for card in low.solve_cards))
        self.assertTrue(any(card.id.startswith("note.") for card in high.solve_cards))
        self.assertEqual(high.primary_subject, "数值分析")
        self.assertEqual(high.subject_confidence, "high")

    def test_secondary_subject_participates_but_primary_remains_stronger(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "数值分析.txt").write_text(
                "Newton method computes polynomial roots.\n", encoding="utf-8"
            )
            (base / "抽象代数.txt").write_text(
                "Polynomial roots can be checked by substitution.\n", encoding="utf-8"
            )
            retriever = CardRetriever(base)
            spec = build_problem_spec(
                "Use Newton method to find the roots of a monic polynomial."
            )
            scores = {
                card.domain: score
                for score, card in retriever._score(spec, "en")
                if card.id.startswith("note.")
            }

        self.assertIn("数值分析", scores)
        self.assertIn("抽象代数", scores)
        self.assertGreater(scores["数值分析"], scores["抽象代数"])


if __name__ == "__main__":
    unittest.main()
