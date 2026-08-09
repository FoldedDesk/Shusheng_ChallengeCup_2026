from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from classifier.target import extract_target_clause
from reasoning.candidate_selector import assess_candidate


# Exact public prompts from the seven affected judge112 records.  Only these
# question strings are passed to the runtime contract builder.
QUESTIONS = {
    31: r"""The unit squares of an $2025 \times 2025$ chessboard are coloured alternately black and white, with the four corners coloured black. An L-tromino is a shape consisting of three unit squares connected in the shape of the letter L. A mysterious alien spaceship has landed on Earth and the aliens have brought with them a variety of L-trominos of various sizes and orientations. Is it possible to cover all the black squares with non-overlapping L-trominos? If it is possible, what is the minimum number of L-trominos needed?
Remember to put your final answer within \boxed{}.""",
    34: r"""Consider a checkerboard consisting of $38$ by $38$ unit squares. At the midpoints of some of these unit squares, there is an ant. At time 0, each ant starts moving with speed 1 parallel to some edge of the checkerboard. When two ants moving in opposite directions meet, they both turn $90^{\circ}$ clockwise and continue moving with speed 1. When more than two ants meet, or when two ants moving in perpendicular directions meet, the ants continue moving in the same direction as before they met. When an ant reaches one of the edges of the checkerboard, it falls off and will not re-appear. Also, there are some spiders on the checkerboard that do not move at all.

Considering all possible starting positions, determine the latest possible moment at which the last ant falls off the checkerboard or prove that such a moment does not necessarily exist.
Remember to put your final answer within \boxed{}.""",
    57: r"""Let $n,s,$ and $t$ be positive integers and $0<\lambda<1.$ A simple graph on $n$ vertices with at least $\lambda n^2$ edges is given. We say that $(x_1,\ldots,x_s,y_1,\ldots,y_t)$ is a good intersection if letters $x_i$ and $y_j$ denote not necessarily distinct vertices and every $x_iy_j$ is an edge of the graph $(1\leq i\leq s,$ $1\leq j\leq t).$ Find the minimum number of good insertions.

Remember to put your final answer within \boxed{}.""",
    64: r"""A convex $m$-gon $Q$, where $m > 3$, is divided into identical triangles by diagonals that do not intersect within it. For which values of $m$ is it possible for $Q$ to be circumscribed?

Remember to put your final answer within \boxed{}.""",
    86: r"""$x^4+5\in\mathbb{Q}[x]$在$\mathbb{Q}$上的分裂域(记为$E$)是$(\quad)$.
$[E:\mathbb{Q}]=(\quad)$.
$E/\mathbb{Q}$ $(\quad)$(填“是”或“否”.)为Galois扩张.
Remember to put your final answer within \boxed{}.""",
    95: r"""12. Consider the message

aabababcabcde.

Describe the decomposition into phrases that will be used by Lempel-Ziv, and give the encoded string obtained using Lempel-Ziv. When encoding a letter, use the mapping

\[
a\rightarrow000,\quad b\rightarrow001,\quad c\rightarrow010,\quad d\rightarrow011,\quad e\rightarrow100.
\]
Remember to put your final answer within \boxed{}.""",
    98: r"""7. 矩阵A 的条件数定义是:

A. \(\kappa(A)=\sqrt{|A|_{1}|A^{-1}|_{1}}\) B. \(\kappa(A)=|A|_{1}|A^{-1}|_{1}\) C. \(\kappa(A)=\sqrt{|A|_{2}|A^{-1}|_{2}}\) D. \(\kappa(A)=|A|_{2}|A^{-1}|_{2}\)
Remember to put your final answer within \boxed{}.""",
    104: r"""10．对于时间序列的季节调整，常用的方法有( )、( )
Remember to put your final answer within \boxed{}.""",
}


# Offline references are used only after ProblemSpec construction.
REFERENCES = {
    31: "1026169",
    34: "56",
    57: r"$\lambda^{st}n^{s+t}$",
    64: "4",
    86: r"\(\mathbb{Q}(5^{1/4},\zeta_8),\ 16,\ \text{是}\)",
    95: (
        r"(0,a),\ (1,b),\ (2,a),\ (1,c),\ (5,d),\ (0,e); "
        r"000000\,001001\,010000\,001010\,101011\,000100"
    ),
    98: r"\(\boxed{B}\)",
    104: "移动平均法、时间序列分解法",
}


class Judge112ContractRegressionTest(unittest.TestCase):
    @staticmethod
    def _spec(index: int):
        return build_problem_spec(QUESTIONS[index])

    @staticmethod
    def _assessment(index: int, answer: str):
        spec = Judge112ContractRegressionTest._spec(index)
        return assess_candidate(answer, "offline_reference", spec, ())

    def test_57_definition_does_not_create_intersection_obligation(self):
        spec = self._spec(57)

        self.assertEqual(extract_target_clause(QUESTIONS[57]).lower(), "find the minimum number of good insertions")
        self.assertNotIn("intersection", spec.goals[0].instruction.lower())
        self.assertNotIn("intersection", {item.name for item in spec.goals[0].requirements})
        self.assertTrue(self._assessment(57, REFERENCES[57]).accepted)

    def test_64_value_query_is_not_a_truth_contract(self):
        spec = self._spec(64)

        self.assertEqual(spec.profile.answer_shape, "number")
        self.assertNotIn("judgement", {item.name for item in spec.goals[0].requirements})
        self.assertTrue(self._assessment(64, REFERENCES[64]).accepted)
        self.assertFalse(self._assessment(64, "Yes").accepted)

    def test_98_latex_wrapped_choice_is_valid(self):
        spec = self._spec(98)
        assessment = self._assessment(98, REFERENCES[98])

        self.assertEqual(spec.profile.answer_shape, "choice")
        self.assertEqual(spec.goals[0].answer_shape, "choice")
        self.assertEqual(spec.goals[0].kind, "choice_selection")
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.shape_valid)

    def test_31_conditional_question_has_two_gradable_parts(self):
        spec = self._spec(31)

        self.assertEqual(len(spec.answer_contract.parts), 2)
        self.assertEqual(
            [part.validation_requirements for part in spec.answer_contract.parts],
            [("feasibility_or_numeric",), ("numeric_result",)],
        )
        self.assertTrue(self._assessment(31, REFERENCES[31]).accepted)
        self.assertFalse(self._assessment(31, "No").accepted)

    def test_34_result_or_nonexistence_alternative_accepts_numeric_result(self):
        spec = self._spec(34)

        self.assertEqual(spec.profile.problem_type, "calculation")
        self.assertEqual(spec.answer_contract.mode, "answer_only")
        self.assertEqual(spec.goals[0].kind, "alternative_result")
        self.assertTrue(self._assessment(34, REFERENCES[34]).accepted)

    def test_86_three_blanks_require_field_degree_and_verdict(self):
        spec = self._spec(86)

        self.assertEqual(len(spec.answer_contract.parts), 3)
        self.assertEqual([part.answer_shape for part in spec.answer_contract.parts], ["expression", "number", "truth"])
        self.assertTrue(self._assessment(86, REFERENCES[86]).accepted)
        self.assertFalse(self._assessment(86, r"\mathbb{Q}(5^{1/4},\zeta_8), 16").accepted)
        self.assertFalse(self._assessment(86, r"16, \text{是}").accepted)

    def test_95_requires_both_phrase_decomposition_and_encoding(self):
        spec = self._spec(95)

        self.assertEqual(len(spec.answer_contract.parts), 2)
        self.assertEqual(
            [part.validation_requirements for part in spec.answer_contract.parts],
            [("phrase_decomposition",), ("encoded_string",)],
        )
        self.assertTrue(self._assessment(95, REFERENCES[95]).accepted)
        self.assertFalse(self._assessment(95, "(0,a), (1,b), (2,a)").accepted)
        self.assertFalse(self._assessment(95, r"000000\,001001\,010000\,001010").accepted)

    def test_104_two_blanks_require_two_answer_items(self):
        spec = self._spec(104)

        self.assertEqual(len(spec.answer_contract.parts), 2)
        self.assertTrue(all(
            part.validation_requirements == ("two_items",)
            for part in spec.answer_contract.parts
        ))
        self.assertTrue(self._assessment(104, REFERENCES[104]).accepted)
        self.assertFalse(self._assessment(104, "移动平均法").accepted)

    def test_narrative_commands_do_not_create_false_subquestions(self):
        cases = (
            (
                "The researchers want to determine the smallest initial number k. "
                "What is the smallest such k that guarantees full infection?",
                "smallest such k",
            ),
            (
                "Player B must then give as many cookies as there are numbers. "
                "Player B wants to give as few as possible. Determine the number "
                "of cookies received under optimal play.",
                "determine the number",
            ),
            (
                "Apply v+w or max(v,w), and then write this tuple on the board. "
                "Minh can write any integer tuple. What is the smallest possible s?",
                "smallest possible s",
            ),
        )
        for problem, target in cases:
            spec = build_problem_spec(problem)
            self.assertEqual(len(spec.goals), 1, problem)
            self.assertIn(target, spec.goals[0].instruction.lower())

    def test_find_all_contract_requires_an_exhaustive_final_statement(self):
        spec = build_problem_spec(
            "Find all positive integers n satisfying the stated divisibility condition."
        )
        names = {item.name for item in spec.goals[0].requirements}

        self.assertIn("exhaustive_result", names)
        self.assertFalse(assess_candidate("2", "solve", spec, ()).accepted)
        self.assertTrue(assess_candidate("The only value is n=2.", "solve", spec, ()).accepted)
        self.assertTrue(assess_candidate("n=2,4,6; there are no others.", "solve", spec, ()).accepted)

    def test_displayed_condition_after_target_is_kept_in_the_goal(self):
        problem = (
            "Find the number of triples (x,y,z) of real numbers satisfying\n"
            r"\[x^2+y^2+z^2=xy^3+yz^3+zx^3=3.\]"
        )

        target = extract_target_clause(problem)
        spec = build_problem_spec(problem)

        self.assertIn("x^2+y^2+z^2", target)
        self.assertIn("xy^3+yz^3+zx^3", spec.goals[0].instruction)


if __name__ == "__main__":
    unittest.main()
