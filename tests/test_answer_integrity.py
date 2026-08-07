from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.stage_budget import plan_stage_budget
from reasoning.candidate_selector import ToolEvidence, assess_candidate, choose_candidate
from reasoning.finalizer import Finalizer
from user_agent import ReasoningAgent


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, **kwargs):
        return self.responses.pop(0)


class AnswerIntegrityTest(unittest.TestCase):
    def test_nested_boxed_answer_is_extracted(self):
        result = Finalizer.extract_result(r"推导。\boxed{x=\frac{1}{\sqrt{2}}}")

        self.assertTrue(result.valid)
        self.assertEqual(result.answer, r"x=\frac{1}{\sqrt{2}}")

    def test_placeholders_and_unclosed_latex_are_rejected(self):
        for text in ("Final Answer: check formatting", "【最终答案】完整结论", "【最终答案】...", "conclusion.", "check on constraints:", "`.", r"\boxed{x=\frac{1}{2}"):
            result = Finalizer.extract_result(text)
            self.assertFalse(result.valid, text)
            self.assertTrue(result.rejected_reasons)

    def test_line_anchored_label_ignores_prompt_echo_after_answer(self):
        raw = (
            "依据概率加法公式，P(A\\cup B)=2/3。\n"
            "【最终答案】2/3\n"
            "Check spacing in final line: `【最终答案】2/3`。"
        )

        result = Finalizer.extract_result(raw)

        self.assertTrue(result.valid)
        self.assertTrue(result.raw_has_meta)
        self.assertTrue(result.explicit_answer)
        self.assertEqual(result.answer, "2/3")

    def test_final_check_sentence_is_not_an_english_answer_label(self):
        result = Finalizer.extract_result("Thinking Process:\nFinal check on constraints:")

        self.assertFalse(result.valid)
        self.assertEqual(result.method, "meta_without_explicit_answer")

    def test_meta_without_explicit_answer_is_never_submittable(self):
        result = Finalizer.extract_result("Analysis: 先完成推导，再给出 response.")

        self.assertFalse(result.valid)
        self.assertEqual(result.rejected_reasons, ("meta_without_explicit_answer",))

    def test_legitimate_short_math_answers_remain_valid(self):
        for text in ("B", "-1", "x", "x=1"):
            self.assertTrue(Finalizer.extract_result(text).valid, text)

    def test_explicit_single_symbol_conclusion_is_not_rejected_as_short_proof(self):
        spec = build_problem_spec("设布尔代数中x+y=1且xy=0，化简表达式(x+z)(y+z)，并使用分配律说明。")

        self.assertTrue(assess_candidate("z", "review", spec, (), "label", (), True, True).accepted)

    def test_newton_formula_is_not_replaced_by_sympy_root(self):
        problem = "用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。"
        spec = build_problem_spec(problem)
        evidence = ReasoningAgent(RecordingClient([])).agent._tool_evidence(
            ["SymPy 方程解: x=-sqrt(3)，x=sqrt(3)"], spec
        )
        candidate = r"x_{n+1}=(x_n+3/x_n)/2，x_1=7/4"

        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(evidence[0].scope, "subexpression")
        self.assertTrue(assess_candidate(candidate, "solve", spec, evidence).accepted)
        self.assertFalse(assess_candidate("x=sqrt(3)", "sympy_verified", spec, evidence).complete_goals)

    def test_missing_multi_part_candidate_loses_to_complete_review(self):
        spec = build_problem_spec("令f_n=n·1_(0,1/n)，说明f_n逐点收敛到何函数，并计算其积分以说明不可直接交换极限。")
        partial = assess_candidate("逐点极限为0。", "solve", spec, ())
        complete = assess_candidate("逐点极限为0，积分恒为1。", "review", spec, ())

        self.assertTrue(partial.accepted)
        self.assertTrue(partial.coverage_uncertain)
        self.assertEqual(choose_candidate([partial, complete]).source, "review")

    def test_equation_in_a_pde_verification_is_not_a_root_problem(self):
        spec = build_problem_spec("对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解，需分别求时间和空间导数。")

        self.assertEqual(spec.profile.answer_shape, "truth")
        self.assertEqual(spec.goals[0].required_terms, ())

    def test_uncovered_but_structurally_complete_answer_avoids_fallback(self):
        spec = build_problem_spec("用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。")
        equivalent = "方程的正根为 sqrt(3)。"

        candidate = assess_candidate(equivalent, "solve", spec, ())
        self.assertTrue(candidate.accepted)
        self.assertTrue(candidate.coverage_uncertain)

    def test_repair_recovers_when_both_prior_candidates_are_scratchpads(self):
        client = RecordingClient([
            "Thinking Process: unfinished",
            "Analysis: still unfinished",
            "FINAL: 4",
        ])

        result = ReasoningAgent(client).solve("给出一个满足条件的构造。", {})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(next(item for item in result["trace"] if item["step"] == "selection")["content"]["source"], "repair")

    def test_review_receives_only_explicit_answer_from_meta_response(self):
        raw = "analysis follows\n【最终答案】x=2\nCheck formatting before submitting."

        self.assertEqual(ReasoningAgent(RecordingClient([])).agent._review_evidence(raw), "x=2")

    def test_equivalent_requirement_forms_are_covered(self):
        spec = build_problem_spec("用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。")
        candidate = assess_candidate("迭代公式为 x(k+1)=(x(k)+3/x(k))/2，第一次迭代值为7/4。", "solve", spec, ())

        self.assertTrue(candidate.complete_goals)

    def test_hard_problem_reserves_repair_budget(self):
        spec = build_problem_spec("证明紧致空间的闭子集紧致。")
        budget = plan_stage_budget(spec, False)

        self.assertTrue(budget.allow_review)
        self.assertTrue(budget.allow_repair)
        self.assertGreater(budget.repair_min_remaining_seconds, 0)

    def test_truncated_first_response_triggers_review_and_keeps_complete_answer(self):
        client = RecordingClient([
            r"由定义可知\boxed{紧致空间的闭子集仍紧致",
            "由闭子集的开覆盖扩展到母空间，利用紧致性取得有限子覆盖。\n\\boxed{紧致空间的闭子集紧致}",
        ])

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集仍紧致，并说明开覆盖限制到闭子集的步骤。", {})

        self.assertIn("闭子集紧致", result["final_response"])
        self.assertEqual(len(client.responses), 0)
        validation = next(item for item in result["trace"] if item["step"] == "validation")
        self.assertIn("unclosed_latex_brace", validation["content"]["solve"]["rejected_reasons"])


if __name__ == "__main__":
    unittest.main()
