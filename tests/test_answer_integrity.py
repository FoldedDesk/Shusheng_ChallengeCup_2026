from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.stage_budget import plan_stage_budget
from reasoning.candidate_selector import ToolEvidence, assess_candidate, choose_candidate
from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
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

    def test_tagged_answer_inside_format_check_is_recovered(self):
        raw = (
            "Thinking Process:\nI need to check the required first line.\n"
            "Final check on first line: `【最终答案】是，积分值为 2。`"
        )

        result = Finalizer.extract_result(raw)

        self.assertTrue(result.valid)
        self.assertTrue(result.explicit_answer)
        self.assertEqual(result.method, "bracket_label")
        self.assertEqual(result.answer, "是，积分值为 2。")

    def test_tagged_placeholder_does_not_override_real_answer(self):
        raw = "【最终答案】x=2\nInstruction: use `【最终答案】<完整答案>`。"

        result = Finalizer.extract_result(raw)

        self.assertTrue(result.valid)
        self.assertEqual(result.answer, "x=2")

    def test_final_check_sentence_is_not_an_english_answer_label(self):
        result = Finalizer.extract_result("Thinking Process:\nFinal check on constraints:")

        self.assertFalse(result.valid)
        self.assertEqual(result.method, "meta_without_explicit_answer")

    def test_meta_without_explicit_answer_is_never_submittable(self):
        result = Finalizer.extract_result("Analysis: 先完成推导，再给出 response.")

        self.assertFalse(result.valid)
        self.assertEqual(result.rejected_reasons, ("meta_without_explicit_answer",))

    def test_quoted_prompt_fragment_is_not_recovered_as_answer(self):
        result = Finalizer.extract_result('I should output “计算题给结论和必要算式”。')

        self.assertFalse(result.valid)
        self.assertNotEqual(result.method, "quoted")

    def test_tail_math_conclusion_can_be_recovered_from_planning_text(self):
        result = Finalizer.extract_result(
            "所以，通项公式是 \\(a_n=1\\)。\n\n现在输出时，我应该整理格式。"
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.method, "tail_segment")
        self.assertIn("a_n=1", result.answer)

    def test_tail_recovery_rejects_plan_and_final_answer_content(self):
        for raw in (
            "Plan: 1. derive. 2. output x=2.",
            "Final Answer Content: `x=2`.",
            "Content for first line: `x=2`.",
            "Structure:\nLine 1: answer.\nLine 2: derivation.",
        ):
            self.assertFalse(Finalizer.extract_result(raw).valid, raw)

    def test_mathematical_inequalities_are_not_treated_as_markup(self):
        answer = r"因 m(E_1)=1<\infty，且 x>0，故交集测度为0。"

        self.assertNotIn("markup_fragment", Finalizer.validate_structure(answer))

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

    def test_all_generators_requires_an_enumeration(self):
        spec = build_problem_spec("在模12加法群中，求所有生成元并说明个数为何等于欧拉函数。")

        self.assertFalse(assess_candidate("生成元个数等于欧拉函数。", "solve", spec, ()).complete_goals)
        self.assertTrue(assess_candidate(
            "生成元为 {1,5,7,11}，共4个；因为它们恰为与12互素的剩余类，个数为φ(12)=4。",
            "solve", spec, (),
        ).complete_goals)

    def test_missing_multi_part_candidate_loses_to_complete_review(self):
        spec = build_problem_spec("令f_n=n·1_(0,1/n)，说明f_n逐点收敛到何函数，并计算其积分以说明不可直接交换极限。")
        partial = assess_candidate("逐点极限为0。", "solve", spec, ())
        complete = assess_candidate("逐点极限为0，积分恒为1。", "review", spec, ())

        self.assertFalse(partial.accepted)
        self.assertTrue(partial.coverage_uncertain)
        self.assertIn("missing_required_goal", partial.rejected_reasons)
        self.assertEqual(choose_candidate([partial, complete]).source, "review")

    def test_equation_in_a_pde_verification_is_not_a_root_problem(self):
        spec = build_problem_spec("对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解，需分别求时间和空间导数。")

        self.assertEqual(spec.profile.answer_shape, "truth")
        self.assertEqual(spec.goals[0].required_terms, ())

    def test_bare_truth_answer_is_rendered_with_its_object(self):
        problem = "对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解。"
        agent = ReasoningAgent(RecordingClient(["【最终答案】是。"])).agent

        result = agent.solve(problem, {})

        self.assertIn("u(x,t)=e^{-t}sin x", result["final_response"])
        self.assertTrue(result["final_response"].endswith("是。"))
        spec_trace = next(item for item in result["trace"] if item["step"] == "spec")
        self.assertTrue(spec_trace["content"]["answer_frame"]["subject"])

    def test_json_backspace_in_bar_command_is_restored(self):
        client = RecordingClient(["【最终答案】否。"])

        result = ReasoningAgent(client).solve("求函数f(z)=\x08ar z在z=0处是否复可导。", {})

        self.assertIn(r"\bar z", result["final_response"])
        self.assertNotIn("\x08", result["final_response"])

    def test_truth_summary_need_not_copy_the_whole_question_prefix(self):
        spec = build_problem_spec("给定f(x)=x^{-1/2}定义在(0,1]上，判断其是否属于L^1[0,1]并计算积分。")

        self.assertTrue(assess_candidate("是。积分值为2。", "solve", spec, ()).shape_valid)
        self.assertFalse(assess_candidate("是。", "solve", spec, ()).shape_valid)
        self.assertTrue(assess_candidate("是。积分值为2。", "solve", spec, ()).accepted)
        self.assertFalse(assess_candidate("给定函数属于L^1。", "solve", spec, ()).accepted)

    def test_missing_strict_integral_value_triggers_rescue(self):
        client = RecordingClient([
            "【最终答案】是。",
            "【最终答案】是。积分值为2。",
        ])

        result = ReasoningAgent(client).solve(
            "给定f(x)=x^{-1/2}定义在(0,1]上，判断其是否属于L^1[0,1]并计算积分。",
            {},
        )

        self.assertIn("积分值为2", result["final_response"])
        self.assertEqual(len(client.responses), 0)

    def test_lone_latex_escape_is_rejected(self):
        result = Finalizer.extract_result("【最终答案】\\")

        self.assertFalse(result.valid)
        self.assertIn("meaningless_fragment", result.rejected_reasons)

    def test_uncovered_but_structurally_complete_answer_avoids_fallback(self):
        spec = build_problem_spec("用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。")
        equivalent = "方程的正根为 sqrt(3)。"

        candidate = assess_candidate(equivalent, "solve", spec, ())
        self.assertTrue(candidate.accepted)
        self.assertTrue(candidate.coverage_uncertain)

    def test_repair_recovers_when_both_prior_candidates_are_scratchpads(self):
        client = RecordingClient(["Thinking Process: unfinished", "FINAL: 4"])

        result = ReasoningAgent(client).solve("给出一个满足条件的构造。", {})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(next(item for item in result["trace"] if item["step"] == "selection")["content"]["source"], "rescue")

    def test_review_receives_only_explicit_answer_from_meta_response(self):
        raw = "analysis follows\n【最终答案】x=2\nCheck formatting before submitting."

        self.assertEqual(ReasoningAgent(RecordingClient([])).agent._review_evidence(raw), "x=2")

    def test_equivalent_requirement_forms_are_covered(self):
        spec = build_problem_spec("用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。")
        candidate = assess_candidate("迭代公式为 x(k+1)=(x(k)+3/x(k))/2，第一次迭代值为7/4。", "solve", spec, ())

        self.assertTrue(candidate.complete_goals)

    def test_numerical_integral_comparison_need_not_repeat_the_word_integral(self):
        spec = build_problem_spec(
            "用复化梯形公式将[0,2]分为两段近似积分∫_0^2 x^2dx，求近似值并与精确值比较。"
        )
        candidate = assess_candidate("近似值为3，精确值为8/3，误差为1/3。", "solve", spec, ())

        self.assertTrue(candidate.complete_goals)

    def test_bare_numeric_integral_value_is_gradable(self):
        spec = build_problem_spec(
            "设μ为计数测度，求函数f(k)=1/2^k在正整数集合上的积分，并化为级数计算。"
        )

        for answer in ("1", "$1$", "级数值为1。", r"\frac{\pi i}{2}"):
            self.assertTrue(assess_candidate(answer, "solve", spec, ()).accepted, answer)

    def test_domain_specific_judgement_word_is_accepted(self):
        spec = build_problem_spec(
            "在多项式环F_2[x]中判断x^3+x+1是否不可约，说明只需检查何种次数的因子。"
        )
        candidate = assess_candidate(
            "该多项式不可约；三次多项式只需检查一次因子，且0、1均不是根。",
            "solve", spec, (),
        )

        self.assertTrue(candidate.accepted)
        self.assertTrue(candidate.complete_goals)

    def test_contour_location_and_symbolic_integral_are_gradable(self):
        location_spec = build_problem_spec(
            "计算围道|z|=2上积分∮ dz/(z-1)，说明留数定理中极点是否位于围道内。"
        )
        symbolic_spec = build_problem_spec(
            "求∮_{|z|=1} z^2/(z-1/2) dz，使用柯西积分公式而非直接参数化。"
        )

        self.assertTrue(assess_candidate(
            "极点位于围道内，积分值为2πi。", "solve", location_spec, ()
        ).accepted)
        self.assertTrue(assess_candidate(r"\frac{\pi i}{2}", "solve", symbolic_spec, ()).accepted)

    def test_regression_invariance_is_a_valid_judgement(self):
        spec = build_problem_spec(
            "简单线性回归含截距，若所有x_i同时加常数c，说明斜率估计是否改变并给出理由。"
        )
        candidate = assess_candidate(
            "斜率估计不变，因为平移后x_i-均值不变。", "solve", spec, ()
        )

        self.assertTrue(candidate.accepted)

    def test_stage_placeholders_and_goal_metadata_are_rejected(self):
        for text in (
            'adjudicated complete answer." Wait, check the prompt.',
            '<完整答案>，随后只写最短核验依据。',
            '完整答案，随后只写最短核验依据。',
            '结论成立。\ng1 [proof]: 关键依据为紧致性。',
        ):
            self.assertTrue(Finalizer.validate_structure(text), text)

    def test_hard_problem_reserves_repair_budget(self):
        spec = build_problem_spec("证明紧致空间的闭子集紧致。")
        budget = plan_stage_budget(spec, False)

        self.assertTrue(budget.allow_review)
        self.assertTrue(budget.allow_repair)
        self.assertEqual(budget.repair_tokens, 1536)
        self.assertEqual(budget.solve_tokens, 5120)

    def test_complete_high_risk_answer_is_independently_verified(self):
        client = RecordingClient([
            "【最终答案】x=2",
            "【校验】通过\n【最终答案】x=2",
        ])

        result = ReasoningAgent(client).solve("给出一个满足条件的构造。", {})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(len(client.responses), 0)
        admission = next(item for item in result["trace"] if item["step"] == "review_admission")["content"]
        self.assertTrue(admission["admitted"])
        self.assertEqual(admission["mode"], "verify")

    def test_two_invalid_rounds_trigger_a_short_last_chance(self):
        client = RecordingClient([
            "Thinking Process: unfinished",
            "Analysis: still unfinished",
            "【最终答案】极大元12，极小元1，最长链含4个元素。",
        ])
        result = ReasoningAgent(client).solve(
            "设偏序集为正整数12的全部正因子并按整除关系排序，求极大元、极小元及最长链长度。",
            {},
        )

        self.assertNotEqual(result["final_response"], "未能生成可验证的数学答案。")
        self.assertIn("极大元12", result["final_response"])
        equivalence = next(item["content"] for item in result["trace"] if item["step"] == "equivalence")
        self.assertEqual(equivalence["repair_mode"], "last_chance")

    def test_problem_goals_are_typed_and_split(self):
        spec = build_problem_spec(
            "给定f(x)=x^{-1/2}定义在(0,1]上，判断其是否属于L^1[0,1]并计算积分。"
        )

        self.assertEqual([goal.kind for goal in spec.goals], ["truth_judgement", "scalar_or_result"])
        self.assertEqual(len(spec.goals), 2)
        self.assertTrue(any(item.name == "integral_value" and item.strict for item in spec.goals[1].requirements))

    def test_determine_maximal_interval_is_not_forced_into_yes_no_shape(self):
        spec = build_problem_spec(
            "判断方程 y'=y^2,y(0)=1 的最大右侧存在区间，并通过分离变量求解。"
        )
        candidate = assess_candidate(
            r"y=\frac{1}{1-x}，最大右侧存在区间为[0,1)。", "solve", spec, ()
        )

        self.assertNotEqual(spec.goals[0].kind, "truth_judgement")
        self.assertFalse(any(item.name == "judgement" for item in spec.goals[0].requirements))
        self.assertEqual(spec.profile.subject, "常微分方程")
        self.assertEqual(spec.profile.answer_shape, "expression")
        self.assertTrue(spec.verification_required)
        self.assertTrue(candidate.accepted)

    def test_unmarked_placeholder_arbitration_cannot_override_a_correct_solution(self):
        spec = build_problem_spec("证明极大理想的商环是域。")
        solve = assess_candidate("由理想对应定理，商环只有平凡理想，故商环是域。", "solve", spec, ())
        fake = assess_candidate(
            '裁决后的完整答案。"". This looks like noise or a specific test case description.',
            "arbitration", spec, (), verification_verdict="",
        )

        self.assertFalse(fake.accepted)
        self.assertEqual(choose_candidate([solve, fake]).source, "solve")

    def test_echoed_answer_instruction_is_not_an_explicit_answer(self):
        result = Finalizer.extract_result(
            '【最终答案】并给出全部结论，再写必要依据。" This phrasing suggests the basis follows.'
        )

        self.assertFalse(result.valid)

    def test_richest_tagged_candidate_beats_a_later_truncated_draft(self):
        raw = (
            "Thinking Process: preparing the response.\n"
            "【最终答案】生成元为 1, 5, 7, 11，共4个，等于φ(12)。\n"
            "依据：Z_n中k为生成元当且仅当gcd(k,n)=1；故这四个数恰为全部生成元。\n"
            "This looks solid.\nWait, check formatting.\n"
            "【最终答案】生成元为 1,"
        )
        spec = build_problem_spec("在模12加法群Z_12中，求所有生成元并说明其个数为何等于欧拉函数φ(12)。")
        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)
        candidate = assess_candidate(result.answer, "solve", spec, ())

        self.assertTrue(result.valid)
        self.assertIn("1, 5, 7, 11", result.answer)
        self.assertIn("gcd", result.answer)
        self.assertNotIn("Thinking Process", result.answer)
        self.assertTrue(candidate.complete_goals)

    def test_equivalent_latex_forms_do_not_create_a_false_conflict(self):
        self.assertTrue(equivalent_answers("x=1/2", r"x=\frac{1}{2}"))
        self.assertFalse(equivalent_answers("x=1/2", "x=2"))

    def test_tagged_proof_block_drops_meta_preamble_but_keeps_argument(self):
        raw = (
            "Analysis: hidden planning.\n"
            "【最终答案】闭子集紧致。\n"
            "任取闭子集的开覆盖，补上其补集得到母空间的开覆盖；"
            "由紧致性取有限子覆盖，故原覆盖也有有限子覆盖。"
        )
        result = ReasoningAgent(RecordingClient([])).agent._finalize(
            raw, build_problem_spec("证明紧致空间的闭子集紧致。")
        )

        self.assertTrue(result.valid)
        self.assertNotIn("Analysis", result.answer)
        self.assertIn("有限子覆盖", result.answer)

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
