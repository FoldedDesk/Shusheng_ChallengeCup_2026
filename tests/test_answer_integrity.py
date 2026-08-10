from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from core.stage_budget import plan_stage_budget
from reasoning.candidate_selector import ToolEvidence, assess_candidate, choose_candidate
from reasoning.finalizer import Finalizer
from reasoning.math_equivalence import equivalent_answers
from tools.latex_parser import normalize_latex
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

    def test_latex_normalization_preserves_final_inline_math_delimiter(self):
        answer = "由独立增量可得 $E[S_5]=0$，且 $\\operatorname{Var}(S_5)=5$"

        self.assertEqual(normalize_latex(answer), answer)
        self.assertEqual(normalize_latex("$x=1$"), "x=1")
        self.assertEqual(normalize_latex("$$x=1$$"), "x=1")
        self.assertFalse(Finalizer.validate_structure(Finalizer.extract_solution(answer)))

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

    def test_labelled_final_outranks_a_later_unlabelled_check_box(self):
        result = Finalizer.extract_result(
            r"FINAL: \boxed{42}" "\n" r"Check: \boxed{6}"
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.method, "label_boxed")
        self.assertEqual(result.answer, "42")

    def test_common_inline_answer_labels_are_extracted(self):
        cases = {
            "The final answer is 42": "42",
            "答案为42": "42",
            "**Final Answer:** 42": "42",
        }
        for raw, expected in cases.items():
            result = Finalizer.extract_result(raw)
            self.assertTrue(result.valid, raw)
            self.assertTrue(result.explicit_answer, raw)
            self.assertEqual(result.answer, expected)

    def test_unmatched_fences_groups_and_environments_are_rejected(self):
        cases = (
            "```latex\nx=2",
            "x=2\n```",
            "x=(1+2",
            "x=[1,2",
            r"x=1\end{align}",
            "x=",
        )
        for raw in cases:
            self.assertTrue(Finalizer.validate_structure(raw), raw)
            self.assertFalse(Finalizer.extract_result(raw).valid, raw)

    def test_escaped_currency_and_left_arrow_are_not_corrupted(self):
        self.assertNotIn(
            "unclosed_inline_math",
            Finalizer.validate_structure(r"The cost is \$5."),
        )
        self.assertEqual(normalize_latex(r"x\leftarrow y"), r"x\leftarrow y")

    def test_boxed_scratch_prose_is_checked_by_its_inner_semantics(self):
        spec = build_problem_spec(
            r"Find the value. Remember to put your final answer within \boxed{}."
        )
        candidate = assess_candidate(
            r"\boxed{Suppose one variable is positive and two are negative.}",
            "offline",
            spec,
            (),
        )

        self.assertFalse(candidate.shape_valid)
        self.assertFalse(candidate.accepted)

    def test_formal_proof_requires_reasoning_but_keeps_normal_proof_language(self):
        spec = build_problem_spec(
            r"Prove that every closed subset of a compact space is compact."
        )
        bare = assess_candidate("Every closed subset is compact.", "solve", spec, ())
        supported = assess_candidate(
            "Let us prove the claim. Since the complement is open, add it to any "
            "open cover. By compactness there is a finite subcover. This implies "
            "the closed subset is compact.",
            "solve",
            spec,
            (),
        )

        self.assertFalse(bare.accepted)
        self.assertIn("missing_proof_reasoning", bare.rejected_reasons)
        self.assertTrue(supported.accepted)
        self.assertFalse(Finalizer.contains_meta(supported.answer))

    def test_marker_only_line_preserves_all_multipart_answers(self):
        problem = (
            r"计算行列式；再求矩阵的迹。"
            r"Remember to put your final answer within \boxed{}."
        )
        spec = build_problem_spec(problem)
        agent = ReasoningAgent(RecordingClient([])).agent

        finalized = agent._finalize("FINAL:\ndet(A)=1\ntr(A)=2", spec)
        rendered = agent._render_submission(finalized.answer, spec, problem)

        self.assertEqual(len(spec.goals), 2)
        self.assertTrue(finalized.valid)
        self.assertIn("det(A)=1", rendered)
        self.assertIn("tr(A)=2", rendered)
        self.assertTrue(rendered.startswith(r"\boxed{"))

        boxed = agent._finalize(
            r"FINAL: \boxed{det(A)=1}" "\n" r"\boxed{tr(A)=2}",
            spec,
        )
        boxed_rendered = agent._render_submission(boxed.answer, spec, problem)
        self.assertTrue(boxed.valid)
        self.assertEqual(boxed_rendered.count(r"\boxed{"), 1)
        self.assertIn("det(A)=1", boxed_rendered)
        self.assertIn("tr(A)=2", boxed_rendered)

    def test_marker_only_line_preserves_single_goal_multiline_values(self):
        agent = ReasoningAgent(RecordingClient([])).agent
        matrix_problem = (
            r"写出矩阵A。Remember to put your final answer within \boxed{}."
        )
        matrix_spec = build_problem_spec(matrix_problem)
        matrix = agent._finalize(
            "FINAL:\n\\begin{pmatrix}1&0\\\\0&1\\end{pmatrix}",
            matrix_spec,
        )

        roots_problem = (
            r"解方程x^2-3x+2=0。Remember to put your final answer within \boxed{}."
        )
        roots_spec = build_problem_spec(roots_problem)
        roots = agent._finalize("FINAL:\nx=1\nx=2", roots_spec)

        self.assertTrue(matrix.valid)
        self.assertEqual(matrix.method, "tagged_multiline_body")
        self.assertIn(r"\end{pmatrix}", matrix.answer)
        self.assertTrue(roots.valid)
        self.assertEqual(roots.method, "tagged_multiline_body")
        self.assertIn("x=1", roots.answer)
        self.assertIn("x=2", roots.answer)

    def test_multipart_support_contract_boxes_every_conclusion(self):
        problem = (
            r"令f_n=n·1_(0,1/n)，说明f_n逐点收敛到何函数，"
            r"并计算其积分以说明不可直接交换极限。"
            r"Remember to put your final answer within \boxed{}."
        )
        spec = build_problem_spec(problem)
        agent = ReasoningAgent(RecordingClient([])).agent
        finalized = agent._finalize("FINAL:\n逐点极限=0\n积分=1", spec)
        rendered = agent._render_submission(finalized.answer, spec, problem)
        extracted = Finalizer.extract_result(rendered)

        self.assertEqual(spec.answer_contract.mode, "answer_with_support")
        self.assertTrue(finalized.valid)
        self.assertEqual(rendered.count(r"\boxed{"), 1)
        self.assertIn("逐点极限=0", extracted.answer)
        self.assertIn("积分=1", extracted.answer)

    def test_explicit_method_cannot_collapse_to_a_single_symbol(self):
        spec = build_problem_spec("设布尔代数中x+y=1且xy=0，化简表达式(x+z)(y+z)，并使用分配律说明。")

        self.assertFalse(assess_candidate("z", "review", spec, (), "label", (), True, True).accepted)
        self.assertTrue(assess_candidate(
            "由分配律，(x+z)(y+z)=xy+z=0+z=z。", "review", spec, (), "label", (), True, True,
        ).accepted)

    def test_newton_formula_is_not_replaced_by_sympy_root(self):
        problem = "用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。"
        spec = build_problem_spec(problem)
        agent = ReasoningAgent(RecordingClient([])).agent
        evidence = agent._tool_evidence(agent.sympy.results_for(problem), spec)
        candidate = r"x_{n+1}=(x_n+3/x_n)/2，x_1=7/4"

        self.assertFalse(spec.tool_can_answer_whole)
        self.assertEqual(evidence[0].scope, "subexpression")
        self.assertTrue(evidence[0].verified)
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
        complete = assess_candidate(
            "逐点极限为0，积分恒为1，因此两者极限不等，不能直接交换极限。",
            "review", spec, (),
        )

        self.assertFalse(partial.accepted)
        self.assertTrue(partial.coverage_uncertain)
        self.assertIn("missing_required_goal", partial.rejected_reasons)
        self.assertEqual(choose_candidate([partial, complete]).source, "review")

    def test_equation_in_a_pde_verification_is_not_a_root_problem(self):
        spec = build_problem_spec("对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解，需分别求时间和空间导数。")

        self.assertEqual(spec.profile.answer_shape, "truth")
        self.assertEqual(spec.goals[0].required_terms, ())
        self.assertTrue(any(item.name == "pde_time_space_derivatives" and item.strict for item in spec.goals[0].requirements))

    def test_bare_truth_answer_is_rendered_with_its_object(self):
        problem = "对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解。"
        agent = ReasoningAgent(RecordingClient(["【最终答案】是。"])).agent

        result = agent.solve(problem, {})

        self.assertIn("u_t=", result["final_response"])
        self.assertIn("u_{xx}=", result["final_response"])
        self.assertIn("是解", result["final_response"])
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
        self.assertTrue(assess_candidate("属于L^1，积分值为2。", "solve", spec, ()).shape_valid)
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
        self.assertFalse(candidate.accepted)
        self.assertTrue(candidate.degraded)
        self.assertTrue(candidate.coverage_uncertain)
        self.assertEqual(choose_candidate([candidate]), candidate)

    def test_repair_recovers_when_both_prior_candidates_are_scratchpads(self):
        client = RecordingClient(["Thinking Process: unfinished", "FINAL: 4"])

        result = ReasoningAgent(client).solve("某项统计量已知为四，求其数值。", {})

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
        bare = assess_candidate(r"\frac{\pi i}{2}", "solve", symbolic_spec, ())
        supported = assess_candidate(
            r"由柯西积分公式，积分值为\frac{\pi i}{2}。", "solve", symbolic_spec, ()
        )
        self.assertTrue(bare.accepted)
        self.assertFalse(all(bare.support_coverage))
        self.assertTrue(supported.accepted)
        self.assertTrue(all(supported.support_coverage))

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
        self.assertEqual(budget.repair_tokens, 2048)
        self.assertEqual(budget.solve_tokens, 8192)
        self.assertEqual(budget.review_tokens, 6144)
        self.assertTrue(budget.require_independent_review)
        self.assertEqual(budget.emergency_tokens, 1024)
        self.assertEqual(budget.max_calls, 4)

    def test_complete_high_risk_answer_is_independently_verified(self):
        client = RecordingClient([
            "【最终答案】取x=2，代入得2^2=4且2>0，满足条件。",
            "【校验】通过\n【最终答案】取x=2，代入得2^2=4且2>0，满足条件。",
        ])

        result = ReasoningAgent(client).solve("构造一个整数x，使x^2=4且x>0。", {})

        self.assertIn("x=2", result["final_response"])
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
            r"分离变量得 y=\frac{1}{1-x}，最大右侧存在区间为[0,1)。", "solve", spec, ()
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

    def test_same_stage_variants_do_not_outvote_an_independent_verifier(self):
        spec = build_problem_spec("求这个数值。")
        solve = assess_candidate("42", "solve", spec, ())
        solve_variant = assess_candidate("6*7", "solve#2", spec, ())
        verify = assess_candidate("43", "verify", spec, ())

        self.assertEqual(choose_candidate([solve, solve_variant, verify]).source, "verify")

    def test_novel_arbitration_candidate_has_no_priority(self):
        spec = build_problem_spec("求这个数值。")
        solve = assess_candidate("42", "solve", spec, ())
        verify = assess_candidate("43", "verify", spec, ())
        arbitration = assess_candidate("44", "arbitration#2", spec, ())

        self.assertEqual(
            choose_candidate([solve, verify, arbitration]).source,
            "verify",
        )

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

    def test_reordered_lambda_polynomial_is_equivalent(self):
        ascending = r"Z(\lambda)=1+4\lambda+3\lambda^{2}+\lambda^{3}"
        descending = r"Z(\lambda)=\lambda^{3}+3\lambda^{2}+4\lambda+1"

        self.assertTrue(equivalent_answers(ascending, descending))

    def test_different_lambda_polynomial_is_not_equivalent(self):
        expected = r"Z(\lambda)=1+4\lambda+3\lambda^{2}+\lambda^{3}"
        different_coefficient = r"Z(\lambda)=\lambda^{3}+2\lambda^{2}+4\lambda+1"

        self.assertFalse(equivalent_answers(expected, different_coefficient))

    def test_equivalent_bare_judgements_are_normalized(self):
        self.assertTrue(equivalent_answers(r"\boxed{否}", "错误"))
        self.assertTrue(equivalent_answers("是", "正确"))
        self.assertFalse(equivalent_answers("是", "错误"))

    def test_formal_adjoint_product_rule_forms_are_equivalent(self):
        divergence = (
            r"L^*v=\sum_{i,j=1}^n\partial_j(a_{ij}\partial_i v)"
            r"-\sum_{j=1}^n\partial_j(b_jv)+cv"
        )
        expanded = (
            r"L^*v=\sum_{i,j=1}^n\partial_j(a_{ij}\partial_i v)"
            r"-\sum_{j=1}^n b_j\partial_jv"
            r"-\sum_{j=1}^n(\partial_jb_j)v+cv"
        )

        self.assertTrue(equivalent_answers(divergence, expanded))

    def test_indeterminate_variance_direction_wordings_are_equivalent(self):
        self.assertTrue(equivalent_answers(
            "参数估计量的方差不一定增大。",
            "传统方差估计可能低估或高估真实方差。",
        ))

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

    def test_answer_first_proof_keeps_the_support_after_the_box(self):
        raw = (
            r"FINAL: \boxed{Every closed subset is compact.}" "\n"
            "Let F be closed. Add its open complement to an open cover of F; "
            "compactness gives a finite subcover, hence F is compact."
        )
        spec = build_problem_spec(
            r"Prove that every closed subset of a compact space is compact. "
            r"Put your final answer in \boxed{}."
        )

        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)
        rendered = ReasoningAgent(RecordingClient([])).agent._render_submission(
            result.answer, spec, spec.goals[0].instruction,
        )

        self.assertTrue(result.valid)
        self.assertIn("finite subcover", result.answer)
        self.assertIn("Conclusion:", result.answer)
        self.assertNotIn("结论", result.answer)
        self.assertIn(r"\boxed{Every closed subset is compact.}", rendered)

    def test_answer_first_required_method_keeps_the_calculation(self):
        raw = (
            r"FINAL: \boxed{36}" "\n"
            r"By inclusion-exclusion, the number is $3^4-3\cdot2^4+3=36$."
        )
        spec = build_problem_spec(
            r"Find the number of surjections from a four-element set to a three-element set "
            r"using inclusion-exclusion. Put your final answer in \boxed{}."
        )

        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)

        self.assertTrue(result.valid)
        self.assertIn("inclusion-exclusion", result.answer)
        self.assertIn("3^4", result.answer)

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

    def test_uncertain_meta_and_internal_fields_are_rejected(self):
        for answer in (
            "a_n=3^{n-1} (or similar)",
            "证明：... 结论：... 依据：不等式成立",
            "g1: x-x^2/2；g2: x-x^2/2",
            "依据 [note.abstract.algebra.5] 可得商环为域。",
            "结论正确。 Looks compliant.",
            '结论正确。 Is there any risk of violating the prompt? So it is fine.',
            "积分结果 integral_result=1，积分极限 integral_value=0。",
            "expects just the value/expression. I'll write",
            "Count lines:\n1. 【最终答案】x=1\n2. Check line count",
            "结论：r\n\n[Explanation text]",
        ):
            self.assertTrue(Finalizer.validate_structure(answer), answer)

    def test_optional_note_residue_is_removed_from_a_tagged_answer(self):
        problem = (
            "对于时间序列的季节调整，常用的方法有( )、( )。"
            r"Remember to put your final answer within \boxed{}."
        )
        raw = (
            r"FINAL: \boxed{移动平均法、经典时间序列分解法}"
            "\n\n(Optional brief note)"
        )

        self.assertTrue(Finalizer.contains_meta(raw))
        self.assertIn("meta_text", Finalizer.validate_structure("optional note"))

        blocks = Finalizer.extract_tagged_submissions(raw)
        self.assertEqual(blocks, (r"FINAL: \boxed{移动平均法、经典时间序列分解法}",))

        result = ReasoningAgent(RecordingClient([])).agent._finalize(
            raw, build_problem_spec(problem)
        )
        self.assertTrue(result.valid)
        self.assertNotIn("Optional", result.answer)
        self.assertIn("移动平均法、经典时间序列分解法", result.answer)

    def test_random_walk_tagged_body_keeps_final_variance_line(self):
        raw = (
            "【最终答案】E[S_5]=0，Var(S_5)=5\n"
            "设 $X_i=S_i-S_{i-1}$ 为独立增量，$E[X_i]=0$ 且 $Var(X_i)=1$。\n"
            "由独立增量，$E[S_5]=\\sum_{i=1}^5E[X_i]=0$。\n"
            "$Var(S_5)=\\sum_{i=1}^5Var(X_i)=5$"
        )
        spec = build_problem_spec(
            "简单对称随机游走S_n从0出发，求E[S_5]与Var(S_5)，利用独立增量性质。"
        )

        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)

        self.assertTrue(result.valid)
        self.assertIn("独立增量", result.answer)
        self.assertIn("Var(S_5)", result.answer)
        self.assertTrue(result.answer.endswith("$"))

    def test_last_complete_tag_beats_an_earlier_hedged_draft(self):
        result = Finalizer.extract_result(
            "【最终答案】a_n=3^{n-1} (or similar)\n"
            "核验初值后修正。\n"
            "【最终答案】a_n=1"
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.answer, "a_n=1")

    def test_dangling_proof_body_falls_back_to_explicit_conclusion(self):
        spec = build_problem_spec(
            "设集合A有n个元素，求满足B⊆A且|B|为偶数的子集个数，并说明n≥1时为何两类子集数相等。"
        )
        result = ReasoningAgent(RecordingClient([])).agent._finalize(
            "【最终答案】偶数子集数为2^{n-1}，且奇偶子集数相等。\n\n设集合A有n个元素",
            spec,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.method, "tagged_answer_body")
        self.assertNotIn("设集合A", result.answer)

    def test_complementary_late_conclusion_does_not_replace_a_complete_proof_block(self):
        spec = build_problem_spec(
            "设集合A有n个元素，求满足B⊆A且|B|为偶数的子集个数，并说明n≥1时为何两类子集数相等。"
        )
        raw = (
            "Thinking Process: drafting.\n"
            "【最终答案】偶数子集数为2^{n-1}，奇偶子集数相等。\n"
            "由二项式定理，偶数项和与奇数项和之差为(1-1)^n=0，"
            "而二者之和为2^n，故各为2^{n-1}。\n"
            "结论：n≥1时二者相等。"
        )

        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)

        self.assertTrue(result.valid)
        self.assertIn("2^{n-1}", result.answer)
        self.assertIn("二项式定理", result.answer)

    def test_required_calculation_body_is_kept_and_a_cut_suffix_is_removed(self):
        spec = build_problem_spec(
            "求递推a_n=3a_{n-1}-2且a_1=1的通项，并利用不动点平移验证结果。"
        )
        raw = (
            "【最终答案】a_n=1\n"
            "求不动点得x=1，令b_n=a_n-1，则b_n=3b_{n-1}, b_1=0，故a_n=1。\n"
            "最后再验证 $n \\ge"
        )

        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)

        self.assertTrue(result.valid)
        self.assertEqual(result.method, "tagged_answer_body")
        self.assertIn("b_n=3b_{n-1}", result.answer)
        self.assertNotIn("最后再验证", result.answer)

    def test_nonproof_required_support_is_compacted_to_contract_complete_lines(self):
        cases = (
            (
                "求长度为10的二进制串中恰有4个1且不含相邻两个1的串数，要求先选取1的位置再计算。",
                r"""FINAL: \boxed{35}

**解题步骤：**
1. **问题描述：** 求长度为10且有4个1的二进制串数。
2. **方法：** 使用插空法（先选取1的位置，再计算）。
3. **步骤：** 将6个0排成一排，形成7个空位。
- 选择方法数为组合数 \(C(7,4)=35\)。
4. **结论：** 所求串数为35。""",
                ("35", "位置"),
            ),
            (
                "在5个不同元素的排列中求a在b之前且c不在首位的排列数，使用容斥或条件计数。",
                r"""FINAL: \boxed{48}
采用条件计数法。
1. 由对称性，a在b之前的排列数为60。
2. 固定c在首位时有12种。
3. 故所求排列数为60-12=48。""",
                ("48", "条件计数法"),
            ),
            (
                "求从集合{1,2,3,4}到{a,b,c}的满射个数，要求使用容斥原理而非直接枚举。",
                r"""FINAL: \boxed{36}

**步骤：**
1. **定义集合和映射：** 设S为所有映射的集合。
2. **使用容斥原理：**
3. **计算：** \(3^4-3\cdot2^4+3\cdot1^4=36\)。
因此满射个数为36。""",
                ("36", "容斥原理"),
            ),
        )
        agent = ReasoningAgent(RecordingClient([])).agent
        for problem, raw, required in cases:
            spec = build_problem_spec(problem)
            result = agent._finalize(raw, spec)

            self.assertTrue(result.valid, problem)
            self.assertTrue(assess_candidate(result.answer, "compact", spec, ()).accepted, result.answer)
            self.assertLess(len(result.answer), 240, result.answer)
            self.assertNotRegex(result.answer, r"\*\*|解题步骤|(?:^|\n)\s*步骤\s*[:：]")
            self.assertTrue(all(term in result.answer for term in required), result.answer)

    def test_english_required_support_is_compacted_without_step_headings(self):
        problem = (
            "Find the number of surjections from a four-element set to a "
            "three-element set using inclusion-exclusion."
        )
        raw = r"""FINAL: \boxed{36}

**Solution Steps:**
1. **Method:** Use inclusion-exclusion.
2. There are \(3^4\) total maps and three ways to omit a target.
3. Therefore \(3^4-3\cdot2^4+3=36\)."""

        spec = build_problem_spec(problem)
        result = ReasoningAgent(RecordingClient([])).agent._finalize(raw, spec)

        self.assertTrue(result.valid)
        self.assertTrue(assess_candidate(result.answer, "compact", spec, ()).accepted)
        self.assertLess(len(result.answer), 240)
        self.assertIn("inclusion-exclusion", result.answer.lower())
        self.assertNotRegex(result.answer.lower(), r"\*\*|solution steps|(?:^|\n)steps\s*:")

    def test_explicit_calculation_methods_do_not_override_result_correctness(self):
        cases = (
            ("求长度为10的二进制串中恰有4个1且不含相邻两个1的串数，要求先选取1的位置再计算。", "35"),
            ("求满足x_1+x_2+x_3+x_4=13且每个x_i为正整数、x_1≥3的解数，需通过变量平移化为隔板问题。", "120"),
            ("在5个不同元素的排列中求a在b之前且c不在首位的排列数，使用容斥或条件计数。", "48"),
            ("求从四元素集合到三元素集合的满射个数，要求使用容斥原理而非直接枚举。", "36"),
            ("求恰有3个不动点的7元对合置换数，先将其余元素配对。", "105"),
            ("求a+b+c=15且a≤b≤c的正整数解数，需要按a分类讨论。", "19"),
        )
        for problem, bare_answer in cases:
            spec = build_problem_spec(problem)
            rendered = SubmissionAgent._render_answer(bare_answer, spec)
            candidate = assess_candidate(rendered, "solve", spec, ())
            self.assertTrue(candidate.accepted, problem)
            self.assertFalse(all(candidate.support_coverage), problem)

    def test_recurrence_tool_rejects_a_conflicting_closed_form(self):
        problem = "求递推a_n=3a_{n-1}-2且a_1=1的通项，并利用不动点平移验证结果。"
        agent = ReasoningAgent(RecordingClient([])).agent
        spec = build_problem_spec(problem)
        evidence = agent._tool_evidence(agent.sympy.results_for(problem), spec)

        wrong = assess_candidate("不动点平移后 a_n=3^{n-1}。", "solve", spec, evidence)
        correct = assess_candidate("不动点为1，令b_n=a_n-1，则b_n=3b_{n-1}, b_1=0，故a_n=1。", "solve", spec, evidence)

        self.assertEqual(wrong.tool_status, "conflict")
        self.assertFalse(wrong.accepted)
        self.assertEqual(correct.tool_status, "partial_pass")
        self.assertTrue(correct.accepted)

    def test_exact_geometry_and_pde_tools_prevent_empty_fallbacks(self):
        problems_and_terms = (
            ("对曲线γ(t)=(cos t,sin t,t)，求速度长度并判断该参数是否为弧长参数。", ("sqrt(2)", "不是弧长参数")),
            ("对参数曲面X(u,v)=(u,v,u+v)，计算第一基本形式系数E,F,G。", ("E=2", "F=1", "G=2")),
            ("设曲面为图形z=f(x,y)，且在一点∇f=0，写出高斯曲率用Hessian行列式表示的公式。", ("K=", "f_{xy}^2")),
            ("对热方程u_t=u_{xx}，验证u(x,t)=e^{-t}sin x是否为解，需分别求时间和空间导数。", ("u_t=", "u_{xx}=", "是解")),
            ("求拉普拉斯方程u_{xx}+u_{yy}=0中函数u=x^2-y^2是否调和，并进行二阶求导。", ("u_{xx}=2", "u_{yy}=-2", "是调和")),
            ("求所有满足a+b+c=15、a≤b≤c且a,b,c为正整数的三元组个数，需要按a分类讨论。", ("按a分类", "共19个")),
            ("简单对称随机游走S_n从0出发，求E[S_5]与Var(S_5)，利用独立增量性质。", ("独立增量", "E[S_5]=0", "Var(S_5)=5")),
            ("设简单图有8个顶点且每个顶点度数至少4，证明该图必含长度为3的路径，并给出所用度数条件。", ("最长路径", "k≥4", "δ(G)≥4")),
            ("给定命题(p→q)∧(q→r)∧p，写出其合取范式下必然推出的最简结论，并说明推理规则。", ("假言推理", "得 q", "得 r")),
        )
        for problem, terms in problems_and_terms:
            result = ReasoningAgent(RecordingClient([])).solve(problem, {})
            self.assertNotEqual(result["final_response"], "未能生成可验证的数学答案。")
            self.assertTrue(all(term in result["final_response"] for term in terms), result["final_response"])
            selection = next(item for item in result["trace"] if item["step"] == "selection")
            self.assertEqual(selection["content"]["source"], "sympy_verified")

    def test_full_proof_beats_a_short_confirmed_summary(self):
        spec = build_problem_spec(
            "证明Hilbert空间中闭凸集上的最近点若存在则唯一，指出使用的严格凸性等式。"
        )
        short = assess_candidate(
            "最近点唯一，依据平行四边形恒等式。", "verify", spec, (),
            verification_verdict="confirmed",
        )
        full = assess_candidate(
            "设y_1,y_2均为最近点。由凸性其中点仍在集合中，且"
            "||u+v||^2+||u-v||^2=2(||u||^2+||v||^2)，故||y_1-y_2||=0，所以y_1=y_2。",
            "solve", spec, (),
        )

        self.assertFalse(short.accepted)
        self.assertTrue(full.accepted)
        self.assertEqual(choose_candidate([short, full]).source, "solve")


if __name__ == "__main__":
    unittest.main()
