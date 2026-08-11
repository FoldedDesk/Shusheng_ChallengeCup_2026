from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.model_response import ModelCallResult
from core.stage_budget import plan_stage_budget
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate, choose_candidate
from user_agent import ReasoningAgent


class StructuredRecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_result(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class ScoreFirstPipelineTest(unittest.TestCase):
    @staticmethod
    def _step(result, name):
        return next(item for item in result["trace"] if item["step"] == name)

    def test_complete_boxed_prefix_survives_provider_truncated_suffix(self):
        client = StructuredRecordingClient([
            ModelCallResult(
                "FINAL: \\boxed{x=2}\nVerification: $x",
                finish_reason="length",
                usage={"completion_tokens": 4096},
            ),
        ])

        result = ReasoningAgent(client).solve("求 x。", {})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(len(client.calls), 1)
        call = self._step(result, "model_call_solve")["content"]
        self.assertEqual(call["finish_reason"], "length")
        self.assertTrue(call["provider_truncated"])
        self.assertEqual(call["truncation_signal"], "provider_length")
        self.assertEqual(call["usage"]["completion_tokens"], 4096)
        self.assertEqual(self._step(result, "selection")["content"]["source"], "solve")

    def test_truncated_unlabelled_box_is_not_treated_as_the_final_answer(self):
        client = StructuredRecordingClient([
            ModelCallResult(
                r"During a trial case we get \boxed{6}, and next consider",
                finish_reason="length",
            ),
            ModelCallResult(r"FINAL: \boxed{42}"),
            ModelCallResult(
                "FINAL: \\boxed{42}\nVERDICT: CONFIRMED\n"
                "CHECK: direct substitution gives 42=42."
            ),
        ])

        result = ReasoningAgent(client).solve(
            r"Find the value. Remember to put your final answer within \boxed{}.",
            {},
        )

        self.assertEqual(result["final_response"], r"\boxed{42}")
        self.assertEqual(len(client.calls), 3)
        validation = self._step(result, "validation")["content"]
        rejected = [
            reason
            for source, item in validation.items()
            if source.startswith("solve")
            for reason in item["rejected_reasons"]
        ]
        self.assertIn("provider_truncated_ambiguous_box", rejected)

    def test_truncated_correction_invalidates_an_earlier_labelled_answer(self):
        client = StructuredRecordingClient([
            ModelCallResult(
                r"FINAL: \boxed{x=1}" "\nCorrection:\n" r"FINAL: \boxed{x=",
                finish_reason="length",
            ),
            ModelCallResult(r"FINAL: \boxed{x=2}"),
            ModelCallResult(
                "FINAL: \\boxed{x=2}\nVERDICT: CONFIRMED\n"
                "CHECK: substituting x=2 satisfies the equation."
            ),
        ])

        result = ReasoningAgent(client).solve("求 x。", {})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            self._step(result, "review_admission")["content"]["mode"],
            "continue",
        )
        validation = self._step(result, "validation")["content"]
        solve_reasons = {
            reason
            for source, item in validation.items()
            if source.startswith("solve")
            for reason in item["rejected_reasons"]
        }
        self.assertIn("provider_truncated_ambiguous_box", solve_reasons)

    def test_provider_truncation_without_answer_continues_the_same_draft(self):
        client = StructuredRecordingClient([
            ModelCallResult("Analysis: derive the invariant and now consider", finish_reason="length"),
            ModelCallResult(r"the last case. Therefore FINAL: \boxed{x=2}"),
            ModelCallResult(
                "FINAL: \\boxed{x=2}\nVERDICT: CONFIRMED\n"
                "CHECK: substituting x=2 verifies the invariant."
            ),
        ])

        result = ReasoningAgent(client).solve("求 x。", {})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self._step(result, "review_admission")["content"]["mode"], "continue")
        messages = client.calls[1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertIn("derive the invariant", messages[2]["content"])
        self.assertIn("不要继续、重启或解释草稿", messages[3]["content"])
        self.assertFalse(client.calls[0]["thinking_mode"])
        self.assertFalse(client.calls[1]["thinking_mode"])
        self.assertEqual(client.calls[1]["max_tokens"], 1024)
        self.assertEqual(len(client.calls[2]["messages"]), 2)
        self.assertIn("x=2", client.calls[2]["messages"][1]["content"])
        self.assertFalse(client.calls[2]["thinking_mode"])
        self.assertEqual(
            self._step(result, "equivalence")["content"]["repair_mode"],
            "verify_recovered",
        )

    def test_structurally_cut_draft_uses_the_same_recovery_path(self):
        client = StructuredRecordingClient([
            ModelCallResult("Analysis: after substitution x =", finish_reason="stop"),
            ModelCallResult(r"FINAL: \boxed{x=2}"),
            ModelCallResult(
                "FINAL: \\boxed{x=2}\nVERDICT: CONFIRMED\n"
                "CHECK: direct substitution gives x=2."
            ),
        ])

        result = ReasoningAgent(client).solve("求 x。", {})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(self._step(result, "review_admission")["content"]["mode"], "continue")
        self.assertEqual(
            self._step(result, "model_call_solve")["content"]["truncation_signal"],
            "structural",
        )
        self.assertEqual(len(client.calls), 3)

    def test_twice_truncated_draft_gets_clean_context_last_chance(self):
        client = StructuredRecordingClient([
            ModelCallResult("Analysis: first half", finish_reason="length"),
            ModelCallResult("continuing: second half", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{7}"),
        ])

        result = ReasoningAgent(client).solve(
            r"Find the value. Remember to put your final answer within \boxed{}.", {}
        )

        self.assertEqual(result["final_response"], r"\boxed{7}")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self._step(result, "equivalence")["content"]["repair_mode"], "last_chance")
        self.assertEqual(len(client.calls[2]["messages"]), 2)
        self.assertNotIn("first half", client.calls[2]["messages"][1]["content"])
        self.assertNotIn("second half", client.calls[2]["messages"][1]["content"])
        self.assertFalse(client.calls[2]["thinking_mode"])

    def test_recovered_hard_answer_is_adversarially_audited_and_corrected(self):
        client = StructuredRecordingClient([
            ModelCallResult("Deep analysis: unfinished", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{9}"),
            ModelCallResult(
                "FINAL: \\boxed{45}\nVERDICT: CORRECTED\n"
                "CHECK: 45^2=2025 is divisible by 2025, while 9^2 is not."
            ),
        ])
        problem = (
            "Find the least positive integer n in this Diophantine divisibility problem "
            "such that n^2 is divisible by 2025. "
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(result["final_response"], r"\boxed{45}")
        self.assertTrue(client.calls[0]["thinking_mode"])
        self.assertFalse(client.calls[1]["thinking_mode"])
        self.assertEqual(client.calls[1]["max_tokens"], 2048)
        self.assertFalse(client.calls[2]["thinking_mode"])
        self.assertEqual(client.calls[2]["max_tokens"], 4096)
        self.assertEqual(len(client.calls[2]["messages"]), 2)
        self.assertIn("9", client.calls[2]["messages"][1]["content"])
        self.assertEqual(
            self._step(result, "model_call_solve")["content"]["routing_role"],
            "deep_reasoning",
        )
        self.assertEqual(
            self._step(result, "model_call_continue")["content"]["routing_role"],
            "quick_response",
        )
        self.assertEqual(
            self._step(result, "model_call_verify_recovered")["content"]["routing_role"],
            "quick_response",
        )
        selection = self._step(result, "selection")["content"]
        self.assertEqual(selection["source"], "verify_recovered")
        self.assertEqual(selection["recovered_answer_verification"], "complete")

    def test_truncated_independent_verifier_is_replaced_by_fresh_candidate_audit(self):
        client = StructuredRecordingClient([
            ModelCallResult(r"FINAL: \boxed{9}"),
            ModelCallResult("Independent deep calculation: unfinished", finish_reason="length"),
            ModelCallResult(
                "FINAL: \\boxed{45}\nVERDICT: CORRECTED\n"
                "CHECK: 45^2=2025 is divisible by 2025, while 9^2 is not."
            ),
        ])
        problem = (
            "Find the least positive integer n in this Diophantine divisibility problem "
            "such that n^2 is divisible by 2025. "
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(result["final_response"], r"\boxed{45}")
        self.assertTrue(client.calls[1]["thinking_mode"])
        self.assertFalse(client.calls[2]["thinking_mode"])
        self.assertEqual(client.calls[2]["max_tokens"], 4096)
        self.assertEqual(
            [message["role"] for message in client.calls[2]["messages"]],
            ["system", "user"],
        )
        self.assertNotIn("Independent deep calculation", client.calls[2]["messages"][1]["content"])
        self.assertIn("9", client.calls[2]["messages"][1]["content"])
        self.assertEqual(
            self._step(result, "equivalence")["content"]["repair_mode"],
            "retry_verify",
        )
        self.assertEqual(self._step(result, "selection")["content"]["source"], "retry_verify")

    def test_truncated_verifier_continuation_cannot_override_complete_primary(self):
        client = StructuredRecordingClient([
            ModelCallResult(
                r"FINAL: \boxed{120}" "\n令 $y_1=x_1-2$，则四个正整数之和为11，故有 $\binom{10}{3}=120$。"
            ),
            ModelCallResult(
                "Independent recount reaches 120, but the final check is $",
                finish_reason="length",
            ),
            ModelCallResult(
                r"FINAL: \boxed{35}" "\nYet the direct count gives $\binom{10}{3}=120$, and now $",
                finish_reason="length",
            ),
            ModelCallResult(
                "FINAL: \\boxed{120}\nVERDICT: CONFIRMED\n"
                "CHECK: the direct count is binom(10,3)=120."
            ),
        ])

        result = ReasoningAgent(client).solve(
            r"求满足\sum_{i=1}^4 x_i=13且每个x_i为正整数、"
            "x_1>=3的解数，需通过变量平移化为隔板问题。",
            {},
        )

        self.assertIn("120", result["final_response"])
        self.assertNotIn("35", result["final_response"])
        self.assertNotIn("VERDICT", result["final_response"])
        self.assertNotIn("CHECK:", result["final_response"])
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(
            self._step(result, "equivalence")["content"]["repair_mode"],
            "retry_verify",
        )
        validation = self._step(result, "validation")["content"]
        self.assertIn(
            "provider_truncated_ambiguous_box",
            validation["retry_verify"]["rejected_reasons"],
        )
        self.assertEqual(self._step(result, "selection")["content"]["source"], "audit_retry")

    def test_uncertified_recovered_answer_audit_gets_a_fresh_retry(self):
        client = StructuredRecordingClient([
            ModelCallResult("Primary deep analysis: unfinished", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{9}"),
            ModelCallResult("FINAL: \\boxed{45}\nVERDICT: CORRECTED"),
            ModelCallResult(
                "FINAL: \\boxed{45}\nVERDICT: CORRECTED\n"
                "CHECK: 45^2=2025 is divisible by 2025, while 9^2 is not."
            ),
        ])
        problem = (
            "Find the least positive integer n in this Diophantine divisibility problem "
            "such that n^2 is divisible by 2025. "
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(result["final_response"], r"\boxed{45}")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(
            [call["thinking_mode"] for call in client.calls],
            [True, False, False, False],
        )
        self.assertEqual(
            [call["max_tokens"] for call in client.calls],
            [8192, 2048, 4096, 2048],
        )
        equivalence = self._step(result, "equivalence")["content"]
        self.assertTrue(equivalence["final_verification_completion_used"])
        self.assertEqual(equivalence["repair_mode"], "verify_recovered")
        selection = self._step(result, "selection")["content"]
        self.assertEqual(selection["source"], "audit_retry")
        self.assertEqual(selection["recovered_answer_verification"], "complete")

    def test_conflicting_complete_answers_trigger_third_round_arbitration(self):
        client = StructuredRecordingClient([
            ModelCallResult(r"FINAL: \boxed{取x=1，代入检验得1^2=1。}"),
            ModelCallResult(r"FINAL: \boxed{取x=2，代入检验得2^2=4。}"),
            ModelCallResult(r"FINAL: \boxed{取x=2，代入检验得2^2=4。}"),
        ])

        result = ReasoningAgent(client).solve("构造一个正整数x，使x^2=4。", {})

        self.assertIn("x=2", result["final_response"])
        self.assertEqual(len(client.calls), 3)
        equivalence = self._step(result, "equivalence")["content"]
        self.assertTrue(equivalence["conflict"])
        self.assertTrue(equivalence["arbitration_used"])
        self.assertEqual(equivalence["repair_mode"], "arbitration")
        self.assertEqual(self._step(result, "selection")["content"]["source"], "verify")
        self.assertEqual(equivalence["arbitration_disposition"], "implicit_supports_b")

    def test_incomplete_verifier_box_uses_support_body_without_arbitration(self):
        client = StructuredRecordingClient([
            ModelCallResult(
                r"FINAL: \boxed{x_{n+1}=\frac{x_n^2+3}{2x_n},\ x_1=\frac74}"
            ),
            ModelCallResult(
                r"FINAL: \boxed{x_1=\frac74}" "\n"
                r"牛顿迭代公式为 $x_{n+1}=\frac{x_n^2+3}{2x_n}$，"
                r"代入 $x_0=2$ 得 $x_1=\frac74$。"
            ),
            ModelCallResult(r"FINAL: \boxed{SHOULD_NOT_BE_USED}"),
        ])

        result = ReasoningAgent(client).solve(
            "用牛顿法求方程x^2-3=0的迭代公式，并由x_0=2计算x_1。",
            {},
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("x_{n+1}", result["final_response"])
        self.assertIn("x_1", result["final_response"])
        equivalence = self._step(result, "equivalence")["content"]
        self.assertFalse(equivalence["conflict"])
        self.assertFalse(equivalence["arbitration_used"])
        self.assertEqual(self._step(result, "selection")["content"]["source"], "solve")

    def test_all_empty_calls_return_scoreable_fallback_not_failure_sentinel(self):
        client = StructuredRecordingClient([ModelCallResult("") for _ in range(3)])

        result = ReasoningAgent(client).solve("求一个满足条件的构造。", {})

        self.assertEqual(len(client.calls), 3)
        self.assertTrue(result["final_response"].strip())
        self.assertNotEqual(result["final_response"], "未能生成可验证的数学答案。")
        self.assertNotIn("unable to generate", result["final_response"].lower())
        selection = self._step(result, "selection")["content"]
        self.assertEqual(selection["source"], "fallback")
        self.assertEqual(selection["degraded_reason"], "degraded_all_empty")

    def test_best_effort_never_mines_an_unlabelled_multiline_draft(self):
        problem = "求这个数值。"
        spec = build_problem_spec(problem)

        answer, reason = SubmissionAgent._best_effort_answer(
            (("Assume n=10.\n2+3=5", False),), spec, problem
        )

        self.assertEqual(answer, "0")
        self.assertEqual(reason, "degraded_all_empty")

    def test_best_effort_can_keep_a_standalone_bare_scalar(self):
        problem = "求这个数值。"
        spec = build_problem_spec(problem)

        answer, reason = SubmissionAgent._best_effort_answer(
            (("42", False),), spec, problem
        )

        self.assertEqual(answer, "42")
        self.assertEqual(reason, "degraded_finalized_candidate")

    def test_remember_to_box_contract_is_preserved(self):
        client = StructuredRecordingClient([
            ModelCallResult(r"FINAL: \boxed{715}"),
            ModelCallResult(r"FINAL: \boxed{715}"),
        ])
        problem = (
            "How many positive integers less than 100000 have digit sum 9? "
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(result["final_response"], r"\boxed{715}")
        self.assertEqual(len(client.calls), 2)
        spec = self._step(result, "spec")["content"]
        self.assertEqual(spec["answer_contract"]["wrapper"], "boxed")
        self.assertEqual(self._step(result, "finalize")["content"]["contract_wrapper"], "boxed")

    def test_boxed_wrapper_does_not_inflate_a_routine_problem_budget(self):
        spec = build_problem_spec(
            r"Calculate a contest value. Remember to put your final answer within \boxed{}."
        )

        budget = plan_stage_budget(spec, False)

        self.assertFalse(budget.require_independent_review)
        self.assertEqual(budget.solve_tokens, 4096)
        self.assertEqual(budget.review_tokens, 4096)
        self.assertEqual(budget.repair_tokens, 2048)
        self.assertEqual(budget.review_min_remaining_seconds, 0)
        self.assertEqual(budget.emergency_tokens, 1024)
        self.assertEqual(budget.max_calls, 4)

    def test_boxed_wrapper_is_not_a_deep_reasoning_signal(self):
        contest = build_problem_spec(
            r"Calculate a contest value. Remember to put your final answer within \boxed{}."
        )
        routine = build_problem_spec("求 x。")

        self.assertFalse(SubmissionAgent._use_deep_reasoning(contest))
        self.assertFalse(SubmissionAgent._use_deep_reasoning(routine))

    def test_long_nonchoice_contest_problem_uses_deep_reasoning(self):
        problem = (
            "Find the minimum integer N with the following property. Given a collection "
            "of non-degenerate triangles satisfying several pairwise incidence constraints, "
            "prove that one can select a subcollection whose vertices obey the stated "
            "intersection condition and determine the exact extremal value. "
            r"Remember to put your final answer within \boxed{}."
        )
        spec = build_problem_spec(problem)
        budget = plan_stage_budget(spec, False, deep_reasoning=True)

        self.assertTrue(SubmissionAgent._use_deep_reasoning(spec, problem))
        self.assertEqual(budget.solve_tokens, 8192)
        self.assertEqual(budget.review_tokens, 8192)
        self.assertEqual(budget.repair_tokens, 4096)
        self.assertEqual(budget.review_min_remaining_seconds, 0)
        self.assertEqual(budget.emergency_tokens, 2048)
        self.assertEqual(budget.max_calls, 4)
        self.assertTrue(budget.require_independent_review)

    def test_multipart_and_nonlinear_solution_counts_use_deep_reasoning(self):
        multipart_problem = "计算行列式；再求矩阵的迹。"
        nonlinear_problem = (
            "Find the number of triples (x,y,z) of real numbers satisfying "
            "x^2+y^2+z^2=3 and xy^3+yz^3+zx^3=3."
        )

        multipart = build_problem_spec(multipart_problem)
        nonlinear = build_problem_spec(nonlinear_problem)

        self.assertGreater(len(multipart.goals), 1)
        self.assertTrue(SubmissionAgent._use_deep_reasoning(multipart, multipart_problem))
        self.assertTrue(SubmissionAgent._use_deep_reasoning(nonlinear, nonlinear_problem))

    def test_hard_structure_choice_uses_deep_reasoning(self):
        problem = (
            r"设$D_8$是正方形上的二面体群，下列关于元的阶、中心、"
            r"换位子群和四阶子群的多个选项中，选出全部正确选项。"
        )
        spec = build_problem_spec(problem)

        self.assertEqual(spec.profile.difficulty, "hard")
        self.assertTrue(SubmissionAgent._use_deep_reasoning(spec, problem))

    def test_hard_choice_uses_deep_solve_but_quick_independent_review(self):
        client = StructuredRecordingClient([
            ModelCallResult(r"FINAL: \boxed{B,D}"),
            ModelCallResult(r"FINAL: \boxed{B,D}"),
        ])
        problem = (
            "设$D_8$是正方形上的二面体群，请判断下列命题中全部正确的是：\n"
            r"\begin{itemize}" "\n"
            r"\item[A.] $D_8$中存在$8$阶元." "\n"
            r"\item[B.] $D_8$的四阶子群一定是Abel群." "\n"
            r"\item[C.] $C(D_8)=\{1\}$." "\n"
            r"\item[D.] $[D_8,D_8]$是$2$阶群." "\n"
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            [call["thinking_mode"] for call in client.calls],
            [True, False],
        )
        self.assertEqual(
            self._step(result, "model_call_verify")["content"]["routing_role"],
            "quick_response",
        )

    def test_recovered_hard_choice_also_gets_a_quick_independent_review(self):
        client = StructuredRecordingClient([
            ModelCallResult("Deep option analysis: unfinished", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{B,D}"),
            ModelCallResult(
                "FINAL: \\boxed{B,D}\nVERDICT: CONFIRMED\n"
                "CHECK: A=false, B=true, C=false, D=true by direct option checks."
            ),
        ])
        problem = (
            "设$D_8$是正方形上的二面体群，请判断下列命题中全部正确的是：\n"
            r"\begin{itemize}" "\n"
            r"\item[A.] $D_8$中存在$8$阶元." "\n"
            r"\item[B.] $D_8$的四阶子群一定是Abel群." "\n"
            r"\item[C.] $C(D_8)=\{1\}$." "\n"
            r"\item[D.] $[D_8,D_8]$是$2$阶群." "\n"
            r"Remember to put your final answer within \boxed{}."
        )

        result = ReasoningAgent(client).solve(problem, {})

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            [call["thinking_mode"] for call in client.calls],
            [True, False, False],
        )
        self.assertEqual(
            self._step(result, "equivalence")["content"]["repair_mode"],
            "verify_recovered",
        )
        self.assertEqual(
            self._step(result, "model_call_verify_recovered")["content"]["routing_role"],
            "quick_response",
        )

    def test_short_multi_blank_recall_is_quick_but_galois_fill_in_is_deep(self):
        recall_problem = (
            "10．对于时间序列的季节调整，常用的方法有( )、( )\n"
            r"Remember to put your final answer within \boxed{}."
        )
        galois_problem = (
            r"$x^4+5\in\mathbb{Q}[x]$在$\mathbb{Q}$上的分裂域(记为$E$)是$(\quad)$." "\n"
            r"$[E:\mathbb{Q}]=(\quad)$." "\n"
            r"$E/\mathbb{Q}$ $(\quad)$(填“是”或“否”.)为Galois扩张."
        )
        recall = build_problem_spec(recall_problem)
        galois = build_problem_spec(galois_problem)

        self.assertEqual(len(recall.goals), 2)
        self.assertTrue(SubmissionAgent._is_simple_multi_blank(recall, recall_problem))
        self.assertFalse(SubmissionAgent._use_deep_reasoning(recall, recall_problem))
        self.assertEqual(len(galois.goals), 3)
        self.assertFalse(SubmissionAgent._is_simple_multi_blank(galois, galois_problem))
        self.assertTrue(SubmissionAgent._use_deep_reasoning(galois, galois_problem))
        self.assertFalse(SubmissionAgent._use_deep_verification(galois, True))

    def test_missing_explicit_method_keeps_result_complete_but_loses_support_score(self):
        spec = build_problem_spec(
            "求长度为10的二进制串中恰有4个1且不含相邻两个1的串数，要求先选取1的位置再计算。"
        )

        candidate = assess_candidate("35", "solve", spec, ())

        self.assertTrue(candidate.accepted)
        self.assertFalse(all(candidate.support_coverage))
        self.assertTrue(candidate.complete_goals)
        self.assertTrue(candidate.formatting_valid)
        self.assertIs(choose_candidate([candidate]), candidate)


if __name__ == "__main__":
    unittest.main()
