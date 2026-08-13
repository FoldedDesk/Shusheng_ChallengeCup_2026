import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import classify_profile
from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from rag.card_retriever import CardRetriever
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


class RecordingClient:
    def __init__(self, responses="【最终答案】4"):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.responses.pop(0)


class SubmissionContractTest(unittest.TestCase):
    @staticmethod
    def _step(result, name):
        return next(item for item in result["trace"] if item["step"] == name)

    def test_platform_constructor_and_solve_signatures(self):
        init = list(inspect.signature(ReasoningAgent.__init__).parameters)
        solve = list(inspect.signature(ReasoningAgent.solve).parameters)

        self.assertEqual(init, ["self", "client", "args", "kwargs"])
        self.assertEqual(solve, ["self", "problem", "metadata"])

    def test_submission_prompt_requires_checked_stage_specific_final_placement(self):
        prompt = (Path("prompts") / "submission.txt").read_text(encoding="utf-8")

        self.assertIn("within 3000 reasoning tokens", prompt)
        self.assertIn("Complete the decisive calculation or check before committing to FINAL", prompt)
        self.assertIn("quick-response stages require a short check first", prompt)
        self.assertIn("put the object in FINAL", prompt)
        self.assertIn("complete label set in FINAL", prompt)

    def test_quick_and_deep_requests_use_different_final_placement(self):
        quick_problem = "一棵树有20个顶点和5个叶子，求度为3的顶点数。"
        deep_problem = "证明每个有限维赋范空间的子空间都是闭集。"
        quick_spec = build_problem_spec(quick_problem)
        deep_spec = build_problem_spec(deep_problem)

        quick = SubmissionAgent._stage_answer_instruction(quick_spec, quick_problem)
        deep = SubmissionAgent._stage_answer_instruction(deep_spec, deep_problem)

        self.assertIn("最后一行", quick)
        self.assertIn("不得先写暂定答案", quick)
        self.assertIn("第一行", deep)
        self.assertIn("隐藏推理", deep)

    def test_solve_request_adds_a_silent_type_specific_reasoning_protocol(self):
        problem = "求方程 x^2-1=0 的全部实根。"
        spec = build_problem_spec(problem)

        request = SubmissionAgent._solve_request(
            problem, spec, CardRetriever().retrieve(spec), ()
        )

        self.assertIn("内部解题协议", request)
        self.assertIn("求尽全部分支", request)
        self.assertIn("否证式检查", request)
        self.assertIn("不要输出这些标签", request)

    def test_independent_review_protocol_tries_to_falsify_the_candidate(self):
        problem = "Find all real roots of x^2-1=0."
        spec = build_problem_spec(problem)
        self.assertEqual(spec.profile.answer_shape, "roots")

        request = SubmissionAgent._verification_request(
            problem, spec, CardRetriever().retrieve(spec), (), []
        )

        self.assertIn("Independent internal protocol", request)
        self.assertIn("Try to falsify", request)
        self.assertIn("substitute every root back", request)

    def test_truth_verifier_gets_at_most_one_high_confidence_theorem_fact(self):
        problem = "在多项式环F_2[x]中判断x^3+x+1是否不可约，并说明理由。"
        spec = build_problem_spec(problem)
        cards = CardRetriever().retrieve(spec)
        fact = cards.verification_fact_context()

        request = SubmissionAgent._verification_request(
            problem, spec, cards, (), []
        )

        self.assertTrue(fact)
        self.assertIn("一条经校订的领域事实", request)
        self.assertEqual(request.count(fact), 1)

    def test_low_confidence_verifier_does_not_reuse_solve_method_card(self):
        problem = "证明每个有限集合都有有限个子集。"
        spec = build_problem_spec(problem)
        cards = CardRetriever().retrieve(spec)

        self.assertEqual(cards.verification_fact_context(), "")

    def test_verified_tool_route_uses_one_public_client_call(self):
        client = RecordingClient([
            "计算得到 2+2=4。\n【最终答案】4",
            "核验无误。\n\\boxed{4}",
        ])

        result = ReasoningAgent(client=client).solve("计算 2+2。", {"idx": 0})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(len(client.calls), 0)
        self.assertTrue(all(call["temperature"] == 0.2 for call in client.calls))
        self.assertTrue(all(call["max_tokens"] == 4096 for call in client.calls))
        self.assertEqual(self._step(result, "tool_evidence")["content"]["whole_goal_count"], 1)
        self.assertEqual(self._step(result, "selection")["content"]["source"], "sympy_verified")
        json.dumps(result, ensure_ascii=False)

    def test_proof_keeps_reasoning_and_conclusion(self):
        client = RecordingClient([
            "由紧致性定义，任一开覆盖有有限子覆盖。\n【最终答案】闭子集紧致。",
            "由开覆盖定义取有限子覆盖，故结论成立。\n\\boxed{闭子集紧致}",
        ])

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 1})

        self.assertIn("闭子集紧致", result["final_response"])
        self.assertNotIn(r"\boxed", result["final_response"])
        self.assertEqual(len(client.calls), 2)

    def test_sympy_is_a_non_network_fallback(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("计算 2+2。", {"idx": 2})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(self._step(result, "selection")["content"]["source"], "sympy_verified")

    def test_common_final_label_is_extracted_without_special_brackets(self):
        client = RecordingClient(["计算过程略。\n最终答案：x=3", "\\boxed{x=3}"])

        result = ReasoningAgent(client).solve("求 x。", {"idx": 4})

        self.assertEqual(result["final_response"], "x=3")

    def test_english_final_answer_in_a_code_fence_is_cleaned(self):
        client = RecordingClient(["```latex\nFinal Answer: x=3\n```", "\\boxed{x=3}"])

        result = ReasoningAgent(client).solve("求 x。", {"idx": 41})

        self.assertEqual(result["final_response"], "x=3")

    def test_boxed_answer_wins_over_a_thinking_scratchpad(self):
        client = RecordingClient([
            "Thinking Process: lengthy scratchpad.\n\\boxed{x=\\frac{1}{2}}",
            "\\boxed{x=\\frac{1}{2}}",
        ])

        result = ReasoningAgent(client).solve("求 x。", {"idx": 5})

        self.assertEqual(result["final_response"], r"x=\frac{1}{2}")

    def test_proof_scratchpad_returns_its_boxed_conclusion_only(self):
        client = RecordingClient([
            "Analysis: hidden draft.\n\\boxed{闭子集紧致}",
            "\\boxed{闭子集紧致}",
        ])

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 6})

        self.assertEqual(result["final_response"], "闭子集紧致")

    def test_empty_model_response_has_an_explicit_fallback_trace(self):
        result = ReasoningAgent(RecordingClient(["\\boxed{x=2}", "Thinking Process: truncated"])).solve("求 x。", {"idx": 7})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(result["trace"][-1]["content"]["source"], "solve")

    def test_second_stage_corrects_first_stage_candidate(self):
        client = RecordingClient([
            "\\boxed{取x=3，代入检验得3^2=9。}",
            "【校验】修正\n【最终答案】取x=4，代入检验得4^2=16。",
            "【校验】修正\n【最终答案】取x=4，代入检验得4^2=16。",
        ])

        result = ReasoningAgent(client).solve("构造一个正整数x，使x^2=16。", {"idx": 8})

        self.assertIn("x=4", result["final_response"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self._step(result, "selection")["content"]["source"], "verify")
        self.assertEqual(
            self._step(result, "equivalence")["content"]["arbitration_disposition"],
            "implicit_supports_b",
        )

    def test_second_stage_failure_keeps_first_explicit_answer(self):
        class SecondCallFails:
            def __init__(self):
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return r"\boxed{取x=2，代入检验得2^2=4。}"
                raise RuntimeError("temporary failure")

        result = ReasoningAgent(SecondCallFails()).solve(
            "构造一个正整数x，使x^2=4。", {"idx": 9}
        )

        self.assertIn("x=2", result["final_response"])
        self.assertTrue(self._step(result, "review_admission")["content"]["admitted"])
        self.assertEqual(self._step(result, "review_admission")["content"]["mode"], "verify")

    def test_chinese_scratchpad_without_an_explicit_answer_is_not_submitted(self):
        client = RecordingClient(["思考过程：尚未完成。", r"\boxed{x=2}"])

        result = ReasoningAgent(client).solve("求 x。", {"idx": 10})

        self.assertEqual(result["final_response"], "x=2")

    def test_failed_non_symbolic_problem_still_returns_non_empty_text(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("证明所有有限维子空间闭。", {"idx": 3})

        self.assertTrue(result["final_response"].strip())
        json.dumps(result, ensure_ascii=False)

    def test_sympy_plain_text_equation_and_infinity_format(self):
        tool = SympyTool()

        self.assertIn("SymPy 方程解: x=-1，x=1", tool.hints_for("求方程 x^2-1=0 的所有根。"))
        self.assertEqual(tool.limit("x", "x", "oo"), "∞")

    def test_english_problem_profile_routes_algebra_and_proof(self):
        roots = classify_profile("Solve the equation x^2 - 1 = 0.")
        proof = classify_profile("Prove that every finite dimensional subspace is closed.")

        self.assertEqual(roots.subject, "高等代数")
        self.assertEqual(roots.answer_shape, "roots")
        self.assertEqual(proof.problem_type, "proof")
        self.assertEqual(proof.difficulty, "hard")

    def test_equation_roots_are_not_rendered_as_an_interval(self):
        client = RecordingClient(["\\boxed{[-1, 1]}"])

        result = ReasoningAgent(client).solve("求方程 x^2-1=0 的所有根。", {"idx": 11})

        self.assertEqual(result["final_response"], "x=-1，x=1")
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(result["trace"][-1]["content"]["source"], "sympy_verified")

    def test_infinity_normalization_does_not_modify_english_words(self):
        profile = classify_profile("Prove a statement.")

        self.assertEqual(ReasoningAgent(RecordingClient()).agent._normalize_answer("proof with oo", profile), "proof with ∞")

    def test_non_symbolic_construction_stops_after_complete_rescue(self):
        client = RecordingClient([
            "\\boxed{A}",
            "【校验】修正\n【最终答案】取x=4，代入检验得4^2=16。",
            "【校验】修正\n【最终答案】取x=4，代入检验得4^2=16。",
        ])

        result = ReasoningAgent(client).solve("构造一个正整数x，使x^2=16。", {"idx": 12})

        self.assertIn("x=4", result["final_response"])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(self._step(result, "review_admission")["content"]["mode"], "rescue")

    def test_age_question_renders_a_bare_number_as_a_readable_sentence(self):
        result = ReasoningAgent(RecordingClient([r"\boxed{14}", r"\boxed{14}"])).solve(
            "小明今年14岁，问小明多少岁？", {"idx": 13}
        )

        self.assertEqual(result["final_response"], "小明14岁。")

    def test_count_and_probability_answers_are_rendered_as_sentences(self):
        count = ReasoningAgent(RecordingClient([r"\boxed{16}", r"\boxed{16}"])).solve(
            "共有多少个满足条件的整数？", {"idx": 14}
        )
        probability = ReasoningAgent(RecordingClient([r"\boxed{1/2}", r"\boxed{1/2}"])).solve(
            "事件 A 发生的概率是多少？", {"idx": 15}
        )

        self.assertEqual(count["final_response"], "所求数量为16个。")
        self.assertEqual(probability["final_response"], "所求概率为1/2。")


if __name__ == "__main__":
    unittest.main()
