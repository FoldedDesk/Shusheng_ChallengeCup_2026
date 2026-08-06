import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    def test_platform_constructor_and_solve_signatures(self):
        init = list(inspect.signature(ReasoningAgent.__init__).parameters)
        solve = list(inspect.signature(ReasoningAgent.solve).parameters)

        self.assertEqual(init, ["self", "client", "args", "kwargs"])
        self.assertEqual(solve, ["self", "problem", "metadata"])

    def test_two_public_client_calls_use_platform_arguments(self):
        client = RecordingClient([
            "计算得到 2+2=4。\n【最终答案】4",
            "核验无误。\n\\boxed{4}",
        ])

        result = ReasoningAgent(client=client).solve("计算 2+2。", {"idx": 0})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(call["temperature"] == 0.2 for call in client.calls))
        self.assertTrue(all(call["max_tokens"] == 4096 for call in client.calls))
        self.assertEqual(result["trace"][1]["content"]["status"], "completed")
        self.assertEqual(result["trace"][2]["content"]["status"], "completed")
        self.assertEqual(result["trace"][3]["content"]["source"], "model_explicit")
        self.assertIn("4", client.calls[1]["messages"][1]["content"])
        json.dumps(result, ensure_ascii=False)

    def test_proof_keeps_reasoning_and_conclusion(self):
        client = RecordingClient([
            "由紧致性定义，任一开覆盖有有限子覆盖。\n【最终答案】闭子集紧致。",
            "由开覆盖定义取有限子覆盖，故结论成立。\n\\boxed{闭子集紧致}",
        ])

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 1})

        self.assertIn("由开覆盖定义", result["final_response"])
        self.assertIn("闭子集紧致", result["final_response"])
        self.assertEqual(len(client.calls), 2)

    def test_sympy_is_a_non_network_fallback(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("计算 2+2。", {"idx": 2})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(result["trace"][1]["content"]["status"], "failed")
        self.assertEqual(result["trace"][2]["content"]["status"], "failed")
        self.assertEqual(result["trace"][3]["content"]["source"], "sympy")

    def test_common_final_label_is_extracted_without_special_brackets(self):
        client = RecordingClient(["计算过程略。\n最终答案：x=3", "\\boxed{x=3}"])

        result = ReasoningAgent(client).solve("求 x。", {"idx": 4})

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
        self.assertEqual(result["trace"][3]["content"]["source"], "model_explicit")

    def test_second_stage_corrects_first_stage_candidate(self):
        client = RecordingClient(["\\boxed{3}", "复核后应为 \\boxed{4}"])

        result = ReasoningAgent(client).solve("计算 2+2。", {"idx": 8})

        self.assertEqual(result["final_response"], "4")
        self.assertIn("3", client.calls[1]["messages"][1]["content"])

    def test_second_stage_failure_keeps_first_explicit_answer(self):
        class SecondCallFails:
            def __init__(self):
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return r"\boxed{x=2}"
                raise RuntimeError("temporary failure")

        result = ReasoningAgent(SecondCallFails()).solve("求 x。", {"idx": 9})

        self.assertEqual(result["final_response"], "x=2")
        self.assertEqual(result["trace"][2]["content"]["status"], "failed")

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


if __name__ == "__main__":
    unittest.main()
