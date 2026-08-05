import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


class RecordingClient:
    def __init__(self, response="【最终答案】4"):
        self.response = response
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.response


class SubmissionContractTest(unittest.TestCase):
    def test_platform_constructor_and_solve_signatures(self):
        init = list(inspect.signature(ReasoningAgent.__init__).parameters)
        solve = list(inspect.signature(ReasoningAgent.solve).parameters)

        self.assertEqual(init, ["self", "client", "args", "kwargs"])
        self.assertEqual(solve, ["self", "problem", "metadata"])

    def test_one_public_client_call_uses_platform_arguments(self):
        client = RecordingClient("计算得到 2+2=4。\n【最终答案】4")

        result = ReasoningAgent(client=client).solve("计算 2+2。", {"idx": 0})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["temperature"], 0.2)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)
        self.assertEqual(result["trace"][1]["content"]["status"], "completed")
        self.assertEqual(result["trace"][2]["content"]["source"], "model")
        json.dumps(result, ensure_ascii=False)

    def test_proof_keeps_reasoning_and_conclusion(self):
        client = RecordingClient(
            "由紧致性定义，任一开覆盖有有限子覆盖。\n【最终答案】闭子集紧致。"
        )

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 1})

        self.assertIn("由紧致性定义", result["final_response"])
        self.assertIn("闭子集紧致", result["final_response"])
        self.assertEqual(len(client.calls), 1)

    def test_sympy_is_a_non_network_fallback(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("计算 2+2。", {"idx": 2})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(result["trace"][1]["content"]["status"], "failed")
        self.assertEqual(result["trace"][2]["content"]["source"], "sympy")

    def test_common_final_label_is_extracted_without_special_brackets(self):
        client = RecordingClient("计算过程略。\n最终答案：x=3")

        result = ReasoningAgent(client).solve("求 x。", {"idx": 4})

        self.assertEqual(result["final_response"], "x=3")

    def test_boxed_answer_wins_over_a_thinking_scratchpad(self):
        client = RecordingClient("Thinking Process: lengthy scratchpad.\n\\boxed{x=\\frac{1}{2}}")

        result = ReasoningAgent(client).solve("求 x。", {"idx": 5})

        self.assertEqual(result["final_response"], r"x=\frac{1}{2}")

    def test_proof_scratchpad_returns_its_boxed_conclusion_only(self):
        client = RecordingClient("Analysis: hidden draft.\n\\boxed{闭子集紧致}")

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 6})

        self.assertEqual(result["final_response"], "闭子集紧致")

    def test_empty_model_response_has_an_explicit_fallback_trace(self):
        result = ReasoningAgent(RecordingClient("")).solve("求 x。", {"idx": 7})

        self.assertTrue(result["final_response"].strip())
        self.assertEqual(result["trace"][2]["content"]["source"], "fallback")

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
