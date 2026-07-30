from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import classify_difficulty, classify_problem_type, classify_subject
from core.client_adapter import ClientAdapter
from reasoning.finalizer import Finalizer
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


class SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.responses.pop(0)


class ArchitectureTest(unittest.TestCase):
    def test_adapter_uses_only_public_chat_method(self):
        client = SequencedClient(["ok"])

        self.assertEqual(ClientAdapter(client).chat([], 0.1, 12), "ok")
        self.assertEqual(client.calls[0]["temperature"], 0.1)

    def test_classifier_covers_required_categories(self):
        self.assertEqual(classify_subject("证明紧致空间的闭子集仍紧致"), "拓扑学")
        self.assertEqual(classify_problem_type("证明 x=1"), "proof")
        self.assertEqual(classify_difficulty("证明 x=1", "proof"), "hard")

    def test_easy_problem_runs_the_full_review_pipeline(self):
        client = SequencedClient([
            "【最终答案】7",
            "计算正确",
            "CHOICE: 0\nFINAL: 7\nREASON: 候选正确",
        ])

        result = ReasoningAgent(client).solve("计算 3+4。", {})

        self.assertEqual(result["final_response"], "7")
        self.assertEqual(len(client.calls), 3)

    def test_medium_problem_uses_solver_then_verifier(self):
        client = SequencedClient([
            "【最终答案】x=±1",
            "方程解完整",
            "CHOICE: 0\nFINAL: x=±1\nREASON: 候选满足方程",
        ])

        result = ReasoningAgent(client).solve("求方程 x^2-1=0 的所有根。", {})

        self.assertEqual(result["final_response"], "x=±1")
        self.assertEqual(len(client.calls), 3)

    def test_sympy_hints_are_injected_into_the_solver_context(self):
        client = SequencedClient([
            "【最终答案】3*x**2",
            "导数正确",
            "CHOICE: 0\nFINAL: 3*x**2\nREASON: 导数正确",
        ])

        result = ReasoningAgent(client).solve("求函数 f(x)=x^3 的导数。", {})

        self.assertEqual(result["final_response"], "3*x**2")
        self.assertIn("SymPy 导数: 3*x**2", client.calls[0]["messages"][1]["content"])
        self.assertIn({"step": "sympy", "content": {"hint_count": 1}}, result["trace"])

    def test_sympy_refuses_non_mathematical_input(self):
        self.assertIsNone(SympyTool().derivative("__import__('os')"))

    def test_hard_problem_uses_three_candidates_and_a_verifier(self):
        client = SequencedClient([
            "由条件可得 x=1。\n【最终答案】x=1",
            "由等式变形可得 x=1。\n【最终答案】x=1",
            "因此 x=1。\n【最终答案】x=1",
            "结论完整",
            "结论完整",
            "结论完整",
            "CHOICE: 1\nFINAL: x=1\nREASON: 第二份推导完整",
        ])

        result = ReasoningAgent(client).solve("证明 x=1。", {})

        self.assertEqual(result["final_response"], "x=1")
        self.assertEqual(len(client.calls), 7)

    def test_state_isolated_between_calls(self):
        client = SequencedClient([
            "【最终答案】1", "计算正确", "CHOICE: 0\nFINAL: 1\nREASON: 正确",
            "【最终答案】2", "计算正确", "CHOICE: 0\nFINAL: 2\nREASON: 正确",
        ])
        agent = ReasoningAgent(client)

        first = agent.solve("计算 1。", {})
        second = agent.solve("计算 2。", {})

        self.assertEqual(first["final_response"], "1")
        self.assertEqual(second["final_response"], "2")
        self.assertIsNot(first["trace"], second["trace"])

    def test_finalizer_recovers_quoted_chinese_answer_and_balances_latex(self):
        leaked = (
            '【最终答案】<可直接判分的答案>". Usually, write '
            '"甲地市场价格 p1=100，乙地市场价格 p2=60".'
        )

        self.assertEqual(
            Finalizer.extract(leaked), "甲地市场价格 p1=100，乙地市场价格 p2=60"
        )
        self.assertEqual(Finalizer.extract("【最终答案】最大面积为 $2R^2"), "最大面积为 $2R^2$")

    def test_verifier_final_answer_replaces_an_incomplete_candidate(self):
        client = SequencedClient([
            "组合数为 C_7^4=35。\n【最终答案】35",
            "候选结论不完整",
            "CHOICE: 0\nFINAL: 不含相邻整数的选法数为35\nREASON: 组合数计算正确",
        ])

        result = ReasoningAgent(client).solve(
            "在集合{1,2,…,10}中任选4个元素，求其中不含相邻整数的选法数。", {}
        )

        self.assertEqual(result["final_response"], "不含相邻整数的选法数为35")
        self.assertNotIn("REASON", str(result["trace"]))

    def test_sympy_result_falls_back_when_all_model_calls_fail(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("求函数 f(x)=x^3 的导数。", {})

        self.assertEqual(result["final_response"], "3*x**2")
        self.assertIn({"step": "solver_fallback", "content": "sympy_hint_used"}, result["trace"])


if __name__ == "__main__":
    unittest.main()
