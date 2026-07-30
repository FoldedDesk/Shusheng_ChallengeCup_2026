from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_agent import ReasoningAgent


class FakeClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        if "CHOICE:" in messages[0]["content"]:
            return "CHOICE: 0\nREASON: 候选完整"
        return "由计算得结果。\n【最终答案】4"


class OutputTest(unittest.TestCase):
    def test_solve_returns_a_non_empty_final_response(self):
        result = ReasoningAgent(client=FakeClient()).solve("计算 2+2。", {})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["final_response"], "4")
        self.assertIsInstance(result.get("trace"), list)


if __name__ == "__main__":
    unittest.main()
