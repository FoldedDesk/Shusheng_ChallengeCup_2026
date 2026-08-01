import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier import classify_difficulty, classify_problem_type, classify_subject
from core.client_adapter import ClientAdapter
from core.execution_limits import (
    OVERALL_TIMEOUT_SECONDS,
    PER_ITEM_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    dataset_worst_case_seconds,
    hard_item_worst_case_seconds,
)
from reasoning.finalizer import Finalizer
from reasoning.verifier import Verifier
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

    def test_hard_pipeline_execution_budget_fits_evaluation_limits(self):
        self.assertEqual(REQUEST_TIMEOUT_SECONDS, 60)
        self.assertEqual(hard_item_worst_case_seconds(), 9 * 60)
        self.assertEqual(dataset_worst_case_seconds(112), 342 * 60)
        self.assertLessEqual(hard_item_worst_case_seconds(), PER_ITEM_TIMEOUT_SECONDS)
        self.assertLessEqual(dataset_worst_case_seconds(112), OVERALL_TIMEOUT_SECONDS)

    def test_classifier_covers_required_categories(self):
        self.assertEqual(classify_subject("证明紧致空间的闭子集仍紧致"), "拓扑学")
        self.assertEqual(classify_problem_type("证明 x=1"), "proof")
        self.assertEqual(classify_problem_type("求子集个数，并说明两类子集为何相等"), "explanation")
        self.assertEqual(classify_difficulty("证明 x=1", "proof"), "hard")
        self.assertEqual(classify_difficulty("求子集个数，并说明两类子集为何相等", "explanation"), "hard")

    def test_easy_problem_skips_critic_but_keeps_verification(self):
        client = SequencedClient([
            "【最终答案】7",
            "CHOICE: 0\nFINAL: 7\nREASON: 候选正确",
        ])

        result = ReasoningAgent(client).solve("计算 3+4。", {})

        self.assertEqual(result["final_response"], "7")
        self.assertEqual(len(client.calls), 2)

    def test_medium_problem_uses_solver_then_verifier(self):
        client = SequencedClient([
            "【最终答案】x=±1",
            "CHOICE: 0\nFINAL: x=±1\nREASON: 候选满足方程",
        ])

        result = ReasoningAgent(client).solve("求方程 x^2-1=0 的所有根。", {})

        self.assertEqual(result["final_response"], "x=±1")
        self.assertEqual(len(client.calls), 2)

    def test_sympy_hints_are_injected_into_the_solver_context(self):
        client = SequencedClient([
            "【最终答案】3*x**2",
            "CHOICE: 0\nFINAL: 3*x**2\nREASON: 导数正确",
        ])

        result = ReasoningAgent(client).solve("求函数 f(x)=x^3 的导数。", {})

        self.assertEqual(result["final_response"], "3*x**2")
        self.assertIn("SymPy 导数: 3*x^2", client.calls[0]["messages"][1]["content"])
        self.assertIn({"step": "sympy", "content": {"hint_count": 1}}, result["trace"])

    def test_sympy_supports_plain_text_equations_and_arithmetic_fallbacks(self):
        tool = SympyTool()

        self.assertIn("SymPy 方程解: x=-1，x=1", tool.hints_for("求方程 x^2-1=0 的所有根。"))
        self.assertIn("SymPy 计算: 4", tool.hints_for("计算 2+2。"))

        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("计算 2+2。", {})
        self.assertEqual(result["final_response"], "4")

    def test_sympy_supports_raw_latex_limits_and_definite_integrals(self):
        tool = SympyTool()

        self.assertIn("SymPy 极限: 3", tool.hints_for(r"求极限 \lim_{x \to 0} \frac{\sin 3x}{x}"))
        self.assertIn("SymPy 定积分: 2", tool.hints_for(r"计算定积分 \int_0^1 (2x+1)\, dx"))
        self.assertIn("SymPy 导数: ln(x) + 1", tool.hints_for(r"y=x\ln x，求导数。"))
        self.assertIn("SymPy 导数: 3*x^2 + 2", tool.hints_for(r"求函数 $f(x)=x^3+2x-5$ 的导数。"))
        self.assertIn("SymPy 导数: e^x*sin(x) + e^x*cos(x)", tool.hints_for(r"求函数 $f(x)=e^x\sin x$ 的导数。"))
        self.assertEqual(tool.limit("x", "x", "oo"), "∞")
        self.assertIn(
            "SymPy 偏导数: 2*x*y + y^2",
            tool.hints_for(r"求函数 $f(x,y)=x^2y+xy^2$ 关于 $x$ 的偏导数。"),
        )
        self.assertIn("SymPy 极限: 2", tool.hints_for(r"求极限 $\lim_{x\to0}\frac{e^{2x}-1}{\sin x}$。"))
        self.assertIn(
            "SymPy 导数: 1/(x*(ln(x)^2 + 1))",
            tool.hints_for(r"求函数 $f(x)=\arctan(\ln x)$ 的导数。"),
        )
        self.assertIn(
            "本地同余方程解: x=4 (mod 7)",
            tool.hints_for(r"求解同余方程 $3x \equiv 5 \pmod{7}$，给出最小正整数解。"),
        )
        self.assertIn("本地模幂计算: 1", tool.hints_for(r"计算 $2^{10} \bmod 11$。"))

    def test_sympy_refuses_non_mathematical_input(self):
        self.assertIsNone(SympyTool().derivative("__import__('os')"))

    def test_incomplete_solver_response_is_retried_before_verification(self):
        client = SequencedClient([
            "Thinking Process: compute 2+2.",
            "计算可得 2+2=4。\n【最终答案】4",
            "CHOICE: 0\nFINAL: 4\nREASON: 计算正确",
        ])

        result = ReasoningAgent(client).solve("计算 2+2。", {})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(len(client.calls), 3)

    def test_solver_scratchpad_is_retried_before_verification(self):
        client = SequencedClient([
            "Thinking Process:\n【最终答案】4",
            "计算可得 2+2=4。\n【最终答案】4",
            "CHOICE: 0\nFINAL: 4\nREASON: 计算正确",
        ])

        result = ReasoningAgent(client).solve("计算 2+2。", {})

        self.assertEqual(result["final_response"], "4")
        self.assertEqual(len(client.calls), 3)

    def test_hard_solver_scratchpad_is_retried(self):
        client = SequencedClient([
            "1. 分析条件。\n2. 给出结论。",
            "Thinking Process:\n【最终答案】x=1",
            "由题设可得 x=1。\n【最终答案】x=1",
            "独立核对得 x=1。\n【最终答案】x=1",
            "再次核对得 x=1。\n【最终答案】x=1",
            "结论一致。",
            "CHOICE: 0\nFINAL: x=1\nREASON: 条件充分",
        ])

        result = ReasoningAgent(client).solve("证明 x=1。", {})

        self.assertEqual(result["final_response"], "x=1")
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(client.calls[1]["max_tokens"], 8192)

    def test_hard_problem_uses_three_candidates_and_a_verifier(self):
        client = SequencedClient([
            "1. 明确已知条件和待证结论。\n2. 逐步变形并核对结论。",
            "由条件可得 x=1。\n【最终答案】x=1",
            "由等式变形可得 x=1。\n【最终答案】x=1",
            "因此 x=1。\n【最终答案】x=1",
            "结论完整",
            "CHOICE: 1\nFINAL: x=1\nREASON: 第二份推导完整",
        ])

        result = ReasoningAgent(client).solve("证明 x=1。", {})

        self.assertEqual(result["final_response"], "x=1")
        self.assertEqual(len(client.calls), 6)

    def test_hard_dataset_source_forces_the_hard_pipeline(self):
        client = SequencedClient([
            "1. 分析关键条件。\n2. 核对结论。",
            "由定义可得结论。\n【最终答案】4",
            "独立核对可得 4。\n【最终答案】4",
            "结果为 4。\n【最终答案】4",
            "结论一致。",
            "CHOICE: 0\nFINAL: 4\nREASON: 核对正确",
        ])

        result = ReasoningAgent(client).solve(
            "求一个有限结构的计数结果。",
            {"source": "official_distribution_112_hard"},
        )

        classification = next(step for step in result["trace"] if step["step"] == "classification")
        self.assertEqual(classification["content"]["difficulty"], "hard")
        self.assertEqual(len(client.calls), 6)

    def test_hard_problem_decomposition_executes_safe_subproblem_tool(self):
        client = SequencedClient([
            (
                "1. 将方程化为等于零的形式。\n"
                "2. 求出根并代回原方程核验。\n"
                "TOOL: equation|x^2-1|x"
            ),
            "由 x^2-1=(x-1)(x+1) 可得两根。\n【最终答案】x=-1 或 x=1",
            "因式分解后逐一验证。\n【最终答案】x=-1 或 x=1",
            "代回可知均成立。\n【最终答案】x=-1 或 x=1",
            "三个候选均使用了方程根的核验。",
            (
                "CHOICE: 0\nFINAL: x^2-1=(x-1)(x+1)，"
                "故根只能为 -1 或 1；代回均满足原方程。\n"
                "【最终答案】x=-1 或 x=1\nREASON: 两根均满足原方程"
            ),
        ])

        result = ReasoningAgent(client).solve("求证方程 x^2-1=0 的全部解为 x=-1 或 x=1。", {})

        self.assertIn("【最终答案】x=-1 或 x=1", result["final_response"])
        self.assertEqual(len(client.calls), 6)
        solver_context = client.calls[1]["messages"][1]["content"]
        self.assertIn("分解后的子目标", solver_context)
        self.assertIn("分解工具 equation: x=-1, x=1", solver_context)
        decomposition_trace = next(step for step in result["trace"] if step["step"] == "decomposition")
        self.assertEqual(decomposition_trace["content"]["tool_count"], 1)

    def test_proof_response_keeps_reasoning_from_multiline_verifier_final(self):
        client = SequencedClient([
            "1. 写出平方非负这一基础事实。\n2. 将其与常数项相加得到下界。",
            "由 x^2\ge0 可知 x^2+1>0。\n【最终答案】对任意实数 x，x^2+1>0。",
            "同理可得结论。\n【最终答案】x^2+1>0。",
            "因为 x^2 非负。\n【最终答案】x^2+1>0。",
            "论证完整",
            (
                "CHOICE: 0\nFINAL: 对任意实数 x，有 x^2\ge0，"
                "故 x^2+1\ge1>0。\n【最终答案】x^2+1>0。\nREASON: 证明完整"
            ),
        ])

        result = ReasoningAgent(client).solve("证明任意实数 x 都有 x^2+1>0。", {})

        self.assertIn("x^2\\ge0", result["final_response"])
        self.assertIn("【最终答案】x^2+1>0", result["final_response"])
        self.assertIn("题设与目标", client.calls[1]["messages"][1]["content"])

    def test_all_regression_proofs_preserve_reasoning_and_a_final_conclusion(self):
        class ProofClient:
            def chat(self, messages, temperature=0.2, max_tokens=4096):
                system = messages[0]["content"]
                if "验证器" in system:
                    return (
                        "CHOICE: 0\nFINAL: 由题设和相关定义逐步推出所求结论。"
                        "\n【最终答案】命题成立。\nREASON: 论证完整"
                    )
                if "批评器" in system:
                    return "关键条件已使用，推导完整。"
                return "由题设和相关定义逐步推出所求结论。\n【最终答案】命题成立。"

        path = Path(__file__).resolve().parents[1] / "sample_data" / "full_solution_output_regression_23.jsonl"
        problems = [
            item["problem"] for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if "证明" in item["problem"] or "求证" in item["problem"]
        ]
        agent = ReasoningAgent(ProofClient())

        for problem in problems:
            result = agent.solve(problem, {})
            critic_trace = next(step for step in result["trace"] if step["step"] == "critic")
            self.assertGreater(critic_trace["content"]["review_count"], 0)
            self.assertIn("逐步推出", result["final_response"])
            self.assertIn("【最终答案】", result["final_response"])
            self.assertNotEqual(result["final_response"], "TRUNCATED_ALL")

    def test_state_isolated_between_calls(self):
        client = SequencedClient([
            "【最终答案】1", "CHOICE: 0\nFINAL: 1\nREASON: 正确",
            "【最终答案】2", "CHOICE: 0\nFINAL: 2\nREASON: 正确",
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
            "CHOICE: 0\nFINAL: 不含相邻整数的选法数为35\nREASON: 组合数计算正确",
        ])

        result = ReasoningAgent(client).solve(
            "在集合{1,2,…,10}中任选4个元素，求其中不含相邻整数的选法数。", {}
        )

        self.assertEqual(result["final_response"], "不含相邻整数的选法数为35")
        self.assertNotIn("REASON", str(result["trace"]))

    def test_verifier_preserves_requested_object_when_final_only_has_a_check_value(self):
        client = SequencedClient(["CHOICE: 0\nFINAL: 验证 p(2)=5\nREASON: 数据一致"])
        verifier = Verifier(ClientAdapter(client), "验证器")
        state = type("State", (), {
            "problem": "求二次插值多项式p(x)通过三点，并验证。",
            "problem_type": "calculation",
            "difficulty": "medium",
        })()

        result = verifier.verify(state, ["推导得 p(x)=x^2+1。\n【最终答案】p(x)=x^2+1"], [])

        self.assertEqual(result["final_answer"], "p(x)=x^2+1")

    def test_sympy_result_falls_back_when_all_model_calls_fail(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("求函数 f(x)=x^3 的导数。", {})

        self.assertEqual(result["final_response"], "3*x^2")
        self.assertIn({"step": "solver_fallback", "content": "sympy_hint_used"}, result["trace"])

    def test_model_failures_are_exposed_in_trace(self):
        class FailingClient:
            def chat(self, **kwargs):
                raise RuntimeError("offline")

        result = ReasoningAgent(FailingClient()).solve("计算 2+2。", {})

        solver_trace = next(step for step in result["trace"] if step["step"] == "solver_0")
        self.assertEqual(solver_trace["content"]["status"], "candidate_failed")
        self.assertEqual(solver_trace["content"]["errors"], ["RuntimeError", "RuntimeError"])

    def test_incomplete_verifier_response_is_retried(self):
        client = SequencedClient([
            "【最终答案】4",
            "Thinking Process: compare the candidate.",
            "CHOICE: 0\nFINAL: 4\nREASON: 计算正确",
        ])

        result = ReasoningAgent(client).solve("计算 2+2。", {})

        self.assertEqual(result["final_response"], "4")
        verification = next(step for step in result["trace"] if step["step"] == "verification")
        self.assertEqual(verification["content"]["error_count"], 1)

    def test_empty_verifier_final_is_retried(self):
        client = SequencedClient([
            "【最终答案】4",
            "CHOICE: 0\nFINAL:\nREASON: 空结论",
            "CHOICE: 0\nFINAL: 4\nREASON: 计算正确",
        ])

        result = ReasoningAgent(client).solve("计算 2+2。", {})

        self.assertEqual(result["final_response"], "4")


if __name__ == "__main__":
    unittest.main()
