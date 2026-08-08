from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from rag.card_retriever import CardRetriever
from reasoning.candidate_selector import ToolEvidence, assess_candidate, choose_candidate
from user_agent import ReasoningAgent


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return self.responses.pop(0)


class AdaptivePipelineTest(unittest.TestCase):
    def test_finite_field_spec_retrieves_domain_specific_cards(self):
        spec = build_problem_spec("设 F_81 为 81 元有限域，求生成整个扩张的元素个数。")
        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(spec.profile.subject, "抽象代数")
        self.assertIn("definition_or_structure_conditions", spec.risk_flags)
        self.assertTrue(any("有限域" in card.id for card in bundle.solve_cards))

    def test_proof_review_receives_reasoning_and_check_cards(self):
        client = RecordingClient([
            "由开覆盖定义，任意开覆盖可限制到闭子集。\n【最终答案】闭子集紧致。",
            "由闭子集的任意开覆盖扩展到母空间，利用紧致性取有限子覆盖，故闭子集紧致。\n\\boxed{闭子集紧致}",
        ])

        result = ReasoningAgent(client).solve("证明紧致空间的闭子集紧致。", {"idx": 1})

        self.assertEqual(len(client.calls), 2)
        self.assertIn("你是数学答案审查者", client.calls[1]["messages"][1]["content"])
        self.assertIn("关键依据、必要推导、明确结论", client.calls[1]["messages"][1]["content"])
        self.assertIn("闭子集", result["final_response"])

    def test_partial_tool_evidence_cannot_select_a_whole_answer(self):
        spec = build_problem_spec("证明函数连续并求其在 0 点的导数。")
        evidence = (ToolEvidence("1", "subexpression", "derivative", False),)
        solve = assess_candidate("函数连续，因此导数为 1。", "solve", spec, evidence)
        tool_like = assess_candidate("1", "sympy_verified", spec, evidence)

        chosen = choose_candidate([tool_like, solve])

        self.assertEqual(chosen.source, "solve")
        self.assertEqual(tool_like.tool_status, "unknown")


if __name__ == "__main__":
    unittest.main()
