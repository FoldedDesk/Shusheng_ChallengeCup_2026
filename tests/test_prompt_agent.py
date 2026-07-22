import unittest

from user_agent import AgentConfig, AgentMessage, ReasoningAgent, _PromptAgent


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return "wrapped response"


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


class PromptAgentTest(unittest.TestCase):
    def test_wraps_client_chat_with_system_and_user_messages(self):
        client = FakeClient()
        agent = _PromptAgent(client, template="system prompt", name="policy")

        response = agent(
            AgentMessage(sender="user", content="problem text"),
            session_id="ignored",
            temperature=0.6,
            max_tokens=123,
        )

        self.assertEqual(response, AgentMessage(sender="policy", content="wrapped response"))
        self.assertEqual(client.calls, [{
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "problem text"},
            ],
            "temperature": 0.6,
            "max_tokens": 123,
        }])


class FullSolutionFlowTest(unittest.TestCase):
    def _agent(self, responses):
        return ReasoningAgent(
            SequencedClient(responses),
            AgentConfig(policy_sample_times=2, use_llm_extraction=False),
        )

    def test_english_proof_with_agreeing_complete_candidates_skips_model_review(self):
        candidate = (
            "设 x=1。由已知等式 x=x 可知该等式两端相等，"
            "因此 x=1 满足题设并且推导完成。\n【最终答案】x=1"
        )
        agent = self._agent([candidate, candidate])

        result = agent.solve("Prove that x = 1. Show all steps.", {"idx": 1})

        self.assertEqual(result["final_response"], candidate)
        self.assertEqual(len(agent.client.calls), 2)
        self.assertTrue(any(
            item["step"] == "full_solution_consensus" for item in result["trace"]
        ))

    def test_answer_only_candidates_do_not_enter_full_solution_fallback(self):
        agent = self._agent(["【最终答案】1", "【最终答案】1"])

        result = agent.solve("请证明结论为 1", {"idx": 2})

        self.assertEqual(result["final_response"], "TRUNCATED_ALL")

    def test_conflicting_complete_candidates_are_audited_and_repaired(self):
        first = "由定义逐步计算，先代入已知条件，再化简等式，故最终得到数值 1。\n【最终答案】1"
        second = "由另一条完整推导，先列出关系式，再逐项化简，故最终得到数值 2。\n【最终答案】2"
        repaired = (
            "【解答】\n由第二个候选先列关系式、再逐项化简的推导可知结论成立。"
            "\n【结论】\n答案为 2。"
        )
        agent = self._agent([first, second, "CHOICE: 1\nISSUES: 统一表述", repaired])

        result = agent.solve("请推导结果", {"idx": 3})

        self.assertEqual(result["final_response"], repaired)
        self.assertEqual(len(agent.client.calls), 4)

    def test_invalid_audit_choice_falls_back_to_highest_quality_candidate(self):
        weak = "根据定义先写出等式，再进行一次代入计算，继续化简后最后得到数值 1。\n【最终答案】1"
        strong = (
            "根据定义进行两步推导，先得到中间关系式，再代入已知条件化简，"
            "最终得到数值 2。\n【最终答案】2"
        )
        agent = self._agent([weak, strong, "CHOICE: 9\nISSUES: 无"])

        result = agent.solve("请证明该结论", {"idx": 4})

        self.assertEqual(result["final_response"], strong)
        audit = next(item for item in result["trace"] if item["step"] == "solution_audit")
        self.assertEqual(audit["content"]["selected_candidate"], 1)


class GarbageDetectionTest(unittest.TestCase):
    def test_accepts_bracketed_mathematical_intervals(self):
        self.assertFalse(ReasoningAgent._looks_like_garbage("[1, 1.5]"))
        self.assertFalse(ReasoningAgent._looks_like_garbage("[0, 1]"))
        self.assertFalse(ReasoningAgent._looks_like_garbage("[-1, 1]"))

    def test_rejects_quoted_bracketed_placeholder(self):
        self.assertTrue(ReasoningAgent._looks_like_garbage('["placeholder"]'))
        self.assertTrue(ReasoningAgent._looks_like_garbage("['placeholder']"))


if __name__ == "__main__":
    unittest.main()
