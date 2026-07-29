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


class CandidateSelectionTest(unittest.TestCase):
    def _agent(self, responses, **config_overrides):
        config = AgentConfig(use_llm_extraction=False, **config_overrides)
        return ReasoningAgent(SequencedClient(responses), config)

    def test_uses_three_short_candidates_before_majority_selection(self):
        agent = self._agent(
            ["【最终答案】7", "【最终答案】7", "【最终答案】8"],
        )

        result = agent.solve("计算 3+4。", {"idx": 10})

        self.assertEqual(result["final_response"], "7")
        self.assertEqual(len(agent.client.calls), 3)
        self.assertTrue(any(item["step"] == "majority_vote" for item in result["trace"]))
        self.assertTrue(all(call["max_tokens"] == 6144 for call in agent.client.calls))

    def test_disagreement_uses_independent_verifier_to_select_matching_candidate(self):
        agent = self._agent([
            "【最终答案】1",
            "【最终答案】2",
            "【最终答案】3",
            "ANSWER: 2\nCONFIDENCE: 95",
            "FINAL: 2",
        ])

        result = agent.solve("计算 1+1。", {"idx": 11})

        self.assertEqual(result["final_response"], "2")
        verification = next(
            item for item in result["trace"] if item["step"] == "independent_verification"
        )
        self.assertEqual(verification["content"]["selected_candidate"], 1)
        verifier_call = agent.client.calls[3]
        self.assertEqual(verifier_call["max_tokens"], 4096)
        self.assertEqual(verifier_call["messages"][1]["content"], "题目：\n计算 1+1。")

    def test_unparseable_verifier_response_keeps_deterministic_candidate_fallback(self):
        agent = self._agent([
            "【最终答案】1",
            "【最终答案】2",
            "【最终答案】3",
            "无法确定",
            "FINAL: 1",
        ])

        result = agent.solve("计算 1+1。", {"idx": 12})

        self.assertEqual(result["final_response"], "1")
        verification = next(
            item for item in result["trace"] if item["step"] == "independent_verification"
        )
        self.assertIsNone(verification["content"]["verifier_answer"])

    def test_parser_clamps_verifier_confidence(self):
        self.assertEqual(
            ReasoningAgent._parse_verifier_response("ANSWER: x=1\nCONFIDENCE: 120"),
            ("x=1", 100),
        )

    def test_high_risk_majority_is_overridden_by_independent_solution(self):
        agent = self._agent([
            "【最终答案】84",
            "【最终答案】84",
            "【最终答案】84",
            "ANSWER: 120\nCONFIDENCE: 100",
        ])

        result = agent.solve("求正整数解数，需用隔板法计数。", {"idx": 13})

        self.assertEqual(result["final_response"], "120")
        verification = next(
            item for item in result["trace"] if item["step"] == "majority_verification"
        )
        self.assertTrue(verification["content"]["overrode_majority"])

    def test_all_garbage_falls_back_to_independent_solution(self):
        agent = self._agent([
            "【最终答案】<final answer>",
            "【最终答案】<final answer>",
            "【最终答案】<final answer>",
            "ANSWER: 取Y=X，P(X=Y)=1\nCONFIDENCE: 100",
        ])

        result = agent.solve("构造两个不独立的 Bernoulli(1/2) 变量。", {"idx": 14})

        self.assertEqual(result["final_response"], "取Y=X,P(X=Y)=1")

    def test_punctuation_only_finalizer_output_is_rejected(self):
        agent = self._agent([
            "【最终答案】速度长度为√2，不是弧长参数",
            "【最终答案】速度长度为√2，不是弧长参数",
            "【最终答案】速度长度为√2，不是弧长参数",
        ])

        self.assertFalse(ReasoningAgent._is_useful_final_answer(")."))
        result = agent.solve("求速度长度并判断是否为弧长参数。", {"idx": 15})
        self.assertEqual(result["final_response"], "速度长度为√2,不是弧长参数")

    def test_construction_answer_preserves_the_constructed_variables(self):
        self.assertEqual(
            ReasoningAgent._normalize_construction_answer(
                "构造边缘均为Bernoulli(1/2)但不独立的变量，并给出P(X=Y)。",
                "1",
                "令 Y = X，则两个变量不独立。",
            ),
            "取Y=X,P(X=Y)=1",
        )


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
