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

    def test_template_override_replaces_the_system_prompt(self):
        client = FakeClient()
        agent = _PromptAgent(client, template="default prompt", name="policy")

        agent(
            AgentMessage(sender="user", content="problem text"),
            session_id="ignored",
            temperature=0.2,
            max_tokens=321,
            template_override="routed prompt",
        )

        self.assertEqual(client.calls[0]["messages"][0]["content"], "routed prompt")


class SubjectRoutingTest(unittest.TestCase):
    def test_classifies_priority_subjects(self):
        cases = {
            "用牛顿法求迭代公式": "数值分析",
            "求曲线的弧长参数与主曲率": "微分几何",
            "求微分方程的通解": "常微分方程",
            "计算 z=i 处的留数": "复分析",
            "证明函数几乎处处收敛": "测度积分",
            "构造 Bernoulli(1/2) 随机变量": "概率论",
            "求图的哈密顿路径个数": "离散数学",
            "证明群同态的核是正规子群": "抽象代数",
            "对热方程计算 u_t 与 u_xx": "偏微分方程",
            "证明样本均值是无偏估计": "统计推断",
            "计算决定系数 R^2": "线性回归",
            "计算平稳过程的协方差函数": "随机过程",
            "求评价泛函的算子范数": "泛函分析",
            "证明紧致空间的闭子集紧致": "拓扑学",
        }
        for problem, subject in cases.items():
            self.assertEqual(ReasoningAgent._classify_subject(problem), subject)

    def test_routed_prompt_includes_subject_checklist(self):
        prompt = ReasoningAgent._routed_policy_prompt("用牛顿法求迭代公式", False)

        self.assertIn("当前题型：数值分析", prompt)
        self.assertIn("迭代格式和初值", prompt)

    def test_dual_path_uses_distinct_candidate_roles(self):
        prompts = [
            ReasoningAgent._candidate_policy_prompt("用牛顿法求迭代公式", index, False)
            for index in range(3)
        ]

        self.assertIn("严谨的数学推理智能体", prompts[0])
        self.assertIn("结构化快速求解器", prompts[1])
        self.assertIn("独立数学审计求解器", prompts[2])


class ObjectCompletenessSelectionTest(unittest.TestCase):
    def test_complete_candidate_overrides_a_bare_number_majority(self):
        agent = ReasoningAgent(
            SequencedClient([
                "【最终答案】1.75",
                "【最终答案】1.75",
                "【最终答案】x_{n+1}=(x_n+3/x_n)/2，x_1=7/4",
                "ANSWER: 1.75\nCONFIDENCE: 100",
                "FINAL: x_{n+1}=(x_n+3/x_n)/2，x_1=7/4",
            ]),
            AgentConfig(use_llm_extraction=False),
        )

        result = agent.solve("用牛顿法求迭代公式，并由x_0=2计算x_1。", {"idx": 16})

        self.assertIn("x_{n+1}", result["final_response"])
        self.assertTrue(any(
            item["step"] == "object_completeness_override" for item in result["trace"]
        ))


class FullSolutionFlowTest(unittest.TestCase):
    def _agent(self, responses):
        return ReasoningAgent(
            SequencedClient(responses),
            AgentConfig(policy_sample_times=2, use_llm_extraction=False),
        )

    def test_chinese_proof_with_agreeing_complete_candidates_skips_model_review(self):
        candidate = (
            "设 x=1。由已知等式两端相等可知该结论成立，"
            "因此 x=1 满足题设。\n【最终答案】x=1"
        )
        agent = self._agent([candidate, candidate])

        result = agent.solve("请证明 x=1。", {"idx": 1})

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

    def test_uses_two_short_candidates_for_low_risk_majority_selection(self):
        agent = self._agent(
            ["【最终答案】7", "【最终答案】7", "【最终答案】8"],
        )

        result = agent.solve("计算 3+4。", {"idx": 10})

        self.assertEqual(result["final_response"], "7")
        self.assertEqual(len(agent.client.calls), 2)
        self.assertTrue(any(item["step"] == "majority_vote" for item in result["trace"]))
        self.assertTrue(all(call["max_tokens"] == 6144 for call in agent.client.calls))

    def test_deterministic_decision_coefficient_bypasses_model_sampling(self):
        agent = self._agent([])

        result = agent.solve("已知相关系数r=-0.8，求决定系数R^2。", {"idx": 17})

        self.assertEqual(result["final_response"], "R^2=0.64")
        self.assertEqual(agent.client.calls, [])
        self.assertEqual(result["trace"][0]["step"], "deterministic_solver")

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


class TruncationRecoveryTest(unittest.TestCase):
    def test_recovers_a_labeled_terminal_count_without_a_final_marker(self):
        response = "逐步计数可得。Count = 4 × 3 = 12"
        agent = ReasoningAgent(
            SequencedClient([response, response, response]),
            AgentConfig(use_llm_extraction=False),
        )

        result = agent.solve("在图中求长度为3的简单路径数。", {"idx": 4002})

        self.assertEqual(result["final_response"], "12")
        self.assertEqual(sum(
            item["step"].endswith("_recovered") for item in result["trace"]
        ), 3)

    def test_prompt_marker_and_thinking_are_not_accepted_as_a_proof(self):
        malformed = (
            "Here's a thinking process that leads to the solution.\n"
            "请在最后一行使用【最终答案】<答案>的格式。\n"
            "后面还有未完成的讨论。"
        )
        agent = ReasoningAgent(
            SequencedClient([
                malformed,
                malformed,
                "ANSWER: 该图必含长度为3的路径。\nCONFIDENCE: 95",
            ]),
            AgentConfig(policy_sample_times=3, use_llm_extraction=False),
        )

        result = agent.solve("证明该图必含长度为3的路径。", {"idx": 4003})

        self.assertEqual(result["final_response"], "该图必含长度为3的路径.")
        self.assertNotIn("thinking process", result["final_response"].lower())
        self.assertEqual(len(agent.client.calls), 3)
        fallback = next(item for item in result["trace"] if item["step"] == "all_truncated")
        self.assertEqual(fallback["content"]["verifier_answer"], "该图必含长度为3的路径。")

    def test_real_late_final_marker_is_accepted(self):
        self.assertTrue(ReasoningAgent._has_usable_final_answer(
            "由计算得结果。\n【最终答案】12"
        ))

    def test_embedded_final_marker_is_not_accepted(self):
        self.assertFalse(ReasoningAgent._has_usable_final_answer(
            "Final Answer: 【最终答案】12\nWait, I should check the result again."
        ))

    def test_final_marker_with_trailing_thinking_is_not_accepted(self):
        self.assertFalse(ReasoningAgent._has_usable_final_answer(
            "【最终答案】12\nWait, I should check the result again."
        ))

    def test_malformed_verifier_uses_only_the_requested_proof_claim(self):
        malformed = "Thinking Process:\n请按【最终答案】<答案>格式作答。"
        verifier = "Thinking Process:\n因此图中存在长度为3的路径。"
        agent = ReasoningAgent(
            SequencedClient([malformed, malformed, verifier]),
            AgentConfig(policy_sample_times=3, use_llm_extraction=False),
        )

        result = agent.solve(
            "设简单图有8个顶点且每个顶点度数至少4，证明该图必含长度为3的路径，并给出所用度数条件。",
            {"idx": 4003},
        )

        self.assertEqual(
            result["final_response"],
            "该图必含长度为3的路径；所用度数条件为每个顶点度数至少4",
        )
        fallback = next(item for item in result["trace"] if item["step"] == "all_truncated")
        self.assertEqual(
            fallback["content"]["proof_fallback"],
            "该图必含长度为3的路径；所用度数条件为每个顶点度数至少4",
        )

    def test_english_reasoning_is_retried_and_redacted_from_trace(self):
        english = (
            "First analyze the problem. Then let us compute the result carefully.\n"
            "【最终答案】12"
        )
        chinese = "由计算可得结果。\n【最终答案】12"
        agent = ReasoningAgent(
            SequencedClient([english, chinese]),
            AgentConfig(policy_sample_times=1, use_llm_extraction=False),
        )

        result = agent.solve("计算 3+9。", {"idx": 99})

        self.assertEqual(result["final_response"], "12")
        first_call = next(item for item in result["trace"] if item["step"] == "policy_call_0")
        self.assertEqual(first_call["content"]["language_gate"], "blocked")
        self.assertTrue(first_call["content"]["response"]["redacted"])


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
