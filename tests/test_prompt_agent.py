import unittest

from user_agent import AgentMessage, ReasoningAgent, _PromptAgent


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
