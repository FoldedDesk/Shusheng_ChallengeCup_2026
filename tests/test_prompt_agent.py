import unittest

from user_agent import AgentMessage, _PromptAgent


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


if __name__ == "__main__":
    unittest.main()
