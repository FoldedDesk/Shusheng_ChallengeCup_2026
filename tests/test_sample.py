import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_agent import ReasoningAgent


class FakeClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        if "CHOICE:" in messages[0]["content"]:
            return "CHOICE: 0\nREASON: 候选可用"
        return "必要推导。\n【最终答案】测试答案"


class SampleTest(unittest.TestCase):
    def test_dev_samples_produce_serializable_responses(self):
        path = Path(__file__).resolve().parents[1] / "sample_data" / "dev.jsonl"
        agent = ReasoningAgent(client=FakeClient())
        with path.open(encoding="utf-8") as handle:
            for line in list(handle)[:3]:
                item = json.loads(line)
                result = agent.solve(item["problem"], {"idx": item["idx"]})
                self.assertTrue(result["final_response"])
                json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
