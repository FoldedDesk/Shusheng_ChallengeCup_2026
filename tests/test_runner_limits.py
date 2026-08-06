import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import FAILED_ANSWER, preflight_client, process_item


class RunnerLimitsTest(unittest.TestCase):
    def test_runner_records_a_non_empty_answer_when_one_item_raises(self):
        class BrokenAgent:
            def solve(self, problem, metadata):
                raise RuntimeError("simulated failure")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            asyncio.run(process_item(
                BrokenAgent(),
                {"idx": 1, "problem": "测试题"},
                output_dir,
                asyncio.Semaphore(1),
            ))
            record = json.loads((output_dir / "1.json").read_text(encoding="utf-8"))

        self.assertEqual(record["status"], "error")
        self.assertEqual(record["final_response"], FAILED_ANSWER)
        self.assertTrue(record["final_response"].strip())

    def test_preflight_rejects_empty_client_response(self):
        class EmptyClient:
            def chat(self, **kwargs):
                return ""

        with self.assertRaises(RuntimeError):
            preflight_client(EmptyClient())


if __name__ == "__main__":
    unittest.main()
