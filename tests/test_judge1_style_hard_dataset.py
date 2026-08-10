import json
from pathlib import Path
import unittest

from main import solve_item
from scripts.build_judge1_style_hard112 import OUTPUT, SHARDS, validate


class _CapturingAgent:
    def __init__(self) -> None:
        self.problem = None
        self.metadata = None

    def solve(self, problem, metadata):
        self.problem = problem
        self.metadata = metadata
        return {"final_response": "1", "trace": []}


class Judge1StyleHardDatasetTest(unittest.TestCase):
    @staticmethod
    def _load(path: Path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_generated_dataset_passes_static_contract(self):
        rows = self._load(OUTPUT)

        self.assertEqual(
            validate(rows),
            {"rows": 112, "english": 87, "chinese": 25, "unique": 112},
        )

    def test_generated_dataset_matches_ordered_shards(self):
        expected = [row for path in SHARDS for row in self._load(path)]

        self.assertEqual(self._load(OUTPUT), expected)

    def test_local_reference_fields_are_not_passed_to_solve(self):
        item = self._load(OUTPUT)[0]
        agent = _CapturingAgent()

        output = solve_item(agent, item)

        self.assertEqual(agent.problem, item["problem"])
        self.assertEqual(agent.metadata, {"idx": item["idx"], "source": item["source"]})
        self.assertNotIn("answer", agent.metadata)
        self.assertNotIn("subject", agent.metadata)
        self.assertNotIn(item["answer"], json.dumps(agent.metadata, ensure_ascii=False))
        self.assertEqual(output["final_response"], "1")


if __name__ == "__main__":
    unittest.main()
