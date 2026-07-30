from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ImportTest(unittest.TestCase):
    def test_reasoning_agent_can_be_imported(self):
        from user_agent import ReasoningAgent

        self.assertTrue(callable(ReasoningAgent))


if __name__ == "__main__":
    unittest.main()
