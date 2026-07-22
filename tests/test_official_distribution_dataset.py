import json
import unittest
from collections import Counter
from pathlib import Path


BASE_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "official_distribution_112.jsonl"
HARD_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "official_distribution_112_hard.jsonl"
EXPECTED_COUNTS = {
    "离散数学": 24, "数值分析": 13, "测度积分": 11, "微分几何": 9,
    "概率论": 8, "抽象代数": 8, "随机过程": 7, "复分析": 7,
    "常微分方程": 5, "统计推断": 4, "泛函分析": 4, "线性回归": 3,
    "偏微分方程": 3, "非基础及进阶课程": 2, "高等代数": 1,
    "运筹学": 1, "数学分析": 1, "拓扑学": 1,
}


class OfficialDistributionDatasetTest(unittest.TestCase):
    def assert_dataset(self, path, start_idx, source, min_problem_length=1):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 112)
        self.assertEqual([row["idx"] for row in rows], list(range(start_idx, start_idx + 112)))
        self.assertEqual(Counter(row["subject"] for row in rows), EXPECTED_COUNTS)
        for row in rows:
            self.assertEqual(row["source"], source)
            self.assertTrue(all(str(row[key]).strip() for key in ("problem", "answer", "subject")))
            self.assertGreaterEqual(len(row["problem"]), min_problem_length)

    def test_base_dataset_has_required_structure_and_distribution(self):
        self.assert_dataset(BASE_DATASET_PATH, 3000, "official_distribution_112")

    def test_hard_dataset_has_required_structure_and_distribution(self):
        self.assert_dataset(HARD_DATASET_PATH, 4000, "official_distribution_112_hard", min_problem_length=30)


if __name__ == "__main__":
    unittest.main()
