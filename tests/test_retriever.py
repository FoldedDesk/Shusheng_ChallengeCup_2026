from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retriever import LocalRetriever


class LocalRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.retriever = LocalRetriever()

    def test_bundled_knowledge_base_has_broad_offline_coverage(self):
        self.assertGreaterEqual(len(self.retriever.documents), 28)

    def test_retrieves_relevant_chinese_method_cards(self):
        cases = {
            "用牛顿法求方程的迭代公式": "数值分析.txt",
            "证明紧致空间的闭子集仍然紧致": "泛函分析与拓扑.txt",
            "求参数为 p 的伯努利分布方差": "概率统计.txt",
            "计算留数并使用柯西积分公式": "复分析.txt",
            "判断单调收敛定理的适用条件": "测度积分.txt",
            "用中国剩余定理解同余方程组": "数论与代数进阶.txt",
            "使用 KKT 条件求凸优化问题": "优化与运筹学进阶.txt",
            "用 Fourier 级数解热方程边值问题": "偏微分方程进阶.txt",
            "求曲面的第一基本形式和主曲率": "微分几何进阶.txt",
        }
        for query, document_name in cases.items():
            results = self.retriever.retrieve(query, top_k=5)
            self.assertTrue(results)
            self.assertIn(document_name, "\n".join(results))

        self.assertTrue(
            self.retriever.retrieve("用牛顿法求方程的迭代公式", top_k=1)[0]
            .startswith("数值分析.txt:")
        )

    def test_retrieval_stays_local_and_respects_top_k_limit(self):
        results = self.retriever.retrieve("组合计数与图论路径", top_k=99)

        self.assertLessEqual(len(results), 5)
        self.assertTrue(all(":" in result for result in results))


if __name__ == "__main__":
    unittest.main()
