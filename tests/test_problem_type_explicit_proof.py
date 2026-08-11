from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_type import classify_problem_type


class ExplicitResultProofClassificationTest(unittest.TestCase):
    def test_explicit_english_proof_request_is_primary(self):
        problems = (
            "Determine, with proof, all integers n for which n^2+n is even.",
            "Find all possible values of x, and provide a rigorous proof.",
            "With a proof, determine whether every finite domain is a field.",
            "Determine whether the sequence converges and prove your conclusion.",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "proof")

    def test_explicit_chinese_proof_request_is_primary(self):
        problems = (
            "求所有满足 n^2+n 为偶数的整数 n，并证明你的结论。",
            "确定该数列是否收敛，并给出严格证明。",
            "求所有可能的函数，且写出完整的论证。",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "proof")

    def test_nonexistence_alternative_remains_calculation(self):
        problems = (
            "Find such an integer n or prove that no such integer exists.",
            "Determine a matrix with this property, or prove no such matrix exists.",
            "Compute the requested invariant,\nor prove that no such invariant exists.",
            "求一个满足条件的整数，否则证明不存在这样的整数。",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "calculation")

    def test_justification_and_exhaustiveness_support_remain_calculation(self):
        problems = (
            "Compute 2+3 and justify your answer.",
            "计算 2+3，并说明理由。",
            "Find all real roots of x^2=1 and prove there are no others.",
            "求 x^2=1 的全部实根并证明没有其他根。",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "calculation")

    def test_leading_proof_is_not_overridden_by_a_trailing_result_clause(self):
        problems = (
            "设关系R定义在整数集上，证明R为等价关系并写出[3]。",
            "设H为群G的指数为2的子群，证明H必为正规子群，并写出左右陪集数目。",
            "若f非负可测且积分为0，证明每个正水平集为零测集，并写出结论。",
            "设曲线曲率恒为0且连通，证明其像包含在一条直线中，写出关键结论。",
            "证明群同态的核为正规子群，并写出共轭封闭计算式。",
            "设平稳过程均值为m，证明协方差只依赖时间差，并写出协方差公式。",
            "证明闭凸集上的最近点若存在则唯一，指出所用严格凸性等式。",
            "试证紧致空间的闭子集紧致，并指出所用开覆盖步骤。",
            "Show every subgroup of index two is normal, and state the coset count.",
            "Demonstrate that the kernel is normal, and write the conjugation identity.",
            "Establish that the nearest point is unique, and identify the strict-convexity identity.",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "proof")

    def test_proof_word_inside_result_support_does_not_take_priority(self):
        problems = (
            "写出逆否命题，并说明用它证明原命题的理由。",
            "Find all roots and prove that there are no others.",
        )

        for problem in problems:
            with self.subTest(problem=problem):
                self.assertEqual(classify_problem_type(problem), "calculation")


if __name__ == "__main__":
    unittest.main()
