from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from classifier.target import extract_target_clause
from core.model_response import ModelCallResult
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_result(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


PARAMETER_PROBLEM = (
    "Let m >= 3. Find the largest constant T=T(m) such that the inequality "
    "x_1+x_2 >= T holds for every admissible m-tuple. "
    r"Remember to put your final answer within \boxed{}."
)


class HardProbePipelineRegressionTest(unittest.TestCase):
    def test_target_keeps_leading_find_before_for_which(self):
        problem = (
            "For a real number T, call a configuration feasible when its constraints hold. "
            "Find the minimum value of T for which this is possible."
        )

        self.assertEqual(
            extract_target_clause(problem),
            "Find the minimum value of T for which this is possible",
        )

    def test_target_keeps_every_condition_after_such_that(self):
        problem = (
            "Compute the number of permutations of S such that:\n"
            "- the first element is 0;\n"
            "- for every i, consecutive elements are adjacent;\n"
            "- the resulting path contains every vertex.\n"
            r"Remember to put your final answer within \boxed{}."
        )

        target = extract_target_clause(problem)

        self.assertIn("first element", target)
        self.assertIn("consecutive elements", target)
        self.assertIn("contains every vertex", target)

    def test_parameterized_extremum_requires_parameter_in_final_result(self):
        spec = build_problem_spec(PARAMETER_PROBLEM)

        self.assertEqual(spec.profile.answer_shape, "expression")
        self.assertEqual(spec.profile.topic, "olympiad_inequality")
        self.assertIn("parameter_dependency", spec.risk_flags)
        self.assertIn(
            "parameter_dependency_m",
            {requirement.name for requirement in spec.goals[0].requirements},
        )
        self.assertFalse(assess_candidate("-4", "solve", spec, ()).accepted)
        self.assertTrue(assess_candidate("2-2m", "solve", spec, ()).accepted)

    def test_collection_of_sorted_triangles_routes_to_extremal_combinatorics(self):
        problem = (
            "Find the minimum integer N that satisfies the following condition:\n"
            "Given 2025 non-degenerate triangles, color one side of each green, purple, and orange.\n"
            "Sort the green, purple, and orange lengths separately in decreasing order.\n"
            "The number of indices whose three indexed lengths do not form a triangle is at most N."
        )

        spec = build_problem_spec(problem)

        self.assertEqual(spec.profile.topic, "olympiad_combinatorics")
        self.assertEqual(spec.profile.subject, "离散数学")
        self.assertIn("extremal_two_sided_bound", spec.risk_flags)
        self.assertNotIn("diagram_dependency", spec.risk_flags)
        self.assertIn("Sort the green", spec.goals[0].instruction)

    def test_exhaustive_result_accepts_a_list_but_not_an_uncertified_singleton(self):
        spec = build_problem_spec("Find all possible values of a_2025.")

        singleton = assess_candidate("2026", "solve", spec, ())
        listed = assess_candidate("2026, 2030", "solve", spec, ())
        certified_singleton = assess_candidate(
            "FINAL: \\boxed{2026}\nCHECK: 2026 is the only value; no others satisfy the recurrence.",
            "verify",
            spec,
            (),
        )

        self.assertFalse(singleton.accepted)
        self.assertTrue(listed.accepted)
        self.assertTrue(certified_singleton.accepted)

    def test_truncated_deep_solve_uses_bounded_candidate_audit(self):
        client = RecordingClient([
            ModelCallResult("A long unfinished derivation", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{-4}"),
            ModelCallResult(
                "FINAL: \\boxed{2-2m}\n"
                "VERDICT: CORRECTED\n"
                "CHECK: the universal upper bound is T<=2-2m and a matching construction gives "
                "T>=2-2m; m=3 gives -4 and m=4 gives -6."
            ),
        ])

        result = SubmissionAgent(client).solve(PARAMETER_PROBLEM, {})

        self.assertEqual(result["final_response"], r"\boxed{2-2m}")
        self.assertEqual(len(client.calls), 3)
        self.assertFalse(client.calls[2]["thinking_mode"])
        self.assertEqual(client.calls[2]["max_tokens"], 4096)
        self.assertIn("Candidate recovered from a truncated deep draft", client.calls[2]["messages"][1]["content"])
        self.assertIn("-4", client.calls[2]["messages"][1]["content"])

    def test_uncertified_audit_retries_in_a_fresh_context(self):
        client = RecordingClient([
            ModelCallResult("A long unfinished derivation", finish_reason="length"),
            ModelCallResult(r"FINAL: \boxed{-4}"),
            ModelCallResult(
                "FINAL: \\boxed{2-2m}\nVERDICT: CORRECTED"
            ),
            ModelCallResult(
                "FINAL: \\boxed{2-2m}\n"
                "VERDICT: CORRECTED\n"
                "CHECK: the universal upper bound is T<=2-2m and a matching construction gives "
                "T>=2-2m; m=3 gives -4 and m=4 gives -6."
            ),
        ])

        result = SubmissionAgent(client).solve(PARAMETER_PROBLEM, {})

        self.assertEqual(result["final_response"], r"\boxed{2-2m}")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(client.calls[3]["messages"]), 2)
        self.assertFalse(client.calls[3]["thinking_mode"])


if __name__ == "__main__":
    unittest.main()
