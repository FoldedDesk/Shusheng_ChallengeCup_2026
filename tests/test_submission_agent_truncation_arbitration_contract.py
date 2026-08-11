from pathlib import Path
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.model_response import ModelCallResult
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate


class NoopClient:
    def chat_result(self, **kwargs):  # pragma: no cover - these tests do not call the model
        raise AssertionError(f"unexpected model call: {kwargs}")


class StructuredRecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_result(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class SubmissionAgentTruncationArbitrationContractTest(unittest.TestCase):
    def setUp(self):
        self.agent = SubmissionAgent(NoopClient())
        self.spec = build_problem_spec("求这个数值。")

    def _verifier_candidates(self, response: str):
        candidates = self.agent._assess_candidates(
            r"FINAL: \boxed{42}",
            response,
            "",
            "",
            self.spec,
            (),
            second_source="verify",
            second_truncated=True,
        )
        return [
            item for item in candidates
            if self.agent._source_stage(item.source) == "verify"
        ]

    def test_truncated_verifier_keeps_a_closed_first_line_final_with_benign_support(self):
        candidates = self._verifier_candidates(
            "FINAL: \\boxed{42}\n"
            "Verification substitutes the value into the defining relation and then checks"
        )

        self.assertTrue(candidates)
        self.assertTrue(any(item.answer == "42" for item in candidates))
        self.assertTrue(any(
            item.validation_tier in {"complete", "degraded"}
            and "provider_truncated_ambiguous_box" not in item.rejected_reasons
            for item in candidates
        ))

    def test_truncated_verifier_rejects_a_first_line_final_before_reversal_text(self):
        reversals = (
            "Yet a recalculation gives 43.",
            "Actually the corrected value is 43.",
            "但是重新计算得到43。",
            "更正：正确值应为43。",
            r"A conflicting calculation instead gives FINAL: \boxed{43}.",
        )

        for suffix in reversals:
            with self.subTest(suffix=suffix):
                candidates = self._verifier_candidates(
                    f"FINAL: \\boxed{{42}}\n{suffix}"
                )
                self.assertTrue(candidates)
                self.assertTrue(all(item.validation_tier == "rejected" for item in candidates))
                self.assertTrue(any(
                    "provider_truncated_ambiguous_box" in item.rejected_reasons
                    for item in candidates
                ))

    def test_corrected_arbitration_can_select_a_complete_new_answer(self):
        solve = assess_candidate("42", "solve", self.spec, ())
        verify = assess_candidate("43", "verify", self.spec, ())
        corrected = assess_candidate("44", "arbitration", self.spec, ())
        response = (
            r"FINAL: \boxed{44}" "\n"
            r"DECISION: CORRECTED" "\n"
            r"The decisive recomputation is (4\cdot 11=44), not 42 or 43."
        )

        selected, disposition, decision = self.agent._resolve_arbitration(
            response, [solve, verify, corrected]
        )

        self.assertIs(selected, corrected)
        self.assertEqual(selected.answer, "44")
        self.assertEqual(decision, "CORRECTED")
        self.assertTrue(disposition.startswith("corrected"))

    def test_corrected_arbitration_without_a_check_is_rejected_and_falls_back(self):
        solve = assess_candidate("42", "solve", self.spec, ())
        verify = assess_candidate("43", "verify", self.spec, ())
        unsupported = assess_candidate("44", "arbitration", self.spec, ())
        response = r"FINAL: \boxed{44}" "\nDECISION: CORRECTED"

        selected, disposition, decision = self.agent._resolve_arbitration(
            response, [solve, verify, unsupported]
        )

        self.assertIsNot(selected, unsupported)
        self.assertIn(selected, {solve, verify})
        self.assertEqual(decision, "CORRECTED")
        self.assertNotEqual(disposition, "corrected_answer")

    def test_unresolved_arbitration_keeps_primary_candidate(self):
        solve = assess_candidate("42", "solve", self.spec, ())
        verify = assess_candidate("43", "verify", self.spec, ())

        selected, disposition, decision = self.agent._resolve_arbitration(
            "DECISION: UNRESOLVED", [solve, verify]
        )

        self.assertIs(selected, solve)
        self.assertEqual(disposition, "unresolved_fallback")
        self.assertEqual(decision, "UNRESOLVED")

    def test_mismatched_ab_declaration_falls_back_to_candidate_evidence(self):
        solve = assess_candidate("42", "solve", self.spec, ())
        verify = assess_candidate("43", "verify", self.spec, ())
        arbitration = assess_candidate("42", "arbitration", self.spec, ())
        response = (
            r"FINAL: \boxed{42}" "\n"
            "DECISION: B\nThe displayed answer agrees with candidate A."
        )

        selected, disposition, decision = self.agent._resolve_arbitration(
            response, [solve, verify, arbitration]
        )

        self.assertIs(selected, solve)
        self.assertEqual(selected.answer, "42")
        self.assertEqual(decision, "B")
        self.assertEqual(disposition, "decision_answer_mismatch")

    def test_declared_b_maps_matching_refined_conclusion_to_complete_b_body(self):
        solve = assess_candidate("x=1", "solve", self.spec, ())
        verify_body = replace(
            assess_candidate(
                "由积分单调性可得每个水平集为零测集，因此 f=0 a.e.",
                "verify",
                self.spec,
                (),
            ),
            validation_tier="complete",
            complete_goals=True,
            shape_valid=True,
            formatting_valid=True,
            explicit_answer=False,
        )
        verify_conclusion = replace(
            assess_candidate(
                r"\mu(\{f \ge \epsilon\})=0 \text{ 且 } f=0 \text{ a.e.}",
                "verify#2",
                self.spec,
                (),
            ),
            validation_tier="degraded",
            shape_valid=True,
            formatting_valid=True,
            explicit_answer=True,
        )
        arbitration_conclusion = replace(
            assess_candidate(
                r"\text{对任意 } \epsilon>0, \mu(\{f \ge \epsilon\})=0, "
                r"\text{ 从而 } f=0 \text{ a.e.}",
                "arbitration#2",
                self.spec,
                (),
            ),
            validation_tier="degraded",
            shape_valid=True,
            formatting_valid=True,
            explicit_answer=True,
        )

        selected, disposition, decision = self.agent._resolve_arbitration(
            "DECISION: B",
            [solve, verify_body, verify_conclusion, arbitration_conclusion],
        )

        self.assertIs(selected, verify_body)
        self.assertEqual(selected.source, "verify")
        self.assertIn("积分单调性", selected.answer)
        self.assertEqual(disposition, "supports_b")
        self.assertEqual(decision, "B")

    def test_declared_b_rejects_refined_conclusion_with_one_changed_relation(self):
        solve = assess_candidate("x=1", "solve", self.spec, ())
        verify_body = replace(
            assess_candidate("由代入检验可得 x=2, y=3。", "verify", self.spec, ()),
            validation_tier="complete",
            complete_goals=True,
            shape_valid=True,
            formatting_valid=True,
        )
        verify_conclusion = replace(
            assess_candidate(r"x=2 \text{ 且 } y=3", "verify#2", self.spec, ()),
            validation_tier="degraded",
            shape_valid=True,
            formatting_valid=True,
            explicit_answer=True,
        )
        arbitration_conclusion = replace(
            assess_candidate(r"x=2 \text{ 且 } y=4", "arbitration#2", self.spec, ()),
            validation_tier="degraded",
            shape_valid=True,
            formatting_valid=True,
            explicit_answer=True,
        )

        selected, disposition, decision = self.agent._resolve_arbitration(
            "DECISION: B",
            [solve, verify_body, verify_conclusion, arbitration_conclusion],
        )

        self.assertIs(selected, solve)
        self.assertEqual(disposition, "decision_answer_mismatch")
        self.assertEqual(decision, "B")

    def test_4040_end_to_end_arbitration_returns_complete_verify_body(self):
        problem = "若f≥0可测且∫f dμ=0，证明对任意ε>0集合{f≥ε}为零测集，并写出结论。"
        solve = (
            r"FINAL: \boxed{\text{依据：积分单调性；推导：} "
            r"\int f d\mu \ge \int_{\{f \ge \epsilon\}} f d\mu "
            r"\ge \epsilon \mu(\{f \ge \epsilon\}) \text{，因 } "
            r"\int f d\mu=0, \epsilon>0 \text{，故 } "
            r"\mu(\{f \ge \epsilon\})=0 \text{；结论：对任意 } "
            r"\epsilon>0 \text{，集合 } \{f \ge \epsilon\} \text{ 为零测集。}}"
        )
        verify = (
            r"FINAL: \boxed{\mu(\{f \ge \epsilon\})=0 \text{ 且 } "
            r"f=0 \text{ a.e.}}" "\n"
            r"证明：设 $E_\epsilon=\{x\mid f(x)\ge\epsilon\}$。因 $f\ge0$，"
            r"由积分单调性知 $\int f d\mu\ge\int_{E_\epsilon}f d\mu"
            r"\ge\epsilon\mu(E_\epsilon)$。已知 $\int f d\mu=0$ 且 "
            r"$\epsilon>0$，故 $\mu(E_\epsilon)=0$。对任意 $\epsilon>0$ "
            r"均成立，从而 $f=0$ 几乎处处。"
        )
        arbitration = (
            "CHECK: Both candidates use integral monotonicity to prove every positive "
            "level set has measure zero; B also states the required almost-everywhere conclusion.\n"
            "DECISION: B\n"
            r"FINAL: \boxed{\text{对任意 } \epsilon>0, "
            r"\mu(\{f \ge \epsilon\})=0, \text{ 从而 } f=0 \text{ a.e.}}"
        )
        client = StructuredRecordingClient([
            ModelCallResult(solve),
            ModelCallResult(verify),
            ModelCallResult(arbitration),
        ])

        # The stricter 4040 result contract now sends incomplete A to rescue
        # before the historical verify/conflict path can start. Reproduce only
        # those two historical admission decisions; extraction, arbitration,
        # selection, and rendering still execute through production code.
        with (
            patch.object(
                SubmissionAgent,
                "_review_decision",
                return_value=("verify", "historical_4040_independent_check"),
            ),
            patch.object(SubmissionAgent, "_candidate_conflict", return_value=True),
        ):
            result = SubmissionAgent(client).solve(problem, {})

        selection = next(
            item["content"] for item in result["trace"]
            if item["step"] == "selection"
        )
        equivalence = next(
            item["content"] for item in result["trace"]
            if item["step"] == "equivalence"
        )
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(selection["source"], "verify")
        self.assertEqual(equivalence["arbitration_disposition"], "supports_b")
        self.assertIn("积分单调性", result["final_response"])
        self.assertRegex(result["final_response"], r"f\s*=\s*0")
        self.assertTrue(
            "几乎处处" in result["final_response"]
            or "a.e." in result["final_response"]
        )

    def test_literal_arbitration_label_selects_the_declared_candidate(self):
        solve = assess_candidate("错误", "solve", self.spec, ())
        verify = assess_candidate("正确", "verify", self.spec, ())
        arbitration = assess_candidate("A", "arbitration", self.spec, ())

        selected, disposition, decision = self.agent._resolve_arbitration(
            "FINAL: \\boxed{A}\nDECISION: A",
            [solve, verify, arbitration],
        )

        self.assertIs(selected, solve)
        self.assertEqual(decision, "A")
        self.assertEqual(disposition, "label_supports_a")

    def test_arbitration_prompt_requires_actual_answer_and_decisive_check(self):
        request = self.agent._arbitration_request(
            "求这个数值。", self.spec, self.agent.retriever.retrieve(self.spec), [], ()
        )

        self.assertIn("FINAL框内不得只写候选标签A或B", request)
        self.assertIn("CHECK:", request)

    def test_long_boxed_contest_contract_triggers_independent_verification(self):
        problem = (
            "Find the value of the contest quantity described here. The surrounding "
            "prose supplies background for a mathematical competition setting and "
            "deliberately makes the statement long enough for careful reasoning while "
            "requesting only one numerical result. Remember to put your final answer within "
            r"\boxed{}."
        )
        self.assertGreaterEqual(len(problem), 180)
        spec = build_problem_spec(problem)
        self.assertEqual(spec.answer_contract.wrapper, "boxed")
        self.assertTrue(SubmissionAgent._use_deep_reasoning(spec, problem))

        client = StructuredRecordingClient([
            ModelCallResult(r"FINAL: \boxed{42}"),
            ModelCallResult(r"FINAL: \boxed{42}"),
        ])
        result = SubmissionAgent(client).solve(problem, {})
        review = next(
            item["content"] for item in result["trace"]
            if item["step"] == "review_admission"
        )

        self.assertEqual(review["mode"], "verify")
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
