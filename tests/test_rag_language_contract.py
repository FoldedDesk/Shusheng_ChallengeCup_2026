from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from rag.card_retriever import CardRetriever, KnowledgeCard, RetrievalBundle


class RagLanguageContractTest(unittest.TestCase):
    def test_english_olympiad_uses_english_method_and_review_cards(self):
        spec = build_problem_spec(
            "In triangle ABC, the incircle touches BC at D. Prove that angle "
            r"BAD equals angle CAD. Put your final answer in \boxed{}."
        )

        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(bundle.language, "en")
        self.assertEqual(bundle.solve_cards[0].id, "method.olympiad.geometry")
        self.assertEqual(bundle.review_cards[0].id, "check.olympiad.exhaustiveness")
        self.assertGreaterEqual(bundle.solve_scores[0], 9)
        self.assertGreaterEqual(bundle.review_scores[0], 9)
        self.assertIn("Record collinear", bundle.solve_context())
        self.assertIn("Independently check every branch", bundle.review_context())
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", bundle.solve_context()))
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", bundle.review_context()))

    def test_chinese_contract_keeps_chinese_card_rendering(self):
        spec = build_problem_spec("证明紧致空间的闭子集紧致。")

        bundle = CardRetriever().retrieve(spec)

        self.assertEqual(bundle.language, "zh")
        self.assertRegex(bundle.solve_context(), r"[\u4e00-\u9fff]")
        self.assertRegex(bundle.review_context(), r"[\u4e00-\u9fff]")

    def test_loaded_note_without_translation_is_rendered_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "抽象代数.txt"
            note.write_text("Finite field multiplicative groups are cyclic.\n", encoding="utf-8")
            spec = build_problem_spec(
                "For a finite field, find a generator of its multiplicative group."
            )

            bundle = CardRetriever(Path(directory)).retrieve(spec)

        self.assertEqual(bundle.language, "en")
        self.assertTrue(bundle.solve_cards[0].id.startswith("note."))
        self.assertEqual(
            bundle.solve_context(),
            "- Finite field multiplicative groups are cyclic.",
        )

    def test_new_language_fields_are_backwards_compatible(self):
        card = KnowledgeCard("legacy", "method", "answer", "原文", ("legacy",))
        bundle = RetrievalBundle((card,), ())

        self.assertEqual(bundle.language, "zh")
        self.assertEqual(bundle.solve_context(), "- 原文")
        self.assertEqual(card.render("en"), "原文")


if __name__ == "__main__":
    unittest.main()
