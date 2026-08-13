from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.math_equivalence import equivalent_answers


def test_matching_final_cannot_hide_conflicting_english_exact_values():
    left = "FINAL: 2\nexact value=8/3"
    right = "FINAL: 2\nexact value=3"

    assert not equivalent_answers(left, right)
    assert not equivalent_answers(right, left)


def test_matching_final_with_equivalent_exact_fields_remains_equivalent():
    left = "FINAL: 2\nexact value=8/3"
    right = r"FINAL: 2" "\n" r"exact value=\frac{8}{3}"

    assert equivalent_answers(left, right)
    assert equivalent_answers(right, left)


def test_matching_final_cannot_hide_conflicting_chinese_approximate_values():
    left = "FINAL: 2\n近似值：2.50"
    right = "FINAL: 2\n近似值：2.6"

    assert not equivalent_answers(left, right)


def test_single_sided_semantic_field_does_not_change_final_equivalence():
    labelled = "FINAL: 2\n精确值：8/3"

    assert equivalent_answers(labelled, "FINAL: 2")


def test_unparseable_semantic_prose_preserves_ordinary_final_behavior():
    left = "FINAL: 2\nexact result: established by the argument above"
    right = "FINAL: 2\nexact result: follows from continuity"

    assert equivalent_answers(left, right)
