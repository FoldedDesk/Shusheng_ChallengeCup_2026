from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.math_equivalence import equivalent_answers


def test_fraction_style_commands_are_equivalent_to_frac_and_exact_decimals():
    assert equivalent_answers(r"\dfrac{1}{2}", r"\frac{1}{2}")
    assert equivalent_answers(r"\tfrac{3}{4}", "0.75")
    assert equivalent_answers(r"x=\tfrac34", "x=0.75")


def test_finite_decimal_comparison_is_exact_not_tolerance_based():
    assert equivalent_answers("1/8", "0.125")
    assert equivalent_answers(r"x=\frac{1}{2}", "0.5")
    assert not equivalent_answers("1/3", "0.333333")
    assert not equivalent_answers(r"x=\frac{1}{2}", "x=0.5001")


def test_indexed_roots_match_exact_power_not_wrong_root():
    assert equivalent_answers(r"\sqrt[3]{2}", r"2^{1/3}")
    assert equivalent_answers(r"\sqrt[4]{16}", "2")
    assert equivalent_answers(r"\sqrt[n]{x}", r"x^{1/n}")
    assert not equivalent_answers(r"\sqrt[3]{2}", r"2^{1/2}")
    assert not equivalent_answers(r"\sqrt[3]{8}", "3")


def test_diagonal_notation_matches_the_same_explicit_matrix():
    diagonal = r"\operatorname{diag}(2,2,12)"
    explicit = r"\begin{pmatrix}2&0&0\\0&2&0\\0&0&12\end{pmatrix}"
    assert equivalent_answers(diagonal, explicit)

    fractional_diagonal = r"\mathrm{diag}(\tfrac12,3)"
    fractional_explicit = r"\begin{bmatrix}0.5&0\\0&3\end{bmatrix}"
    assert equivalent_answers(fractional_diagonal, fractional_explicit)


def test_matrix_comparison_rejects_wrong_entries_and_off_diagonal_terms():
    diagonal = r"\operatorname{diag}(2,2,12)"
    wrong_entry = r"\begin{pmatrix}2&0&0\\0&2&0\\0&0&13\end{pmatrix}"
    wrong_position = r"\begin{pmatrix}2&0&0\\0&12&0\\0&0&2\end{pmatrix}"
    off_diagonal = r"\begin{pmatrix}2&1&0\\0&2&0\\0&0&12\end{pmatrix}"
    assert not equivalent_answers(diagonal, wrong_entry)
    assert not equivalent_answers(diagonal, wrong_position)
    assert not equivalent_answers(diagonal, off_diagonal)


def test_tuple_matches_column_vector_componentwise_but_not_row_vector():
    tuple_value = r"(\frac12,0.25,3)"
    column = r"\begin{pmatrix}0.5\\\frac14\\3\end{pmatrix}"
    wrong_column = r"\begin{pmatrix}0.5\\\frac13\\3\end{pmatrix}"
    row = r"\begin{pmatrix}0.5&\frac14&3\end{pmatrix}"
    assert equivalent_answers(tuple_value, column)
    assert not equivalent_answers(tuple_value, wrong_column)
    assert not equivalent_answers(tuple_value, row)


def test_interval_notation_is_not_conflated_with_tuple_or_wrong_interval():
    assert not equivalent_answers("[1,2]", "(1,2)")
    assert not equivalent_answers("[1,2]", "[1,3]")
    assert not equivalent_answers("(1,2)", "(1,3)")


def test_single_equation_can_omit_only_its_left_hand_side():
    assert equivalent_answers(r"[K:\mathbb Q]=16", "16")
    assert equivalent_answers(r"\lVert T^n\rVert=\frac{1}{n!}", r"\frac{1}{n!}")
    assert equivalent_answers(
        r"A=\operatorname{diag}(2,2,12)",
        r"\begin{pmatrix}2&0&0\\0&2&0\\0&0&12\end{pmatrix}",
    )
    assert not equivalent_answers("x=1", "y=1")
    assert not equivalent_answers("x=1", "2")


def test_multi_assignment_components_use_root_and_decimal_normalization():
    left = r"x=\sqrt[3]{2}; y=\tfrac34"
    right = r"y=0.75; x=2^{1/3}"
    wrong = r"y=0.75; x=2^{1/2}"
    assert equivalent_answers(left, right)
    assert not equivalent_answers(left, wrong)
