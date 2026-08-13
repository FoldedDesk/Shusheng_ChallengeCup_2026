from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.math_equivalence import equivalent_answers


REFERENCE = (
    r"{{0,1,3},{1,2,4},{2,3,5},{3,4,6},{0,4,5},{1,5,6},{0,2,6}}"
)


def test_nested_integer_sets_ignore_inner_and_outer_order():
    reordered = (
        r"\boxed{\{\{3,1,0\},\{4,2,1\},\{5,3,2\},"
        r"\{6,4,3\},\{5,0,4\},\{6,1,5\},\{6,0,2\}\}}"
    )
    assert equivalent_answers(reordered, REFERENCE)


def test_nested_integer_sets_reject_changed_missing_or_extra_blocks():
    variants = (
        REFERENCE.replace("{0,2,6}", "{0,2,5}"),
        REFERENCE.replace(",{0,2,6}", ""),
        REFERENCE[:-1] + ",{0,1,2}}",
        REFERENCE[:-1] + ",{0,1,3}}",
        REFERENCE.replace("{0,1,3}", "{0,0,1,3}", 1),
    )
    for value in variants:
        assert not equivalent_answers(value, REFERENCE)


def test_ordered_tuple_families_are_not_treated_as_nested_sets():
    tuples = "((0,1,3),(1,2,4),(2,3,5),(3,4,6),(0,4,5),(1,5,6),(0,2,6))"
    assert not equivalent_answers(tuples, REFERENCE)


def test_explicit_sixth_roots_accept_fraction_exponent_notation():
    explicit = (
        r"\boxed{0,1,e^{i\frac{\pi}{3}},e^{i\frac{2\pi}{3}},"
        r"-1,e^{i\frac{4\pi}{3}},e^{i\frac{5\pi}{3}}}"
    )
    parameterized = r"\{0\}\cup\{e^{k\pi i/3}:k=0,1,2,3,4,5\}"
    assert equivalent_answers(explicit, parameterized)
