from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.math_equivalence import equivalent_answers


def test_equivalent_newton_iteration_and_first_iterate_match_as_a_whole():
    quotient_form = r"x_{n+1}=(x_n^2+3)/(2x_n), x_1=1.75"
    averaged_form = r"x_{n+1}=1/2(x_n+3/x_n), x_1=7/4"

    assert equivalent_answers(quotient_form, averaged_form)


def test_equivalent_newton_latex_forms_accept_spacing_command_between_parts():
    quotient_form = (
        r"x_{n+1} = \frac{x_n^2 + 3}{2x_n}, x_1 = \frac{7}{4}"
    )
    averaged_form = (
        r"x_{n+1} = \frac{1}{2}\left(x_n + \frac{3}{x_n}\right),\ x_1 = 1.75"
    )

    assert equivalent_answers(quotient_form, averaged_form)


def test_newton_answers_with_different_iteration_formula_are_not_equivalent():
    expected = r"x_{n+1}=(x_n^2+3)/(2x_n)，x_1=1.75"
    wrong_formula = r"x_{n+1}=(x_n^2+5)/(2x_n)，x_1=1.75"

    assert not equivalent_answers(expected, wrong_formula)


def test_newton_answers_with_different_first_iterate_are_not_equivalent():
    expected = r"x_{n+1}=(x_n^2+3)/(2x_n), x_1=1.75"
    wrong_iterate = r"x_{n+1}=1/2(x_n+3/x_n), x_1=2"

    assert not equivalent_answers(expected, wrong_iterate)


def test_newton_answer_missing_a_required_component_is_not_equivalent():
    complete = r"x_{n+1}=(x_n^2+3)/(2x_n), x_1=7/4"
    missing_iterate = r"x_{n+1}=1/2(x_n+3/x_n)"

    assert not equivalent_answers(complete, missing_iterate)


def test_shared_assignment_does_not_make_multi_part_answers_equivalent():
    assert not equivalent_answers("x=1，y=2", "x=1，y=3")


def test_reordered_multi_part_assignments_remain_equivalent():
    assert equivalent_answers("x=1/2; y=2", r"y=2; x=\frac{1}{2}")


def test_assignment_list_matches_same_named_results_in_support_body():
    listed = r"x_{n+1}=\frac{x_n^2+3}{2x_n}, x_1=\frac74"
    supported = (
        r"结论：x_1=\frac74" "\n"
        r"牛顿迭代公式为 $x_{n+1}=\frac{x_n^2+3}{2x_n}$，"
        r"代入 $x_0=2$ 得 $x_1=\frac74$。"
    )

    assert equivalent_answers(listed, supported)


def test_assignment_list_matches_final_expression_in_derivation_chain():
    listed = r"x_{n+1}=\frac{x_n^2+3}{2x_n}, x_1=\frac74"
    supported = (
        r"结论：x_1=\frac74" "\n"
        r"代入得 $x_{n+1}=x_n-\frac{x_n^2-3}{2x_n}"
        r"=\frac{x_n^2+3}{2x_n}$。"
    )

    assert equivalent_answers(listed, supported)


def test_support_body_last_named_value_cannot_hide_a_contradiction():
    listed = r"x_{n+1}=\frac{x_n^2+3}{2x_n}, x_1=\frac74"
    contradicted = (
        r"先算得 $x_1=\frac74$，公式为 "
        r"$x_{n+1}=\frac{x_n^2+3}{2x_n}$，但最终 $x_1=2$。"
    )

    assert not equivalent_answers(listed, contradicted)
