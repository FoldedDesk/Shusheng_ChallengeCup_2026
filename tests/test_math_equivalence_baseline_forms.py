from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.math_equivalence import equivalent_answers


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (r"\dfrac{8}{5}", r"\frac85"),
        (r"\tfrac{-4}{6}", "-2/3"),
        (r"5\sqrt[5]{4}", r"5\cdot2^{2/5}"),
        (r"\lVert T^n\rVert=\frac1{n!}", r"\frac{1}{n!}"),
        (
            r"\operatorname{diag}(2,2,12)",
            r"\begin{pmatrix}2&0&0\\0&2&0\\0&0&12\end{pmatrix}",
        ),
        (
            r"(11/10,1/2)^{\mathsf T}",
            r"\begin{pmatrix}1.1\\0.5\end{pmatrix}",
        ),
        ("5/2", "2.5"),
        (
            r"[18/16.919,18/3.325]\approx[1.064,5.414]",
            "(1.0639,5.4135)",
        ),
        (
            r"否；逐点极限为0，但\lVert f_n\rVert_1=1",
            r"f_n \text{ 在 }L^1\text{ 中不收敛到其逐点极限}",
        ),
        ("(x,y)=(2,3)，最优值为12", "最优解 (2,3), 最优值 12"),
        (r"(x,y)=(2,3)，最优值为12", r"x=2,y=3,\max=12"),
        (
            r"\frac23H_2(1/4)+\frac13",
            r"\frac53-\frac12\log_2 3",
        ),
        (
            r"f(x)=\cos(ax)\ (a\in\mathbb R),\ "
            r"\text{or }f(x)=\cosh(bx)\ (b\in\mathbb R)",
            r"f(x)=\cos(tx)\text{ or }f(x)=\cosh(tx),\ t\in\mathbb R",
        ),
        (
            r"\{0\}\cup\{x^n:n\in\mathbb Z_{\ge0}\}",
            r"P(x)=0\text{ or }P(x)=x^m"
            r"\text{ for some non-negative integer }m",
        ),
        (
            r"f(x)=x^2+cx+d\quad(c,d\in\mathbb R)",
            r"f(x)=x^2+ax+b\text{ for some }a,b\in\mathbb R",
        ),
    ],
)
def test_baseline_equivalent_forms_are_recognized(left, right):
    assert equivalent_answers(left, right)
    assert equivalent_answers(right, left)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (r"\dfrac23", r"\frac34"),
        (r"\sqrt[5]{4}", r"2^{1/2}"),
        (r"\lVert T^n\rVert=1/n!", r"\lVert S^n\rVert=1/n!"),
        (
            r"\operatorname{diag}(2,2,12)",
            r"\begin{pmatrix}2&1&0\\0&2&0\\0&0&12\end{pmatrix}",
        ),
        (
            r"(11/10,1/2)^{\mathsf T}",
            r"\begin{pmatrix}0.5\\1.1\end{pmatrix}",
        ),
        (r"(11/10,1/2)^{\mathsf T}", r"\begin{pmatrix}1.1&0.5\end{pmatrix}"),
        ("2.5", "2.5001"),
        (
            r"[18/16.919,18/3.325]\approx[1.064,5.414]",
            "(1.062,5.414)",
        ),
        (r"[0,1]\approx[0,1]", "(1,0)"),
        ("[0,1]", "(0,1)"),
        (
            r"否；逐点极限为0，但\lVert f_n\rVert_1=1",
            r"f_n\text{ 在 }L^2\text{ 中不收敛到其逐点极限}",
        ),
        (
            r"否；逐点极限为0，但\lVert f_n\rVert_1=1",
            r"f_n\text{ 在 }L^1\text{ 中收敛到其逐点极限}",
        ),
        ("(x,y)=(2,3)，最优值为12", "最优解 (2,3), 最优值 13"),
        ("(x,y)=(2,3)，最优值为12", "最优解 (3,2), 最优值 12"),
        ("(x,y)=(2,3)，最优值为12", "最优值 12"),
        (r"(x,y)=(2,3)，最优值为12", r"x=2,y=4,\max=12"),
        (
            r"\frac23H_2(1/3)+\frac13",
            r"\frac53-\frac12\log_2 3",
        ),
        (
            r"\frac23H_2(1/4)+\frac13",
            r"\frac53-\frac13\log_2 3",
        ),
        (
            r"f(x)=\cos(ax),a\in\mathbb R\text{ or }"
            r"f(x)=\cosh(bx),b\in\mathbb R",
            r"f(x)=\cos(tx),t\in\mathbb R",
        ),
        (
            r"f(x)=\cos(ax),a\in\mathbb R\text{ or }"
            r"f(x)=\cosh(bx),b\in\mathbb R",
            r"f(x)=\cos(nx),n\in\mathbb Z\text{ or }"
            r"f(x)=\cosh(nx),n\in\mathbb Z",
        ),
        (
            r"f(x)=\cos(ax),a\in\mathbb R\text{ or }"
            r"f(x)=\cosh(bx),b\in\mathbb R",
            r"f(x)=\cos(tx)+1,t\in\mathbb R\text{ or }"
            r"f(x)=\cosh(tx),t\in\mathbb R",
        ),
        (
            r"\{0\}\cup\{x^n:n\in\mathbb Z_{\ge0}\}",
            r"P(x)=x^m\text{ for some non-negative integer }m",
        ),
        (
            r"\{0\}\cup\{x^n:n\in\mathbb Z_{\ge0}\}",
            r"P(x)=0\text{ or }P(x)=x^m\text{ for an integer }m\ge1",
        ),
        (
            r"f(x)=x^2+cx+d\quad(c,d\in\mathbb R)",
            r"f(x)=x^2+ax+a\quad(a\in\mathbb R)",
        ),
        (
            r"f(x)=x^2+cx+d\quad(c,d\in\mathbb R)",
            r"f(x)=2x^2+ax+b\quad(a,b\in\mathbb R)",
        ),
    ],
)
def test_nearby_but_different_forms_are_rejected(left, right):
    assert not equivalent_answers(left, right)
    assert not equivalent_answers(right, left)
