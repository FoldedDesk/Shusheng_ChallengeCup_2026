"""Conservative equivalence checks for candidate comparison and offline evaluation."""

from __future__ import annotations

from fractions import Fraction
import re

from tools.latex_parser import normalize_latex
from classifier.choice import answer_choice_labels


def equivalent_answers(left: str, right: str) -> bool:
    """Return true only for high-confidence textual or mathematical equivalence."""
    left_value = _answer_value(left)
    right_value = _answer_value(right)
    a = _compact(left_value)
    b = _compact(right_value)
    if not a or not b:
        return False
    if a == b:
        return True

    a_conclusions = _conclusions(left_value)
    b_conclusions = _conclusions(right_value)
    if any(
        x == y
        for x in a_conclusions
        for y in b_conclusions
    ):
        return True

    a_judgement = _judgement(left_value)
    b_judgement = _judgement(right_value)
    if a_judgement and b_judgement and a_judgement != b_judgement:
        return False
    if (
        a_judgement
        and a_judgement == b_judgement
        and _is_bare_judgement(left_value)
        and _is_bare_judgement(right_value)
    ):
        return True

    if _same_indeterminate_variance_direction(left_value, right_value):
        return True

    if _adjoint_product_rule_match(left_value, right_value):
        return True

    a_choice = answer_choice_labels(left_value)
    b_choice = answer_choice_labels(right_value)
    if a_choice or b_choice:
        return bool(a_choice and b_choice and a_choice == b_choice)

    a_numbers = _numbers(left_value)
    b_numbers = _numbers(right_value)
    numeric_a = _is_numeric_answer(left_value)
    numeric_b = _is_numeric_answer(right_value)
    if numeric_a and numeric_b and a_numbers and a_numbers == b_numbers:
        return True
    if a_numbers and a_numbers == b_numbers and _shared_semantic_anchor(left_value, right_value):
        return True

    a_expressions = _expressions(left_value)
    b_expressions = _expressions(right_value)
    return bool(a_expressions and b_expressions and _symbolically_match(a_expressions, b_expressions))


def equivalence_key(value: str) -> str:
    answer = _answer_value(value)
    compact = _compact(answer)
    numbers = _numbers(answer)
    if numbers:
        return f"{_judgement(answer)}|{'|'.join(str(item) for item in numbers)}|{compact[:80]}"
    return compact


def _answer_value(value: str) -> str:
    text = str(value or "").strip()
    try:
        from reasoning.finalizer import Finalizer

        extracted = Finalizer.extract_result(text)
        if extracted.answer and extracted.explicit_answer:
            return extracted.answer
    except Exception:
        pass
    return text


def _is_numeric_answer(value: str) -> bool:
    text = normalize_latex(str(value or "")).strip()
    text = re.sub(r"(?i)^\s*(?:answer|final|答案|结论)\s*[:：]?\s*", "", text)
    return bool(re.fullmatch(
        r"(?:\\boxed\{)?\s*[-+]?(?:\d+(?:\.\d+)?(?:/[-+]?\d+)?|"
        r"\\frac\{[-+]?\d+\}\{[-+]?\d+\})(?:\s*\\?[A-Za-z]+)?\s*(?:\})?",
        text,
    ))


def _compact(value: str) -> str:
    text = normalize_latex(str(value or "")).lower().replace("−", "-")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return re.sub(r"[\s{}\\,，。；;：:`'$]", "", text)


def _judgement(value: str) -> str:
    text = str(value or "")
    negative = re.search(r"(?:不是|不属于|不成立|不可|错误|发散|否)", text)
    positive = re.search(r"(?:是|属于|成立|可以|正确|收敛)", text)
    if negative:
        return "negative"
    if positive:
        return "positive"
    return ""


def _is_bare_judgement(value: str) -> bool:
    text = normalize_latex(str(value or "")).strip(" \t\r\n。.!?")
    return bool(re.fullmatch(
        r"(?:是|否|正确|错误|成立|不成立|可以|不可以|"
        r"yes|no|true|false|correct|incorrect)",
        text,
        re.IGNORECASE,
    ))


def _same_indeterminate_variance_direction(left: str, right: str) -> bool:
    pattern = re.compile(
        r"不一定(?:增大|变大)|可能(?:高估|低估).*(?:低估|高估)|"
        r"(?:高估|低估).*(?:低估|高估)|"
        r"(?:may|can)\s+(?:increase|decrease).*(?:decrease|increase)",
        re.IGNORECASE | re.DOTALL,
    )
    return bool(pattern.search(str(left or "")) and pattern.search(str(right or "")))


def _adjoint_product_rule_match(left: str, right: str) -> bool:
    """Recognize the conservative product-rule form of a formal adjoint."""
    a = _compact(left)
    b = _compact(right)

    def has_divergence(value: str) -> bool:
        return "partial_j(b_jv)" in value and "b_jpartial_jv" not in value

    def has_expansion(value: str) -> bool:
        return "b_jpartial_jv" in value and "(partial_jb_j)v" in value

    shared = (
        "partial_j(a_ijpartial_iv)" in a and "partial_j(a_ijpartial_iv)" in b
        and "cv" in a and "cv" in b
    )
    return bool(shared and (
        (has_divergence(a) and has_expansion(b))
        or (has_divergence(b) and has_expansion(a))
    ))


def _numbers(value: str) -> tuple[Fraction | str, ...]:
    text = str(value or "").replace(r"\frac", "frac")
    fractions = re.findall(r"frac\s*\{?([+-]?\d+)\}?\s*\{?([+-]?\d+)\}?", text)
    consumed = re.sub(r"frac\s*\{?[+-]?\d+\}?\s*\{?[+-]?\d+\}?", " ", text)
    result: list[Fraction | str] = []
    for numerator, denominator in fractions:
        if int(denominator):
            result.append(Fraction(int(numerator), int(denominator)))
    for token in re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?", consumed):
        try:
            result.append(Fraction(token))
        except (ValueError, ZeroDivisionError):
            result.append(token)
    return tuple(result)


def _shared_semantic_anchor(left: str, right: str) -> bool:
    anchors = (
        "概率", "积分", "极限", "误差", "方差", "期望", "定义域", "区间", "阶", "曲率",
        "det", "tr", "rank", "var", "cov", "lim", "p(", "x=", "y=", "="
    )
    a = str(left or "").lower()
    b = str(right or "").lower()
    return any(anchor in a and anchor in b for anchor in anchors)


def _conclusions(value: str) -> tuple[str, ...]:
    text = str(value or "")
    labelled = re.findall(r"(?:结论|最终答案)\s*[:：]?\s*([^。；;\n]+)", text)
    sentences = [item.strip() for item in re.split(r"[。；;\n]+", text) if item.strip()]
    candidates = [*labelled, *(sentences[-2:] if sentences else [])]
    generic = {"结论成立", "命题成立", "得证", "证毕", "成立"}
    return tuple(
        compact for compact in (_compact(item) for item in candidates)
        if compact and compact not in generic
    )


def _expressions(value: str) -> tuple[str, ...]:
    text = str(value or "").replace("−", "-")
    fragments = re.findall(r"(?:[A-Za-z][A-Za-z0-9_{}()]*\s*=\s*)?([^，。；;\n]+)", text)
    return tuple(fragment.strip(" $`") for fragment in fragments if "=" in fragment or re.fullmatch(r"[$\\A-Za-z0-9_{}().+\-*/^\s]+", fragment))


def _symbolically_match(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    try:
        import sympy as sp
        from tools.sympy_tool import SympyTool
    except ImportError:
        return False

    def parse_lambda_polynomial(fragment: str):
        value = fragment.split("=", 1)[-1]
        if r"\lambda" not in value:
            return None
        # ``SympyTool`` deliberately accepts only single-letter symbols.  Use a
        # collision-free placeholder for this one known LaTex variable, then
        # require the result to be a univariate polynomial before comparing it.
        if re.search(r"(?<![A-Za-z\\])L(?![A-Za-z])", value):
            return None
        value = value.replace(r"\lambda", "L")
        value = SympyTool._latex_to_sympy(value)
        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", value):
            return None
        expression = SympyTool()._parse(value)
        symbol = sp.Symbol("L")
        if expression.free_symbols - {symbol} or not expression.is_polynomial(symbol):
            return None
        return sp.Poly(expression, symbol)

    def parse(fragment: str):
        value = fragment.split("=", 1)[-1]
        value = SympyTool._latex_to_sympy(value)
        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", value):
            raise ValueError
        return sp.sympify(value, locals={name: sp.Symbol(name) for name in re.findall(r"\b[A-Za-z]\b", value)})

    for a in left:
        for b in right:
            try:
                polynomial_a = parse_lambda_polynomial(a)
                polynomial_b = parse_lambda_polynomial(b)
                if polynomial_a is not None or polynomial_b is not None:
                    if polynomial_a is not None and polynomial_b is not None and polynomial_a == polynomial_b:
                        return True
                    continue
                if sp.simplify(parse(a) - parse(b)) == 0:
                    return True
            except Exception:
                continue
    return False
