"""Conservative equivalence checks for candidate comparison and offline evaluation."""

from __future__ import annotations

from fractions import Fraction
import re

from tools.latex_parser import normalize_latex


def equivalent_answers(left: str, right: str) -> bool:
    """Return true only for high-confidence textual or mathematical equivalence."""
    a = _compact(left)
    b = _compact(right)
    if not a or not b:
        return False
    if a == b or (len(a) >= 4 and len(b) >= 4 and (a in b or b in a)):
        return True

    a_conclusions = _conclusions(left)
    b_conclusions = _conclusions(right)
    if any(
        len(x) >= 4 and len(y) >= 4 and (x == y or x in y or y in x)
        for x in a_conclusions
        for y in b_conclusions
    ):
        return True

    a_judgement = _judgement(left)
    b_judgement = _judgement(right)
    if a_judgement and b_judgement and a_judgement != b_judgement:
        return False

    a_numbers = _numbers(left)
    b_numbers = _numbers(right)
    if a_numbers and a_numbers == b_numbers and _shared_semantic_anchor(left, right):
        return True

    a_expressions = _expressions(left)
    b_expressions = _expressions(right)
    return bool(a_expressions and b_expressions and _symbolically_match(a_expressions, b_expressions))


def equivalence_key(value: str) -> str:
    compact = _compact(value)
    numbers = _numbers(value)
    if numbers:
        return f"{_judgement(value)}|{'|'.join(str(item) for item in numbers)}|{compact[:80]}"
    return compact


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

    def parse(fragment: str):
        value = fragment.split("=", 1)[-1]
        value = SympyTool._latex_to_sympy(value)
        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", value):
            raise ValueError
        return sp.sympify(value, locals={name: sp.Symbol(name) for name in re.findall(r"\b[A-Za-z]\b", value)})

    for a in left:
        for b in right:
            try:
                if sp.simplify(parse(a) - parse(b)) == 0:
                    return True
            except Exception:
                continue
    return False
