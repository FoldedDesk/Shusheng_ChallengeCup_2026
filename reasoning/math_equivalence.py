"""Conservative equivalence checks for candidate comparison and offline evaluation."""

from __future__ import annotations

from fractions import Fraction
import re

from tools.latex_parser import normalize_latex
from classifier.choice import answer_choice_labels


def equivalent_answers(left: str, right: str) -> bool:
    """Return true only for high-confidence textual or mathematical equivalence."""
    left_raw = str(left or "").strip()
    right_raw = str(right or "").strip()
    # Finalizer intentionally prefers an explicit leading FINAL marker.  Before
    # applying that projection, preserve contradictions in repeated semantic
    # fields: identical headline answers must not hide different exact or
    # approximate values stated in the supporting response.
    if _explicit_semantic_field_conflict(left_raw, right_raw):
        return False
    left_value = _answer_value(left_raw)
    right_value = _answer_value(right_raw)
    a = _compact(left_value)
    b = _compact(right_value)
    if not a or not b:
        return False
    if a == b:
        return True

    # These answer families carry semantics that are lost by generic string
    # or expression comparison.  Each matcher has a deliberately narrow
    # trigger and becomes authoritative once that trigger is present.
    for matcher in (
        _finite_nested_integer_set_match,
        _finite_roots_of_unity_set_match,
        _integer_tuple_parameter_family_match,
        _polynomial_family_match,
        _trigonometric_family_match,
        _entropy_identity_match,
        _optimization_result_match,
        _interval_inequality_match,
        _approximate_exact_pair_match,
        _approximate_interval_match,
        _negative_convergence_match,
    ):
        specialized_match = matcher(left_value, right_value)
        if specialized_match is not None:
            return specialized_match

    # A multi-part answer is one mathematical object, not a bag of formulas.
    # Resolve conservative assignment lists before the legacy conclusion and
    # expression fallbacks, which intentionally look for any shared fragment.
    # Once both (or either) sides clearly claim multiple labelled results, all
    # labels and all corresponding values must agree.
    assignment_match = _assignment_list_match(left_raw, right_raw)
    if assignment_match is not None:
        return assignment_match

    # Compare typed mathematical objects before prose/conclusion heuristics.
    # Once both sides can be parsed as the same kind of object, a disagreement
    # is authoritative: shared digits must not make two different matrices,
    # vectors, roots, or exact scalar values equivalent.
    equation_match = _single_equation_match(left_value, right_value)
    if equation_match is not None:
        return equation_match
    structured_match = _structured_math_match(left_value, right_value)
    if structured_match is not None:
        return structured_match
    scalar_match = _scalar_math_match(left_value, right_value)
    if scalar_match is not None:
        return scalar_match

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


_SEMANTIC_SCALAR_FIELD_LABELS = {
    "exact": (
        r"精确值",
        r"准确值",
        r"确切值",
        r"精确结果",
        r"exact\s+value",
        r"exact\s+result",
    ),
    "approximate": (
        r"近似值",
        r"近似结果",
        r"近似解",
        r"approximate\s+value",
        r"approximate\s+result",
        r"approximation",
    ),
}


def _explicit_semantic_field_conflict(left: str, right: str) -> bool:
    """Detect a parseable disagreement under the same explicit field label.

    This is a veto rather than a general equivalence matcher.  A missing or
    non-scalar field supplies no evidence either way, so ordinary answers and
    proof prose continue through the existing comparison pipeline.
    """
    left_fields = _explicit_semantic_scalar_fields(left)
    right_fields = _explicit_semantic_scalar_fields(right)
    for field in left_fields.keys() & right_fields.keys():
        matched = _math_object_match(left_fields[field], right_fields[field])
        if matched is False:
            return True
    return False


def _explicit_semantic_scalar_fields(value: str) -> dict[str, str]:
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    fields: dict[str, str] = {}
    for field, labels in _SEMANTIC_SCALAR_FIELD_LABELS.items():
        label_pattern = "|".join(f"(?:{label})" for label in labels)
        for match in re.finditer(
            rf"(?:{label_pattern})\s*(?:为|是|is|equals?|=|[:：])?\s*",
            text,
            re.IGNORECASE,
        ):
            # Semantic result fields are conventionally short.  Keeping the
            # extraction on the current clause avoids treating a derivation or
            # a following labelled field as part of the scalar value.
            clause = re.split(r"[\r\n,，;；]", text[match.end():], maxsplit=1)[0]
            candidate = clause.strip().rstrip("。.!?").strip()
            if candidate.startswith("$") and candidate.endswith("$"):
                candidate = candidate[1:-1].strip()
            if candidate.startswith(r"\(") and candidate.endswith(r"\)"):
                candidate = candidate[2:-2].strip()
            if _parse_scalar_expression(candidate) is not None:
                # The last explicit occurrence is the terminal assertion for
                # this field and is the least likely to be an intermediate.
                fields[field] = candidate
    return fields


def _approximate_exact_pair_match(left: str, right: str) -> bool | None:
    """Compare explicitly labelled approximation/exact-value pairs componentwise."""

    def labelled_values(value: str) -> tuple[str, str] | None:
        text = re.sub(
            r"\\(?:text|mathrm)\s*\{([^{}]*)\}",
            r"\1",
            normalize_latex(str(value or "")),
            flags=re.IGNORECASE,
        )
        scalar = (
            r"(?:\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
            r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)"
        )

        def extract(label: str) -> str:
            match = re.search(
                rf"(?:{label})\s*(?:为|是|is|=|[:：])?\s*"
                rf"(?:\$|\\\()?\s*({scalar})",
                text,
                re.IGNORECASE,
            )
            return match.group(1).strip() if match else ""

        approximate = extract(r"近似值|近似结果|approximate\s+value|approximation")
        exact = extract(r"精确值|准确值|exact\s+value")
        return (approximate, exact) if approximate and exact else None

    left_pair = labelled_values(left)
    right_pair = labelled_values(right)
    if left_pair is None and right_pair is None:
        return None
    if left_pair is None or right_pair is None:
        return False
    return all(
        _math_object_match(left_item, right_item)
        for left_item, right_item in zip(left_pair, right_pair)
    )


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
    text = _normalize_fraction_commands(normalize_latex(str(value or ""))).strip()
    text = re.sub(r"(?i)^\s*(?:answer|final|答案|结论)\s*[:：]?\s*", "", text)
    if re.fullmatch(
        r"(?:\\boxed\{)?\s*[-+]?(?:\d+(?:\.\d+)?(?:/[-+]?\d+)?|"
        r"\\frac(?:\{[-+]?\d+\}|[-+]?\d)(?:\{[-+]?\d+\}|[-+]?\d))"
        r"(?:\s*\\?[A-Za-z]+)?\s*(?:\})?",
        text,
    ):
        return True
    expression = _parse_scalar_expression(text)
    return bool(expression is not None and getattr(expression, "is_number", False))


def _compact(value: str) -> str:
    # Preserve norm delimiters before normalize_latex canonicalizes both
    # single and double vertical bars to ``|``.  Norm notations remain
    # mutually equivalent without becoming equal to absolute value.
    source = str(value or "")
    source = (
        source.replace(r"\lVert", "__norm__")
        .replace(r"\rVert", "__norm__")
        .replace(r"\Vert", "__norm__")
        .replace(r"\|", "__norm__")
        .replace("||", "__norm__")
    )
    text = _normalize_fraction_commands(
        normalize_latex(source)
    ).lower().replace("−", "-")
    text = (
        text.replace(r"\lvert", "|")
        .replace(r"\rvert", "|")
        .replace(r"\vert", "|")
        .replace(r"\|", "|")
    )
    for _ in range(3):
        flattened = re.sub(
            r"\\(?:text|mathrm)\s*\{([^{}]*)\}",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        if flattened == text:
            break
        text = flattened
    return re.sub(r"[\s{}\\,，。；;：:`'$]", "", text)


def _normalize_fraction_commands(value: str) -> str:
    """Normalize TeX fraction style without changing the represented value."""
    text = str(value or "").replace(r"\dfrac", r"\frac").replace(
        r"\tfrac", r"\frac"
    )
    # TeX permits a single unbraced token in either argument.  Bracing only
    # these deliberately narrow forms lets the restricted parser handle
    # ``\frac12`` while avoiding guesses about longer unbraced expressions.
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac\s*([A-Za-z0-9])\s*\{([^{}]+)\}",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        text,
    )
    return text


def _strip_math_wrappers(value: str) -> str:
    text = _normalize_fraction_commands(normalize_latex(str(value or ""))).strip()
    text = re.sub(
        r"(?i)^\s*(?:【\s*)?(?:最终答案|答案|结论|final\s+answer|answer)"
        r"(?:\s*】)?\s*[:：]?\s*",
        "",
        text,
    ).strip()
    boxed = re.fullmatch(r"\\boxed\s*\{(.*)\}\s*", text, re.DOTALL)
    if boxed:
        text = boxed.group(1).strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    elif text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()
    return text.strip(" \t\r\n。.!?")


def _single_equation_match(left: str, right: str) -> bool | None:
    """Compare one labelled equality with its labelled or bare right side."""
    left_equation = _split_single_top_level_equals(_strip_math_wrappers(left))
    right_equation = _split_single_top_level_equals(_strip_math_wrappers(right))
    if left_equation is None and right_equation is None:
        return None
    if left_equation is not None and right_equation is not None:
        left_lhs, left_rhs = left_equation
        right_lhs, right_rhs = right_equation
        if _compact(left_lhs) != _compact(right_lhs):
            return False
        return _math_object_match(left_rhs, right_rhs)
    if left_equation is not None:
        return _math_object_match(left_equation[1], right)
    return _math_object_match(left, right_equation[1])


_MATRIX_ENVIRONMENT = re.compile(
    r"^\s*\\begin\{(?P<env>matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|smallmatrix)\}"
    r"(?P<body>.*?)\\end\{(?P=env)\}\s*$",
    re.DOTALL,
)


def _split_top_level_items(value: str, separators: str = ",，") -> list[str]:
    pieces: list[str] = []
    start = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(value):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            char in separators
            and round_depth == square_depth == brace_depth == 0
        ):
            pieces.append(value[start:index].strip())
            start = index + 1
    pieces.append(value[start:].strip())
    return pieces


def _parse_matrix(value: str) -> tuple[tuple[str, ...], ...] | None:
    text = _strip_math_wrappers(value)
    diagonal = re.fullmatch(
        r"(?:\\operatorname\s*\{\s*diag\s*\}|\\mathrm\s*\{\s*diag\s*\}|"
        r"\\text\s*\{\s*diag\s*\}|diag)\s*\((.*)\)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if diagonal:
        entries = _split_top_level_items(diagonal.group(1))
        if len(entries) < 2 or any(not item for item in entries):
            return None
        return tuple(tuple(
            entries[row] if row == column else "0"
            for column in range(len(entries))
        ) for row in range(len(entries)))

    matrix = _MATRIX_ENVIRONMENT.fullmatch(text)
    if not matrix:
        return None
    rows = [
        row.strip()
        for row in re.split(r"\\\\(?:\s*\[[^\]]*\])?", matrix.group("body"))
        if row.strip()
    ]
    parsed = tuple(tuple(cell.strip() for cell in row.split("&")) for row in rows)
    if not parsed or any(not row or any(not cell for cell in row) for row in parsed):
        return None
    if len({len(row) for row in parsed}) != 1:
        return None
    return parsed


def _parse_tuple_vector(value: str) -> tuple[str, ...] | None:
    text = _strip_math_wrappers(value)
    transpose = re.search(
        r"\^\s*(?:\{\s*(?:\\(?:mathsf|mathrm)\s*\{?\s*T\s*\}?|"
        r"\\top|T)\s*\}|\\top|T)\s*$",
        text,
    )
    if transpose:
        text = text[:transpose.start()].rstrip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    entries = _split_top_level_items(text[1:-1])
    if len(entries) < 2 or any(not entry for entry in entries):
        return None
    return tuple(entries)


def _matrix_entries_match(
    left: tuple[tuple[str, ...], ...],
    right: tuple[tuple[str, ...], ...],
) -> bool:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right)
    ):
        return False
    return all(
        _scalar_entry_match(left_entry, right_entry)
        for left_row, right_row in zip(left, right)
        for left_entry, right_entry in zip(left_row, right_row)
    )


def _scalar_entry_match(left: str, right: str) -> bool:
    match = _scalar_math_match(left, right)
    return match if match is not None else _compact(left) == _compact(right)


def _structured_math_match(left: str, right: str) -> bool | None:
    left_matrix = _parse_matrix(left)
    right_matrix = _parse_matrix(right)
    left_tuple = _parse_tuple_vector(left)
    right_tuple = _parse_tuple_vector(right)

    if left_matrix is not None or right_matrix is not None:
        if left_matrix is not None and right_matrix is not None:
            return _matrix_entries_match(left_matrix, right_matrix)
        if left_matrix is not None and right_tuple is not None:
            if any(len(row) != 1 for row in left_matrix):
                return False
            return len(left_matrix) == len(right_tuple) and all(
                _scalar_entry_match(row[0], entry)
                for row, entry in zip(left_matrix, right_tuple)
            )
        if right_matrix is not None and left_tuple is not None:
            if any(len(row) != 1 for row in right_matrix):
                return False
            return len(right_matrix) == len(left_tuple) and all(
                _scalar_entry_match(entry, row[0])
                for entry, row in zip(left_tuple, right_matrix)
            )
        return False

    if left_tuple is not None and right_tuple is not None:
        return len(left_tuple) == len(right_tuple) and all(
            _scalar_entry_match(a, b) for a, b in zip(left_tuple, right_tuple)
        )
    return None


def _replace_indexed_roots(value: str) -> str:
    r"""Convert balanced ``\sqrt[index]{radicand}`` forms to exact powers."""
    text = str(value or "")
    search_from = 0
    while True:
        match = re.search(r"\\sqrt\s*\[([^\[\]]+)\]\s*\{", text[search_from:])
        if not match:
            return text
        start = search_from + match.start()
        brace_start = search_from + match.end() - 1
        depth = 0
        end = None
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            return text
        root_index = match.group(1).strip()
        radicand = text[brace_start + 1:end]
        replacement = f"(({radicand})^(1/({root_index})))"
        text = text[:start] + replacement + text[end + 1:]
        search_from = start + len(replacement)


def _decimal_literals_to_rationals(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        fraction = Fraction(match.group(0))
        return f"({fraction.numerator}/{fraction.denominator})"

    return re.sub(
        r"(?<![A-Za-z0-9_.])(?:\d+\.\d+|\.\d+)(?![A-Za-z0-9_.])",
        replacement,
        value,
    )


def _parse_scalar_expression(value: str):
    text = _strip_math_wrappers(value)
    if not text or len(text) > 600 or "=" in text or "&" in text:
        return None
    if _MATRIX_ENVIRONMENT.search(text) or len(_split_top_level_items(text)) > 1:
        return None
    text = _replace_indexed_roots(_normalize_fraction_commands(text))
    text = text.replace(r"\lambda", "L").replace(r"\cdot", "*").replace(r"\times", "*")
    try:
        from tools.sympy_tool import SympyTool

        prepared = SympyTool._latex_to_sympy(text)
        prepared = _decimal_literals_to_rationals(prepared)
        if not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]+", prepared):
            return None
        expression = SympyTool()._parse(prepared)
        if not hasattr(expression, "free_symbols") or getattr(expression, "is_Matrix", False):
            return None
        return expression
    except Exception:
        return None


def _scalar_math_match(left: str, right: str) -> bool | None:
    left_expression = _parse_scalar_expression(left)
    right_expression = _parse_scalar_expression(right)
    if left_expression is None or right_expression is None:
        return None
    if left_expression.free_symbols != right_expression.free_symbols:
        return False
    try:
        import sympy as sp

        return bool(sp.simplify(left_expression - right_expression) == 0)
    except Exception:
        return False


def _math_object_match(left: str, right: str) -> bool | None:
    if _compact(left) == _compact(right):
        return True
    structured = _structured_math_match(left, right)
    if structured is not None:
        return structured
    return _scalar_math_match(left, right)


def _unwrap_text_commands(value: str) -> str:
    """Keep simple TeX prose while removing its presentation wrapper."""
    text = str(value or "")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1 ", text)
    return text.replace(r"\left", "").replace(r"\right", "")


def _real_parameter_names(value: str) -> set[str]:
    text = _unwrap_text_commands(value).lower()
    text = re.sub(
        r"\\(?:mathbb|mathbf)\s*(?:\{\s*r\s*\}|r)",
        " REAL ",
        text,
    )
    text = text.replace(r"\in", " in ")
    names: set[str] = set()
    patterns = (
        r"(?P<names>[a-z](?:\s*,\s*[a-z])*)\s+in\s+REAL",
        r"\bfor\s+(?P<names>[a-z](?:\s*,\s*[a-z])*)\s+in\s+REAL",
        r"(?:for\s+some\s+)?(?P<names>[a-z](?:\s*,\s*[a-z])*)"
        r"\s+(?:are|is|in)\s+(?:arbitrary\s+)?real(?:\s+numbers?)?",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            names.update(re.findall(r"[a-z]", match.group("names")))
    return names


def _trigonometric_family_signature(value: str) -> frozenset[str] | tuple[()] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value)).lower()
    real_parameters = _real_parameter_names(text)
    if not real_parameters or not re.search(r"\\?(?:cosh|cos)\s*\(", text):
        return None
    if re.search(r"\\mathbb\s*(?:\{\s*z\s*\}|z)|\bintegers?\b|"
                 r"(?:\\ge|>=|\\le|<=)\s*0", text):
        return ()

    call_pattern = re.compile(
        r"(?<![A-Za-z0-9_)])\\?(?P<name>cosh|cos)\s*\(\s*"
        r"(?P<parameter>[a-z])\s*(?:\\cdot|\*)?\s*x\s*\)",
    )
    calls = list(call_pattern.finditer(text))
    all_calls = list(re.finditer(r"\\?(?:cosh|cos)\s*\(", text))
    if not calls or len(calls) != len(all_calls):
        return ()
    for match in calls:
        if match.group("parameter") not in real_parameters:
            return ()
        suffix = text[match.end():]
        next_token = re.search(r"\S", suffix)
        if next_token and next_token.group(0) in "+-*/^":
            return ()
    return frozenset(match.group("name") for match in calls)


def _trigonometric_family_match(left: str, right: str) -> bool | None:
    left_signature = _trigonometric_family_signature(left)
    right_signature = _trigonometric_family_signature(right)
    if left_signature is None and right_signature is None:
        return None
    return bool(
        left_signature
        and right_signature
        and left_signature == right_signature
    )


def _has_nonnegative_integer_domain(value: str, parameter: str) -> bool:
    text = _unwrap_text_commands(value).lower()
    escaped = re.escape(parameter.lower())
    latex_integer = re.search(
        rf"{escaped}\s*\\in\s*\\mathbb\s*(?:\{{\s*z\s*\}}|z)"
        rf"\s*_\s*\{{?\s*(?:\\ge|>=)\s*0\s*\}}?",
        text,
    )
    natural_zero = re.search(
        rf"{escaped}\s*\\in\s*\\mathbb\s*(?:\{{\s*n\s*\}}|n)"
        rf"\s*_\s*\{{?\s*0\s*\}}?",
        text,
    )
    prose_integer = re.search(
        rf"(?:non[-\s]?negative\s+integer\s+{escaped}|"
        rf"{escaped}\s+(?:is\s+)?(?:a\s+)?non[-\s]?negative\s+integer)",
        text,
    )
    bounded_integer = (
        bool(re.search(r"\\mathbb\s*(?:\{\s*z\s*\}|z)|\bintegers?\b", text))
        and bool(re.search(rf"{escaped}\s*(?:\\ge|>=)\s*0", text))
    )
    return bool(latex_integer or natural_zero or prose_integer or bounded_integer)


def _integer_tuple_parameter_signature(
    value: str,
) -> tuple[str, int, tuple[str, ...], bool] | None:
    """Parse a narrowly stated ordered-tuple family with an integer lower bound."""
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    text = (
        text.replace("$", "")
        .replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(r"\ ", " ")
        .replace("≥", r"\ge")
    )
    text = re.sub(r"\\(?:quad|qquad|,|;)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    integer_domain = re.search(
        r"(?P<parameter>[A-Za-z])\s*\\in\s*\\mathbb\s*"
        r"(?:\{\s*Z\s*\}|Z)\s*_\s*\{?\s*(?:\\ge|>=)\s*"
        r"(?P<lower>-?\d+)\s*\}?",
        text,
        re.IGNORECASE,
    )
    prose_domain = re.search(
        r"(?:for\s+)?(?P<parameter>[A-Za-z])\s*(?:\\ge|>=)\s*"
        r"(?P<lower>-?\d+)",
        text,
        re.IGNORECASE,
    )
    domain = integer_domain or prose_domain
    if domain is None:
        return None

    prefix = text[:domain.start()].rstrip(" ,:;")
    prefix = re.sub(r"^\s*\\\{\s*", "", prefix)
    prefix = re.sub(r"\s*\\\}\s*$", "", prefix)
    prefix = re.sub(r"\bfor\s*$", "", prefix, flags=re.IGNORECASE).rstrip(" ,:;")
    if not (prefix.startswith("(") and prefix.endswith(")")):
        return None
    entries = tuple(_split_top_level_items(prefix[1:-1]))
    if len(entries) < 2 or any(not entry for entry in entries):
        return None
    suffix = text[domain.end():].strip()
    if suffix not in {"", r"\}"}:
        return None
    return (
        domain.group("parameter").lower(),
        int(domain.group("lower")),
        entries,
        integer_domain is not None,
    )


def _integer_tuple_parameter_family_match(left: str, right: str) -> bool | None:
    """Compare equivalent set-builder and prose-domain tuple families."""
    left_signature = _integer_tuple_parameter_signature(left)
    right_signature = _integer_tuple_parameter_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    left_parameter, left_lower, left_entries, left_explicit = left_signature
    right_parameter, right_lower, right_entries, right_explicit = right_signature
    if not left_explicit and not right_explicit:
        return None
    if (
        left_parameter != right_parameter
        or left_lower != right_lower
        or len(left_entries) != len(right_entries)
    ):
        return False
    return all(
        _scalar_entry_match(left_entry, right_entry)
        for left_entry, right_entry in zip(left_entries, right_entries)
    )


def _monomial_family_signature(value: str) -> tuple[str, bool] | tuple[()] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value)).lower()
    powers = list(re.finditer(
        r"(?<![A-Za-z])(?P<base>[a-z])\s*\^\s*\{?\s*"
        r"(?P<parameter>[a-z])\s*\}?",
        text,
    ))
    if not powers or not re.search(
        r"\\mathbb\s*(?:\{\s*[zn]\s*\}|[zn])|\binteger\b",
        text,
    ):
        return None
    if len(powers) != 1:
        return ()
    power = powers[0]
    base = power.group("base")
    parameter = power.group("parameter")
    if not _has_nonnegative_integer_domain(text, parameter):
        return ()
    prefix = text[:power.start()].rstrip()
    if prefix and prefix[-1] not in "{=(,:;" and not re.search(
        r"(?:or|或)\s*(?:[a-z]\s*\([^)]*\)\s*=)?\s*$",
        prefix,
    ):
        return ()
    has_zero = bool(
        re.search(r"\\?\{\s*0\s*\\?\}", text)
        or re.search(r"[a-z]\s*\(\s*[a-z]\s*\)\s*=\s*0", text)
    )
    if not re.search(r"\\cup|\bor\b|或", text):
        return ()
    return base, has_zero


def _quadratic_family_signature(value: str) -> tuple[str, str] | tuple[()] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    real_parameters = _real_parameter_names(text)
    if not real_parameters or not re.search(r"[A-Za-z]\s*\^\s*\{?\s*2\s*\}?", text):
        return None
    equation = _split_single_top_level_equals(text)
    if equation is None:
        return ()
    lhs, rhs = equation
    variable_match = re.search(r"[A-Za-z]\s*\(\s*([A-Za-z])\s*\)", lhs)
    if not variable_match:
        return ()
    variable_name = variable_match.group(1)
    rhs = re.split(
        r"\\quad|\\qquad|\(\s*[A-Za-z]\s*,|\bfor\s+(?:some\s+)?"
        r"[a-z](?:\s*,\s*[a-z])*(?:\s+(?:in|are|is)\b|\s*\\in)|\bwhere\b",
        rhs,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    rhs = re.sub(r"(?<=[A-Za-z])(?=[A-Za-z])", "*", rhs)
    expression = _parse_scalar_expression(rhs)
    if expression is None:
        return ()
    try:
        import sympy as sp

        variable = sp.Symbol(variable_name)
        polynomial = sp.Poly(expression, variable)
        if polynomial.degree() != 2 or sp.simplify(polynomial.nth(2) - 1) != 0:
            return ()
        linear = polynomial.nth(1)
        constant = polynomial.nth(0)
        if not isinstance(linear, sp.Symbol) or not isinstance(constant, sp.Symbol):
            return ()
        if linear == constant or variable in (linear, constant):
            return ()
        if expression.free_symbols != {variable, linear, constant}:
            return ()
        if {str(linear).lower(), str(constant).lower()} - real_parameters:
            return ()
        return variable_name.lower(), "monic-quadratic-free-linear-constant"
    except Exception:
        return ()


def _polynomial_family_match(left: str, right: str) -> bool | None:
    left_monomial = _monomial_family_signature(left)
    right_monomial = _monomial_family_signature(right)
    if left_monomial is not None or right_monomial is not None:
        return bool(
            left_monomial
            and right_monomial
            and left_monomial == right_monomial
        )

    left_quadratic = _quadratic_family_signature(left)
    right_quadratic = _quadratic_family_signature(right)
    if left_quadratic is None and right_quadratic is None:
        return None
    return bool(
        left_quadratic
        and right_quadratic
        and left_quadratic == right_quadratic
    )


def _finite_roots_of_unity_signature(value: str) -> frozenset[int] | tuple[()] | None:
    """Canonicalize the narrow finite family ``0 plus all m-th roots``.

    This intentionally does not attempt generic complex-set parsing.  It only
    recognizes a consecutive parameter range or an explicit list whose terms
    can be reduced exactly to powers of one primitive root.
    """
    text = _unwrap_text_commands(_strip_math_wrappers(value)).lower()
    text = (
        text.replace(r"\;", "")
        .replace(r"\,", "")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
    )
    # A finite root list may be written either as a bare set/list or as one
    # assignment such as ``z=0,1,e^{...}``.  Strip only that leading scalar
    # target; arbitrary equations and later assignments remain unsupported.
    text = re.sub(r"^\s*[a-z]\s*=\s*(?=0\s*[,，])", "", text, count=1)
    text = re.sub(
        r"\\(?:d?frac)\s*\{\s*(?:(\d+)\s*)?\\?pi\s*\}\s*\{\s*(\d+)\s*\}",
        lambda match: f"{match.group(1) or ''}pi/{match.group(2)}",
        text,
    )
    if not re.search(r"e\s*\^", text) or not re.search(r"\{.*\}", text, re.DOTALL):
        return None
    # Strip recognized exponent/range payloads, then reject any standalone
    # integer set member other than the separately normalized 0, 1, and -1.
    # Without this guard, adding an unrelated element such as 2 was silently
    # ignored and two different finite sets were treated as equivalent.
    residue_scan = re.sub(r"e\s*\^\s*\{[^{}]+\}", "ROOT", text)
    residue_scan = re.sub(
        r"[a-z]\s*=\s*0(?:\s*[,，]\s*\d+)+",
        "RANGE",
        residue_scan,
    )
    explicit_integers = {
        int(token)
        for token in re.findall(
            r"(?:^|[\{,，;；])\s*([-+]?\d+)\s*(?=[,，;；\}]|$)",
            residue_scan,
        )
    }
    if explicit_integers - {-1, 0, 1}:
        return ()

    parameterized = re.search(
        r"e\s*\^\s*\{?\s*i\s*\\?pi\s*(?P<parameter>[a-z])\s*/\s*(?P<half>\d+)\s*\}?",
        text,
    ) or re.search(
        r"e\s*\^\s*\{?\s*(?P<parameter>[a-z])\s*\\?pi\s*i\s*/\s*(?P<half>\d+)\s*\}?",
        text,
    )
    if parameterized:
        parameter = re.escape(parameterized.group("parameter"))
        upper = re.search(
            rf"{parameter}\s*=\s*0\s*[,，]\s*1(?:\s*[,，]\s*\d+)+",
            text,
        )
        values = [int(item) for item in re.findall(r"\d+", upper.group(0))] if upper else []
        half = int(parameterized.group("half"))
        order = 2 * half
        if values == list(range(order)):
            has_zero = bool(re.search(r"(?:^|[\{,，;；]|\bor\b)\s*(?:[a-z]\s*=\s*)?0(?:\s|[,，;；\}]|$)", text))
            return frozenset({-1, *range(order)}) if has_zero else frozenset(range(order))
        return ()

    # The Judge-style family has denominator 3 and hence order 6.  Each
    # exponential must be a recognizable integer multiple of pi/3.
    order = 6
    residues: set[int] = set()
    if re.search(r"(?:^|[\{,，])\s*0\s*(?:[,，\}]|$)", text):
        residues.add(-1)
    if re.search(r"(?:^|[\{,，])\s*1\s*(?:[,，\}]|$)", text):
        residues.add(0)
    if re.search(r"(?:^|[\{,，])\s*-1\s*(?:[,，\}]|$)", text):
        residues.add(3)
    exponentials = re.findall(r"e\s*\^\s*\{([^{}]+)\}", text)
    if not exponentials:
        return ()
    for exponent in exponentials:
        compact = re.sub(r"\s+", "", exponent).replace(r"\pi", "pi")
        match = re.fullmatch(
            r"i(?:(\d+)\*?)?pi/3|i(\d+)pi/3|i?pii?/3|i(\d+)\\?pi/3",
            compact,
        )
        if not match:
            return ()
        multiplier = next((int(group) for group in match.groups() if group), 1)
        residues.add(multiplier % order)
    if re.search(r"(?:^|[\{,，])\s*0\s*(?:[,，\}]|$)", text):
        residues.add(-1)
    if re.search(r"(?:^|[\{,，])\s*1\s*(?:[,，\}]|$)", text):
        residues.add(0)
    if re.search(r"(?:^|[\{,，])\s*-1\s*(?:[,，\}]|$)", text):
        residues.add(3)
    # The explicit notation conventionally writes the real roots as 1 and -1
    # instead of repeating e^0 and e^{i pi}; normalize those omissions.
    return frozenset(residues)


def _finite_nested_integer_set_signature(
    value: str,
) -> frozenset[frozenset[int]] | tuple[()] | None:
    """Parse only a finite set whose members are finite integer sets."""
    text = _strip_math_wrappers(value)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = re.sub(r"[\s$]", "", text)
    integer_set = r"\{[-+]?\d+(?:[,，][-+]?\d+)*\}"
    if not re.fullmatch(rf"\{{{integer_set}(?:[,，]{integer_set})*\}}", text):
        return () if re.search(r"\{\{.*\}\}", text, re.DOTALL) else None

    members: list[frozenset[int]] = []
    for payload in re.findall(r"\{([^{}]+)\}", text[1:-1]):
        values = [int(item) for item in re.split(r"[,，]", payload)]
        member = frozenset(values)
        if len(member) != len(values):
            return ()
        members.append(member)
    signature = frozenset(members)
    if len(signature) != len(members):
        return ()
    return signature


def _finite_nested_integer_set_match(left: str, right: str) -> bool | None:
    left_signature = _finite_nested_integer_set_signature(left)
    right_signature = _finite_nested_integer_set_signature(right)
    if left_signature is None and right_signature is None:
        return None
    return bool(
        left_signature
        and right_signature
        and left_signature == right_signature
    )


def _finite_roots_of_unity_set_match(left: str, right: str) -> bool | None:
    left_signature = _finite_roots_of_unity_signature(left)
    right_signature = _finite_roots_of_unity_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return None
    return bool(left_signature and right_signature and left_signature == right_signature)


def _expand_binary_entropy(value: str):
    text = _normalize_fraction_commands(_strip_math_wrappers(value))
    text = (
        text.replace(r"\left", "")
        .replace(r"\right", "")
        .replace(r"\!", "")
        .replace(r"\,", "")
    )
    text = re.sub(
        r"\\(?:d?frac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}",
        r"(\1/\2)",
        text,
    )
    text = re.sub(
        r"\(\(\s*([-+]?\d+\s*/\s*[-+]?\d+)\s*\)\)",
        r"(\1)",
        text,
    )
    entropy_pattern = re.compile(
        r"(?:\\?H)\s*_\s*\{?\s*2\s*\}?\s*"
        r"\(\s*(?P<argument>[+-]?\d+(?:\s*/\s*[+-]?\d+)?)\s*\)",
        re.IGNORECASE,
    )
    text, entropy_count = entropy_pattern.subn(
        lambda match: (
            "(-({p})*log({p})/log(2)-(1-({p}))*log(1-({p}))/log(2))"
        ).format(p=match.group("argument")),
        text,
    )
    log_pattern = re.compile(
        r"\\?log\s*_\s*\{?\s*2\s*\}?\s*"
        r"(?:\{\s*(?P<braced>[+-]?\d+(?:\s*/\s*[+-]?\d+)?)\s*\}|"
        r"\(\s*(?P<parenthesized>[+-]?\d+(?:\s*/\s*[+-]?\d+)?)\s*\)|"
        r"(?P<bare>[+-]?\d+(?:\s*/\s*[+-]?\d+)?))",
        re.IGNORECASE,
    )
    text = log_pattern.sub(
        lambda match: "(log({})/log(2))".format(
            match.group("braced")
            or match.group("parenthesized")
            or match.group("bare")
        ),
        text,
    )
    expression = _parse_scalar_expression(text)
    return expression, entropy_count


def _entropy_identity_match(left: str, right: str) -> bool | None:
    left_expression, left_entropy = _expand_binary_entropy(left)
    right_expression, right_entropy = _expand_binary_entropy(right)
    if not left_entropy and not right_entropy:
        return None
    if left_expression is None or right_expression is None:
        return False
    if left_expression.free_symbols or right_expression.free_symbols:
        return False
    try:
        import sympy as sp

        return bool(sp.simplify(left_expression - right_expression) == 0)
    except Exception:
        return False


def _optimization_signature(value: str) -> tuple[tuple[str, ...], str] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    coordinate = re.search(
        r"\(\s*(?:[A-Za-z]\s*,\s*)+[A-Za-z]\s*\)\s*=\s*"
        r"(?P<tuple>\([^()]+\))",
        text,
    )
    if coordinate is None:
        coordinate = re.search(
            r"(?:最优解|optimal\s+solution|optimizer)\s*"
            r"(?:为|是|is|equals?|=|[:：])?\s*(?P<tuple>\([^()]+\))",
            text,
            re.IGNORECASE,
        )
    coordinate_entries: tuple[str, ...] | None = None
    if coordinate is None:
        assignments = re.findall(
            r"(?<![A-Za-z])([A-Za-z])\s*=\s*([^,，;；。\n]+)",
            text,
        )
        if len(assignments) >= 2 and len({name.lower() for name, _ in assignments}) == len(assignments):
            coordinate_entries = tuple(item.strip() for _, item in assignments)
    value_match = re.search(
        r"(?:最优值|optimal\s+value|\\max|max(?:imum)?)\s*"
        r"(?:为|是|is|equals?|=|[:：])?\s*"
        r"(?P<value>[^,，;；。\n]+)",
        text,
        re.IGNORECASE,
    )
    if (coordinate is None and coordinate_entries is None) or value_match is None:
        return None
    entries = (
        _parse_tuple_vector(coordinate.group("tuple"))
        if coordinate is not None else coordinate_entries
    )
    scalar = value_match.group("value").strip(" \t\r\n$。.!?")
    if entries is None or _parse_scalar_expression(scalar) is None:
        return None
    return entries, scalar


def _optimization_result_match(left: str, right: str) -> bool | None:
    marker = re.compile(
        r"最优(?:解|值)|optimal\s+(?:solution|value)|optimizer|\\max\s*=|\bmaximum\s*=",
        re.IGNORECASE,
    )
    if not marker.search(left) and not marker.search(right):
        return None
    left_signature = _optimization_signature(left)
    right_signature = _optimization_signature(right)
    if left_signature is None or right_signature is None:
        return False
    left_coordinates, left_value = left_signature
    right_coordinates, right_value = right_signature
    return bool(
        len(left_coordinates) == len(right_coordinates)
        and all(
            _scalar_entry_match(a, b)
            for a, b in zip(left_coordinates, right_coordinates)
        )
        and _scalar_entry_match(left_value, right_value)
    )


def _intervals(value: str) -> list[tuple[str, str, str, str]]:
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    pattern = re.compile(
        r"(?P<open>[\[(])\s*(?P<lower>[^,，\])\n]+)\s*[,，]\s*"
        r"(?P<upper>[^\])\n]+)\s*(?P<close>[\])])",
    )
    return [
        (
            match.group("lower").strip(),
            match.group("upper").strip(),
            match.group("open"),
            match.group("close"),
        )
        for match in pattern.finditer(text)
    ]


_INTERVAL_BOUND = (
    r"(?:[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
    r"[-+]?\\(?:d?frac)\{[^{}]+\}\{[^{}]+\}|"
    r"[-+]?\\sqrt\{[^{}]+\}|[-+]?(?:\\infty|∞))"
)


def _set_interval_signature(value: str) -> tuple[str, str, bool, bool] | None:
    """Parse one-dimensional interval notation or a chained inequality."""
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    text = text.replace(r"\leqslant", "<=").replace(r"\geqslant", ">=")
    text = text.replace(r"\leq", "<=").replace(r"\geq", ">=")
    text = text.replace("≤", "<=").replace("≥", ">=")

    chain = re.search(
        rf"(?P<lower>{_INTERVAL_BOUND})\s*(?P<lower_op><=|<)\s*"
        rf"[A-Za-z](?:_\{{?[^}}\s]+\}}?)?\s*(?P<upper_op><=|<)\s*"
        rf"(?P<upper>{_INTERVAL_BOUND})",
        text,
        re.IGNORECASE,
    )
    if chain:
        return (
            chain.group("lower"),
            chain.group("upper"),
            chain.group("lower_op") == "<",
            chain.group("upper_op") == "<",
        )

    reverse = re.search(
        rf"(?P<upper>{_INTERVAL_BOUND})\s*(?P<upper_op>>=|>)\s*"
        rf"[A-Za-z](?:_\{{?[^}}\s]+\}}?)?\s*(?P<lower_op>>=|>)\s*"
        rf"(?P<lower>{_INTERVAL_BOUND})",
        text,
        re.IGNORECASE,
    )
    if reverse:
        return (
            reverse.group("lower"),
            reverse.group("upper"),
            reverse.group("lower_op") == ">",
            reverse.group("upper_op") == ">",
        )

    intervals = _intervals(text)
    if not intervals:
        return None
    lower, upper, opening, closing = intervals[-1]
    interval_semantics = bool(
        re.search(
            r"\\in\b|∈|区间|范围|定义域|解集|"
            r"\b(?:interval|range|domain|solution\s+set)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(r"\\infty|∞", f"{lower} {upper}")
    )
    if not interval_semantics:
        return None
    return lower, upper, opening == "(", closing == ")"


def _infinite_bound(value: str) -> int | None:
    compact = re.sub(r"\s+", "", value).replace(r"\infty", "∞")
    if compact in {"∞", "+∞"}:
        return 1
    if compact == "-∞":
        return -1
    return None


def _same_interval_bound(left: str, right: str) -> bool:
    left_infinite = _infinite_bound(left)
    right_infinite = _infinite_bound(right)
    if left_infinite is not None or right_infinite is not None:
        return left_infinite is not None and left_infinite == right_infinite
    return _scalar_entry_match(left, right)


def _interval_inequality_match(left: str, right: str) -> bool | None:
    left_signature = _set_interval_signature(left)
    right_signature = _set_interval_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    left_lower, left_upper, left_lower_open, left_upper_open = left_signature
    right_lower, right_upper, right_lower_open, right_upper_open = right_signature
    return bool(
        left_lower_open == right_lower_open
        and left_upper_open == right_upper_open
        and _same_interval_bound(left_lower, right_lower)
        and _same_interval_bound(left_upper, right_upper)
    )


def _displayed_decimal_places(value: str) -> int | None:
    match = re.fullmatch(r"[+-]?(?:\d+)?\.(\d+)", value.strip())
    return len(match.group(1)) if match else None


def _rounded_scalar_match(left: str, right: str) -> bool:
    left_expression = _parse_scalar_expression(left)
    right_expression = _parse_scalar_expression(right)
    if left_expression is None or right_expression is None:
        return False
    try:
        import sympy as sp

        if sp.simplify(left_expression - right_expression) == 0:
            return True
        precisions = tuple(
            places for places in (
                _displayed_decimal_places(left),
                _displayed_decimal_places(right),
            ) if places is not None
        )
        if not precisions:
            return False
        tolerance = 0.5 * 10 ** (-min(precisions)) + 1e-12
        return abs(float(sp.N(left_expression - right_expression, 30))) <= tolerance
    except Exception:
        return False


def _approximate_interval_match(left: str, right: str) -> bool | None:
    approximate = re.compile(r"\\(?:approx|simeq)|≈|~=", re.IGNORECASE)
    if not approximate.search(left) and not approximate.search(right):
        return None
    left_intervals = _intervals(left)
    right_intervals = _intervals(right)
    if not left_intervals or not right_intervals:
        return False
    left_lower, left_upper, left_open, left_close = left_intervals[-1]
    right_lower, right_upper, right_open, right_close = right_intervals[-1]
    interval_semantics = bool(re.search(
        r"区间|置信区间|取值范围|解集|定义域|值域|"
        r"\b(?:interval|confidence\s+interval|range|solution\s+set|domain)\b",
        f"{left} {right}",
        re.IGNORECASE,
    ))
    if interval_semantics and (left_open, left_close) != (right_open, right_close):
        return False
    return bool(
        _rounded_scalar_match(left_lower, right_lower)
        and _rounded_scalar_match(left_upper, right_upper)
    )


def _negative_convergence_signature(value: str) -> tuple[str, str, str] | None:
    text = _unwrap_text_commands(str(value or "")).lower()
    negative = _judgement(text) == "negative" or bool(re.search(
        r"不\s*收敛|未\s*收敛|不能[^。；;]*收敛|"
        r"\b(?:does\s+not|do\s+not|not)\s+converge\b|\bdiverges?\b|\bno\b",
        text,
    ))
    pointwise = bool(re.search(r"逐点(?:极限|收敛)|pointwise(?:\s+limit|\s+converg)", text))
    l_one = bool(re.search(
        r"(?<![A-Za-z])l\s*\^?\s*\{?\s*1\s*\}?|"
        r"\\(?:lvert|rvert|lVert|rVert).*?_\s*\{?\s*1\s*\}?|l[-_ ]?1\s+norm",
        text,
    ))
    if negative and pointwise and l_one:
        return "negative", "pointwise-limit", "L1"
    return None


def _negative_convergence_match(left: str, right: str) -> bool | None:
    left_signature = _negative_convergence_signature(left)
    right_signature = _negative_convergence_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if not (left_signature and right_signature and left_signature == right_signature):
        return False
    left_values = _negative_convergence_values(left)
    right_values = _negative_convergence_values(right)
    for field in left_values.keys() & right_values.keys():
        if _math_object_match(left_values[field], right_values[field]) is False:
            return False
    return True


def _negative_convergence_values(value: str) -> dict[str, str]:
    """Extract only explicit scalar checks from an L1 non-convergence answer."""
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    scalar = (
        r"(?:\\(?:d?frac|tfrac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
        r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)"
    )
    fields: dict[str, str] = {}
    pointwise = re.search(
        r"(?:逐点(?:极限|收敛)?|pointwise(?:\s+(?:limit|convergence|converges?))?)"
        r"[^;；。.!?\n]{0,80}?(?:为|是|于|到|=|\bis\b|\bto\b|\\to|→)\s*"
        rf"(?P<value>{scalar})",
        text,
        re.IGNORECASE,
    )
    if pointwise:
        fields["pointwise"] = pointwise.group("value")
    norm = re.search(
        r"(?:\\lVert[^\n]{0,100}?\\rVert|\\Vert[^\n]{0,100}?\\Vert|"
        r"\|{1,2}[^\n]{1,100}?\|{1,2})"
        r"\s*_\s*\{?\s*1\s*\}?\s*=\s*"
        rf"(?P<value>{scalar})",
        text,
        re.IGNORECASE,
    )
    if norm:
        fields["l1_norm"] = norm.group("value")
    return fields


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
    text = _normalize_fraction_commands(str(value or "")).replace(r"\frac", "frac")
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


def _assignment_list_match(left: str, right: str) -> bool | None:
    """Compare complete lists such as "iteration formula, first iterate".

    None means neither answer has the deliberately narrow assignment-list
    shape, so the remaining equivalence rules may decide.  If exactly one side
    has that shape, the other side omitted at least one claimed component and
    is therefore not considered equivalent.
    """
    left_parts = _assignment_parts(_answer_value(left))
    right_parts = _assignment_parts(_answer_value(right))
    if left_parts is None and right_parts is None:
        return None
    if left_parts is None:
        left_parts = _embedded_assignment_parts(left, tuple(right_parts or ()))
    if right_parts is None:
        right_parts = _embedded_assignment_parts(right, tuple(left_parts or ()))
    if left_parts is None or right_parts is None:
        return False
    if left_parts.keys() != right_parts.keys():
        return False
    return all(
        _component_match(left_parts[key], right_parts[key])
        for key in left_parts
    )


def _assignment_parts(
    value: str,
    minimum_parts: int = 2,
) -> dict[str, str] | None:
    """Parse only answer-list-shaped, top-level variable assignments.

    Requiring the entire text to consist of at least two simple labelled
    assignments keeps this parser away from derivations and proof prose.  The
    top-level splitter preserves commas inside functions and grouped formulas.
    """
    text = normalize_latex(str(value or "")).strip()
    text = re.sub(
        r"(?i)^\s*(?:【\s*)?(?:最终答案|答案|final\s+answer|answer)(?:\s*】)?\s*[:：]?\s*",
        "",
        text,
    )
    pieces = _split_assignment_pieces(text)
    if len(pieces) < minimum_parts:
        return None

    parts: dict[str, str] = {}
    for piece in pieces:
        item = piece.strip(" \t\r\n$")
        item = re.sub(
            r"^(?:(?:\\[,;:!])|(?:\\(?:quad|qquad))|(?:\\\s+))+\s*",
            "",
            item,
        )
        item = re.sub(r"^\s*(?:\(\d+\)|（\d+）|\d+[.)、])\s*", "", item)
        if ":" in item or "：" in item:
            item = re.split(r"[:：]", item)[-1].strip()
        equation = _split_single_top_level_equals(item)
        if equation is None:
            return None
        lhs, rhs = equation
        key = _assignment_key(lhs)
        if not key or not rhs.strip() or key in parts:
            return None
        parts[key] = rhs.strip()
    return parts if len(parts) >= minimum_parts else None


def _embedded_assignment_parts(
    value: str,
    required_keys: tuple[str, ...],
) -> dict[str, str] | None:
    """Recover named final values from a bounded prose answer.

    This is used only opposite an already parsed multi-part assignment list.
    Assignments already present in the explicit conclusion take precedence;
    the body may only fill missing labels and cannot overwrite a conclusion.
    """
    if len(required_keys) < 2:
        return None
    explicit = _assignment_parts(_answer_value(value), minimum_parts=1) or {}
    recovered = {
        key: rhs for key, rhs in explicit.items() if key in required_keys
    }
    missing_keys = set(required_keys) - set(recovered)
    if not missing_keys:
        return recovered

    text = normalize_latex(str(value or ""))
    assignment = re.compile(
        r"(?P<lhs>(?:[A-Za-z]|\\[A-Za-z]+)"
        r"(?:_\s*(?:\{[^{}]+\}|[A-Za-z0-9]+))?)\s*=",
    )
    matches = list(assignment.finditer(text))
    for index, match in enumerate(matches):
        key = _assignment_key(match.group("lhs"))
        if key not in missing_keys:
            continue
        next_assignment = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tail = text[match.end():next_assignment]
        rhs = re.split(
            r"\$|\\\)|\\\]|[，,；;。！？!?\n]",
            tail,
            maxsplit=1,
        )[0].strip(" \t\r\n")
        if "=" in rhs:
            rhs = rhs.rsplit("=", 1)[-1].strip()
        if rhs:
            recovered[key] = rhs
    return recovered if set(recovered) == set(required_keys) else None


def _split_assignment_pieces(value: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(value):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            char in ",，;；\n"
            and round_depth == square_depth == brace_depth == 0
        ):
            piece = value[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + 1
    tail = value[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _split_single_top_level_equals(value: str) -> tuple[str, str] | None:
    positions: list[int] = []
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(value):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "=" and round_depth == square_depth == brace_depth == 0:
            positions.append(index)
    if len(positions) != 1:
        return None
    position = positions[0]
    return value[:position].strip(), value[position + 1:].strip()


def _assignment_key(value: str) -> str:
    text = normalize_latex(value).strip(" \t\r\n$")
    match = re.fullmatch(
        r"(?P<base>[A-Za-z]|\\[A-Za-z]+)"
        r"(?:_\s*(?:\{(?P<braced>[^{}]+)\}|(?P<bare>[A-Za-z0-9]+)))?",
        text,
    )
    if not match:
        return ""
    base = match.group("base")
    index = match.group("braced") or match.group("bare")
    if index is None:
        return base
    normalized_index = re.sub(r"\s+", "", normalize_latex(index))
    return f"{base}_[{normalized_index}]"


def _component_match(left: str, right: str) -> bool:
    typed_match = _math_object_match(left, right)
    if typed_match is not None:
        return typed_match
    if _is_numeric_answer(left) and _is_numeric_answer(right):
        left_numbers = _numbers(left)
        right_numbers = _numbers(right)
        if len(left_numbers) == len(right_numbers) == 1:
            return left_numbers == right_numbers

    try:
        import sympy as sp
        from tools.sympy_tool import SympyTool

        def parse(value: str):
            # Braced single-token subscripts are notation variants of the
            # identifier form accepted by the local restricted parser.
            normalized = re.sub(r"([A-Za-z])_\{([A-Za-z0-9]+)\}", r"\1_\2", value)
            expression = SympyTool._latex_to_sympy(normalized)
            return SympyTool()._parse(expression)

        parsed_left = parse(left)
        parsed_right = parse(right)
        if parsed_left.free_symbols != parsed_right.free_symbols:
            return False
        return bool(sp.simplify(parsed_left - parsed_right) == 0)
    except Exception:
        return False


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
