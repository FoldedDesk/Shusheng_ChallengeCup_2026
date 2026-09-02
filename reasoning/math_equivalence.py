"""Conservative equivalence checks for candidate comparison and offline evaluation."""

from __future__ import annotations

from fractions import Fraction
import re

from tools.latex_parser import find_matching_brace, normalize_latex
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
    if _has_ambiguous_unparenthesized_function(left_value) or _has_ambiguous_unparenthesized_function(right_value):
        return False

    # These answer families carry semantics that are lost by generic string
    # or expression comparison.  Each matcher has a deliberately narrow
    # trigger and becomes authoritative once that trigger is present.
    for matcher in (
        _finite_game_outcome_match,
        _complex_disk_match,
        _finite_integer_set_match,
        _finite_nested_integer_set_match,
        _finite_roots_of_unity_set_match,
        _jordan_block_multiset_match,
        _multistep_stability_summary_match,
        _integer_tuple_parameter_family_match,
        # Resolve explicit finite function lists before broad polynomial-family
        # signatures.  Otherwise one quadratic member can make the broad
        # matcher reject two equivalent, reordered solution sets.
        _repeated_function_family_match,
        _polynomial_family_match,
        _trigonometric_family_match,
        _entropy_identity_match,
        _runge_kutta_stability_match,
        _gaussian_curvature_function_match,
        _poisson_exponential_umvu_match,
        _operator_norm_spectrum_point_match,
        _wasserstein_transport_match,
        _rouche_zero_count_match,
        _uniform_series_summary_match,
        _optimization_result_match,
        _finite_exact_order_count_match,
        _vitali_conclusion_match,
        _uniform_integrability_match,
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
    # Tool-backed and concise model answers often introduce each requested
    # object with a short prose label, for example ``iteration matrix
    # $B_J=...$`` followed by ``$x^{(1)}=...$``.  The older assignment parser
    # deliberately accepts only an all-mathematics answer line, so it misses
    # this common, still unambiguous representation.  Compare the complete set
    # of named assignments found in explicit math spans before falling back to
    # any-fragment heuristics.  This remains fail closed: both sides need at
    # least two labels, their label sets must agree, and every value must match.
    span_assignment_match = _math_span_assignment_list_match(left_raw, right_raw)
    if span_assignment_match is not None:
        return span_assignment_match

    assignment_match = _assignment_list_match(left_raw, right_raw)
    if assignment_match is not None:
        return assignment_match

    # Topic signatures need the unprojected text: supporting prose can carry
    # a state index, maximal domain, or named result that a leading FINAL line
    # intentionally omits. They still run after multi-assignment comparison so
    # a single matching component cannot hide a disagreement in another one.
    for matcher in (
        _stationary_distribution_match,
        _sample_sum_estimator_match,
        _smith_cokernel_match,
        _homology_groups_match,
        _count_result_match,
        _expectation_result_match,
        _hitting_probability_result_match,
        _contour_integral_result_match,
        _ode_solution_domain_match,
        _nilpotent_summary_match,
        _tournament_hamilton_proof_match,
    ):
        semantic_match = matcher(left_raw, right_raw)
        if semantic_match is not None:
            return semantic_match

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

    labelled_equation_match = _labelled_equation_in_body_match(left_raw, right_raw)
    if labelled_equation_match is not None:
        return labelled_equation_match

    named_scalar_match = _named_scalar_in_body_match(left_raw, right_raw)
    if named_scalar_match is not None:
        return named_scalar_match

    explicit_scalar_match = _explicit_result_scalar_in_body_match(
        left_raw, right_raw
    )
    if explicit_scalar_match is not None:
        return explicit_scalar_match

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
    if a_judgement and a_judgement == b_judgement:
        if _is_bare_judgement(left_value):
            return _terminal_judgement(right_value) == a_judgement
        if _is_bare_judgement(right_value):
            return _terminal_judgement(left_value) == b_judgement

    if _same_indeterminate_variance_direction(left_value, right_value):
        return True

    if _adjoint_product_rule_match(left_value, right_value):
        return True

    textual_list_match = _short_textual_item_list_match(left_value, right_value)
    if textual_list_match is not None:
        return textual_list_match

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


def _short_textual_item_list_match(left: str, right: str) -> bool | None:
    """Compare short unordered lists across punctuation/conjunction spelling.

    The matcher is deliberately unavailable for mathematical expressions or
    prose sentences. At least one side must use an explicit list separator;
    this lets ``A和B`` match ``A、B`` without treating every Chinese ``和``
    inside ordinary prose as a list boundary.
    """

    def prepare(value: str) -> str:
        text = _unwrap_text_commands(normalize_latex(str(value or ""))).strip()
        text = re.sub(
            r"(?i)^\s*(?:最终答案|答案|answer|final\s+answer)\s*[:：]?\s*",
            "",
            text,
        )
        return text.strip(" \t\r\n。.!?")

    first, second = prepare(left), prepare(right)
    if (
        not first
        or not second
        or max(len(first), len(second)) > 180
        or re.search(r"[$=<>\\^{}\[\]()\n]", first + second)
        or re.search(r"[。.!?：:]", first + second)
        or not re.search(r"[、,，;；]", first + second)
    ):
        return None

    separator = r"\s*(?:、|,|，|;|；|以及|和|与|\band\b)\s*"

    def parts(value: str) -> tuple[str, ...]:
        items = tuple(
            _compact(item)
            for item in re.split(separator, value, flags=re.IGNORECASE)
            if _compact(item)
        )
        return items if 2 <= len(items) <= 8 else ()

    left_parts, right_parts = parts(first), parts(second)
    if not left_parts or not right_parts:
        return None
    return sorted(left_parts) == sorted(right_parts)


def _finite_game_outcome_match(left: str, right: str) -> bool | None:
    """Compare an explicit winner and, when stated, all winning first moves."""

    def parse(value: str) -> tuple[bool, frozenset[int] | None] | None:
        text = _unwrap_text_commands(normalize_latex(str(value or "")))
        losing = bool(re.search(
            r"先手(?:必败|会输|不能必胜)|后手必胜|"
            r"\b(?:first\s+player\s+(?:loses|is\s+losing)|"
            r"second\s+player\s+wins?)\b",
            text,
            re.IGNORECASE,
        ))
        winning = bool(re.search(
            r"先手必胜|先手有必胜策略|"
            r"\b(?:first\s+player\s+wins?|winning\s+for\s+the\s+first\s+player)\b",
            text,
            re.IGNORECASE,
        ))
        if losing == winning:
            return None
        if losing:
            return False, None
        move_clause = re.search(
            r"(?:必胜(?:的)?(?:第一步|首步|着法)|winning\s+(?:first\s+)?moves?)"
            r"[^。.;；\n]{0,120}",
            text,
            re.IGNORECASE,
        )
        if move_clause is None:
            return True, None
        moves = frozenset(
            int(item) for item in re.findall(r"(?<![A-Za-z0-9_.])-?\d+", move_clause.group(0))
        )
        return True, moves or None

    first = parse(left)
    second = parse(right)
    if first is None or second is None:
        return None
    if first[0] != second[0]:
        return False
    if not first[0]:
        return True
    if first[1] is None or second[1] is None:
        return None
    return first[1] == second[1]


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
    left_text = _strip_math_wrappers(left)
    right_text = _strip_math_wrappers(right)
    # ``解为 $u=...$`` and ``solution is $u=...$`` are prose-wrapped body
    # equations.  Treating the prose plus opening dollar as the mathematical
    # lhs creates a false mismatch; the conclusive-body matcher below parses
    # the displayed equation with its real label.
    prose_equation = re.compile(
        r"^\s*(?:解(?:为|是)|solution(?:\s+is)?)\s*[:：]?\s*\$",
        re.IGNORECASE,
    )
    if prose_equation.search(left_text) or prose_equation.search(right_text):
        return None
    left_equation = (
        None
        if re.search(r"[；;\n]", left_text)
        else _split_single_top_level_equals(left_text)
    )
    right_equation = (
        None
        if re.search(r"[；;\n]", right_text)
        else _split_single_top_level_equals(right_text)
    )
    if left_equation is None and right_equation is None:
        return None
    if left_equation is not None and right_equation is not None:
        left_lhs, left_rhs = left_equation
        right_lhs, right_rhs = right_equation
        if _compact(left_lhs) != _compact(right_lhs):
            return False
        return _math_object_match(left_rhs, right_rhs)
    if left_equation is not None:
        if "=" in _strip_math_wrappers(right):
            return None
        return _math_object_match(left_equation[1], right)
    if "=" in _strip_math_wrappers(left):
        return None
    return _math_object_match(left, right_equation[1])


def _conclusive_body_equations(value: str) -> tuple[tuple[str, str], ...]:
    """Extract explicitly displayed answer equations, not derivation chains."""
    text = normalize_latex(str(value or ""))
    spans = re.compile(r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)")
    equations: list[tuple[str, str]] = []
    for match in spans.finditer(text):
        prefix = text[:match.start()]
        conclusive = not prefix.strip() or bool(re.search(
            r"(?:最终答案|答案|结论|故|因此|所以|从而|于是|"
            r"解(?:为|是)|final(?:\s+answer)?|answer|solution(?:\s+is)?|"
            r"thus|hence|therefore)\s*[:：]?\s*$",
            prefix[-48:],
            re.IGNORECASE,
        ))
        if not conclusive:
            continue
        equation = _split_single_top_level_equals(
            (match.group("dollar") or match.group("paren") or "").strip()
        )
        if equation is not None and all(equation):
            equations.append(equation)
    return tuple(equations)


def _labelled_equation_in_body_match(left: str, right: str) -> bool | None:
    """Match a concise labelled result to the same conclusive body equation."""

    def one_way(direct: str, body: str) -> bool | None:
        direct_equation = _split_single_top_level_equals(
            _strip_math_wrappers(_answer_value(direct))
        )
        if direct_equation is None:
            return None
        lhs, rhs = direct_equation
        lhs_key = _compact(lhs)
        matches = [
            other_rhs
            for other_lhs, other_rhs in _conclusive_body_equations(body)
            if lhs_key and _compact(other_lhs) == lhs_key
        ]
        if not matches:
            return None
        # The last conclusive assertion of a repeated label is authoritative.
        return _math_object_match(rhs, matches[-1])

    forward = one_way(left, right)
    reverse = one_way(right, left)
    decisions = [item for item in (forward, reverse) if item is not None]
    if not decisions:
        return None
    return all(decisions)


def _named_scalar_values(value: str) -> tuple[str, ...]:
    """Extract scalar results after an explicit mathematical object label."""
    text = normalize_latex(str(value or ""))
    label = (
        r"(?:围道)?积分(?:值|结果)?|极限(?:值)?|概率|期望|方差|"
        r"行列式|迹|秩|曲率|最优值|结果|"
        r"integral|limit|probability|expectation|variance|determinant|"
        r"trace|rank|curvature|optimum|result"
    )
    pattern = re.compile(
        rf"(?:故|因此|所以|从而|于是|thus|hence|therefore)?\s*"
        rf"(?:{label})\s*(?:为|是|等于|is|equals?|=|[:：])\s*"
        rf"(?:\$(?P<math>[^$\n]+)\$|(?P<plain>[^，,。；;\n]+))",
        re.IGNORECASE,
    )
    values: list[str] = []
    for match in pattern.finditer(text):
        candidate = (match.group("math") or match.group("plain") or "").strip()
        if _parse_scalar_expression(candidate) is not None:
            values.append(candidate)
    return tuple(values)


def _named_scalar_in_body_match(left: str, right: str) -> bool | None:
    """Match a bare scalar answer to an explicitly named result in prose."""

    def one_way(direct: str, body: str) -> bool | None:
        value = _strip_math_wrappers(_answer_value(direct))
        parsed = _parse_scalar_expression(value)
        if parsed is None or parsed.free_symbols:
            return None
        named = _named_scalar_values(body)
        if not named:
            return None
        return _scalar_math_match(value, named[-1])

    decisions = [
        decision
        for decision in (one_way(left, right), one_way(right, left))
        if decision is not None
    ]
    return all(decisions) if decisions else None


def _explicit_result_math_values(value: str) -> tuple[str, ...]:
    """Extract math spans immediately asserted as the requested result."""
    text = normalize_latex(str(value or ""))
    cue = re.compile(
        r"(?:为|是|等于|结果(?:为|是)?|值(?:为|是)?|"
        r"is|equals?|is\s+given\s+by)\s*[:：=]?\s*$",
        re.IGNORECASE,
    )
    values: list[str] = []
    for match in re.finditer(
        r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)",
        text,
    ):
        if not cue.search(text[max(0, match.start() - 100):match.start()]):
            continue
        candidate = (match.group("dollar") or match.group("paren") or "").strip()
        if candidate:
            values.append(candidate)
    return tuple(values)


def _explicit_result_scalar_values(value: str) -> tuple[str, ...]:
    return tuple(
        item
        for item in _explicit_result_math_values(value)
        if _parse_scalar_expression(item) is not None
    )


def _direct_scalar_result(value: str) -> str:
    direct = _strip_math_wrappers(_answer_value(value))
    equation = _split_single_top_level_equals(direct)
    candidate = equation[1] if equation is not None else direct
    return candidate if _parse_scalar_expression(candidate) is not None else ""


def _explicit_result_scalar_in_body_match(left: str, right: str) -> bool | None:
    """Match a concise scalar against one explicitly asserted prose result."""

    def one_way(direct: str, body: str) -> bool | None:
        scalar = _direct_scalar_result(direct)
        asserted = _explicit_result_scalar_values(body)
        if not scalar or not asserted:
            return None
        decisions = tuple(_scalar_math_match(scalar, item) for item in asserted)
        return any(decision is True for decision in decisions)

    decisions = [
        decision
        for decision in (one_way(left, right), one_way(right, left))
        if decision is not None
    ]
    return all(decisions) if decisions else None


def _math_result_fragments(value: str) -> tuple[str, ...]:
    text = normalize_latex(str(value or ""))
    answer = _answer_value(text)
    fragments = [answer]
    fragments.extend(
        match.group("dollar") or match.group("paren") or ""
        for match in re.finditer(
            r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)",
            text,
        )
    )
    fragments.extend(re.split(r"[。；;\n]+", answer))
    return tuple(dict.fromkeys(item.strip() for item in fragments if item.strip()))


_SIMPLE_SAMPLE_SUM = re.compile(
    r"\\sum\s*"
    r"(?:_\s*(?:\{[^{}]*\}|[A-Za-z](?:\s*=\s*\d+)?))?\s*"
    r"(?:\^\s*(?:\{[^{}]*\}|[A-Za-z0-9]+))?\s*"
    r"(?P<variable>[A-Za-z])\s*_\s*\{?(?P<index>[A-Za-z])\}?",
    re.IGNORECASE,
)


def _sample_sum_estimator_signature(value: str) -> str:
    """Canonicalize a threshold estimator written with a sample-sum alias."""
    text = normalize_latex(str(value or ""))
    aliases: set[str] = set()
    fragments = _math_result_fragments(text)
    for fragment in fragments:
        equation = _split_single_top_level_equals(fragment.strip(" $，,；;"))
        if equation is None:
            continue
        lhs, rhs = equation
        if re.fullmatch(r"[A-Za-z]", lhs.strip()) and _SIMPLE_SAMPLE_SUM.fullmatch(
            rhs.strip()
        ):
            aliases.add(lhs.strip())

    piecewise_condition = ""
    piecewise = re.search(
        r"(?:当|when)\s*(?:\$(?P<math>[^$\n]+)\$|"
        r"(?P<plain>[^，,。；;\n]+?))\s*(?:时|,|，)\s*[,，]?\s*"
        r"(?:否则(?:为|是)?|otherwise(?:\s+(?:is|equals?))?)\s*"
        r"\$?\s*0\s*\$?",
        text,
        re.IGNORECASE,
    )
    if piecewise is not None:
        piecewise_condition = (
            piecewise.group("math") or piecewise.group("plain") or ""
        ).strip()

    estimator_candidates: list[str] = []
    for fragment in fragments:
        equation = _split_single_top_level_equals(fragment.strip(" $，,；;"))
        if equation is None:
            continue
        lhs, rhs = equation
        if not re.search(r"\\(?:widehat|hat)\b", lhs):
            continue
        has_indicator = bool(re.search(
            r"\\(?:mathbf|mathbb)\s*\{?1\}?|"
            r"(?<![A-Za-z])I\s*\(",
            rhs,
        ))
        if not has_indicator and not piecewise_condition:
            continue
        estimator_candidates.append(
            rhs.strip()
            if has_indicator
            else f"{rhs.strip()}I({piecewise_condition})"
        )
    if not estimator_candidates:
        return ""
    # ``_math_result_fragments`` intentionally includes both the projected
    # answer and its displayed equations.  The shortest complete estimator is
    # the equation itself; a longer candidate can append proof prose after it.
    estimator_rhs = min(estimator_candidates, key=len)

    canonical = re.sub(
        r"\\(?:mathbf|mathbb)\s*\{?1\}?\s*_\s*"
        r"\{\s*\\?\{\s*([^{}]+)\s*\\?\}\s*\}",
        r"I(\1)",
        estimator_rhs,
    )
    canonical = re.sub(
        r"\\(?:mathbf|mathbb)\s*\{?1\}?\s*_\s*\{([^{}]+)\}",
        r"I(\1)",
        canonical,
    )
    canonical = _SIMPLE_SAMPLE_SUM.sub("TOTAL", canonical)
    for alias in aliases:
        canonical = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
            "TOTAL",
            canonical,
        )
    return _compact(canonical)


def _sample_sum_estimator_match(left: str, right: str) -> bool | None:
    left_signature = _sample_sum_estimator_signature(left)
    right_signature = _sample_sum_estimator_signature(right)
    if not left_signature and not right_signature:
        return None
    if not left_signature or not right_signature:
        return False
    return left_signature == right_signature


def _poisson_exponential_umvu_signature(value: str) -> str:
    """Extract an explicit exp(-lambda) UMVU estimator without comparing prose."""
    text = normalize_latex(str(value or ""))
    if not re.search(r"Poisson|泊松", text, re.IGNORECASE) or not re.search(
        r"\bUMVU\b|一致最小方差无偏|uniformly\s+minimum\s+variance",
        text,
        re.IGNORECASE,
    ):
        return ""

    aliases: set[str] = set()
    fragments = _math_result_fragments(text)
    for fragment in fragments:
        equation = _split_single_top_level_equals(fragment.strip(" $，,；;"))
        if equation is None:
            continue
        lhs, rhs = equation
        if re.fullmatch(r"[A-Za-z]", lhs.strip()) and _SIMPLE_SAMPLE_SUM.fullmatch(
            rhs.strip()
        ):
            aliases.add(lhs.strip())

    estimators: list[str] = []
    for fragment in fragments:
        equation = _split_single_top_level_equals(fragment.strip(" $，,；;"))
        if equation is None:
            continue
        lhs, rhs = equation
        if not re.search(r"\\(?:widehat|hat)", lhs) or not re.search(
            r"(?:\\lambda|lambda|λ)", lhs,
            re.IGNORECASE,
        ):
            continue
        canonical = _SIMPLE_SAMPLE_SUM.sub("TOTAL", rhs.strip())
        for alias in aliases:
            canonical = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                "TOTAL",
                canonical,
            )
        if _parse_scalar_expression(canonical) is not None:
            estimators.append(canonical)
    return min(estimators, key=len) if estimators else ""


def _poisson_exponential_umvu_match(left: str, right: str) -> bool | None:
    left_signature = _poisson_exponential_umvu_signature(left)
    right_signature = _poisson_exponential_umvu_signature(right)
    if not left_signature and not right_signature:
        return None
    if not left_signature or not right_signature:
        return False
    return _scalar_math_match(left_signature, right_signature)


def _operator_norm_spectrum_point_signature(
    value: str,
) -> tuple[str, str, str, bool] | None:
    """Parse the norm, interval spectrum, and explicitly empty point spectrum."""
    text = normalize_latex(str(value or ""))
    norm = re.search(
        r"(?:\\?\|\s*T\s*\\?\||\\lVert\s*T\s*\\rVert)\s*=\s*"
        r"(?P<value>\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
        r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?)",
        text,
    )
    spectrum = re.search(
        r"\\sigma\s*\(\s*T\s*\)\s*=\s*\[\s*"
        r"(?P<lower>[^,，\]\n]+)\s*[,，]\s*(?P<upper>[^\]\n]+)\s*\]",
        text,
        re.IGNORECASE,
    )
    point_empty = bool(re.search(
        r"(?:\\sigma\s*_?\s*\{?p\}?\s*\(\s*T\s*\)|点谱|point\s+spectrum)"
        r"[^。；;\n]{0,40}(?:=|为|是|is)\s*\$?\s*"
        r"(?:\\varnothing|\\emptyset|∅|空集|空)",
        text,
        re.IGNORECASE,
    ))
    if norm is None or spectrum is None or not point_empty:
        return None
    values = (
        norm.group("value").strip(),
        spectrum.group("lower").strip(),
        spectrum.group("upper").strip(),
    )
    if any(_parse_scalar_expression(item) is None for item in values):
        return None
    return values[0], values[1], values[2], point_empty


def _operator_norm_spectrum_point_match(left: str, right: str) -> bool | None:
    left_signature = _operator_norm_spectrum_point_signature(left)
    right_signature = _operator_norm_spectrum_point_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    return bool(
        left_signature[3] and right_signature[3]
        and all(
            _scalar_math_match(left_item, right_item)
            for left_item, right_item in zip(left_signature[:3], right_signature[:3])
        )
    )


def _abelian_group_signature(value: str) -> tuple[int, tuple[int, ...]] | None:
    """Parse a finite-rank Z-module written as free and cyclic summands."""
    text = _strip_math_wrappers(value).replace(r"\left", "").replace(r"\right", "")
    if re.fullmatch(r"(?:0|\{0\}|\\?{0\\?})", text.strip()):
        return 0, ()

    cyclic = re.compile(
        r"(?:\\mathbb\s*\{?\s*Z\s*\}?|\\mathbf\s*\{?\s*Z\s*\}?|Z)"
        r"\s*/\s*(?P<order>\d+)\s*"
        r"(?:\\mathbb\s*\{?\s*Z\s*\}?|\\mathbf\s*\{?\s*Z\s*\}?|Z)",
        re.IGNORECASE,
    )
    torsion = tuple(sorted(int(match.group("order")) for match in cyclic.finditer(text)))
    remainder = cyclic.sub(" ", text)
    free = re.compile(
        r"(?:\\mathbb\s*\{?\s*Z\s*\}?|\\mathbf\s*\{?\s*Z\s*\}?|"
        r"(?<![A-Za-z])Z(?![A-Za-z]))"
        r"(?:\s*\^\s*\{?(?P<rank>\d+)\}?)?",
        re.IGNORECASE,
    )
    free_rank = sum(int(match.group("rank") or 1) for match in free.finditer(remainder))
    if free_rank == 0 and not torsion:
        return None
    return free_rank, torsion


def _cokernel_group(value: str) -> tuple[int, tuple[int, ...]] | None:
    text = normalize_latex(str(value or ""))
    for fragment in _math_result_fragments(text):
        if not re.search(r"coker|余核", fragment, re.IGNORECASE):
            continue
        relation = re.search(r"(?:\\cong|≅|=)", fragment)
        if relation is not None:
            signature = _abelian_group_signature(fragment[relation.end():])
            if signature is not None:
                return signature
    prose = re.search(
        r"(?:余核|coker(?:nel)?)[^$。；;\n]{0,40}?"
        r"(?:为|是|is|=|\\cong|≅|[:：])\s*\$([^$]+)\$",
        text,
        re.IGNORECASE,
    )
    return _abelian_group_signature(prose.group(1)) if prose else None


def _smith_invariants(value: str) -> tuple[int, ...] | None:
    text = normalize_latex(str(value or ""))
    if not re.search(r"Smith|史密斯", text, re.IGNORECASE):
        return None
    matrix = re.search(
        r"\\begin\{(?P<env>matrix|pmatrix|bmatrix|Bmatrix|smallmatrix)\}"
        r".*?\\end\{(?P=env)\}",
        text,
        re.DOTALL,
    )
    if matrix is not None:
        parsed = _parse_matrix(matrix.group(0))
        if parsed is not None:
            diagonal: list[int] = []
            for index in range(min(len(parsed), len(parsed[0]))):
                expression = _parse_scalar_expression(parsed[index][index])
                if expression is None or expression.is_integer is not True:
                    return None
                integer = abs(int(expression))
                if integer:
                    diagonal.append(integer)
            if diagonal:
                return tuple(diagonal)

    labelled = re.search(
        r"(?:Smith|史密斯)[^。；;\n]{0,48}?(?:不变量|invariant\s+factors?)"
        r"[^$。；;\n]{0,24}?\$([^$]+)\$",
        text,
        re.IGNORECASE,
    )
    if labelled is None:
        return None
    values: list[int] = []
    for item in _split_top_level_items(labelled.group(1)):
        expression = _parse_scalar_expression(item)
        if expression is None or expression.is_integer is not True:
            return None
        integer = abs(int(expression))
        if integer:
            values.append(integer)
    return tuple(values) if values else None


def _smith_cokernel_match(left: str, right: str) -> bool | None:
    left_invariants = _smith_invariants(left)
    right_invariants = _smith_invariants(right)
    left_cokernel = _cokernel_group(left)
    right_cokernel = _cokernel_group(right)
    present = (left_invariants, right_invariants, left_cokernel, right_cokernel)
    if all(item is None for item in present):
        return None
    if any(item is None for item in present):
        return False
    return bool(
        left_invariants == right_invariants
        and left_cokernel == right_cokernel
    )


def _homology_group_signature(
    value: str,
) -> dict[int, tuple[int, tuple[int, ...]]] | None:
    # Prefer an explicit conclusion. A complete proof commonly repeats the
    # same H_k labels while deriving kernels and images; those intermediate
    # equations are not additional answer components.
    text = normalize_latex(_answer_value(value))
    # A proof often repeats H_k while computing kernels and quotients.  When
    # an explicit support marker is present, the headline before it is the
    # answer object; later assignments are derivation facts, not extra groups.
    text = re.split(
        r"(?:[；;。]\s*)?(?:依据|证明|论证|because|proof)\s*[:：]",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    assignment = re.compile(
        r"H\s*_\s*\{?(?P<degree>\d+)\}?"
        r"(?:\s*\([^()]{0,40}\))?\s*(?:\\cong|≅|=)\s*",
        re.IGNORECASE,
    )
    matches = list(assignment.finditer(text))
    if len(matches) < 2:
        return None
    groups: dict[int, tuple[int, tuple[int, ...]]] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rhs = text[match.end():stop]
        rhs = re.split(r"[$，,；;。\n]", rhs, maxsplit=1)[0].strip()
        signature = _abelian_group_signature(rhs)
        if signature is None:
            return None
        degree = int(match.group("degree"))
        if degree in groups:
            return None
        groups[degree] = signature
    return groups


def _homology_groups_match(left: str, right: str) -> bool | None:
    left_signature = _homology_group_signature(left)
    right_signature = _homology_group_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    return left_signature == right_signature


def _top_level_equal_positions(value: str) -> tuple[int, ...]:
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
    return tuple(positions)


def _terminal_chain_expression(value: str) -> str:
    text = _clean_assignment_piece(value)
    positions = _top_level_equal_positions(text)
    candidate = text[positions[-1] + 1:] if positions else text
    candidate = candidate.strip(" \t\r\n$。；;")
    return candidate if _parse_scalar_expression(candidate) is not None else ""


def _count_result_signature(value: str) -> str | None:
    text = normalize_latex(str(value or ""))
    if not re.search(
        r"总数|总计|计数(?:结果)?|个数|数量|"
        r"(?:生成树|同态|自同构|排列|组合|方案|方法|元素|解)数|"
        r"\\tau\s*\(|"
        r"(?:共有|共计)[^。.;\n]{0,20}\d|"
        r"Pr(?:ü|u)fer|普吕弗|\\binom|"
        r"纤维[^。.;\n]{0,40}(?:排列|分配)|"
        r"\b(?:total|count|number\s+of|there\s+are)\b",
        text,
        re.IGNORECASE,
    ):
        return None
    def candidates(payload: str) -> list[str]:
        terminal_candidates: list[str] = []
        for fragment in _math_result_fragments(payload):
            terminal = _terminal_chain_expression(fragment)
            expression = _parse_scalar_expression(terminal) if terminal else None
            if expression is not None and not expression.free_symbols:
                terminal_candidates.append(terminal)
        return terminal_candidates

    # Prefer the stated answer clause. Later proof text may contain small-case
    # checks or intermediate cardinalities that are not the requested count.
    headline = re.split(
        r"(?:[；;]\s*(?:依据|证明|推导|核验|because|proof|derivation)\b)|\n",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    headline_candidates = candidates(headline)
    if headline_candidates:
        return headline_candidates[-1]

    # An explicit answer marker is authoritative. Supporting proofs often
    # contain later intermediate counts, which must not replace the headline.
    if re.search(r"(?:FINAL(?:\s+ANSWER)?|最终答案|最后答案)\s*[:：]", text, re.IGNORECASE):
        marked_candidates = candidates(_answer_value(text))
        if marked_candidates:
            return marked_candidates[-1]

    # In ``总数为 4\cdot...=30240`` the first integer is a factor, not the
    # answer.  A fully parsed terminal equality is therefore stronger than a
    # number immediately following the prose label.
    terminal_candidates = candidates(text)
    return terminal_candidates[-1] if terminal_candidates else None


def _count_result_match(left: str, right: str) -> bool | None:
    left_signature = _count_result_signature(left)
    right_signature = _count_result_signature(right)
    if left_signature is None or right_signature is None:
        return None
    return _scalar_math_match(left_signature, right_signature)


def _expectation_signatures(value: str) -> tuple[tuple[str, str], ...]:
    signatures: list[tuple[str, str]] = []
    for fragment in _math_result_fragments(value):
        positions = _top_level_equal_positions(fragment)
        if not positions:
            continue
        lhs = fragment[:positions[0]].strip(" $，,；;")
        lhs = re.sub(
            r"\\mathbb\s*\{?\s*E\s*\}?",
            "E",
            lhs,
            flags=re.IGNORECASE,
        )
        if not re.match(r"^E\s*(?:\[|[A-Za-z])", lhs, re.IGNORECASE):
            continue
        rhs = _terminal_chain_expression(fragment)
        if not rhs:
            continue
        key = re.sub(r"[\[\]()\s{}\\]", "", lhs).casefold()
        signatures.append((key, rhs))

    text = normalize_latex(str(value or ""))
    for match in re.finditer(
        r"X\s*_\s*\{?\(\s*(?P<order>\d+)\s*\)\}?"
        r"[^。；;\n]{0,120}(?:条件期望|conditional\s+expectation)"
        r"\s*(?:为|是|is|=|[:：])\s*\$?(?P<value>[^$。；;\n]+)\$?",
        text,
        re.IGNORECASE,
    ):
        rhs = match.group("value").strip()
        if _parse_scalar_expression(rhs) is not None:
            signatures.append((f"conditionalx{match.group('order')}", rhs))
    return tuple(signatures)


def _headline_expectation_scalar(value: str) -> str:
    text = normalize_latex(str(value or ""))
    headline = re.split(r"[；;。\n]", text, maxsplit=1)[0]
    match = re.search(
        r"(?:期望(?:值)?|均值|expected\s+value|expectation|mean)"
        r"[^$。；;\n]{0,24}?(?:为|是|is|equals?|=|[:：])\s*"
        r"(?:\$\s*(?P<math>[^$]+)\s*\$|(?P<plain>[^,，。；;\n]+))",
        headline,
        re.IGNORECASE,
    )
    if match is None:
        return ""
    candidate = (match.group("math") or match.group("plain") or "").strip()
    terminal = _terminal_chain_expression(candidate)
    if terminal:
        candidate = terminal
    expression = _parse_scalar_expression(candidate)
    return candidate if expression is not None and not expression.free_symbols else ""


def _expectation_result_match(left: str, right: str) -> bool | None:
    left_signatures = _expectation_signatures(left)
    right_signatures = _expectation_signatures(right)
    left_headline = _headline_expectation_scalar(left)
    right_headline = _headline_expectation_scalar(right)
    if left_headline and right_headline:
        return _scalar_math_match(left_headline, right_headline)
    if left_headline and right_signatures:
        decisions = [
            _scalar_math_match(left_headline, signature[1])
            for signature in right_signatures
        ]
        if any(decision is True for decision in decisions):
            return True
    if right_headline and left_signatures:
        decisions = [
            _scalar_math_match(signature[1], right_headline)
            for signature in left_signatures
        ]
        if any(decision is True for decision in decisions):
            return True
    if not left_signatures or not right_signatures:
        return None

    def compatible_key(left_key: str, right_key: str) -> bool:
        if left_key == right_key:
            return True
        left_order = re.search(r"x_?(\d+)", left_key)
        right_order = re.search(r"x_?(\d+)", right_key)
        return bool(
            left_order
            and right_order
            and left_order.group(1) == right_order.group(1)
            and "conditional" in f"{left_key}{right_key}"
        )

    decisions = [
        _scalar_math_match(left_rhs, right_rhs)
        for left_key, left_rhs in left_signatures
        for right_key, right_rhs in right_signatures
        if compatible_key(left_key, right_key)
    ]
    if not decisions:
        # A single expectation may use a descriptive random-variable label on
        # one side (for example ``max``) and a named symbol on the other. With
        # exactly one asserted expectation on each side there is no component
        # ambiguity, so compare their terminal values directly.
        if len(left_signatures) == len(right_signatures) == 1:
            return _scalar_math_match(
                left_signatures[0][1], right_signatures[0][1]
            )
        return None
    return any(decision is True for decision in decisions)


def _hitting_probability_signature(value: str) -> tuple[int, str] | None:
    candidates: list[tuple[int, str]] = []
    for fragment in _math_result_fragments(value):
        match = re.search(r"h\s*_\s*\{?(\d+)\}?\s*=", fragment, re.IGNORECASE)
        if match is None:
            continue
        rhs = _terminal_chain_expression(fragment)
        if rhs:
            candidates.append((int(match.group(1)), rhs))
    return candidates[-1] if candidates else None


def _hitting_probability_result_match(left: str, right: str) -> bool | None:
    left_signature = _hitting_probability_signature(left)
    right_signature = _hitting_probability_signature(right)
    if left_signature is not None and right_signature is not None:
        return bool(
            left_signature[0] == right_signature[0]
            and _scalar_math_match(left_signature[1], right_signature[1])
        )

    def bare_scalar(value: str) -> str:
        answer = _strip_math_wrappers(_answer_value(value))
        parsed = _parse_scalar_expression(answer)
        return answer if parsed is not None and not parsed.free_symbols else ""

    if left_signature is not None:
        direct = bare_scalar(right)
        return _scalar_math_match(left_signature[1], direct) if direct else None
    if right_signature is not None:
        direct = bare_scalar(left)
        return _scalar_math_match(direct, right_signature[1]) if direct else None
    return None


def _contour_integral_signature(value: str) -> str:
    candidates: list[str] = []
    for fragment in _math_result_fragments(value):
        if not re.search(r"\\oint|围道积分|contour\s+integral", fragment, re.IGNORECASE):
            continue
        terminal = _terminal_chain_expression(fragment)
        if terminal:
            candidates.append(terminal)
    # Concise residue solutions often omit the word "contour" in the final
    # sentence after the contour was explicit in the problem.  Accept only an
    # explicit integral-result assignment inside a math span; residue values
    # elsewhere in the proof are not eligible as the requested integral.
    if not candidates and re.search(r"\\operatorname\s*\{?Res\}?|\bresidue\b|留数", value, re.IGNORECASE):
        for match in re.finditer(
            r"(?:围道)?积分\s*(?:为|等于|=)\s*\$([^$]+)\$|"
            r"(?:contour\s+)?integral\s*(?:is|equals?|=)\s*\$([^$]+)\$",
            value,
            re.IGNORECASE,
        ):
            body = next(group for group in match.groups() if group is not None)
            terminal = _terminal_chain_expression(body) or body.strip()
            if terminal:
                candidates.append(terminal)
    return candidates[-1] if candidates else ""


def _contour_integral_result_match(left: str, right: str) -> bool | None:
    left_signature = _contour_integral_signature(left)
    right_signature = _contour_integral_signature(right)
    if left_signature and right_signature:
        return _scalar_math_match(left_signature, right_signature)

    def bare_scalar(value: str) -> str:
        answer = _strip_math_wrappers(_answer_value(value))
        parsed = _parse_scalar_expression(answer)
        return answer if parsed is not None and not parsed.free_symbols else ""

    if left_signature:
        direct = bare_scalar(right)
        return _scalar_math_match(left_signature, direct) if direct else None
    if right_signature:
        direct = bare_scalar(left)
        return _scalar_math_match(direct, right_signature) if direct else None
    return None


def _ode_solution_signature(value: str) -> str:
    candidates: list[str] = []
    for fragment in _math_result_fragments(value):
        positions = _top_level_equal_positions(fragment)
        if not positions:
            continue
        raw_lhs = fragment[:positions[0]]
        # Apostrophes are presentation punctuation to ``_compact``. Reject a
        # differential equation before compacting so it cannot overwrite the
        # explicit solution collected earlier in the response.
        if re.search(r"['′]|\\prime|\\dot|d\s*y\s*/\s*d\s*x", raw_lhs):
            continue
        lhs = _compact(raw_lhs)
        if lhs not in {"y", "y(x)"}:
            continue
        rhs = fragment[positions[0] + 1:].strip(" $，,；;")
        if _parse_scalar_expression(rhs) is not None:
            candidates.append(rhs)
    return candidates[-1] if candidates else ""


def _whole_real_line_domain(value: str) -> bool:
    return bool(re.search(
        r"(?:最大(?:存在|解)?区间|maximal\s+(?:existence\s+)?interval)"
        r"[^。；;\n]{0,60}(?:\\mathbb\s*\{?R\}?|实轴|real\s+line|all\s+real)",
        value,
        re.IGNORECASE,
    ))


def _ode_solution_domain_match(left: str, right: str) -> bool | None:
    left_solution = _ode_solution_signature(left)
    right_solution = _ode_solution_signature(right)
    if not left_solution or not right_solution:
        return None
    left_real = _whole_real_line_domain(left)
    right_real = _whole_real_line_domain(right)
    if left_real != right_real:
        return False
    solution_match = _scalar_math_match(left_solution, right_solution)
    if solution_match is not True:
        return solution_match

    maximal_marker = re.compile(
        r"最大(?:存在|解)?区间|maximal\s+(?:existence\s+)?interval",
        re.IGNORECASE,
    )
    left_has_domain = bool(maximal_marker.search(left))
    right_has_domain = bool(maximal_marker.search(right))
    if left_has_domain != right_has_domain:
        return False
    if not left_has_domain or (left_real and right_real):
        return True
    left_intervals = _intervals(left)
    right_intervals = _intervals(right)
    if not left_intervals or not right_intervals:
        return False
    left_lower, left_upper, left_open, left_close = left_intervals[-1]
    right_lower, right_upper, right_open, right_close = right_intervals[-1]
    return bool(
        (left_open, left_close) == (right_open, right_close)
        and _same_interval_bound(left_lower, right_lower)
        and _same_interval_bound(left_upper, right_upper)
    )


def _nilpotent_summary_signature(
    value: str,
) -> tuple[tuple[int, ...], str, int, int] | None:
    text = normalize_latex(str(value or ""))
    blocks = re.search(
        r"(?:Jordan\s*)?(?:全部)?块(?:大小)?\s*(?:为|是|=|[:：])\s*\$?\s*"
        r"[\[(]?\s*((?:\d+\s*[,，]\s*)+\d+)\s*[\])]??\s*\$?",
        text,
        re.IGNORECASE,
    )
    rank = re.search(
        r"(?:\\operatorname\s*\{\s*rank\s*\}|\\?rank|秩)\s*([A-Za-z])"
        r"\s*=\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    minimal = re.search(
        r"(?:最小多项式|minimal\s+polynomial)[^。；;\n]{0,40}"
        r"(?:为|是|=|[:：])\s*\$?\s*[A-Za-z]\s*\^\s*\{?(\d+)\}?",
        text,
        re.IGNORECASE,
    )
    if blocks is None or rank is None or minimal is None:
        return None
    sizes = tuple(int(item) for item in re.split(r"[,，]", blocks.group(1)))
    return sizes, rank.group(1).casefold(), int(rank.group(2)), int(minimal.group(1))


def _nilpotent_summary_match(left: str, right: str) -> bool | None:
    left_signature = _nilpotent_summary_signature(left)
    right_signature = _nilpotent_summary_signature(right)
    if left_signature is None or right_signature is None:
        return None
    return left_signature == right_signature


def _tournament_hamilton_proof_match(left: str, right: str) -> bool | None:
    def signature(value: str) -> tuple[bool, bool, bool] | None:
        text = str(value or "")
        hamilton = bool(re.search(r"Hamilton(?:ian)?\s*(?:path|路)|哈密顿(?:路径|路)", text, re.IGNORECASE))
        if re.search(
            r"(?:不存在|不能(?:得到|构造)?|无法(?:得到|构造)?)[^。；;\n]{0,24}"
            r"(?:Hamilton|哈密顿)|(?:no|not)\s+Hamilton",
            text,
            re.IGNORECASE,
        ):
            return None
        induction = bool(re.search(r"归纳|induction", text, re.IGNORECASE))
        insertion = bool(re.search(r"插入|插在|insert", text, re.IGNORECASE))
        return (hamilton, induction, insertion) if hamilton else None

    left_signature = signature(left)
    right_signature = signature(right)
    if left_signature is None or right_signature is None:
        return None
    return all(left_signature) and all(right_signature)


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

    # SymPy commonly prints a matrix environment inside presentation brackets
    # (``[\begin{matrix}...\end{matrix}]``), whereas hand-written references
    # use ``pmatrix`` or ``bmatrix``.  Strip only brackets whose complete
    # interior is already a recognized matrix environment; ordinary intervals
    # and lists retain their type.
    bracketed = text.replace(r"\left", "").replace(r"\right", "").strip()
    if (
        bracketed.startswith("[")
        and bracketed.endswith("]")
        and _MATRIX_ENVIRONMENT.fullmatch(bracketed[1:-1].strip())
    ):
        text = bracketed[1:-1].strip()

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


def _complex_disk_signature(value: str) -> tuple[str, str, bool] | None:
    """Parse narrow complex disk notations as (center, radius, closed)."""
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    equation = _split_single_top_level_equals(text)
    if equation is not None:
        text = equation[1]
    text = (
        text.replace(r"\left", "")
        .replace(r"\right", "")
        .replace(r"\,", "")
        .replace(r"\!", "")
        .strip()
    )

    disk_body = (
        r"(?:D|\\mathbb\s*\{?D\}?|\\mathcal\s*\{?D\}?)"
        r"\s*(?:\(\s*(?P<center>[^,()]+)\s*,\s*"
        r"(?P<radius>[^,()]+)\s*\))?"
    )
    named = re.fullmatch(
        rf"(?P<bar>\\overline\s*\{{\s*)?{disk_body}"
        rf"(?(bar)\s*\}})",
        text,
        re.IGNORECASE,
    )
    if named:
        center = (named.group("center") or "0").strip()
        radius = (named.group("radius") or "1").strip()
        return center, radius, bool(named.group("bar"))

    # Require an explicitly complex set, or a conventional complex variable,
    # so that an ordinary real interval/norm inequality is not retyped as a disk.
    complex_context = bool(
        re.search(r"\\mathbb\s*\{?C\}?|\bcomplex\b|复数", text, re.IGNORECASE)
        or re.search(r"\\(?:lambda|zeta)\b|(?<![A-Za-z])z(?![A-Za-z])", text)
    )
    if not complex_context or not re.search(r"\\?\{|\{", text):
        return None
    inequality = re.search(
        r"(?:\\lvert|\|)\s*(?P<variable>\\?[A-Za-z]+)\s*"
        r"(?:(?P<sign>[+-])\s*(?P<center>[^|]+?))?\s*"
        r"(?:\\rvert|\|)\s*(?P<operator>\\leq?|<=|≤|<)\s*"
        r"(?P<radius>[^,;；}\\\n]+)",
        text,
    )
    if inequality is None:
        return None
    center = (inequality.group("center") or "0").strip()
    if inequality.group("sign") == "+":
        center = f"-({center})"
    radius = inequality.group("radius").strip()
    closed = inequality.group("operator") in {r"\le", r"\leq", "<=", "≤"}
    return center, radius, closed


def _complex_disk_match(left: str, right: str) -> bool | None:
    left_signature = _complex_disk_signature(left)
    right_signature = _complex_disk_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    left_center, left_radius, left_closed = left_signature
    right_center, right_radius, right_closed = right_signature
    return bool(
        left_closed == right_closed
        and _scalar_entry_match(left_center, right_center)
        and _scalar_entry_match(left_radius, right_radius)
    )


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


def _replace_balanced_fractions(value: str) -> str:
    """Convert nested TeX fractions without relying on a full LaTeX parser."""
    text = str(value or "")

    def group_at(source: str, start: int) -> tuple[str, int] | None:
        while start < len(source) and source[start].isspace():
            start += 1
        if start >= len(source) or source[start] != "{":
            return None
        depth = 0
        for index in range(start, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[start + 1:index], index + 1
        return None

    while True:
        matches = list(re.finditer(r"\\(?:d?frac|tfrac)", text))
        replaced = False
        for match in reversed(matches):
            numerator = group_at(text, match.end())
            if numerator is None:
                continue
            denominator = group_at(text, numerator[1])
            if denominator is None:
                atom = _balanced_tex_atom(text, numerator[1])
                if atom is None:
                    continue
                denominator = (atom[0].strip("{}"), atom[1])
            replacement = (
                f"(({_replace_balanced_fractions(numerator[0])})/"
                f"({_replace_balanced_fractions(denominator[0])}))"
            )
            text = text[:match.start()] + replacement + text[denominator[1]:]
            replaced = True
            break
        if not replaced:
            return text


def _replace_balanced_e_powers(value: str) -> str:
    """Rewrite balanced ``e^{...}`` forms as exact ``exp(...)`` calls."""
    text = str(value or "")
    search_from = 0
    while True:
        match = re.search(r"e\s*\^\s*\{", text[search_from:])
        if match is None:
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
        payload = _replace_balanced_e_powers(text[brace_start + 1:end])
        product = "*" if start > 0 and (text[start - 1].isalnum() or text[start - 1] == ")") else ""
        replacement = f"{product}exp({payload})"
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


def _replace_parenthesized_log_bases(value: str) -> str:
    r"""Rewrite exact ``\log_b(argument)`` forms using natural logarithms."""
    text = str(value or "")
    search_from = 0
    pattern = re.compile(
        r"\\log\s*_\s*(?:\{([^{}]+)\}|([A-Za-z0-9]+))\s*"
        r"(?:\\!\s*)?"
    )
    while True:
        match = pattern.search(text, search_from)
        if match is None:
            return text
        argument_start = match.end()
        while argument_start < len(text) and text[argument_start].isspace():
            argument_start += 1
        if argument_start >= len(text) or text[argument_start] != "(":
            search_from = match.end()
            continue
        depth = 0
        argument_end = None
        for index in range(argument_start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    argument_end = index
                    break
        if argument_end is None:
            return text
        base = match.group(1) or match.group(2)
        argument = text[argument_start + 1:argument_end]
        replacement = f"(log({argument})/log({base}))"
        text = text[:match.start()] + replacement + text[argument_end + 1:]
        search_from = match.start() + len(replacement)


def _balanced_tex_atom(value: str, start: int) -> tuple[str, int] | None:
    """Return one conservative TeX atom and its exclusive end offset."""
    text = str(value or "")
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    while text.startswith(r"\!", index):
        index += 2
        while index < len(text) and text[index].isspace():
            index += 1
    if index >= len(text):
        return None

    def balanced(opening: str, closing: str, offset: int) -> int | None:
        depth = 0
        for cursor in range(offset, len(text)):
            if text[cursor] == opening:
                depth += 1
            elif text[cursor] == closing:
                depth -= 1
                if depth == 0:
                    return cursor + 1
        return None

    if text[index] in "({":
        closing = ")" if text[index] == "(" else "}"
        end = balanced(text[index], closing, index)
        return (text[index:end], end) if end is not None else None

    command = re.match(r"\\(?:sqrt|(?:d?frac|tfrac))", text[index:])
    if command:
        is_fraction = "frac" in command.group(0)
        cursor = index + command.end()
        if text[index:cursor].endswith("sqrt") and cursor < len(text) and text[cursor] == "[":
            root_end = balanced("[", "]", cursor)
            if root_end is None:
                return None
            cursor = root_end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        first_end = balanced("{", "}", cursor) if cursor < len(text) and text[cursor] == "{" else None
        if first_end is None:
            return None
        cursor = first_end
        if is_fraction:
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            second_end = balanced("{", "}", cursor) if cursor < len(text) and text[cursor] == "{" else None
            if second_end is None:
                return None
            cursor = second_end
        return text[index:cursor], cursor

    token = re.match(r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z]|\\pi)\b", text[index:])
    if token:
        end = index + token.end()
        return text[index:end], end
    return None


def _has_ambiguous_unparenthesized_function(value: str) -> bool:
    """Detect function arguments whose TeX extent cannot be inferred safely."""
    text = _strip_math_wrappers(value)
    pattern = re.compile(
        r"\\(?:sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|"
        r"coth|sech|csch|exp|ln)(?![A-Za-z])\s*(?:\\!\s*)?"
    )
    for match in pattern.finditer(text):
        atom = _balanced_tex_atom(text, match.end())
        if atom is None:
            suffix = text[match.end():].lstrip()
            if suffix and suffix[0] not in "+-=,;:.)]}|":
                return True
            continue
        if atom[0].startswith("("):
            continue
        suffix = text[atom[1]:].lstrip()
        if suffix and (
            suffix[0].isalnum()
            or suffix.startswith((r"\sqrt", r"\frac", r"\dfrac", r"\tfrac"))
        ):
            return True
    return False


def _replace_bare_log_bases(value: str) -> str:
    r"""Rewrite ``\log_b x`` when ``x`` is one unambiguous TeX atom."""
    text = str(value or "")
    pattern = re.compile(
        r"\\log\s*_\s*(?:\{([^{}]+)\}|([A-Za-z0-9]+))\s*(?:\\!\s*)?"
    )
    search_from = 0
    while True:
        match = pattern.search(text, search_from)
        if match is None:
            return text
        atom = _balanced_tex_atom(text, match.end())
        if atom is None or atom[0].startswith("("):
            search_from = match.end()
            continue
        suffix = text[atom[1]:].lstrip()
        if suffix and (
            suffix[0].isalnum()
            or suffix.startswith((r"\sqrt", r"\frac", r"\dfrac", r"\tfrac"))
        ):
            search_from = atom[1]
            continue
        base = match.group(1) or match.group(2)
        replacement = f"(log({atom[0]})/log({base}))"
        text = text[:match.start()] + replacement + text[atom[1]:]
        search_from = match.start() + len(replacement)


def _parenthesize_function_atoms(value: str) -> str:
    """Parenthesize a single TeX atom following an elementary function."""
    text = str(value or "")
    pattern = re.compile(
        r"\\(?:sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|"
        r"coth|sech|csch|exp|ln|Gamma)(?![A-Za-z])\s*(?:\\!\s*)?"
    )
    search_from = 0
    while True:
        match = pattern.search(text, search_from)
        if match is None:
            return text
        atom = _balanced_tex_atom(text, match.end())
        if atom is None or atom[0].startswith("("):
            search_from = match.end()
            continue
        suffix = text[atom[1]:].lstrip()
        if suffix and (
            suffix[0].isalnum()
            or suffix.startswith((r"\sqrt", r"\frac", r"\dfrac", r"\tfrac"))
        ):
            search_from = atom[1]
            continue
        replacement = text[match.start():match.end()] + f"({atom[0]})"
        text = text[:match.start()] + replacement + text[atom[1]:]
        search_from = match.start() + len(replacement)


def _replace_symbolic_binomial_coefficients(value: str) -> str:
    r"""Translate unambiguous scalar ``\binom{u}{k}`` terms for SymPy.

    TeX also uses ``\binom{a}{b}`` as a compact two-entry column vector, so
    treating every occurrence as a binomial coefficient is unsafe.  The
    scalar equivalence layer only translates the conservative form whose
    lower argument is a nonnegative integer and whose upper argument contains
    a variable or arithmetic operation.  Purely numeric pairs remain
    untouched and are handled, if applicable, by the structured-object
    parser.
    """
    text = str(value or "")
    search_from = 0
    while True:
        match = re.search(r"\\binom(?![A-Za-z])", text[search_from:])
        if match is None:
            return text
        start = search_from + match.start()
        cursor = search_from + match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            search_from = cursor
            continue
        upper_end = find_matching_brace(text, cursor)
        if upper_end < 0:
            return text
        lower_start = upper_end + 1
        while lower_start < len(text) and text[lower_start].isspace():
            lower_start += 1
        if lower_start >= len(text) or text[lower_start] != "{":
            search_from = upper_end + 1
            continue
        lower_end = find_matching_brace(text, lower_start)
        if lower_end < 0:
            return text

        upper = text[cursor + 1:upper_end].strip()
        lower = text[lower_start + 1:lower_end].strip()
        safe_upper = bool(
            upper
            and re.fullmatch(r"[A-Za-z0-9_+\-*/^().\s]+", upper)
            and (re.search(r"[A-Za-z_]", upper) or re.search(r"[+\-*/^()]", upper))
        )
        safe_lower = bool(re.fullmatch(r"\d+", lower))
        if not safe_upper or not safe_lower or int(lower) > 1000:
            search_from = lower_end + 1
            continue

        replacement = f"binomial(({upper}),({lower}))"
        text = text[:start] + replacement + text[lower_end + 1:]
        search_from = start + len(replacement)


def _parse_scalar_expression(value: str):
    text = _strip_math_wrappers(value)
    if not text or len(text) > 600 or "=" in text or "&" in text:
        return None
    if _MATRIX_ENVIRONMENT.search(text) or len(_split_top_level_items(text)) > 1:
        return None
    text = re.sub(r"√\s*\(\s*([^()]+)\s*\)", r"\\sqrt{\1}", text)
    text = re.sub(r"√\s*([0-9A-Za-z]+)", r"\\sqrt{\1}", text)
    text = re.sub(
        r"\\sqrt\s*([0-9A-Za-z])",
        r"\\sqrt{\1}",
        _normalize_fraction_commands(text),
    )
    # Both ``2\pi i`` and ``2 i\pi`` are conventional scalar notation.
    # Canonicalize the latter before implicit multiplication is inserted.
    text = re.sub(r"(?<![A-Za-z0-9_])i\s*\\pi", r"\\pi i", text)
    text = _replace_symbolic_binomial_coefficients(text)
    text = _replace_bare_log_bases(
        _replace_parenthesized_log_bases(
            _parenthesize_function_atoms(
                _replace_balanced_e_powers(
                    _replace_balanced_fractions(_replace_indexed_roots(text))
                )
            )
        )
    )
    text = text.replace(r"\Gamma", "gamma")
    text = re.sub(
        r"\\lvert\s*([^|\n]+?)\s*\\rvert|(?<!\\)\|\s*([^|\n]+?)\s*\|",
        lambda match: f"Abs({match.group(1) or match.group(2)})",
        text,
    )
    # TeX commonly omits spacing in ``\log|x|``.  Replacing the absolute
    # value delimiters first creates ``\logAbs(x)``; restore the function
    # boundary before the restricted symbolic parser sees it.
    text = re.sub(
        r"\\(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|coth|"
        r"sech|csch|exp|log|ln)(?=Abs\()",
        r"\\\1 ",
        text,
    )
    text = text.replace(r"\lambda", "L").replace(r"\cdot", "*").replace(r"\times", "*")
    try:
        from tools.sympy_tool import SympyTool

        text = re.sub(
            r"\\(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|coth|sech|csch|exp|log|ln)\s*\{\s*\(([^()]*)\)\s*\}",
            lambda match: rf"\{match.group(1)}({match.group(2)})",
            text,
        )
        text = re.sub(
            r"\\(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|coth|sech|csch|exp|log|ln)\s*\{([^{}]+)\}",
            lambda match: rf"\{match.group(1)}({match.group(2)})",
            text,
        )
        text = re.sub(
            r"\\(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|coth|sech|csch|exp|log|ln)(?![A-Za-z])",
            r" \1 ",
            text,
        )
        text = (
            text.replace("arcsin", "asin")
            .replace("arccos", "acos")
            .replace("arctan", "atan")
        )
        prepared = SympyTool._latex_to_sympy(text)
        prepared = re.sub(
            r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|coth|sech|csch|exp|log)\s+"
            r"Abs\(([^()]*)\)",
            r"\1(Abs(\2))",
            prepared,
        )
        prepared = re.sub(
            r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|coth|sech|csch|exp|log)\s+([0-9A-Za-z]+)\b",
            r"\1(\2)",
            prepared,
        )
        prepared = re.sub(
            r"\b(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|coth|sech|csch|exp|log|Abs)\s*\(",
            r"\1(",
            prepared,
        )
        # In a scalar mathematical expression, standalone e and i use their
        # conventional Euler/imaginary meanings. Labelled variables are
        # handled by the equation/assignment paths before this parser.
        prepared = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "E", prepared)
        prepared = re.sub(r"(?<![A-Za-z0-9_])i(?![A-Za-z0-9_])", "I", prepared)
        prepared = re.sub(
            r"(?<=[0-9A-Za-z)])\s+(?=[0-9A-Za-z(])",
            "*",
            prepared,
        )
        prepared = re.sub(
            r"(?<![A-Za-z0-9_])([A-Za-z])(?=\()",
            r"\1*",
            prepared,
        )
        # SymPy reserves a few uppercase one-letter names (notably ``S``).
        # Lowercase them only when that cannot merge two distinct variables.
        uppercase = set(re.findall(
            r"(?<![A-Za-z0-9_])([A-DF-HJ-Z])(?![A-Za-z0-9_])",
            prepared,
        ))
        if any(re.search(
            rf"(?<![A-Za-z0-9_]){letter.lower()}(?![A-Za-z0-9_])",
            prepared,
        ) for letter in uppercase):
            return None
        prepared = re.sub(
            r"(?<![A-Za-z0-9_])([A-DF-HJ-Z])(?![A-Za-z0-9_])",
            lambda match: match.group(1).lower(),
            prepared,
        )
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

        difference = sp.simplify(left_expression - right_expression)
        if difference == 0:
            return True
        # SymPy deliberately leaves some exact inverse-trigonometric
        # identities unevaluated.  For closed, exact expressions only, a
        # high-precision agreement is a conservative secondary certificate.
        if (
            not difference.free_symbols
            and not left_expression.atoms(sp.Float)
            and not right_expression.atoms(sp.Float)
        ):
            numeric = sp.N(difference, 80)
            if numeric.is_finite is True and abs(complex(numeric)) < 1e-50:
                return True
        return False
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
    text = (
        text.replace(r"\(", "")
        .replace(r"\)", "")
        .replace("$", "")
    )
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
        r"(?:for\s+some\s+)?(?:arbitrary\s+)?real\s+constants?\s+"
        r"(?P<names>[a-z](?:\s*(?:,|and)\s*[a-z])*)",
        r"for\s+some\s+real\s+constant\s+"
        r"(?P<names>[a-z])",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            names.update(re.findall(
                r"(?<![a-z])[a-z](?![a-z])", match.group("names")
            ))
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
        r"(?P<parameter>[a-z])\s*(?:\\cdot|\*)?\s*"
        r"(?P<variable>[a-z])\s*\)",
    )
    calls = list(call_pattern.finditer(text))
    all_calls = list(re.finditer(r"\\?(?:cosh|cos)\s*\(", text))
    if not calls or len(calls) != len(all_calls):
        return ()
    for match in calls:
        if (
            match.group("parameter") not in real_parameters
            or match.group("parameter") == match.group("variable")
        ):
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


def _jordan_block_signature(value: str, *, marker_present: bool) -> tuple[int, ...] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    blocks = re.findall(
        r"(\d+)\s*个\s*(?:大小|阶数|阶)?\s*(?:为|是)?\s*(\d+)\s*"
        r"(?:阶)?\s*的?\s*Jordan\s*块|"
        r"(\d+)\s+Jordan\s+blocks?\s+of\s+size\s+(\d+)",
        text,
        re.IGNORECASE,
    )
    if blocks:
        sizes: list[int] = []
        for chinese_count, chinese_size, english_count, english_size in blocks:
            count = int(chinese_count or english_count)
            size = int(chinese_size or english_size)
            if count < 1 or size < 1 or count * size > 10000:
                return None
            sizes.extend([size] * count)
        return tuple(sorted(sizes, reverse=True))
    if not marker_present:
        return None
    entries = _parse_tuple_vector(text)
    if entries is None or not entries:
        return None
    if any(not re.fullmatch(r"\s*\d+\s*", item) or int(item) < 1 for item in entries):
        return None
    return tuple(sorted((int(item) for item in entries), reverse=True))


def _jordan_block_multiset_match(left: str, right: str) -> bool | None:
    marker = re.compile(r"Jordan\s*块|Jordan\s+blocks?", re.IGNORECASE)
    left_marker = bool(marker.search(left))
    right_marker = bool(marker.search(right))
    if not left_marker and not right_marker:
        return None
    # A response that also states rank or a minimal polynomial is a composite
    # nilpotent-operator answer.  Comparing only its block multiset would hide
    # disagreements in those required fields, so defer it to the stricter
    # summary matcher below.
    composite = re.compile(
        r"\\operatorname\s*\{\s*rank\s*\}|\\?rank|秩|"
        r"minimal\s+polynomial|最小多项式",
        re.IGNORECASE,
    )
    if composite.search(left) or composite.search(right):
        return None
    left_signature = _jordan_block_signature(left, marker_present=right_marker)
    right_signature = _jordan_block_signature(right, marker_present=left_marker)
    if left_signature is None or right_signature is None:
        return False
    return left_signature == right_signature


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
    equations = [
        equation
        for fragment in _math_result_fragments(value)
        if (equation := _split_single_top_level_equals(
            _unwrap_text_commands(_strip_math_wrappers(fragment))
        )) is not None
        and re.search(r"[A-Za-z]\s*\(\s*[A-Za-z]\s*\)", equation[0])
    ]
    if not equations:
        return ()
    equation = equations[-1]
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
    text = re.sub(
        r"^\s*[a-z]\s*(?:\\in|∈|\bin\b)\s*(?=\\?\{)",
        "",
        text,
        count=1,
    )
    # Exponential formulas are ubiquitous. Enter this specialized matcher
    # only when the answer is syntactically a finite set/list of roots; a lone
    # Laplace transform, stability boundary, or PDE solution must continue to
    # the ordinary equation comparators.
    root_container = bool(
        (r"\{" in text and r"\}" in text)
        or re.match(r"^\s*(?:[a-z]\s*=\s*)?0\s*[,，]\s*1\s*[,，]", text)
    )
    if not root_container:
        return None
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
        r"e\s*\^\s*\{?\s*i\s*(?P<parameter>[a-z])\s*\\?pi\s*/\s*(?P<half>\d+)\s*\}?",
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
        # Only an actual nested integer-set prefix activates the authoritative
        # malformed-set verdict. TeX level sets such as ``\{|f|>K\}`` and
        # families ``\{f_n\}`` are unrelated mathematical objects.
        return () if re.search(r"\{\s*\{\s*[-+]?\d", text, re.DOTALL) else None

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


def _finite_integer_set_signature(value: str) -> frozenset[int] | tuple[()] | None:
    """Parse a finite integer set or a bare comma-separated set answer."""
    text = _strip_math_wrappers(value)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = re.sub(r"[\s$]", "", text)
    braced = text.startswith("{") and text.endswith("}")
    payload = text[1:-1] if braced else text
    if not re.fullmatch(r"[-+]?\d+(?:[,，][-+]?\d+)+", payload):
        return None
    values = [int(item) for item in re.split(r"[,，]", payload)]
    signature = frozenset(values)
    return signature if len(signature) == len(values) else ()


def _finite_integer_set_match(left: str, right: str) -> bool | None:
    left_signature = _finite_integer_set_signature(left)
    right_signature = _finite_integer_set_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    # A bare ordered list becomes set-like only when the other side explicitly
    # uses braces. Two bare lists retain their ordinary ordering semantics.
    left_braced = "{" in _strip_math_wrappers(left) or r"\{" in left
    right_braced = "{" in _strip_math_wrappers(right) or r"\{" in right
    if not (left_braced or right_braced):
        return None
    return bool(left_signature and left_signature == right_signature)


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
    def candidates(value: str) -> tuple[str, ...]:
        direct = _strip_math_wrappers(value)
        asserted = _explicit_result_math_values(value)
        return tuple(dict.fromkeys((direct, *asserted)))

    left_candidates = candidates(left)
    right_candidates = candidates(right)
    expanded_left = tuple(_expand_binary_entropy(item) for item in left_candidates)
    expanded_right = tuple(_expand_binary_entropy(item) for item in right_candidates)
    left_entropy = any(count for _, count in expanded_left)
    right_entropy = any(count for _, count in expanded_right)
    if not left_entropy and not right_entropy:
        return None
    try:
        import sympy as sp

        for left_expression, _ in expanded_left:
            for right_expression, _ in expanded_right:
                if left_expression is None or right_expression is None:
                    continue
                if left_expression.free_symbols or right_expression.free_symbols:
                    continue
                if sp.simplify(left_expression - right_expression) == 0:
                    return True
        return False
    except Exception:
        return False


def _runge_kutta_stability_signature(
    value: str,
) -> tuple[object, tuple[str, ...], float] | None:
    """Parse a complete explicit-RK negative-axis stability result."""
    text = normalize_latex(str(value or ""))
    if not (
        re.search(r"(?:绝对)?稳定(?:函数|域|区间)|stability", text, re.IGNORECASE)
        and re.search(r"R\s*\(\s*z\s*\)", text, re.IGNORECASE)
    ):
        return None

    stability_function = None
    boundary_coefficients: tuple[str, ...] | None = None
    try:
        import sympy as sp

        for fragment in _math_result_fragments(text):
            equation = _split_single_top_level_equals(fragment.strip(" $，,；;。"))
            if equation is None:
                continue
            lhs, rhs = equation
            lhs_key = _compact(lhs)
            if lhs_key == "r(z)":
                expression = _parse_scalar_expression(rhs)
                if expression is not None and {
                    symbol.name.lower() for symbol in expression.free_symbols
                } <= {"z"}:
                    stability_function = expression
                continue
            if _compact(rhs) != "0":
                continue
            expression = _parse_scalar_expression(lhs)
            if expression is None or len(expression.free_symbols) != 1:
                continue
            symbol = next(iter(expression.free_symbols))
            name = symbol.name.lower()
            if name not in {"r", "z"}:
                continue
            positive_radius = sp.Symbol("positive_radius")
            transformed = sp.expand(
                expression.subs(symbol, -positive_radius if name == "z" else positive_radius)
            )
            polynomial = sp.Poly(transformed, positive_radius)
            if polynomial.degree() < 2:
                continue
            boundary_coefficients = tuple(
                str(sp.simplify(coefficient))
                for coefficient in polynomial.monic().all_coeffs()
            )
    except Exception:
        return None

    endpoint_match = re.search(
        r"(?:z|r)\s*_\s*(?:\{\s*\*\s*\}|\*)?[^$，,。；;\n]{0,12}?"
        r"(?:\\approx|≈|~=)\s*([-+]?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if endpoint_match is None:
        endpoint_match = re.search(
            r"(?:\\approx|≈|~=)\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*0\s*\]",
            text,
            re.IGNORECASE,
        )
    if endpoint_match is None:
        return None
    endpoint = abs(float(endpoint_match.group(1)))
    if stability_function is None or boundary_coefficients is None or endpoint <= 0:
        return None
    return stability_function, boundary_coefficients, endpoint


def _runge_kutta_stability_match(left: str, right: str) -> bool | None:
    left_signature = _runge_kutta_stability_signature(left)
    right_signature = _runge_kutta_stability_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    try:
        import sympy as sp

        function_equal = bool(sp.simplify(left_signature[0] - right_signature[0]) == 0)
    except Exception:
        return False
    return bool(
        function_equal
        and left_signature[1] == right_signature[1]
        and abs(left_signature[2] - right_signature[2]) <= 1e-7
    )


def _multistep_stability_summary_match(left: str, right: str) -> bool | None:
    """Compare multi-obligation linear-multistep summaries conservatively."""

    def conclusion(text: str, positive: str, negative: str) -> bool | None:
        if re.search(negative, text, re.IGNORECASE):
            return False
        if re.search(positive, text, re.IGNORECASE):
            return True
        return None

    def order(text: str) -> int | None:
        match = re.search(
            r"(?:阶数|精度阶|order)\s*(?:为|是|=|:|：|is)?\s*"
            r"(?P<first>\d+|一|二|三|四)|"
            r"(?P<second>\d+|一|二|三|四)\s*阶",
            text,
            re.IGNORECASE,
        )
        if match is None:
            return None
        token = match.group("first") or match.group("second")
        return int(token) if token.isdigit() else {"一": 1, "二": 2, "三": 3, "四": 4}.get(token)

    def signature(value: str):
        text = _unwrap_text_commands(normalize_latex(str(value or "")))
        if not re.search(r"(?:\\?xi|ξ)", text, re.IGNORECASE):
            return None
        if not re.search(
            r"z\s*\(\s*(?:\\?theta|θ)\s*\)|稳定边界|stability\s+boundary",
            text,
            re.IGNORECASE,
        ):
            return None
        equations = {
            _compact(span)
            for span in re.findall(r"\$([^$\n]+)\$", text)
            if "=" in span and re.search(r"(?:\\?xi|ξ|z\s*\()", span, re.IGNORECASE)
        }
        zero_stable = conclusion(
            text,
            r"零稳定|zero[- ]stable|满足(?:了)?根条件|"
            r"satisf(?:y|ies|ied)[^.;\n]{0,30}root\s+condition|"
            r"root\s+condition\s+(?:holds|is\s+satisfied)",
            r"非零稳定|不零稳定|不是零稳定|not\s+zero[- ]stable|"
            r"不满足根条件|root\s+condition\s+(?:fails|is\s+not\s+satisfied)",
        )
        a_stable = conclusion(
            text,
            r"A\s*[- ]?稳定|A[- ]stable",
            r"(?:不|非|不是|并非)\s*A\s*[- ]?稳定|not\s+A[- ]stable",
        )
        parsed_order = order(text)
        if len(equations) < 2 or zero_stable is None or a_stable is None or parsed_order is None:
            return None
        return equations, zero_stable, parsed_order, a_stable

    first = signature(left)
    second = signature(right)
    if first is None or second is None:
        return None
    return bool(
        len(first[0].intersection(second[0])) >= 2
        and first[1:] == second[1:]
    )


def _gaussian_curvature_signature(value: str) -> str:
    candidates: list[str] = []
    for fragment in _math_result_fragments(value):
        if not re.match(r"\s*K(?:\s*\([^)]*\))?\s*=", fragment, re.IGNORECASE):
            continue
        terminal = _terminal_chain_expression(fragment)
        if terminal and _parse_scalar_expression(terminal) is not None:
            candidates.append(terminal)
    text = normalize_latex(str(value or ""))
    for match in re.finditer(
        r"(?:Gauss(?:ian)?\s*(?:曲率|curvature)|高斯曲率)\s*"
        r"(?:为|是|等于|is|equals?|=|[:：])\s*\$?\s*"
        r"(?P<value>[^$，,。；;\n]+)",
        text,
        re.IGNORECASE,
    ):
        candidate = match.group("value").strip()
        if _parse_scalar_expression(candidate) is not None:
            candidates.append(candidate)
    return candidates[-1] if candidates else ""


def _gaussian_curvature_function_match(left: str, right: str) -> bool | None:
    left_signature = _gaussian_curvature_signature(left)
    right_signature = _gaussian_curvature_signature(right)
    if not left_signature or not right_signature:
        return None
    return _scalar_math_match(left_signature, right_signature)


def _wasserstein_transport_signature(value: str) -> tuple[str, str] | None:
    """Parse W_2^2 together with an explicitly optimal one-dimensional map."""
    text = normalize_latex(str(value or ""))
    if not re.search(
        r"(?:最优[^。；;\n]{0,12}(?:映射|传输)|"
        r"optimal[^.\n]{0,24}(?:map|transport))",
        text,
        re.IGNORECASE,
    ):
        return None

    distance = ""
    transport = ""
    for fragment in _math_result_fragments(text):
        equation = _split_single_top_level_equals(fragment.strip(" $，,；;。"))
        if equation is None:
            continue
        lhs, rhs = equation
        if re.match(
            r"\s*W\s*_\s*\{?2\}?\s*\^\s*\{?2\}?",
            lhs,
            re.IGNORECASE,
        ) and _parse_scalar_expression(rhs) is not None:
            distance = rhs.strip()
        if re.fullmatch(r"\s*T\s*\(\s*x\s*\)\s*", lhs, re.IGNORECASE):
            expression = _parse_scalar_expression(rhs)
            if expression is not None and {
                symbol.name.lower() for symbol in expression.free_symbols
            } <= {"x"}:
                transport = rhs.strip()
    return (distance, transport) if distance and transport else None


def _wasserstein_transport_match(left: str, right: str) -> bool | None:
    left_signature = _wasserstein_transport_signature(left)
    right_signature = _wasserstein_transport_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if left_signature is None or right_signature is None:
        return False
    return bool(
        _scalar_math_match(left_signature[0], right_signature[0])
        and _scalar_math_match(left_signature[1], right_signature[1])
    )


def _rouche_zero_count_signature(value: str) -> tuple[str, int, bool] | None:
    text = normalize_latex(str(value or ""))
    theorem_marker = bool(re.search(r"Rouch(?:é|e)|儒歇", text, re.IGNORECASE))
    count_matches = list(re.finditer(
        r"(?:恰有|共有|有|为|即为)?\s*\$?\s*(\d+)\s*\$?\s*个?\s*零点|"
        r"(?:exactly|has|have|there\s+are)\s*\$?\s*(\d+)\s*\$?\s*zeros?",
        text,
        re.IGNORECASE,
    ))
    if not count_matches:
        return None
    count = int(next(group for group in count_matches[-1].groups() if group is not None))
    radius = "1" if re.search(r"单位圆盘|unit\s+disk", text, re.IGNORECASE) else ""
    boundary = re.search(
        r"(?:\\lvert|\|)\s*z\s*(?:\\rvert|\|)\s*(?:=|<)\s*"
        r"([+-]?\d+(?:/\d+|\.\d+)?)",
        text,
    )
    if boundary:
        radius = boundary.group(1)
    if not radius or _parse_scalar_expression(radius) is None:
        return None
    return radius, count, theorem_marker


def _rouche_zero_count_match(left: str, right: str) -> bool | None:
    left_signature = _rouche_zero_count_signature(left)
    right_signature = _rouche_zero_count_signature(right)
    if left_signature is None or right_signature is None:
        return None
    if not left_signature[2] and not right_signature[2]:
        return None
    return bool(
        left_signature[1] == right_signature[1]
        and _scalar_math_match(left_signature[0], right_signature[0])
    )


def _uniform_series_signature(value: str) -> tuple[str, bool, bool] | None:
    text = normalize_latex(str(value or ""))
    compact = re.sub(r"\s+", "", text)
    if "[0,r]" not in compact or "[0,1)" not in compact:
        return None
    local = bool(
        re.search(
            r"\[\s*0\s*[,，]\s*r\s*\][^。；;\n]{0,90}(?<!不)一致收敛|"
            r"uniform(?:ly)?\s+conver[^.;\n]{0,90}\[\s*0\s*,\s*r\s*\]|"
            r"\[\s*0\s*,\s*r\s*\][^.;\n]{0,90}uniform(?:ly)?\s+conver",
            text,
            re.IGNORECASE,
        )
    )
    global_negative = bool(
        re.search(
            r"\[\s*0\s*[,，]\s*1\s*\)[^。；;\n]{0,90}不一致收敛|"
            r"不一致收敛[^。；;\n]{0,90}\[\s*0\s*[,，]\s*1\s*\)|"
            r"(?:not|does\s+not)\s+(?:converge\s+uniformly|uniformly\s+converge)"
            r"[^.;\n]{0,90}\[\s*0\s*,\s*1\s*\)",
            text,
            re.IGNORECASE,
        )
    )
    sum_candidates: list[str] = []
    for fragment in _math_result_fragments(text):
        if not re.search(r"\\?log\s*\(", fragment, re.IGNORECASE):
            continue
        terminal = _terminal_chain_expression(fragment)
        candidate = terminal or _strip_math_wrappers(fragment)
        if _parse_scalar_expression(candidate) is not None:
            sum_candidates.append(candidate)
    if not sum_candidates:
        return None
    return sum_candidates[-1], local, global_negative


def _uniform_series_summary_match(left: str, right: str) -> bool | None:
    left_signature = _uniform_series_signature(left)
    right_signature = _uniform_series_signature(right)
    if left_signature is None or right_signature is None:
        return None
    return bool(
        left_signature[1:] == right_signature[1:] == (True, True)
        and _scalar_math_match(left_signature[0], right_signature[0])
    )


def _optimization_signature(value: str) -> tuple[tuple[str, ...], str] | None:
    text = _unwrap_text_commands(_strip_math_wrappers(value))
    text = text.replace(r"\left", "").replace(r"\right", "")
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
        r"(?:最优值|最大值|最小值|optimal\s+value|\\max|\\min|"
        r"max(?:imum)?|min(?:imum)?)\s*"
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
        r"最优(?:解|值)|最大值|最小值|optimal\s+(?:solution|value)|optimizer|"
        r"\\(?:max|min)\s*=|\b(?:maximum|minimum)\s*=",
        re.IGNORECASE,
    )
    left_signature = _optimization_signature(left)
    right_signature = _optimization_signature(right)
    if left_signature is None and right_signature is None:
        return None
    if not marker.search(left) and not marker.search(right):
        return None
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


def _finite_exact_order_count_signature(value: str) -> tuple[int, int] | None:
    """Parse a stated count of elements having one exact finite order."""
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    patterns = (
        r"阶\s*(?:恰为|正好为|等于)\s*\$?\s*(?P<order>\d+)\s*\$?"
        r"(?P<body>[^。；;\n]{0,180})",
        r"(?:elements?\s+of\s+exact\s+order|exact(?:ly)?\s+order)\s*"
        r"(?P<order>\d+)(?P<body>[^.；;\n]{0,180})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        body = match.group("body")
        if not re.search(
            r"(?:元素(?:数|共有)?|共有|个|number\s+of\s+elements|"
            r"there\s+are|count\s+is)",
            body,
            re.IGNORECASE,
        ):
            continue
        integers = [int(item) for item in re.findall(r"(?<![A-Za-z_])\d+", body)]
        if integers:
            return int(match.group("order")), integers[-1]
    return None


def _finite_exact_order_count_match(left: str, right: str) -> bool | None:
    left_signature = _finite_exact_order_count_signature(left)
    right_signature = _finite_exact_order_count_signature(right)
    if left_signature is not None and right_signature is not None:
        return left_signature == right_signature

    def bare_integer(value: str) -> int | None:
        answer = _strip_math_wrappers(_answer_value(value)).strip()
        return int(answer) if re.fullmatch(r"[-+]?\d+", answer) else None

    if left_signature is not None:
        direct = bare_integer(right)
        return left_signature[1] == direct if direct is not None else None
    if right_signature is not None:
        direct = bare_integer(left)
        return right_signature[1] == direct if direct is not None else None
    return None


def _vitali_conclusion_signature(value: str) -> tuple[bool, bool] | None:
    """Recognize the two conclusions in the finite-measure Vitali theorem."""
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    integrable = bool(re.search(
        r"(?<![A-Za-z])f\s*(?:\\in|∈)\s*L\s*\^?\s*\{?\s*1\s*\}?",
        text,
        re.IGNORECASE,
    ))
    l_one_convergence = bool(
        re.search(
            r"(?:\\lVert|\\\||\|)\s*f\s*_\s*\{?n\}?\s*-\s*f\s*"
            r"(?:\\rVert|\\\||\|)\s*_\s*\{?\s*1\s*\}?\s*"
            r"(?:\\to|→)\s*0",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\\int[^。；;\n]{0,80}(?:\\lvert|\|)\s*f\s*_\s*\{?n\}?\s*-\s*f\s*"
            r"(?:\\rvert|\|)[^。；;\n]{0,40}(?:\\to|→)\s*0",
            text,
            re.IGNORECASE,
        )
    )
    return (integrable, l_one_convergence) if integrable and l_one_convergence else None


def _vitali_conclusion_match(left: str, right: str) -> bool | None:
    left_signature = _vitali_conclusion_signature(left)
    right_signature = _vitali_conclusion_signature(right)
    if left_signature is None or right_signature is None:
        return None
    return left_signature == right_signature == (True, True)


def _stationary_distribution_signature(
    value: str,
) -> tuple[tuple[str, ...], bool] | None:
    """Parse a finite stationary vector and whether detailed balance is asserted."""
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    labelled = bool(re.search(
        r"平稳分布|稳态分布|stationary\s+distribution", text, re.IGNORECASE
    ))
    match = re.search(
        r"\\pi\s*=\s*\((?P<entries>[^()]+)\)\s*"
        r"(?:/\s*(?P<denominator>\d+))?",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    entries = tuple(_split_top_level_items(match.group("entries")))
    if len(entries) < 2 or any(not entry for entry in entries):
        return None
    denominator = match.group("denominator")
    if denominator:
        entries = tuple(f"({entry})/({denominator})" for entry in entries)
    if any(_parse_scalar_expression(entry) is None for entry in entries):
        return None
    detailed_balance = bool(re.search(
        r"细致平衡|详细平衡|detailed\s+balance|"
        r"\\pi\s*_\s*\{?i\}?\s*\\lambda\s*_\s*\{?i\}?\s*=\s*"
        r"\\pi\s*_\s*\{?i\s*\+\s*1\}?\s*\\mu\s*_\s*\{?i\s*\+\s*1\}?",
        text,
        re.IGNORECASE,
    ))
    if not labelled and not detailed_balance:
        return None
    return entries, detailed_balance


def _stationary_distribution_match(left: str, right: str) -> bool | None:
    left_signature = _stationary_distribution_signature(left)
    right_signature = _stationary_distribution_signature(right)
    if left_signature is None or right_signature is None:
        return None
    left_entries, left_balance = left_signature
    right_entries, right_balance = right_signature
    return bool(
        len(left_entries) == len(right_entries)
        and all(
            _scalar_entry_match(a, b)
            for a, b in zip(left_entries, right_entries)
        )
        and left_balance == right_balance
    )


def _uniform_integrability_signature(
    value: str,
) -> tuple[bool, bool, str] | None:
    """Capture UI, a.e.-zero convergence, and the asserted L1 norm."""
    text = _unwrap_text_commands(normalize_latex(str(value or "")))
    ui_match = re.search(r"一致可积|uniformly\s+integrable", text, re.IGNORECASE)
    if ui_match is None:
        return None
    prefix = text[max(0, ui_match.start() - 10):ui_match.start()]
    uniformly_integrable = not bool(re.search(r"不|非|not\s*$", prefix, re.IGNORECASE))
    almost_everywhere_zero = bool(
        re.search(
            r"(?:f\s*_\s*\{?n\}?[^。；;\n]{0,40}(?:\\to|→)\s*0"
            r"[^。；;\n]{0,30}(?:几乎处处|a\.?e\.?))|"
            r"(?:(?:几乎处处|a\.?e\.?)[^。；;\n]{0,40}"
            r"f\s*_\s*\{?n\}?[^。；;\n]{0,20}(?:\\to|→)\s*0)",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:每个|任意)\s*\$?\s*x\s*\$?[^。；;\n]{0,80}"
            r"f\s*_\s*\{?n\}?\s*\(\s*x\s*\)\s*(?:\\to|→)\s*0",
            text,
            re.IGNORECASE,
        )
    )
    norm = re.search(
        r"(?:\\lVert|\\\||\|)\s*f\s*_\s*\{?n\}?\s*"
        r"(?:\\rVert|\\\||\|)\s*_\s*\{?\s*1\s*\}?\s*=\s*"
        r"(?:(?:[A-Za-z]|\\[A-Za-z]+)\s*\([^\)\n]{1,48}\)\s*=\s*)?"
        r"(?P<value>\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}|"
        r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not almost_everywhere_zero or norm is None:
        return None
    norm_value = norm.group("value")
    if _parse_scalar_expression(norm_value) is None:
        return None
    return uniformly_integrable, almost_everywhere_zero, norm_value


def _uniform_integrability_match(left: str, right: str) -> bool | None:
    left_signature = _uniform_integrability_signature(left)
    right_signature = _uniform_integrability_signature(right)
    if left_signature is None or right_signature is None:
        return None
    return bool(
        left_signature[:2] == right_signature[:2]
        and _scalar_math_match(left_signature[2], right_signature[2])
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
    interval_literal = (
        r"[\[(]\s*[^,，\]\)\n]+\s*[,，]\s*[^,，\]\)\n]+\s*[\])]"
    )
    interval_semantics = bool(
        re.fullmatch(
            rf"\s*(?:\\boxed\s*\{{)?{interval_literal}(?:\}})?\s*",
            text,
        )
        or re.search(rf"(?:\\in\b|∈)\s*{interval_literal}", text, re.IGNORECASE)
        or re.search(
            rf"(?:区间|范围|定义域|解集|"
            rf"\b(?:interval|range|domain|solution\s+set)\b)"
            rf"\s*(?:约为|为|是|is|=|[:：])\s*{interval_literal}",
            text,
            re.IGNORECASE,
        )
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
    if re.search(r"\\(?:approx|simeq)|≈|~=", f"{left} {right}", re.IGNORECASE):
        return None
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
    negative = re.search(
        r"(?:不是|不属于|不成立|不可|错误|发散|否)|"
        r"\b(?:false|no|incorrect)\b",
        text,
        re.IGNORECASE,
    )
    positive = re.search(
        r"(?:是|属于|成立|可以|正确|收敛)|\b(?:true|yes|correct)\b",
        text,
        re.IGNORECASE,
    )
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


def _terminal_judgement(value: str) -> str:
    text = _strip_math_wrappers(value)
    match = re.search(
        r"(?:为|是|[:：])?\s*(不成立|错误|否|false|incorrect|"
        r"成立|正确|是|true|correct)\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    token = match.group(1).casefold()
    return (
        "negative"
        if token in {"不成立", "错误", "否", "false", "incorrect"}
        else "positive"
    )


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


def _math_span_assignment_list_match(left: str, right: str) -> bool | None:
    """Compare complete multi-result assignments embedded in concise prose.

    Only explicit dollar or ``\\(...\\)`` math spans are inspected.  This is
    intentionally narrower than scanning arbitrary proof text: it is meant for
    answer lines such as a matrix, spectral radius, and requested iterates.
    """
    left_parts = _math_span_assignment_parts(_answer_value(left))
    right_parts = _math_span_assignment_parts(_answer_value(right))
    if left_parts is None and right_parts is None:
        return None
    if left_parts is None or right_parts is None:
        # A one-sided collection may simply be explanatory derivation beside
        # a concise scalar answer.  It supplies no safe list-level verdict;
        # dedicated scalar/count/family matchers below may still compare it.
        return None
    if left_parts.keys() != right_parts.keys():
        return False
    return all(
        _component_match(left_parts[key], right_parts[key])
        for key in left_parts
    )


def _math_span_assignment_parts(value: str) -> dict[str, str] | None:
    text = normalize_latex(str(value or ""))
    parts: dict[str, str] = {}
    conflicting_key = False
    for match in re.finditer(
        r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)",
        text,
        re.DOTALL,
    ):
        fragment = (match.group("dollar") or match.group("paren") or "").strip()
        equation = _split_single_top_level_equals(fragment)
        if equation is None:
            continue
        lhs, rhs = equation
        key = _assignment_key(lhs) or _safe_expression_assignment_key(lhs)
        if not key or not rhs.strip():
            continue
        if key in parts:
            # A repeated label is acceptable only when it repeats the same
            # value.  Conflicting revisions must never be hidden by this
            # equivalence path.
            if not _component_match(parts[key], rhs):
                conflicting_key = True
                break
            continue
        parts[key] = rhs.strip()
    if conflicting_key or len(parts) < 2:
        return None
    return parts


def _safe_expression_assignment_key(value: str) -> str:
    """Canonicalize a bounded symbolic lhs, never a matrix or prose clause."""
    text = normalize_latex(str(value or "")).strip()
    if (
        not text
        or len(text) > 180
        or "=" in text
        or _MATRIX_ENVIRONMENT.search(text)
        or re.search(r"(?:<=|>=|<|>|≤|≥|\\\\leq?|\\\\geq?)", text)
        or not re.search(r"[A-Za-z]|\\\\[A-Za-z]+", text)
        or not re.fullmatch(r"[A-Za-z0-9_{}^()\\.\s+*/-]+", text)
    ):
        return ""
    return _compact(text)


def _repeated_function_family_match(left: str, right: str) -> bool | None:
    """Compare short unordered families written as repeated function assignments."""
    left_family = _repeated_function_family(_answer_value(left))
    right_family = _repeated_function_family(_answer_value(right))
    if left_family is None and right_family is None:
        return None
    if left_family is None or right_family is None:
        # Set-builder and parameterized-family notations may describe the same
        # solutions without repeating a function assignment.  Leave those
        # asymmetric representations to the dedicated family matchers below.
        return None
    left_name, left_values = left_family
    right_name, right_values = right_family
    if left_name != right_name or len(left_values) != len(right_values):
        return False
    remaining = list(right_values)
    for left_args, value in left_values:
        match_index = next(
            (
                index
                for index, (right_args, other) in enumerate(remaining)
                if _alpha_function_component_match(
                    value,
                    left_args,
                    other,
                    right_args,
                )
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _alpha_function_component_match(
    left: str,
    left_args: tuple[str, ...],
    right: str,
    right_args: tuple[str, ...],
) -> bool:
    """Compare two function formulas modulo bound-argument renaming.

    Free parameter names remain authoritative.  The bound variables are
    replaced only after both scalar expressions have parsed successfully, so
    a rename cannot silently merge a free parameter with an argument.
    """
    if len(left_args) != len(right_args):
        return False
    left_expression = _parse_scalar_expression(left)
    right_expression = _parse_scalar_expression(right)
    if left_expression is None or right_expression is None:
        return False
    try:
        import sympy as sp

        left_bound = tuple(sp.Symbol(item.lower()) for item in left_args)
        right_bound = tuple(sp.Symbol(item.lower()) for item in right_args)
        if any(item not in left_expression.free_symbols for item in left_bound):
            return False
        if any(item not in right_expression.free_symbols for item in right_bound):
            return False
        if (
            left_expression.free_symbols - set(left_bound)
            != right_expression.free_symbols - set(right_bound)
        ):
            return False
        canonical = tuple(sp.Dummy(f"bound_{index}") for index in range(len(left_args)))
        normalized_left = left_expression.xreplace(dict(zip(left_bound, canonical)))
        normalized_right = right_expression.xreplace(dict(zip(right_bound, canonical)))
        return bool(sp.simplify(normalized_left - normalized_right) == 0)
    except Exception:
        return False


def _repeated_function_family(
    value: str,
) -> tuple[str, tuple[tuple[tuple[str, ...], str], ...]] | None:
    text = _unwrap_text_commands(normalize_latex(str(value or ""))).strip()
    text = re.sub(
        r"(?i)^\s*(?:【\s*)?(?:最终答案|答案|final\s+answer|answer)(?:\s*】)?\s*[:：]?\s*",
        "",
        text,
    )
    # TeX spacing commands such as ``\ `` and ``\,`` are presentation-only
    # here.  Removing them before recognizing separators also lets a comma
    # followed by ``\ `` start the next assignment cleanly.
    text = re.sub(r"\\(?:quad|qquad)|\\(?=\s)|\\[,;:!]", " ", text)
    # Finite solution families are commonly separated by prose ``or`` rather
    # than commas.  Split only when the following clause starts another
    # function assignment; an ``or`` inside a formula or explanation remains
    # untouched.
    text = re.sub(
        r"\s+(?:or|或)\s+(?=(?:[A-Za-z]|\\[A-Za-z]+)\s*\()",
        ",",
        text,
        flags=re.IGNORECASE,
    )
    pieces = _split_assignment_pieces(text)
    if not 2 <= len(pieces) <= 8:
        return None
    function_key = ""
    values: list[tuple[tuple[str, ...], str]] = []
    for piece in pieces:
        equation = _split_single_top_level_equals(
            _clean_assignment_piece(piece).strip(" \t\r\n$")
        )
        if equation is None:
            return None
        lhs, rhs = equation
        function = re.fullmatch(
            r"\s*(?P<name>[A-Za-z]|\\[A-Za-z]+)\s*"
            r"\(\s*(?P<args>[A-Za-z](?:\s*,\s*[A-Za-z]){0,3})\s*\)\s*",
            lhs,
        )
        if function is None or not rhs.strip():
            return None
        # A real-domain qualifier is redundant when attached to a displayed
        # function identity.  Do not discard integer, positive, interval, or
        # parameter restrictions: those can change the represented family.
        rhs = re.sub(
            r"\s*(?:for\s+(?:all|every)\s+(?:real\s+)?[A-Za-z]"
            r"(?:\s*(?:\\in|in)\s*(?:\\mathbb\s*\{?R\}?|R|"
            r"the\s+reals?))?|"
            r"对(?:任意|所有)实数\s*[A-Za-z]|"
            r"对于(?:任意|所有)实数\s*[A-Za-z])\s*$",
            "",
            rhs,
            flags=re.IGNORECASE,
        ).strip()
        if not rhs:
            return None
        arguments = tuple(
            item.strip().casefold()
            for item in function.group("args").split(",")
        )
        current = f"{function.group('name').casefold()}/{len(arguments)}"
        if function_key and current != function_key:
            return None
        function_key = current
        values.append((arguments, rhs.strip()))
    return (
        (function_key, tuple(values))
        if function_key and len(values) >= 2
        else None
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
    # A reference answer may append one short positive consequence after a
    # complete assignment list (for example ``故对任意初值收敛``).  Treat it as
    # support only under a deliberately narrow grammar.  Negative or arbitrary
    # prose remains authoritative and prevents assignment-list equivalence.
    if pieces and _terminal_assignment_support_clause(pieces[-1]):
        pieces = pieces[:-1]
    if len(pieces) < minimum_parts:
        return None

    parts: dict[str, str] = {}
    for piece in pieces:
        item = _clean_assignment_piece(piece)
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


def _clean_assignment_piece(value: str) -> str:
    """Remove only balanced display wrappers and terminal punctuation."""
    item = str(value or "").strip()
    for _ in range(2):
        item = item.rstrip("。.!?").strip()
        if len(item) >= 2 and item.startswith("$") and item.endswith("$"):
            item = item[1:-1].strip()
        elif item.startswith(r"\(") and item.endswith(r"\)"):
            item = item[2:-2].strip()
    return item.strip()


def _terminal_assignment_support_clause(value: str) -> bool:
    item = _clean_assignment_piece(value)
    if not item or len(item) > 96 or "=" in item:
        return False
    if re.search(r"(?:不|未|无|否|not|never|diverg|fail)", item, re.IGNORECASE):
        return False
    return bool(
        re.match(
            r"^(?:故|因此|所以|从而|于是|(?:thus|hence|therefore)\b)",
            item,
            re.IGNORECASE,
        )
        and re.search(
            r"(?:收敛|稳定|成立|可逆|存在|唯一|converg|stable|holds?|"
            r"invertible|exists?|unique)",
            item,
            re.IGNORECASE,
        )
    )


def _terminal_assignment_negative_clause(value: str) -> bool:
    item = _clean_assignment_piece(value)
    return bool(
        len(item) <= 96
        and re.match(
            r"^(?:故|因此|所以|从而|于是|(?:thus|hence|therefore)\b)",
            item,
            re.IGNORECASE,
        )
        and re.search(r"(?:不|未|无|否|not|never|diverg|fail)", item, re.IGNORECASE)
        and re.search(
            r"(?:收敛|稳定|成立|可逆|存在|唯一|converg|stable|holds?|"
            r"invertible|exists?|unique)",
            item,
            re.IGNORECASE,
        )
    )


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
    if any(
        _terminal_assignment_negative_clause(piece)
        for piece in _split_assignment_pieces(normalize_latex(str(value or "")))
    ):
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
    atom = r"(?:[A-Za-z]|\\[A-Za-z]+)"
    decorated_atom = rf"(?:{atom}|\\(?:widehat|hat|bar|tilde)\s*\{{{atom}\}})"
    index = r"(?:\{[A-Za-z0-9()+\-]+\}|[A-Za-z0-9]+)"
    labelled_atom = rf"{decorated_atom}(?:\s*[_^]\s*{index}){{0,2}}"
    function_label = rf"{decorated_atom}\s*\(\s*{labelled_atom}\s*\)"
    if not re.fullmatch(rf"(?:{labelled_atom}|{function_label})", text):
        return ""
    return _compact(text)


def _strip_verified_terminal_bound(value: str) -> str:
    """Strip a true, closed numeric bound from an assigned scalar value.

    This permits ``rho=1/sqrt(6)<1`` to match the same exact scalar without
    accepting a false inequality or discarding symbolic domain information.
    """
    text = _clean_assignment_piece(value)
    relation = re.fullmatch(
        r"(?P<value>.+?)\s*(?P<op><|>|≤|≥|\\leq?|\\geq?)\s*(?P<bound>.+)",
        text,
        re.DOTALL,
    )
    if relation is None:
        return text
    left = _parse_scalar_expression(relation.group("value"))
    right = _parse_scalar_expression(relation.group("bound"))
    if (
        left is None
        or right is None
        or left.free_symbols
        or right.free_symbols
    ):
        return text
    try:
        difference = left - right
        op = relation.group("op")
        verified = {
            "<": difference.is_negative,
            ">": difference.is_positive,
            "≤": difference.is_nonpositive,
            r"\le": difference.is_nonpositive,
            r"\leq": difference.is_nonpositive,
            "≥": difference.is_nonnegative,
            r"\ge": difference.is_nonnegative,
            r"\geq": difference.is_nonnegative,
        }.get(op)
        return relation.group("value").strip() if verified is True else text
    except Exception:
        return text


def _component_match(left: str, right: str) -> bool:
    left_value = _strip_verified_terminal_bound(left)
    right_value = _strip_verified_terminal_bound(right)
    typed_match = _math_object_match(left_value, right_value)
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

        indexed = sorted(set(re.findall(
            r"[A-Za-z]_\{?[A-Za-z0-9]+\}?",
            f"{left} {right}",
        )))
        available = [
            symbol for symbol in "qruvwxyz"
            if not re.search(
                rf"(?<![A-Za-z0-9_]){symbol}(?![A-Za-z0-9_])",
                f"{left} {right}",
            )
        ]
        if len(indexed) > len(available):
            return False
        indexed_mapping = dict(zip(indexed, available))

        def parse(value: str):
            # Braced single-token subscripts are notation variants of the
            # identifier form accepted by the local restricted parser.
            normalized = str(value)
            for token, replacement in indexed_mapping.items():
                normalized = normalized.replace(token, replacement)
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
