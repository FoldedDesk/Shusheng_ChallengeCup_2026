"""Case-sensitive option parsing shared by classification and validation."""

from __future__ import annotations

import re
import unicodedata


_OPTION_PATTERN = re.compile(
    r"\\item\s*\[\s*([A-E])(?:[.．、)])?\s*\]|"
    r"(?<![A-Za-z])([A-E])[.．、)]|"
    # A parenthesized label is an option only when it is not a function or
    # measure argument such as P(A), mu(B), or μ(A). Start/whitespace and
    # punctuation still allow the common standalone "(A)" option form.
    r"(?<![A-Za-z0-9_\u0370-\u03ff])\(([A-E])\)",
)


def _option_matches(text: str) -> tuple[re.Match[str], ...]:
    """Return option markers after rejecting function-argument lookalikes."""
    matches: list[re.Match[str]] = []
    for match in _OPTION_PATTERN.finditer(text):
        if match.group(2) and match.group(0).endswith(")"):
            # The compact `A)` option spelling shares a suffix with a function
            # argument. Do not reinterpret the closing part of `mu(A)` or
            # `P(A intersection B)` as an answer option.
            line_prefix = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
            if line_prefix.rfind("(") > line_prefix.rfind(")"):
                continue
        matches.append(match)
    return tuple(matches)


def option_labels(problem: str) -> tuple[str, ...]:
    text = str(problem or "")
    labels: list[str] = []
    for match in _option_matches(text):
        label = next((group for group in match.groups() if group), "")
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def choice_options(problem: str) -> tuple[tuple[str, str], ...]:
    """Return ordered ``(label, body)`` pairs for validated option markers.

    Duplicate labels make a choice statement ambiguous, so the parser rejects
    the whole option list rather than silently merging two different bodies.
    """
    text = str(problem or "")
    matches = _option_matches(text)
    if len(matches) < 2:
        return ()
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        label = next((group for group in match.groups() if group), "")
        if not label or label in seen:
            return ()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip(" \t\r\n:：;；,，")
        if not body:
            return ()
        seen.add(label)
        options.append((label, body))
    return tuple(options)


def has_choice_options(problem: str) -> bool:
    return len(option_labels(problem)) >= 2


def choice_stem(problem: str) -> str:
    """Return the text before the first validated option marker."""
    text = str(problem or "")
    for match in _option_matches(text):
        return text[:match.start()].strip(" \t\r\n。.!?？:：")
    return text.strip()


def answer_choice_labels(answer: str) -> tuple[str, ...]:
    value = str(answer or "").strip()
    for _ in range(4):
        value = value.strip()
        if value.startswith(r"\(") and value.endswith(r"\)"):
            value = value[2:-2].strip()
            continue
        if value.startswith(r"\[") and value.endswith(r"\]"):
            value = value[2:-2].strip()
            continue
        if value.startswith("$") and value.endswith("$") and len(value) >= 2:
            value = value[1:-1].strip()
            continue
        unwrapped = re.fullmatch(
            r"\\(?:boxed|text|mathrm|mathbf)\s*\{\s*(.*?)\s*\}",
            value,
            re.DOTALL,
        )
        if not unwrapped:
            break
        value = unwrapped.group(1).strip()
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "").replace("$", "")
    value = re.sub(r"(?i)\b(?:answer|choice|choices|option|options|and)\b|答案|选项|选择", "", value)
    compact = re.sub(r"[\s,，、;；/&+()（）.。:：-]", "", value)
    if not compact or len(compact) > 5 or not re.fullmatch(r"[A-E]+", compact):
        return ()
    labels: list[str] = []
    for label in compact:
        if label not in labels:
            labels.append(label)
    return tuple(labels)


_CHOICE_CUE = re.compile(
    r"(?:正确(?:答案|选项)|答案|选项|应选|故选|选择)\s*(?:为|是|[:：=])?\s*"
    r"[\(（\[]?\s*(?P<zh>[A-E](?:\s*[,，、/&+]\s*[A-E])*)\s*[\)）\]]?|"
    r"\b(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\s*(?:is|are|[:=])?\s*"
    r"[\(\[]?\s*(?P<en>[A-E](?:\s*[,/&+]\s*[A-E])*)\s*[\)\]]?",
    re.IGNORECASE,
)
_LEADING_LABEL = re.compile(
    r"^\s*[\(（\[]?\s*(?P<label>[A-E])\s*[\)）\]]?\s*[.．、:：]\s*\S",
)
_PRESENTATION_COMMANDS = (
    r"\displaystyle",
    r"\textstyle",
    r"\left",
    r"\right",
)
_OUTER_COMMANDS = {"boxed", "text", "mathrm", "mathbf", "mathit"}


def _matching_brace(text: str, open_pos: int) -> int:
    if not (0 <= open_pos < len(text)) or text[open_pos] != "{":
        return -1
    depth = 0
    index = open_pos
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _unwrap_choice_presentation(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    for _ in range(6):
        if text.startswith(r"\(") and text.endswith(r"\)"):
            text = text[2:-2].strip()
            continue
        if text.startswith(r"\[") and text.endswith(r"\]"):
            text = text[2:-2].strip()
            continue
        if text.startswith("$") and text.endswith("$") and len(text) >= 2:
            text = text[1:-1].strip()
            continue
        command = re.match(r"^\\([A-Za-z]+)\s*\{", text)
        if command and command.group(1) in _OUTER_COMMANDS:
            open_pos = text.find("{", command.start())
            close_pos = _matching_brace(text, open_pos)
            if close_pos == len(text) - 1:
                text = text[open_pos + 1:close_pos].strip()
                continue
        break
    return text


def _choice_body_key(value: str) -> str:
    text = _unwrap_choice_presentation(value)
    text = re.sub(
        r"^(?:(?:最终|正确)?答案|正确选项|结论|"
        r"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option|conclusion))"
        r"\s*(?:为|是|is|are|[:：=])\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for command in _PRESENTATION_COMMANDS:
        text = text.replace(command, "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "").replace("$", "")
    text = re.sub(r"[\s,，;；.。:：]+", "", text)
    return text.casefold()


def canonical_choice_answer(answer: str, problem: str) -> str:
    """Return canonical labels only when the answer identifies them uniquely.

    Models sometimes emit the full text of the selected option instead of its
    label.  Shape validation must not turn that transport difference into a
    guessed first option.  Ambiguous prose is deliberately left unchanged.
    """
    original = str(answer or "").strip()
    available_options = choice_options(problem)
    available = {label for label, _ in available_options}
    if not original or not available:
        return original

    direct = answer_choice_labels(original)
    if direct and set(direct) <= available:
        return ",".join(direct)

    visible = _unwrap_choice_presentation(original)
    cue = _CHOICE_CUE.search(visible)
    if cue:
        raw_labels = cue.group("zh") or cue.group("en") or ""
        labels = tuple(dict.fromkeys(re.findall(r"[A-E]", raw_labels.upper())))
        if labels and set(labels) <= available:
            return ",".join(labels)

    leading = _LEADING_LABEL.match(visible)
    if leading and leading.group("label") in available:
        return leading.group("label")

    candidate_key = _choice_body_key(visible)
    if not candidate_key:
        return original
    matches = [
        label
        for label, body in available_options
        if candidate_key == _choice_body_key(body)
    ]
    return matches[0] if len(matches) == 1 else original
