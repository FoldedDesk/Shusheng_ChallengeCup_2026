"""Case-sensitive option parsing shared by classification and validation."""

from __future__ import annotations

import re


_OPTION_PATTERN = re.compile(
    r"\\item\s*\[\s*([A-E])(?:[.．、)])?\s*\]|"
    r"(?<![A-Za-z])([A-E])[.．、)]|"
    # A parenthesized label is an option only when it is not a function or
    # measure argument such as P(A), mu(B), or μ(A). Start/whitespace and
    # punctuation still allow the common standalone "(A)" option form.
    r"(?<![A-Za-z0-9_\u0370-\u03ff])\(([A-E])\)",
)


def option_labels(problem: str) -> tuple[str, ...]:
    text = str(problem or "")
    labels: list[str] = []
    for match in _OPTION_PATTERN.finditer(text):
        if match.group(2) and match.group(0).endswith(")"):
            # The compact `A)` option spelling shares a suffix with a function
            # argument. Do not reinterpret the closing part of `mu(A)` or
            # `P(A intersection B)` as an answer option.
            line_prefix = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
            if line_prefix.rfind("(") > line_prefix.rfind(")"):
                continue
        label = next((group for group in match.groups() if group), "")
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def has_choice_options(problem: str) -> bool:
    return len(option_labels(problem)) >= 2


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
