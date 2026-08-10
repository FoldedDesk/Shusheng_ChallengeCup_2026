from __future__ import annotations

import re

from classifier.choice import has_choice_options


_RESULT_COMMAND = re.compile(
    r"(?:^|[。！？!?；;：:，,]\s*|\n)\s*(?:请|试)?\s*"
    r"(?:求(?!证)|计算|求解|解(?:出|方程|不等式)?|判断|验证|比较|构造|列出|写出|给出|指出|确定|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|verify|compare|construct|classify|list|"
    r"what\s+(?:is|are)|how\s+many)\b)",
    re.IGNORECASE,
)

_CONTEXTUAL_RESULT_COMMAND = re.compile(
    r"判断|验证|比较|列出|写出|确定|"
    r"说明[^。！？!?\n]{0,100}是否|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|verify|compare|classify|list|"
    r"what\s+(?:is|are)|how\s+many)\b",
    re.IGNORECASE,
)


def classify_problem_type(problem: str) -> str:
    """Classify the primary task, keeping support requests subordinate.

    For example, ``计算……并说明理由`` and ``compute ... and justify`` are
    calculations. A proof or explanation type is reserved for prompts whose
    primary request is the proof or explanation itself.
    """
    text = str(problem or "")
    if has_choice_options(text) or re.search(
        r"选择题|选项|multiple choice|choose (?:the )?(?:correct|best)",
        text,
        re.IGNORECASE,
    ):
        return "choice"
    if re.search(r"填空|填入|fill (?:in|the blank)", text, re.IGNORECASE):
        return "fill_blank"
    if re.search(
        r"^\s*(?:请|试)?\s*(?:构造|给出[^。！？!?\n]{0,20}构造)|"
        r"^\s*(?:please\s+)?construct\b|"
        r"^\s*(?:please\s+)?(?:give|provide)\b[^.!?\n]{0,30}"
        r"\b(?:construction|example)\b",
        text,
        re.IGNORECASE,
    ):
        return "construction"
    if re.search(
        r"\b(?:determine|find|compute)\b.+\bor\s+prove\b.+"
        r"(?:does\s+not|need\s+not|no\s+such|not\s+necessarily\s+exist)",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return "calculation"
    if re.search(
        r"^\s*(?:请)?\s*(?:给出|写出)\s*(?:一个|完整的|严格的)?\s*(?:证明|论证)|"
        r"^\s*(?:请)?\s*(?:给出|写出)[^。！？!?\n]{1,100}(?:的证明|的论证)|"
        r"^\s*(?:please\s+)?(?:give|provide|write)\s+(?:a\s+|the\s+)?proof\b",
        text,
        re.IGNORECASE,
    ):
        return "proof"
    if re.search(
        r"^\s*(?:请)?\s*(?:解释|说明)\s*(?:为什么|为何)|"
        r"^\s*(?:请)?\s*给出\s*(?:一个|完整的)?\s*(?:解释|说明)|"
        r"^\s*(?:please\s+)?explain\s+why\b",
        text,
        re.IGNORECASE,
    ):
        return "explanation"
    if _RESULT_COMMAND.search(text) or _CONTEXTUAL_RESULT_COMMAND.search(text):
        return "calculation"
    if re.search(r"证明|求证|论证|\b(?:prove|proof|show\s+that)\b", text, re.IGNORECASE):
        return "proof"
    if re.search(r"推导|\b(?:derive|derivation)\b", text, re.IGNORECASE):
        return "derivation"
    if re.search(r"解释|说明|\b(?:explain|justify)\b", text, re.IGNORECASE):
        return "explanation"
    return "calculation"


def classify_task_kind(problem: str) -> str:
    """Explicit name for the primary-task classifier."""
    return classify_problem_type(problem)
