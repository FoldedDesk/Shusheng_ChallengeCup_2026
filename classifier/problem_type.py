from __future__ import annotations

import re

from classifier.choice import has_choice_options


def classify_problem_type(problem: str) -> str:
    if re.search(
        r"\b(?:determine|find|compute)\b.+\bor\s+prove\b.+"
        r"(?:does\s+not|need\s+not|no\s+such|not\s+necessarily\s+exist)",
        problem,
        re.IGNORECASE | re.DOTALL,
    ):
        return "calculation"
    if re.search(r"证明|求证|论证|prove|proof", problem, re.IGNORECASE):
        return "proof"
    if re.search(r"推导|derive|derivation", problem, re.IGNORECASE):
        return "derivation"
    if re.search(r"解释|说明|explain|justify", problem, re.IGNORECASE):
        return "explanation"
    if has_choice_options(problem) or re.search(
        r"选择题|选项|multiple choice|choose (?:the )?(?:correct|best)",
        problem,
        re.IGNORECASE,
    ):
        return "choice"
    if re.search(r"填空|填入|fill (?:in|the blank)", problem, re.IGNORECASE):
        return "fill_blank"
    return "calculation"
