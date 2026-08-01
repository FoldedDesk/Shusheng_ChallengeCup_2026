from __future__ import annotations

import re


def classify_problem_type(problem: str) -> str:
    if re.search(r"证明|求证|论证|prove|proof", problem, re.IGNORECASE):
        return "proof"
    if re.search(r"推导|derive|derivation", problem, re.IGNORECASE):
        return "derivation"
    if re.search(r"解释|说明|explain|justify", problem, re.IGNORECASE):
        return "explanation"
    if re.search(r"选择题|选项|A[．.、]", problem):
        return "choice"
    if re.search(r"填空|填入", problem):
        return "fill_blank"
    return "calculation"
