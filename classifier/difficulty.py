from __future__ import annotations

import re


HARD_MARKERS = r"证明|求证|推导|论证|构造|隔板|计数|组合数|排列数|微分方程|通解|留数|曲率|正规子群|热方程|协方差函数|算子范数|开覆盖|线性规划"


def classify_difficulty(problem: str, problem_type: str) -> str:
    if problem_type in {"proof", "derivation", "explanation"}:
        return "hard"
    if re.search(HARD_MARKERS, problem, re.IGNORECASE):
        return "hard"
    if re.search(r"^(?:计算|求)\s*[-+()\d\s./^×÷]+[。？?]?$|简单选择题", problem):
        return "easy"
    if len(problem) <= 30 and re.search(r"^(?:计算|求).*(?:积分|导数|极限|函数值|数值|结果)", problem):
        return "easy"
    return "medium"
