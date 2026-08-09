from __future__ import annotations

import re


HARD_MARKERS = (
    r"证明|求证|推导|论证|构造|隔板|计数|组合数|排列数|微分方程|通解|留数|曲率|"
    r"正规子群|热方程|协方差函数|算子范数|开覆盖|线性规划|"
    r"prove|construct|counterexample|bijection|induction|differential equation|residue|"
    r"functional equation|diophantine|pigeonhole|inclusion[- ]exclusion|cauchy[- ]schwarz|am[- ]gm"
)

_EXHAUSTIVE_OLYMPIAD = re.compile(
    r"\b(?:find|determine|classify)\s+all\s+(?:positive\s+)?(?:integers?|functions?|polynomials?|"
    r"sequences?|triples?|quadruples?|solutions?|configurations?)\b|"
    r"\b(?:least|greatest|minimum|maximum)\s+possible\b|"
    r"\bfor\s+all\s+(?:positive\s+)?(?:integers?|reals?)\b",
    re.IGNORECASE,
)

_GEOMETRY_CONFIGURATION = re.compile(
    r"\b(?:triangle|quadrilateral|circle|cyclic|tangent|circumcircle|incircle|orthocenter|incenter)\b",
    re.IGNORECASE,
)


def classify_difficulty(problem: str, problem_type: str) -> str:
    if problem_type in {"proof", "derivation", "explanation"}:
        return "hard"
    if re.search(HARD_MARKERS, problem, re.IGNORECASE):
        return "hard"
    if _EXHAUSTIVE_OLYMPIAD.search(problem):
        return "hard"
    if len(_GEOMETRY_CONFIGURATION.findall(problem)) >= 2:
        return "hard"
    if len(problem) >= 240 and re.search(r"\b(?:integer|triangle|polynomial|sequence|function)\b", problem, re.IGNORECASE):
        return "hard"
    if re.search(r"^(?:计算|求)\s*[-+()\d\s./^×÷]+[。？?]?$|简单选择题", problem):
        return "easy"
    if len(problem) <= 30 and re.search(r"^(?:计算|求).*(?:积分|导数|极限|函数值|数值|结果)", problem):
        return "easy"
    return "medium"
