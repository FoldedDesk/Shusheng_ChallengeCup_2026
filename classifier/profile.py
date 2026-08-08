"""Bilingual, conservative routing metadata for the public submission path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.difficulty import classify_difficulty
from classifier.problem_type import classify_problem_type
from classifier.subject import classify_subject


@dataclass(frozen=True)
class ProblemProfile:
    subject: str
    problem_type: str
    difficulty: str
    answer_shape: str
    language: str
    confidence: str
    tool_eligible: bool

    def trace_content(self) -> dict:
        return asdict(self)


_ENGLISH_SUBJECTS = (
    ("抽象代数", r"\b(group|ring|field|ideal|homomorphism|coset|quotient)\b"),
    ("高等代数", r"\b(equation|polynomial|roots?|zeros?)\b"),
    ("线性代数", r"\b(matrix|determinant|eigenvalue|eigenvector|linear transformation|vector space)\b"),
    ("概率论", r"\b(probability|random variable|expectation|variance|distribution|conditional)\b"),
    ("离散数学", r"\b(graph|combinatorics?|permutation|combination|recurrence|counting)\b"),
    ("数论", r"\b(congruen|modulo|prime|divisibility|gcd|integer)\b"),
    ("常微分方程", r"\b(differential equation|initial value|ordinary differential)\b"),
    ("复分析", r"\b(residue|holomorphic|analytic function|contour|laurent)\b"),
    ("数学分析", r"\b(limit|derivative|differentiate|integral|continuous|series|convergen)\b"),
)


def classify_profile(problem: str) -> ProblemProfile:
    """Describe the requested result without guessing an unsafe tool route."""
    text = str(problem or "")
    lowered = text.lower()
    problem_type = classify_problem_type(text)
    subject = classify_subject(text)
    if subject == "进阶数学":
        for candidate, pattern in _ENGLISH_SUBJECTS:
            if re.search(pattern, lowered):
                subject = candidate
                break

    answer_shape = _answer_shape(text, problem_type)
    language = "mixed" if re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[A-Za-z]", text) else (
        "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
    )
    difficulty = classify_difficulty(text, problem_type)
    if re.search(r"\b(prove|show|construct|counterexample|bijection|induction)\b", lowered):
        difficulty = "hard"
    elif subject in {"抽象代数", "复分析", "常微分方程"} and difficulty == "medium":
        difficulty = "hard"

    confidence = "high" if answer_shape in {"number", "roots", "expression", "matrix"} else "medium"
    if problem_type in {"proof", "derivation", "explanation"}:
        confidence = "low"
    tool_eligible = problem_type == "calculation" and answer_shape in {"number", "roots", "expression", "matrix"}
    return ProblemProfile(subject, problem_type, difficulty, answer_shape, language, confidence, tool_eligible)


def _answer_shape(problem: str, problem_type: str) -> str:
    lowered = problem.lower()
    if problem_type in {"proof", "derivation", "explanation"}:
        return "proof"
    # Verification and yes/no questions frequently contain equations, but do
    # not ask for equation roots (for example PDE solution checks).
    if re.search(r"是否|是不是|能否|可否|验证.*(?:为解|调和)|is it|whether|verify", lowered):
        return "truth"
    if re.search(r"不等式|inequal", lowered):
        return "interval"
    if re.search(r"矩阵|matrix|determinant|行列式", lowered):
        return "matrix"
    if re.search(r"微分方程|热方程|波动方程|laplace方程|曲线|曲面|函数|function|导数|derivative|积分|integral|极限|limit", lowered):
        return "expression"
    if re.search(r"\b[xy]\s*'\s*=|d[xy]/d[xt]", lowered):
        return "expression"
    if re.search(r"(?:求解|解|求)\s*(?:代数)?方程|方程.*(?:所有)?(?:根|解)|equations?|roots?|solutions?|零点|zeros?", lowered):
        return "roots"
    return "number"
