"""Bilingual, conservative routing metadata for the public submission path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.difficulty import classify_difficulty
from classifier.problem_type import classify_problem_type
from classifier.choice import has_choice_options
from classifier.subject import classify_subject
from classifier.target import extract_target_clause


@dataclass(frozen=True)
class ProblemProfile:
    subject: str
    problem_type: str
    difficulty: str
    answer_shape: str
    language: str
    confidence: str
    tool_eligible: bool
    topic: str = "general"

    def trace_content(self) -> dict:
        return asdict(self)


_ENGLISH_SUBJECTS = (
    ("初等几何", r"\b(triangle|quadrilateral|polygon|circle|cyclic|tangent|collinear|concurrent|circumcircle|incircle)\b"),
    ("抽象代数", r"\b(group|ring|field|ideal|homomorphism|coset|quotient)\b"),
    ("线性代数", r"\b(matrix|determinant|eigenvalue|eigenvector|linear transformation|vector space)\b"),
    ("概率论", r"\b(probability|random variable|expectation|variance|distribution|conditional)\b"),
    ("离散数学", r"\b(graph|combinatorics?|permutation|combination|recurrence|counting|colorings?|pigeonhole)\b"),
    ("数论", r"\b(congruen\w*|modulo|prime|divisibility|divisible|gcd|diophantine|integer solutions?)\b"),
    ("常微分方程", r"\b(differential equation|initial value|ordinary differential)\b"),
    ("复分析", r"\b(residue|holomorphic|analytic function|contour|laurent)\b"),
    ("高等代数", r"\b(equation|polynomial|roots?|zeros?|functional equation|inequalit)\w*\b"),
    ("数学分析", r"\b(limit|derivative|differentiate|integral|continuous|series|convergen)\b"),
)


_OLYMPIAD_TOPICS = (
    ("olympiad_functional_equation", r"\bfind\s+all\s+functions?\b|\bfunctional equations?\b|f\s*\([^)]*\)\s*[+=].*f\s*\("),
    ("olympiad_geometry", r"\b(triangle|quadrilateral|polygon|circle|cyclic|tangent|collinear|concurrent|circumcircle|incircle|orthocenter|incenter|angle bisector)\b"),
    ("olympiad_number_theory", r"\b(positive integers?|integer solutions?|diophantine|divisibility|divisible|congruence|modulo|prime numbers?|gcd|lcm)\b"),
    ("olympiad_combinatorics", r"\b(colorings?|arrangements?|permutations?|combinations?|pigeonhole|double counting|tournament|subsets?|ways\s+to)\b"),
    ("olympiad_inequality", r"\b(inequalit\w*|am[- ]gm|cauchy[- ]schwarz|positive reals?|minimum possible|maximum possible)\b"),
    ("olympiad_polynomial", r"\b(polynomials?|vieta|integer roots?|real roots?|monic polynomial)\b"),
    ("olympiad_sequence", r"\b(sequences?|recurrence relations?|recursive sequence)\b"),
    ("olympiad_general", r"\b(olympiad|imo|aime|amc|math contest|mathematical competition)\b"),
)


_OLYMPIAD_SIGNAL = re.compile(
    r"\b(?:olympiad|imo|aime|amc|math contest|mathematical competition|"
    r"prove|show\s+that|find\s+all|determine\s+all|classify\s+all|for\s+all|"
    r"least\s+possible|greatest\s+possible|minimum\s+possible|maximum\s+possible|"
    r"functional equations?|diophantine|pigeonhole|double counting|"
    r"number\s+of\s+ways|how\s+many\s+ways)\b",
    re.IGNORECASE,
)


_OLYMPIAD_GEOMETRY_MARKERS = re.compile(
    r"\b(?:triangle|quadrilateral|polygon|circle|cyclic|tangent|collinear|concurrent|"
    r"circumcircle|incircle|orthocenter|incenter|angle bisector)\b",
    re.IGNORECASE,
)


_TOPIC_SUBJECTS = {
    "olympiad_geometry": "初等几何",
    "olympiad_number_theory": "数论",
    "olympiad_combinatorics": "离散数学",
    "olympiad_functional_equation": "高等代数",
    "olympiad_inequality": "高等代数",
    "olympiad_polynomial": "高等代数",
    "olympiad_sequence": "离散数学",
}


def classify_profile(problem: str) -> ProblemProfile:
    """Describe the requested result without guessing an unsafe tool route."""
    text = str(problem or "")
    lowered = text.lower()
    target = extract_target_clause(text)
    problem_type = classify_problem_type(text)
    subject = classify_subject(text)
    topic = _classify_topic(lowered)
    if topic in _TOPIC_SUBJECTS:
        subject = _TOPIC_SUBJECTS[topic]
    if subject == "进阶数学":
        for candidate, pattern in _ENGLISH_SUBJECTS:
            if re.search(pattern, lowered):
                subject = candidate
                break

    answer_shape = _answer_shape(text, problem_type, target)
    language = "mixed" if re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[A-Za-z]", text) else (
        "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
    )
    difficulty = classify_difficulty(text, problem_type)
    if re.search(r"\b(prove|show|construct|counterexample|bijection|induction)\b", lowered):
        difficulty = "hard"
    elif topic.startswith("olympiad_") and topic != "olympiad_general":
        difficulty = "hard"
    elif subject in {"抽象代数", "复分析", "常微分方程"} and difficulty == "medium":
        difficulty = "hard"

    confidence = "high" if answer_shape in {"number", "roots", "expression", "matrix"} else "medium"
    if problem_type in {"proof", "derivation", "explanation"}:
        confidence = "low"
    elif topic.startswith("olympiad_"):
        confidence = "medium"
    tool_eligible = (
        problem_type == "calculation"
        and answer_shape in {"number", "roots", "expression", "matrix"}
        and not topic.startswith("olympiad_")
    )
    return ProblemProfile(subject, problem_type, difficulty, answer_shape, language, confidence, tool_eligible, topic)


def _classify_topic(lowered: str) -> str:
    has_olympiad_signal = bool(_OLYMPIAD_SIGNAL.search(lowered))
    geometry_markers = set(_OLYMPIAD_GEOMETRY_MARKERS.findall(lowered))
    for topic, pattern in _OLYMPIAD_TOPICS:
        if not re.search(pattern, lowered, re.IGNORECASE | re.DOTALL):
            continue
        inherently_high_risk = topic == "olympiad_functional_equation" or bool(re.search(
            r"\b(?:diophantine|pigeonhole|double counting)\b",
            lowered,
            re.IGNORECASE,
        ))
        configured_geometry = topic == "olympiad_geometry" and len(geometry_markers) >= 2
        if has_olympiad_signal or inherently_high_risk or configured_geometry:
            return topic
    return "general"


def _answer_shape(problem: str, problem_type: str, target: str = "") -> str:
    target_text = target or extract_target_clause(problem)
    lowered = target_text.lower()
    if problem_type in {"proof", "derivation", "explanation"}:
        return "proof"
    if has_choice_options(problem):
        return "choice"
    # Verification and yes/no questions frequently contain equations, but do
    # not ask for equation roots (for example PDE solution checks).
    value_after_verdict = bool(re.search(
        r"\b(?:if\s+(?:it\s+)?is\s+possible|for\s+which|what|minimum|maximum|"
        r"how\s+many|number\s+of)\b",
        lowered,
    )) or bool(re.search(
        r"^\s*判断(?![^。？！?]*(?:是否|真假|正确|错误))[^。？！?]*(?:区间|数值|值|公式|解集)",
        target_text,
    ))
    truth_query = bool(re.search(
        r"是否|是不是|能否|可否|填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|^\s*判断|"
        r"验证.*(?:为解|调和)|\bis\s+it\b|\bwhether\b|\bverify\b|yes\s+or\s+no",
        target_text,
        re.IGNORECASE,
    ))
    if truth_query and not value_after_verdict:
        return "truth"
    if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", lowered) and re.search(
        r"方程|求解|solve", lowered
    ):
        return "expression"
    if re.search(r"(?:解|solve).*(?:不等式|inequal)|(?:区间|interval|range|domain)", lowered):
        return "interval"
    if re.search(r"高斯曲率|hessian", lowered):
        return "expression"
    if re.search(r"矩阵|matrix|determinant|行列式", lowered):
        return "matrix"
    if re.search(r"微分方程|热方程|波动方程|laplace方程|曲线|曲面|函数|function|导数|derivative|积分|integral|极限|limit", lowered):
        return "expression"
    if re.search(r"数列|递推|sequence|recurrence", lowered):
        return "expression"
    if re.search(r"\b[xy]\s*'\s*=|d[xy]/d[xt]", lowered):
        return "expression"
    if re.search(
        r"(?:求解|解)\s*(?:代数)?方程|方程.*(?:所有)?(?:根|解)|"
        r"\b(?:solve\s+(?:the\s+)?equation|find\s+(?:all\s+)?(?:roots?|zeros?|solutions?))\b",
        lowered,
    ):
        return "roots"
    return "number"
