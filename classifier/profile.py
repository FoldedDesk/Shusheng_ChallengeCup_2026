"""General problem profile derived solely from the submitted statement."""

from __future__ import annotations

from dataclasses import dataclass
import re

from classifier.choice import has_choice_options
from classifier.difficulty import classify_difficulty
from classifier.problem_type import classify_problem_type
from classifier.subject import classify_subjects
from classifier.target import extract_target_clause


@dataclass(frozen=True)
class ProblemProfile:
    subject: str
    problem_type: str
    difficulty: str
    answer_shape: str
    language: str
    tool_eligible: bool
    task_kind: str = "calculation"
    result_kind: str = "expression"
    topic: str = "general"
    confidence: str = "medium"
    primary_subject: str = "进阶数学"
    secondary_subject: str = ""
    subject_confidence: str = "low"
    matched_signals: tuple[str, ...] = ()

    @property
    def primary(self) -> str:
        return self.primary_subject

    @property
    def secondary(self) -> str:
        return self.secondary_subject

    def trace_content(self) -> dict:
        return {
            "subject": self.subject,
            "primary_subject": self.primary_subject,
            "secondary_subject": self.secondary_subject,
            "subject_confidence": self.subject_confidence,
            "matched_signals": list(self.matched_signals),
            "problem_type": self.problem_type,
            "task_kind": self.task_kind,
            "difficulty": self.difficulty,
            "answer_shape": self.answer_shape,
            "result_kind": self.result_kind,
            "language": self.language,
            "topic": self.topic,
            "confidence": self.confidence,
            "tool_eligible": self.tool_eligible,
        }


_TRUTH = re.compile(
    r"是否|能否|可否|判断(?:下列|该|此)?.*(?:正确|错误|成立)|真假|"
    r"\b(?:true or false|whether|is it true|decide if|determine whether|yes or no)\b|"
    r"^\s*(?!(?:what|which|who|where|when|why|how)\b)"
    r"(?:is|are|does|do|can|could|will|would)\b[^?]{1,300}\?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_ROOTS = re.compile(
    r"解集|解方程|方程[^。！？!?\n]{0,300}(?:全部|所有)(?:解|根)|"
    r"(?:求|找出|确定|列出)[^。！？!?\n]{0,200}(?:根|零点)|"
    r"\bsolve\s+(?:the\s+)?(?:equation|inequality)\b|"
    r"\b(?:find|determine|list)\s+(?:all\s+)?(?:real\s+|complex\s+)?(?:roots?|zeros?)\b|"
    r"\b(?:find|determine|list)\s+all\s+solutions?\b",
    re.IGNORECASE | re.DOTALL,
)

_INTERVAL = re.compile(
    r"解集|定义域|值域|收敛区间|参数范围|在哪些区间|"
    r"\b(?:solution set|domain|range|interval of convergence|parameter range)\b",
    re.IGNORECASE,
)

_COUNT = re.compile(
    r"多少(?:个|种|条|次|对|组|棵|张|项|类)?|数目|数量|个数|"
    r"\b(?:how many|number of|count the|cardinality)\b",
    re.IGNORECASE,
)


def _language(text: str) -> str:
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z]{2,}\b", text))
    return "zh" if han >= max(2, latin_words) else "en"


def _answer_shape(text: str, task_kind: str) -> str:
    target = extract_target_clause(text) or text
    if has_choice_options(text):
        return "choice"
    result_target = re.search(
        r"求(?!证)|计算|确定|"
        r"\b(?:find|determine|compute|calculate|evaluate|solve)\b",
        target,
        re.IGNORECASE,
    )
    if task_kind in {"proof", "explanation"} and not result_target:
        return "proof"
    if _TRUTH.search(target) or _TRUTH.search(text):
        return "truth"
    if re.search(
        r"基本解|弱解|通解|特解|格林函数|Green函数|谱(?!半径)|协方差函数|特征函数|"
        r"最优解|分拆|基本群|同构类型|估计量|估计式|初值问题|边值问题|"
        r"\b(?:fundamental solution|weak solution|general solution|particular solution|"
        r"green(?:'s)? function|spectrum|covariance function|characteristic function|"
        r"optimal solution|partition|fundamental group|isomorphism type|estimator|"
        r"initial value problem|boundary value problem)\b",
        target,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(r"谱半径|\bspectral radius\b", target, re.IGNORECASE):
        return "number"
    if re.search(r"行列式|\\det|\b(?:determinant|rank)\b|(?:矩阵的?)?秩", target, re.IGNORECASE):
        return "number"
    if re.search(r"特征值|\beigenvalues?\b", target, re.IGNORECASE):
        return "expression"
    if re.search(r"逆矩阵|\b(?:inverse matrix|matrix inverse)\b", target, re.IGNORECASE):
        return "matrix"
    if _INTERVAL.search(target):
        return "interval"
    if _ROOTS.search(target) or _ROOTS.search(text):
        return "roots"
    if re.search(r"概率|\bprobability\s+(?:that|of)|\bhitting probability\b", target, re.IGNORECASE):
        return "probability"
    if _COUNT.search(target):
        return "count"
    if re.search(
        r"最小值|最大值|数值|函数值|面积|体积|周期|容量|维数|重数|判别式|范数|风险|常数|系数|曲率|"
        r"\b(?:minimum value|maximum value|value|area|volume|period|capacity|dimension|"
        r"multiplicity|discriminant|norm|risk|constant|coefficient|curvature)\b",
        target,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"公式|表达式|通解|多项式|函数|导数|原函数|级数|推导|"
        r"\b(?:formula|expression|general solution|polynomial|function|derivative|antiderivative|series|derive|derivation)\b",
        target,
        re.IGNORECASE,
    ):
        return "expression"
    return "number"


def _topic(text: str, subject: str, shape: str, task_kind: str) -> str:
    if shape == "choice":
        return "choice"
    if task_kind == "construction":
        return "construction"
    if task_kind in {"proof", "derivation", "explanation"}:
        return "proof"
    rules = (
        ("numerical_method", r"牛顿法|二分法|割线法|有限差分|数值积分|Runge.?Kutta|\b(?:newton|bisection|secant|finite difference|quadrature|runge[- ]kutta)\b"),
        ("calculus", r"极限|导数|积分|级数|\b(?:limit|derivative|integral|series)\b"),
        ("equation", r"方程|不等式|\b(?:equation|inequality|solve)\b"),
        ("linear_algebra", r"矩阵|行列式|特征值|秩|\b(?:matrix|determinant|eigenvalue|rank)\b"),
        ("combinatorics", r"计数|排列|组合|容斥|生成函数|\b(?:counting|permutation|combination|inclusion[- ]exclusion|generating function)\b"),
        ("graph", r"顶点|边集|图论|生成树|\b(?:vertices?|edges?|graph|spanning tree)\b"),
        ("probability", r"概率|随机变量|期望|方差|\b(?:probability|random variable|expectation|variance)\b"),
        ("optimization", r"最小值|最大值|最优|线性规划|\b(?:minimum|maximum|optimal|linear programming)\b"),
    )
    for topic, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return topic
    return subject if subject != "进阶数学" else "general"


def classify_profile(problem: str) -> ProblemProfile:
    text = str(problem or "").strip()
    task_kind = classify_problem_type(text)
    subject_route = classify_subjects(text)
    shape = _answer_shape(text, task_kind)
    difficulty = classify_difficulty(text, task_kind)
    result_kind = {
        "choice": "choice_labels",
        "truth": "judgement",
        "roots": "solution_set",
        "interval": "interval",
        "matrix": "matrix",
        "probability": "probability",
        "count": "integer",
        "proof": "supported_conclusion",
    }.get(shape, shape)
    confidence = subject_route.confidence
    if not text or (subject_route.confidence == "low" and len(text) > 180):
        confidence = "low"
    tool_eligible = bool(
        task_kind in {"calculation", "fill_blank"}
        and shape not in {"choice", "truth", "proof"}
    )
    return ProblemProfile(
        subject=subject_route.primary,
        problem_type=task_kind,
        difficulty=difficulty,
        answer_shape=shape,
        language=_language(text),
        tool_eligible=tool_eligible,
        task_kind=task_kind,
        result_kind=result_kind,
        topic=_topic(text, subject_route.primary, shape, task_kind),
        confidence=confidence,
        primary_subject=subject_route.primary,
        secondary_subject=subject_route.secondary,
        subject_confidence=subject_route.confidence,
        matched_signals=subject_route.matched_signals,
    )
