"""Bilingual, conservative routing metadata for the public submission path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.difficulty import classify_difficulty
from classifier.problem_type import classify_problem_type
from classifier.choice import has_choice_options
from classifier.advanced_families import (
    LACUNARY_NATURAL_BOUNDARY_PATTERN,
    RUNGE_KUTTA_STABILITY_PATTERN,
    SPECIALIZED_TOPIC_PATTERNS,
    SPECIALIZED_TOPICS,
    TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
)
from classifier.subject import classify_subjects
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
    task_kind: str = "calculation"
    result_kind: str = "number"
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
    (
        "olympiad_functional_equation",
        r"\bfunctional equations?\b|"
        r"\b(?:find|determine|classify)\s+all\s+(?:(?:continuous|differentiable|"
        r"twice[- ]differentiable|real[- ]valued)\s+)*functions?\s+[A-Za-z]?\s*"
        r"(?::|such\s+that|satisfying)\s*[^.!?]{0,500}"
        r"[A-Za-z]\s*\([^)]*\)\s*=\s*[^.!?]{0,300}[A-Za-z]\s*\(",
    ),
    ("olympiad_geometry", r"\b(triangles?|quadrilaterals?|polygons?|hexagons?|hypotenuse|circle|cyclic|tangent|collinear|concurrent|circumcircle|incircle|orthocenter|circumcenter|incenter|angle bisector)\b"),
    (
        "olympiad_number_theory",
        r"\b(integer solutions?|pairs?\s+of\s+(?:(?:positive|nonnegative|nonzero)\s+)?integers?|"
        r"triples?\s+of\s+(?:(?:positive|nonnegative|nonzero)\s+)?integers?|diophantine|"
        r"divisibility|divisible|divides|congruence|modulo|prime numbers?|gcd|lcm|totient|"
        r"pell equation|factorials?|positive divisors?|numerical semigroups?)\b|"
        r"\b(?:(?:positive|nonnegative|whole)\s+)?integers?[^.!?]{0,160}"
        r"(?:represented|representable|expressed|expressible)\s+as\s+a\s+sum\b|\\pmod|\\mid",
    ),
    (
        "olympiad_combinatorics",
        r"\b(colorings?|arrangements?|permutations?|combinations?|pigeonhole|double counting|"
        r"tournaments?|subsets?|ways\s+to|spanning trees?|labeled trees?|posets?|linear extensions?|"
        r"bracelets?|necklaces?|binary strings?|lattice paths?|generating functions?|coefficients?|"
        r"heaps?|losing positions?|winning positions?|codewords?|hamming weight|tilings?|dominoes?|"
        r"tiles?|boards?|chessboards?|partitions?|surjective functions?|injective functions?|boolean variables?|"
        r"satisfying assignments?|conjunctive normal form|selected seats?|graphs?)\b",
    ),
    ("olympiad_inequality", r"\b(inequalit\w*|am[- ]gm|cauchy[- ]schwarz|positive reals?|minimum possible|maximum possible|minimum value|maximum value)\b"),
    ("olympiad_polynomial", r"\b(polynomials?|vieta|integer roots?|real roots?|complex roots?|roots?,\s*counted\s+with\s+multiplicity|monic polynomial)\b"),
    ("olympiad_sequence", r"\b(sequences?|recurrence relations?|recursive sequences?|fibonacci numbers?|lucas numbers?)\b"),
    ("olympiad_general", r"\b(olympiad|imo|aime|amc|math contest|mathematical competition)\b"),
)


_OLYMPIAD_SIGNAL = re.compile(
    r"\b(?:olympiad|imo|aime|amc|math contest|mathematical competition|"
    r"prove|show\s+that|find\s+all|determine\s+all|classify\s+all|for\s+all|"
    r"least\s+possible|greatest\s+possible|smallest\s+possible|largest\s+possible|minimum\s+possible|maximum\s+possible|"
    r"functional equations?|diophantine|pigeonhole|double counting|"
    r"number\s+of\s+ways|how\s+many(?:\s+ways)?|find\s+the\s+number|"
    r"determine\s+the\s+number|find\s+the\s+coefficient|construct|for\s+every|"
    r"least\s+nonnegative\s+residue|greatest\s+integer|least\s+positive\s+integer|"
    r"smallest\s+integer|"
    r"minimum\s+value|maximum\s+value|smallest\s+(?:value|size|cardinality)|"
    r"largest\s+(?:value|size|cardinality)|smallest\s+and\s+largest\s+possible\s+value|find\s+the\s+exact)\b",
    re.IGNORECASE,
)


_OLYMPIAD_GEOMETRY_MARKERS = re.compile(
    r"\b(?:triangles?|quadrilaterals?|polygons?|hexagons?|hypotenuse|circle|cyclic|tangent|collinear|concurrent|"
    r"circumcircle|incircle|orthocenter|circumcenter|incenter|angle bisector)\b",
    re.IGNORECASE,
)


_COMBINATORIAL_OBJECT_PRIORITY = re.compile(
    r"\b(?:famil(?:y|ies).{0,50}subsets?|set systems?|blocks?|surjective functions?|"
    r"injective functions?|bracelets?|necklaces?|binary strings?|lattice paths?|"
    r"linear extensions?|spanning trees?|labeled trees?|tournaments?|colorings?)\b",
    re.IGNORECASE | re.DOTALL,
)


_ADDITIVE_COUNTING_PRIORITY = re.compile(
    r"\b(?:number\s+of\s+ways|how\s+many\s+ways?)\b[^.!?]{0,180}"
    r"\b(?:represented|representable|expressed|expressible|written)\s+as\s+a\s+sum\b|"
    r"(?:多少种|多少个|几种)[^。！？]{0,100}(?:表示|写)(?:成|为)[^。！？]{0,80}(?:之和|的和)",
    re.IGNORECASE | re.DOTALL,
)


_DECISION_GAME_PRIORITY = re.compile(
    r"轮流(?:选择|取|放|移动)|无法(?:行动|选择|移动)[^。！？]{0,40}(?:输|失败)|"
    r"\b(?:players?|contestants?)\s+take\s+turns\s+"
    r"(?:choos(?:e|ing)|select(?:ing)?|remov(?:e|ing)|plac(?:e|ing)|mov(?:e|ing))\b|"
    r"\b(?:no\s+legal\s+move|unable\s+to\s+move|winning\s+strategy|optimal\s+play|"
    r"initial\s+position\s+(?:is\s+)?losing)\b",
    re.IGNORECASE,
)


_STOCHASTIC_MARKERS = re.compile(
    r"概率|随机|期望|方差|掷|骰子|硬币|马尔可夫|"
    r"\b(?:probability|random(?:ly)?|expected|expectation|variance|fair\s+(?:die|dice|coin)|"
    r"roll(?:ing|s|ed)?|markov)\b",
    re.IGNORECASE,
)


_GEOMETRIC_COLLECTION_COMBINATORICS = re.compile(
    r"\b(?:\d+|n|m)\s+(?:non[- ]degenerate\s+)?triangles?\b[\s\S]{0,1000}"
    r"(?:colou?r(?:ed|ing)?|sort(?:ed)?|increasing\s+order|indices?|permutation|"
    r"at\s+least|at\s+most|minimum|maximum|how\s+many)",
    re.IGNORECASE | re.DOTALL,
)


_PARAMETRIC_INEQUALITY_PRIORITY = re.compile(
    r"(?:\binequalit\w*\b|\\geq|\\leq|[≤≥])[^.!?]{0,800}"
    r"(?:largest|greatest|smallest|least|best|optimal)\s+(?:real\s+)?constant|"
    r"(?:largest|greatest|smallest|least|best|optimal)\s+(?:real\s+)?constant"
    r"[^.!?]{0,800}(?:\binequalit\w*\b|\\geq|\\leq|[≤≥])",
    re.IGNORECASE | re.DOTALL,
)


_TOPIC_SUBJECTS = {
    "directed_euler_circuits": "离散数学",
    "plane_rooted_tree_enumeration": "离散数学",
    "lacunary_natural_boundary": "复分析",
    "runge_kutta_stability": "数值分析",
    "spherical_triangle_area": "微分几何",
    "weierstrass_sine_product": "复分析",
    "two_dimensional_polyharmonic_fundamental_solution": "偏微分方程",
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
    subject_route = classify_subjects(text)
    subject = subject_route.primary
    secondary_subject = subject_route.secondary
    subject_confidence = subject_route.confidence
    matched_signals = subject_route.matched_signals
    topic = _classify_topic(lowered)
    if topic in _TOPIC_SUBJECTS:
        topic_subject = _TOPIC_SUBJECTS[topic]
        strong_subject_conflict = bool(
            subject_route.confidence == "high"
            and subject not in {"进阶数学", topic_subject}
        )
        if strong_subject_conflict:
            # A generic contest word must not turn an explicit Fourier, matrix,
            # probability, or numerical-analysis problem into another field.
            topic = "general"
        else:
            if subject not in {"进阶数学", topic_subject}:
                secondary_subject = subject
            subject = topic_subject
            subject_confidence = "high" if subject_route.primary == topic_subject else "medium"
            matched_signals = (*matched_signals, f"{topic_subject}:topic:{topic}")

    answer_shape = _answer_shape(text, problem_type, target)
    if subject == "进阶数学" and answer_shape == "roots" and re.search(
        r"方程|根|零点|\b(?:equation|roots?|zeros?|solutions?)\b",
        target or text,
        re.IGNORECASE,
    ):
        subject = "高等代数"
        subject_confidence = "low"
        matched_signals = (*matched_signals, "高等代数:root-object-fallback")
    language = "mixed" if re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[A-Za-z]", text) else (
        "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
    )
    difficulty = classify_difficulty(text, problem_type)
    if topic in SPECIALIZED_TOPICS:
        difficulty = "hard"
    elif re.search(r"\b(prove|show|construct|counterexample|bijection|induction)\b", lowered):
        difficulty = "hard"
    elif topic.startswith("olympiad_") and topic != "olympiad_general":
        difficulty = "hard"
    elif subject in {"抽象代数", "复分析", "常微分方程"} and difficulty == "medium":
        difficulty = "hard"

    confidence = "high" if answer_shape in {"number", "roots", "expression", "matrix"} else "medium"
    if problem_type in {"proof", "derivation", "explanation"}:
        confidence = "low"
    elif topic in SPECIALIZED_TOPICS:
        confidence = "medium"
    elif topic.startswith("olympiad_"):
        confidence = "medium"
    tool_eligible = (
        problem_type == "calculation"
        and answer_shape in {"number", "roots", "expression", "matrix"}
        and topic not in SPECIALIZED_TOPICS
        and not topic.startswith("olympiad_")
    )
    return ProblemProfile(
        subject=subject,
        problem_type=problem_type,
        difficulty=difficulty,
        answer_shape=answer_shape,
        language=language,
        confidence=confidence,
        tool_eligible=tool_eligible,
        topic=topic,
        task_kind=problem_type,
        result_kind=answer_shape,
        primary_subject=subject,
        secondary_subject=secondary_subject,
        subject_confidence=subject_confidence,
        matched_signals=tuple(dict.fromkeys(matched_signals)),
    )


def _classify_topic(lowered: str) -> str:
    for topic, pattern in SPECIALIZED_TOPIC_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE | re.DOTALL):
            return topic
    has_olympiad_signal = bool(_OLYMPIAD_SIGNAL.search(lowered))
    if _GEOMETRIC_COLLECTION_COMBINATORICS.search(lowered):
        return "olympiad_combinatorics"
    if _PARAMETRIC_INEQUALITY_PRIORITY.search(lowered):
        return "olympiad_inequality"
    if _ADDITIVE_COUNTING_PRIORITY.search(lowered):
        return "olympiad_combinatorics"
    if (
        has_olympiad_signal
        and _DECISION_GAME_PRIORITY.search(lowered)
        and not _STOCHASTIC_MARKERS.search(lowered)
    ):
        return "olympiad_combinatorics"
    if has_olympiad_signal and _COMBINATORIAL_OBJECT_PRIORITY.search(lowered):
        return "olympiad_combinatorics"
    geometry_markers = set(_OLYMPIAD_GEOMETRY_MARKERS.findall(lowered))
    for topic, pattern in _OLYMPIAD_TOPICS:
        if not re.search(pattern, lowered, re.IGNORECASE | re.DOTALL):
            continue
        inherently_high_risk = topic == "olympiad_functional_equation" or bool(re.search(
            r"\b(?:diophantine|pigeonhole|double counting|roots?,\s*counted\s+with\s+multiplicity|"
            r"recurrence\s+relations?|recursive\s+sequence|a_\{?n\+2\}?)\b",
            lowered,
            re.IGNORECASE,
        ))
        if topic == "olympiad_number_theory" and re.search(
            r"\b(?:totient|pell\s+equation|primitive\s+pythagorean|positive\s+divisors?|"
            r"factorials?|greatest\s+integer|least\s+positive\s+integer|gcd\s*\(|"
            r"divides?\b|divisor\s+sum)\b",
            lowered,
            re.IGNORECASE,
        ):
            inherently_high_risk = True
        configured_geometry = topic == "olympiad_geometry" and len(geometry_markers) >= 2
        if has_olympiad_signal or inherently_high_risk or configured_geometry:
            return topic
    return "general"


def _answer_shape(problem: str, problem_type: str, target: str = "") -> str:
    target_text = target or extract_target_clause(problem)
    lowered = target_text.lower()
    # A requested fundamental solution is a formula even when the prompt also
    # mandates a proof of its distributional normalization.
    if re.search(
        TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
        problem,
        re.IGNORECASE | re.DOTALL,
    ) and re.search(
        r"(?:求|确定|写出|给出|构造)[^。.!?\n]{0,160}(?:基本解|基解)|"
        r"\b(?:find|determine|write|give|construct)\b[^.!?\n]{0,180}"
        r"\bfundamental\s+solution\b",
        problem,
        re.IGNORECASE,
    ):
        return "expression"
    if problem_type in {"proof", "derivation", "explanation"}:
        return "proof"
    if problem_type == "construction":
        return "expression"
    if has_choice_options(problem):
        return "choice"
    if re.search(LACUNARY_NATURAL_BOUNDARY_PATTERN, target_text, re.IGNORECASE | re.DOTALL):
        return "expression"
    if re.search(RUNGE_KUTTA_STABILITY_PATTERN, target_text, re.IGNORECASE | re.DOTALL) and re.search(
        r"稳定函数|stability\s+function|R\s*\(\s*z\s*\)",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    # An interval in a minimax or approximation problem is the domain of the
    # requested polynomial, not the answer itself.  Keep this before the broad
    # interval/range rule below.
    polynomial_is_requested = bool(re.search(
        r"(?:求|确定|构造|写出|给出)[^。.!?\n]{0,180}(?:多项式|逼近式)|"
        r"\b(?:find|determine|construct|give)\b[^.!?\n]{0,180}\bpolynomial\b",
        target_text,
        re.IGNORECASE,
    ))
    roots_are_requested = bool(re.search(
        r"(?:求|确定|写出|给出)\s*(?:该|这个|此|所有|全部|多项式的)*\s*(?:根|零点)|"
        r"(?:求|确定|写出|给出)[^。.!?\n]{0,100}多项式[^。.!?\n]{0,40}"
        r"(?:的)?(?:全部|所有)?(?:根|零点)|"
        r"\b(?:find|determine|give|list)\s+(?:all\s+|the\s+)?(?:roots?|zeros?)\b",
        target_text,
        re.IGNORECASE,
    ))
    if polynomial_is_requested and not roots_are_requested:
        return "expression"
    if re.search(
        r"(?:求|确定|计算|写出|给出)[^。.!?\n]{0,160}(?:同调群|胞腔链复形)|"
        r"\b(?:find|determine|compute|write|give)\b[^.!?\n]{0,160}"
        r"(?:H_?\{?\d+\}?\s*\(|homology\s+groups?|cellular\s+chain\s+complex)",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(
        r"(?:化简|简化)[^。.!?\n]{0,120}(?:表达式|式子)|"
        r"\bsimplif(?:y|ication)\b[^.!?\n]{0,120}\b(?:expression|formula)\b",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    distribution_is_support = bool(re.search(
        r"(?:求|计算)[^。.!?\n]{0,120}概率[^。.!?\n]{0,80}"
        r"(?:说明|指出|识别|认出)[^。.!?\n]{0,40}几何分布|"
        r"\b(?:find|compute|calculate)\b[^.!?\n]{0,120}\bprobability\b"
        r"[^.!?\n]{0,80}\b(?:identify|recognize|describe|state)\b"
        r"[^.!?\n]{0,40}\bgeometric\s+distribution\b",
        target_text,
        re.IGNORECASE,
    ))
    if not distribution_is_support and re.search(
        r"(?:求|确定|写出|给出)[^。.!?\n]{0,80}(?:条件)?(?<!初始)(?<!同)分布(?:律)?|"
        r"\b(?:find|determine|write|give)\s+(?:the\s+)?distribution\s+of\b|"
        r"\b(?:find|determine|write|give)\b[^.!?\n]{0,100}"
        r"\b(?:conditional|joint|marginal|sampling|limiting|stationary|probability)\s+distribution\b|"
        r"\bwhat\s+is\b[^.!?\n]{0,80}\bdistribution\b",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(
        r"\b(?:as\s+a\s+function\s+of|in\s+terms\s+of)\s+\$?[A-Za-z](?![A-Za-z])|"
        r"\b[A-Za-z]\s*=\s*[A-Za-z]\s*\(\s*[A-Za-z]\s*\)",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    # Verification and yes/no questions frequently contain equations, but do
    # not ask for equation roots (for example PDE solution checks).
    value_after_verdict = bool(re.search(
        r"\b(?:if\s+(?:it\s+)?is\s+possible|for\s+which|what|minimum|maximum|"
        r"how\s+many|number\s+of)\b",
        lowered,
    )) or bool(re.search(
        r"^\s*判断(?![^。？！?]*(?:是否|真假|正确|错误))[^。？！?]*(?:区间|数值|值|公式|解集)",
        problem,
    ))
    truth_query = bool(re.search(
        r"是否|是不是|能否|可否|填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|^\s*判断|"
        r"验证.*(?:为解|调和)|\bis\s+it\b|\bwhether\b|\bverify\b|yes\s+or\s+no",
        target_text,
        re.IGNORECASE,
    ))
    result_then_support = bool(re.search(
        r"(?:^|[，,；;。.!?]\s*)(?:计算|求|calculate|compute|find)"
        r"[^。.!?\n]{0,160}(?:积分|范数|\|\|[^\n]{0,40}\|\||contour\s+integral|norm)"
        r"[^。.!?\n]{0,100}(?:说明|解释|并|and|explain)[^。.!?\n]{0,60}"
        r"(?:是否|whether|if)",
        problem,
        re.IGNORECASE,
    ))
    if truth_query and not value_after_verdict and not result_then_support:
        return "truth"
    if re.search(
        r"(?:映射|函数)[^。！？!?\n]{0,50}(?:个数|多少个)|"
        r"\b(?:number of|how many)\s+(?:(?:surjective|injective|bijective)\s+)?(?:functions?|maps?)\b",
        lowered,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", lowered) and re.search(
        r"方程|求解|solve", lowered
    ):
        return "expression"
    # For quadrature and definite integrals, an interval is the integration
    # domain rather than the shape of the requested result.
    if re.search(r"积分|∫|\\int|\bintegral\b", lowered, re.IGNORECASE) and re.search(
        r"计算|近似|数值|精确值|compute|calculate|approx(?:imate|imation)?|exact\s+value",
        lowered,
        re.IGNORECASE,
    ) and not re.search(
        r"参数[^。.!?\n]{0,30}(?:区间|范围)|"
        r"(?:parameter|values?\s+of)[^.!?\n]{0,40}(?:range|interval)",
        lowered,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(
        r"(?:解|solve).*(?:不等式|inequal)|区间|\b(?:interval|range|domain)\b",
        lowered,
    ):
        return "interval"
    if re.search(
        r"\b(?:determine|find)\s+all\s+real\s+(?:numbers?|values?)\b",
        lowered,
    ) and re.search(r"(?:inequal|>=|<=|\\ge|\\le|holds?\s+for\s+every)", lowered):
        return "interval"
    if re.search(r"高斯曲率|hessian", lowered):
        return "expression"
    if re.search(
        r"范数|\|\|[^\n]{0,40}\|\||\bnorm\b",
        problem,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(r"矩阵|matrix|determinant|行列式", lowered):
        return "matrix"
    if re.search(
        r"多项式.*(?:所有)?(?:根|零点)|"
        r"\bfind\s+(?:all\s+|the\s+)?(?:roots?|zeros?)\s+of\s+(?:the\s+|a\s+)?polynomial\b",
        lowered,
    ):
        return "roots"
    if re.search(
        r"\b(?:find\s+the\s+complete\s+set\s+of|find\s+all|determine\s+all)\s+integers?\b",
        lowered,
    ):
        return "roots"
    if re.search(
        r"(?:一个)?(?:指标|方法|定理|性质|图形|统计量|概念)(?:是|为)?\s*"
        r"(?:[（(]\s*[）)]\s*)?$|"
        r"导致[^。！？!?\n]{0,60}(?:方向|变化|方差)\s*[（(]\s*[）)]\s*$",
        target_text,
        re.IGNORECASE,
    ):
        return "expression"
    if re.search(r"微分方程|热方程|波动方程|laplace方程|曲线|曲面|函数|function|导数|derivative|积分|integral|极限|limit", lowered):
        return "expression"
    if re.search(r"数列|递推|sequence|recurrence", lowered):
        return "expression"
    if re.search(r"\b[xy]\s*'\s*=|d[xy]/d[xt]", lowered):
        return "expression"
    if re.search(
        r"(?:求解|解)\s*(?:代数)?方程|方程.*(?:所有)?(?:根|解)|"
        r"\b(?:solve\s+(?:the\s+)?equation|"
        r"(?:find|determine|list)\s+(?:all\s+|the\s+)?"
        r"(?:(?:real|complex|integer|integral|rational|positive|nonnegative)\s+)?"
        r"(?:roots?|zeros?|solutions?))\b",
        lowered,
    ):
        return "roots"
    return "number"
