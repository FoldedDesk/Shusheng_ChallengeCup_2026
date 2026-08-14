"""Weighted, bilingual subject routing for mathematics problems."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from classifier.advanced_families import (
    DIRECTED_EULER_CIRCUIT_PATTERN,
    LACUNARY_NATURAL_BOUNDARY_PATTERN,
    PLANE_ROOTED_TREE_PATTERN,
    RUNGE_KUTTA_STABILITY_PATTERN,
    SPHERICAL_TRIANGLE_AREA_PATTERN,
    TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
    WEIERSTRASS_SINE_PRODUCT_PATTERN,
)


@dataclass(frozen=True)
class SubjectClassification:
    """Ranked subject evidence derived only from the public problem text."""

    primary: str
    secondary: str
    confidence: str
    matched_signals: tuple[str, ...]
    scores: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _SubjectSignal:
    subject: str
    name: str
    pattern: str
    weight: int


# Generic words such as ``function`` or ``equation`` are deliberately absent:
# they identify the requested object, not a mathematical field.
_SUBJECT_SIGNALS = (
    _SubjectSignal(
        "偏微分方程",
        "two-dimensional-polyharmonic-fundamental-solution",
        TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
        13,
    ),
    _SubjectSignal("偏微分方程", "pde-explicit", r"偏微分方程|偏微分|\bpartial differential equations?\b|\bPDEs?\b", 7),
    _SubjectSignal("偏微分方程", "pde-named", r"热方程|波动方程|Laplace方程|\b(?:heat|wave|laplace) equation\b", 6),
    _SubjectSignal("偏微分方程", "pde-derivatives", r"u\s*_?\{?t\}?\s*=.*u\s*_?\{?xx\}?|u_t.*u_\{?xx\}?", 5),
    _SubjectSignal("线性回归", "regression", r"线性回归|非线性回归|逐步回归|普通最小二乘|\b(?:linear|nonlinear|stepwise) regression\b|\bOLS\b", 7),
    _SubjectSignal("线性回归", "regression-diagnostics", r"异方差|决定系数|回归系数|heteroscedastic|coefficient of determination", 5),
    _SubjectSignal("统计推断", "inference", r"无偏估计|估计量|置信区间|假设检验|\b(?:unbiased estimator|confidence interval|hypothesis test)\b", 7),
    _SubjectSignal("统计推断", "descriptive-statistics", r"样本均值|标准差|时间序列|时间数列|季节调整|数据分散|\b(?:sample mean|standard deviation|time series)\b", 4),
    _SubjectSignal("随机过程", "stochastic-process", r"布朗运动|泊松过程|更新过程|平稳过程|马尔可夫|随机过程|随机游走|\b(?:brownian motion|poisson process|renewal process|markov chain|stochastic process|random walk)\b", 7),
    _SubjectSignal("随机过程", "covariance-function", r"协方差函数|covariance function", 4),
    _SubjectSignal("泛函分析", "functional-analysis", r"Banach|Hilbert|算子范数|有界线性(?:算子|泛函)|\b(?:operator norm|bounded linear operator|bounded linear functional)\b", 7),
    _SubjectSignal(
        "拓扑学",
        "algebraic-topology",
        r"CW\s*(?:复形|complex(?:es)?)|胞腔(?:同调|链复形|边界(?:映射|算子)?)|同调群|"
        r"附着映射|粘附映射|"
        r"\b(?:cellular\s+(?:homology|chain\s+complex|boundary(?:\s+map)?)|"
        r"homology\s+groups?|attaching\s+maps?)\b",
        10,
    ),
    _SubjectSignal("拓扑学", "topology", r"紧致|开覆盖|同胚|拓扑空间|\b(?:compactness|open cover|homeomorphism|topological space)\b", 7),
    _SubjectSignal("运筹学", "optimization", r"线性规划|单纯形|可行域|对偶问题|KKT|\b(?:linear programming|simplex method|feasible region|dual problem)\b", 7),
    _SubjectSignal("抽象代数", "group", r"群同态|正规子群|陪集|商群|群论|二面体群|循环群|阿贝尔群|\b(?:group theory|dihedral group|cyclic group|abelian group|normal subgroup|quotient group|coset)\b", 7),
    _SubjectSignal("抽象代数", "ring-field", r"环同态|多项式环|商环|理想|有限域|域扩张|分裂域|伽罗瓦|Galois扩张|"
                   r"\b(?:quotient ring|polynomial ring|finite field|field extension|splitting field|galois extension|ring homomorphism|ideal)\b", 7),
    _SubjectSignal("抽象代数", "structured-group", r"\bgroup\b.{0,80}\b(?:subgroup|normal|abelian|homomorphism|identity element)\b", 6),
    _SubjectSignal("初等几何", "plane-geometry", r"三角形|四边形|圆周角|外接圆|内切圆|共线|共圆|切线|\b(?:triangles?|quadrilaterals?|polygons?|hexagons?|cyclic|tangent|collinear|concurrent|circumcircle|incircle|orthocenter|circumcenter|incenter|angle bisector)\b", 6),
    _SubjectSignal("初等几何", "circle-geometry", r"\bcircle\b.{0,80}\b(?:radius|diameter|chord|tangent|angle|area)\b", 4),
    _SubjectSignal(
        "微分几何",
        "spherical-triangle-area",
        SPHERICAL_TRIANGLE_AREA_PATTERN,
        12,
    ),
    _SubjectSignal("微分几何", "differential-geometry", r"曲面|弧长参数|主曲率|高斯曲率|第一基本形式|第二基本形式|\b(?:principal curvatures?|gaussian curvature|first fundamental form)\b", 7),
    _SubjectSignal("微分几何", "parametric-curve", r"参数曲线|曲率和挠率|\bcurvature and torsion\b", 5),
    _SubjectSignal(
        "数值分析",
        "runge-kutta-stability",
        RUNGE_KUTTA_STABILITY_PATTERN,
        12,
    ),
    _SubjectSignal("数值分析", "root-method", r"牛顿法|二分法|割线法|\b(?:newton(?:'s)? method|bisection method|secant method)\b", 7),
    _SubjectSignal(
        "数值分析",
        "minimax-approximation",
        r"切比雪夫|极小极大(?:多项式|逼近)?|最佳一致逼近|等振荡|交错定理|"
        r"\b(?:chebyshev|minimax\s+(?:polynomial|approximation)|best\s+uniform\s+approximation|"
        r"equioscillation|alternation\s+theorem)\b",
        10,
    ),
    _SubjectSignal("数值分析", "numerical-method", r"数值积分|插值|条件数|截断误差|有限差分|辛普森|高斯求积|\b(?:numerical integration|interpolation|condition number|truncation error|finite[- ]difference|simpson(?:'s)? rule|gauss[- ]legendre|quadrature)\b", 6),
    _SubjectSignal("线性代数", "matrix", r"矩阵|行列式|特征值|特征向量|线性空间|\b(?:matrix|matrices|determinant|eigenvalue|eigenvector|vector space)\b", 6),
    _SubjectSignal("线性代数", "linear-map", r"线性变换|线性映射|秩与零化度|\b(?:linear transformation|linear map|rank.nullity)\b", 6),
    _SubjectSignal("高等代数", "polynomial", r"多项式(?:的)?(?:根|零点|因式|整除)|韦达|\b(?:polynomials?|polynomial roots?|monic polynomial|irreducible polynomial|vieta)\b|P\s*.{0,8}mathbb\s*\{?[A-Z]\}?\s*\[", 6),
    _SubjectSignal("高等代数", "functional-equation-explicit", r"函数方程|\bfunctional equations?\b", 7),
    _SubjectSignal("常微分方程", "ode-explicit", r"常微分方程|初值问题|相平面|平衡点|\b(?:ordinary differential equations?|initial value problem|phase plane)\b", 7),
    _SubjectSignal("常微分方程", "ode-notation", r"\b[xy]\s*'\s*=|d[xy]\s*/\s*d[xt]", 5),
    _SubjectSignal(
        "复分析",
        "lacunary-natural-boundary",
        LACUNARY_NATURAL_BOUNDARY_PATTERN,
        12,
    ),
    _SubjectSignal(
        "复分析",
        "weierstrass-sine-product",
        WEIERSTRASS_SINE_PRODUCT_PATTERN,
        12,
    ),
    _SubjectSignal("复分析", "complex-analysis", r"留数|复可导|柯西积分|Laurent|全纯|\b(?:residue|holomorphic|contour integral|laurent series|cauchy integral)\b", 7),
    _SubjectSignal("复分析", "complex-analytic", r"解析函数.{0,40}(?:复|区域)|\banalytic function\b.{0,40}\bcomplex\b", 5),
    _SubjectSignal("测度积分", "measure-theory", r"勒贝格|可测|几乎处处|单调收敛定理|支配收敛定理|\b(?:lebesgue|measurable|almost everywhere|dominated convergence theorem)\b", 7),
    _SubjectSignal("概率论", "probability", r"概率|随机变量|条件概率|几何分布|Bernoulli|\b(?:probability|random variable|conditional probability|geometric distribution|bernoulli)\b", 6),
    _SubjectSignal("概率论", "event-probability", r"(?:事件|独立)[^。！？!?\n]{0,100}P\s*\(\s*[A-Z]\s*\)|\bindependent\s+events?\b[^.!?\n]{0,100}\bP\s*\(", 6),
    _SubjectSignal(
        "概率论",
        "moments",
        r"期望|方差|\b(?:expectation|expected value|variance)\b|"
        r"\bexpected\s+(?:total\s+)?(?:winnings?|payoffs?|value|number)\b",
        4,
    ),
    # A bare domain declaration such as "positive integer n" occurs throughout
    # combinatorics, geometry, and games. It is not field evidence by itself.
    _SubjectSignal(
        "数论", "number-theory",
        r"同余|整除|素数|最大公约数|丢番图|数值半群|"
        r"(?:方程|等式)[^。！？\n]{0,120}(?:全部|所有)正整数(?:有序)?解|"
        r"(?:正整数|非负整数|整数|数)[^。！？\n]{0,40}表示成[^。！？\n]{0,60}"
        r"(?:整数|素数|平方数|幂|集合[^。！？\n]{0,15}(?:元素|中的数))(?:之)?和|"
        r"\b(?:integer solutions?|diophantine|divisibility|divisible|congruence|modulo|"
        r"primes?|prime numbers?|gcd|lcm|numerical semigroups?)\b|"
        r"\b(?:(?:positive|nonnegative|whole)\s+)?integers?[^.!?]{0,160}"
        r"(?:represented|representable|expressed|expressible)\s+as\s+a\s+sum\b",
        6,
    ),
    _SubjectSignal("数论", "mod-notation", r"\bmod\s*\d+|\\pmod\s*\{?\d+\}?", 4),
    _SubjectSignal(
        "离散数学",
        "finite-field-graph-flow",
        r"无处零流|处处非零流|图流多项式|循环空间|圈空间|Tutte\s*多项式|"
        r"\b(?:nowhere[- ]zero\s+(?:graph\s+)?flows?|flow\s+polynomial|cycle\s+space|"
        r"tutte\s+polynomial)\b",
        11,
    ),
    _SubjectSignal(
        "离散数学",
        "directed-euler-circuits",
        DIRECTED_EULER_CIRCUIT_PATTERN,
        12,
    ),
    _SubjectSignal(
        "离散数学",
        "plane-rooted-tree-enumeration",
        PLANE_ROOTED_TREE_PATTERN,
        11,
    ),
    _SubjectSignal("离散数学", "graph-theory", r"图论|简单图|有向图|无向图|二分图|图的顶点|图的边|生成树|\b(?:graph theory|graphs?|simple graph|directed graph|undirected graph|bipartite graph|vertices and edges|spanning trees?)\b", 7),
    _SubjectSignal(
        "离散数学",
        "latin-square",
        r"拉丁方|拉丁矩阵|行列(?:均|各)(?:为|是)?排列|"
        r"每个符号[^。！？\n]{0,50}每行(?:和|与|、)?每列[^。！？\n]{0,30}(?:恰好|正好)?出现一次|"
        r"\b(?:latin\s+squares?|rows?\s+and\s+columns?\s+(?:are|form)\s+permutations?|"
        r"each\s+row\s+and\s+(?:each\s+)?column\s+(?:is|forms?)\s+a\s+permutation|"
        r"row(?:\s*[/&-]\s*|\s+and\s+)column\s+permutations?|"
        r"each\s+symbol\s+(?:must\s+)?occurs?\s+exactly\s+once\s+in\s+every\s+row\s+"
        r"and\s+(?:in\s+)?every\s+column|"
        r"every\s+row\s+and\s+(?:every\s+)?column\s+contains?\s+each\s+symbol\s+exactly\s+once)\b",
        10,
    ),
    _SubjectSignal("离散数学", "combinatorics", r"排列|组合|计数|鸽巢|容斥|\b(?:combinatorics?|permutations?|combinations?|counting|colorings?|pigeonhole|inclusion.exclusion|arrangements?|bracelets?|necklaces?|binary strings?|lattice paths?|linear extensions?)\b", 6),
    _SubjectSignal("离散数学", "finite-structures", r"\b(?:posets?|tournaments?|hypercubes?|labeled trees?|codewords?|hamming weight|generating functions?|domino tilings?|unit squares?|chessboards?|tiles?|heaps?|losing positions?|winning positions?)\b", 6),
    _SubjectSignal(
        "离散数学",
        "grid-growth",
        r"(?:染色|着色|绿色)[^。！？\n]{0,80}(?:格子|方格|网格)[^。！？\n]{0,120}"
        r"(?:邻域|相邻|扩散|传播|增长)|"
        r"\b(?:green|colou?red)\s+cells?\b[^.!?\n]{0,160}"
        r"\b(?:spread|grow(?:th)?|neighbou?rhood)\b|"
        r"\b(?:spread|grow(?:th)?)\b[^.!?\n]{0,160}"
        r"\b(?:grid\s+neighbou?rhood|neighbou?r(?:ing)?\s+cells?)\b",
        7,
    ),
    _SubjectSignal(
        "离散数学",
        "colored-array-slices",
        r"(?:单位立方体|小立方体)[^。！？\n]{0,160}(?:染色|颜色)[\s\S]{0,500}"
        r"(?:截面|薄片|方向)|"
        r"\bunit\s+cubes?\b[^.!?]{0,180}\b(?:colou?red|painted)\b[\s\S]{0,700}"
        r"\b(?:slices?|rectangular\s+prisms?|orientations?)\b|"
        r"\b(?:colou?red|painted)\s+unit\s+cubes?\b[\s\S]{0,700}"
        r"\b(?:slices?|rectangular\s+prisms?|orientations?)\b",
        7,
    ),
    _SubjectSignal("离散数学", "finite-game", r"轮流(?:选择|取|放|移动)|无法(?:行动|选择|移动)[^。！？]{0,40}(?:输|失败)|"
                   r"\b(?:(?:players?|contestants?)\s+take\s+turns\s+(?:choos(?:e|ing)|select(?:ing)?|remov(?:e|ing)|plac(?:e|ing)|mov(?:e|ing))|"
                   r"no\s+legal\s+move|unable\s+to\s+move|winning\s+strategy|optimal\s+play|initial\s+position\s+(?:is\s+)?losing)\b", 7),
    _SubjectSignal("离散数学", "finite-set-extremal", r"集合[^。！？!?]{0,80}(?:最小规模|最少元素|最大规模)|"
                   r"\b(?:smallest|minimum|largest|maximum)\s+(?:size|cardinality)\s+of\s+a\s+set\b", 6),
    _SubjectSignal("离散数学", "function-counting", r"(?:映射|函数)[^。！？!?]{0,50}(?:个数|多少)|\b(?:number of|how many)\s+(?:(?:surjective|injective|bijective)\s+)?(?:functions?|maps?)\b", 6),
    _SubjectSignal("离散数学", "bijection-structure", r"(?:双射|一一对应)[^。！？\n]{0,120}(?:坐标平移|横纵坐标|格点|有限集)|"
                   r"\b(?:bijection|bijective map)\b[^.!?\n]{0,140}(?:coordinate translations?|nonnegative integer lattice|finite sets?)", 5),
    _SubjectSignal("离散数学", "additive-counting", r"\b(?:number\s+of\s+ways|how\s+many\s+ways?)\b[^.!?]{0,180}"
                   r"\b(?:represented|representable|expressed|expressible|written)\s+as\s+a\s+sum\b|"
                   r"(?:多少种|多少个|几种)[^。！？]{0,100}(?:表示|写)(?:成|为)[^。！？]{0,80}(?:之和|的和)", 8),
    _SubjectSignal("离散数学", "recurrence", r"递推(?:关系|数列)|斐波那契数|卢卡斯数|"
                   r"\b(?:recurrence relations?|recursive sequences?|fibonacci numbers?|lucas numbers?)\b", 5),
    _SubjectSignal("数学分析", "calculus", r"极限|积分|导数|可导|连续|极值|\b(?:limit|derivative|differentiate|integral|continuous|convergence)\b", 5),
    _SubjectSignal("数学分析", "series", r"级数|幂级数|\b(?:infinite series|power series)\b", 5),
    _SubjectSignal("数学分析", "fourier", r"傅里叶变换|Fourier\s*(?:变换|transform(?:ation)?)", 7),
)


# Compatibility for callers that imported the old first-match table.
SUBJECT_RULES = tuple((signal.subject, signal.pattern) for signal in _SUBJECT_SIGNALS)


def classify_subjects(problem: str) -> SubjectClassification:
    text = str(problem or "")
    stochastic_turn_taking = bool(
        re.search(r"轮流|回合|\b(?:take\s+turns|turn[- ]based|moves? first)\b", text, re.IGNORECASE)
        and re.search(
            r"概率|随机|期望|方差|掷|骰子|硬币|马尔可夫|"
            r"\b(?:probability|random(?:ly)?|expected|expectation|variance|fair\s+(?:die|dice|coin)|"
            r"roll(?:ing|s|ed)?|markov)\b",
            text,
            re.IGNORECASE,
        )
    )
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for signal in _SUBJECT_SIGNALS:
        if signal.name == "finite-game" and stochastic_turn_taking:
            continue
        if not re.search(signal.pattern, text, re.IGNORECASE | re.DOTALL):
            continue
        scores[signal.subject] = scores.get(signal.subject, 0) + signal.weight
        matches.setdefault(signal.subject, []).append(signal.name)

    if not scores:
        return SubjectClassification("进阶数学", "", "low", (), ())

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary, top_score = ranked[0]
    second_subject, second_score = ranked[1] if len(ranked) > 1 else ("", 0)
    secondary = second_subject if second_score >= 4 and second_score * 2 >= top_score else ""
    margin = top_score - second_score
    if top_score >= 7 and margin >= 3:
        confidence = "high"
    elif top_score >= 5 and margin >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    matched_signals = tuple(
        f"{subject}:{name}"
        for subject, _ in ranked
        for name in matches.get(subject, ())
    )
    return SubjectClassification(
        primary,
        secondary,
        confidence,
        matched_signals,
        tuple(ranked),
    )


def classify_subject(problem: str) -> Optional[str]:
    """Return the primary subject, preserving the original public API."""
    return classify_subjects(problem).primary
