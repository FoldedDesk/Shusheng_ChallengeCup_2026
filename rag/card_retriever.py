"""Local theorem, method, and checklist cards for the public solve path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

from classifier.advanced_families import (
    DIRECTED_EULER_CIRCUIT_PATTERN,
    LACUNARY_NATURAL_BOUNDARY_PATTERN,
    PLANE_ROOTED_TREE_PATTERN,
    RUNGE_KUTTA_STABILITY_PATTERN,
    SPHERICAL_TRIANGLE_AREA_PATTERN,
    TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
    WEIERSTRASS_SINE_PRODUCT_PATTERN,
)

if TYPE_CHECKING:
    from classifier.problem_spec import ProblemSpec


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    kind: str
    domain: str
    text: str
    keywords: tuple[str, ...]
    topics: tuple[str, ...] = ()
    text_en: str = ""
    domains: tuple[str, ...] = ()

    @property
    def effective_domains(self) -> tuple[str, ...]:
        return self.domains or (self.domain,)

    def render(self, language: str) -> str:
        """Render in the contract language, falling back to the source note."""
        if language == "en" and self.text_en:
            return self.text_en
        return self.text

    def supports(self, language: str) -> bool:
        """Whether this card has prose suitable for the requested language."""
        if language == "en":
            return bool(self.text_en) or not re.search(r"[\u4e00-\u9fff]", self.text)
        return bool(re.search(r"[\u4e00-\u9fff]", self.text)) or not re.search(
            r"[A-Za-z]{2,}", self.text
        )


@dataclass(frozen=True)
class RetrievalBundle:
    solve_cards: tuple[KnowledgeCard, ...]
    review_cards: tuple[KnowledgeCard, ...]
    solve_scores: tuple[int, ...] = ()
    review_scores: tuple[int, ...] = ()
    language: str = "zh"
    primary_subject: str = ""
    secondary_subject: str = ""
    subject_confidence: str = "low"

    def solve_context(self) -> str:
        return _render(self.solve_cards, self.language)

    def review_context(self) -> str:
        return _render(self.review_cards, self.language)

    def verification_fact_context(self) -> str:
        """Return at most one high-confidence theorem fact for a fresh review."""
        if self.subject_confidence != "high":
            return ""
        for card, score in zip(self.solve_cards, self.solve_scores):
            if card.kind == "theorem" and score >= 14:
                return card.render(self.language)
        return ""

    def trace_content(self) -> dict:
        return {
            "solve_card_ids": [card.id for card in self.solve_cards],
            "review_card_ids": [card.id for card in self.review_cards],
            "solve_card_scores": list(self.solve_scores),
            "review_card_scores": list(self.review_scores),
            "language": self.language,
            "primary_subject": self.primary_subject,
            "secondary_subject": self.secondary_subject,
            "subject_confidence": self.subject_confidence,
        }


_DOMAIN_BY_NAME = {
    "回归": "线性回归",
    "抽象代数": "抽象代数", "高等代数": "高等代数", "数论": "数论",
    "数值分析": "数值分析", "数值线性": "数值分析", "线性代数": "线性代数",
    "离散": "离散数学", "图论": "离散数学", "组合": "离散数学", "概率": "概率论",
    "统计": "统计推断", "实分析": "数学分析", "微积分": "数学分析", "测度": "测度积分",
    "常微分": "常微分方程", "偏微分": "偏微分方程", "PDE": "偏微分方程", "复分析": "复分析", "拓扑": "拓扑学",
    "泛函": "泛函分析", "优化": "运筹学", "几何": "微分几何", "答案": "answer",
}

_COMPOSITE_DOMAINS = {
    "数论与代数": ("数论", "抽象代数"),
    "泛函分析与拓扑": ("泛函分析", "拓扑学"),
    "拓扑与泛函": ("拓扑学", "泛函分析"),
    "数据编码与数值PDE": ("离散数学", "数值分析", "偏微分方程"),
    "线性代数与优化": ("线性代数", "运筹学"),
    "概率统计": ("概率论", "统计推断"),
    "描述统计与时间序列": ("统计推断", "随机过程"),
}

_OPERATION_TERMS = (
    "牛顿法", "newton method", "newton's method",
    "二分法", "bisection method",
    "割线法", "secant method",
    "有限差分", "finite difference", "finite-difference",
    "中心差分", "central difference",
    "欧拉法", "euler method", "euler's method",
    "辛普森", "simpson", "梯形公式", "trapezoidal",
    "容斥", "inclusion-exclusion", "inclusion exclusion",
    "spanning tree", "spanning trees of", "complete bipartite", "complete tripartite", "deleting one edge",
    "surjective functions", "bracelet", "necklace", "lattice path", "monotone lattice path",
    "binary string", "hypercube", "no two", "rotation or a reflection", "contain neither",
    "pell equation", "positive divisors", "factorial quotient",
    "拉格朗日插值", "lagrange interpolation",
    "柯西积分", "cauchy integral", "留数", "residue",
    "高斯曲率", "gaussian curvature", "主曲率", "principal curvature",
    "胞腔同调", "cellular homology", "胞腔边界", "cellular boundary", "smith normal form",
    "切比雪夫", "chebyshev", "极小极大", "minimax polynomial", "等振荡", "equioscillation",
    "拉丁方", "latin square", "行列均为排列", "rows and columns are permutations",
    "无处零流", "nowhere-zero flow", "cycle space", "flow polynomial", "tutte polynomial",
    "欧拉回路", "欧拉闭迹", "euler circuit", "eulerian circuit", "euler tour", "best theorem",
    "平面根树", "有序根树", "plane rooted tree", "ordered rooted tree", "lukasiewicz", "cycle lemma",
    "自然边界", "natural boundary", "lacunary series", "fabry gap", "hadamard gap",
    "龙格库塔", "runge-kutta", "butcher tableau", "dirk", "sdirk", "stability function", "l-stability",
    "球面三角形", "球面余弦定理", "spherical triangle", "spherical excess", "girard theorem",
    "正弦乘积", "双曲正弦", "weierstrass sine product", "infinite product", "hyperbolic sine",
    "双调和", "多调和", "biharmonic", "polyharmonic", "fundamental solution", "flux normalization",
)


_SPECIALIZED_CARD_GATES = {
    "fact.number_theory.n_good_exotic": re.compile(
        r"\bn\W*good\s+functions?\b[\s\S]{0,1000}"
        r"\bg\s*\([^)]*\)\s*-\s*g\s*\([^)]*\)[\s\S]{0,700}"
        r"\bexotic\s+integers?\b|"
        r"n[- ]?好函数[\s\S]{0,1000}奇异整数",
        re.IGNORECASE,
    ),
    "fact.combinatorics.colored_cube_slices": re.compile(
        r"(?:单位立方体|小立方体)[\s\S]{0,700}(?:截面|薄片|三个方向)[\s\S]{0,500}"
        r"(?:颜色集合|不同颜色)|"
        r"\bunit\s+cubes?\b[\s\S]{0,700}\b(?:slices?|rectangular\s+prisms?)\b"
        r"[\s\S]{0,500}\b(?:sets?\s+of\s+(?:distinct\s+)?colou?rs?|orientations?)\b",
        re.IGNORECASE,
    ),
    "method.tiling.invariant_profile": re.compile(
        r"多米诺|多连方|铺砌|铺满|"
        r"\b(?:domino(?:es)?|tromino(?:es)?|tetromino(?:es)?|hexomino(?:es)?|"
        r"polyomino(?:es)?|til(?:e|es|ing))\b",
        re.IGNORECASE,
    ),
    "method.number_theory.vieta_jumping": re.compile(
        r"(?:方程|等式)[\s\S]{0,320}(?:全部|所有)正整数(?:有序)?解|"
        r"\b(?:determine|find|classify)\s+all\s+positive\s+integer"
        r"(?:\s+ordered)?\s+(?:pairs?|solutions?)\b",
        re.IGNORECASE,
    ),
    "method.finite_game.minimax": re.compile(
        r"轮流(?:选择|取|放|移动)|无法(?:行动|选择|移动)[^。！？]{0,40}(?:输|失败)|"
        r"\b(?:players?|contestants?)\s+take\s+turns\s+"
        r"(?:choos(?:e|ing)|select(?:ing)?|remov(?:e|ing)|plac(?:e|ing)|mov(?:e|ing))\b|"
        r"\b(?:no\s+legal\s+move|unable\s+to\s+move|winning\s+strategy|optimal\s+play|"
        r"initial\s+position\s+(?:is\s+)?losing)\b",
        re.IGNORECASE,
    ),
    "method.algebra.splitting_field": re.compile(
        r"分裂域|伽罗瓦|Galois扩张|\b(?:splitting\s+field|galois(?:\s+extension)?)\b",
        re.IGNORECASE,
    ),
    "method.analysis.fourier": re.compile(
        r"傅里叶|\bFourier(?:\s+(?:transform|transformation))?\b",
        re.IGNORECASE,
    ),
    "method.topology.cellular_homology": re.compile(
        r"CW\s*(?:复形|complex(?:es)?)|胞腔(?:同调|链复形|边界(?:映射|算子)?)|同调群|"
        r"附着映射|粘附映射|"
        r"\b(?:cellular\s+(?:homology|chain\s+complex|boundary(?:\s+map)?)|"
        r"homology\s+groups?|attaching\s+maps?)\b",
        re.IGNORECASE,
    ),
    "method.numerical.chebyshev_minimax": re.compile(
        r"切比雪夫|极小极大(?:多项式|逼近)?|最佳一致逼近|等振荡|交错定理|"
        r"\b(?:chebyshev|minimax\s+(?:polynomial|approximation)|best\s+uniform\s+approximation|"
        r"equioscillation|alternation\s+theorem)\b",
        re.IGNORECASE,
    ),
    "method.combinatorics.latin_square": re.compile(
        r"拉丁方|拉丁矩阵|行列(?:均|各)(?:为|是)?排列|"
        r"每个符号[^。！？\n]{0,50}每行(?:和|与|、)?每列[^。！？\n]{0,30}(?:恰好|正好)?出现一次|"
        r"\b(?:latin\s+squares?|rows?\s+and\s+columns?\s+(?:are|form)\s+permutations?|"
        r"each\s+row\s+and\s+(?:each\s+)?column\s+(?:is|forms?)\s+a\s+permutation|"
        r"row(?:\s*[/&-]\s*|\s+and\s+)column\s+permutations?|"
        r"each\s+symbol\s+(?:must\s+)?occurs?\s+exactly\s+once\s+in\s+every\s+row\s+"
        r"and\s+(?:in\s+)?every\s+column|"
        r"every\s+row\s+and\s+(?:every\s+)?column\s+contains?\s+each\s+symbol\s+exactly\s+once)\b",
        re.IGNORECASE,
    ),
    "method.graph.nowhere_zero_flow": re.compile(
        r"无处零流|处处非零流|图流多项式|循环空间|圈空间|Tutte\s*多项式|"
        r"\b(?:nowhere[- ]zero\s+(?:graph\s+)?flows?|flow\s+polynomial|cycle\s+space|"
        r"tutte\s+polynomial)\b",
        re.IGNORECASE,
    ),
    "method.graph.directed_euler_circuits": re.compile(
        DIRECTED_EULER_CIRCUIT_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.combinatorics.plane_rooted_trees": re.compile(
        PLANE_ROOTED_TREE_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.complex.lacunary_natural_boundary": re.compile(
        LACUNARY_NATURAL_BOUNDARY_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.numerical.runge_kutta_stability": re.compile(
        RUNGE_KUTTA_STABILITY_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.geometry.spherical_triangle_area": re.compile(
        SPHERICAL_TRIANGLE_AREA_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.complex.weierstrass_sine_product": re.compile(
        WEIERSTRASS_SINE_PRODUCT_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
    "method.pde.polyharmonic_fundamental_solution": re.compile(
        TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
        re.IGNORECASE | re.DOTALL,
    ),
}

_STOCHASTIC_GAME_MARKERS = re.compile(
    r"概率|随机|期望|方差|掷|骰子|硬币|马尔可夫|"
    r"\b(?:probability|random(?:ly)?|expected|expectation|variance|fair\s+(?:die|dice|coin)|"
    r"roll(?:ing|s|ed)?|markov)\b",
    re.IGNORECASE,
)

_FINITE_GAME_LEGALITY_MARKERS = re.compile(
    r"合法(?:着法|移动|选择)|每(?:一)?步(?:必须|可以|可)|玩家(?:必须|可以|可)|"
    r"\b(?:legal\s+moves?|on\s+each\s+move|a\s+player\s+(?:may|must|can)|"
    r"players?\s+(?:may|must|can))\b",
    re.IGNORECASE,
)

_FINITE_GAME_TERMINAL_MARKERS = re.compile(
    r"获胜|胜者|输|失败|无法(?:行动|选择|移动)|游戏结束|终局|"
    r"\b(?:wins?|loses?|winner|no\s+legal\s+move|unable\s+to\s+move|"
    r"game\s+ends?|terminal\s+(?:state|position|payoff))\b",
    re.IGNORECASE,
)


_METHOD_CARDS = (
    KnowledgeCard(
        "fact.number_theory.n_good_exotic", "theorem", "数论",
        "对满足 g(1)=1 且任意不同整数 a,b 都有 g(a)-g(b) 整除 a^n-b^n 的 n-好函数，完整分类的奇偶性结论是：恰好当 n 为奇数的平方时，n-好函数总数是二倍奇数。因此按递增顺序第 k 个奇异整数为 (2k-1)^2。应用时必须核对定义域为全体整数、量词覆盖所有不同 a,b，并核对题目采用通常的非零除数约定。",
        ("good", "function", "functions", "exotic", "integer", "integers", "divides", "奇异整数", "好函数"),
        ("olympiad_number_theory",),
        "For n-good maps g: Z to Z with g(1)=1 and g(a)-g(b) dividing a^n-b^n for every distinct a,b, the parity consequence of the full classification is: the number of n-good maps is twice an odd integer exactly when n is an odd square. Hence the kth exotic integer in increasing order is (2k-1)^2. Before applying this, check the full-integer domain, the universal distinct-pair quantifier, and the usual nonzero-divisor convention.",
    ),
    KnowledgeCard(
        "fact.combinatorics.colored_cube_slices", "theorem", "离散数学",
        "在 n×n×n 单元阵列中，若每个坐标方向的每个薄片颜色集合都必须在另外两个方向各出现一次，则把每种颜色编码为它出现的三组薄片指标。按首次出现层排序并作嵌套化，可得颜色数上界为 1^2+2^2+...+n^2；用第 k 层新增 k^2 种颜色的嵌套构造达到该界。使用前必须逐项核对题目确实要求三个方向的薄片集合相互复现，而不只是颜色数相等。",
        ("unit", "cubes", "coloured", "colored", "painted", "slices", "prisms", "orientations", "colors", "colours", "截面", "薄片", "颜色集合"),
        text_en="For an n by n by n unit array in which every slice color-set in each coordinate direction must reappear in both other directions, encode each color by the three families of slice indices where it occurs. Sorting first appearances and nesting supports gives the upper bound 1^2+2^2+...+n^2; a nested construction adding k^2 new colors at layer k attains it. Apply this only when the statement requires the slice sets themselves to reappear in all three directions, not merely equal color counts.",
    ),
    KnowledgeCard(
        "method.tiling.invariant_profile", "method", "离散数学",
        "多连方铺砌极值题先分别做下界与构造：面积整除只是一项必要条件，必须再检查棋盘染色、带权和或边界割线不变量。对固定宽度可用边界轮廓状态DP验证小规模并猜测周期；最后给出达到下界的可重复铺法，不能由总面积可整除直接断言无需小块。",
        ("domino", "dominoes", "tromino", "tetromino", "tetrominoes", "hexomino", "polyomino", "tile", "tiling", "铺砌", "多米诺", "多连方"),
        text_en="For an extremal polyomino tiling, prove the lower bound and construction separately. Area divisibility is only one necessary condition: also test checkerboard or weighted colorings and cut-line invariants. For fixed width, use a boundary-profile DP on small cases to identify a period, then give a repeatable tiling attaining the bound; never infer that zero small pieces suffice from area alone.",
    ),
    KnowledgeCard(
        "method.proof.direct", "method", "proof",
        "证明时先写清题设与目标；引用定理前核对前提，再给出从条件到结论的关键推导。",
        ("证明", "prove", "show"),
        text_en="State the hypotheses and target first. Check every theorem's assumptions, then give the key deductions from the hypotheses to the conclusion.",
    ),
    KnowledgeCard(
        "method.count.inclusion", "method", "离散数学",
        "计数题先明确对象是否有序、是否允许重复；容斥法必须定义违例事件并检查交集层数。",
        ("计数", "组合", "排列", "count"),
        text_en="For counting, determine whether order matters and repetition is allowed. For inclusion-exclusion, define the bad events and check every required intersection level.",
    ),
    KnowledgeCard(
        "method.algebra.structure", "method", "抽象代数",
        "代数结构题先展开定义，再检查运算、子结构、同态或商结构的必要条件。",
        ("群", "环", "域", "同态", "ideal"),
        text_en="Expand the definitions first, then check every required condition on the operation, substructure, homomorphism, or quotient.",
    ),
    KnowledgeCard(
        "method.analysis.conditions", "method", "数学分析",
        "分析题使用定理前检查连续、可导、可积、收敛或支配等前提，并单独检查端点。",
        ("极限", "积分", "级数", "limit", "integral"),
        text_en="Before applying an analysis theorem, verify continuity, differentiability, integrability, convergence, or domination assumptions, and check endpoints separately.",
    ),
    KnowledgeCard(
        "method.olympiad.geometry", "method", "初等几何",
        "几何题先记录共线、共圆、平行和切线等结构；优先检查相似、圆周角、点幂或面积比，并验证退化情形。",
        ("triangle", "circle", "cyclic", "tangent", "collinear", "angle", "geometry"),
        ("olympiad_geometry",),
        "Record collinear, cyclic, parallel, and tangent structures first. Test similarity, inscribed angles, power of a point, or area ratios, and verify degenerate cases.",
    ),
    KnowledgeCard(
        "method.olympiad.number_theory", "method", "数论",
        "整数题先固定整除与互素条件；结合模运算、因式分解、估值或无限递降，并证明所得解已穷尽。",
        ("integer", "divisible", "congruence", "modulo", "prime", "gcd", "diophantine"),
        ("olympiad_number_theory",),
        "Fix the divisibility and coprimality conditions first. Use congruences, factorization, valuations, or descent, and prove that the resulting solutions are exhaustive.",
    ),
    KnowledgeCard(
        "method.number_theory.vieta_jumping", "method", "数论",
        "二元二次丢番图方程要求全部正整数解时，可把方程视为关于较大变量的一元二次式。由韦达定理写出另一整数根，证明它仍为正且严格更小，从而下降到有限个最小解；再反向迭代得到递推族，并分别证明递推保持原方程、覆盖全部解以及交换变量后的对称分支。",
        ("vieta", "jumping", "descent", "diophantine", "integer", "pairs", "韦达", "下降", "正整数", "方程"),
        text_en="For a binary quadratic Diophantine equation asking for all positive-integer pairs, view it as a quadratic in the larger variable. Vieta's formulas give the other integral root; prove that it remains positive and is strictly smaller, reducing every solution to finitely many minimal seeds. Reverse the descent to obtain the recurrence, then verify invariance, exhaustiveness, and the branch obtained by swapping the variables.",
    ),
    KnowledgeCard(
        "method.olympiad.combinatorics", "method", "离散数学",
        "组合题先明确对象、有序性和重复规则。固定宽度、局部转移或网格路径计数应优先定义有限状态与转移矩阵，并用小规模边界核验；其余再考虑双计数、鸽巢、不变量、极端原理或容斥。",
        ("coloring", "arrangement", "pigeonhole", "counting", "subset", "tournament"),
        ("olympiad_combinatorics",),
        "Specify the objects, ordering, and repetition rules. For fixed-width grids, local transitions, or path counts, define a finite state and transfer recurrence first and verify small boundary cases. Otherwise use double counting, pigeonhole, an invariant, an extremal argument, or inclusion-exclusion.",
    ),
    KnowledgeCard(
        "method.olympiad.functional_equation", "method", "高等代数",
        "函数方程先代入特殊值寻找常数项、零点和对称性，再检验单射、满射或加法性；最后逐一回代全部候选。",
        ("function", "functional", "substitution", "injective", "surjective"),
        ("olympiad_functional_equation",),
        "Substitute special values to find constants, zeros, and symmetries. Test injectivity, surjectivity, or additivity, then substitute every candidate back into the original equation.",
    ),
    KnowledgeCard(
        "method.olympiad.inequality", "method", "高等代数",
        "不等式先检查齐次性、定义域和等号条件，再选择 AM-GM、Cauchy、凸性或变量代换；极值结论必须验证可达。",
        ("inequality", "positive", "minimum", "maximum", "cauchy", "am-gm"),
        ("olympiad_inequality",),
        "Check homogeneity, domain, and equality conditions before choosing AM-GM, Cauchy, convexity, or a substitution. Verify that any claimed extremum is attained.",
    ),
    KnowledgeCard(
        "method.olympiad.polynomial", "method", "高等代数",
        "多项式题结合根与系数、整除性、重根和插值约束；列出候选后检查次数、首项系数及所有根。",
        ("polynomial", "roots", "vieta", "monic", "divisibility"),
        ("olympiad_polynomial",),
        "Use relations between roots and coefficients, divisibility, repeated roots, and interpolation constraints. Check the degree, leading coefficient, and every root for each candidate.",
    ),
    KnowledgeCard(
        "method.olympiad.sequence", "method", "离散数学",
        "数列题先列初值并寻找单调性、不变量或周期；递推变形后用归纳法验证通项和全部边界。",
        ("sequence", "recurrence", "recursive", "induction", "invariant"),
        ("olympiad_sequence",),
        "List initial values and look for monotonicity, invariants, or periodicity. After transforming the recurrence, verify the closed form and all boundary cases by induction.",
    ),
    KnowledgeCard(
        "method.finite_game.minimax", "method", "离散数学",
        "有限完全信息博弈必须明确状态、当前玩家、合法着法和终局收益；先对最小规模做精确 minimax，再从状态模式提炼并分别证明双方策略。",
        ("game", "turn", "move", "player", "博弈", "轮流", "回合"),
        ("olympiad_combinatorics",),
        "For a finite perfect-information game, define the state, player to move, legal moves, and terminal payoff. Compute exact minimax on the smallest cases, then turn the observed pattern into separately proved strategies for both players.",
    ),
    KnowledgeCard(
        "method.algebra.splitting_field", "method", "抽象代数",
        "分裂域题先列出全部根，证明给定生成元包含这些根；再用不可约多项式逐层计算次数。分裂域在特征零下自动正规且可分，但不能把更大的含根域误当最小分裂域。",
        ("分裂域", "伽罗瓦", "splitting", "galois", "extension", "degree"),
        text_en="For a splitting field, list every root and prove that the proposed generators contain them. Compute each tower degree from minimal polynomials. In characteristic zero a splitting field is normal and separable, but a larger field containing the roots is not the minimal splitting field.",
    ),
    KnowledgeCard(
        "method.analysis.fourier", "method", "数学分析",
        "Fourier变换先声明归一化与指数符号约定，再利用平移、尺度和标准变换对；最后在零频率或逆变换处核对常数因子。",
        ("Fourier", "傅里叶", "transform", "变换"),
        text_en="State the Fourier-transform normalization and exponential-sign convention first. Use shift, scaling, and standard transform pairs, then check constants at zero frequency or by inversion.",
    ),
    KnowledgeCard(
        "method.geometry.spherical_triangle_area", "method", "微分几何",
        "适用门槛：三角形的边是球面大圆弧，目标是球面面积。先确认给出的边长是中心角还是物理弧长；半径为 R 时物理弧长须先除以 R。用球面余弦定理分别求三个内角，反三角函数取与几何范围一致的支；再用 Girard 定理，面积为 R^2 乘球面超额 A+B+C-pi。核验清单：所有角用弧度、边满足球面三角形可行条件、反余弦舍入值限制在 [-1,1]、单位球与一般半径的 R^2 因子，并用向量/Gram 矩阵或立体角公式独立检查。",
        ("球面", "三角形", "面积", "球面余弦", "Girard", "球面超额", "spherical", "triangle", "area", "cosine", "law", "excess", "radius", "solid", "angle"),
        text_en="Applicability gate: the sides are great-circle arcs and the target is spherical area. Decide first whether the supplied lengths are central angles or physical arc lengths; divide physical lengths by sphere radius R. Use the spherical law of cosines to recover all three interior angles with geometrically valid inverse-cosine branches. Girard then gives area R^2 times the spherical excess A+B+C-pi. Checklist: radians throughout, feasible spherical side data, clamp numerical cosine values to [-1,1], retain the R^2 factor away from the unit sphere, and independently check with vectors, a Gram matrix, or a solid-angle formula.",
    ),
    KnowledgeCard(
        "method.complex.weierstrass_sine_product", "method", "复分析",
        "适用门槛：题目要求由正弦的 Weierstrass 乘积推导或计算正弦/双曲正弦型无穷乘积。先写 sin(pi z)/(pi z) 的成对零点乘积，并检查零点、重数以及 z=0 处归一化；再作 z=i x 代换，使用 sin(iu)=i sinh(u)，逐项把减号变为加号。核验清单：pi 的尺度、是否漏掉前因子、乘积从哪个指标开始、虚数因子与符号、局部一致收敛，以及令 x 趋于 0 或比较对数导数的复核。",
        ("Weierstrass", "正弦", "双曲正弦", "无穷乘积", "零点", "归一化", "sine", "sinh", "hyperbolic", "infinite", "product", "zeros", "normalization", "imaginary", "substitution"),
        text_en="Applicability gate: derive or evaluate a sine/hyperbolic-sine infinite product from the Weierstrass product for sin(pi z)/(pi z). Check the paired zeros, multiplicities, and normalization at z=0 first. Then substitute z=i*x and use sin(iu)=i*sinh(u), which changes each quadratic minus sign to a plus sign. Checklist: the pi scaling, the prefactor, starting index, imaginary factors and signs, locally uniform convergence, and an independent check from the x->0 limit or the logarithmic derivative.",
    ),
    KnowledgeCard(
        "method.pde.polyharmonic_fundamental_solution", "method", "偏微分方程",
        "适用门槛：在二维全空间中求双调和或多调和算子的分布基本解。先固定算子是 Delta^m 还是 (-Delta)^m，并声明 Delta 与 Fourier 变换约定。对原点外的径向方程使用 Delta f(r)=f''(r)+f'(r)/r；二维共振会产生 r^(2m-2) log r 型候选。常数不能凭记忆填写：逐次施加 Laplace 算子降到二维调和基本解，再用小圆边界通量或测试函数积分确定 delta 的系数与符号。核验清单：原点外确为多调和、分布等式的符号、尺度量纲、允许添加的齐次多调和项，以及 Fourier 符号法的独立检查。",
        ("二维", "双调和", "多调和", "基本解", "分布", "通量", "biharmonic", "polyharmonic", "fundamental", "solution", "distribution", "radial", "laplacian", "flux", "fourier"),
        text_en="Applicability gate: find a distributional fundamental solution of a biharmonic or polyharmonic operator on the whole two-dimensional space. Fix whether the operator is Delta^m or (-Delta)^m and state the Laplacian and Fourier conventions. Off the origin use the radial identity Delta f=f''+f'/r; the two-dimensional resonance produces a candidate of the form r^(2m-2) log r. Do not recall the coefficient blindly: apply Laplacians successively until reaching the planar harmonic fundamental solution, then determine the delta coefficient and sign by small-circle flux or test-function integration. Checklist: polyharmonicity away from zero, distributional sign, scaling, allowable homogeneous additions, and an independent Fourier-symbol check.",
    ),
    KnowledgeCard(
        "method.graph.directed_euler_circuits", "method", "离散数学",
        "适用门槛：有限有向欧拉图满足每个顶点入度等于出度，且非零度顶点位于相应的强连通部分。先明确边是否有标签、回路是否按循环移位等价、起点或首弧是否固定。对带标签弧且固定根 r 发出的首弧，BEST 定理给出 t_r(G) 乘所有顶点 (d^+(v)-1)! 的乘积，其中 t_r 是以 r 为汇的有向生成树数；用有向矩阵树定理核验 t_r。不能在未建立双射时随意乘除边数或根的出度。核验清单：自环和平行弧约定、根与首弧、入出度平衡、生成树取向、局部阶乘，以及小图逐条枚举。",
        ("有向图", "弧", "欧拉回路", "欧拉闭迹", "BEST", "矩阵树", "directed", "digraph", "arc", "euler", "eulerian", "circuit", "tour", "fixed", "rotation", "arborescence"),
        text_en="Applicability gate: a finite directed Eulerian graph has equal indegree and outdegree at every vertex and the nonisolated part has the required strong connectivity. First decide whether arcs are labeled, whether tours are identified under cyclic rotation, and whether a start vertex or first arc is fixed. For labeled arcs with a fixed first arc leaving root r, BEST gives t_r(G) times the product of (outdeg(v)-1)! over all vertices, where t_r counts in-arborescences rooted at r; verify t_r by the directed matrix-tree theorem. Never multiply or divide by the number of arcs or the root outdegree without an explicit bijection. Checklist: loops and parallel arcs, root and first-arc convention, degree balance, arborescence orientation, local factorials, and exact enumeration on a small graph.",
    ),
    KnowledgeCard(
        "method.combinatorics.plane_rooted_trees", "method", "离散数学",
        "适用门槛：对象是子节点有左右次序的无标号平面根树，并给定各出度顶点数。设 n_i 为出度 i 的顶点数，先检查 sum n_i=n 与 sum i n_i=n-1；否则计数为零。按先序遍历写 Łukasiewicz 增量 i-1，合法树恰对应总和 -1 且所有真前缀和非负的词。循环引理从具有给定字母重数的多项式系数词中选出唯一合法循环移位，得到 (1/n) 乘多项式系数。核验清单：平面有序而非普通无序、是否带顶点标签、前缀不等式方向、周期词的循环移位，以及最小规模递推枚举。",
        ("平面根树", "有序根树", "出度", "度数分布", "Łukasiewicz", "Lukasiewicz", "循环引理", "plane", "ordered", "rooted", "tree", "outdegree", "profile", "cycle", "lemma", "prefix"),
        text_en="Applicability gate: the objects are unlabeled plane rooted trees, so children are ordered, and the number n_i of vertices of each outdegree i is prescribed. Check sum n_i=n and sum i*n_i=n-1; otherwise the count is zero. Preorder traversal gives Lukasiewicz increments i-1; a tree corresponds exactly to a word of total -1 whose proper prefix sums satisfy the required nonnegativity convention. The cycle lemma selects one valid cyclic shift among the multinomial words, yielding one over n times the multinomial coefficient. Checklist: plane ordered versus ordinary unordered trees, labeled versus unlabeled vertices, the prefix-sign convention, periodic words and cyclic shifts, and a recurrence enumeration at the smallest sizes.",
    ),
    KnowledgeCard(
        "method.complex.lacunary_natural_boundary", "method", "复分析",
        "适用门槛：目标是确定稀疏幂级数的收敛圆及其边界能否解析延拓。先由 Cauchy-Hadamard 公式或非零项子列求收敛半径；收敛圆周只是边界，不自动是自然边界。若指数满足 Hadamard 比值间隙或 Fabry 间隙条件，逐项核对系数与半径前提后应用相应定理；具有函数恒等式时，也可证明单位根等处奇异并利用其稠密性排除任何边界圆弧上的延拓。核验清单：指数严格递增、间隙定理的准确假设、边界奇点的稠密性，以及结论明确写出收敛域和整条自然边界。",
        ("稀疏", "幂级数", "自然边界", "收敛半径", "解析延拓", "Fabry", "Hadamard", "lacunary", "power", "series", "natural", "boundary", "radius", "analytic", "continuation", "gap"),
        text_en="Applicability gate: determine the convergence disk of a lacunary power series and whether analytic continuation crosses its boundary. First obtain the radius from Cauchy-Hadamard or the nonzero coefficient subsequence; a convergence circle is not automatically a natural boundary. If the exponents satisfy a Hadamard ratio-gap or Fabry gap condition, check its coefficient and radius hypotheses before invoking the theorem. When a functional identity is available, one may instead prove singularities at roots of unity and use their density to exclude continuation through every boundary arc. Checklist: strictly increasing exponents, the exact gap-theorem hypotheses, density of boundary singularities, and an explicit statement of both the convergence domain and the whole natural boundary.",
    ),
    KnowledgeCard(
        "method.numerical.runge_kutta_stability", "method", "数值分析",
        "适用门槛：题目给出 Runge-Kutta 或 DIRK/SDIRK 的 Butcher 系数并要求阶数、稳定函数、A 稳定或 L 稳定。先施加结构约束和阶条件（例如二阶需 b^T e=1、b^T c=1/2），再从测试方程 y'=lambda y 推导 R(z)=1+z b^T(I-zA)^{-1}e。A 稳定必须同时检查左半平面无极点且 |R(z)|<=1；对有理函数可用虚轴模与最大模原理。L 稳定还要求 A 稳定并核验 z 趋于无穷时 R(z) 趋于 0。若直接计算的极限或模条件冲突，不能用“经典方法”之类称谓覆盖冲突。",
        ("龙格库塔", "Butcher", "DIRK", "SDIRK", "稳定函数", "阶条件", "A稳定", "L稳定", "runge", "kutta", "tableau", "stability", "function", "order", "conditions", "infinity", "limit"),
        text_en="Applicability gate: a Runge-Kutta or DIRK/SDIRK tableau is given and the task asks for order, the stability function, A-stability, or L-stability. Apply the structural constraints and order conditions first (for order two, b^T e=1 and b^T c=1/2), then derive R(z)=1+z b^T(I-zA)^{-1}e from y'=lambda y. A-stability requires both no poles in the left half-plane and |R(z)|<=1 there; for a rational function, an imaginary-axis modulus check plus the maximum-modulus argument can certify this. L-stability additionally requires A-stability and R(z)->0 as z tends to infinity. If a computed limit or modulus contradicts a familiar method name, resolve the computation rather than appealing to the name.",
    ),
    KnowledgeCard(
        "method.topology.cellular_homology", "method", "拓扑学",
        "适用门槛：题目给出有限 CW 分胞及足以确定附着映射次数的数据。先写胞腔链复形和各边界矩阵，由附着映射的次数或生成元指数和计算边界，再用 Smith 标准形求核模像；可用基本群表示的阿贝尔化独立复核。核验清单：胞腔数与矩阵维数、取向符号、相邻边界复合为零、自由秩和每个挠系数。",
        ("CW", "胞腔", "同调", "边界", "附着", "smith", "cellular", "homology", "attaching", "abelianization"),
        text_en="Applicability gate: the problem gives a finite CW decomposition and enough attaching-map data to determine degrees. Write the cellular chain complex and boundary matrices, obtain entries from attaching degrees or exponent sums, and compute kernel modulo image with Smith normal form; independently check by abelianizing a fundamental-group presentation. Verification checklist: cell counts and matrix dimensions, orientation signs, consecutive boundaries composing to zero, free rank, and every torsion invariant.",
    ),
    KnowledgeCard(
        "method.numerical.chebyshev_minimax", "method", "数值分析",
        "适用门槛：目标是在区间上一致范数下优化具有固定首项系数或线性约束的多项式。先仿射映射到 [-1,1]，按首项系数归一化切比雪夫多项式，并用交错点给出不可改进的下界；约束更一般时解等振荡线性方程组。核验清单：区间缩放、次数与全部约束、归一化常数、至少 n+1 个交替极值点，以及候选的实际一致范数。",
        ("切比雪夫", "极小极大", "一致逼近", "等振荡", "chebyshev", "minimax", "uniform", "equioscillation", "alternation"),
        text_en="Applicability gate: optimize a polynomial with fixed leading coefficient or linear constraints in the uniform norm on an interval. Affinely map the interval to [-1,1], normalize a Chebyshev polynomial to match the leading coefficient, and use alternating extremal points for the sharp lower bound; for general constraints solve the equioscillation linear system. Verification checklist: interval scaling, degree and every constraint, normalization constant, at least n+1 alternating extrema, and the candidate's actual sup norm.",
    ),
    KnowledgeCard(
        "method.combinatorics.latin_square", "method", "离散数学",
        "适用门槛：数组的每行每列都是同一符号集的排列，即每个符号在每行每列恰好出现一次，且题目要求在对角线等附加限制下计数。先利用符号、行、列置换归一化首行或首列，明确稳定子与轨道倍率，再穷尽归一化结构并恢复标签。核验清单：允许的对称作用、是否重复计数、两条对角线及所有附加条件、归一化计数乘回轨道大小。",
        ("拉丁方", "拉丁矩阵", "行列", "排列", "对角线", "latin", "square", "rows", "columns", "permutations", "diagonals", "normalize", "symmetry", "structural", "orbit"),
        text_en="Applicability gate: every row and column is a permutation of one symbol set, equivalently each symbol occurs exactly once in every row and column, and the task counts arrays under extra conditions such as diagonals. Normalize a first row or column using symbol, row, and column permutations, record stabilizers and orbit factors, exhaust the normalized structures, then restore labels. Verification checklist: allowed symmetry actions, duplicate counting, both diagonals and every extra constraint, and multiplication of the normalized count by the correct orbit size.",
    ),
    KnowledgeCard(
        "method.graph.nowhere_zero_flow", "method", "离散数学",
        "适用门槛：有限图上的有限域流满足每个顶点的守恒律，并要求每条边取非零值。任取取向和生成树，以圈空间坐标表示所有流，把每条边为零视为线性超平面并作容斥；也可用 Tutte 流多项式或逐边枚举复核。核验清单：桥存在时无无处零流、圈空间维数 m-n+c、有限域阶、取向不变性，以及非零条件必须施加到每一条边而不只是圈基坐标。",
        ("无处零流", "图流", "循环空间", "圈空间", "有限域", "nowhere", "zero", "flow", "cycle", "space", "tutte", "finite", "field"),
        text_en="Applicability gate: a finite-graph flow over a finite field obeys conservation at every vertex and must be nonzero on every edge. Orient the graph, choose a spanning forest, express flows in cycle-space coordinates, and use inclusion-exclusion on the linear hyperplanes where an edge value is zero; cross-check with the Tutte flow polynomial or exact edge enumeration. Verification checklist: a bridge forces zero solutions, cycle-space dimension m-n+c, field order, orientation invariance, and edgewise nonzero constraints rather than merely nonzero cycle coordinates.",
    ),
    KnowledgeCard(
        "fact.lz78.encoding", "theorem", "离散数学",
        "LZ78 从只含空串的字典出发；每次输出（最长已有前缀的索引，下一个新字符），再把“前缀+新字符”加入字典。编码串必须同时编码索引和新字符，不能只串联新字符的代码；应明确索引起点与字段宽度。",
        ("lempel", "ziv", "lz78", "phrase", "phrases", "decomposition", "encode", "encoded", "string", "dictionary"),
        text_en=(
            "LZ78 starts with a dictionary containing only the empty string. At each step, output "
            "(the index of the longest dictionary prefix, the next new character), then insert prefix+character. "
            "The encoded bit string must encode both the index and the new character; concatenating only the "
            "new-character codes is incomplete. State the index origin and fixed field width used by the exercise."
        ),
    ),
    KnowledgeCard(
        "check.all_goals", "check", "answer",
        "最终答案必须覆盖题目的全部所求对象；多问题按题目顺序分别作答，不能只给中间量。",
        ("所有", "分别", "证明", "求"),
        text_en="The final answer must cover every requested object. Answer multipart questions in order, and do not substitute an intermediate quantity for the requested result.",
    ),
    KnowledgeCard(
        "check.roots", "check", "answer",
        "方程题必须列出全部根、检查定义域和伪根；离散根不能写成区间。",
        ("方程", "根", "solve", "equation"),
        text_en="List all roots, enforce the domain, and reject extraneous solutions. Do not express a discrete root set as an interval.",
    ),
    KnowledgeCard(
        "check.proof", "check", "proof",
        "证明必须具备关键条件、依据、推导链和结论；仅写定理名称或结论不构成证明。",
        ("证明", "prove", "show"),
        text_en="A proof needs the key hypotheses, justification, deduction chain, and conclusion. A theorem name or bare conclusion alone is not a proof.",
    ),
    KnowledgeCard(
        "check.counterexample", "check", "proof",
        "审查全称命题或逆命题时，尝试检查边界情形、反例以及量词方向。",
        ("是否", "当且仅当", "every", "if and only if"),
        text_en="For universal or converse claims, test boundary cases, possible counterexamples, and the direction of every quantifier.",
    ),
    KnowledgeCard(
        "check.olympiad.exhaustiveness", "check", "answer",
        "竞赛题复核时必须检查全部分支、整数或几何边界、等号情形，并确认没有把必要条件误当充分条件。",
        ("all", "integer", "boundary", "equality", "所有", "穷尽"),
        (
            "olympiad_geometry", "olympiad_number_theory", "olympiad_combinatorics",
            "olympiad_functional_equation", "olympiad_inequality", "olympiad_polynomial",
            "olympiad_sequence", "olympiad_general",
        ),
        "Independently check every branch, integer or geometric boundary, and equality case. Confirm that no necessary condition was used as though it were sufficient.",
    ),
)


class CardRetriever:
    """A dependency-free diversified retriever over bundled knowledge notes."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or Path("rag") / "knowledge"
        self.cards = [*_METHOD_CARDS, *self._load_cards()]

    def retrieve(self, spec: "ProblemSpec") -> RetrievalBundle:
        contract = getattr(spec, "answer_contract", None)
        raw_language = getattr(contract, "language", None) or getattr(spec.profile, "language", "zh")
        language = "en" if raw_language == "en" else "zh"
        primary_subject = getattr(spec.profile, "primary_subject", spec.profile.subject)
        secondary_subject = getattr(spec.profile, "secondary_subject", "")
        subject_confidence = getattr(spec.profile, "subject_confidence", "low")
        scored = self._score(spec, language)
        solve = self._top_confident(scored, include_kinds={"theorem", "method"}, min_score=9)
        solve_ids = {card.id for _, card in solve}
        review_scored = sorted(
            ((score + (3 if card.kind == "check" else 0), card) for score, card in scored),
            key=lambda item: (item[0], item[1].id),
            reverse=True,
        )
        review = self._top_confident(
            review_scored,
            include_kinds={"check", "method"},
            min_score=9,
            excluded_ids=solve_ids,
        )
        return RetrievalBundle(
            tuple(card for _, card in solve),
            tuple(card for _, card in review),
            tuple(score for score, _ in solve),
            tuple(score for score, _ in review),
            language,
            primary_subject,
            secondary_subject,
            subject_confidence,
        )

    def _load_cards(self) -> list[KnowledgeCard]:
        if not self.knowledge_dir.is_dir():
            return []
        cards = []
        for path in sorted(self.knowledge_dir.glob("*.txt")):
            domains = self._domains(path.stem)
            domain = domains[0]
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                text = line.strip()
                if not text or text.startswith("主题："):
                    continue
                cards.append(KnowledgeCard(
                    id=f"note.{path.stem}.{index}",
                    kind="theorem",
                    domain=domain,
                    text=text[:420],
                    keywords=tuple(_tokens(text)),
                    domains=domains,
                ))
        return cards

    @staticmethod
    def _domain(name: str) -> str:
        return CardRetriever._domains(name)[0]

    @staticmethod
    def _domains(name: str) -> tuple[str, ...]:
        for marker, domains in _COMPOSITE_DOMAINS.items():
            if marker in name:
                return domains
        found = []
        for marker, domain in _DOMAIN_BY_NAME.items():
            if marker in name and domain not in found:
                found.append(domain)
        return tuple(found) or ("进阶数学",)

    def _score(self, spec: "ProblemSpec", language: str = "zh") -> list[tuple[int, KnowledgeCard]]:
        topic = getattr(spec.profile, "topic", "general")
        primary_subject = getattr(spec.profile, "primary_subject", spec.profile.subject)
        secondary_subject = getattr(spec.profile, "secondary_subject", "")
        subject_confidence = getattr(spec.profile, "subject_confidence", "low")
        contract = getattr(spec, "answer_contract", None)
        support = tuple(getattr(contract, "explicit_support_requirements", ()))
        query = set(_tokens(" ".join([
            getattr(spec, "problem_text", ""),
            primary_subject, secondary_subject, spec.profile.problem_type, topic, spec.primary_method,
            spec.alternative_method, *spec.constraints, *spec.risk_flags,
            *support,
            *(goal.instruction for goal in spec.goals),
        ])))
        domain_tokens = set(_tokens(" ".join((primary_subject, secondary_subject))))
        proof_goal = any(goal.kind == "proof" for goal in spec.goals)
        problem_text = " ".join((
            getattr(spec, "problem_text", ""),
            *(goal.instruction for goal in spec.goals),
        )).lower()
        problem_tokens = set(_tokens(problem_text))
        scored = []
        for card in self.cards:
            if not card.supports(language):
                continue
            if card.id.startswith("note.") and not problem_tokens.intersection(
                card.keywords
            ):
                # Loaded notes need a direct semantic anchor in the public
                # statement. Subject or inferred-method bonuses alone must
                # never inject an unrelated theorem.
                continue
            specialized_gate = _SPECIALIZED_CARD_GATES.get(card.id)
            if specialized_gate is not None and not specialized_gate.search(problem_text):
                continue
            if card.id == "method.finite_game.minimax" and _STOCHASTIC_GAME_MARKERS.search(problem_text):
                continue
            if card.id == "method.finite_game.minimax" and not (
                _FINITE_GAME_LEGALITY_MARKERS.search(problem_text)
                and _FINITE_GAME_TERMINAL_MARKERS.search(problem_text)
            ):
                continue
            if card.id == "fact.lz78.encoding" and not re.search(
                r"\bLZ[- ]?\s*78\b", problem_text, re.IGNORECASE
            ):
                continue
            if card.topics and topic not in card.topics:
                continue
            card_domains = set(card.effective_domains)
            # Low-confidence classification never injects a card solely from
            # a guessed domain. Such cards must earn their score from concrete
            # operation/topic tokens in the public problem.
            primary_bonus = {"high": 12, "medium": 9, "low": 0}.get(subject_confidence, 0)
            secondary_bonus = {"high": 6, "medium": 5, "low": 0}.get(subject_confidence, 0)
            score = primary_bonus if primary_subject in card_domains else 0
            if secondary_subject and secondary_subject in card_domains:
                score += secondary_bonus
            if card.domain == "proof" and proof_goal:
                score += 6
            if card.domain == "answer":
                score += 2
            if topic != "general" and topic in card.topics:
                score += 7
            # Domain membership already has its own score.  Counting every
            # overlapping n-gram of the domain name made generic cards outrank
            # a card matching the actual requested operation.
            score += 2 * len(query.intersection(card.keywords) - domain_tokens)
            card_text = f"{card.text} {card.text_en}".lower()
            score += 5 * sum(
                1 for term in _OPERATION_TERMS
                if term in problem_text and term in card_text
            )
            if card.id == "fact.lz78.encoding" and re.search(
                r"\bLZ[- ]?\s*78\b", problem_text, re.IGNORECASE
            ):
                score += 3
            if card.kind == "method" and (
                (
                    subject_confidence != "low"
                    and bool(card_domains.intersection({primary_subject, secondary_subject}))
                )
                or (card.domain == "proof" and proof_goal)
            ):
                score += 3
            if card.kind == "check" and any(flag in {"missing_roots", "theorem_scope", "double_counting", "multiple_goals"} for flag in spec.risk_flags):
                score += 2
            if score:
                scored.append((score, card))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)

    @staticmethod
    def _top_confident(
        scored: list[tuple[int, KnowledgeCard]],
        include_kinds: set[str],
        min_score: int,
        excluded_ids: set[str] | None = None,
    ) -> list[tuple[int, KnowledgeCard]]:
        excluded = excluded_ids or set()
        for score, card in scored:
            if score < min_score:
                break
            if card.kind in include_kinds and card.id not in excluded:
                return [(score, card)]
        return []

    @staticmethod
    def _diverse(
        scored: list[tuple[int, KnowledgeCard]],
        limit: int,
        include_kinds: set[str],
        min_score: int,
    ) -> list[tuple[int, KnowledgeCard]]:
        selected: list[tuple[int, KnowledgeCard]] = []
        domains = set()
        kinds = set()
        for score, card in scored:
            if score < min_score:
                continue
            if card.kind not in include_kinds:
                continue
            if card.id in {item.id for _, item in selected}:
                continue
            if card.domain in domains and card.kind in kinds and len(selected) >= 2:
                continue
            selected.append((score, card))
            domains.add(card.domain)
            kinds.add(card.kind)
            if len(selected) == limit:
                break
        return selected


_ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "from", "for", "with", "without",
    "all", "is", "are", "be", "in", "on", "by", "as", "at", "when", "where", "that",
    "this", "then", "than", "each", "every", "one", "only", "has", "have", "having",
    "find", "determine", "evaluate", "number", "value", "possible", "using", "under",
    "into", "after", "before", "given", "such", "which", "whose", "does", "not",
    "it", "its", "we", "our", "you", "your", "they", "their", "these", "those",
    "can", "could", "may", "might", "must", "shall", "should", "would", "will",
    "what", "who", "why", "how", "also", "called", "call", "requires", "required",
    "used", "use", "two", "three", "four",
}


def _tokens(text: str) -> list[str]:
    tokens = [
        token for token in re.findall(r"[A-Za-z]{2,}", text.lower())
        if token not in _ENGLISH_STOPWORDS
    ]
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(
            run[index:index + width]
            for width in (2, 3, 4)
            for index in range(max(0, len(run) - width + 1))
        )
    return tokens


def _render(cards: tuple[KnowledgeCard, ...], language: str = "zh") -> str:
    return "\n".join(f"- {card.render(language)}" for card in cards)
