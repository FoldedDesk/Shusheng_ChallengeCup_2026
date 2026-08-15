"""Weighted bilingual subject classification using mathematical terminology only."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class SubjectClassification:
    primary: str
    secondary: str
    confidence: str
    matched_signals: tuple[str, ...]
    scores: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _Signal:
    subject: str
    name: str
    pattern: str
    weight: int


# Signals describe fields and standard mathematical objects.  They deliberately
# exclude story nouns and instantiated problem statements.
_SIGNALS = (
    _Signal("离散数学", "graph", r"图论|顶点|边集|生成树|欧拉(?:路|回路)|哈密顿|二部图|平面图|\b(?:graph|vertices?|edges?|spanning tree|eulerian|hamiltonian|bipartite|planar graph)\b", 7),
    _Signal("离散数学", "combinatorics", r"排列|组合|计数|容斥|鸽巢|生成函数|递推计数|染色|\b(?:combinatorics?|permutations?|combinations?|counting|inclusion[- ]exclusion|pigeonhole|generating function|colorings?)\b", 6),
    _Signal("离散数学", "logic", r"命题逻辑|谓词逻辑|真值表|偏序|等价关系|\b(?:propositional logic|predicate logic|truth table|partial order|equivalence relation)\b", 6),
    _Signal("数值分析", "root-method", r"牛顿法|二分法|割线法|迭代法|\b(?:newton(?:'s)? method|bisection|secant method|fixed[- ]point iteration)\b", 8),
    _Signal("数值分析", "approximation", r"数值积分|求积公式|Gauss.?Legendre|高斯求积|插值|有限差分|截断误差|舍入误差|条件数|龙格.?库塔|\b(?:numerical integration|interpolation|finite difference|truncation error|roundoff|condition number|runge[- ]kutta|quadrature|gauss[- ]legendre)\b", 7),
    _Signal("测度积分", "measure", r"勒贝格|可测|测度空间|几乎处处|支配收敛|单调收敛|可积(?:控制|支配)函数|Fatou|\b(?:lebesgue|measurable|measure space|almost everywhere|dominated convergence|monotone convergence|integrable dominat|fatou)\b", 8),
    _Signal("微分几何", "differential-geometry", r"微分流形|黎曼|曲率|挠率|测地线|第一基本形式|第二基本形式|高斯曲率|主曲率|\b(?:differential manifold|riemannian|curvature|torsion|geodesic|fundamental form|gaussian curvature|principal curvature)\b", 8),
    _Signal("概率论", "probability", r"条件概率|随机变量|分布函数|期望|方差|独立事件|均匀(?:抽取|分布)|样本空间|大数定律|中心极限定理|\b(?:conditional probability|random variable|distribution function|expectation|variance|independent events?|uniform(?:ly)? (?:sample|distributed)|sample space|law of large numbers|central limit theorem)\b", 7),
    _Signal("抽象代数", "groups-rings", r"群同态|正规子群|商群|环同态|理想|商环|有限域|不可约多项式|域扩张|分裂域|伽罗瓦|\b(?:group homomorphism|normal subgroup|quotient group|ring homomorphism|ideal|quotient ring|finite field|irreducible polynomial|field extension|splitting field|galois)\b", 8),
    _Signal("随机过程", "stochastic-process", r"随机过程|随机游走|吸收(?:状态|边界)?|首达|击中概率|马尔可夫链|布朗运动|泊松过程|更新过程|平稳过程|鞅|\b(?:stochastic process|random walk|absorbing|hitting probability|first passage|markov chain|brownian motion|poisson process|renewal process|stationary process|martingale)\b", 8),
    _Signal("复分析", "complex-analysis", r"全纯|复可导|复变函数|解析延拓|留数|Laurent|柯西积分|共形映射|辐角原理|Rouch|\b(?:holomorphic|complex differentiab(?:le|ility)|complex analysis|complex function|analytic continuation|residue|laurent|cauchy integral|conformal map|argument principle|rouch)\b", 8),
    _Signal("常微分方程", "ode", r"常微分方程|初值问题|边值问题|相平面|稳定性|Wronskian|\b(?:ordinary differential equation|initial value problem|boundary value problem|phase plane|wronskian)\b", 8),
    _Signal("常微分方程", "ode-notation", r"(?<![A-Za-z])[xyu]\s*['′]{1,3}\s*=|d[xyu]\s*/\s*d[xt]", 5),
    _Signal("统计推断", "inference", r"估计量|充分统计量|Fisher信息|极大似然|置信区间|假设检验|Wald|似然比|\b(?:estimator|sufficient statistic|fisher information|maximum likelihood|confidence interval|hypothesis test|wald|likelihood ratio)\b", 8),
    _Signal("泛函分析", "functional", r"Banach|Hilbert|有界线性算子|算子范数|弱收敛|紧算子|Hahn.?Banach|开映射定理|\b(?:bounded linear operator|operator norm|weak convergence|compact operator|open mapping theorem)\b", 8),
    _Signal("线性回归", "regression", r"线性回归|非线性回归|最小二乘|回归系数|异方差|残差|\b(?:linear regression|nonlinear regression|least squares|regression coefficient|heteroscedastic|residuals?|OLS|GLS)\b", 8),
    _Signal("偏微分方程", "pde", r"偏微分方程|热方程|波动方程|Laplace方程|Poisson方程|调和函数|Poisson核|圆盘边值|基本解|弱解|\b(?:partial differential equation|heat equation|wave equation|laplace equation|poisson equation|harmonic function|poisson kernel|fundamental solution|weak solution|PDE)\b", 8),
    _Signal("高等代数", "linear-algebra", r"矩阵|行列式|特征值|特征向量|线性空间|线性变换|秩|Jordan|Smith标准形|\b(?:matrix|determinant|eigenvalue|eigenvector|vector space|linear transformation|rank|jordan|smith normal form)\b", 6),
    _Signal("高等代数", "polynomial", r"多项式|最小多项式|特征多项式|不可约多项式|\b(?:polynomial|minimal polynomial|characteristic polynomial|irreducible polynomial)\b", 5),
    _Signal("运筹学", "operations-research", r"线性规划|整数规划|对偶问题|单纯形|网络流|动态规划|KKT|\b(?:linear programming|integer programming|dual problem|simplex|network flow|dynamic programming|KKT)\b", 8),
    _Signal("数学分析", "analysis", r"一致收敛|逐点收敛|函数列|级数收敛|连续性|可微性|极限|中值定理|\b(?:uniform convergence|pointwise convergence|sequence of functions|series convergence|continuity|differentiability|limit|mean value theorem)\b", 5),
    _Signal("拓扑学", "topology", r"拓扑空间|开集|闭集|紧致|连通|同胚|基本群|同调群|CW复形|\b(?:topological space|open set|closed set|compactness|connectedness|homeomorphism|fundamental group|homology group|CW complex)\b", 8),
    _Signal("数论", "number-theory", r"整除|同余|素数|丢番图|最大公约数|二次剩余|p进赋值|\b(?:divisibility|congruence|prime|diophantine|greatest common divisor|quadratic residue|p[- ]adic valuation)\b", 7),
    _Signal("进阶数学", "proof-structure", r"证明|求证|构造反例|\b(?:prove|proof|construct a counterexample)\b", 2),
)


def classify_subjects(problem: str) -> SubjectClassification:
    text = str(problem or "")
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for signal in _SIGNALS:
        occurrences = len(re.findall(signal.pattern, text, re.IGNORECASE | re.DOTALL))
        if not occurrences:
            continue
        scores[signal.subject] = scores.get(signal.subject, 0) + signal.weight + min(2, occurrences - 1)
        matches.setdefault(signal.subject, []).append(signal.name)

    if not scores:
        return SubjectClassification("进阶数学", "", "low", (), ())

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary, top = ranked[0]
    second, second_score = ranked[1] if len(ranked) > 1 else ("", 0)
    secondary = second if second_score >= 5 and second_score * 2 >= top else ""
    margin = top - second_score
    confidence = "high" if top >= 8 and margin >= 3 else ("medium" if top >= 5 else "low")
    matched = tuple(
        f"{subject}:{name}"
        for subject, _ in ranked
        for name in matches.get(subject, ())
    )
    return SubjectClassification(primary, secondary, confidence, matched, tuple(ranked))


def classify_subject(problem: str) -> Optional[str]:
    return classify_subjects(problem).primary
