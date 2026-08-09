from __future__ import annotations

import re
from typing import Optional


SUBJECT_RULES = (
    ("偏微分方程", r"热方程|Laplace方程|波动方程|偏微分|u_t|u_{xx}|\b(?:partial differential|heat equation|wave equation|laplace equation)\b"),
    ("线性回归", r"线性回归|非线性回归|逐步回归|异方差|普通最小二乘|\bOLS\b|决定系数|相关系数|回归系数|R\^2|\blinear regression\b"),
    ("统计推断", r"无偏估计|样本均值|估计量|置信区间|假设检验|正态分布|标准差|时间序列|时间数列|季节调整|数据分散|统计图形|\b(?:unbiased estimator|confidence interval|hypothesis test)\b"),
    ("随机过程", r"布朗运动|平稳过程|马尔可夫|协方差函数|随机过程|\b(?:brownian motion|markov chain|stochastic process)\b"),
    ("泛函分析", r"Banach|Hilbert|算子范数|有界线性|评价泛函|\b(?:operator norm|bounded linear operator)\b"),
    ("拓扑学", r"紧致|开覆盖|同胚|拓扑空间|\b(?:compactness|open cover|homeomorphism|topological space)\b"),
    ("运筹学", r"线性规划|单纯形|目标函数|可行域|对偶|\b(?:linear programming|simplex method|feasible region)\b"),
    (
        "抽象代数",
        r"群同态|正规子群|陪集|商群|群论|二面体群|循环群|交换群|阿贝尔群|Abel群|"
        r"环同态|多项式环|商环|理想|有限域|域扩张|"
        r"\b(?:group theory|dihedral group|cyclic group|abelian group|normal subgroup|"
        r"quotient group|quotient ring|polynomial ring|finite field|field extension|"
        r"ideal|homomorphism|coset)\b|"
        r"\bgroup\b.{0,80}\b(?:subgroup|normal|abelian|homomorphism|identity element)\b",
    ),
    ("初等几何", r"\b(?:triangle|quadrilateral|polygon|circle|cyclic|tangent|collinear|concurrent|circumcircle|incircle|orthocenter|incenter|angle bisector)\b"),
    ("微分几何", r"曲线|曲面|弧长参数|主曲率|高斯曲率|Hessian"),
    ("数值分析", r"牛顿法|二分法|迭代|插值|条件数|误差|\b(?:newton method|bisection|interpolation|condition number|numerical error)\b"),
    ("线性代数", r"\b(?:matrix|determinant|eigenvalue|eigenvector|linear transformation|vector space)\b"),
    ("高等代数", r"行列式|特征值|特征向量|矩阵|线性空间|\b(?:polynomial|functional equation|vieta)\b"),
    ("常微分方程", r"微分方程|通解|初值问题|相平面|平衡点|\b[xy]\s*'\s*=|d[xy]/d[xt]|\b(?:ordinary differential equation|initial value problem)\b"),
    ("复分析", r"留数|解析|复可导|幂级数|柯西|Laurent|\b(?:residue|holomorphic|analytic function|contour integral|laurent series)\b"),
    ("测度积分", r"勒贝格|可测|几乎处处|单调收敛|支配收敛|L\^1|\b(?:lebesgue|measurable|almost everywhere|dominated convergence)\b"),
    ("概率论", r"Bernoulli|概率|随机变量|分布|期望|方差|条件概率|\b(?:probability|random variable|expectation|variance|conditional probability)\b"),
    ("数论", r"\b(?:positive integers?|integer solutions?|diophantine|divisibility|divisible|congruence|modulo|prime numbers?|gcd|lcm)\b|同余|整除|素数|最大公约数"),
    ("离散数学", r"集合|图论|简单图|有向图|无向图|二分图|图中|图的|关系|命题|排列|组合|计数|递推|\b(?:graph|combinatorics?|permutation|combination|recurrence|counting|coloring|pigeonhole)\b"),
    ("数学分析", r"连续|可导|极限|定积分|积分|极值|\b(?:limit|derivative|integral|continuous|series|convergence)\b"),
)


def classify_subject(problem: str) -> Optional[str]:
    for subject, pattern in SUBJECT_RULES:
        if re.search(pattern, problem, re.IGNORECASE):
            return subject
    return "进阶数学"
