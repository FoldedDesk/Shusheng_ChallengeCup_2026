from __future__ import annotations

import re
from typing import Optional


SUBJECT_RULES = (
    ("抽象代数", r"群同态|正规子群|陪集|商群|单位元|逆元|环|域"),
    ("偏微分方程", r"热方程|Laplace方程|波动方程|偏微分|u_t|u_{xx}"),
    ("线性回归", r"线性回归|决定系数|相关系数|回归系数|R\^2"),
    ("统计推断", r"无偏估计|样本均值|估计量|置信区间|假设检验"),
    ("随机过程", r"布朗运动|平稳过程|马尔可夫|协方差函数|随机过程"),
    ("泛函分析", r"Banach|Hilbert|算子范数|有界线性|评价泛函"),
    ("拓扑学", r"紧致|开覆盖|同胚|拓扑空间"),
    ("运筹学", r"线性规划|单纯形|目标函数|可行域|对偶"),
    ("高等代数", r"行列式|特征值|特征向量|矩阵|线性空间"),
    ("数值分析", r"牛顿法|二分法|迭代|插值|条件数|误差"),
    ("微分几何", r"曲线|曲面|弧长参数|主曲率|高斯曲率"),
    ("常微分方程", r"微分方程|通解|初值问题|相平面|平衡点"),
    ("复分析", r"留数|解析|复可导|幂级数|柯西|Laurent"),
    ("测度积分", r"勒贝格|可测|几乎处处|单调收敛|支配收敛|L\^1"),
    ("概率论", r"Bernoulli|概率|随机变量|分布|期望|方差|条件概率"),
    ("离散数学", r"集合|图|关系|命题|排列|组合|计数|递推|二分图"),
    ("数学分析", r"连续|可导|极限|定积分|积分|极值"),
)


def classify_subject(problem: str) -> Optional[str]:
    for subject, pattern in SUBJECT_RULES:
        if re.search(pattern, problem, re.IGNORECASE):
            return subject
    return "进阶数学"
