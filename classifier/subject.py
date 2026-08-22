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
    _Signal(
        "离散数学",
        "latin-square",
        r"拉丁方|拉丁方阵|\bLatin squares?\b|"
        r"(?=.*\b(?:symbols?|entries)\b)(?=.*\bevery row\b)(?=.*\bevery column\b)"
        r"(?=.*\b(?:exactly once|permutations?)\b)",
        10,
    ),
    _Signal("离散数学", "graph", r"图论|有限图|简单图|图中[^。；;\n]{0,40}(?:顶点|边|度数|团|独立集)|顶点|边集|生成树|欧拉(?:路|回路|有向图)|有向图|强连通|哈密顿|二部图|平面图|\b(?:finite graph|simple graph|graph|digraph|vertices?|edges?|spanning tree|eulerian|strongly connected|hamiltonian|bipartite|planar graph)\b", 7),
    _Signal(
        "离散数学",
        "directed-euler-circuits",
        r"有向(?:多重)?图[^。；;\n]{0,100}欧拉(?:闭迹|回路)|"
        r"欧拉(?:闭迹|回路)[^。；;\n]{0,100}(?:有向|固定首弧)|"
        r"固定首弧|BEST\s*定理|"
        r"\b(?:directed\s+(?:multi)?graph[^.;\n]{0,220}Eulerian circuits?|"
        r"Eulerian circuits?[^.;\n]{0,220}(?:directed|fixed first arc|specified first arc)|"
        r"(?:fixed|specified) first arc|BEST theorem)\b",
        10,
    ),
    _Signal(
        "离散数学",
        "plane-rooted-trees",
        r"平面有根树|有序有根树|Lukasiewicz|Łukasiewicz|"
        r"(?:有根树|树)[^。；;\n]{0,80}出度(?:剖面|分布|序列)|"
        r"\b(?:ordered plane rooted trees?|plane rooted trees?|"
        r"Lukasiewicz words?|out[- ]degree profile)\b",
        10,
    ),
    _Signal("离散数学", "graph-flow", r"无处(?:为)?零流|流多项式|(?:圈|循环)空间|\b(?:nowhere[- ]zero flows?|flow polynomial|cycle space)\b", 10),
    _Signal("离散数学", "combinatorics", r"排列|置换|对换|不动点|组合|计数|容斥|鸽巢|生成函数|递推计数|染色|二进制串|(?:字符串|序列)(?:数|[^。；;\n]{0,40}(?:个数|计数|递推))|\b(?:combinatorics?|permutations?|involutions?|transpositions?|fixed points?|combinations?|counting|inclusion[- ]exclusion|pigeonhole|generating function|colorings?|binary strings?)\b", 6),
    _Signal(
        "离散数学",
        "finite-combinatorial-objects",
        r"圆桌|字母排列|项链|手镯|布尔变量|满足赋值|格路径|集合族|反链|标号树|无标号组|"
        r"\b(?:circular tables?|arrangements? of (?:the )?letters|"
        r"necklaces?|bracelets?|boolean variables?|satisfying assignments?|"
        r"monotone lattice paths?|set famil(?:y|ies)|antichains?|labeled trees?|"
        r"unlabeled groups?)\b",
        8,
    ),
    _Signal(
        "离散数学",
        "explicit-recurrence-sequence",
        r"(?:数列|序列)[^。；;\n]{0,100}(?:定义为|满足)[^。；;\n]{0,180}"
        r"[A-Za-z]_?\{?n\s*\+\s*\d+\}?\s*=|"
        r"\bsequence\b[^.;\n]{0,100}\bdefined by\b[^.;\n]{0,180}"
        r"[A-Za-z]_?\{?n\s*\+\s*\d+\}?\s*=",
        8,
    ),
    _Signal("离散数学", "finite-games", r"(?![^。；;\n]{0,180}(?:随机|概率|骰子|硬币|掷|抛|random|probability|dice?|coin|roll|toss))(?:博弈|游戏|玩家|两人|双方)[^。；;\n]{0,100}(?:轮流|先手|后手|必胜|策略|不能行动)|(?![^。；;\n]{0,180}(?:随机|概率|骰子|硬币|掷|抛|random|probability|dice?|coin|roll|toss))(?:轮流|先手|后手)[^。；;\n]{0,100}(?:取走|移动|操作|获胜|策略)|\b(?:finite game|impartial game|winning strategy)\b", 7),
    _Signal("离散数学", "logic", r"命题逻辑|谓词逻辑|真值表|偏序|等价关系|\b(?:propositional logic|predicate logic|truth table|partial order|equivalence relation)\b", 6),
    _Signal(
        "离散数学",
        "order-theory",
        r"偏序集|线性扩张|哈斯图|Hasse\s*图|"
        r"\b(?:posets?|partially ordered sets?|linear extensions?|Hasse diagrams?)\b",
        8,
    ),
    _Signal("数值分析", "root-method", r"牛顿法|二分法|割线法|迭代法|\b(?:newton(?:'s)? method|bisection|secant method|fixed[- ]point iteration)\b", 8),
    _Signal("数值分析", "approximation", r"数值积分|求积公式|复化(?:中点|梯形|辛普森)公式|Gauss.?Legendre|高斯求积|插值|有限差分|中心差分|截断误差|舍入误差|条件数|龙格.?库塔|Chebyshev|切比雪夫|极小极大|交错极值|上确界范数[^。；;\n]{0,40}最小|\b(?:numerical integration|composite[- ](?:midpoint|trapezoid|simpson)(?:'s)?\s+rule|interpolation|finite[- ]difference|central[- ]difference|truncation error|roundoff|condition number|runge[- ]kutta|quadrature|gauss[- ]legendre|chebyshev|minimax|equioscillation)\b", 7),
    _Signal(
        "数值分析",
        "runge-kutta-stability",
        r"SDIRK|Butcher\s*(?:表|数组)|稳定函数\s*R\s*\(\s*z\s*\)|"
        r"L\s*[-－— ]?稳定|Runge.?Kutta[^。；;\n]{0,100}(?:稳定函数|稳定域|阶条件)|"
        r"\b(?:SDIRK|Butcher tableau|stability function\s+R\s*\(\s*z\s*\)|"
        r"L[- ]stability|L[- ]stable|Runge[- ]Kutta[^.;\n]{0,100}"
        r"(?:stability function|stability region|order conditions?))\b",
        10,
    ),
    _Signal(
        "数值分析",
        "chebyshev-minimax",
        r"切比雪夫[^。；;\n]{0,100}(?:极小极大|交错|上确界)|"
        r"(?:极小极大|交错极值)[^。；;\n]{0,100}(?:多项式|首项系数)|"
        r"\b(?:Chebyshev[^.;\n]{0,100}(?:minimax|equioscillation)|"
        r"minimax polynomial|equioscillation theorem)\b",
        10,
    ),
    _Signal("数值分析", "multistep-iteration", r"线性多步法|多步法|向后差分公式|后向差分公式|Jacobi\s*法|雅可比迭代|\bJacobi\b(?=[^。；;\n]{0,40}(?:迭代|矩阵|谱半径))|Gauss.?Seidel|高斯.?赛德尔|超松弛|迭代矩阵|零稳定性|绝对稳定域|\b(?:BDF\s*\d*|backward differentiation formula|linear multistep method|Adams[- ](?:Bashforth|Moulton)|Jacobi iteration|Gauss[- ]Seidel|successive over[- ]relaxation|iteration matrix|zero[- ]stability|A[- ]stable|absolute stability region)\b", 9),
    _Signal("测度积分", "measure", r"勒贝格|可测|测度空间|几乎处处|支配收敛|单调收敛|可积(?:控制|支配)函数|交换积分(?:次序|顺序)?|Fubini|Tonelli|富比尼|托内利|Fatou|(?:属于|不属于)\s*L\s*\^?\s*\{?\s*[0-9pP]+\s*\}?|\b(?:lebesgue|measurable|measure space|almost everywhere|dominated convergence|monotone convergence|change the order of integration|iterated integrals?|integrable dominat|belongs?\s+to\s+L\s*\^?\s*\{?\s*[0-9pP]+\s*\}?|fubini|tonelli|fatou)\b", 8),
    _Signal("微分几何", "differential-geometry", r"微分流形|黎曼|曲率|挠率|测地线|第一基本形式|第二基本形式|高斯曲率|主曲率|\b(?:differential manifold|riemannian|curvature|torsion|geodesic|fundamental form|gaussian curvature|principal curvature)\b", 8),
    _Signal(
        "微分几何",
        "spherical-triangle-area",
        r"球面三角形|球面余弦定理|Girard\s*定理|球面盈量|"
        r"\b(?:spherical triangle|spherical law of cosines|Girard(?:'s)? theorem|"
        r"spherical excess)\b",
        10,
    ),
    _Signal("概率论", "probability", r"概率|条件概率|随机变量|分布函数|正态分布|标准正态|期望|方差|独立事件|独立等可能|均匀(?:抽取|分布)|次序统计量|随机间距|最大间距|样本空间|大数定律|中心极限定理|\b(?:probability|conditional probability|random variable|distribution function|normal distribution|standard normal|expectation|variance|independent events?|independent and equiprobable|uniform(?:ly)? (?:sample|distributed)|order statistics?|random spacings?|maximum spacing|sample space|law of large numbers|central limit theorem)\b", 7),
    _Signal("抽象代数", "groups-rings", r"群同态|正规子群|商群|环同态|理想|商环|有限域|不可约多项式|域扩张|分裂域|伽罗瓦|Galois|二面体群|交换子群|导出子群|群的中心|\b(?:group homomorphism|normal subgroup|quotient group|ring homomorphism|ideal|quotient ring|finite field|irreducible polynomial|field extension|splitting field|galois|dihedral group|commutator subgroup|derived subgroup|center of (?:a |the )?group)\b", 8),
    _Signal(
        "抽象代数",
        "finite-field-notation",
        r"\\mathbb\s*\{?F\}?\s*_\s*\{?\s*[A-Za-z0-9^]+\s*\}?|"
        r"\b(?:monic\s+)?irreducible polynomials?\b[^.;\n]{0,100}"
        r"\bover\s+(?:the\s+)?field\b",
        9,
    ),
    _Signal("随机过程", "stochastic-process", r"随机过程|随机游走|随机游动|出生死亡链|吸收(?:状态|边界)?|首达|击中概率|马尔可夫链|布朗运动|泊松过程|更新过程|平稳过程|鞅|\b(?:stochastic process|random walk|birth[- ]death (?:chain|process)|absorbing|hitting probability|first passage|markov chain|brownian motion|poisson process|renewal process|stationary process|martingale)\b", 9),
    _Signal(
        "复分析",
        "complex-analysis",
        r"全纯|复可导|复变函数|解析延拓|留数|Laurent|柯西积分|共形映射|辐角原理|Rouch|"
        r"\b(?:holomorphic|complex differentiab(?:le|ility)|complex analysis|"
        r"complex function|analytic continuation|laurent|cauchy integral|"
        r"conformal map|argument principle|rouch)\b|"
        r"\b(?:residue theorem|"
        r"residues?\s+(?:at|of)\b(?![^.;\n]{0,140}\bmodulo\b)|"
        r"(?:find|compute|calculate|determine)\s+(?:the\s+)?residues?\b"
        r"(?![^.;\n]{0,140}\bmodulo\b))",
        8,
    ),
    _Signal(
        "复分析",
        "lacunary-natural-boundary",
        r"稀疏幂级数|空隙幂级数|自然边界|Hadamard\s*空隙|Fabry\s*空隙|"
        r"\b(?:lacunary power series|Hadamard gap theorem|Fabry gap theorem|"
        r"natural boundary)\b|"
        r"(?=.*\bpower series\b)(?=.*\bdense boundary singularities\b)"
        r"(?=.*\banalytic continuation\b)(?=.*\bboundary arc\b)",
        10,
    ),
    _Signal(
        "复分析",
        "weierstrass-sine-product",
        r"(?:正弦|双曲正弦)[^。；;\n]{0,80}(?:Weierstrass|无穷乘积|乘积公式)|"
        r"Weierstrass[^。；;\n]{0,80}(?:正弦|双曲正弦|无穷乘积)|"
        r"\b(?:Weierstrass sine product|hyperbolic sine infinite product|"
        r"sine infinite product)\b",
        10,
    ),
    _Signal("常微分方程", "ode", r"常微分方程|初值问题|边值问题|相平面|稳定性|Wronskian|\b(?:ordinary differential equation|initial value problem|boundary value problem|phase plane|wronskian)\b", 8),
    _Signal("常微分方程", "ode-notation", r"(?<![A-Za-z])[xyu]\s*['′]{1,3}\s*=|d[xyu]\s*/\s*d[xt]", 5),
    _Signal("统计推断", "inference", r"估计量|充分统计量|完备充分|(?:统计量|分布族)[^。；;\n]{0,40}完备性|完备性[^。；;\n]{0,40}(?:统计量|分布族)|Rao.?Blackwell|Lehmann.?Scheff|UMVU|Fisher信息|极大似然|置信区间|假设检验|Wald|似然比|\b(?:estimator|sufficient statistic|completeness of (?:a |the )?(?:statistic|family)|complete sufficient|rao[- ]blackwell|lehmann[- ]scheff|UMVU|fisher information|maximum likelihood|confidence interval|hypothesis test|wald|likelihood ratio)\b", 8),
    _Signal("统计推断", "descriptive-statistics", r"描述统计|数据分散程度|标准差|统计图形|直方图|箱线图|时间序列|时间数列|长期趋势|季节变动|循环变动|不规则变动|季节调整|移动平均|时间序列分解|\b(?:descriptive statistics|data dispersion|standard deviation|histogram|box plot|time series|seasonal adjustment|moving average|time series decomposition)\b", 8),
    _Signal("泛函分析", "functional", r"Banach|Hilbert|有界线性算子|乘法算子|算子范数|点谱|本质值域|弱收敛|紧算子|完备度量空间|Cauchy数列|加权移位(?:算子)?|单边移位(?:算子)?|算子[^。；;\n]{0,40}(?:谱|谱半径)|Hahn.?Banach|开映射定理|\b(?:bounded linear operator|multiplication operator|operator norm|point spectrum|essential range|weak convergence|compact operator|complete metric space|cauchy sequence|weighted shift(?: operator)?|unilateral shift(?: operator)?|operator.{0,40}(?:spectrum|spectral radius)|open mapping theorem)\b", 8),
    _Signal("线性回归", "regression", r"线性回归|非线性回归|逐步回归|岭回归|套索回归|最小二乘|回归系数|异方差|残差|\b(?:linear regression|nonlinear regression|stepwise regression|ridge regression|lasso regression|least squares|regression coefficient|heteroscedastic|residuals?|OLS|GLS)\b", 8),
    _Signal("偏微分方程", "pde", r"偏微分方程|热方程|波动方程|Laplace方程|拉普拉斯方程|Poisson方程|调和函数|是否调和|调和延拓|Poisson\s*核|圆盘边值|圆盘[^。；;\n]{0,24}Dirichlet|Dirichlet[^。；;\n]{0,24}圆盘|基本解|弱解|\b(?:partial differential equation|heat equation|wave equation|laplace equation|poisson equation|harmonic function|whether[^.\n]{0,30}harmonic|harmonic extension|poisson kernel|disk.{0,24}dirichlet|dirichlet.{0,24}disk|fundamental solution|weak solution|PDE)\b", 8),
    _Signal("偏微分方程", "pde-derivative-notation", r"(?=.*?u\s*_?\{?t\}?\s*)(?=.*?u\s*_?\{?x\}?\s*)", 8),
    _Signal("高等代数", "linear-algebra", r"矩阵|行列式|特征值|特征向量|线性空间|线性变换|秩|Jordan|Smith标准形|\b(?:matrix|determinant|eigenvalue|eigenvector|vector space|linear transformation|rank|jordan|smith normal form)\b", 6),
    _Signal("高等代数", "polynomial", r"多项式|最小多项式|特征多项式|不可约多项式|\b(?:polynomial|minimal polynomial|characteristic polynomial|irreducible polynomial)\b", 5),
    _Signal(
        "高等代数",
        "symmetric-root-data",
        r"(?:复根|实根)[^。；;\n]{0,100}(?:计重数|含重数)|"
        r"\b(?:real|complex)?\s*roots?\b[^.;\n]{0,100}"
        r"\bcounted with multiplicity\b|"
        r"\b(?:Vieta(?:'s)? formulas?|resultants?|symmetric polynomial in the roots?)\b",
        8,
    ),
    _Signal("运筹学", "operations-research", r"线性规划|整数规划|对偶问题|单纯形|网络流|动态规划|KKT|\b(?:linear programming|integer programming|dual problem|simplex|network flow|dynamic programming|KKT)\b", 8),
    _Signal("运筹学", "network-optimization", r"最大(?:费用)?流|最小费用流|网络流|\b(?:maximum flow|max[- ]flow|min[- ]cost flow|network flow)\b", 10),
    _Signal("数学分析", "analysis", r"一致收敛|逐点收敛|函数列|级数收敛|连续性|可微性|极限|中值定理|\b(?:uniform convergence|pointwise convergence|sequence of functions|series convergence|continuity|differentiability|limit|mean value theorem)\b", 5),
    _Signal("拓扑学", "topology", r"拓扑空间|开集|闭集|紧致|连通空间|路径连通|局部连通|同胚|基本群|同调群|CW\s*复形|\b(?:topological space|open set|closed set|compactness|connectedness|path connected|locally connected|homeomorphism|fundamental group|homology group|CW complex)\b", 8),
    _Signal(
        "拓扑学",
        "cellular-homology",
        r"胞腔(?:链复形|边界(?:映射|矩阵)?|同调)|CW\s*复形[^。；;\n]{0,120}附着|"
        r"\b(?:cellular (?:chain complex|boundary(?: map| matrix)?|homology)|"
        r"CW complex[^.;\n]{0,140}attaching (?:map|word)s?)\b",
        10,
    ),
    _Signal("数论", "number-theory", r"整除|同余|素数|丢番图|最大公约数|二次剩余|p进赋值|\b(?:divisibility|divisible|divides?|congruence|prime|diophantine|greatest common divisor|quadratic residue|p[- ]adic valuation)\b", 7),
    _Signal(
        "数论",
        "modular-arithmetic",
        r"\\(?:pmod|bmod)\s*\{?|"
        r"\\equiv[^。；;\n]{0,100}\\(?:pmod|bmod)|"
        r"(?:最小非负(?:剩余|余数)|剩余类|同余类|模乘法群|模\s*\d+\s*的乘法群)|"
        r"(?:余数|剩余)[^。；;\n]{0,80}(?:模\s*\d+|除以\s*\d+)|"
        r"\b(?:least nonnegative residue|residue classes?)\b"
        r"[^.;\n]{0,120}\bmodulo\b|"
        r"\b(?:multiplicative group|group of units)\s+modulo\b|"
        r"\b(?:congruences?|congruent)\b[^.;\n]{0,120}\bmodulo\b",
        10,
    ),
    _Signal("离散数学", "discrete-structures", r"集合族|幂集|基数|布尔代数|递推关系|组合恒等式|匹配数|独立集|团数|\b(?:set family|power set|cardinality|boolean algebra|recurrence relation|combinatorial identity|matching number|independent set|clique number)\b", 7),
    _Signal(
        "离散数学",
        "function-counting",
        r"(?:映射|函数)[^。！？!?\n]{0,80}(?:个数|多少|数量)|"
        r"\b(?:number\s+of|how\s+many|count(?:ing)?)\b[^.!?\n]{0,80}"
        r"\b(?:functions?|maps?)\b",
        7,
    ),
    _Signal("离散数学", "information-coding", r"Lempel.?Ziv|LZ\d*|短语分解|无损编码|信息编码|\b(?:lempel[- ]ziv|phrase parsing|lossless coding|information coding)\b", 8),
    _Signal("概率论", "distribution-calculus", r"概率密度|联合密度|边缘密度|条件密度|矩母函数|特征函数|相关系数|\b(?:probability density|joint density|marginal density|conditional density|moment generating function|characteristic function|correlation coefficient)\b", 7),
    _Signal("统计推断", "sampling-statistics", r"样本均值|样本方差|参数估计|无偏估计|一致估计|充分性|完备统计量|\b(?:sample mean|sample variance|parameter estimation|unbiased estimat|consistent estimat|sufficiency|complete statistic)\b", 7),
    _Signal("抽象代数", "algebraic-objects", r"群作用|共轭类|循环群|交换群|群的阶|环的特征|素理想|极大理想|模同态|\b(?:group action|conjugacy class|cyclic group|abelian group|order of (?:the )?group|characteristic of (?:the )?ring|prime ideal|maximal ideal|module homomorphism)\b", 8),
    _Signal(
        "抽象代数",
        "finite-abelian-elements",
        r"有限阿贝尔群|有限交换群|群的直和|直和群|"
        r"(?:群|直和)[^。；;\n]{0,80}(?:元素的阶|阶恰为|阶等于|阶整除)|"
        r"\b(?:finite abelian group|finite commutative group|direct sum of (?:finite )?groups?)\b|"
        r"\b(?:abelian group|direct sum)[^.\n]{0,100}"
        r"(?:elements? of (?:exact )?order|order (?:exactly|dividing))\b",
        10,
    ),
    _Signal("泛函分析", "normed-spaces", r"赋范空间|内积空间|对偶空间|线性泛函|强收敛|弱星收敛|一致有界|闭图像|\b(?:normed space|inner product space|dual space|linear functional|strong convergence|weak[- ]star convergence|uniform boundedness|closed graph)\b", 8),
    _Signal("微分几何", "geometry-forms", r"切空间|余切空间|联络|Christoffel|外微分|微分形式|曲面面积|法曲率|\b(?:tangent space|cotangent space|connection|christoffel|exterior derivative|differential form|surface area|normal curvature)\b", 8),
    _Signal("复分析", "complex-local", r"复积分|围道|奇点|极点|本性奇点|解析函数|调和共轭|最大模原理|\b(?:complex integral|contour|singularit|poles?|essential singularity|analytic function|harmonic conjugate|maximum modulus)\b", 7),
    _Signal("常微分方程", "ode-structure", r"线性微分方程|自治系统|特征方程|积分因子|存在唯一性|\b(?:linear differential equation|autonomous system|characteristic equation|integrating factor|existence and uniqueness)\b", 7),
    _Signal("偏微分方程", "pde-operators", r"梯度|散度|旋度|拉普拉斯算子|弱导数|Sobolev空间|Dirichlet边界|Neumann边界|\b(?:gradient|divergence|curl|laplacian|weak derivative|sobolev space|dirichlet boundary|neumann boundary)\b", 7),
    _Signal("偏微分方程", "formal-adjoint", r"伴随算子|形式伴随|散度型算子|定义域为\s*C_0\^?\\?infty|\b(?:formal adjoint|adjoint operator|divergence[- ]form operator)\b", 8),
    _Signal("数学分析", "fourier-analysis", r"Fourier\s*变换|傅里叶变换|Fourier\s*transform|频域|\bfrequency domain\b", 8),
    _Signal("数学分析", "real-analysis", r"函数项级数|幂级数|绝对收敛|条件收敛|上极限|下极限|可积性|黎曼积分|反常积分|凸函数|\b(?:series of functions|power series|absolute convergence|conditional convergence|limsup|liminf|integrability|riemann integral|improper integral|convex function)\b", 7),
    _Signal("拓扑学", "topological-invariants", r"商空间|积拓扑|子空间拓扑|邻域|闭包|内部|边界点|分离公理|同伦|覆盖空间|\b(?:quotient space|product topology|subspace topology|neighbou?rhood|closure|interior|boundary point|separation axiom|homotopy|covering space)\b", 8),
    _Signal("欧氏几何", "euclidean-geometry", r"三角形|四边形|圆周角|外接圆|内切圆|垂心|内心|旁心|中线|角平分线|相似三角形|\b(?:triangle|quadrilateral|cyclic quadrilateral|circumcircle|incircle|orthocenter|incenter|excenter|median|angle bisector|similar triangles)\b", 7),
    _Signal("数论", "number-theory-structure", r"质数|素因子|欧拉函数|费马小定理|中国剩余定理|原根|乘法阶|整数解|(?:正整数|整数)\s*[A-Za-z][^。；;\n]{0,80}(?:满足|使得)[^。；;\n]{0,100}(?<![<>!])=(?!=)|\b(?:primes?|prime factor|euler phi|fermat(?:'s)? little theorem|chinese remainder theorem|primitive root|multiplicative order|integer solutions?|positive integer (?:pairs?|tuples?)|pairs? of positive integers?)\b", 7),
    _Signal(
        "数论",
        "elementary-number-theory-objects",
        r"正(?:因数|约数)|约数个数|因数个数|最大公因数|最小公倍数|勾股数|佩尔方程|"
        r"\b(?:positive divisors?|number of (?:positive )?divisors?|divisor function|"
        r"gcd|lcm|primitive Pythagorean triples?|Pell equations?|prime factorization|"
        r"[pq][- ]adic valuations?)\b|"
        r"\b(?:greatest|largest)\s+integer\b[^.;\n]{0,160}\\mid",
        9,
    ),
    _Signal("非基础及进阶课程", "olympiad-structures", r"函数方程|不等式证明|对称不等式|循环不等式|极值构造|\b(?:functional equation|prove (?:the )?inequality|symmetric inequality|cyclic inequality|extremal construction)\b", 6),
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
    # A second field close to the top score means the statement is genuinely
    # mixed-domain. Keep confidence low there so routing can retain both
    # protocols instead of overcommitting to a one-word lead.
    confidence = (
        "high" if top >= 10 and margin >= 4
        else "medium" if top >= 7 and margin >= 3
        else "low"
    )
    matched = tuple(
        f"{subject}:{name}"
        for subject, _ in ranked
        for name in matches.get(subject, ())
    )
    return SubjectClassification(primary, secondary, confidence, matched, tuple(ranked))


def classify_subject(problem: str) -> Optional[str]:
    return classify_subjects(problem).primary
