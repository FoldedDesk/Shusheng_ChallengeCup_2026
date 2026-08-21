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
    r"是否|能否|可否|(?:^|[\n。；;])\s*判断(?:题|命题)?\s*[:：.]|"
    r"判断(?:下列|该|此)?.*(?:正确|错误|成立)|真假|"
    r"\b(?:true or false|whether|is it true|decide if|determine whether|yes or no)\b|"
    r"(?:^|[,，;；]\s*)(?!(?:what|which|who|where|when|why|how)\b)"
    r"(?:is|are|does|do|can|could|will|would)\b[^?]{1,300}\?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_ROOTS = re.compile(
    r"解集|(?:求解?|解)\s*(?:该|此|下列)?方程|"
    r"(?:求|找出|确定)[^。！？!?\n]{0,100}(?:正整数|整数)\s*[A-Za-z]"
    r"[^。！？!?\n]{0,100}(?:满足|使得)[^。！？!?\n]{0,100}(?<![<>!])=(?!=)|"
    r"方程[^。！？!?\n]{0,300}(?:(?:全部|所有)[^。！？!?\n]{0,40}(?:解|根)|"
    r"(?:实数|复数|有理数|整数)(?:范围内的?)?(?:解|根))|"
    r"(?:求|找出|确定|列出)[^。！？!?\n]{0,200}(?:根|零点)|"
    r"\bsolve\s+(?:the\s+)?(?:equation|inequality)\b|"
    r"\bsolve\b[^.!?\n]{0,240}(?:(?<![<>!])=(?!=)|\bsolutions?\b|\broots?\b)|"
    r"\bfind\s+([A-Za-z])\s+if\s+\1\b[^.!?\n]{0,160}(?<![<>!])=(?!=)|"
    r"\b(?:find|determine|list)\s+(?:the\s+)?(?:all\s+)?(?:real\s+|complex\s+)?(?:roots?|zeros?)\b|"
    r"\b(?:find|determine|list)\s+all\s+(?:positive\s+)?integer\s+"
    r"(?:pairs?|tuples?)\b[^.!?\n]{0,180}(?:satisfying|such that)[^.!?\n]{0,120}(?<![<>!])=(?!=)|"
    r"\b(?:find|determine|list)\s+all\s+solutions?\b",
    re.IGNORECASE | re.DOTALL,
)

_INTERVAL = re.compile(
    r"解集|定义域|值域|收敛区间|参数范围|在哪些区间|"
    r"\b(?:solution set|domain|range|interval of convergence|parameter range)\b",
    re.IGNORECASE,
)

_COUNT = re.compile(
    r"多少(?:个|种|条|次|对|组|棵|张|项|类)?|数目|数量|个数|总数|总计|"
    r"(?:排列|置换|组合|选法|方案|方式|配置|着色|染色|标记|匹配|分拆|划分|"
    r"路径|回路|生成树|子集|序列|字符串|映射)(?:的)?(?:总)?(?:数|数目|数量|个数)|"
    r"(?:^|[。；;\n])\s*计数(?=所有|全部|满足|具有|[A-Za-z\u4e00-\u9fff])|"
    r"\b(?:how many|number of|total number|count\s+(?:all\s+|the\s+)?|cardinality)\b",
    re.IGNORECASE,
)

_PARAMETRIC_RESULT = re.compile(
    r"\bin\s+terms\s+of\s+(?:the\s+)?(?:parameter\s+)?"
    r"(?:\$[^$]{1,80}\$|[A-Za-z][A-Za-z0-9_]*)|"
    r"(?:as|write|express)\s+(?:a\s+)?function\s+of\s+"
    r"(?:\$[^$]{1,80}\$|[A-Za-z][A-Za-z0-9_]*)|"
    r"用\s*(?:参数)?\s*(?:\$[^$]{1,80}\$|[A-Za-z][A-Za-z0-9_]*)\s*表示|"
    r"关于\s*(?:\$[^$]{1,80}\$|[A-Za-z][A-Za-z0-9_]*)\s*的(?:公式|表达式|函数)|"
    r"\$?\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*\1\s*"
    r"\(\s*[A-Za-z][A-Za-z0-9_,\s]*\s*\)",
    re.IGNORECASE,
)

_REQUESTED_EXPRESSION_OBJECT = re.compile(
    r"(?:求|确定|找出|写出|给出|构造|导出)[^。！？!?\n]{0,260}"
    r"(?:多项式|函数|算子|伴随算子|估计量|估计式|公式|表达式|通解|特解|极小元|极大元)"
    r"(?:\s*\$[^$]{1,80}\$|\s*[A-Za-z](?:\([^)]*\))?)?\s*$|"
    r"\b(?:find|determine|construct|derive|give)\b[^.!?\n]{0,220}"
    r"\b(?:polynomial|function|operator|adjoint|estimator|formula|expression|minimizer|maximizer)\b",
    re.IGNORECASE,
)

_ALL_OBJECTS = re.compile(
    r"(?:求|确定|找出|列出|分类)[^。！？!?\n]{0,180}(?:所有|全部)"
    r"[^。！？!?\n]{0,100}(?:整数|实数|复数|有理数|多项式|函数|映射|矩阵|"
    r"集合|子集|点|轨迹|参数|取值|走法|步骤|对象)|"
    r"\b(?:find|determine|list|classify)\s+all\s+"
    r"(?:possible\s+)?(?:(?:first|initial|legal|admissible)\s+)?"
    r"(?:(?:positive|negative|nonnegative|nonzero|real|complex|rational|integer-valued)\s+)*"
    r"(?:(?:[A-Za-z][A-Za-z-]*)\s+){0,3}"
    r"(?:integers?|numbers?|polynomials?|functions?|maps?|matrices|sets?|subsets?|"
    r"points?|loci|locus|parameters?|values?|moves?|steps?|configurations?|objects?)\b",
    re.IGNORECASE,
)

_SET_OR_TEXT_OBJECT = re.compile(
    r"轨迹|所有可能(?:的)?(?:走法|步骤|取值|参数|集合|点)|"
    r"(?:方法|指标|统计量|检验|模型|图形|图表|定理|术语|名称)\s*(?:是|为)?\s*$|"
    r"\b(?:find|determine|describe)\s+(?:the\s+)?locus\b|"
    r"\ball\s+possible\s+(?:(?:first|initial|legal|admissible)\s+)?"
    r"(?:moves?|steps?|values?|parameters?|sets?|points?|configurations?)\b|"
    r"\b(?:which|what)\s+(?:method|statistic|estimator|test|model|plot|chart|theorem|term|name)\b|"
    r"\b(?:name|state)\s+(?:the\s+)?(?:method|statistic|estimator|test|model|plot|chart|theorem|term)\b",
    re.IGNORECASE,
)

_BLANK = re.compile(
    r"填空|填入|fill (?:in|the blank)|"
    r"(?:\(\s*(?:\\(?:quad|qquad|;|,|\s))?\s*\)|（\s*）)|"
    r"(?<![A-Za-z0-9])_{3,}(?![A-Za-z0-9])|□+",
    re.IGNORECASE,
)

_CONTEST_SIGNAL = re.compile(
    r"证明|求证|求出所有|求所有|找出所有|确定所有|分类所有|对任意|对所有|"
    r"最小可能|最大可能|最小正整数|最大整数|构造|反例|染色|着色|涂色|"
    r"\b(?:prove|show\s+that|find\s+all|determine\s+all|classify\s+all|"
    r"for\s+(?:all|every)|least\s+possible|greatest\s+possible|"
    r"minimum\s+possible|maximum\s+possible|construct|counterexample|"
    r"colou?red|colorings?|"
    r"olympiad|math(?:ematical)?\s+contest|imo|aime)\b",
    re.IGNORECASE,
)

_STRUCTURAL_CONTEST_SIGNAL = re.compile(
    r"(?:棋盘|网格|方格|表格)[^。！？!?\n]{0,180}"
    r"(?:格子|行|列|操作|移动|标记|染色|铺砌|最少|最多|多少)|"
    r"(?:博弈|游戏)[^。！？!?\n]{0,180}(?:玩家|轮流|先手|后手|必胜|策略|操作|移动)|"
    r"(?![^。！？!?\n]{0,180}(?:随机|概率|骰子|硬币|掷|抛))(?:玩家|两人|双方)"
    r"[^。！？!?\n]{0,180}(?:轮流|先手|后手|必胜|策略|不能行动)|"
    r"(?![^。！？!?\n]{0,180}(?:随机|概率|骰子|硬币|掷|抛))(?:轮流|先手|后手)"
    r"[^。！？!?\n]{0,180}(?:取走|移动|操作|获胜|策略)|"
    r"(?:多米诺|骨牌|多联骨牌|铺砌|覆盖)[^。！？!?\n]{0,180}(?:棋盘|网格|方格|区域)|"
    r"(?:棋盘|网格|方格|区域)[^。！？!?\n]{0,180}(?:多米诺|骨牌|多联骨牌|铺砌|覆盖)|"
    r"(?:递推|递归)[^。！？!?\n]{0,120}(?:数列|序列|整数|项|字符串|二进制串)|"
    r"(?:字符串|二进制串|01串)[^。！？!?\n]{0,180}(?:递推|状态|不含|避免|个数|多少)|"
    r"\b(?:board|grid|checkerboard|table)\b[^.!?\n]{0,180}"
    r"\b(?:cells?|rows?|columns?|moves?|markings?|tilings?|colou?rings?|minimum|maximum|how many)\b|"
    r"\b(?:game|players?)\b[^.!?\n]{0,180}"
    r"\b(?:players?|turns?|take\s+turns|alternately|moves?|winning|strategy)\b|"
    r"\b(?:domino(?:es)?|polyomino(?:es)?|tilings?)\b[^.!?\n]{0,180}"
    r"\b(?:board|grid|region)\b|"
    r"\b(?:board|grid|region)\b[^.!?\n]{0,180}"
    r"\b(?:domino(?:es)?|polyomino(?:es)?|til(?:e|ed|ing))\b|"
    r"\b(?:recursive sequence|recurrence relation)\b",
    re.IGNORECASE | re.DOTALL,
)

_CONTEST_TOPICS = (
    (
        "olympiad_functional_equation",
        r"函数方程|(?:求出所有|求所有|找出所有|确定所有)[^。！？!?\n]{0,80}函数|"
        r"\bfunctional equations?\b|"
        r"\b(?:find|determine|classify)\s+all\s+[^.!?\n]{0,80}functions?\b",
        "高等代数",
    ),
    (
        "olympiad_geometry",
        r"三角形|四边形|多边形|圆周角|圆内接|共圆|切线|垂心|外心|内心|角平分线|"
        r"\b(?:triangles?|quadrilaterals?|polygons?|cyclic|tangent|collinear|"
        r"concurrent|circumcircle|incircle|orthocenter|circumcenter|incenter|"
        r"angle\s+bisector)\b",
        "欧氏几何",
    ),
    (
        "olympiad_number_theory",
        r"整数解|整除|同余|模\s*\d|素数|质数|最大公约数|最小公倍数|"
        r"(?:求出所有|求所有|找出所有|确定所有)[^。！？!?\n]{0,100}正整数|"
        r"(?:求|找出|确定)[^。！？!?\n]{0,100}(?:全部|所有)"
        r"[^。！？!?\n]{0,40}(?:正整数|整数)(?:有序)?解|"
        r"(?:正整数|整数)\s*[A-Za-z][^。！？!?\n]{0,80}(?:满足|使得)"
        r"[^。！？!?\n]{0,100}(?<![<>!])=(?!=)|"
        r"丢番图|佩尔方程|\b(?:integer solutions?|(?:pairs?|triples?)\s+of\s+"
        r"(?:(?:positive|nonnegative|nonzero)\s+)?integers?|"
        r"positive integer (?:pairs?|tuples?)|"
        r"(?:find|determine|classify)\s+all\s+(?:positive\s+)?integers?|divisib|"
        r"congruen|modulo|prime numbers?|gcd|lcm|diophantine|pell equation)\w*\b|"
        r"\\pmod|\\mid",
        "数论",
    ),
    (
        "olympiad_combinatorics",
        r"染色|着色|涂色|排列|组合|容斥|抽屉原理|鸽巢原理|双计数|生成函数|子集族|锦标赛|"
        r"棋盘|网格|方格|表格|格子|博弈|游戏|玩家|轮流|先手|后手|必胜策略|"
        r"有限图|简单图|图中[^。！？!?\n]{0,60}(?:顶点|边|度数|团|独立集)|"
        r"二进制串|01串|字符串[^。！？!?\n]{0,60}(?:个数|计数|递推|不含|避免)|"
        r"格路径|铺砌|多米诺|骨牌|多联骨牌|项链|手镯|\b(?:colorings?|arrangements?|permutations?|"
        r"colou?red|combinations?|pigeonhole|double counting|inclusion[- ]exclusion|"
        r"set systems?|tournaments?|lattice paths?|tilings?|domino(?:es)?|polyomino(?:es)?|"
        r"boards?|grids?|checkerboards?|games?|players?|winning strateg(?:y|ies)|"
        r"necklaces?|bracelets?)\b",
        "离散数学",
    ),
    (
        "olympiad_inequality",
        r"不等式|均值不等式|柯西不等式|最小值|最大值|最佳常数|"
        r"\b(?:inequalit\w*|am[- ]gm|cauchy[- ]schwarz|minimum value|"
        r"maximum value|best constant)\b",
        "高等代数",
    ),
    (
        "olympiad_polynomial",
        r"多项式|韦达|根的重数|首一多项式|\b(?:polynomials?|vieta|"
        r"roots?,\s*counted\s+with\s+multiplicity|monic polynomial)\b",
        "高等代数",
    ),
    (
        "olympiad_sequence",
        r"递推数列|递归数列|数列.*递推|\b(?:sequences?|recurrence relations?|"
        r"recursive sequence)\b",
        "离散数学",
    ),
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
    if task_kind == "fill_blank" or _BLANK.search(target):
        if re.search(
            r"多少|数值|函数值|最小值|最大值|长度|面积|体积|概率|数量|个数|"
            r"\b(?:how many|numeric(?:al)? value|value of|minimum|maximum|length|"
            r"area|volume|probability|number of)\b",
            target,
            re.IGNORECASE,
        ):
            return "number"
        return "text"
    if re.search(
        r"牛顿法|二分法|割线法|欧拉法|迭代公式|"
        r"\b(?:newton|bisection|secant|euler method|iteration formula)\b",
        text,
        re.IGNORECASE,
    ) and re.search(
        r"迭代公式|第一次迭代|第一步迭代|x_?\{?1\}?|"
        r"\b(?:iteration formula|first iterate|first iteration)\b",
        target,
        re.IGNORECASE,
    ):
        # The requested object is the scheme and iterate, not the exact root
        # set of the equation being approximated.
        return "expression"
    if re.search(
        r"牛顿法|二分法|割线法|"
        r"\b(?:newton(?:'s)?\s+method|bisection|secant(?:\s+method)?)\b",
        text,
        re.IGNORECASE,
    ) and re.search(
        r"近似|数值解|误差|精度|"
        r"\b(?:approximate|numerical\s+solution|tolerance|accuracy)\b",
        target,
        re.IGNORECASE,
    ):
        # Numerical root finding requests an approximation, not the complete
        # exact solution set of the embedded equation.
        return "number"
    truth_requested = bool(_TRUTH.search(target) or _TRUTH.search(text))
    parameter_result_requested = bool(re.search(
        r"(?:哪些|何种|何值|什么)(?:参数)?(?:值|取值)|"
        r"(?:求|确定|找出)[^。！？!?\n]{0,100}(?:参数)?(?:的)?(?:所有)?取值|"
        r"\bfor\s+which\s+(?:values?|parameters?)\b|"
        r"\bwhich\s+(?:values?|parameters?)\b",
        target,
        re.IGNORECASE,
    ))
    if parameter_result_requested:
        # "For which values ... is it possible?" asks for a parameter set.
        # The embedded possibility predicate does not make it a yes/no task.
        return "expression"
    explicit_nonbinary_result = bool(re.search(
        r"计算|求(?!证)|化简|写出|导出|推导|构造|"
        r"\b(?:compute|calculate|evaluate|solve|simplify|derive|construct|write)\b",
        target,
        re.IGNORECASE,
    ))
    change_predicate_requested = bool(re.search(
        r"是否[^。！？!?\n]{0,80}(?:改变|变化|不变|保持|相同|依赖)|"
        r"\bwhether\b[^.!?\n]{0,100}\b(?:change|remain|invariant|depend)|"
        r"\b(?:changed|unchanged|invariant)\b",
        target,
        re.IGNORECASE,
    ))
    if change_predicate_requested:
        return "text"
    if truth_requested and not explicit_nonbinary_result:
        # A matrix, polynomial, operator, or interval can be part of the
        # hypothesis. When the only requested result is whether a property
        # holds, those setup nouns must not turn the contract into a symbolic
        # expression request.
        return "truth"
    composite_expression = bool(re.search(
        r"(?:特征方程|稳定边界|边界参数式|迭代矩阵|Jordan\s*块|若尔当块|"
        r"最小多项式|点谱|本质值域|加权正规方程|"
        r"characteristic equation|stability boundary|boundary parametrization|"
        r"iteration matrix|Jordan blocks?|minimal polynomial|point spectrum|"
        r"essential range|weighted normal equations?)",
        target,
        re.IGNORECASE,
    ) and re.search(
        r"(?:并|以及|同时|和|与|、)|\b(?:and|as well as|together with)\b",
        target,
        re.IGNORECASE,
    ))
    if composite_expression:
        # A trailing yes/no item (for example A-stability) is only one part of
        # a composite mathematical result.  It must not collapse the whole
        # answer shape to a bare truth value or scalar.
        return "expression"
    compound_nonbinary_request = bool(
        (_TRUTH.search(target) or _TRUTH.search(text))
        and re.search(
            r"(?:并|以及|同时|且|、)|[,，]\s*(?:并|且|还|说明|判断|检验|验证)|"
            r"\b(?:and|as well as|together with)\b|[,;]\s*(?:and\s+)?"
            r"(?:explain|determine|decide|check|state)\b",
            target,
            re.IGNORECASE,
        )
        and re.search(
            r"计算|求(?!证)|写出|导出|推导|构造|"
            r"\b(?:compute|calculate|evaluate|solve|derive|construct)\b",
            target,
            re.IGNORECASE,
        )
    )
    if compound_nonbinary_request:
        # A binary predicate may be only one component of the requested
        # result.  Preserve the computed/derived object as well instead of
        # reducing the whole contract to yes/no.  A request merely to justify
        # a judgement does not trigger this gate.
        return "expression"
    if _TRUTH.search(target) or _TRUTH.search(text):
        # An explicit binary request controls the answer shape even when its
        # subject is a polynomial, function, operator, or formula.  Those
        # object words are descriptors here, not requests to return the
        # object itself.  Non-binary parameter and composite requests have
        # already returned above.
        return "truth"
    if re.search(
        r"(?:最小|最低|最大|最高)[^。！？!?\n]{0,80}(?:容量|座位数?|值)|"
        r"(?:容量|座位数?)[^。！？!?\n]{0,80}(?:最小|最低|最大|最高)|"
        r"\b(?:minimum|least|maximum|greatest)\b[^.!?\n]{0,80}"
        r"\b(?:capacity|seats?|value)\b",
        target,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"(?:严格)?(?:计算|求|求值|evaluate|compute|calculate)[^。！？!?\n]{0,120}"
        r"(?:\\int|积分|integral)",
        target,
        re.IGNORECASE,
    ):
        # A trailing request to name the theorem used for an integral does
        # not turn the requested mathematical object into free-form text.
        return "expression"
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
    if re.search(
        r"(?:零点|根|解)的?(?:总)?(?:个数|数目|数量)|"
        r"\b(?:number|count)\s+of\s+(?:zeros?|roots?|solutions?)\b",
        target + "\n" + text,
        re.IGNORECASE,
    ):
        return "count"
    if _ROOTS.search(target) or _ROOTS.search(text):
        return "roots"
    if re.search(
        r"(?:计算|求|确定)\s*\$?\s*[A-Za-z][A-Za-z0-9_{}]*\s*\$?\s*的?"
        r"(?:最小值|最大值)|"
        r"\b(?:find|compute|determine)\s+(?:the\s+)?(?:minimum|maximum)\s+"
        r"(?:value\s+)?of\s+\$?[A-Za-z][A-Za-z0-9_{}]*\$?",
        target,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"期望(?:吸收|到达|首达|首次通过)?时间|平均(?:等待|吸收|到达)时间|"
        r"\bexpected\s+(?:absorption|hitting|first[- ]passage|waiting)?\s*time\b|"
        r"\bmean\s+(?:absorption|hitting|first[- ]passage|waiting)\s*time\b",
        target,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"期望(?![^。！？!?\n]{0,12}时间)|方差|协方差|矩|"
        r"(?<![A-Za-z])E\s*\[|(?<![A-Za-z])Var\s*[\[(]|"
        r"\b(?:expectation|variance|covariance|moments?)\b",
        target,
        re.IGNORECASE,
    ):
        # Moments are not probabilities and need not lie in [0,1].  Keep an
        # expression contract because a single prompt may request several
        # moments or a symbolic answer in parameters.
        return "expression"
    if re.search(
        r"概率(?!\s*(?:分布|密度|质量(?:函数)?|测度))|"
        r"\bprobability\s+(?:that|of)|\bhitting probability\b",
        target,
        re.IGNORECASE,
    ):
        return "probability"
    if _COUNT.search(target):
        return "count"
    if _PARAMETRIC_RESULT.search(target):
        return "expression"
    # The requested object outranks descriptor words such as "real
    # coefficients".  Conversely, a target ending in "maximum value" does
    # not match this gate and remains scalar-valued below.
    if _ALL_OBJECTS.search(target) or _REQUESTED_EXPRESSION_OBJECT.search(target):
        return "expression"
    if _SET_OR_TEXT_OBJECT.search(target):
        if re.search(r"轨迹|locus|all possible|所有可能", target, re.IGNORECASE):
            return "expression"
        return "text"
    if re.search(
        r"最小值|最大值|数值|函数值|面积|体积|周期|容量|维数|重数|判别式|范数|风险|常数|系数|曲率|"
        r"距离(?!函数|公式|表达式)|"
        r"\b(?:minimum value|maximum value|value|area|volume|period|capacity|dimension|"
        r"multiplicity|discriminant|norm|risk|constant|coefficient|curvature|"
        r"distance(?!\s+(?:function|formula|expression)))\b",
        target,
        re.IGNORECASE,
    ):
        return "number"
    if re.search(
        r"公式|表达式|通项|结论|命题|多项式|函数|导数|原函数|级数|推导|"
        r"\b(?:formula|expression|general solution|polynomial|function|derivative|"
        r"differentiat\w*|antiderivative|series|derive|derivation)\b",
        target,
        re.IGNORECASE,
    ):
        return "expression"
    # An unrecognized mathematical target is safer as a symbolic expression.
    # Explicit scalar/count/value phrases have already returned above.  The
    # former numeric default incorrectly imposed a digit requirement on logic
    # conclusions, recurrences, algebraic structures, and other symbolic
    # answers, causing otherwise correct candidates to be discarded.
    return "expression"


def _contest_topic(text: str) -> tuple[str, str]:
    if not (_CONTEST_SIGNAL.search(text) or _STRUCTURAL_CONTEST_SIGNAL.search(text)):
        return "", ""
    # A geometric object used only as a colored or counted carrier is a
    # combinatorics problem; this strong structural signal outranks shape
    # words such as triangle, polygon, or circle.
    combinatorics = next(
        item for item in _CONTEST_TOPICS
        if item[0] == "olympiad_combinatorics"
    )
    if re.search(combinatorics[1], text, re.IGNORECASE | re.DOTALL):
        return combinatorics[0], combinatorics[2]
    for topic, pattern, subject in _CONTEST_TOPICS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return topic, subject
    return "", ""


def _topic(
    text: str,
    subject: str,
    shape: str,
    task_kind: str,
    *,
    allow_contest: bool = True,
) -> str:
    specialized_rules = (
        (
            "directed_euler_circuits",
            r"有向(?:多重)?图[^。；;\n]{0,120}欧拉(?:闭迹|回路)|"
            r"欧拉(?:闭迹|回路)[^。；;\n]{0,120}(?:有向|固定首弧)|"
            r"固定首弧|BEST\s*定理|"
            r"\b(?:directed\s+(?:multi)?graph[^.;\n]{0,220}Eulerian circuits?|"
            r"Eulerian circuits?[^.;\n]{0,220}(?:directed|fixed first arc|specified first arc)|"
            r"(?:fixed|specified) first arc|BEST theorem)\b",
        ),
        (
            "plane_rooted_tree_enumeration",
            r"平面有根树|有序有根树|Lukasiewicz|Łukasiewicz|"
            r"(?:有根树|树)[^。；;\n]{0,100}出度(?:剖面|分布|序列)|"
            r"\b(?:ordered plane rooted trees?|plane rooted trees?|"
            r"Lukasiewicz words?|out[- ]degree profile)\b",
        ),
        (
            "lacunary_natural_boundary",
            r"稀疏幂级数|空隙幂级数|自然边界|Hadamard\s*空隙|Fabry\s*空隙|"
            r"\b(?:lacunary power series|Hadamard gap theorem|Fabry gap theorem|"
            r"natural boundary)\b|"
            r"(?=.*\bpower series\b)(?=.*\bdense boundary singularities\b)"
            r"(?=.*\banalytic continuation\b)(?=.*\bboundary arc\b)",
        ),
        (
            "runge_kutta_stability",
            r"SDIRK|Butcher\s*(?:表|数组)|稳定函数\s*R\s*\(\s*z\s*\)|"
            r"L\s*[-－— ]?稳定|Runge.?Kutta[^。；;\n]{0,120}(?:稳定函数|稳定域|阶条件)|"
            r"\b(?:SDIRK|Butcher tableau|stability function\s+R\s*\(\s*z\s*\)|"
            r"L[- ]stability|L[- ]stable|Runge[- ]Kutta[^.;\n]{0,120}"
            r"(?:stability function|stability region|order conditions?))\b",
        ),
        (
            "spherical_triangle_area",
            r"球面三角形|球面余弦定理|Girard\s*定理|球面盈量|"
            r"\b(?:spherical triangle|spherical law of cosines|Girard(?:'s)? theorem|"
            r"spherical excess)\b",
        ),
        (
            "weierstrass_sine_product",
            r"(?:正弦|双曲正弦)[^。；;\n]{0,100}(?:Weierstrass|无穷乘积|乘积公式)|"
            r"Weierstrass[^。；;\n]{0,100}(?:正弦|双曲正弦|无穷乘积)|"
            r"\b(?:Weierstrass sine product|hyperbolic sine infinite product|"
            r"sine infinite product)\b",
        ),
        (
            "two_dimensional_polyharmonic_fundamental_solution",
            r"(?:二维|R\s*\^?\s*2|\\mathbb\s*(?:\{R\}|R)\s*\^\s*\{?2\}?)"
            r"[^。；;\n]{0,140}(?:多调和|双调和|\\Delta\s*\^\s*(?:m|\{?2\}?)|"
            r"polyharmonic|biharmonic)[^。；;\n]{0,140}(?:基本解|fundamental solution)|"
            r"(?:多调和|双调和|\\Delta\s*\^\s*(?:m|\{?2\}?)|polyharmonic|biharmonic)"
            r"[^。；;\n]{0,140}(?:基本解|fundamental solution)[^。；;\n]{0,140}"
            r"(?:二维|R\s*\^?\s*2|\\mathbb\s*(?:\{R\}|R)\s*\^\s*\{?2\}?)|"
            r"(?:基本解|fundamental solution)[^。；;\n]{0,100}"
            r"(?:多调和|双调和|\\Delta\s*\^\s*(?:m|\{?2\}?)|polyharmonic|biharmonic)"
            r"[^。；;\n]{0,100}(?:二维|R\s*\^?\s*2|\\mathbb\s*(?:\{R\}|R)\s*\^\s*\{?2\}?)",
        ),
        (
            "cellular_homology",
            r"胞腔(?:链复形|边界(?:映射|矩阵)?|同调)|"
            r"CW\s*复形[^。；;\n]{0,140}(?:附着|同调)|"
            r"\b(?:cellular (?:chain complex|boundary(?: map| matrix)?|homology)|"
            r"CW complex[^.;\n]{0,160}(?:attaching|homology))\b",
        ),
        (
            "chebyshev_minimax",
            r"切比雪夫[^。；;\n]{0,120}(?:极小极大|交错|上确界)|"
            r"(?:极小极大|交错极值)[^。；;\n]{0,120}(?:多项式|首项系数)|"
            r"\b(?:Chebyshev[^.;\n]{0,120}(?:minimax|equioscillation)|"
            r"minimax polynomial|equioscillation theorem)\b",
        ),
        (
            "latin_square",
            r"拉丁方|拉丁方阵|\bLatin squares?\b|"
            r"(?=.*\b(?:symbols?|entries)\b)(?=.*\bevery row\b)"
            r"(?=.*\bevery column\b)(?=.*\b(?:exactly once|permutations?)\b)",
        ),
        (
            "nowhere_zero_flow",
            r"无处(?:为)?零流|流多项式|\b(?:nowhere[- ]zero flows?|flow polynomial)\b",
        ),
    )
    for specialized_topic, pattern in specialized_rules:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return specialized_topic
    if allow_contest:
        contest_topic, _ = _contest_topic(text)
        if contest_topic:
            return contest_topic
    if shape == "choice":
        return "choice"
    if task_kind == "construction":
        return "construction"
    rules = (
        ("numerical_method", r"牛顿法|二分法|割线法|有限差分|数值积分|Runge.?Kutta|\b(?:newton|bisection|secant|finite difference|quadrature|runge[- ]kutta)\b"),
        ("calculus", r"极限|导数|积分|级数|\b(?:limit|derivative|differentiat\w*|integral|series)\b"),
        ("equation", r"方程|不等式|\b(?:equation|inequality|solve)\b"),
        ("linear_algebra", r"矩阵|行列式|特征值|秩|\b(?:matrix|determinant|eigenvalue|rank)\b"),
        (
            "combinatorics",
            r"计数|排列|组合|容斥|生成函数|偏序集|线性扩张|"
            r"\b(?:counting|permutation|combination|inclusion[- ]exclusion|"
            r"generating function|posets?|partially ordered sets?|linear extensions?)\b",
        ),
        ("graph", r"顶点|边集|图论|生成树|\b(?:vertices?|edges?|graph|spanning tree)\b"),
        ("probability", r"概率|随机变量|期望|方差|\b(?:probability|random variable|expectation|variance)\b"),
        ("optimization", r"最小值|最大值|最优|线性规划|\b(?:minimum|maximum|optimal|linear programming)\b"),
    )
    for topic, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return topic
    if task_kind in {"proof", "derivation", "explanation"}:
        return "proof"
    return subject if subject != "进阶数学" else "general"


def classify_profile(problem: str) -> ProblemProfile:
    text = str(problem or "").strip()
    task_kind = classify_problem_type(text)
    subject_route = classify_subjects(text)
    shape = _answer_shape(text, task_kind)
    difficulty = classify_difficulty(text, task_kind)
    contest_topic, contest_subject = _contest_topic(text)
    primary_subject = subject_route.primary
    secondary_subject = subject_route.secondary
    subject_confidence = subject_route.confidence
    matched_signals = subject_route.matched_signals
    explicit_numerical_method = bool(re.search(
        r"线性多步法|多步法|向后差分公式|后向差分公式|Jacobi\s*法|雅可比迭代|"
        r"\bJacobi\b(?=[^。；;\n]{0,40}(?:迭代|矩阵|谱半径))|"
        r"Gauss.?Seidel|高斯.?赛德尔|超松弛|"
        r"\b(?:BDF\s*\d*|backward differentiation formula|linear multistep method|"
        r"Adams[- ](?:Bashforth|Moulton)|Jacobi iteration|Gauss[- ]Seidel|"
        r"successive over[- ]relaxation)\b",
        text,
        re.IGNORECASE,
    ))
    if explicit_numerical_method and primary_subject != "数值分析":
        secondary_subject = primary_subject
        primary_subject = "数值分析"
        subject_confidence = "high"
        matched_signals = tuple(dict.fromkeys((
            *matched_signals,
            "数值分析:explicit-numerical-method",
        )))
    strong_distinct_domain = False
    if contest_topic and contest_subject:
        # A contest-style quantifier must not erase a strong mathematical
        # domain.  For example, "positive integer N" occurs in the Cauchy
        # criterion and a recurrence can still be fundamentally number
        # theoretic.  The contest topic remains useful for the method route.
        subject_scores = dict(subject_route.scores)
        primary_score = subject_scores.get(primary_subject, 0)
        competing_score = max(
            (
                score
                for candidate, score in subject_route.scores
                if candidate != primary_subject
            ),
            default=0,
        )
        distinct_domain_signal = bool(
            subject_confidence == "high"
            or (primary_score >= 7 and primary_score - competing_score >= 3)
            or (
                primary_subject == "数值分析"
                and explicit_numerical_method
            )
        )
        strong_distinct_domain = bool(
            distinct_domain_signal
            and primary_subject not in {
                "进阶数学", "非基础及进阶课程", contest_subject,
            }
            and not (
                contest_topic == "olympiad_combinatorics"
                and primary_subject == "欧氏几何"
                and re.search(r"染色|着色|涂色|\bcolou?red\b", text, re.IGNORECASE)
            )
        )
        if strong_distinct_domain:
            secondary_subject = contest_subject
        else:
            if primary_subject not in {"进阶数学", "非基础及进阶课程", contest_subject}:
                secondary_subject = primary_subject
            primary_subject = contest_subject
            subject_confidence = (
                "high" if subject_route.primary == contest_subject else "medium"
            )
        matched_signals = tuple(dict.fromkeys((
            *matched_signals,
            f"{contest_subject}:topic:{contest_topic}",
        )))
        difficulty = "hard"
    selected_topic = _topic(
        text,
        primary_subject,
        shape,
        task_kind,
        allow_contest=not strong_distinct_domain,
    )
    if selected_topic in {
        "directed_euler_circuits",
        "plane_rooted_tree_enumeration",
        "lacunary_natural_boundary",
        "runge_kutta_stability",
        "spherical_triangle_area",
        "weierstrass_sine_product",
        "two_dimensional_polyharmonic_fundamental_solution",
    }:
        difficulty = "hard"
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
        subject=primary_subject,
        problem_type=task_kind,
        difficulty=difficulty,
        answer_shape=shape,
        language=_language(text),
        tool_eligible=tool_eligible,
        task_kind=task_kind,
        result_kind=result_kind,
        topic=selected_topic,
        confidence=confidence,
        primary_subject=primary_subject,
        secondary_subject=secondary_subject,
        subject_confidence=subject_confidence,
        matched_signals=matched_signals,
    )
