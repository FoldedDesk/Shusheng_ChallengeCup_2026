"""Conservative detection of certifiable local-computation opportunities.

This module never derives an answer. It only decides whether a statement
contains an independently describable, explicitly bounded subproblem that a
small deterministic operation can execute. False negatives are preferable to
exposing an attractive but semantically unrelated tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class LocalToolOpportunityKind(str, Enum):
    NONE = "NONE"
    FINITE_ENUM = "FINITE_ENUM"
    FINITE_STATE = "FINITE_STATE"
    SUBTRACTION_GAME = "SUBTRACTION_GAME"
    PERMUTATION_CYCLES = "PERMUTATION_CYCLES"
    LATTICE_POLYGON = "LATTICE_POLYGON"
    FACTORIAL_VALUATION = "FACTORIAL_VALUATION"
    MODULAR_POWER = "MODULAR_POWER"
    RECURRENCE = "RECURRENCE"
    MODULAR = "MODULAR"
    DIGIT_DP = "DIGIT_DP"
    ALGEBRAIC_VERIFICATION = "ALGEBRAIC_VERIFICATION"
    DIFFERENTIAL_VERIFICATION = "DIFFERENTIAL_VERIFICATION"
    MATRIX_VERIFICATION = "MATRIX_VERIFICATION"
    NORMALIZATION_VERIFICATION = "NORMALIZATION_VERIFICATION"
    FUNCTIONAL_EQUATION_VERIFICATION = "FUNCTIONAL_EQUATION_VERIFICATION"
    EQUATION_SOLVE = "EQUATION_SOLVE"
    SYMBOLIC_CALCULUS = "SYMBOLIC_CALCULUS"
    LINEAR_ALGEBRA = "LINEAR_ALGEBRA"


class ToolEligibilityLevel(str, Enum):
    """How a certified operation may be used by a solver.

    Detection alone never certifies that a model-translated operation matches
    the statement.  The level only limits the strongest permitted use after a
    separate operation contract succeeds.
    """

    NONE = "NONE"
    VERIFICATION_ONLY = "VERIFICATION_ONLY"
    LOCAL_FACT = "LOCAL_FACT"
    FULLY_COVERED = "FULLY_COVERED"


@dataclass(frozen=True)
class LocalToolOpportunity:
    kind: LocalToolOpportunityKind = LocalToolOpportunityKind.NONE
    allowed_tools: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    scope: str = "none"

    @property
    def level(self) -> ToolEligibilityLevel:
        return {
            "verification_only": ToolEligibilityLevel.VERIFICATION_ONLY,
            "derived_subproblem": ToolEligibilityLevel.LOCAL_FACT,
            "statement_exact": ToolEligibilityLevel.FULLY_COVERED,
        }.get(self.scope, ToolEligibilityLevel.NONE)

    @property
    def eligible(self) -> bool:
        return self.kind is not LocalToolOpportunityKind.NONE and bool(
            self.allowed_tools
        )

    def trace_content(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "level": self.level.value,
            "eligible": self.eligible,
            "allowed_tools": list(self.allowed_tools),
            "evidence": list(self.evidence),
            "scope": self.scope,
        }


_COUNT_QUERY = re.compile(
    r"个数|数目|总数|多少(?:个|种|组)?|计数|"
    r"\b(?:count|how\s+many|number\s+of)\b",
    re.IGNORECASE,
)
_LIST_QUERY = re.compile(
    r"(?:求|找出|确定|列出)\s*(?:全部|所有)|全部(?:的)?解|所有(?:的)?解|"
    r"\b(?:find|determine|list)\s+all\b|\ball\s+solutions?\b",
    re.IGNORECASE,
)
_OPTIMIZE_QUERY = re.compile(
    r"最大(?:值|的)?|最小(?:值|的)?|至多|至少|"
    r"\b(?:maximum|minimum|largest|smallest|maximize|minimize)\b",
    re.IGNORECASE,
)
_INTEGER_DOMAIN = re.compile(
    r"整数|自然数|非负整数|正整数|"
    r"\b(?:integers?|natural\s+numbers?|nonnegative\s+integers?|"
    r"positive\s+integers?)\b",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"(?<![<>!])=(?!=)|(?:<=|>=|<|>|≤|≥|\\leq(?:slant)?|\\geq(?:slant)?)|"
    r"(?:\\equiv|≡)|同余|"
    r"\b(?:congruent|divisible)\b|整除",
    re.IGNORECASE,
)
_CHAINED_BOUND = re.compile(
    r"(?P<lo>-?\d{1,7})\s*(?:<=|≤|<|\\leq(?:slant)?)\s*"
    r"(?P<var>[A-Za-z](?:\s*_\s*\{?[A-Za-z0-9]+\}?)?)\s*"
    r"(?:<=|≤|<|\\leq(?:slant)?)\s*(?P<hi>-?\d{1,7})"
    r"(?![A-Za-z0-9_])"
    r"(?!\s*(?:[*/^]|[+-]\s*[A-Za-z]))",
    re.IGNORECASE,
)
_FINITE_SET_DOMAIN = re.compile(
    r"(?P<var>[A-Za-z])\s*(?:\\in|∈|in)\s*"
    r"(?:\\?\{(?P<items>[-+]?\d+(?:\s*[,，]\s*[-+]?\d+){1,40})\\?\})",
    re.IGNORECASE,
)

_MODULUS = re.compile(
    r"(?:\\pmod\s*\{?\s*|\bmod(?:ulo)?\s+|模\s*)"
    r"[$\\({\[]*\s*(?P<value>\d{1,10})",
    re.IGNORECASE,
)
_MODULAR_RELATION = re.compile(
    r"\\equiv|≡|同余|\bcongruen(?:t|ce)\b|\\pmod|\bmodulo\b",
    re.IGNORECASE,
)
_RESIDUE_DOMAIN = re.compile(
    r"剩余类|完全剩余系|模\s*\d+\s*(?:的)?(?:解|剩余)|"
    r"\b(?:residue\s+classes?|complete\s+residue\s+system|"
    r"solutions?\s+(?:modulo|mod))\b",
    re.IGNORECASE,
)

_RECURRENCE_WORD = re.compile(
    r"递推|递归定义|数列|序列|斐波那契|"
    r"\b(?:recurrence|recursively|sequence|Fibonacci)\b",
    re.IGNORECASE,
)
_RECURRENCE_RELATION = re.compile(
    r"[A-Za-z]\s*_\s*\{?\s*n\s*(?:[+-]\s*\d+)?\s*\}?\s*="
    r"[^\n;；。]{1,320}",
    re.IGNORECASE,
)
_INITIAL_VALUE = re.compile(
    r"[A-Za-z]\s*_\s*\{?\s*\d{1,4}\s*\}?\s*=\s*"
    r"[-+]?\d+(?:\s*/\s*\d+)?",
    re.IGNORECASE,
)
_TARGET_TERM = re.compile(
    r"(?:求|计算|确定|find|compute|determine)[^\n。;；]{0,100}?"
    r"[A-Za-z]\s*_\s*\{?\s*(?P<index>\d{1,6})\s*\}?",
    re.IGNORECASE,
)

_DIGIT_SIGNAL = re.compile(
    r"十进制|数位|位数|数字串|"
    r"\b(?:base[- ]?10|decimal|digit\s+strings?|digits?)\b",
    re.IGNORECASE,
)
_DIGIT_LENGTH = re.compile(
    r"\b\d{1,5}[- ]digit\b|"
    r"\b(?:length|digits?\s+long)\s*(?:is|=|of)?\s*\d{1,5}\b|"
    r"\d{1,5}\s*位(?:正整数|整数|数|数字串)",
    re.IGNORECASE,
)
_DIGIT_SET = re.compile(
    r"(?:数字|digits?)\s*(?:只能|仅能|取自|来自|are|from|in|chosen\s+from|using)"
    r"[^\n。;；]{0,30}(?:\\?\{\s*)?"
    r"\d(?:\s*[,，]\s*\d){1,9}(?:\s*\\?\})?",
    re.IGNORECASE,
)
_UNSUPPORTED_DIGIT_RESTRICTION = re.compile(
    r"互不相同|各不相同|不重复|无重复|相邻|数位和|每个数字|恰好出现|"
    r"\b(?:distinct|without\s+repetition|no\s+repetition|adjacent|"
    r"digit\s+sum|each\s+digit|exactly\s+once)\b",
    re.IGNORECASE,
)
_HIGH_ORDER_FINITE_OBJECT = re.compile(
    r"排列|置换|序列|数列|路径|回路|图|棋盘|网格|博弈|游戏|着色|染色|"
    r"子集|函数|铺砌|多米诺|"
    r"\b(?:permutations?|sequences?|paths?|cycles?|graphs?|boards?|grids?|"
    r"games?|colourings?|colorings?|subsets?|functions?|tilings?|dominoes?)\b",
    re.IGNORECASE,
)

_FINITE_ASSIGNMENT_OBJECT = re.compile(
    r"排列|置换|双射|完美匹配|匹配数|分配方式|安排方式|"
    r"\b(?:permutations?|bijections?|perfect\s+matchings?|assignments?|"
    r"ways?[^.。;；]{0,80}(?:distribut\w*|assign\w*|arrang\w*))\b",
    re.IGNORECASE,
)
_EXPLICIT_SMALL_ASSIGNMENT_SIZE = re.compile(
    r"(?:there\s+(?:are|is)|given|among|with|有|共)\s*"
    r"[$\\({\[]*\s*(?P<size>\d{1,2})\s*[$\\)}\]]*\s*"
    r"(?:people|persons?|guests?|workers?|tasks?|jobs?|objects?|items?|"
    r"rows?|columns?|人|位|个(?:人|任务|对象|物品))\b",
    re.IGNORECASE,
)
_INDEXED_NUMERIC_TERM = re.compile(
    r"[A-Za-z]\s*_\s*\{?\s*\d{1,6}\s*\}?",
    re.IGNORECASE,
)

_FINITE_STATE_SIGNAL = re.compile(
    r"状态转移|转移矩阵|邻接矩阵|有限状态|长度为\s*\d+\s*的游走|"
    r"\b(?:finite[- ]state|transition\s+matrix|adjacency\s+matrix|"
    r"walks?\s+of\s+length\s+\d+)\b",
    re.IGNORECASE,
)
_SUBTRACTION_GAME_SIGNAL = re.compile(
    r"取石子|取走石子|每次(?:可以|可)(?:取|拿|移走)|减法博弈|"
    r"石子[^。；;\n]{0,100}每次(?:恰好)?(?:可以|可)?(?:取走|取|拿|移走)|"
    r"\b(?:subtraction|take[- ]away)\s+game\b|"
    r"\b(?:remov(?:e|es|ing)|tak(?:e|es|ing))\s+"
    r"(?:one\s+of\s+)?[^.。;；]{0,50}\bstones?\b",
    re.IGNORECASE,
)
_GAME_OUTCOME_QUERY = re.compile(
    r"先手|后手|必胜|胜负|获胜策略|谁(?:能|会)?赢|"
    r"\b(?:winning|losing|winner|who\s+wins?|winning\s+strategy)\b",
    re.IGNORECASE,
)
_NORMAL_PLAY_SIGNAL = re.compile(
    r"无法操作者输|不能(?:再)?操作(?:者)?输|无子可取(?:者)?输|正常玩法|"
    r"\b(?:normal\s+play|player\s+unable\s+to\s+move\s+loses?|"
    r"last\s+(?:move|player\s+to\s+move)\s+wins?)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SUBTRACTION_RULE = re.compile(
    r"两堆|多堆|若干堆|分成|拆分|最后取走者输|反常玩法|"
    r"\b(?:multiple\s+heaps?|two\s+heaps?|split(?:ting)?|mis[eè]re)\b",
    re.IGNORECASE,
)
_PERMUTATION_SIGNAL = re.compile(
    r"排列|置换|\bpermutations?\b",
    re.IGNORECASE,
)
_CYCLE_INVENTORY_SIGNAL = re.compile(
    r"循环分解|轮换|\d+\s*[- ]?循环|长度为\s*\d+\s*的循环|不动点|"
    r"\b(?:cycle\s+(?:decomposition|lengths?|structure)|"
    r"\d+[- ]cycles?|fixed\s+points?)\b",
    re.IGNORECASE,
)
_LATTICE_POLYGON_SIGNAL = re.compile(
    r"格点多边形|整点多边形|皮克定理|Pick\s*定理|"
    r"\b(?:lattice\s+polygon|Pick(?:'s)?\s+theorem)\b",
    re.IGNORECASE,
)
_INTERIOR_LATTICE_QUERY = re.compile(
    r"内部格点|内点(?:数|个数)|边界格点|面积|"
    r"\b(?:interior|boundary)\s+lattice\s+points?\b|\barea\b",
    re.IGNORECASE,
)
_FACTORIAL_VALUATION_SIGNAL = re.compile(
    r"(?:阶乘|!)[^。；;\n]{0,140}(?:p进赋值|(?:p|\d+)[- ]?adic|valuation|指数|幂次|整除)|"
    r"(?:p进赋值|(?:p|\d+)[- ]?adic|valuation)[^.。;；\n]{0,140}(?:factorial|阶乘|!)|"
    r"\bv\s*_?\s*\{?p\}?\s*\([^)]*!",
    re.IGNORECASE,
)
_MODULAR_POWER_SIGNAL = re.compile(
    r"(?:幂|指数|\^)[^。；;\n]{0,180}(?:模|同余|余数)|"
    r"(?:模|余数)[^。；;\n]{0,180}(?:幂|指数|\^)|"
    r"\^[^.。;；\n]{0,180}\b(?:modulo|remainder)\b|"
    r"\b(?:modulo|remainder)[^.。;；\n]{0,180}\^|"
    r"\b(?:power|exponent)[^.。;；\n]{0,180}\b(?:modulo|remainder)\b|"
    r"\b(?:modulo|remainder)[^.。;；\n]{0,180}\b(?:power|exponent)\b",
    re.IGNORECASE,
)
_MODULAR_VALUE_QUERY = re.compile(
    r"求余|余数|模\s*\d+\s*(?:下)?(?:的)?(?:值|结果)|"
    r"\b(?:compute|evaluate|find|determine)[^.。;；]{0,120}"
    r"\b(?:modulo|remainder)\b",
    re.IGNORECASE,
)

_VERIFY_REQUEST = re.compile(
    r"验证|检验|核验|代入检查|是否满足|是否为.{0,20}解|"
    r"\b(?:verify|check|confirm|is\s+(?:a|the)\s+solution)\b",
    re.IGNORECASE,
)
_SOLVE_REQUEST = re.compile(
    r"(?:求|解|确定|计算)[^。；;\n]{0,80}(?:方程|方程组|解)|"
    r"\b(?:solve|find|determine|compute)[^.。;；\n]{0,80}"
    r"(?:equation|system|solution|root)\b",
    re.IGNORECASE,
)
_IDENTITY_SIGNAL = re.compile(
    r"恒等式|恒等于|等式.{0,30}成立|"
    r"\b(?:identity|identically\s+equal|holds?\s+identically)\b",
    re.IGNORECASE,
)
_EXPLICIT_RELATION = re.compile(
    r"(?<![<>!])=(?!=)|(?:\^|\*\*)|\\(?:equiv|frac|sqrt|mid)|"
    r"≡|∣|整除|\bdivides?\b",
    re.IGNORECASE,
)
_DIFFERENTIAL_EQUATION_SIGNAL = re.compile(
    r"微分方程|偏微分方程|常微分方程|初值问题|边值问题|"
    # A prime denotes a derivative only next to mathematical syntax.  The
    # narrower lookahead avoids treating prose quotes such as ``x 'special'``
    # as a differential equation.
    r"[A-Za-z]\s*(?:''|′′|″|'|′)(?=\s*(?:\(|=|[+\-*/^]))|"
    r"[A-Za-z]\s*_\s*\{?(?:x|t|xx|tt|xt|tx)\}?|"
    r"\b(?:ODE|PDE|differential\s+equation|initial[- ]value|boundary[- ]value)\b",
    re.IGNORECASE,
)
_DIFFERENTIAL_TARGET = re.compile(
    r"通解|特解|解为|满足|求解|验证|"
    r"\b(?:solution|solve|satisfy|verify)\b",
    re.IGNORECASE,
)
_MATRIX_SIGNAL = re.compile(
    r"矩阵|行列式|特征值|特征向量|逆矩阵|线性方程组|"
    r"\\begin\s*\{(?:pmatrix|bmatrix|vmatrix|matrix)\}|"
    r"\b(?:matrix|matrices|determinant|eigenvalue|eigenvector|inverse)\b",
    re.IGNORECASE,
)
_MATRIX_TARGET = re.compile(
    r"求|计算|确定|验证|检验|是否|"
    r"\b(?:compute|calculate|find|determine|verify|check)\b",
    re.IGNORECASE,
)
_PROBABILITY_NORMALIZATION_SIGNAL = re.compile(
    r"概率密度|密度函数|概率质量函数|分布列|归一化常数|"
    r"\b(?:probability\s+density|density\s+function|probability\s+mass|"
    r"pmf|pdf|normalizing\s+constant|normalization)\b",
    re.IGNORECASE,
)
_NORMALIZATION_TARGET = re.compile(
    r"是否.{0,20}(?:密度|分布)|验证|求.{0,20}常数|"
    r"\b(?:verify|check|find|determine|normalize)\b",
    re.IGNORECASE,
)
_FUNCTIONAL_EQUATION_SIGNAL = re.compile(
    r"(?:求|找出|确定)\s*(?:出)?\s*(?:全部|所有)\s*(?:函数|映射)|"
    r"(?:求|找出|确定)[^。；;\n]{0,24}(?:函数|映射)[^。；;\n]{0,80}"
    r"(?:满足|使得)|"
    r"\b(?:find|determine)\s+all"
    r"(?:\s+[A-Za-z][A-Za-z-]*){0,3}\s+(?:functions?|maps?)\b|"
    r"\b(?:find|determine)\s+[$\\({\[]*\s*[A-Za-z]\s*"
    r"(?::|\\colon)\s*[^.。;；\n]{0,100}",
    re.IGNORECASE,
)
_UNKNOWN_FUNCTION_CALL = re.compile(
    r"(?<![A-Za-z0-9_\\])(?P<name>[A-Za-z])\s*"
    r"(?:\\left\s*)?\([^()\n]{0,160}\)",
)
_FIXED_POLYNOMIAL_DECLARATION = re.compile(
    r"(?:\b(?:let|given)\s+(?P<en_name>[A-Za-z])\s+(?:be|is)\b"
    r"[^.。;；\n]{0,80}\bpolynomial\b)|"
    r"(?:\b(?:a|the)\s+polynomial\s+(?P<en_after>[A-Za-z])\s*\()|"
    r"(?:多项式\s*(?P<zh_name>[A-Za-z])\s*\()",
    re.IGNORECASE,
)
_UNKNOWN_FUNCTION_DECLARATION = re.compile(
    r"(?:\b(?:functions?|maps?|mappings?)\b|函数|映射)"
    r"[^.。;；\n]{0,48}?(?P<after>[A-Za-z])\s*(?::|\\colon|\(|\b)|"
    r"(?:\b(?:let|suppose|assume)\b|设)\s*[$\\({\[]*\s*"
    r"(?P<before>[A-Za-z])[^.。;；\n]{0,100}?"
    r"(?:\b(?:be|is)\s+(?:a\s+)?(?:function|map|mapping)\b|为.{0,12}(?:函数|映射))",
    re.IGNORECASE,
)
_HIGH_ORDER_FUNCTION_QUANTIFIER = re.compile(
    r"\b(?:for\s+(?:each|every|any)|given\s+any)\s+(?:a\s+)?"
    r"(?:function|map|mapping)\b|对(?:任意|每个|所有)(?:一个)?(?:函数|映射)|"
    r"任给(?:函数|映射)",
    re.IGNORECASE,
)
_POLYNOMIAL_SOLUTION_SIGNAL = re.compile(
    r"(?:求|找出|确定)\s*(?:全部|所有)?\s*多项式|"
    r"\b(?:find|determine)\s+all(?:\s+[A-Za-z-]+){0,3}\s+polynomials?\b|"
    r"\bpolynomial\b[^.。;；\n]{0,180}\b(?:for\s+all|identically)\b",
    re.IGNORECASE,
)


def _has_repeated_unknown_function_relation(text: str) -> bool:
    """Recognize an unknown function without mistaking named functions.

    Requiring the same standalone one-letter symbol at least twice preserves
    ordinary ``f(x+y)=f(x)+f(y)`` statements while excluding expressions such
    as ``arctan(k)=arctan(16)``. A polynomial already fixed by the statement is
    not an unknown-function search target either.
    """
    if _HIGH_ORDER_FUNCTION_QUANTIFIER.search(text):
        return False
    fixed_polynomials = {
        match.group(name).lower()
        for match in _FIXED_POLYNOMIAL_DECLARATION.finditer(text)
        for name in ("en_name", "en_after", "zh_name")
        if match.group(name)
    }
    declared_functions = {
        match.group(name).lower()
        for match in _UNKNOWN_FUNCTION_DECLARATION.finditer(text)
        for name in ("after", "before")
        if match.group(name)
    }
    if not declared_functions:
        return False
    counts: dict[str, int] = {}
    for match in _UNKNOWN_FUNCTION_CALL.finditer(text):
        name = match.group("name").lower()
        if name in fixed_polynomials or name not in declared_functions:
            continue
        counts[name] = counts.get(name, 0) + 1
    return any(count >= 2 for count in counts.values())
_DIRECT_EQUATION_QUERY = re.compile(
    r"(?:解|求解|求.{0,20}根|求.{0,20}方程组)|"
    r"\b(?:solve|find|determine)\b[^.。;；\n]{0,120}"
    r"\b(?:equation|system|roots?|solutions?)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_DIRECT_EQUATION = re.compile(
    r"\\sqrt|\\lfloor|\\rfloor|\\lvert|\\rvert|"
    r"\b(?:sin|cos|tan|log|ln|exp|floor|ceiling|absolute)\b|"
    r"\\frac\s*(?:\{[^{}]{0,80}\}|[^\s])\s*"
    r"\{?\s*[A-Za-z][^}]{0,80}\}?",
    re.IGNORECASE,
)
_EXPLICIT_SOLVE_VARIABLE = re.compile(
    r"(?:解|求解)\s*关于\s*(?P<zh_var>[A-Za-z])\s*的\s*方程|"
    r"\bsolve\b[^.。;；\n]{0,160}\bequation\b"
    r"[^.。;；\n]{0,160}\bfor\s+(?P<en_var>[A-Za-z])\b",
    re.IGNORECASE,
)
_DIRECT_INTEGRAL_QUERY = re.compile(
    r"(?:求|计算|确定|evaluate|compute|find|determine)"
    r"[^.。;；\n]{0,100}(?:\\int|积分)|"
    r"(?:\\int|积分)[^.。;；\n]{0,160}"
    r"(?:的值|等于多少|evaluate|compute)",
    re.IGNORECASE,
)
_DIRECT_LIMIT_QUERY = re.compile(
    r"(?:求|计算|确定|evaluate|compute|find|determine)"
    r"[^.。;；\n]{0,100}(?:\\lim(?![A-Za-z])|极限)|"
    r"(?:\\lim(?![A-Za-z])|极限)[^.。;；\n]{0,160}"
    r"(?:的值|evaluate|compute)",
    re.IGNORECASE,
)
_UNSUPPORTED_DIRECT_LIMIT = re.compile(
    r"\\mathbb\s*\{?[EP]\}?|期望|概率|"
    r"\b(?:expectation|probability)\b|"
    r"[A-Za-z]\s*_\s*\{?\s*n\s*\}?",
    re.IGNORECASE,
)
_DIRECT_DERIVATIVE_QUERY = re.compile(
    r"(?:求|计算|确定)[^。；;\n]{0,80}(?:导数|微分)|"
    r"\b(?:compute|calculate|find|determine)\b"
    r"[^.。;；\n]{0,80}\b(?:derivative|differential)\b",
    re.IGNORECASE,
)
_NUMERICAL_DERIVATIVE_METHOD = re.compile(
    r"数值微分|差分公式|(?:中心|前向|后向)差分|"
    r"\b(?:finite[- ]difference|central[- ]difference|forward[- ]difference|"
    r"backward[- ]difference|numerical\s+differentiat)\w*\b|"
    r"(?<![A-Za-z])h\s*=\s*[-+]?\d",
    re.IGNORECASE,
)
_EXPLICIT_MATRIX_LITERAL = re.compile(
    r"\[\s*\[[^\n]{1,800}\]\s*\]|"
    r"\\begin\s*\{(?:pmatrix|bmatrix|vmatrix|matrix|array)\}",
    re.IGNORECASE,
)
_DIRECT_MATRIX_OPERATION = re.compile(
    r"行列式|秩|逆矩阵|特征值|"
    r"\b(?:determinant|rank|inverse|eigenvalues?)\b",
    re.IGNORECASE,
)
_EXPLICIT_INTEGRAL = re.compile(r"\\int\s*_", re.IGNORECASE)
_TRANSFORM_VALUE_QUERY = re.compile(
    r"求.{0,40}(?:显式|表达式|形式|变换)|"
    r"\b(?:find|determine|compute)\b[^.。;；\n]{0,100}"
    r"\b(?:explicit\s+forms?|transforms?|values?)\b",
    re.IGNORECASE,
)
_POLYNOMIAL_FACTOR_CHECK = re.compile(
    r"多项式[^。；;\n]{0,180}(?:因式|分解|乘积)|"
    r"\bpolynomials?\b[^.。;；\n]{0,180}"
    r"\b(?:factor|factorization|factorisation|product)\b",
    re.IGNORECASE,
)
_PROVE_OR_DISPROVE = re.compile(
    r"证明或否定|证明或举反例|"
    r"\b(?:prove\s+or\s+disprove|prove\s+or\s+give\s+a\s+counterexample)\b",
    re.IGNORECASE,
)
_DIGIT_VARIABLES = re.compile(
    r"(?:digits?|数字)\s*[$\\({\[]*\s*"
    r"(?:[A-Za-z]\s*[,，]\s*){1,8}[A-Za-z]\s*[$\\)}\]]*|"
    r"[A-Za-z](?:\s*[,，]\s*[A-Za-z]){1,8}\s*(?:are\s+)?digits?",
    re.IGNORECASE,
)
_EXPLICIT_FINITE_SUM_STRUCTURE = re.compile(
    r"[A-Za-z]\s*_\s*\{?n\}?\s*=\s*[^.。;；\n]{0,500}"
    r"(?:\\sum|\\frac|\+\s*1\s*/\s*\d)",
    re.IGNORECASE,
)
_POLYNOMIAL_TRANSFORMATION_SIGNAL = re.compile(
    r"(?:多项式|polynomial)[^.。;；\n]{0,240}"
    r"(?:系数.{0,30}(?:交换|置换)|switch.{0,40}coefficients?|"
    r"replace[^.。;；\n]{0,80}P\s*\(\s*x\s*\+\s*1\s*\))",
    re.IGNORECASE,
)


def _finite_domain_size(text: str) -> tuple[int | None, int]:
    """Return a conservative state-count estimate and bounded variable count."""
    domains: dict[str, int] = {}
    for match in _CHAINED_BOUND.finditer(text):
        lower = int(match.group("lo"))
        upper = int(match.group("hi"))
        width = max(0, upper - lower + 1)
        domains[re.sub(r"\s+", "", match.group("var"))] = width
    for match in _FINITE_SET_DOMAIN.finditer(text):
        items = re.split(r"\s*[,，]\s*", match.group("items"))
        domains[match.group("var")] = len(set(items))
    if not domains or len(domains) > 3:
        return None, len(domains)
    states = 1
    for width in domains.values():
        states *= width
        if width <= 0 or states > 1_000_000:
            return None, len(domains)
    return states, len(domains)


def _finite_state_opportunity(text: str) -> LocalToolOpportunity | None:
    if not (
        _FINITE_STATE_SIGNAL.search(text)
        and _COUNT_QUERY.search(text)
        and re.search(r"\d", text)
    ):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.FINITE_STATE,
        ("finite_state_walk_count",),
        (
            "explicit_finite_transition_system",
            "numeric_walk_length",
            "count_query",
            "model_must_supply_complete_transition_matrix",
        ),
        "statement_exact",
    )


def _subtraction_game_opportunity(text: str) -> LocalToolOpportunity | None:
    if not (
        _SUBTRACTION_GAME_SIGNAL.search(text)
        and _GAME_OUTCOME_QUERY.search(text)
        and _NORMAL_PLAY_SIGNAL.search(text)
        and re.search(r"\d", text)
    ):
        return None
    if _UNSUPPORTED_SUBTRACTION_RULE.search(text):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.SUBTRACTION_GAME,
        ("subtraction_game_outcome",),
        (
            "normal_play_subtraction_game_candidate",
            "numeric_heap_or_moves_present",
            "winner_or_strategy_query",
            "model_must_supply_complete_move_set",
        ),
        "statement_exact",
    )


def _permutation_cycle_opportunity(text: str) -> LocalToolOpportunity | None:
    if not (
        _PERMUTATION_SIGNAL.search(text)
        and _CYCLE_INVENTORY_SIGNAL.search(text)
        and _COUNT_QUERY.search(text)
        and re.search(r"\d", text)
    ):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.PERMUTATION_CYCLES,
        ("permutation_cycle_count",),
        (
            "labelled_permutation_object",
            "explicit_cycle_inventory_restriction",
            "count_query",
            "model_must_supply_all_allowed_lengths_and_bounds",
        ),
        "statement_exact",
    )


def _lattice_polygon_opportunity(text: str) -> LocalToolOpportunity | None:
    if not (
        _LATTICE_POLYGON_SIGNAL.search(text)
        and _INTERIOR_LATTICE_QUERY.search(text)
        and re.search(r"[-+]?\d+\s*[,，]\s*[-+]?\d+", text)
    ):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.LATTICE_POLYGON,
        ("lattice_polygon_interior",),
        (
            "integer_coordinate_polygon_candidate",
            "explicit_vertices_present",
            "area_or_lattice_point_query",
            "local_handler_will_reject_non_simple_polygon",
        ),
        "statement_exact",
    )


def _factorial_valuation_opportunity(text: str) -> LocalToolOpportunity | None:
    query = re.search(
        r"求|计算|确定|指数|幂次|最高次幂|"
        r"\b(?:compute|calculate|find|determine|exponent|highest\s+power)\b",
        text,
        re.IGNORECASE,
    )
    if not (
        _FACTORIAL_VALUATION_SIGNAL.search(text)
        and query
        and re.search(r"\d", text)
    ):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.FACTORIAL_VALUATION,
        ("factorial_ratio_prime_valuation",),
        (
            "factorial_ratio_candidate",
            "prime_valuation_query",
            "numeric_factorial_arguments_present",
            "model_must_preserve_numerator_and_denominator_multiplicity",
        ),
        "statement_exact",
    )


def _modular_power_opportunity(text: str) -> LocalToolOpportunity | None:
    modulus = _MODULUS.search(text)
    if not (
        modulus
        and _MODULAR_POWER_SIGNAL.search(text)
        and _MODULAR_VALUE_QUERY.search(text)
    ):
        return None
    if _LIST_QUERY.search(text) or re.search(
        r"全部(?:的)?解|所有(?:的)?解|\bsolutions?\b",
        text,
        re.IGNORECASE,
    ):
        return None
    if not 1 <= int(modulus.group("value")) <= 2_000_000_000:
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.MODULAR_POWER,
        ("modular_power_sum",),
        (
            "explicit_modular_power_expression",
            "numeric_modulus",
            "value_or_remainder_query",
            "model_must_supply_every_term_and_exponent_shape",
        ),
        "statement_exact",
    )


def _digit_dp_opportunity(text: str) -> LocalToolOpportunity | None:
    if not (
        _DIGIT_SIGNAL.search(text)
        and _COUNT_QUERY.search(text)
        and _DIGIT_LENGTH.search(text)
        and _DIGIT_SET.search(text)
        and _MODULUS.search(text)
    ):
        return None
    if _UNSUPPORTED_DIGIT_RESTRICTION.search(text):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.DIGIT_DP,
        ("count_digit_strings",),
        (
            "explicit_digit_alphabet",
            "explicit_length_bound",
            "numeric_modulus",
            "count_query",
        ),
        "statement_exact",
    )


def _modular_opportunity(text: str) -> LocalToolOpportunity | None:
    modulus = _MODULUS.search(text)
    query = _COUNT_QUERY.search(text) or _LIST_QUERY.search(text)
    if not (modulus and query and _MODULAR_RELATION.search(text)):
        return None
    value = int(modulus.group("value"))
    if not 1 <= value <= 200_000:
        return None
    has_domain = bool(_RESIDUE_DOMAIN.search(text) or _CHAINED_BOUND.search(text))
    if not has_domain:
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.MODULAR,
        ("count_modular_solutions",),
        (
            "explicit_congruence",
            "numeric_modulus",
            "finite_residue_domain",
            "count_or_list_query",
        ),
        "statement_exact",
    )


def _recurrence_opportunity(text: str) -> LocalToolOpportunity | None:
    relation = _RECURRENCE_RELATION.search(text)
    target = _TARGET_TERM.search(text)
    initials = tuple(_INITIAL_VALUE.finditer(text))
    if not (
        _RECURRENCE_WORD.search(text)
        and relation
        and initials
        and target
    ):
        return None
    target_index = int(target.group("index"))
    if target_index > 100_000:
        return None
    # Variable forcing such as "+n" is outside the exact affine
    # constant-coefficient operation. Indexed terms are removed first.
    rhs = relation.group(0).split("=", 1)[1]
    stripped = re.sub(
        r"[A-Za-z]\s*_\s*\{?\s*n\s*(?:[+-]\s*\d+)?\s*\}?",
        "TERM",
        rhs,
        flags=re.IGNORECASE,
    )
    if re.search(r"(?<![A-Za-z0-9_])n(?![A-Za-z0-9_])", stripped, re.IGNORECASE):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.RECURRENCE,
        ("linear_recurrence_term",),
        (
            "explicit_recurrence_relation",
            "explicit_initial_values",
            "numeric_target_index",
            "constant_affine_form_candidate",
        ),
        "statement_exact",
    )


def _finite_enum_opportunity(text: str) -> LocalToolOpportunity | None:
    states, variable_count = _finite_domain_size(text)
    query = (
        _COUNT_QUERY.search(text)
        or _LIST_QUERY.search(text)
        or _OPTIMIZE_QUERY.search(text)
    )
    if _HIGH_ORDER_FINITE_OBJECT.search(text):
        return None
    if not (
        states is not None
        and 1 <= variable_count <= 3
        and query
        and _INTEGER_DOMAIN.search(text)
        and _CONSTRAINT.search(text)
    ):
        return None
    return LocalToolOpportunity(
        LocalToolOpportunityKind.FINITE_ENUM,
        ("bounded_integer_search",),
        (
            "explicit_integer_domain",
            "explicit_finite_bounds",
            "bounded_state_space",
            "count_list_or_optimize_query",
            "explicit_constraints",
        ),
        "statement_exact",
    )


def _derived_subproblem_opportunity(text: str) -> LocalToolOpportunity | None:
    """Expose one operation family only when a derived local fact is plausible.

    These guards do not assert that the tool operation answers the statement.
    They only establish that the statement contains a bounded mathematical
    object for which the solver can submit an independently checkable local
    contract.  Translation and exhaustiveness remain uncertified.
    """
    query = bool(
        _COUNT_QUERY.search(text)
        or _LIST_QUERY.search(text)
        or _OPTIMIZE_QUERY.search(text)
    )
    if (
        (
            query
            or re.search(
                r"(?:求|找出|确定)[^。；;\n]{0,30}(?:数字|数码)|"
                r"\b(?:find|determine)\s+digits?\b",
                text,
                re.IGNORECASE,
            )
        )
        and _DIGIT_VARIABLES.search(text)
        and _EXPLICIT_RELATION.search(text)
    ):
        return LocalToolOpportunity(
            LocalToolOpportunityKind.FINITE_ENUM,
            ("bounded_integer_search",),
            (
                "explicit_digit_variables",
                "implicit_zero_to_nine_domains",
                "explicit_relation",
                "model_must_derive_every_constraint",
            ),
            "derived_subproblem",
        )

    if _EXPLICIT_FINITE_SUM_STRUCTURE.search(text) and _INDEXED_NUMERIC_TERM.search(text):
        return LocalToolOpportunity(
            LocalToolOpportunityKind.RECURRENCE,
            ("finite_sum",),
            (
                "explicit_indexed_finite_sum",
                "numeric_target_index",
                "model_must_translate_nested_sum_exactly",
            ),
            "derived_subproblem",
        )
    assignment_sizes = tuple(
        int(match.group("size"))
        for match in _EXPLICIT_SMALL_ASSIGNMENT_SIZE.finditer(text)
    )
    if (
        query
        and _FINITE_ASSIGNMENT_OBJECT.search(text)
        and any(2 <= value <= 18 for value in assignment_sizes)
    ):
        return LocalToolOpportunity(
            LocalToolOpportunityKind.FINITE_ENUM,
            ("matrix_operation",),
            (
                "finite_assignment_or_matching_object",
                "explicit_small_cardinality",
                "count_or_optimize_query",
                "model_must_derive_incidence_matrix",
            ),
            "derived_subproblem",
        )

    if (
        _RECURRENCE_WORD.search(text)
        and _RECURRENCE_RELATION.search(text)
        and _INITIAL_VALUE.search(text)
        and _INDEXED_NUMERIC_TERM.search(text)
    ):
        return LocalToolOpportunity(
            LocalToolOpportunityKind.RECURRENCE,
            ("linear_recurrence_term",),
            (
                "explicit_recurrence_candidate",
                "explicit_initial_values",
                "numeric_index_present",
                "model_must_derive_supported_constant_affine_form",
                "local_term_only_not_global_claim",
            ),
            "derived_subproblem",
        )

    modulus = _MODULUS.search(text)
    if query and modulus and _MODULAR_RELATION.search(text):
        value = int(modulus.group("value"))
        if 1 <= value <= 200_000:
            return LocalToolOpportunity(
                LocalToolOpportunityKind.MODULAR,
                ("count_modular_solutions",),
                (
                    "explicit_numeric_modulus",
                    "finite_residue_computation_plausible",
                    "model_must_derive_polynomial_congruence",
                ),
                "derived_subproblem",
            )

    if (
        _DIGIT_SIGNAL.search(text)
        and _COUNT_QUERY.search(text)
        and _DIGIT_LENGTH.search(text)
        and _MODULUS.search(text)
        and not _UNSUPPORTED_DIGIT_RESTRICTION.search(text)
    ):
        return LocalToolOpportunity(
            LocalToolOpportunityKind.DIGIT_DP,
            ("count_digit_strings",),
            (
                "explicit_digit_length",
                "explicit_numeric_modulus",
                "count_query",
                "model_must_supply_complete_digit_alphabet",
            ),
            "derived_subproblem",
        )
    return None


def _verification_only_opportunities(
    text: str,
) -> tuple[LocalToolOpportunity, ...]:
    """Find claims that a future candidate can be checked mechanically.

    These opportunities deliberately cannot answer the problem.  They only
    identify a residual, substitution, or normalization check that may reject
    a bad candidate after the model has independently derived one.
    """
    opportunities: list[LocalToolOpportunity] = []

    if _POLYNOMIAL_TRANSFORMATION_SIGNAL.search(text):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.ALGEBRAIC_VERIFICATION,
            ("simplify_expression", "substitute_values"),
            (
                "explicit_polynomial_transformation",
                "candidate_step_or_invariant_check_only",
            ),
            "verification_only",
        ))

    if (
        (
            _FUNCTIONAL_EQUATION_SIGNAL.search(text)
            or _has_repeated_unknown_function_relation(text)
        )
        and _EXPLICIT_RELATION.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.FUNCTIONAL_EQUATION_VERIFICATION,
            ("simplify_expression", "substitute_values"),
            (
                "functional_equation_structure",
                "explicit_relation",
                "candidate_function_substitution_only",
            ),
            "verification_only",
        ))

    if _POLYNOMIAL_SOLUTION_SIGNAL.search(text) and _EXPLICIT_RELATION.search(text):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.ALGEBRAIC_VERIFICATION,
            ("simplify_expression", "substitute_values"),
            (
                "polynomial_solution_candidate",
                "explicit_relation",
                "candidate_polynomial_check_only",
            ),
            "verification_only",
        ))

    if (
        _DIFFERENTIAL_EQUATION_SIGNAL.search(text)
        and _DIFFERENTIAL_TARGET.search(text)
        and _EXPLICIT_RELATION.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.DIFFERENTIAL_VERIFICATION,
            ("differentiate_expression", "substitute_values"),
            (
                "explicit_differential_relation",
                "solution_or_verification_target",
                "candidate_residual_only",
            ),
            "verification_only",
        ))

    if (
        _MATRIX_SIGNAL.search(text)
        and _MATRIX_TARGET.search(text)
        and _EXPLICIT_MATRIX_LITERAL.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.MATRIX_VERIFICATION,
            ("matrix_operation",),
            (
                "explicit_matrix_literal",
                "computational_or_verification_target",
                "derived_invariant_or_candidate_check_only",
            ),
            "derived_subproblem",
        ))

    if (
        _PROBABILITY_NORMALIZATION_SIGNAL.search(text)
        and _NORMALIZATION_TARGET.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.NORMALIZATION_VERIFICATION,
            ("finite_sum", "definite_integral"),
            (
                "probability_normalization_object",
                "normalization_or_validity_target",
                "support_and_measure_must_be_supplied_by_model",
            ),
            "verification_only",
        ))

    if _POLYNOMIAL_FACTOR_CHECK.search(text) and _PROVE_OR_DISPROVE.search(text):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.ALGEBRAIC_VERIFICATION,
            ("simplify_expression",),
            (
                "polynomial_factorization_claim",
                "prove_or_disprove_target",
                "candidate_factorization_or_counterexample_only",
            ),
            "verification_only",
        ))

    algebraic_target = (
        _VERIFY_REQUEST.search(text)
        or _SOLVE_REQUEST.search(text)
        or _IDENTITY_SIGNAL.search(text)
    )
    if algebraic_target and _EXPLICIT_RELATION.search(text):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.ALGEBRAIC_VERIFICATION,
            ("simplify_expression", "substitute_values"),
            (
                "explicit_algebraic_relation",
                "solve_identity_or_verification_target",
                "candidate_residual_only",
            ),
            "verification_only",
        ))

    return tuple(opportunities)


def _direct_symbolic_opportunities(
    text: str,
) -> tuple[LocalToolOpportunity, ...]:
    """Find direct requests whose complete target is one whitelisted operation.

    The classification is still provisional: a model-supplied translation must
    pass the operation contract before a result can be considered covered.
    """
    opportunities: list[LocalToolOpportunity] = []
    direct_limit = bool(
        _DIRECT_LIMIT_QUERY.search(text)
        and re.search(r"\\lim(?![A-Za-z])", text)
        and not _UNSUPPORTED_DIRECT_LIMIT.search(text)
        and not _RECURRENCE_RELATION.search(text)
    )
    equation_system = bool(re.search(r"方程组|\bsystem\b", text, re.IGNORECASE))

    if (
        _DIRECT_EQUATION_QUERY.search(text)
        and _EXPLICIT_RELATION.search(text)
        and not _UNSUPPORTED_DIRECT_EQUATION.search(text)
        and not _FUNCTIONAL_EQUATION_SIGNAL.search(text)
        and not _has_repeated_unknown_function_relation(text)
        and (equation_system or _EXPLICIT_SOLVE_VARIABLE.search(text))
    ):
        tools = (
            ("solve_polynomial_system",)
            if equation_system
            else ("solve_equation",)
        )
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.EQUATION_SOLVE,
            tools,
            (
                "direct_equation_solution_target",
                "explicit_relation",
                "domain_and_all_constraints_must_survive_contract",
            ),
            "statement_exact",
        ))

    if _DIRECT_INTEGRAL_QUERY.search(text) and re.search(
        r"\\int\s*_\s*(?:\{|[^\s])[^\n]{0,100}\^", text
    ):
        integral_scope = (
            "derived_subproblem"
            if direct_limit or _OPTIMIZE_QUERY.search(text)
            else "statement_exact"
        )
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.SYMBOLIC_CALCULUS,
            ("definite_integral",),
            (
                (
                    "nested_definite_integral_subproblem"
                    if direct_limit
                    else "optimization_integral_subproblem"
                    if _OPTIMIZE_QUERY.search(text)
                    else "direct_definite_integral_target"
                ),
                "explicit_bounds_candidate",
                "integrand_variable_and_bounds_must_survive_contract",
            ),
            integral_scope,
        ))
    elif _EXPLICIT_INTEGRAL.search(text) and _TRANSFORM_VALUE_QUERY.search(text):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.SYMBOLIC_CALCULUS,
            ("definite_integral",),
            (
                "integral_defined_transform_subproblem",
                "explicit_integration_bounds_candidate",
                "transform_substitution_must_survive_contract",
            ),
            "derived_subproblem",
        ))

    if direct_limit:
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.SYMBOLIC_CALCULUS,
            ("limit_expression",),
            (
                "direct_limit_target",
                "explicit_limit_notation",
                "point_and_direction_must_survive_contract",
            ),
            "statement_exact",
        ))

    if (
        _DIRECT_DERIVATIVE_QUERY.search(text)
        and _EXPLICIT_RELATION.search(text)
        and not _NUMERICAL_DERIVATIVE_METHOD.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.SYMBOLIC_CALCULUS,
            ("differentiate_expression",),
            (
                "direct_derivative_target",
                "explicit_expression_candidate",
                "variable_and_requested_order_must_survive_contract",
            ),
            "statement_exact",
        ))

    if (
        _EXPLICIT_MATRIX_LITERAL.search(text)
        and _DIRECT_MATRIX_OPERATION.search(text)
        and _MATRIX_TARGET.search(text)
    ):
        opportunities.append(LocalToolOpportunity(
            LocalToolOpportunityKind.LINEAR_ALGEBRA,
            ("matrix_operation",),
            (
                "direct_explicit_matrix_operation",
                "matrix_literal_present",
                "requested_operation_must_survive_contract",
            ),
            "statement_exact",
        ))
    return tuple(opportunities)


_STATEMENT_DETECTORS = (
    _finite_state_opportunity,
    _subtraction_game_opportunity,
    _permutation_cycle_opportunity,
    _lattice_polygon_opportunity,
    _factorial_valuation_opportunity,
    _modular_power_opportunity,
    _digit_dp_opportunity,
    _modular_opportunity,
    _recurrence_opportunity,
    _finite_enum_opportunity,
)


def detect_local_tool_opportunities(
    problem: str,
    spec: Any | None = None,
) -> tuple[LocalToolOpportunity, ...]:
    """Return every structurally plausible opportunity, strongest first.

    This is an offline measurement entry point.  It neither exposes tools to
    the production model nor executes an operation.  Each returned operation
    still requires a valid precondition/execution/postcondition contract.
    """
    del spec  # Detection depends only on the current statement text.
    text = str(problem or "").strip()
    if not 1 <= len(text) <= 5_000:
        return ()

    candidates: list[LocalToolOpportunity] = []
    candidates.extend(
        opportunity
        for detector in _STATEMENT_DETECTORS
        if (opportunity := detector(text)) is not None
    )
    candidates.extend(_direct_symbolic_opportunities(text))
    derived = _derived_subproblem_opportunity(text)
    if derived is not None:
        candidates.append(derived)
    candidates.extend(_verification_only_opportunities(text))

    unique: list[LocalToolOpportunity] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for opportunity in candidates:
        key = (
            opportunity.level.value,
            opportunity.kind.value,
            opportunity.allowed_tools,
        )
        if key not in seen:
            seen.add(key)
            unique.append(opportunity)
    return tuple(unique)


def detect_local_tool_opportunity(
    problem: str,
    spec: Any | None = None,
    *,
    allow_derived: bool = False,
) -> LocalToolOpportunity:
    """Classify one local opportunity without deriving any mathematical value."""
    del spec  # Reserved for future statement-level, never metadata-based, guards.
    text = str(problem or "").strip()
    if not 1 <= len(text) <= 5_000:
        return LocalToolOpportunity()
    for detector in _STATEMENT_DETECTORS:
        opportunity = detector(text)
        if opportunity is not None:
            return opportunity
    if allow_derived:
        opportunity = _derived_subproblem_opportunity(text)
        if opportunity is not None:
            return opportunity
    return LocalToolOpportunity()
