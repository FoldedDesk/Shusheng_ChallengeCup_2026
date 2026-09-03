"""Single-problem solve blueprint and gradable answer contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from classifier.choice import answer_choice_labels, choice_stem
from classifier.profile import ProblemProfile, classify_profile
from classifier.semantics import StatementSemantics, extract_statement_semantics
from classifier.target import extract_target_clause


@dataclass(frozen=True)
class Requirement:
    name: str
    alternatives: tuple[tuple[str, ...], ...] = ()
    strict: bool = False
    category: str = "result"

    def matches(self, answer: str) -> bool:
        value = str(answer or "")
        compact = _compact(value)
        lowered = value.casefold()
        if self.name == "result_present":
            return bool(re.search(r"[\w\u4e00-\u9fff\\=<>≤≥+\-*/]", value))
        if self.name == "numeric_result":
            return bool(re.search(
                r"[-+]?\d|\\frac|\\sqrt|\\pi|π|∞|\\infty|\\lambda|"
                r"(?<![A-Za-z])[A-Za-z](?:\s*[_^]|\s*\\(?:cdot|times))",
                value,
            ))
        if self.name == "all_solutions":
            explicit_family = bool(re.search(
                r"所有|全部|解集|无解|不存在|\\varnothing|\\?\{|"
                r"(?<![A-Za-z])[A-Za-z](?:\s*\([^)]*\))?\s*(?:=|\\in)|"
                r"\ball\b|solution set|family|no solutions?",
                value,
                re.IGNORECASE,
            ))
            concise = value.strip().strip("$").strip()
            boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}\s*[。.]?", concise, re.DOTALL)
            if boxed:
                concise = boxed.group(1).strip().strip("$").strip()
            numeric_listing = bool(re.fullmatch(
                r"(?:[-+]?\d+(?:\s*/\s*\d+)?|"
                r"\\frac\s*\{[-+]?\d+\}\s*\{\d+\})"
                r"(?:\s*[,，;；]\s*(?:[-+]?\d+(?:\s*/\s*\d+)?|"
                r"\\frac\s*\{[-+]?\d+\}\s*\{\d+\}|\\(?:ldots|cdots)))+",
                concise,
            ))
            return explicit_family or numeric_listing
        if self.name == "exhaustive_result":
            explicit = bool(re.search(
                r"所有|全部|仅有|只有|恰为|解集|完整(?:集合|列表)|"
                r"\ball\b|\bonly\b|exactly|complete\s+(?:set|list)|solution set",
                value,
                re.IGNORECASE,
            ))
            set_or_range = bool(re.search(
                r"\\?[\{[][^\n]{1,500}\\?[\]}]|\\in|∈|\\leq?|≤|<="
                r"|\b(?:for|where)\s+[A-Za-z][^.;\n]{0,80}\b(?:integer|real|rational)\b",
                value,
                re.IGNORECASE,
            ))
            assignments = re.findall(
                r"(?<![A-Za-z0-9_])[A-Za-z](?:\s*_[A-Za-z0-9{}]+)?\s*=",
                value,
            )
            listed_values = bool(re.search(
                r"(?:\\?[{[]|\()[^\n]{0,300}[,，][^\n]{0,300}(?:\\?[}\]]|\))",
                value,
            ))
            bare_math_list = bool(re.fullmatch(
                r"\s*(?:\\boxed\s*\{)?\$?"
                r"[A-Za-z0-9\\_{}^+\-*/().\s]+"
                r"(?:[,，;；][A-Za-z0-9\\_{}^+\-*/().\s]+)+"
                r"\$?\}?\s*",
                value,
            ))
            multiple = len(assignments) >= 2 or listed_values or bare_math_list
            return explicit or set_or_range or multiple
        if self.name == "phrase_decomposition":
            labelled = bool(re.search(
                r"短语|分解|词组|\bphrases?\b|\bdecomposition\b",
                value,
                re.IGNORECASE,
            ))
            indexed_pairs = len(re.findall(
                r"\(\s*\d+\s*[,，]\s*[^(),，\s]+\s*\)",
                value,
            )) >= 2
            labelled_list = bool(labelled and re.search(
                r"(?:短语|分解|词组|phrases?|decomposition)\s*[:：为]?"
                r"[^。.;\n]{0,300}[,，][^。.;\n]{1,300}",
                value,
                re.IGNORECASE,
            ))
            return labelled and (indexed_pairs or labelled_list)
        if self.name == "encoded_string":
            return bool(
                re.search(
                    r"编码|码串|比特串|\bencoded(?:\s+string)?\b|bit\s*string",
                    value,
                    re.IGNORECASE,
                )
                and re.search(r"(?<![01])(?:[01][\s]*){4,}(?![01])", value)
            )
        if self.name.startswith("parameter_dependency_"):
            symbol = self.name[len("parameter_dependency_"):]
            return bool(re.search(
                rf"(?<![A-Za-z_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                value,
                re.IGNORECASE,
            ))
        if self.name.startswith("target_"):
            symbol = self.name[len("target_"):]
            # A requested symbol can be the first term of a compound target,
            # for example X in the distribution of X+Y.  Requiring the literal
            # fragment ``X=`` rejects a valid relation such as ``X+Y\sim N``.
            compound_relation = re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
                rf"(?:\s*[+\-*/]\s*(?:[A-Za-z][A-Za-z0-9_]*|[-+]?\d+(?:\.\d+)?)){{0,4}}"
                rf"\s*(?:=|\\sim|∼|\\in|∈)",
                value,
                re.IGNORECASE,
            )
            if compound_relation:
                return True
        if self.name == "judgement":
            return bool(re.search(
                r"是|否|正确|错误|成立|不成立|可|不可|收敛|发散|"
                r"必胜|必败|先手胜|后手胜|"
                r"位于|不在|属于|不属于|满足|不满足|等距|非等距|改变|不变|"
                r"true|false|yes|no|holds?|does\s+not|converg|diverg|"
                r"\b(?:wins?|loses?|winning|losing)\b|"
                r"lies?\s+(?:inside|within|outside)|belongs?|satisf(?:y|ies)|isometr|"
                r"changes?|unchanged|invariant|is\s+(?:not\s+)?(?:an?\s+)?solution",
                value,
                re.IGNORECASE,
            ))
        if self.name == "pole_location":
            return bool(re.search(
                r"(?:极点|奇点)[^。；;\n]{0,120}(?:位于|在|不在|属于)"
                r"[^。；;\n]{0,80}(?:围道|曲线|圆)(?:内|外|上)|"
                r"(?:围道|曲线|圆)(?:内|外|上)[^。；;\n]{0,80}(?:有|无|包含)"
                r"[^。；;\n]{0,40}(?:极点|奇点)|"
                r"\b(?:poles?|singularit(?:y|ies))\b[^.;\n]{0,120}"
                r"\b(?:lies?|is|are|falls?)\b[^.;\n]{0,60}"
                r"\b(?:inside|within|outside|on)\b[^.;\n]{0,60}"
                r"\b(?:contour|curve|circle)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "isometry_judgement":
            return bool(re.search(
                r"(?:是|为|并非|不是)[^。；;\n]{0,20}(?:一个)?等距(?:算子|映射)?|"
                r"(?:等距|非等距)(?:算子|映射)?[^。；;\n]{0,20}(?:成立|不成立|是|否)?|"
                r"\b(?:is|is not|isn't|not)\b[^.;\n]{0,30}"
                r"\b(?:an?\s+)?isometr(?:y|ic)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "invariance_judgement":
            return bool(re.search(
                r"改变|不变|保持不变|发生变化|未发生变化|"
                r"\b(?:changes?|does\s+not\s+change|unchanged|invariant|varies?)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "convergence_radius":
            return bool(re.search(
                r"(?:收敛半径|radius\s+of\s+convergence|convergence\s+radius)"
                r"[^。；;\n]{0,80}(?:=|为|是|is)\s*"
                r"(?:[-+]?\d|\\frac|\\sqrt|\\infty|∞)|"
                r"(?<![A-Za-z])R\s*=\s*(?:[-+]?\d|\\frac|\\sqrt|\\infty|∞)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "convergence_domain":
            return bool(re.search(
                r"(?:收敛域|收敛圆|domain\s+of\s+convergence|"
                r"disk\s+of\s+convergence|circle\s+of\s+convergence)"
                r"[^。；;\n]{0,120}(?:\\?[\[{]|<|≤|\\leq?|=|为|是|is)|"
                r"(?:\\?lvert|\|)\s*z\s*(?:\\?rvert|\|)\s*"
                r"(?:<|≤|\\leq?)\s*(?:[-+]?\d|\\frac|\\sqrt)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "natural_boundary_classification":
            return bool(re.search(
                r"自然边界|无法?越过[^。；;\n]{0,40}解析延拓|"
                r"不能[^。；;\n]{0,40}解析延拓|"
                r"\bnatural\s+boundary\b|"
                r"\bno\s+analytic\s+continuation\b[^.;\n]{0,80}"
                r"\b(?:arc|boundary|circle)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "irreducibility_judgement":
            return bool(re.search(
                r"不可约|(?<!不)可约|\birreducible\b|\breducible\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "factor_degree_check":
            return bool(re.search(
                r"(?:只需|需要|检查|枚举)[^。；;\n]{0,60}"
                r"(?:一次|二次|三次|次数(?:不超过|至多|小于等于)\s*\d+)\s*(?:的\s*)?(?:不可约)?因子|"
                r"(?:degree[- ]?(?:one|two|three|\d+)|degree\s*(?:at\s+most|no\s+more\s+than)\s*\d+)"
                r"[^.;\n]{0,30}(?:factor|divisor)|"
                r"(?:factor|divisor)[^.;\n]{0,30}(?:degree\s*(?:one|two|three|\d+|at\s+most\s+\d+))",
                value,
                re.IGNORECASE,
            ))
        if self.name == "equilibrium_point":
            return bool(re.search(
                r"(?:平衡点|equilibrium(?:\s+point)?)[^。；;\n]{0,50}"
                r"(?:为|是|=|is|at)\s*\$?\s*(?:\\left\s*)?[\[(]"
                r"[^\])\n]{1,80}[\])]|"
                r"(?:原点|origin)\s*(?:为|是|is)\s*(?:一个|an?\s+)?"
                r"(?:平衡点|鞍点|结点|焦点|中心|equilibrium|saddle|node|focus|center)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "stability_classification":
            return bool(re.search(
                r"渐近稳定|李雅普诺夫稳定|中性稳定|不稳定|"
                r"\basymptotically\s+stable\b|\blyapunov\s+stable\b|"
                r"\bneutrally\s+stable\b|\bunstable\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "eigenvalue_signs":
            return bool(re.search(
                r"(?:特征值|eigenvalues?)[^。；;\n]{0,100}"
                r"(?:正|负|符号|为|是|=|positive|negative|signs?|are|is)"
                r"[^。；;\n]{0,100}(?:[-+]?\d|\\lambda|λ|positive|negative|正|负)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "membership_judgement":
            return bool(re.search(
                r"(?:属于|不属于)\s*\$?\s*(?:L|l)\s*\^|"
                r"(?:\\(?:notin|in)|∉|∈)\s*(?:L|l)\s*\^|"
                r"(?:是|不是)[^。；;\n]{0,30}(?:可积|integrable)|"
                r"\b(?:belongs?|does\s+not\s+belong|is|is\s+not)\b"
                r"[^.;\n]{0,30}\bL\s*\^?\s*\{?\s*[0-9pP]+\s*\}?",
                value,
                re.IGNORECASE,
            ))
        if self.name == "integral_conclusion":
            return bool(re.search(
                r"(?:积分|积分值|反常积分)[^。；;\n]{0,50}"
                r"(?:=|为|是|等于|收敛于|发散|不存在)|"
                r"\b(?:integral|integral\s+value|improper\s+integral)\b"
                r"[^.;\n]{0,50}(?:=|is|equals?|converges?\s+to|diverges?|does\s+not\s+exist)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "second_derivatives":
            subscripted = re.findall(
                r"[A-Za-z]\s*_\s*\{?\s*([A-Za-z])\s*\1\s*\}?\s*=" ,
                value,
            )
            operator_form = re.findall(
                r"(?:\\frac\s*\{\s*\\partial\s*\^\s*2|"
                r"\\partial\s*\^\s*2|∂\s*\^?\s*2)"
                r"[^=。；;\n]{0,60}=",
                value,
                re.IGNORECASE,
            )
            return len(set(subscripted)) >= 2 or len(operator_form) >= 2
        if self.name == "harmonicity_judgement":
            return bool(re.search(
                r"(?:是|为|不是|并非)[^。；;\n]{0,12}调和|"
                r"调和(?:函数)?[^。；;\n]{0,12}(?:成立|不成立|是|否)|"
                r"\b(?:is|is\s+not|isn't)\s+(?:a\s+)?harmonic\b|"
                r"\b(?:harmonic|not\s+harmonic)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "first_second_derivatives":
            first = bool(re.search(
                r"(?:\\?gamma|γ|[A-Za-z])\s*['′]\s*\([^)]*\)\s*=|"
                r"(?:一阶导数|first\s+derivative)[^。.;\n]{0,80}(?:=|为|is)",
                value,
                re.IGNORECASE,
            ))
            second = bool(re.search(
                r"(?:\\?gamma|γ|[A-Za-z])\s*(?:''|[′']{2})\s*\([^)]*\)\s*=|"
                r"(?:二阶导数|second\s+derivative)[^。.;\n]{0,80}(?:=|为|is)",
                value,
                re.IGNORECASE,
            ))
            return first and second
        if self.name == "variance_identification":
            moment = bool(re.search(
                r"(?:E|\\mathbb\s*\{E\})\s*\[[^\]]+\]\s*(?:=|为|is)|"
                r"(?:期望|expectation)[^。.;\n]{0,80}(?:=|为|is)",
                value,
                re.IGNORECASE,
            ))
            variance = bool(re.search(
                r"(?:\\operatorname\s*\{Var\}|\\mathrm\s*\{Var\}|Var|方差|variance)"
                r"[^。.;\n]{0,80}(?:=|为|is)|"
                r"(?:=|为|is)[^。.;\n]{0,40}(?:方差|variance)",
                value,
                re.IGNORECASE,
            ))
            return moment and variance
        if self.name == "almost_everywhere_zero":
            return bool(
                re.search(r"几乎处处|a\.?e\.?|almost\s+everywhere", value, re.IGNORECASE)
                and re.search(
                    r"(?:f|函数)[^。.;\n]{0,30}(?:=|等于|为|is)\s*0|"
                    r"0[^。.;\n]{0,20}(?:几乎处处|a\.?e\.?|almost\s+everywhere)",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "almost_everywhere_limit":
            return bool(
                re.search(r"几乎处处|a\.?e\.?|almost\s+everywhere", value, re.IGNORECASE)
                and re.search(
                    r"f_?\{?n\}?\s*(?:→|\\to|converges?\s+to)\s*"
                    r"(?:0|[A-Za-z][A-Za-z0-9_{}^]*|\\[A-Za-z]+)|"
                    r"(?:极限|limit)[^。.;\n]{0,50}(?:为|是|=|is)\s*"
                    r"(?:0|[A-Za-z][A-Za-z0-9_{}^]*|\\[A-Za-z]+)",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "uniform_integrability_check":
            named = re.search(r"一致可积|uniform(?:ly)?\s+integrab", value, re.IGNORECASE)
            tail_check = re.search(
                r"\|?f_?\{?n\}?\|?\s*>\s*K|"
                r"sup\s*_?\{?n\}?[^。.;\n]{0,100}(?:\\int|∫)|"
                r"K\s*>[^。.;\n]{0,50}(?:空集|empty|=\s*0)|"
                r"(?:小集合|small\s+sets?)[^。.;\n]{0,100}(?:积分|integral)|"
                r"(?:m|\\?mu|μ)\s*\(\s*E\s*\)\s*<\s*(?:\\?delta|δ)"
                r"[^。.;\n]{0,180}(?:\\int|∫)[^。.;\n]{0,100}(?:f_?\{?n\}?|E)|"
                r"(?:\\int|∫)[^。.;\n]{0,100}(?:f_?\{?n\}?|E)"
                r"[^。.;\n]{0,180}(?:m|\\?mu|μ)\s*\(\s*E\s*\)\s*<",
                value,
                re.IGNORECASE,
            )
            return bool(named and tail_check)
        if self.name == "l1_nonconvergence":
            norm = re.search(
                r"(?:\\?\|\s*f_?\{?n\}?\s*\\?\|\s*_?\{?1\}?|"
                r"L\s*\^?\s*\{?1\}?\s*(?:范数|norm))[^。.;\n]{0,60}(?:=|为|is)",
                value,
                re.IGNORECASE,
            )
            conclusion = re.search(
                r"不趋于\s*0|不(?:在|按)[^。.;\n]{0,30}L\s*\^?\s*\{?1\}?[^。.;\n]{0,30}收敛|"
                r"does\s+not\s+(?:tend|converge)[^.;\n]{0,30}(?:0|L\s*\^?1)|"
                r"no\s+L\s*\^?1\s+convergence",
                value,
                re.IGNORECASE,
            )
            return bool(norm and conclusion)
        if self.name == "l1_limit_conclusion":
            space = r"L\s*\^?\s*\{?\s*1\s*\}?"
            explicit_limit = re.search(
                rf"(?:{space})[^。.;\n]{{0,50}}"
                r"(?:极限|收敛|不收敛|不存在|limit|converg|does\s+not\s+exist)|"
                rf"(?:极限|收敛|不收敛|不存在|limit|converg)[^。.;\n]{{0,50}}"
                rf"(?:{space})|"
                rf"f_?\{{?n\}}?\s*(?:→|\\to)\s*[^。.;\n]{{1,50}}"
                rf"(?:in|于|按)\s*{space}",
                value,
                re.IGNORECASE,
            )
            return bool(explicit_limit)
        if self.name == "uniform_convergence_scope_reason":
            local_reason = re.search(
                r"Weierstrass|M\s*判别|优级数|一致\s*Cauchy|"
                r"(?:上界|控制)[^。.;\n]{0,100}r\s*\^|"
                r"uniform\s+Cauchy|M[- ]test|major(?:ant|ize)|"
                r"bounded[^.;\n]{0,80}r\s*\^",
                value,
                re.IGNORECASE,
            )
            unbounded_limit = re.search(
                r"(?:和函数|极限函数|limit\s+function|sum\s+function)"
                r"[^。.;\n]{0,100}(?:无界|趋于(?:正)?无穷|unbounded|"
                r"tends?\s+to\s+(?:infinity|\\infty))|"
                r"(?:无界|趋于(?:正)?无穷|unbounded|"
                r"tends?\s+to\s+(?:infinity|\\infty))"
                r"[^。.;\n]{0,100}(?:和函数|极限函数|limit\s+function|sum\s+function)",
                value,
                re.IGNORECASE,
            )
            bounded_approximants = re.search(
                r"(?:每个|任一|各个)?部分和[^。.;\n]{0,80}有界|"
                r"有界[^。.;\n]{0,80}(?:部分和|S_?\{?N\}?)|"
                r"(?:each|every)\s+partial\s+sum[^.;\n]{0,80}bounded|"
                r"bounded[^.;\n]{0,80}partial\s+sums?",
                value,
                re.IGNORECASE,
            )
            direct_failure = re.search(
                r"不满足[^。.;\n]{0,50}一致\s*Cauchy|"
                r"尾(?:和|项)[^。.;\n]{0,100}(?:不趋于|不能一致|下界)|"
                r"uniform\s+Cauchy[^.;\n]{0,80}(?:fails?|not)|"
                r"tails?[^.;\n]{0,100}(?:do\s+not|fail|bounded\s+below)"
                r"[^.;\n]{0,30}uniform",
                value,
                re.IGNORECASE,
            )
            global_reason = bool(
                (unbounded_limit and bounded_approximants) or direct_failure
            )
            return bool(local_reason and global_reason)
        if self.name == "executable_insertion_step":
            insertion = re.search(
                r"插(?:入|在)|\b(?:insert|insertion)\b",
                value,
                re.IGNORECASE,
            )
            selection = re.search(
                r"(?:最小|首个|第一个|从左至右|从左到右)[^。.;\n]{0,80}"
                r"(?:指标|下标|位置|顶点|i)|"
                r"\b(?:least|smallest|first|leftmost)\b[^.;\n]{0,80}"
                r"\b(?:index|position|vertex|i)\b",
                value,
                re.IGNORECASE,
            )
            placement = re.search(
                r"(?:之前|之后|之间|最前|末尾|开头|结尾)|"
                r"\b(?:before|after|between|front|beginning|end|append|prepend)\b",
                value,
                re.IGNORECASE,
            )
            return bool(insertion and selection and placement)
        if self.name == "reasoning":
            return _has_reasoning(value)
        if self.name == "construction_object":
            return bool(re.search(r"取|令|定义|构造|例如|\b(?:take|let|define|construct|example)\b|(?<![<>!])=(?!=)|[\[{]", value, re.IGNORECASE))
        if self.name == "construction_check":
            return bool(re.search(r"满足|验证|检查|代入|成立|\b(?:satisf|verify|check|substitut|holds?)\w*\b", value, re.IGNORECASE))
        if self.name == "choice_labels":
            return bool(answer_choice_labels(value))
        if self.name.startswith("decimal_places_"):
            places = int(self.name.rsplit("_", 1)[-1])
            decimals = re.findall(r"[-+]?\d+\.(\d+)", value)
            return any(len(item) == places for item in decimals)
        if self.name == "unit":
            return any(_compact(term) in compact for alt in self.alternatives for term in alt)
        if self.name == "method_formula":
            return bool(re.search(
                r"[A-Za-z]\s*(?:_\{?\s*[nk]\s*\}?\s*\+\s*1|"
                r"\(\s*[nk]\s*\+\s*1\s*\)|\^\s*\{?\(?\s*[nk]\s*\+\s*1)|"
                r"迭代公式|迭代式|递推公式|recurrence|iteration\s+(?:formula|scheme)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "first_iteration":
            return bool(re.search(r"[xuy]_?\{?1\}?\s*=|第一次迭代|first iterate", value, re.IGNORECASE))
        if self.name == "domain_or_conditions":
            if re.search(r"定义域|条件|其中|当且仅当|subject to|domain|provided that|for .* such that", value, re.IGNORECASE):
                return True
            interval = r"[\[(]\s*[^,，\n]{1,80}\s*[,，]\s*[^,，\n]{1,80}\s*[\])]"
            return bool(re.search(
                rf"(?:最大(?:右侧|左侧)?(?:存在|解|定义)?区间|存在区间|"
                rf"maximal\s+(?:interval|interval\s+of\s+existence)|interval\s+of\s+existence)"
                rf"[^\n]{{0,24}}(?:{interval}|\$?\s*\\mathbb\s*\{{?R\}}?\s*\$?|"
                rf"(?:the\s+)?real\s+line)|"
                rf"(?:[A-Za-z]|自变量)\s*(?:\\in|∈)\s*{interval}",
                value,
                re.IGNORECASE,
            ))
        if self.name == "maximal_interval_and_one_sided_part":
            interval = re.compile(
                r"[\[(]\s*[^,，\n]{1,80}\s*[,，]\s*[^,，\n]{1,80}\s*[\])]"
            )
            intervals = {
                _compact(match.group(0)) for match in interval.finditer(value)
            }
            full_interval = bool(re.search(
                r"(?:-\s*(?:\\infty|∞)|(?:[A-Za-z]|自变量)\s*<\s*[-+]?\d)",
                value,
                re.IGNORECASE,
            ))
            one_sided = bool(re.search(
                r"向右|向左|右侧|左侧|right[- ]hand|left[- ]hand|"
                r"to\s+the\s+right|to\s+the\s+left",
                value,
                re.IGNORECASE,
            ))
            return len(intervals) >= 2 and full_interval and one_sided
        if self.name == "normal_equation":
            return bool(re.search(
                r"(?:正规方程|normal\s+equations?)[^。.;\n]{0,180}(?<![<>!])=(?!=)|"
                r"(?:X|A)\s*(?:\^|\\mathsf|\\mathrm)[^=\n]{0,100}"
                r"(?:\\widehat|hat|beta|β)[^=\n]{0,40}=",
                value,
                re.IGNORECASE,
            ))
        if self.name == "coefficient_estimate":
            estimate = (
                r"(?:\\widehat\s*\{?\s*\\?beta\s*\}?|"
                r"\\hat\s*\{?\s*\\?beta\s*\}?|(?:beta|β)\s*_?\s*(?:hat|GLS|WLS)|"
                r"(?:GLS|WLS)\s*(?:估计(?:量|值)?|estimat(?:e|or)))"
            )
            concrete = (
                r"(?:\\begin\s*\{[pbvBV]?matrix\}|\\left\s*[\[(]|"
                r"[\[(]\s*(?:[-+]?\d|\\(?:d?frac|sqrt))|"
                r"[-+]?\d|\\(?:d?frac|sqrt))"
            )
            return bool(re.search(
                rf"{estimate}[^=。.;；\n]{{0,40}}(?<![<>!])=(?!=)\s*{concrete}|"
                rf"{estimate}[^。.;；\n]{{0,40}}(?:为|是|is)\s*\$?\s*{concrete}",
                value,
                re.IGNORECASE,
            ))
        if self.name == "sturm_liouville_argument":
            return bool(
                re.search(r"Sturm[- ]Liouville|斯图姆|施图姆", value, re.IGNORECASE)
                and re.search(
                    r"特征值|特征方程|边界条件|eigenvalues?|eigenvalue\s+equation|"
                    r"boundary\s+conditions?|[A-Za-z]\s*''|\\frac\s*\{d\^?2",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "dual_certificate":
            return bool(
                re.search(r"对偶|dual", value, re.IGNORECASE)
                and re.search(
                    r"(?:[uvy]_?\{?\d*\}?\s*=)|(?:[uvy]\s*[,，]\s*[uvy])\s*=|"
                    r"(?:对偶|dual)[^。.;\n]{0,100}[\[(][^\])\n]+[\])]",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "stationary_distribution":
            return bool(re.search(
                r"(?:平稳分布|稳态分布|stationary\s+distribution)[^。.;\n]{0,80}"
                r"(?:为|是|=|is)\s*\$?\s*(?:\\left\s*)?[\[(][^\])\n]*(?:\d|\\frac)[^\])\n]*[\])]|"
                r"(?:\\pi|π|pi)\s*=\s*(?:\\left\s*)?[\[(][^\])\n]*(?:\d|\\frac)[^\])\n]*[\])]",
                value,
                re.IGNORECASE,
            ))
        if self.name == "detailed_balance_check":
            return bool(
                re.search(r"细致平衡|详细平衡|detailed\s+balance", value, re.IGNORECASE)
                and re.search(
                    r"(?:\\pi|π|pi)[^=\n]{0,30}(?:\\lambda|λ|lambda)[^=\n]{0,20}="
                    r"[^=\n]{0,50}(?:\\pi|π|pi)[^\n]{0,40}(?:\\mu|μ|mu)|"
                    r"(?:\\pi|π|pi)_?\{?\d+\}?[^=\n]{0,30}=",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "stability_function":
            return bool(re.search(
                r"(?:^|[^A-Za-z])R\s*\(\s*z\s*\)\s*=|"
                r"(?:稳定函数|stability\s+function)[^。.;\n]{0,100}(?<![<>!])=(?!=)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "stability_infinity_limit":
            return bool(re.search(
                r"(?:\\lim\s*_\s*\{?\s*z\s*(?:\\to|→)\s*"
                r"(?:\\infty|∞)\s*\}?|lim\s*\(\s*z\s*(?:->|→)\s*"
                r"(?:infinity|∞)\s*\))[^。；;\n]{0,80}"
                r"R\s*\(\s*z\s*\)[^。；;\n]{0,40}(?:=|为|is)\s*0|"
                r"R\s*\(\s*z\s*\)[^。；;\n]{0,80}(?:\\to|→|tends?\s+to)\s*0"
                r"[^。；;\n]{0,40}(?:z\s*(?:\\to|→)|as\s+z\s*(?:->|→))"
                r"[^。；;\n]{0,20}(?:\\infty|∞|infinity)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "stability_boundary_equation":
            return bool(re.search(
                r"(?:边界(?:方程)?|boundary(?:\s+equation)?|端点|endpoint|"
                r"唯一(?:正|负)?根|unique\s+(?:positive|negative\s+)?root)"
                r"[^。;；\n]{0,180}(?<![<>!])=(?!=)\s*0|"
                r"(?<![<>!])=(?!=)\s*0[^。;；\n]{0,120}"
                r"(?:边界|boundary|端点|endpoint|根|root)|"
                r"(?:[xr]\s*(?:\^|\*\*)\s*\d|[xr]\^\{\d+\})"
                r"[^。;；\n]{0,180}(?<![<>!])=(?!=)\s*0",
                value,
                re.IGNORECASE,
            ))
        if self.name == "closed_stability_interval":
            closed_interval = bool(re.search(
                r"(?:稳定(?:区间|域)|数值区间|负实轴|stability(?:\s+(?:interval|region))?|"
                r"negative\s+real\s+axis)[^。;；\n]{0,120}"
                r"\[\s*[^,，\]\n]{1,80}\s*[,，]\s*[^\]\n]{1,80}\s*\]",
                value,
                re.IGNORECASE,
            ))
            closed_chain = bool(re.search(
                r"(?:-|−)?[^。;；\n]{0,60}(?:≤|\\leq?|<=)\s*"
                r"(?:z|x|r|h\s*\\?lambda)\s*(?:≤|\\leq?|<=)\s*0",
                value,
                re.IGNORECASE,
            ))
            compact_stability_tuple = bool(
                re.search(r"R\s*\(\s*z\s*\)\s*=", value, re.IGNORECASE)
                and re.search(
                    r"\[\s*[^,，\]\n]{1,80}\s*[,，]\s*0\s*\]",
                    value,
                )
            )
            return closed_interval or closed_chain or compact_stability_tuple
        if self.name == "multistep_characteristic_equation":
            return bool(re.search(
                r"(?:特征方程|characteristic\s+equation)[^。;；\n]{0,220}"
                r"(?:\\?xi|ξ|z|r)\s*(?:\^|\*\*)\s*\{?2\}?"
                r"[^。;；\n]{0,160}(?<![<>!])=(?!=)\s*0|"
                r"(?:\([^\n]{0,80}(?:z|h\s*\\?lambda)[^\n]{0,80}\))?"
                r"(?:\\?xi|ξ)\s*(?:\^|\*\*)\s*\{?2\}?[^。;；\n]{0,160}"
                r"(?<![<>!])=(?!=)\s*0",
                value,
                re.IGNORECASE,
            ))
        if self.name == "stability_boundary_parametrization":
            return bool(re.search(
                r"z\s*\(\s*(?:\\?theta|θ)\s*\)\s*(?<![<>!])=(?!=)\s*"
                r"[^。;；\n]*(?:e\s*\^|\\exp|cos|sin)|"
                r"(?:稳定边界|边界参数式|stability\s+boundary|boundary\s+parametri[sz]ation)"
                r"[^。;；\n]{0,180}(?<![<>!])=(?!=)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "zero_stability":
            return bool(
                re.search(r"零稳定|zero[- ]stable|zero\s+stability", value, re.IGNORECASE)
                and re.search(
                    r"根条件|root\s+condition|(?:\\?xi|ξ)[^。;；\n]{0,100}"
                    r"(?:=|为|is)\s*[-+]?\d|满足|不满足|yes|no|是|否",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "method_order":
            return bool(re.search(
                r"(?:方法)?(?:阶数|精度阶|order)\s*(?:为|是|=|:|：|is)?\s*"
                r"(?:一|二|三|四|五|六|七|八|九|十|\d+)|"
                r"(?:一|二|三|四|五|六|七|八|九|十|\d+)\s*阶",
                value,
                re.IGNORECASE,
            ))
        if self.name == "a_stability_judgement":
            return bool(re.search(
                r"A\s*[- ]?稳定(?:性)?\s*(?:判断|结论)?\s*"
                r"(?:为|是|:|：|=)?\s*(?:是|否|成立|不成立)|"
                r"(?:是|为|并非|不是)\s*A\s*[- ]?稳定|"
                r"\b(?:is|is\s+not|not)\s+A[- ]stable\b|\bA[- ]stability\s*[:=]?\s*(?:yes|no)\b",
                value,
                re.IGNORECASE,
            ))
        if self.name == "iteration_matrix":
            return bool(re.search(
                r"(?:迭代矩阵|iteration\s+matrix|B_?J|T_?J)[^。;；\n]{0,80}"
                r"(?:=|为|是|is)\s*\$?\s*(?:\\left\s*[\[(])?\s*"
                r"\\begin\s*\{[pbvBV]?matrix\}|"
                r"(?:B|T)_?\{?J\}?\s*=\s*(?:\\left\s*[\[(])?\s*"
                r"\\begin\s*\{[pbvBV]?matrix\}",
                value,
                re.IGNORECASE,
            ))
        if self.name == "spectral_radius":
            return bool(re.search(
                r"(?:\\rho|ρ|rho)\s*\([^)]*\)\s*(?:=|为|是|is)\s*"
                r"(?:[-+]?\d|\\frac|\\sqrt)|"
                r"(?:谱半径|spectral\s+radius)[^。;；\n]{0,80}"
                r"(?:=|为|是|is)\s*(?:[-+]?\d|\\frac|\\sqrt)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "requested_iterates":
            first = re.search(r"[xuy]\s*\^?\s*\{?\(?(?:1)\)?\}?\s*=|[xuy]_?\{?1\}?\s*=", value)
            second = re.search(r"[xuy]\s*\^?\s*\{?\(?(?:2)\)?\}?\s*=|[xuy]_?\{?2\}?\s*=", value)
            return bool(first and second)
        if self.name == "differential_equation_substitution":
            substitution = re.search(
                r"直接(?:代回|代入)|代回(?:原)?方程|direct(?:ly)?\s+substitut|"
                r"substitut\w*\s+(?:back\s+)?into\s+the\s+(?:PDE|ODE|equation)",
                value,
                re.IGNORECASE,
            )
            derivatives = re.search(
                r"u_?\{?[tx]\}?\s*=|u_?\{?xx\}?\s*=|u_?\{?tt\}?\s*=|"
                r"y\s*'{1,2}\s*=|\\partial|d[uy]\s*/\s*d[tx]",
                value,
                re.IGNORECASE,
            )
            residual = re.search(
                r"(?:残差|residual)[^。.;\n]{0,80}(?:=|为|is)\s*"
                r"(?:[-+]?\d|\\frac|\\sqrt|[A-Za-z]|0)|"
                r"u_?\{?t\}?\s*=\s*u_?\{?xx\}?|"
                r"u_?\{?tt\}?\s*=\s*[^。.;\n]{0,80}u_?\{?xx\}?|"
                r"u_?\{?t\}?[^。.;\n]{0,180}u_?\{?x\}?"
                r"[^。.;\n]{0,180}(?<![<>!])=(?!=)[^。.;\n]{1,180}|"
                r"(?:u_?\{?t\}?|u_?\{?x\}?|u_?\{?xx\}?|y\s*'{1,2})"
                r"[^。.;\n]{0,180}=\s*0|满足(?:原)?方程|satisf(?:y|ies)\s+the\s+(?:PDE|ODE|equation)",
                value,
                re.IGNORECASE,
            )
            return bool(derivatives and residual and (substitution or residual))
        if self.name == "initial_condition_check":
            return bool(
                re.search(
                    r"(?:初值|初始条件|initial\s+(?:value|condition))|"
                    r"u\s*\(\s*x\s*[,，]\s*0\s*\)|y\s*\(\s*0\s*\)",
                    value,
                    re.IGNORECASE,
                )
                and re.search(
                    r"(?:t\s*=\s*0|u\s*\(\s*x\s*[,，]\s*0\s*\)|y\s*\(\s*0\s*\))"
                    r"[^。.;\n]{0,120}(?:=|得到|给出|equals?)",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "boundary_value_or_exponential_martingale":
            boundary_value = bool(
                re.search(r"u\s*''|u\s*\\?''|\\frac\s*\{?1\}?\s*\{?2\}?\s*u", value)
                and re.search(r"u\s*\([^)]*\)\s*=\s*1", value)
                and re.search(r"infty|∞|无穷|有界|bounded|decay", value, re.IGNORECASE)
            )
            exponential_martingale = bool(re.search(
                r"exp\s*\([^\n]{0,100}B_?\{?t\}?[^\n]{0,100}t\)|"
                r"e\s*\^\s*\{[^\n]{0,120}B_?\{?t\}?[^\n]{0,120}t[^\n]*\}"
                r"[^。.;\n]{0,80}(?:鞅|martingale)",
                value,
                re.IGNORECASE,
            ))
            return boundary_value or exponential_martingale
        if self.name == "count_conclusion":
            stripped = value.strip().strip("$").strip()
            boxed = re.fullmatch(r"\\boxed\s*\{(.+)\}\s*[。.]?", stripped, re.DOTALL)
            if boxed:
                stripped = boxed.group(1).strip()
            bare_expression = bool(
                1 <= len(stripped) <= 180
                and re.fullmatch(
                    r"[\s0-9A-Za-z_{}()[\],.!+\-*/^=\\]+",
                    stripped,
                )
                and re.search(r"\d|\\(?:frac|binom|sum|prod)|\^", stripped)
            )
            standalone_expression = False
            for line in value.splitlines():
                clean_line = line.strip().strip("$").strip()
                clean_box = re.fullmatch(
                    r"\\boxed\s*\{(.+)\}\s*[。.]?",
                    clean_line,
                    re.DOTALL,
                )
                if clean_box:
                    clean_line = clean_box.group(1).strip()
                if (
                    1 <= len(clean_line) <= 240
                    and re.fullmatch(
                        r"[\s0-9A-Za-z_{}()[\],.!+\-*/^=\\]+",
                        clean_line,
                    )
                    and re.search(r"\d|\\(?:frac|binom|sum|prod)|\^", clean_line)
                    and (
                        "=" in clean_line
                        or re.fullmatch(r"[-+]?\d+(?:/\d+)?", clean_line)
                    )
                ):
                    standalone_expression = True
                    break
            explicit_total = bool(re.search(
                r"(?:总数|总计|合计|共计|计数结果|答案|结论|个数|数目|数量)"
                r"[^。;；\n]{0,50}(?:为|是|等于|=|:|：)?\s*"
                r"\$?\s*(?:\\boxed|\\frac|\\binom|[-+]?\d)|"
                r"(?:共有|共计|共|恰有|有)\s*(?:\*{0,2})?"
                r"\$?\s*(?:\\boxed|\\frac|\\binom|[-+]?\d)|"
                r"\b(?:the\s+)?(?:final(?:\s+answer)?|conclusion|total(?:\s+number)?|number|count|answer)\b"
                r"[^.;\n]{0,60}(?:is|equals|=|:)\s*\$?\s*(?:\\boxed|[-+]?\d)|"
                r"\bthere\s+(?:are|is)\s+(?:exactly\s+)?\$?\s*(?:\\boxed|[-+]?\d)",
                value,
                re.IGNORECASE,
            ))
            return bare_expression or standalone_expression or explicit_total
        if self.name == "quadrature_nodes":
            return bool(re.search(
                r"(?:节点|nodes?)\s*(?:为|是|are|=)\s*"
                r"\$?\s*\([^$\n]{1,500}\)\s*\$?|"
                r"(?:节点|nodes?)[^。;；\n]{0,140}"
                r"(?:x\s*_?\s*\{?[12]\}?|±|\\pm)[^。;；\n]{0,80}"
                r"(?:=|为|是|are)|"
                r"x\s*_?\s*\{?1\}?\s*=.*x\s*_?\s*\{?2\}?\s*=",
                value,
                re.IGNORECASE | re.DOTALL,
            ))
        if self.name == "quadrature_weights":
            return bool(re.search(
                r"(?:权重|weights?)\s*(?:为|是|are|=)\s*"
                r"\$?\s*\([^$\n]{1,500}\)\s*\$?|"
                r"(?:权重|weights?)[^。;；\n]{0,140}"
                r"(?:w\s*_?\s*\{?[12]\}?|均为|都为|both)"
                r"[^。;；\n]{0,80}(?:=|为|是|are)?\s*(?:[-+]?\d|\\frac)|"
                r"w\s*_?\s*\{?1\}?\s*=.*w\s*_?\s*\{?2\}?\s*=|"
                r"(?:a|A\s*_?\s*\{?1\}?|\\omega\s*_?\s*\{?1\}?)\s*=.*"
                r"(?:b|A\s*_?\s*\{?2\}?|\\omega\s*_?\s*\{?2\}?)\s*=",
                value,
                re.IGNORECASE | re.DOTALL,
            ))
        if self.name == "quadrature_value":
            return bool(re.search(
                r"(?:^|[^A-Za-z])(?:Q|[STM]\s*_?\s*\{?(?:n|\d+)\}?)"
                r"\s*(?:=|为|是)\s*(?:[-+]?\d|\\frac|\\sqrt)|"
                r"(?:求积值|求积结果|quadrature\s+value|quadrature\s+result|"
                r"quadrature\s+approximation)[^。;；\n]{0,60}"
                r"(?:=|为|是|is)\s*(?:[-+]?\d|\\frac|\\sqrt)|"
                r"(?:\\int|∫|积分|integral)[^。;；\n]{0,120}"
                r"(?:≈|\\approx|约为|近似(?:为)?|is\s+approximately|approximately\s+equals?)"
                r"\s*(?:[-+]?\d|\\frac|\\sqrt)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "quadrature_error":
            return bool(re.search(
                r"(?:I\s*-\s*Q|Q\s*-\s*I)\s*(?:=|为|是)\s*"
                r"(?:[-+]?\d|\\frac|\\sqrt)|"
                r"(?:精确误差|误差|exact\s+error|quadrature\s+error)"
                r"[^。;；\n]{0,80}(?:I\s*-\s*Q|Q\s*-\s*I)?"
                r"[^。;；\n]{0,30}(?:=|为|是|is)\s*(?:[-+]?\d|\\frac|\\sqrt)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "curvature_function":
            named_function = re.search(
                r"K\s*\(\s*[^)]*[A-Za-z][^)]*\)\s*(?:=|为|是)\s*\S+",
                value,
                re.IGNORECASE,
            )
            variable_rhs = re.search(
                r"(?:曲率函数|curvature\s+function)[^。;；\n]{0,80}"
                r"(?:=|为|是|is)\s*\S+|"
                r"(?:^|[。;；\n])\s*K\s*=\s*[^。;；\n]{0,140}"
                r"(?:[xyuvrs]|\\rho|\\theta)",
                value,
                re.IGNORECASE,
            )
            return bool(named_function or variable_rhs)
        if self.name == "curvature_point_value":
            return bool(re.search(
                r"K\s*\(\s*(?:[-+]?\d+(?:\.\d+)?\s*[,，]\s*)+"
                r"[-+]?\d+(?:\.\d+)?\s*\)\s*(?:=|为|是)\s*\S+|"
                r"(?:原点|指定点|at\s+the\s+(?:origin|specified\s+point))"
                r"[^。;；\n]{0,100}(?:曲率|curvature|K)"
                r"[^。;；\n]{0,40}(?:=|为|是|is)\s*\S+",
                value,
                re.IGNORECASE,
            ))
        if self.name == "dual_optimality_check":
            feasibility = re.search(
                r"对偶[^。;；\n]{0,100}可行|dual[^.;\n]{0,100}feasible|"
                r"(?:2u\+v|u\+2v|A\^?T\s*[A-Za-z])"
                r"[^。;；\n]{0,120}(?:≤|\\leq?|<=)",
                value,
                re.IGNORECASE,
            )
            equal_objectives = re.search(
                r"(?:原|主|primal)[^。;；\n]{0,100}(?:目标值|objective)"
                r"[^。;；\n]{0,100}(?:相等|相同|equal|same)|"
                r"(?:对偶|dual)[^。;；\n]{0,100}(?:目标值|objective)"
                r"[^。;；\n]{0,100}(?:相等|相同|equal|same)|"
                r"强对偶|strong\s+duality|"
                r"(?:原|主|primal)[^=。;；\n]{0,80}="
                r"[^=。;；\n]{0,80}(?:对偶|dual)",
                value,
                re.IGNORECASE,
            )
            return bool(feasibility and equal_objectives)
        if self.name == "series_sum_function":
            return bool(re.search(
                r"(?:和函数|sum\s+function)[^。;；\n]{0,80}"
                r"(?:=|为|是|is)\s*\S+|"
                r"S\s*\(\s*x\s*\)\s*(?:=|为|是)\s*\S+",
                value,
                re.IGNORECASE,
            ))
        if self.name == "local_uniform_convergence":
            return bool(
                re.search(r"\[\s*0\s*[,，]\s*r\s*[\])]", value, re.IGNORECASE)
                and re.search(r"一致收敛|uniform(?:ly)?\s+conver", value, re.IGNORECASE)
            )
        if self.name == "global_nonuniform_convergence":
            return bool(
                re.search(r"\[\s*0\s*[,，]\s*1\s*\)", value)
                and re.search(
                    r"不一致收敛|不是一致收敛|并非一致收敛|"
                    r"(?:not|does\s+not)\s+(?:converge\s+)?uniform(?:ly)?",
                    value,
                    re.IGNORECASE,
                )
            )
        if self.name == "operator_norm":
            return bool(re.search(
                r"\\?\|\s*T\s*\\?\|\s*(?:=|为|是|is)\s*\S|"
                r"(?:算子范数|operator\s+norm)[^。;；\n]{0,80}(?:=|为|是|is)\s*\S",
                value,
                re.IGNORECASE,
            ))
        if self.name == "operator_spectrum":
            return bool(re.search(
                r"(?:\\sigma|σ)\s*\(\s*T\s*\)\s*(?:=|为|是|is)\s*\S|"
                r"(?:^|[^点本])谱(?:集)?[^。;；\n]{0,70}(?:=|为|是|is)\s*\S|"
                r"\bspectrum\b[^.;\n]{0,70}(?:=|is)\s*\S",
                value,
                re.IGNORECASE,
            ))
        if self.name == "point_spectrum":
            return bool(re.search(
                r"(?:\\sigma\s*_?\s*\{?p\}?|σ\s*_?p)\s*\(\s*T\s*\)\s*"
                r"(?:=|为|是|is)\s*(?:\\varnothing|\\emptyset|∅|空集|\S+)|"
                r"(?:点谱|point\s+spectrum)[^。;；\n]{0,70}(?:=|为|是|is)\s*\S",
                value,
                re.IGNORECASE,
            ))
        if self.name == "jordan_blocks":
            return bool(re.search(
                r"(?:Jordan|若尔当)(?:\s*标准形|\s*块)?[^。;；\n]{0,120}"
                r"(?:大小|尺寸|sizes?|blocks?)[^。;；\n]{0,40}(?:为|是|=|:|：)"
                r"[^。;；\n]*\d|"
                r"(?:块大小|block\s+sizes?)[^。;；\n]{0,40}(?:为|是|=|:|：)[^。;；\n]*\d",
                value,
                re.IGNORECASE,
            ))
        if self.name == "operator_rank":
            return bool(re.search(
                r"(?:\\operatorname\s*\{rank\}|\\?rank|秩)\s*(?:\([^)]*\)|[A-Za-z])?"
                r"\s*(?:=|为|是|is)\s*\d+",
                value,
                re.IGNORECASE,
            ))
        if self.name == "minimal_polynomial":
            return bool(re.search(
                r"(?:最小多项式|minimal\s+polynomial)[^。;；\n]{0,80}"
                r"(?:=|为|是|is|:|：)\s*\$?\s*[^。;；\n]*(?:\^|\*\*)",
                value,
                re.IGNORECASE,
            ))
        if self.name == "smith_normal_form":
            return bool(re.search(
                r"(?:Smith|SNF|史密斯)[^。.;；\n]{0,100}(?:=|为|是|is)\s*\$?\s*"
                r"(?:\\begin\s*\{[pbvBV]?matrix\}|\\left\s*[\[(]|[\[(])|"
                r"\\operatorname\s*\{SNF\}\s*\([^)]*\)\s*=\s*"
                r"(?:\\begin\s*\{[pbvBV]?matrix\}|\\left\s*[\[(])",
                value,
                re.IGNORECASE,
            ))
        if self.name == "cokernel_structure":
            return bool(re.search(
                r"(?:\\operatorname\s*\{coker\}|\\?coker|余核)[^。.;；\n]{0,100}"
                r"(?:\\cong|≅|=|为|是|is)\s*\$?\s*"
                r"(?:0|\\mathbb\s*\{?Z\}?|\\mathbf\s*\{?Z\}?|(?:^|[^A-Za-z])Z(?:/|\^))",
                value,
                re.IGNORECASE,
            ))
        if self.name == "wasserstein_squared_value":
            return bool(re.search(
                r"(?:W\s*_?\s*\{?2\}?\s*\^\s*\{?2\}?|W₂²)\s*"
                r"(?:\([^)]*\))?\s*(?:=|为|是|is)\s*\$?\s*"
                r"(?:[-+]?\d|\\(?:d?frac|sqrt)|[A-Za-z][A-Za-z0-9_]*\s*[+\-])",
                value,
                re.IGNORECASE,
            ))
        if self.name == "optimal_transport_map":
            return bool(re.search(
                r"(?:最优传输映射|optimal\s+transport\s+map)[^。.;；\n]{0,80}"
                r"(?:为|是|=|is)\s*\$?\s*(?:T\s*\(\s*x\s*\)|[A-Za-z])\s*=|"
                r"T\s*\(\s*x\s*\)\s*=\s*[^。.;；\n]+",
                value,
                re.IGNORECASE,
            ))
        if self.name == "umvu_estimator":
            return bool(re.search(
                r"(?:(?:\\widehat|\\hat)\s*(?:\{[^}]+\}|\\?[A-Za-z]+|[A-Za-zα-ωΑ-Ω])|"
                r"(?:UMVU|一致最小方差无偏)[^。.;；\n]{0,60}(?:估计量|estimator))"
                r"[^=。.;；\n]{0,50}(?<![<>!])=(?!=)\s*\S|"
                r"(?:UMVU|一致最小方差无偏)[^。.;；\n]{0,80}"
                r"(?:估计量|estimator)[^。.;；\n]{0,20}(?:为|是|is)\s*\$?\s*\S+",
                value,
                re.IGNORECASE,
            ))
        if self.name == "exact_and_approximate":
            exact = bool(re.search(r"精确|exact|\\frac|\\sqrt|\\pi", value, re.IGNORECASE))
            approximate = bool(re.search(r"≈|约为|近似|approx", value, re.IGNORECASE) or re.search(r"\d+\.\d+", value))
            return exact and approximate
        if self.name == "l1_norm_check":
            return bool(re.search(
                r"(?:\\(?:lVert|Vert|lvert)\s*[^\n]{1,120}"
                r"\\(?:rVert|Vert|rvert)\s*_?\{?\s*1\s*\}?|"
                r"\\?\|\s*[^\n]{1,120}\\?\|\s*_?\{?\s*1\s*\}?|"
                r"L\s*\^?\s*\{?\s*1\s*\}?\s*(?:范数|norm)|"
                r"(?:范数|norm)\s*(?:为|是|=|equals?|is)|"
                r"\\int[^\n]{0,180}\\?(?:lvert|\|)[^\n]{1,120}"
                r"\\?(?:rvert|\|)[^=\n]{0,80}=)"
                r"[^。.;\n]{0,160}(?:=|为|是|equals?|is|\\to|→)\s*"
                r"(?:[-+]?\d|\\frac|\\infty|∞|[A-Za-z])",
                value,
                re.IGNORECASE,
            ))
        if self.name.startswith("support_anchor_"):
            return any(
                _support_anchor_matches(term, value)
                for alternative in self.alternatives
                for term in alternative
            )
        if not self.alternatives:
            return True
        for alternative in self.alternatives:
            if all(_compact(term) in compact or term.casefold() in lowered for term in alternative):
                return True
        return False


@dataclass(frozen=True)
class Goal:
    id: str
    instruction: str
    answer_shape: str
    kind: str
    required_terms: tuple[str, ...] = ()
    requirements: tuple[Requirement, ...] = ()

    @property
    def result_requirements(self) -> tuple[Requirement, ...]:
        return tuple(item for item in self.requirements if item.category == "result")

    @property
    def support_requirements(self) -> tuple[Requirement, ...]:
        return tuple(item for item in self.requirements if item.category == "support")

    @property
    def format_requirements(self) -> tuple[Requirement, ...]:
        return tuple(item for item in self.requirements if item.category == "format")


@dataclass(frozen=True)
class AnswerFrame:
    style: str
    subject: str = ""
    predicate: str = ""
    unit: str = ""
    question_kind: str = "math"

    def trace_content(self) -> dict:
        return {
            "style": self.style,
            "subject": self.subject,
            "predicate": self.predicate,
            "unit": self.unit,
            "question_kind": self.question_kind,
        }


@dataclass(frozen=True)
class AnswerPart:
    id: str
    description: str
    category: str
    strict: bool = False
    unit: str = ""

    def trace_content(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "strict": self.strict,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class AnswerContract:
    language: str
    mode: str
    wrapper: str
    parts: tuple[AnswerPart, ...]
    explicit_support_requirements: tuple[str, ...] = ()
    result_kind: str = "expression"
    unit: str = ""

    @property
    def support_requirements(self) -> tuple[str, ...]:
        return self.explicit_support_requirements

    def shape(self) -> str:
        return self.result_kind

    def trace_content(self) -> dict:
        return {
            "language": self.language,
            "mode": self.mode,
            "wrapper": self.wrapper or "none",
            "result_kind": self.result_kind,
            "unit": self.unit,
            "parts": [item.trace_content() for item in self.parts],
            "support_requirements": list(self.explicit_support_requirements),
        }


@dataclass(frozen=True)
class ProblemSpec:
    profile: ProblemProfile
    semantics: StatementSemantics
    goals: tuple[Goal, ...]
    constraints: tuple[str, ...]
    risk_flags: tuple[str, ...]
    primary_method: str
    alternative_method: str
    answer_frame: AnswerFrame
    tool_can_answer_whole: bool
    risk_score: int
    verification_required: bool
    answer_contract: AnswerContract
    problem_text: str = ""

    def trace_content(self) -> dict:
        return {
            "profile": self.profile.trace_content(),
            "semantics": self.semantics.trace_content(),
            "goal_count": len(self.goals),
            "goals": [{
                "id": goal.id,
                "kind": goal.kind,
                "instruction": goal.instruction[:500],
                "requirements": [
                    {"name": item.name, "category": item.category, "strict": item.strict}
                    for item in goal.requirements
                ],
            } for goal in self.goals],
            "constraints": list(self.constraints),
            "risk_flags": list(self.risk_flags),
            "risk_score": self.risk_score,
            "verification_required": self.verification_required,
            "primary_method": self.primary_method,
            "alternative_method": self.alternative_method,
            "answer_frame": self.answer_frame.trace_content(),
            "answer_contract": self.answer_contract.trace_content(),
            "tool_can_answer_whole": self.tool_can_answer_whole,
        }


SolveBlueprint = ProblemSpec


_TRAILING_FORMAT = tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
    r"\s*(?:remember\s+to\s+|please\s+)?(?:put|place|write)\s+(?:your\s+)?final\s+answer\s+(?:within|inside|in)\s+\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
    r"\s*(?:remember\s+to\s+|please\s+)?(?:show|give|provide|write)\s+"
    r"(?:your\s+|the\s+)?final\s+answer\s+(?:within|inside|in|using)\s+"
    r"\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
    r"\s*(?:your\s+|the\s+)?final\s+answer\s+(?:should|must)\s+be\s+"
    r"(?:written|placed|put)\s+(?:within|inside|in|using)\s+"
    r"\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
    r"\s*(?:final\s+answer|answer)\s*[:：]?\s*(?:within|inside|in|using)?\s*"
    r"\\boxed\s*\{\s*\}\s*[.!。]?\s*$",
    r"\s*(?:请|务必)?(?:将)?(?:最终)?答案(?:写|填|放|置)(?:在|于)?\s*(?:方框|框内|\\boxed\s*\{\s*\})(?:中|内)?\s*[。.!]?\s*$",
    r"\s*(?:请|务必)?(?:用|使用)\s*\\boxed\s*\{\s*\}\s*"
    r"(?:给出|写出|填写)?(?:最终)?答案\s*[。.!]?\s*$",
))


def _strip_trailing_answer_instructions(text: str) -> str:
    value = str(text or "").strip()
    previous = None
    while value and value != previous:
        previous = value
        for pattern in _TRAILING_FORMAT:
            value = pattern.sub("", value).strip()
    return value or str(text or "").strip()


def _restore_json_latex_controls(text: str) -> str:
    """Repair common LaTeX commands damaged by JSON/control decoding."""
    value = str(text or "")
    replacements = (
        ("\x08ar", r"\bar"),
        ("\x08egin", r"\begin"),
        ("\x08eta", r"\beta"),
        ("\x08inom", r"\binom"),
        ("\x08oldsymbol", r"\boldsymbol"),
        ("\x0crac", r"\frac"),
        ("\x0corall", r"\forall"),
        ("\theta", r"\theta"),
        ("\text", r"\text"),
        ("\times", r"\times"),
        ("\tan", r"\tan"),
        ("\tau", r"\tau"),
        ("\to", r"\to"),
        ("\rho", r"\rho"),
        ("\right", r"\right"),
        ("\rangle", r"\rangle"),
        ("\x0barphi", r"\varphi"),
        ("\x0bec", r"\vec"),
    )
    for corrupted, repaired in replacements:
        value = value.replace(corrupted, repaired)
    return value


def build_problem_spec(problem: str) -> ProblemSpec:
    original = _restore_json_latex_controls(str(problem or "")).strip()
    text = _strip_trailing_answer_instructions(original)
    profile = classify_profile(text)
    target = (
        choice_stem(text)
        if profile.answer_shape == "choice"
        else extract_target_clause(text) or text
    )
    semantics = extract_statement_semantics(
        text,
        target,
        subject_confidence=profile.subject_confidence,
    )
    if profile.answer_shape == "choice":
        stem_semantics = extract_statement_semantics(
            target,
            target,
            subject_confidence=profile.subject_confidence,
        )
        semantics = replace(
            semantics,
            requested_methods=stem_semantics.requested_methods,
            named_theorems=stem_semantics.named_theorems,
        )
    requirements = _requirements(text, target, profile)
    kind = _goal_kind(profile)
    goal_instruction = (
        text
        if (
            profile.task_kind in {"proof", "derivation", "explanation"}
            or any(item.category == "support" for item in requirements)
        )
        else target
    )
    goal = Goal(
        "g1",
        goal_instruction[:1800],
        profile.answer_shape,
        kind,
        (),
        tuple(requirements),
    )
    constraints = tuple(dict.fromkeys((*_constraints(text), *semantics.domains)))
    risks = tuple(dict.fromkeys((
        *_risks(text, profile, requirements, constraints),
        *(f"semantic_{flag}" for flag in semantics.ambiguity_flags),
    )))
    score = min(8, _risk_score(text, profile, risks))
    primary, alternative = _methods(profile, semantics)
    frame = _answer_frame(text, target, profile, requirements)
    mode = "proof" if profile.task_kind in {"proof", "derivation", "explanation"} else (
        "answer_with_support" if any(item.category == "support" for item in requirements) else "answer_only"
    )
    wrapper = "boxed" if re.search(r"\\boxed\s*\{\s*\}|方框|框内", original, re.IGNORECASE) else ""
    parts = tuple(
        AnswerPart(item.name, _requirement_description(item, profile.language), item.category, item.strict)
        for item in requirements
    )
    support = tuple(item.name for item in requirements if item.category == "support")
    contract = AnswerContract(
        profile.language,
        mode,
        wrapper,
        parts,
        support,
        profile.result_kind,
        frame.unit,
    )
    tool_whole = _tool_whole_possible(text, profile, requirements)
    return ProblemSpec(
        profile=profile,
        semantics=semantics,
        goals=(goal,),
        constraints=constraints,
        risk_flags=risks,
        primary_method=primary,
        alternative_method=alternative,
        answer_frame=frame,
        tool_can_answer_whole=tool_whole,
        risk_score=score,
        verification_required=score >= 3,
        answer_contract=contract,
        problem_text=text,
    )


def _requirements(text: str, target: str, profile: ProblemProfile) -> list[Requirement]:
    items: list[Requirement] = [Requirement("result_present", strict=True)]
    shape = profile.answer_shape
    if shape in {"number", "count", "probability"}:
        items.append(Requirement("numeric_result", strict=True))
    if shape == "count" or re.search(
        r"(?:求|计算|确定|统计)[^。！？!?\n]{0,100}"
        r"(?:总数|总计|数目|数量|个数|满射数|着色数|方案数|选法数)|"
        r"\b(?:find|compute|determine)\s+(?:the\s+)?(?:total\s+)?"
        r"(?:number|count|cardinality)\s+of\b|\bhow\s+many\b|"
        r"\bcount\s+(?:the\s+)?(?:number\s+of\s+)?",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("count_conclusion", strict=True))
    if shape == "choice":
        items.append(Requirement("choice_labels", strict=True))
    explicit_judgement_request = bool(re.search(
        r"是否|能否|可否|判断[^。；;\n]{0,120}(?:成立|正确|错误|收敛|可约|调和)|"
        r"\b(?:whether|determine\s+whether|decide\s+if|true\s+or\s+false|"
        r"is\s+it\s+true)\b",
        target,
        re.IGNORECASE,
    ))
    if shape == "truth" or (
        explicit_judgement_request
        and shape not in {"expression", "number", "count", "roots", "interval", "matrix"}
    ):
        items.append(Requirement("judgement", strict=True))
    contour_pole_request = bool(re.search(
        r"\\oint|∮|围道积分|复积分|contour\s+integral",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"(?:说明|指出|判断|确定|核对)[^。；;\n]{0,120}(?:极点|奇点)"
        r"[^。；;\n]{0,80}(?:是否|位于|在)[^。；;\n]{0,60}(?:围道|曲线|圆)(?:内|外|上)?|"
        r"(?:极点|奇点)[^。；;\n]{0,100}(?:是否|位于|在)"
        r"[^。；;\n]{0,60}(?:围道|曲线|圆)(?:内|外|上)?|"
        r"\b(?:state|determine|decide|check|explain)\b[^.;\n]{0,120}"
        r"\b(?:pole|singularity)\b[^.;\n]{0,100}"
        r"\b(?:inside|within|outside|on)\b[^.;\n]{0,60}\b(?:contour|curve|circle)\b|"
        r"\b(?:pole|singularity)\b[^.;\n]{0,100}\b(?:whether|lies?|is)\b"
        r"[^.;\n]{0,60}\b(?:inside|within|outside|on)\b",
        text,
        re.IGNORECASE,
    ))
    if contour_pole_request:
        items.extend((
            Requirement("judgement", strict=True, category="support"),
            Requirement("pole_location", strict=True, category="support"),
        ))
    operator_isometry_request = bool(re.search(
        r"算子|映射|operator|mapping",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"是否(?:为)?等距|是不是等距|判断[^。；;\n]{0,60}等距|"
        r"\b(?:whether|determine\s+whether|decide\s+if)\b"
        r"[^.;\n]{0,80}\bisometr(?:y|ic)\b",
        text,
        re.IGNORECASE,
    ))
    if operator_isometry_request:
        items.append(Requirement(
            "isometry_judgement", strict=True, category="support"
        ))
    invariance_request = bool(re.search(
        r"是否改变|是否变化|保持不变|改变吗|"
        r"\b(?:whether|determine\s+whether)\b[^.;\n]{0,100}"
        r"\b(?:changes?|remains?\s+unchanged|is\s+invariant)\b",
        text,
        re.IGNORECASE,
    ))
    if invariance_request:
        items.append(Requirement("invariance_judgement", strict=True))
    irreducibility_question = bool(re.search(
        r"(?:判断|确定|检验|判定)[^。；;\n]{0,160}是否(?:为)?不可约|"
        r"\b(?:determine|decide|check)\s+whether\b[^.\n]{0,160}"
        r"\bis\s+(?:an?\s+)?irreducible\b|"
        r"\bis\b[^?.\n]{1,160}\birreducible\b",
        target,
        re.IGNORECASE,
    ))
    if irreducibility_question:
        items.append(Requirement("irreducibility_judgement", strict=True))
        if re.search(
            r"只需[^。；;\n]{0,60}(?:检查|考虑)[^。；;\n]{0,60}(?:次数|因子)|"
            r"说明[^。；;\n]{0,80}(?:何种|哪些|什么|次数)[^。；;\n]{0,40}因子|"
            r"(?:which|what)\s+(?:factor\s+)?degrees?[^.\n]{0,80}"
            r"(?:check|consider)|(?:suffices?|enough)[^.\n]{0,80}"
            r"degree[^.\n]{0,30}factors?",
            target,
            re.IGNORECASE,
        ):
            items.append(Requirement(
                "factor_degree_check", strict=True, category="support"
            ))
    membership_and_integral = bool(re.search(
        r"(?:是否|能否)?(?:属于|在)\s*\$?\s*L\s*\^?\s*\{?\s*[0-9pP]+\s*\}?|"
        r"\b(?:belongs?|membership)\b[^.;\n]{0,40}\bL\s*\^?\s*\{?\s*[0-9pP]+\s*\}?",
        target,
        re.IGNORECASE,
    )) and bool(re.search(
        r"(?:并|且|同时|还)[^。；;\n]{0,24}(?:计算|求)(?:其|该)?积分|"
        r"\band\s+(?:also\s+)?(?:compute|evaluate|find)[^.\n]{0,30}\bintegral\b",
        target,
        re.IGNORECASE,
    ))
    if membership_and_integral:
        items.extend((
            Requirement("membership_judgement", strict=True),
            Requirement("integral_conclusion", strict=True),
        ))
    harmonic_second_derivatives = bool(re.search(
        r"是否调和|是不是调和|判断[^。；;\n]{0,50}调和|"
        r"\b(?:whether|determine\s+whether|is)\b[^.\n]{0,60}\bharmonic\b",
        target,
        re.IGNORECASE,
    )) and bool(re.search(
        r"二阶(?:求导|导数|偏导)|求[^。；;\n]{0,30}二阶(?:偏)?导数|"
        r"\b(?:second(?:-order)?\s+(?:partial\s+)?derivatives?|differentiate\s+twice)\b",
        target,
        re.IGNORECASE,
    ))
    if harmonic_second_derivatives:
        items.extend((
            Requirement("second_derivatives", strict=True),
            Requirement("harmonicity_judgement", strict=True),
        ))
    planar_curve_derivatives = bool(re.search(
        r"(?:平面)?曲线|参数曲线|curve\s+(?:\\?gamma|gamma)|plane\s+curve",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"一阶(?:导数|求导)[^。；;\n]{0,80}二阶(?:导数|求导)|"
        r"二阶(?:导数|求导)[^。；;\n]{0,80}一阶(?:导数|求导)|"
        r"一阶\s*(?:和|与|及、?)\s*二阶(?:导数|求导)|"
        r"first\s+and\s+second\s+derivatives?|"
        r"first\s+derivative[^.;\n]{0,80}second\s+derivative",
        text,
        re.IGNORECASE,
    ))
    if planar_curve_derivatives:
        items.append(Requirement(
            "first_second_derivatives", strict=True, category="support"
        ))
    variance_identification = bool(re.search(
        r"(?:E|\\mathbb\s*\{E\})\s*\[\s*\([^\]]+\)\s*\^\s*2\s*\]|"
        r"(?:中心矩|centered\s+(?:second\s+)?moment)",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"识别(?:为)?(?:其)?方差|由定义[^。；;\n]{0,40}方差|"
        r"identify[^.;\n]{0,60}(?:as\s+)?(?:the\s+)?variance|"
        r"recognize[^.;\n]{0,60}(?:as\s+)?(?:the\s+)?variance",
        text,
        re.IGNORECASE,
    ))
    if variance_identification:
        items.append(Requirement("variance_identification", strict=True))
    zero_integral_conclusion = bool(re.search(
        r"(?:f|函数)[^。；;\n]{0,30}(?:>=|≥|\\geq?)\s*0|"
        r"nonnegative[^.;\n]{0,40}(?:function|f)",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"(?:\\int|∫)[^=。；;\n]{0,80}(?:=|等于|为)\s*0|"
        r"积分[^。；;，,\n]{0,30}(?:=|等于|为)\s*0",
        text,
        re.IGNORECASE,
    )) and bool(re.search(
        r"写出结论|最终结论|由此得出|state\s+the\s+conclusion|conclude",
        text,
        re.IGNORECASE,
    ))
    if zero_integral_conclusion:
        items.append(Requirement("almost_everywhere_zero", strict=True))
    diagonal_stability = bool(re.search(
        r"线性系统|linear\s+system", text, re.IGNORECASE
    )) and bool(re.search(
        r"平衡点|稳定(?:性|类型)?|equilibri|stability|stable|unstable",
        target,
        re.IGNORECASE,
    ))
    if diagonal_stability:
        if re.search(r"平衡点|equilibrium", target, re.IGNORECASE):
            items.append(Requirement("equilibrium_point", strict=True))
        if re.search(r"稳定(?:性|类型)?|stability|stable|unstable", target, re.IGNORECASE):
            items.append(Requirement("stability_classification", strict=True))
        if re.search(r"特征值|eigenvalues?", target, re.IGNORECASE):
            items.append(Requirement("eigenvalue_signs", strict=True))
    count_of_all = bool(re.search(
        r"(?:求|计算|确定|统计)[^。！？!?\n]{0,220}(?:所有|全部)"
        r"[^。！？!?\n]{0,160}(?:个数|数目|数量|总数|多少)|"
        r"\b(?:number|count)\s+of\s+(?:all\s+)?",
        target,
        re.IGNORECASE,
    )) and not bool(re.search(
        r"列出|写出|逐一给出|枚举|\b(?:list|enumerate)\b",
        target,
        re.IGNORECASE,
    ))
    if (shape == "roots" or re.search(
        r"全部解|所有解|all solutions?|all roots?|"
        r"(?:求|确定|找出|列出|分类)[^。！？!?\n]{0,180}(?:所有|全部)"
        r"[^。！？!?\n]{0,100}(?:整数|实数|复数|有理数|多项式|函数|映射|矩阵|对象)|"
        r"\b(?:find|determine|list|classify)\s+all\s+"
        r"(?:(?:positive|negative|nonnegative|nonzero|real|complex|rational|integer-valued)\s+)*"
        r"(?:integers?|numbers?|polynomials?|functions?|maps?|matrices|objects?)\b",
        target,
        re.IGNORECASE,
    )) and not count_of_all and not re.search(
        r"(?:全部|所有)\s*(?:Jordan|若尔当)\s*块|all\s+(?:the\s+)?Jordan\s+blocks?",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("all_solutions", strict=True))
    exhaustive_target = bool(re.search(
        r"(?:求|确定|找出|列出|分类)[^。！？!?\n]{0,180}(?:所有可能|全部可能|所有|全部)"
        r"[^。！？!?\n]{0,100}(?:取值|参数|数对|有序对|元组|配置|构型|走法|步骤|集合|点)|"
        r"(?:哪些|何种|何值|什么)(?:参数)?(?:值|取值)|"
        r"\bfor\s+which\s+(?:values?|parameters?)\b|"
        r"\bwhich\s+(?:values?|parameters?)\b|"
        r"\b(?:find|determine|list|classify)\s+all\s+(?:possible\s+)?"
        r"(?:values?|parameters?|pairs?|tuples?|configurations?|moves?|steps?|sets?|points?)\b",
        target,
        re.IGNORECASE,
    ))
    if exhaustive_target and not count_of_all:
        items.append(Requirement("exhaustive_result", strict=True))
    lz_request = bool(re.search(
        r"Lempel[- ]?Ziv|\bLZ(?:7[678]|W)?\b|短语编码",
        text,
        re.IGNORECASE,
    ))
    if lz_request and re.search(
        r"短语|分解|词组|\bphrases?\b|\bdecomposition\b",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("phrase_decomposition", strict=True))
    if lz_request and re.search(
        r"编码|码串|比特串|encoded\s+string|bit\s*string",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("encoded_string", strict=True))
    explicit_support = re.search(
        r"(?:要求|必须|须|需|应当)\s*先[^。；;\n]{1,160}(?:再|然后)|"
        r"(?:要求|必须|须|需|应当)(?:严格|分别|逐项|完整地?|明确(?:地)?)?"
        r"(?:使用|用|从|通过|由|以|作|解|识别|构造|给出|推导|验证|核对|检查|说明|证明|计算|归一化)"
        r"[^。；;\n]{0,220}|"
        r"(?:证明|论证|推导|计算|归一化)(?:须|必须|应当|要求)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:用|利用|根据|通过|由|从)"
        r"[^。；;\n]{1,160}(?:说明|解释|推导|证明|论证|验证|核对|检查|计算|求|确定)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:说明|解释)[^。；;\n]{0,160}|"
        r"(?:用|使用|利用|通过)[^。；;\n]{2,160}(?:求|计算|确定|推导|证明|论证|说明|解释|验证)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:"
        r"推导|证明|论证|验证|核对|检查|说明[^。；;\n]{0,40}(?:理由|过程|步骤|依据|计算)|"
        r"(?:给出|写出|展示|列出)[^。；;\n]{0,80}"
        r"(?:计算|推导|证明|论证|过程|步骤|依据|公式|方程|矩阵|留数|节点|权重|精度|控制函数|上界|估计))|"
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,140}"
        r"(?:控制函数|控制量|可积上界|支配函数|dominating function|integrable bound)|"
        r"(?:用|利用|根据|通过|由|从)[^。；;\n]{1,120}"
        r"(?:定理|公式|方程|问题|方法|法|定义|矩阵|变换|基本形式|核|原理|"
        r"theorem|formula|equations?|method|definition|transform|kernel|principle)"
        r"[^。；;\n]{0,80}(?:求|计算|推导|证明|说明|验证|find|compute|derive|prove|explain|verify)|"
        r"(?:要求|必须|须|需(?:要)?|应当)[^。；;\n]{0,100}"
        r"(?:分类讨论|分情况讨论|按[^。；;\n]{1,40}分类)|"
        r"\b(?:must|required\s+to|should)\b[^.\n]{0,100}"
        r"(?:case\s+analysis|classify\s+by)|"
        r"\band\s+(?:derive|prove|justify|verify|check|show\s+(?:the\s+)?"
        r"(?:calculation|derivation|proof|steps?|work))\b|"
        r"\busing\b[^.\n]{1,120}\b(?:theorem|formula|method|principle|identity)\b|"
        r"(?:^|[。；;]\s*)(?:验证|检验|核对)[^。；;\n]{2,180}(?:是否|为解|成立)|"
        r"\b(?:verify|check)\b[^.\n]{2,180}\b(?:is\s+a\s+solution|holds?|satisf(?:y|ies))\b",
        text,
        re.IGNORECASE,
    )
    forced_support = re.search(
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,160}"
        r"(?:注明|写明|列明|区分|辨明|陈述|指出|说明|解释|证明|推导|验证|核对|检查)|"
        r"(?:用|利用|根据|通过|由|从)[^。；;\n]{1,160}"
        r"(?:给出|写出|得到|导出|求出|算出|计算出|指出|说明|解释|证明|验证|核对)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:分别|逐项)?"
        r"(?:指出|注明|写明|列明|陈述)[^。；;\n]{1,160}|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:分别|逐项)?"
        r"(?:用|利用|根据|通过|由|从)[^。；;\n]{1,160}"
        r"(?:给出|写出|得到|导出|求出|算出|计算出|说明|解释|证明|验证|核对)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:分别|逐项)?"
        r"(?:给出|写出|展示|列出)[^。；;\n]{0,140}"
        r"(?:依据|理由|过程|步骤|推导|证明|论证|上界|下界|取等|核验)|"
        r"\b(?:must|required\s+to|should)\b[^.\n]{0,160}"
        r"\b(?:state|point\s+out|distinguish|identify|name|justify|derive|verify|explain)\b|"
        r"\band\s+(?:separately\s+|explicitly\s+)?(?:use|distinguish|"
        r"derive|verify|explain|justify)\b[^.\n]{0,180}|"
        r"\band\s+(?:separately\s+|explicitly\s+)?(?:state|give|provide)\b"
        r"[^.\n]{0,100}\b(?:proof|argument|reason|derivation|justification|"
        r"calculation|bound|theorem|equation used)\b|"
        r"(?:给出|写出|展示|列出)[^。；;\n]{1,120}(?:计数)?"
        r"(?:结构|分解|分类|公式|构造)[^。；;\n]{0,100}"
        r"(?:而不只|不能只|不应只|而非只)|"
        r"\b(?:give|show|provide|write)[^.\n]{1,120}"
        r"(?:counting\s+structure|decomposition|classification|formula|construction)"
        r"[^.\n]{0,100}(?:not\s+just|rather\s+than\s+only)|"
        r"(?:作为|用作)(?:一个)?(?:证明|证书|核验)|"
        r"\bas\s+(?:a\s+)?(?:certificate|proof)\b",
        text,
        re.IGNORECASE,
    )
    explicit_support = bool(explicit_support or forced_support)
    generic_support_signal = re.search(
        r"说明理由|给出证明|完整(?:论证|证明|推导)|严格(?:论证|证明|推导)|"
        r"(?:证明|论证|推导)须|须[^。；;\n]{0,40}(?:证明|论证|推导)|"
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,160}(?:证明|论证|推导|验证|核对|检查|说明)|"
        r"证明.*(?:所有|唯一)|\b(?:justify|explain)\b|give (?:a )?proof|show your work|"
        r"\bwith\s+(?:a\s+)?(?:complete\s+|rigorous\s+)?"
        r"(?:proof|derivation|argument|justification)\b|"
        r"(?:complete|rigorous)\s+(?:proof|derivation|argument|normalization|calculation|justification)|"
        r"(?:the\s+)?(?:proof|derivation|argument|normalization|calculation|justification)\s+"
        r"(?:must|should|is required)|"
        r"(?:使用|利用|用|根据|通过|由|从)[^。；;\n]{1,100}"
        r"(?:定理|公式|方程|方法|法|定义|变换|原理)(?:[^。；;\n]{0,80})|"
        r"\busing\b[^.\n]{1,120}\b(?:theorem|formula|method|definition|transform|principle)\b|"
        r"(?:验证|检验|核对)[^。；;\n]{2,180}(?:是否|为解|成立)|"
        r"\b(?:verify|check)\b[^.\n]{2,180}\b(?:is\s+a\s+solution|holds?|satisf(?:y|ies))\b",
        text,
        re.IGNORECASE,
    )
    short_form_task = bool(
        profile.task_kind in {"choice", "fill_blank"}
        or shape in {"choice", "truth"}
    )
    explicit_short_form_support = re.search(
        r"说明(?:你的)?理由|解释(?:原因|理由)?|"
        r"给出(?:一句|一个|简要(?:的)?|相应(?:的)?|你的)?"
        r"(?:完整|严格)?(?:证明|论证|理由)|"
        r"证明|论证|推导|展示(?:计算|推导|证明)?(?:过程|步骤)|写出(?:计算|推导)?过程|"
        r"\b(?:justify|explain)\b|"
        r"\bsupport\s+(?:(?:your|the)\s+)?(?:answer|conclusion|claim|assertion)\b|"
        r"\bwith\s+(?:a\s+)?(?:brief\s+|complete\s+|rigorous\s+)?justification\b|"
        r"\b(?:give|provide)\b[^.\n]{0,80}"
        r"\b(?:proof|argument|reason|justification|derivation)\b|"
        r"\bshow\s+(?:all\s+|your\s+)?work\b|"
        r"\b(?:prove|derive)\b",
        text,
        re.IGNORECASE,
    )
    support_required = bool(
        profile.task_kind in {"proof", "derivation", "explanation"}
        or (
            explicit_short_form_support
            if short_form_task
            else (explicit_support or generic_support_signal)
        )
    )
    if support_required:
        items.append(Requirement("reasoning", strict=True, category="support"))
    direct_substitution = re.search(
        r"(?:直接)?(?:代回|代入)(?:原)?(?:方程|偏微分方程|常微分方程)"
        r"[^。；;\n]{0,100}(?:核验|验证|检验|检查)|"
        r"direct(?:ly)?\s+substitut\w*\s+(?:back\s+)?into\s+the\s+"
        r"(?:PDE|ODE|equation)[^.\n]{0,100}(?:verify|check)",
        text,
        re.IGNORECASE,
    )
    if direct_substitution is None:
        direct_substitution = re.search(
            r"(?:验证|检验|核对)[^。；;\n]{0,140}(?:函数|解|u\s*\(|y\s*\()[^。；;\n]{0,100}"
            r"(?:是否)?(?:为|是)?(?:原)?(?:方程|PDE|ODE)[^。；;\n]{0,30}(?:的)?解|"
            r"(?:方程|PDE|ODE)[^。；;\n]{0,180}(?:验证|检验|核对)"
            r"[^。；;\n]{0,140}(?:是否)?(?:为|是)(?:其|该|此|原方程)?解|"
            r"\b(?:verify|check)\b[^.\n]{0,160}\b(?:function|solution)\b"
            r"[^.\n]{0,100}\b(?:satisf(?:y|ies)|is\s+a\s+solution\s+of)\b"
            r"[^.\n]{0,60}\b(?:PDE|ODE|equation)\b",
            text,
            re.IGNORECASE,
        )
    if direct_substitution:
        items.append(Requirement(
            "differential_equation_substitution", strict=True, category="support"
        ))
        if re.search(
            r"初值|初始条件|initial\s+(?:value|condition)",
            direct_substitution.group(0),
            re.IGNORECASE,
        ) or re.search(
            r"(?:方程|PDE|ODE)[^。；;\n]{0,40}(?:和|及|与|and)"
            r"[^。；;\n]{0,40}(?:初值|初始条件|initial)",
            text,
            re.IGNORECASE,
        ):
            items.append(Requirement(
                "initial_condition_check", strict=True, category="support"
            ))
    if re.search(
        r"指出[^。；;\n]{0,100}(?:边值问题|指数鞅)|"
        r"(?:state|identify|give)[^.\n]{0,100}"
        r"(?:boundary[- ]value problem|exponential martingale)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement(
            "boundary_value_or_exponential_martingale",
            strict=True,
            category="support",
        ))
    for index, term in enumerate(_explicit_support_terms(text), start=1):
        items.append(Requirement(
            f"support_anchor_{index}",
            ((term,),),
            strict=True,
            category="support",
        ))
    if re.search(
        r"(?:证明|说明|验证|判断|确定|求|prove|show|verify|determine)"
        r"[^。.;\n]{0,180}(?:"
        r"f_?\{?n\}?[^。.;\n]{0,80}(?:→|\\to|converges?\s+to)"
        r"[^。.;\n]{0,60}(?:几乎处处|a\.?e\.?|almost\s+everywhere)|"
        r"(?:几乎处处|a\.?e\.?|almost\s+everywhere)[^。.;\n]{0,100}"
        r"(?:极限|limit|f_?\{?n\}?))",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("almost_everywhere_limit", strict=True))
    if re.search(r"一致可积|uniform(?:ly)?\s+integrab", text, re.IGNORECASE) and re.search(
        r"证明|(?:直接)?(?:从|由)[^。.;\n]{0,30}定义|说明|验证|prove|"
        r"(?:from|by)\s+the\s+definition|justify|verify|explain",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement(
            "uniform_integrability_check",
            strict=True,
            category="support",
        ))
    if re.search(
        r"(?:\\?\|\s*f_?\{?n\}?\s*\\?\|\s*_?\{?1\}?|L\s*\^?\s*\{?1\}?\s*(?:范数|norm))"
        r"[^。.;\n]{0,80}(?:不趋于\s*0|不收敛|does\s+not\s+(?:tend|converge)|not\s+converge)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("l1_nonconvergence", strict=True))
    if re.search(
        r"L\s*\^?\s*\{?\s*1\s*\}?[^。.;\n]{0,35}"
        r"(?:极限|收敛性|是否收敛|limit|convergence)|"
        r"(?:判断|确定|求|find|determine)[^。.;\n]{0,80}"
        r"L\s*\^?\s*\{?\s*1\s*\}?[^。.;\n]{0,30}(?:极限|收敛)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("l1_limit_conclusion", strict=True))
    if re.search(
        r"(?:写出|给出|列出|推导|derive|write|give)[^。.;\n]{0,100}"
        r"(?:正规方程|normal\s+equations?)|"
        r"(?:正规方程|normal\s+equations?)[^。.;\n]{0,80}"
        r"(?:写出|给出|列出|推导|derive|write|give)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("normal_equation", strict=True, category="support"))
    if re.search(
        r"(?:精确)?(?:计算|求|确定|find|compute|determine)[^。.;\n]{0,120}"
        r"(?:\\widehat\s*\{?\s*\\?beta|\\hat\s*\{?\s*\\?beta|"
        r"(?:GLS|WLS|广义最小二乘|加权最小二乘)[^。.;\n]{0,35}"
        r"(?:系数|估计(?:量|值)?|coefficient|estimat(?:e|or))|"
        r"(?:回归系数|coefficient\s+vector))",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("coefficient_estimate", strict=True))
    if re.search(r"Sturm[- ]Liouville|斯图姆|施图姆", text, re.IGNORECASE):
        items.append(Requirement("sturm_liouville_argument", strict=True, category="support"))
    if re.search(
        r"(?:给出|写出|求|确定|find|give|provide|determine)[^。.;\n]{0,100}"
        r"(?:对偶(?:最优)?解|对偶证书|dual\s+(?:optimal\s+)?solution|dual\s+certificate)|"
        r"(?:对偶(?:最优)?解|对偶证书|dual\s+(?:optimal\s+)?solution|dual\s+certificate)"
        r"[^。.;\n]{0,80}(?:给出|写出|求|确定|find|give|provide|determine)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("dual_certificate", strict=True))
    if re.search(
        r"(?:作为|用作)(?:一个)?(?:对偶)?(?:证书|核验)|"
        r"\bas\s+(?:a\s+)?(?:dual\s+)?certificate\b",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement(
            "dual_optimality_check",
            strict=True,
            category="support",
        ))
    if re.search(r"平稳分布|稳态分布|stationary\s+distribution", target, re.IGNORECASE):
        items.append(Requirement("stationary_distribution", strict=True))
    if re.search(
        r"(?:验证|检验|核对|verify|check)[^。.;\n]{0,60}"
        r"(?:细致平衡|详细平衡|detailed\s+balance)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("detailed_balance_check", strict=True, category="support"))
    if (
        re.search(r"稳定函数|stability\s+function", text, re.IGNORECASE)
        and re.search(
            r"(?:绝对)?稳定域|绝对稳定|absolute\s+stability|stability\s+region",
            text,
            re.IGNORECASE,
        )
        and re.search(r"负实轴|negative\s+real\s+axis", text, re.IGNORECASE)
    ):
        items.extend((
            Requirement("stability_function", strict=True),
            Requirement("stability_boundary_equation", strict=True),
            Requirement("closed_stability_interval", strict=True),
        ))
    runge_kutta_stability = profile.topic == "runge_kutta_stability" or bool(
        re.search(
            r"SDIRK|Butcher\s*(?:表|数组|tableau)|Runge.?Kutta",
            text,
            re.IGNORECASE,
        )
        and re.search(
            r"稳定函数|稳定域|L\s*[-－— ]?稳定|阶条件|"
            r"stability\s+function|stability\s+region|L[- ]stability|"
            r"order\s+conditions?",
            text,
            re.IGNORECASE,
        )
    )
    if runge_kutta_stability:
        if re.search(r"稳定函数|stability\s+function|R\s*\(\s*z\s*\)", target, re.IGNORECASE):
            items.append(Requirement("stability_function", strict=True))
        if re.search(
            r"无穷远极限|z\s*(?:\\to|→|->)\s*(?:\\infty|∞)|"
            r"L\s*[-－— ]?稳定|L[- ]stability|L[- ]stable|"
            r"limit[^.;\n]{0,80}(?:infinity|infty)",
            target,
            re.IGNORECASE,
        ):
            items.append(Requirement("stability_infinity_limit", strict=True))
        if re.search(r"阶条件|二阶|三阶|order\s+conditions?|(?:method\s+)?order", target, re.IGNORECASE):
            items.append(Requirement("method_order", strict=True))
    multistep = bool(re.search(
        r"线性多步法|多步法|向后差分公式|后向差分公式|"
        r"\b(?:BDF\s*\d*|backward differentiation formula|linear multistep method|"
        r"Adams[- ](?:Bashforth|Moulton))\b",
        text,
        re.IGNORECASE,
    ))
    if multistep:
        if re.search(r"特征方程|characteristic\s+equation", target, re.IGNORECASE):
            items.append(Requirement("multistep_characteristic_equation", strict=True))
        if re.search(
            r"(?:单位圆)?稳定边界(?:参数式)?|边界参数式|"
            r"stability\s+boundary|boundary\s+parametri[sz]ation",
            target,
            re.IGNORECASE,
        ):
            items.append(Requirement("stability_boundary_parametrization", strict=True))
        if re.search(r"零稳定|zero[- ]stability", target, re.IGNORECASE):
            items.append(Requirement("zero_stability", strict=True))
        if re.search(r"阶数|精度阶|(?:method\s+)?order", target, re.IGNORECASE):
            items.append(Requirement("method_order", strict=True))
        if re.search(r"A\s*[- ]?稳定|A[- ]stability|A[- ]stable", target, re.IGNORECASE):
            items.append(Requirement("a_stability_judgement", strict=True))
    stationary_iteration = bool(re.search(
        r"Jacobi\s*法|雅可比迭代|"
        r"\bJacobi\b(?=[^。；;\n]{0,40}(?:迭代|矩阵|谱半径))|"
        r"Gauss.?Seidel|高斯.?赛德尔|超松弛|"
        r"\b(?:Jacobi iteration|Gauss[- ]Seidel|successive over[- ]relaxation)\b",
        text,
        re.IGNORECASE,
    ))
    if stationary_iteration:
        if re.search(r"迭代矩阵|iteration\s+matrix", target, re.IGNORECASE):
            items.append(Requirement("iteration_matrix", strict=True))
        if re.search(r"谱半径|spectral\s+radius", target, re.IGNORECASE):
            items.append(Requirement("spectral_radius", strict=True))
        if (
            re.search(r"[xuy]\s*\^\s*\{?\(?1\)?\}?|[xuy]_?\{?1\}?", target)
            and re.search(r"[xuy]\s*\^\s*\{?\(?2\)?\}?|[xuy]_?\{?2\}?", target)
        ):
            items.append(Requirement("requested_iterates", strict=True))
    quadrature_problem = bool(re.search(
        r"Gauss[-–— ]*Legendre|高斯.?勒让德|高斯求积|求积公式|"
        r"复化(?:中点|梯形|辛普森)公式|"
        r"\b(?:gauss[- ]legendre|gaussian quadrature|quadrature rule|"
        r"composite[- ](?:midpoint|trapezoid|simpson)(?:'s)?\s+rule)\b",
        text,
        re.IGNORECASE,
    ))
    if quadrature_problem:
        if re.search(
            r"(?:给出|写出|列出|求|确定)[^。.;\n]{0,100}节点|"
            r"\b(?:find|give|provide|write|list|determine)\b"
            r"[^.;\n]{0,100}\bnodes?\b",
            target,
            re.IGNORECASE,
        ):
            items.append(Requirement("quadrature_nodes", strict=True))
        if re.search(r"权重|weights?", target, re.IGNORECASE):
            items.append(Requirement("quadrature_weights", strict=True))
        if re.search(
            r"求积值|求积结果|求积近似|近似(?:计算|求)[^。；;\n]{0,40}积分|"
            r"quadrature\s+(?:value|result|approximation)|"
            r"approximate(?:ly)?\s+(?:compute|evaluate)[^.;\n]{0,40}integral|"
            r"\bapproximate\b[^.;\n]{0,80}(?:\\int|\bintegral\b)|"
            r"\b(?:resulting\s+)?approximation\b",
            target,
            re.IGNORECASE,
        ):
            items.append(Requirement("quadrature_value", strict=True))
        if re.search(r"误差|error|I\s*-\s*Q|Q\s*-\s*I", target, re.IGNORECASE):
            items.append(Requirement("quadrature_error", strict=True))
    if re.search(
        r"(?:Gauss|高斯)?曲率函数|curvature\s+function|"
        r"(?:求|find|determine)[^。.;\n]{0,100}K\s*\(\s*[A-Za-z]",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("curvature_function", strict=True))
    if re.search(
        r"(?:原点|指定点)[^。.;\n]{0,100}(?:曲率|K)|"
        r"(?:curvature|K)[^.;\n]{0,100}at\s+the\s+(?:origin|specified\s+point)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("curvature_point_value", strict=True))
    operator_problem = bool(re.search(
        r"算子|operator|L\s*\^\s*\{?\d|L\s*\^\s*[pP]",
        text,
        re.IGNORECASE,
    ))
    if operator_problem:
        if re.search(r"算子范数|\\?\|\s*T\s*\\?\||operator\s+norm", target, re.IGNORECASE):
            items.append(Requirement("operator_norm", strict=True))
        if re.search(r"(?<!点)(?<!本质值域)谱(?:集)?|\bspectrum\b", target, re.IGNORECASE):
            items.append(Requirement("operator_spectrum", strict=True))
        if re.search(r"点谱|point\s+spectrum", target, re.IGNORECASE):
            items.append(Requirement("point_spectrum", strict=True))
    if re.search(r"Jordan\s*块|若尔当块|Jordan\s+blocks?", target, re.IGNORECASE):
        items.append(Requirement("jordan_blocks", strict=True))
    if re.search(r"(?:\\operatorname\s*\{rank\}|\\?rank|秩)\s*[A-Za-z]?", target, re.IGNORECASE):
        items.append(Requirement("operator_rank", strict=True))
    if re.search(r"最小多项式|minimal\s+polynomial", target, re.IGNORECASE):
        items.append(Requirement("minimal_polynomial", strict=True))
    if re.search(r"Smith\s*(?:标准|正规)?形|Smith\s+normal\s+form|史密斯|\bSNF\b", target, re.IGNORECASE):
        items.append(Requirement("smith_normal_form", strict=True))
    if re.search(r"\\?operatorname\s*\{coker\}|\\?coker|余核|cokernel", target, re.IGNORECASE):
        items.append(Requirement("cokernel_structure", strict=True))
    if re.search(r"W\s*_?\s*\{?2\}?\s*\^\s*\{?2\}?|W₂²", target, re.IGNORECASE):
        items.append(Requirement("wasserstein_squared_value", strict=True))
    if re.search(r"最优传输映射|optimal\s+transport\s+map", target, re.IGNORECASE):
        items.append(Requirement("optimal_transport_map", strict=True))
    if re.search(r"UMVU|一致最小方差无偏", target, re.IGNORECASE):
        items.append(Requirement("umvu_estimator", strict=True))
    uniform_series = bool(
        re.search(r"函数项级数|级数|series", text, re.IGNORECASE)
        and re.search(r"\[\s*0\s*[,，]\s*r\s*[\])]", text, re.IGNORECASE)
        and re.search(r"\[\s*0\s*[,，]\s*1\s*\)", text)
    )
    if uniform_series:
        items.extend((
            Requirement("local_uniform_convergence", strict=True),
            Requirement("global_nonuniform_convergence", strict=True),
        ))
        if re.search(r"和函数|sum\s+function|求和|find\s+(?:its|the)\s+sum", text, re.IGNORECASE):
            items.append(Requirement("series_sum_function", strict=True))
        if re.search(
            r"(?:解释|说明)[^。.;\n]{0,100}(?:区间|范围)[^。.;\n]{0,80}"
            r"(?:不同|差异|区别|原因)|"
            r"(?:原因|理由)[^。.;\n]{0,80}(?:不同|差异|区别)|"
            r"\b(?:explain|justify)\b[^.\n]{0,140}"
            r"(?:intervals?|domains?|ranges?)[^.\n]{0,80}"
            r"(?:differ|different|contrast|why)",
            text,
            re.IGNORECASE,
        ):
            items.append(Requirement(
                "uniform_convergence_scope_reason",
                strict=True,
                category="support",
            ))
    if re.search(
        r"可执行[^。.;\n]{0,80}(?:插入|归纳构造)|"
        r"(?:插入|归纳)[^。.;\n]{0,80}(?:构造|算法)[^。.;\n]{0,30}(?:明确|具体|可执行)|"
        r"\b(?:explicit|executable|algorithmic)\b[^.\n]{0,100}"
        r"\b(?:insertion|inductive construction)\b",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement(
            "executable_insertion_step",
            strict=True,
            category="support",
        ))
    if profile.task_kind == "construction" or re.search(r"构造|举例|反例|construct|counterexample", target, re.IGNORECASE):
        items.extend((
            Requirement("construction_object", strict=True),
            Requirement("construction_check", strict=True, category="support"),
        ))
    lacunary_boundary_problem = profile.topic == "lacunary_natural_boundary" or bool(
        re.search(
            r"(?:稀疏|空隙)[^。；;\n]{0,40}幂级数|"
            r"\blacunary\s+power\s+series\b",
            text,
            re.IGNORECASE,
        )
        and re.search(
            r"自然边界|解析延拓|\bnatural\s+boundary\b|"
            r"\banalytic\s+continuation\b",
            text,
            re.IGNORECASE,
        )
    )
    if lacunary_boundary_problem:
        if re.search(
            r"收敛半径|radius\s+of\s+convergence|convergence\s+radius|"
            r"(?:求|确定|find|determine)[^。；;.\n]{0,80}(?:半径|radius)",
            text,
            re.IGNORECASE,
        ):
            items.append(Requirement("convergence_radius", strict=True))
        if re.search(
            r"收敛域|收敛圆|domain\s+of\s+convergence|"
            r"disk\s+of\s+convergence|circle\s+of\s+convergence",
            text,
            re.IGNORECASE,
        ):
            items.append(Requirement("convergence_domain", strict=True))
        if re.search(
            r"自然边界|不能[^。；;\n]{0,60}解析延拓|"
            r"\bnatural\s+boundary\b|\bexclude\b[^.;\n]{0,100}"
            r"\banalytic\s+continuation\b",
            text,
            re.IGNORECASE,
        ):
            items.append(Requirement(
                "natural_boundary_classification", strict=True
            ))
    named_iterative_method = bool(re.search(
        r"牛顿法|二分法|割线法|欧拉法|"
        r"\b(?:newton(?:'s)?\s+method|bisection|secant(?:\s+method)?|euler\s+method)\b",
        text,
        re.IGNORECASE,
    ))
    requests_iteration_formula = bool(re.search(
        r"迭代公式|迭代式|递推公式|格式|"
        r"\b(?:iteration\s+formula|iteration\s+scheme|recurrence\s+formula)\b",
        target,
        re.IGNORECASE,
    ))
    requests_first_iteration = bool(re.search(
        r"第一次迭代|第一步迭代|首步迭代|"
        r"(?:计算|求|得到|给出)\s*(?:出|得)?\s*[xuy]_?\{?1\}?|"
        r"(?:compute|find|give|obtain)\s+(?:the\s+)?(?:value\s+of\s+)?"
        r"[xuy]_?\{?1\}?|first iterate|first iteration",
        target,
        re.IGNORECASE,
    ))
    if requests_iteration_formula or (
        named_iterative_method and requests_first_iteration
    ):
        items.append(Requirement("method_formula", strict=True))
    if named_iterative_method and requests_first_iteration:
        items.append(Requirement("first_iteration", strict=True))
    if re.search(
        r"精确值[^。；;\n]{0,120}(?:近似值|近似计算|近似结果)|"
        r"(?:近似值|近似计算|近似结果)[^。；;\n]{0,120}精确值|"
        r"exact\s+(?:value|form)[^.\n]{0,100}approximate\s+value|"
        r"(?:approximate\s+value|approximate(?:ly)?\s+(?:compute|evaluate))"
        r"[^.\n]{0,120}exact\s+(?:value|form)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("exact_and_approximate", strict=True))
    if re.search(
        r"(?:L\s*\^?\s*\{?\s*1\s*\}?|L[¹1])[^。；;\n]{0,120}"
        r"(?:范数|积分|收敛)|(?:范数|积分)[^。；;\n]{0,100}"
        r"(?:L\s*\^?\s*\{?\s*1\s*\}?|L[¹1])|"
        r"\bL\s*\^?\s*\{?\s*1\s*\}?\b[^.;\n]{0,120}"
        r"\b(?:norm|integral|convergen)",
        text,
        re.IGNORECASE,
    ) and re.search(
        r"范数|\bnorm\b",
        target,
        re.IGNORECASE,
    ):
        items.append(Requirement("l1_norm_check", strict=True))
    places = re.search(r"(?:保留|精确到)\s*(\d+)\s*位小数|(?:to|give)\s*(\d+)\s*decimal places?", text, re.IGNORECASE)
    if places:
        count = next(group for group in places.groups() if group)
        items.append(Requirement(f"decimal_places_{count}", strict=True, category="format"))
    unit = _requested_unit(text)
    if unit:
        items.append(Requirement("unit", ((unit,),), strict=True, category="format"))
    if re.search(
        r"并(?:说明|给出|写出|注明).*条件|写明.*定义域|注明.*范围|"
        r"\bstate\b.*\bconditions?\b|\binclude\b.*\bdomain\b|"
        r"最大(?:右侧|左侧)?(?:存在|解|定义)?区间|存在区间|"
        r"maximal\s+(?:interval|interval\s+of\s+existence)|interval\s+of\s+existence",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("domain_or_conditions", strict=True))
    asks_full_and_one_sided = bool(
        re.search(
            r"最大(?:存在|解|定义)?区间|完整最大区间|"
            r"(?:full\s+)?maximal\s+(?:existence\s+|solution\s+)?interval",
            text,
            re.IGNORECASE,
        )
        and re.search(
            r"(?:并|以及|同时|和|与)[^。；;\n]{0,80}"
            r"(?:右侧|左侧|单侧)(?:存在|解|定义)?区间|"
            r"\b(?:and|as\s+well\s+as)\b[^.;\n]{0,80}"
            r"(?:right|left)(?:[- ]hand)?\s+(?:existence\s+|solution\s+)?interval",
            text,
            re.IGNORECASE,
        )
    )
    if asks_full_and_one_sided:
        items.append(Requirement(
            "maximal_interval_and_one_sided_part", strict=True
        ))
    symbols = _requested_symbols(target)
    for symbol in symbols[:4]:
        normalized_symbol = re.sub(r"[{}\s]", "", symbol).casefold()
        if "first_iteration" in {item.name for item in items} and re.fullmatch(
            r"[xuy]_?1", normalized_symbol
        ):
            continue
        items.append(Requirement(
            f"target_{symbol.casefold()}",
            ((f"{symbol}=",), (f"{symbol} =",), (f"{symbol}为",), (f"{symbol} is",)),
            strict=len(symbols) > 1,
        ))
    for symbol in _dependency_symbols(text, target)[:4]:
        items.append(Requirement(
            f"parameter_dependency_{symbol.casefold()}",
            strict=True,
        ))
    unique: dict[tuple[str, str], Requirement] = {}
    for item in items:
        unique[(item.name, item.category)] = item
    return list(unique.values())


def _goal_kind(profile: ProblemProfile) -> str:
    if profile.task_kind in {"proof", "derivation", "explanation"}:
        return "proof"
    if profile.task_kind == "construction":
        return "construction"
    return {
        "choice": "choice_selection",
        "truth": "truth_judgement",
        "roots": "equation_roots",
        "interval": "domain_or_interval",
    }.get(profile.answer_shape, "result")


def _answer_frame(
    text: str,
    target: str,
    profile: ProblemProfile,
    requirements: list[Requirement],
) -> AnswerFrame:
    if profile.task_kind in {"proof", "derivation", "explanation"}:
        return AnswerFrame("proof", predicate="conclusion", question_kind="proof")
    if profile.answer_shape == "truth":
        leading_auxiliary = re.match(
            r"^\s*(?:is|are|does|do|can|could|will|would)\s+(.+?)\?\s*$",
            target,
            re.IGNORECASE | re.DOTALL,
        )
        prefix = leading_auxiliary.group(1) if leading_auxiliary else re.split(
            r"是否|能否|可否|whether|is it",
            target,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        subject = re.sub(r"^(?:判断|确定|验证|decide|determine|verify)\s*", "", prefix, flags=re.IGNORECASE).strip(" ，,。:：")[-120:]
        return AnswerFrame("sentence", subject=subject, predicate="judgement", question_kind="truth")
    if profile.answer_shape == "choice":
        return AnswerFrame("math", predicate="choice labels", question_kind="choice")
    unit = next((alt[0] for item in requirements if item.name == "unit" for alt in item.alternatives), "")
    if profile.answer_shape in {"count", "probability"} or unit:
        return AnswerFrame("sentence", predicate=profile.answer_shape, unit=unit, question_kind=profile.answer_shape)
    return AnswerFrame("math", question_kind="math")


def _vieta_jump_applicable(semantics: StatementSemantics | None) -> bool:
    """Recognize the structural two-variable quadratic descent setting."""
    if semantics is None or "positive integers" not in semantics.domains:
        return False
    relation_text = " ".join(semantics.relations)
    squared = tuple(dict.fromkeys(re.findall(
        r"(?<![A-Za-z])([A-Za-z])\s*\^\s*(?:\{\s*2\s*\}|2)(?!\d)",
        relation_text,
    )))
    if len(squared) != 2:
        return False
    left, right = map(re.escape, squared)
    mixed_term = rf"(?:{left}\s*(?:\\cdot|\*)?\s*{right}|{right}\s*(?:\\cdot|\*)?\s*{left})"
    return bool(
        re.search(r"(?<![<>!])=(?!=)", relation_text)
        and re.search(mixed_term, relation_text)
    )


def _methods(
    profile: ProblemProfile,
    semantics: StatementSemantics | None = None,
) -> tuple[str, str]:
    by_topic = {
        "directed_euler_circuits": (
            "best_theorem_with_fixed_arc_normalization",
            "directed_matrix_tree_and_exit_ordering_check",
        ),
        "plane_rooted_tree_enumeration": (
            "lukasiewicz_words_and_cycle_lemma",
            "rooted_plane_tree_degree_sequence_formula",
        ),
        "lacunary_natural_boundary": (
            "radius_then_dense_boundary_singularities",
            "fabry_or_hadamard_gap_theorem",
        ),
        "runge_kutta_stability": (
            "order_conditions_then_stability_function",
            "imaginary_axis_modulus_and_infinity_limit",
        ),
        "spherical_triangle_area": (
            "spherical_cosine_law_then_girard_excess",
            "gram_matrix_or_vector_angle_area_check",
        ),
        "weierstrass_sine_product": (
            "weierstrass_sine_product_then_imaginary_substitution",
            "zero_set_normalization_and_log_derivative_check",
        ),
        "two_dimensional_polyharmonic_fundamental_solution": (
            "radial_laplacian_recurrence_and_flux_normalization",
            "fourier_symbol_and_distributional_constant_check",
        ),
        "numerical_method": ("derive the requested iteration and compute with the stated data", "independent residual and error check"),
        "calculus": ("definition or standard theorem with domain checks", "symbolic differentiation, substitution, or boundary check"),
        "equation": ("algebraic reduction with complete branch analysis", "substitute every candidate into the original equation"),
        "linear_algebra": ("row reduction and invariant computation", "independent determinant, rank, or polynomial check"),
        "combinatorics": ("bijection, recurrence, or inclusion-exclusion", "small-case enumeration and symmetry audit"),
        "graph": ("graph invariant or structural theorem", "direct small graph or matrix-tree check"),
        "probability": ("condition on a clear sample space", "normalization and complementary-event check"),
        "optimization": ("derive a sharp bound and attainability", "boundary and equality-case verification"),
        "olympiad_functional_equation": ("special substitutions followed by an exhaustive structural reduction", "substitute the full candidate family into the original equation"),
        "olympiad_geometry": ("one coherent synthetic or coordinate geometry route", "verify incidences, degeneracies, and the requested metric relation"),
        "olympiad_number_theory": ("factorization, congruences, valuations, or descent with all cases explicit", "substitute every parameter family and check exhaustiveness"),
        "olympiad_combinatorics": ("bijection, recurrence, double counting, or inclusion-exclusion", "enumerate the smallest legal cases and audit symmetry factors"),
        "olympiad_inequality": ("normalize and derive a sharp two-sided bound", "check equality or a limiting extremal construction"),
        "olympiad_polynomial": ("degree, factor, interpolation, and root-structure analysis", "check leading terms, multiplicities, and special values"),
        "olympiad_sequence": ("identify an invariant, monotonicity, periodicity, or linearizing substitution", "verify initial indices and substitute into the recurrence"),
        "cellular_homology": (
            "cellular_chain_complex_then_smith_normal_form",
            "fundamental_group_abelianization_check",
        ),
        "chebyshev_minimax": (
            "chebyshev_affine_map_and_normalized_alternation",
            "equioscillation_linear_system_check",
        ),
        "latin_square": (
            "normalize_symmetry_then_exhaust_structural_cases",
            "exact_enumeration_with_orbit_size_check",
        ),
        "nowhere_zero_flow": (
            "cycle_space_coordinate_inclusion_exclusion",
            "tutte_flow_polynomial_or_exact_edge_enumeration",
        ),
        "proof": ("minimal sufficient lemma with hypotheses checked", "counterexample search and converse audit"),
        "construction": ("explicit construction", "verify every condition on the same object"),
        "choice": ("evaluate every option from definitions", "independent option-by-option falsification"),
    }
    if _vieta_jump_applicable(semantics) and profile.primary_subject == "数论":
        selected = (
            "vieta_jumping_descent",
            "height_descent_with_direct_substitution_and_coordinate_symmetry",
        )
    elif profile.topic in by_topic and profile.topic not in {"proof"}:
        selected = by_topic[profile.topic]
    else:
        selected = None
    by_subject = {
        "离散数学": ("use an invariant, recurrence, bijection, or double count with all cases explicit", "verify by a different count and the smallest nontrivial cases"),
        "数值分析": ("derive the requested scheme and verify consistency, stability, and error assumptions", "residual, order-condition, and boundary-of-stability check"),
        "抽象代数": ("definitions, homomorphisms, and quotient structure", "kernel, order, and counterexample check"),
        "测度积分": ("select the convergence theorem and verify hypotheses", "exceptional-set or counterexample check"),
        "概率论": ("condition on a precisely defined sample space", "normalization, complement, and extreme-case check"),
        "泛函分析": ("operator theorem with completeness hypotheses", "norm estimate or counterexample check"),
        "复分析": ("singularity, contour, or analytic continuation analysis", "local expansion and residue check"),
        "微分几何": ("compute invariant geometric quantities", "coordinate or orientation-independent check"),
        "常微分方程": ("solve the equation then impose data", "differentiate and substitute the solution"),
        "偏微分方程": ("identify the PDE principle and boundary data", "differentiate or test the weak identity"),
        "统计推断": ("derive from likelihood or sampling distribution", "bias, variance, and parameter-domain check"),
        "随机过程": ("condition on states or increments", "transition normalization or martingale check"),
        "高等代数": ("row reduction, invariant subspaces, or the relevant polynomial", "trace, determinant, rank, and dimension check"),
        "线性回归": ("derive from the design matrix and error covariance assumptions", "normal equations, bias, and covariance check"),
        "拓扑学": ("work from the definitions and state every separation or compactness hypothesis", "test converse implications and standard counterexamples"),
    }
    if selected is None and profile.primary_subject in by_subject:
        selected = by_subject[profile.primary_subject]
    if selected is None:
        selected = by_topic.get(
            profile.topic,
            ("direct derivation from the statement", "independent substitution or boundary check"),
        )
    if profile.topic in {
        "directed_euler_circuits",
        "plane_rooted_tree_enumeration",
        "lacunary_natural_boundary",
        "runge_kutta_stability",
        "spherical_triangle_area",
        "weierstrass_sine_product",
        "two_dimensional_polyharmonic_fundamental_solution",
        "cellular_homology",
        "chebyshev_minimax",
        "latin_square",
        "nowhere_zero_flow",
    } or (_vieta_jump_applicable(semantics) and profile.primary_subject == "数论"):
        # These routes already encode the explicitly requested method and its
        # independent check more precisely than the generic method wrapper.
        return selected
    if semantics and semantics.requested_methods:
        required = ", ".join(semantics.requested_methods)
        return (
            f"apply the explicitly requested method ({required}) and show its defining formula",
            f"audit the requested method ({required}) by an independent residual, invariant, or boundary check",
        )
    if semantics and semantics.named_theorems:
        named = ", ".join(semantics.named_theorems)
        return (
            f"apply {named} only after checking every hypothesis",
            f"independently test the hypotheses and conclusion of {named}, including boundary cases",
        )
    return selected


def _constraints(text: str) -> list[str]:
    pattern = re.compile(
        r"正整数|非负整数|整数|实数|复数|有理数|互不相同|独立|连续|可测|紧致|可逆|"
        r"\b(?:positive integers?|nonnegative integers?|integers?|real|complex|rational|distinct|independent|continuous|measurable|compact|invertible)\b",
        re.IGNORECASE,
    )
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))[:10]


def _risks(
    text: str,
    profile: ProblemProfile,
    requirements: list[Requirement],
    constraints: tuple[str, ...],
) -> list[str]:
    risks: list[str] = []
    if profile.subject_confidence == "low":
        risks.append("low_subject_confidence")
    if profile.task_kind in {"proof", "derivation", "explanation"}:
        risks.extend(("theorem_scope", "logical_completeness"))
    if profile.task_kind == "construction":
        risks.append("construction_validation")
    if profile.answer_shape == "roots":
        risks.extend(("exhaustiveness", "extraneous_roots"))
    if any(item.name == "exhaustive_result" for item in requirements):
        risks.append("exhaustiveness")
    if any(item.name.startswith("parameter_dependency_") for item in requirements):
        risks.append("parameter_dependency")
    if profile.answer_shape == "choice":
        risks.append("option_exhaustiveness")
        if re.search(r"不正确|错误的是|不能|except|not true|incorrect", text, re.IGNORECASE):
            risks.append("negative_polarity")
    if profile.topic in {"combinatorics", "graph"}:
        risks.append("counting_or_symmetry")
    if profile.topic == "numerical_method":
        risks.append("method_and_requested_iterate")
    if re.search(r"所有|全部|唯一|最小|最大|最优|\b(?:all|unique|least|greatest|minimum|maximum|optimal)\b", text, re.IGNORECASE):
        risks.append("quantifier_or_extremal")
    if re.search(r"最小|最大|最优|\b(?:least|greatest|minimum|maximum|optimal)\b", text, re.IGNORECASE):
        risks.append("extremal_two_sided_bound")
    if re.search(r"连通|路径存在|可达|\b(?:connected|connectivity|reachable|path exists)\b", text, re.IGNORECASE):
        risks.append("global_connectivity")
    if constraints:
        risks.append("domain_constraints")
    if len(text) >= 320:
        risks.append("long_statement")
    if sum(item.strict for item in requirements) >= 3:
        risks.append("multiple_answer_obligations")
    if _has_residual_output_contract(text):
        risks.append("residual_output_contract")
    return list(dict.fromkeys(risks))


def _risk_score(text: str, profile: ProblemProfile, risks: tuple[str, ...]) -> int:
    score = 0
    if profile.difficulty == "hard":
        score += 3
    elif profile.difficulty == "medium":
        score += 1
    score += min(3, len(risks) // 2)
    if len(text) >= 500:
        score += 1
    return score


def _tool_whole_possible(text: str, profile: ProblemProfile, requirements: list[Requirement]) -> bool:
    if _has_residual_output_contract(text):
        return False
    if not profile.tool_eligible or profile.task_kind not in {"calculation", "fill_blank"}:
        return False
    if any(item.category == "support" for item in requirements):
        return False
    if len(text) > 500 or re.search(
        r"近似|误差|证明|论证|推导|说明|比较|构造|"
        r"(?:写出|给出|展示)[^。；;\n]{0,80}(?:计算|过程|步骤|依据)|"
        r"approx|error|prove|derive|justify|explain|compare|construct|show\s+(?:the\s+)?work",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"定义域|参数范围|解的范围|精确到|保留\s*\d+\s*位|"
        r"\b(?:parameter range|solution range|domain|decimal places?|significant figures?)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"牛顿法|二分法|割线法|欧拉法|迭代公式|中心差分|有限差分|"
        r"\b(?:newton(?:'s)? method|bisection|secant method|euler(?:'s)? method|"
        r"iteration formula|central difference|finite difference)\b",
        text,
        re.IGNORECASE,
    ):
        return False

    normalized = text.strip().strip("。.").strip()
    arithmetic_match = re.fullmatch(
        r"(?:(?:请)?(?:计算|求值)\s*.+?(?:的值)?|(?:请)?求\s*.+?\s*的值|"
        r"(?:please\s+)?(?:calculate|compute|evaluate)\s+.+?|"
        r"(?:please\s+)?find\s+the\s+value\s+of\s+.+?)",
        normalized,
        re.IGNORECASE | re.DOTALL,
    )
    if arithmetic_match and not re.search(
        r"面积|体积|周长|半径|直径|概率|期望|方差|"
        r"方程|导数|积分|极限|矩阵|\\(?:int|lim|sum)(?![A-Za-z])|"
        r"\b(?:area|volume|perimeter|radius|diameter|probability|expectation|variance|"
        r"equation|derivative|integral|limit|matrix)\b",
        text,
        re.IGNORECASE,
    ):
        return True

    if profile.answer_shape == "roots" and re.search(
        r"方程|求解|\b(?:solve|find\s+[A-Za-z]\s+if)\b",
        text,
        re.IGNORECASE,
    ):
        if len(re.findall(r"(?<![<>!])=(?!=)", text)) != 1 or re.search(
            r"方程组|system|[,，;；][^。.!?\n]{0,100}(?<![<>!])=(?!=)",
            text,
            re.IGNORECASE,
        ):
            return False
        relation = next((
            group
            for match in re.finditer(
                r"\$([^$\n]{1,240})\$|\\\((.{1,240}?)\\\)|\\\[(.{1,240}?)\\\]",
                text,
                re.DOTALL,
            )
            for group in match.groups()
            if group is not None and re.search(r"(?<![<>!])=(?!=)", group)
        ), "")
        if not relation:
            relation_match = re.search(
                r"([A-Za-z0-9_+\-*/^(){}\\\s]+(?<![<>!])=(?!=)"
                r"[A-Za-z0-9_+\-*/^(){}\\\s]+)",
                text,
            )
            relation = relation_match.group(1) if relation_match else ""
        relation = re.sub(
            r"^\s*(?:solve(?:\s+the\s+equation)?|find\s+[A-Za-z]\s+if)\s+",
            "",
            relation,
            flags=re.IGNORECASE,
        )
        relation = re.split(r"\bfor\s+[A-Za-z]\b", relation, maxsplit=1)[0]
        relation = re.sub(
            r"\\[A-Za-z]+|\b(?:sin|cos|tan|log|exp|sqrt)\b",
            "",
            relation,
            flags=re.IGNORECASE,
        )
        symbols = {
            symbol
            for symbol in re.findall(r"[A-Za-z]", relation)
            if symbol.casefold() not in {"e", "i"}
        }
        return len(symbols) == 1

    if re.search(r"导数|求导|\b(?:derivative|differentiat)\w*\b", text, re.IGNORECASE):
        return bool(
            not re.search(
                r"二阶|三阶|高阶|第\s*[2-9]\s*阶|"
                r"\b(?:second|third|higher|[2-9](?:nd|rd|th))\s+derivative\b",
                text,
                re.IGNORECASE,
            )
            and len(re.findall(
                r"导数|求导|\b(?:derivative|differentiat\w*)\b",
                text,
                re.IGNORECASE,
            )) == 1
        )

    if re.search(r"\\oint|∮|围道积分|contour\s+integral", text, re.IGNORECASE):
        return bool(
            len(re.findall(r"\\oint|∮", text)) == 1
            and re.search(r"(?:\\lvert|\|)\s*z\s*(?:\\rvert|\|)\s*=", text)
            and not re.search(
                r"分别|每个极点|各极点|留数计算|核对|"
                r"separately|each\s+pole|show\s+(?:the\s+)?residue",
                text,
                re.IGNORECASE,
            )
        )

    integral_count = len(re.findall(r"\\int(?![A-Za-z])|∫", text))
    if integral_count:
        if integral_count != 1 or not re.search(
            r"\\int\s*_\s*\{?[^\s{}]+\}?\s*\^\s*\{?[^\s{}]+\}?",
            text,
        ):
            return False
        source_match = re.search(r"\$([^$]*\\int[^$]*)\$", text, re.DOTALL)
        source = source_match.group(1) if source_match else text
        differential = re.search(r"(?:\\,|\\;|\s)d\s*([A-Za-z])\b", source)
        if not differential:
            return False
        variable = differential.group(1)
        scrubbed = re.sub(r"\\[A-Za-z]+|\\[,;!]", " ", source)
        scrubbed = re.sub(r"d\s*" + re.escape(variable) + r"\b", " ", scrubbed)
        free_symbols = {
            symbol
            for symbol in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", scrubbed)
            if symbol.casefold() not in {"e", "i"} and symbol != variable
        }
        return not free_symbols

    limit_count = len(re.findall(r"\\lim(?![A-Za-z])|极限|\blimit\b", text, re.IGNORECASE))
    if limit_count:
        return limit_count == 1

    if profile.topic == "linear_algebra":
        matrix_count = len(re.findall(r"\\begin\s*\{[pbvBV]?matrix\}", text))
        operations = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in (
            r"行列式|\\det|\bdeterminant\b",
            r"(?:矩阵的?)?秩|\brank\b",
            r"逆矩阵|\b(?:inverse matrix|matrix inverse)\b",
            r"特征值|\beigenvalues?\b",
        ))
        return matrix_count == 1 and operations == 1
    return False


def _has_residual_output_contract(text: str) -> bool:
    """Detect post-processing or extra filters outside a tool's core result.

    A local solver may still certify the untransformed subexpression, but it
    must not claim that value is the requested whole answer.  The patterns are
    intentionally limited to explicit command language so mathematical terms
    such as Fourier transform or a recurrence increment are unaffected.
    """
    value = str(text or "")
    transformed_result = re.search(
        r"(?:最终|最后|只需|请)?\s*(?:返回|输出|报告|给出|写出|求)[^。；;\n]{0,90}"
        r"(?:所求(?:量|值|结果)?|上述(?:量|值|结果)?|最终(?:结果|答案)|答案|结果)"
        r"[^。；;\n]{0,35}(?:加上?|减去|乘以|除以|取模|模\s*\d|平方|开方|绝对值)"
        r"[^。；;\n]{0,30}(?:后|的值|输出|返回|报告)|"
        r"(?:把|将)\s*(?:所求|上述|该|最终)?\s*(?:量|值|结果|答案)"
        r"[^。；;\n]{0,30}(?:加上?|减去|乘以|除以|取模|平方|开方|绝对值)"
        r"[^。；;\n]{0,30}(?:后)?\s*(?:返回|输出|报告|给出)|"
        r"\b(?:report|return|output|state|give)\b[^.;\n]{0,90}"
        r"(?:requested\s+(?:quantity|value|result)|final\s+(?:result|answer)|"
        r"the\s+(?:result|answer|value))[^.;\n]{0,40}"
        r"(?:plus|minus|times|multiplied|divided|modulo|mod\s+\d|squared|"
        r"square\s+root|absolute\s+value)|"
        r"\b(?:report|return|output|state|give)\b[^.;\n]{0,45}"
        r"(?:one|two|\d+)\s+(?:plus|minus|times)[^.;\n]{0,30}"
        r"(?:requested\s+(?:quantity|value|result)|result|answer)",
        value,
        re.IGNORECASE,
    )
    replaced_object = re.search(
        r"(?:返回|输出|报告|给出|写出)[^。；;\n]{0,70}(?:数量|数目|个数)"
        r"[^。；;\n]{0,45}(?:而不是|而非|替代|代替)[^。；;\n]{0,50}(?:对象|结果|取值|列表|逐一列出)|"
        r"\b(?:report|return|output|state|give)\b[^.;\n]{0,70}"
        r"(?:number|count|cardinality)[^.;\n]{0,50}"
        r"(?:rather\s+than|instead\s+of|not)[^.;\n]{0,55}"
        r"(?:objects?|results?|values?|moves?|solutions?|list(?:ing)?)",
        value,
        re.IGNORECASE,
    )
    extra_filter = re.search(
        r"(?:另外|此外|额外|附加|另|还|并额外)(?:再)?\s*(?:要求|限制|条件)|"
        r"(?:满足|加上|加入|增加)\s*(?:一个|以下)?\s*(?:额外|附加)(?:限制|条件|要求)|"
        r"\b(?:also|additionally|further)\s+require\b|"
        r"\b(?:extra|additional|further)\s+(?:restriction|constraint|condition|requirement)\b",
        value,
        re.IGNORECASE,
    )
    return bool(transformed_result or replaced_object or extra_filter)


def _requested_unit(text: str) -> str:
    match = re.search(
        r"单位(?:为|是|用)\s*([\u4e00-\u9fffA-Za-z%°]{1,12})|"
        r"以\s*([\u4e00-\u9fffA-Za-z%°]{1,12})(?:为单位|计|表示)|"
        r"\b(?:in|measured in)\s+(seconds?|minutes?|hours?|meters?|centimeters?|degrees?|percent)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    unit = next((group for group in match.groups() if group), "")
    if re.search(r"哪|何|什么|下列|以下|上述|统计", unit):
        return ""
    return unit


def _explicit_support_terms(text: str) -> tuple[str, ...]:
    """Extract only terms the statement explicitly says must be used."""
    explicit_sources = []
    for match in re.finditer(
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,24}?明确(?:地)?(?:使用|运用|说明|验证)"
        r"\s*([^。；;\n]{2,160})|"
        r"\b(?:must|required\s+to|should)\b[^.\n]{0,30}?explicitly\s+"
        r"(?:use|invoke|verify)\s+([^.\n]{2,160})",
        str(text or ""),
        re.IGNORECASE,
    ):
        explicit_sources.append(next(group for group in match.groups() if group))
    explanation_sources = []
    for match in re.finditer(
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:说明|解释)\s*"
        r"([^。；;\n]{2,140})|"
        r"\band\s+(?:explain|justify)\s+([^.\n]{2,140})",
        str(text or ""),
        re.IGNORECASE,
    ):
        explanation_sources.append(next(group for group in match.groups() if group))
    terms = []
    for source in explicit_sources:
        for raw in re.split(r"\s*(?:、|与|和|及|\band\b)\s*", source, flags=re.IGNORECASE):
            term = raw.strip(" ，,：:()（）")
            term = re.sub(r"^(?:the\s+)", "", term, flags=re.IGNORECASE)
            generic_explanation = re.search(
                r"\b(?:where|why|how|what|comes?|from|reasons?|process|steps?|work)\b|"
                r"为什么|为何|何以|如何|理由|过程|步骤|来源",
                term,
                re.IGNORECASE,
            )
            if (
                2 <= len(term) <= 64
                and not generic_explanation
                and not re.fullmatch(r"(?:上述|以上|these|them)", term, re.IGNORECASE)
            ):
                terms.append(term)
    named_pattern = re.compile(
        r"\b[A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)+\b|"
        r"\b[A-Z][A-Za-z-]{2,40}\s+(?:theorem|lemma|formula|criterion|method|"
        r"inequality|completeness|independence)\b|"
        r"[\u4e00-\u9fffA-Za-z0-9φΦ-]{0,24}?"
        r"(?:定理|引理|公式|方程|方法|算法|准则|原理|不等式|完备性|充分性|"
        r"独立性|单调性|连续性|对偶性|正态性|分布|函数|分位数|公因数|假设|平衡)",
        re.IGNORECASE,
    )
    for source in explanation_sources:
        for raw in re.split(r"\s*(?:、|与|和|及|\band\b)\s*", source, flags=re.IGNORECASE):
            cleaned = raw.strip(" ，,：:()（）")
            cleaned = re.sub(r"^(?:其|该|所使用的|所用的|the\s+)", "", cleaned, flags=re.IGNORECASE)
            cue_parts = re.split(
                r"(?:为什么|为何|何以|如何|是否|用在何处|等于|采用|使用|根据)",
                cleaned,
            )
            cleaned = cue_parts[-1].strip() if cue_parts else cleaned
            for match in named_pattern.finditer(cleaned):
                term = match.group(0).strip(" ，,：:()（）")
                term = re.sub(r"^(?:其中的?|中的|所述的?|相应的)", "", term)
                asks_for_unknown_name = bool(re.search(
                    r"所(?:使用|采用|依据|用)的|依据的|采用的|使用的|"
                    r"什么|哪(?:个|一)|which|what",
                    term,
                    re.IGNORECASE,
                ))
                if 2 <= len(term) <= 64 and not asks_for_unknown_name:
                    terms.append(term)
    return tuple(dict.fromkeys(terms))[:6]


def _requested_symbols(target: str) -> tuple[str, ...]:
    symbol = r"[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?![A-Za-z0-9_])"
    matches = list(re.findall(
        r"(?:求|计算|确定|写出|给出|find|compute|determine|give)\s*"
        rf"(?:the\s+value\s+of\s+)?({symbol})"
        r"(?!\s*(?:\^|\*\*|\(|\[|['’][A-Za-z]))",
        target,
        re.IGNORECASE,
    ))
    requested_lists = re.finditer(
        rf"(?:求(?:出)?|计算|确定|写出|给出|find|compute|determine|give)\s*"
        rf"(?:both\s+|the\s+values?\s+of\s+)?"
        rf"(?P<symbols>{symbol}(?:\s*(?:[,，、]|和|与|及|\band\b)\s*{symbol}){{1,5}})",
        target,
        re.IGNORECASE,
    )
    for requested in requested_lists:
        parts = re.split(
            r"\s*(?:[,，、]|和|与|及|\band\b)\s*",
            requested.group("symbols"),
            flags=re.IGNORECASE,
        )
        matches.extend(
            part for part in parts
            if re.fullmatch(symbol, part, re.IGNORECASE)
        )
    unique = tuple(dict.fromkeys(matches))
    indexed_bases = {
        symbol.split("_", 1)[0].casefold()
        for symbol in unique
        if "_" in symbol
    }
    return tuple(
        symbol for symbol in unique
        if not (len(symbol) == 1 and symbol.casefold() in indexed_bases)
    )


def _dependency_symbols(text: str, target: str) -> tuple[str, ...]:
    """Return parameters that the requested result must explicitly retain."""
    source = f"{target}\n{text}"
    patterns = (
        r"\bin\s+terms\s+of\s+(?:the\s+)?(?:parameter\s+)?\$?([A-Za-z])(?![A-Za-z])\$?",
        r"\b(?:as|write|express)\s+(?:a\s+)?function\s+of\s+\$?([A-Za-z])(?![A-Za-z])\$?",
        r"用\s*(?:参数)?\s*\$?([A-Za-z])(?![A-Za-z])\$?\s*表示",
        r"关于\s*\$?([A-Za-z])(?![A-Za-z])\$?\s*的(?:公式|表达式|函数)",
        r"([A-Za-z])\s*=\s*[A-Za-z]\s*\(\s*([A-Za-z])\s*\)",
    )
    symbols: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, source, re.IGNORECASE):
            groups = [group for group in match.groups() if group]
            symbols.append(groups[-1])
    return tuple(dict.fromkeys(symbols))


def _requirement_description(item: Requirement, language: str) -> str:
    zh = {
        "result_present": "明确最终结论",
        "numeric_result": "所求数值",
        "count_conclusion": "明确的最终总数，而非中间数字",
        "choice_labels": "完整选项标签集合",
        "judgement": "带对象的明确判断",
        "pole_location": "极点相对围道的明确位置",
        "isometry_judgement": "明确的等距或非等距判断",
        "invariance_judgement": "明确说明所问对象改变或保持不变",
        "convergence_radius": "明确的收敛半径",
        "convergence_domain": "明确的收敛域",
        "natural_boundary_classification": "明确的自然边界或不可延拓结论",
        "irreducibility_judgement": "明确的可约或不可约结论",
        "factor_degree_check": "判定可约性所需检查的因子次数",
        "equilibrium_point": "明确的平衡点",
        "stability_classification": "明确写出稳定或不稳定及类型",
        "eigenvalue_signs": "特征值及其符号",
        "membership_judgement": "明确的空间成员性判断",
        "integral_conclusion": "明确标注的积分值或发散结论",
        "second_derivatives": "题目要求的各个二阶导数",
        "harmonicity_judgement": "明确的调和性判断",
        "first_second_derivatives": "曲线的一阶与二阶导数",
        "variance_identification": "中心二阶矩的值及其方差识别",
        "almost_everywhere_zero": "由零积分推出函数几乎处处为零的结论",
        "all_solutions": "全部解并执行原条件",
        "exhaustive_result": "所有可能结果及其穷尽性",
        "phrase_decomposition": "完整的短语分解",
        "encoded_string": "完整的编码比特串",
        "reasoning": "必要且完整的论证",
        "construction_object": "明确构造对象",
        "construction_check": "逐条验证构造条件",
        "method_formula": "题目指定的方法或迭代公式",
        "first_iteration": "指定的第一次迭代值",
        "differential_equation_substitution": "把显式解及其导数直接代回微分方程核验",
        "initial_condition_check": "直接核验初值或初始条件",
        "boundary_value_or_exponential_martingale": "题目要求的边值问题或指数鞅及其边界条件",
        "exact_and_approximate": "精确值与近似值",
        "domain_or_conditions": "定义域或适用条件",
        "maximal_interval_and_one_sided_part": "包含初始点的完整最大区间及所问单侧区间",
        "unit": "题目要求的单位",
    }
    en = {
        "result_present": "an explicit final conclusion",
        "numeric_result": "the requested numeric value",
        "count_conclusion": "an explicit final total, not an intermediate number",
        "choice_labels": "the complete set of option labels",
        "judgement": "an explicit judgement naming its object",
        "pole_location": "the explicit location of each relevant pole relative to the contour",
        "isometry_judgement": "an explicit isometry or non-isometry judgement",
        "invariance_judgement": "an explicit statement that the requested object changes or remains invariant",
        "convergence_radius": "the explicit radius of convergence",
        "convergence_domain": "the explicit domain of convergence",
        "natural_boundary_classification": "the explicit natural-boundary or non-continuation conclusion",
        "irreducibility_judgement": "an explicit reducible or irreducible verdict",
        "factor_degree_check": "the factor degrees that must be checked for reducibility",
        "equilibrium_point": "the explicit equilibrium point",
        "stability_classification": "the explicit stable or unstable classification and type",
        "eigenvalue_signs": "the eigenvalues and their signs",
        "membership_judgement": "an explicit function-space membership judgement",
        "integral_conclusion": "an explicitly labelled integral value or divergence conclusion",
        "second_derivatives": "all requested second derivatives",
        "harmonicity_judgement": "an explicit harmonicity judgement",
        "first_second_derivatives": "the first and second curve derivatives",
        "variance_identification": "the centered second moment and its identification as the variance",
        "almost_everywhere_zero": "the conclusion that the function is zero almost everywhere",
        "all_solutions": "all solutions under the original conditions",
        "exhaustive_result": "the complete exhaustive set of possible results",
        "phrase_decomposition": "the complete phrase decomposition",
        "encoded_string": "the complete encoded bit string",
        "reasoning": "the necessary complete justification",
        "construction_object": "an explicit constructed object",
        "construction_check": "verification of every construction condition",
        "method_formula": "the specified method or iteration formula",
        "first_iteration": "the requested first iterate",
        "differential_equation_substitution": "direct substitution of the explicit solution and its derivatives into the differential equation",
        "initial_condition_check": "a direct check of the initial condition",
        "boundary_value_or_exponential_martingale": "the requested boundary-value problem or exponential martingale with its conditions",
        "exact_and_approximate": "both exact and approximate values",
        "domain_or_conditions": "the domain or applicability conditions",
        "maximal_interval_and_one_sided_part": "both the full maximal interval containing the initial point and the requested one-sided interval",
        "unit": "the required unit",
    }
    table = en if language == "en" else zh
    if item.name.startswith("decimal_places_"):
        places = item.name.rsplit("_", 1)[-1]
        return f"{places} decimal places" if language == "en" else f"保留{places}位小数"
    if item.name.startswith("target_"):
        symbol = item.name[len("target_"):]
        return f"the requested value of {symbol}" if language == "en" else f"所求量 {symbol}"
    if item.name.startswith("parameter_dependency_"):
        symbol = item.name[len("parameter_dependency_"):]
        return (
            f"a result retaining its required dependence on {symbol}"
            if language == "en"
            else f"结果中保留参数 {symbol} 的依赖关系"
        )
    if item.name.startswith("support_anchor_"):
        term = item.alternatives[0][0] if item.alternatives else ""
        return f"explicit use of {term}" if language == "en" else f"明确使用 {term}"
    if item.name == "normal_equation":
        return "the requested normal equation" if language == "en" else "题目要求的正规方程"
    if item.name == "coefficient_estimate":
        return "the explicit requested coefficient estimate" if language == "en" else "明确的所求系数估计值"
    if item.name == "sturm_liouville_argument":
        return "the requested Sturm-Liouville reduction" if language == "en" else "题目要求的 Sturm-Liouville 化简"
    if item.name == "dual_certificate":
        return "the requested dual certificate" if language == "en" else "题目要求的对偶证书"
    if item.name == "stationary_distribution":
        return "the explicit stationary distribution" if language == "en" else "明确的平稳分布"
    if item.name == "detailed_balance_check":
        return "an explicit detailed-balance equality" if language == "en" else "细致平衡等式核验"
    if item.name == "stability_function":
        return "the explicit stability function R(z)" if language == "en" else "明确的稳定函数 R(z)"
    if item.name == "stability_infinity_limit":
        return (
            "the explicit limit of R(z) at infinity required for L-stability"
            if language == "en"
            else "L-稳定性所需的稳定函数无穷远极限"
        )
    if item.name == "stability_boundary_equation":
        return "the exact stability-boundary equation" if language == "en" else "精确的稳定边界方程"
    if item.name == "closed_stability_interval":
        return (
            "the closed negative-real-axis stability interval, including both endpoints"
            if language == "en"
            else "包含两个端点的负实轴闭稳定区间"
        )
    specialized = {
        "multistep_characteristic_equation": (
            "the linear-multistep characteristic equation",
            "线性多步法的特征方程",
        ),
        "stability_boundary_parametrization": (
            "the unit-circle stability-boundary parametrization",
            "单位圆稳定边界参数式",
        ),
        "zero_stability": ("the zero-stability conclusion and root check", "零稳定结论及根条件核验"),
        "method_order": ("the method order", "方法阶数"),
        "a_stability_judgement": ("the explicit A-stability judgement", "明确的 A-稳定性判断"),
        "iteration_matrix": ("the iteration matrix", "迭代矩阵"),
        "spectral_radius": ("the spectral radius", "谱半径"),
        "requested_iterates": ("all explicitly requested iterates", "题目明确要求的各次迭代值"),
        "quadrature_nodes": ("all requested quadrature nodes", "全部求积节点"),
        "quadrature_weights": ("all requested quadrature weights", "全部求积权重"),
        "quadrature_value": ("the quadrature value Q", "求积值 Q"),
        "quadrature_error": ("the exact signed quadrature error", "带符号的精确求积误差"),
        "curvature_function": ("the full curvature function", "完整的曲率函数"),
        "curvature_point_value": ("the requested curvature at the stated point", "指定点处的曲率值"),
        "dual_optimality_check": (
            "dual feasibility and equality of primal and dual objective values",
            "对偶可行性及原对偶目标值相等的核验",
        ),
        "almost_everywhere_limit": ("the almost-everywhere limit", "几乎处处极限"),
        "uniform_integrability_check": (
            "a direct tail or small-set verification of uniform integrability",
            "按尾积分或小集合定义核验一致可积性",
        ),
        "l1_nonconvergence": ("the explicit failure of L1 convergence", "明确说明不在 L1 中收敛"),
        "l1_limit_conclusion": ("an explicit conclusion about the L1 limit", "明确说明 L1 极限是否存在及其值"),
        "uniform_convergence_scope_reason": ("separate reasons for local uniform convergence and global failure", "分别说明局部一致收敛与整体不一致收敛的原因"),
        "executable_insertion_step": ("an executable insertion rule with a selection criterion and placement", "含选点准则与落位规则的可执行插入步骤"),
        "series_sum_function": ("the series sum function", "级数的和函数"),
        "local_uniform_convergence": (
            "the uniform-convergence conclusion on every stated compact subinterval",
            "每个指定紧子区间上的一致收敛结论",
        ),
        "global_nonuniform_convergence": (
            "the non-uniform-convergence conclusion on the full stated interval",
            "整个指定区间上的不一致收敛结论",
        ),
        "operator_norm": ("the operator norm", "算子范数"),
        "operator_spectrum": ("the full operator spectrum", "算子的完整谱"),
        "point_spectrum": ("the point spectrum", "点谱"),
        "jordan_blocks": ("all Jordan block sizes", "全部 Jordan 块大小"),
        "operator_rank": ("the requested rank", "题目要求的秩"),
        "minimal_polynomial": ("the minimal polynomial", "最小多项式"),
    }
    if item.name in specialized:
        english, chinese = specialized[item.name]
        return english if language == "en" else chinese
    if item.name == "smith_normal_form":
        return "the explicit Smith normal form" if language == "en" else "明确的 Smith 标准形"
    if item.name == "cokernel_structure":
        return "the resulting cokernel decomposition" if language == "en" else "由此得到的余核分解"
    if item.name == "wasserstein_squared_value":
        return "the explicit squared Wasserstein distance" if language == "en" else "明确的 Wasserstein 距离平方"
    if item.name == "optimal_transport_map":
        return "the explicit optimal transport map" if language == "en" else "明确的最优传输映射"
    if item.name == "umvu_estimator":
        return "the explicit UMVU estimator" if language == "en" else "明确的 UMVU 估计量"
    return table.get(item.name, item.name)


def _has_reasoning(value: str) -> bool:
    if re.search(
        r"因为|由于|根据|由.*(?:得|给出)|"
        r"由[^。；;\n]{0,60}(?:定义|定理|引理|性质|公式|方法|原理)|"
        r"(?:^|[，。；;\s])因(?=\s|\$|\\|[A-Za-z0-9])|"
        r"所以|故|因此|从而|推出|假设|反设|若.*则|矛盾|"
        r"代入|分别取|令.*则|取.*得|计算得|展开得|"
        r"(?:分离变量|积分因子|特征方程|配方|因式分解|消元|换元|求导|积分)"
        r"[^。；;\n]{0,80}(?:得|得到|可得)|"
        r"\bby\b[^.\n]{0,80}\b(?:definition|theorem|lemma|property)\b|"
        r"\b(?:because|since|therefore|hence|thus|by|assume|suppose|contradiction|"
        r"implies?|follows from|if\b.*\bthen|substitut\w*|setting|evaluat\w*|"
        r"expanding)\b",
        value,
        re.IGNORECASE | re.DOTALL,
    ):
        return True

    if re.search(
        r"(?:迭代公式|迭代式|递推公式)[^。；;\n]{0,180}(?<![<>!])=(?!=)|"
        r"\b(?:iteration\s+(?:formula|scheme)|recurrence)\b"
        r"[^.;\n]{0,180}(?<![<>!])=(?!=)|"
        r"(?:逐一)?检查[^。；;\n]{0,100}(?:均)?不是根|"
        r"(?:代入|取)\s*[-+]?\d[^。；;\n]{0,100}(?:不是根|不为\s*0)|"
        r"\b(?:checking|substituting)\b[^.;\n]{0,100}"
        r"\b(?:not\s+(?:a\s+)?root|nonzero)\b",
        value,
        re.IGNORECASE,
    ):
        return True

    relation = re.compile(r"(?<![<>!])=(?!=)|≤|≥|<|>|\\(?:leq|geq|implies)")
    # Bounds such as ``\sum_{i=1}^n`` and ``\prod_{k=0}^m`` are notation,
    # not a second derivation step.  Counting their internal equality made a
    # bare estimator formula look like a proof.
    relation_text = re.sub(
        r"_(?:\{\s*)?[A-Za-z][A-Za-z0-9_]*\s*=\s*[^{}\s]+(?:\s*\})?",
        "_{index-bound}",
        str(value or ""),
    )
    segments = [
        item.strip()
        for item in re.split(r"[\n；;，,。]", relation_text)
        if item.strip()
    ]
    relation_counts = [len(relation.findall(item)) for item in segments]
    if re.search(
        r"(?:检验|验证)[^。；;\n]{1,160}(?:得到|给出|可得|表明|推出)|"
        r"\b(?:testing|verifying)[^.\n]{1,160}\b(?:gives?|yields?|shows?|confirms?|implies?)\b",
        value,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"检验|验证|\b(?:testing|verif\w*)\b", value, re.IGNORECASE):
        if sum(relation_counts) >= 1:
            return True
    # A genuine equality/inequality chain is itself a compact derivation.
    if any(count >= 2 for count in relation_counts):
        return True

    # Multiple bare terminal assignments such as "a=1/2, b=1/2" only state
    # the answer.  At least one relation must instead encode an intermediate
    # constraint, transformed expression, sum, integral, derivative, or limit.
    total_relations = sum(relation_counts)
    if total_relations < 2:
        return False
    for segment, count in zip(segments, relation_counts):
        if not count:
            continue
        lhs = relation.split(segment, maxsplit=1)[0]
        if re.search(
            r"[+\-*/^']|\\(?:int|sum|prod|lim|frac|sqrt|det|operatorname)\b|"
            r"(?:^|\W)(?:tr|det|rank|Var|Cov|E)\s*\(",
            lhs,
            re.IGNORECASE,
        ):
            return True
    return False


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").casefold()).replace("−", "-")


def _support_anchor_matches(term: str, value: str) -> bool:
    anchor = _compact(value).replace("的", "")
    normalized_term = _compact(term).replace("的", "")
    if normalized_term and normalized_term in anchor:
        return True
    aliases = (
        (
            r"欧拉函数|euler(?:'s)?\s+(?:totient|phi)\s+function|totient\s+function",
            r"(?:\\(?:var)?phi|[φΦ]|(?<![A-Za-z])phi)\s*(?:\(|_?\{?)",
        ),
        (
            r"中心极限定理|central\s+limit\s+theorem",
            r"中心极限定理|central\s+limit\s+theorem|(?<![A-Za-z])CLT(?![A-Za-z])",
        ),
        (
            r"大数定律|law\s+of\s+large\s+numbers",
            r"大数定律|law\s+of\s+large\s+numbers|(?<![A-Za-z])(?:WLLN|SLLN|LLN)(?![A-Za-z])",
        ),
        (
            r"柯西[-—– ]?施瓦茨|cauchy[-—– ]?schwarz",
            r"柯西[-—– ]?施瓦茨|cauchy[-—– ]?schwarz|(?<![A-Za-z])C[-–—]?S(?![A-Za-z])",
        ),
        (
            r"容斥(?:原理)?|inclusion[- ]exclusion",
            r"容斥|inclusion[- ]exclusion",
        ),
    )
    return any(
        re.search(term_pattern, term, re.IGNORECASE)
        and re.search(value_pattern, value, re.IGNORECASE)
        for term_pattern, value_pattern in aliases
    )
