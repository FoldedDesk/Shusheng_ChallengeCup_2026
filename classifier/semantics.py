"""Conservative semantic facts extracted only from the current statement."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StatementSemantics:
    target: str
    variables: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    quantifiers: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    initial_data: tuple[str, ...] = ()
    boundary_data: tuple[str, ...] = ()
    requested_methods: tuple[str, ...] = ()
    named_theorems: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    precision: str = ""
    unit: str = ""
    ambiguity_flags: tuple[str, ...] = ()

    def trace_content(self) -> dict:
        return {
            "target": self.target[:600],
            "variables": list(self.variables),
            "domains": list(self.domains),
            "quantifiers": list(self.quantifiers),
            "relations": list(self.relations),
            "initial_data": list(self.initial_data),
            "boundary_data": list(self.boundary_data),
            "requested_methods": list(self.requested_methods),
            "named_theorems": list(self.named_theorems),
            "assumptions": list(self.assumptions),
            "precision": self.precision,
            "unit": self.unit,
            "ambiguity_flags": list(self.ambiguity_flags),
        }

    def prompt_context(self, language: str) -> str:
        """Render compact hints; the original statement remains authoritative."""
        labels = {
            "zh": {
                "target": "目标",
                "variables": "变量",
                "domains": "定义域/对象域",
                "quantifiers": "量词",
                "relations": "需保持的关系",
                "initial_data": "初值",
                "boundary_data": "边界条件",
                "methods": "指定方法",
                "theorems": "题面点名的定理",
                "assumptions": "显式假设",
                "precision": "精度",
                "unit": "单位",
            },
            "en": {
                "target": "Target",
                "variables": "Variables",
                "domains": "Domains",
                "quantifiers": "Quantifiers",
                "relations": "Relations to preserve",
                "initial_data": "Initial data",
                "boundary_data": "Boundary data",
                "methods": "Required methods",
                "theorems": "Theorems named in the statement",
                "assumptions": "Explicit assumptions",
                "precision": "Precision",
                "unit": "Unit",
            },
        }["en" if language == "en" else "zh"]
        values = (
            ("target", (self.target,)),
            ("variables", self.variables),
            ("domains", self.domains),
            ("quantifiers", self.quantifiers),
            ("relations", self.relations),
            ("initial_data", self.initial_data),
            ("boundary_data", self.boundary_data),
            ("methods", self.requested_methods),
            ("theorems", self.named_theorems),
            ("assumptions", self.assumptions),
            ("precision", (self.precision,) if self.precision else ()),
            ("unit", (self.unit,) if self.unit else ()),
        )
        return "\n".join(
            f"- {labels[key]}: {'; '.join(value for value in items if value)}"
            for key, items in values
            if any(items)
        )


_DOMAIN_PATTERNS = (
    ("positive integers", r"正整数|\\mathbb\s*\{?Z\}?\s*_?\s*\+|\bpositive integers?\b"),
    ("nonnegative integers", r"非负整数|\\mathbb\s*\{?Z\}?\s*_?\s*(?:\\geq|≥)\s*0|\bnonnegative integers?\b"),
    ("integers", r"(?<!正)(?<!负)整数|\\mathbb\s*\{?Z\}?|\bintegers?\b"),
    ("real numbers", r"实数|\\mathbb\s*\{?R\}?|\breal numbers?\b|\breal-valued\b"),
    ("complex numbers", r"复数|\\mathbb\s*\{?C\}?|\bcomplex numbers?\b|\bcomplex-valued\b"),
    ("rational numbers", r"有理数|\\mathbb\s*\{?Q\}?|\brational numbers?\b"),
    ("almost everywhere", r"几乎处处|几乎到处|\ba\.?e\.?\b|\balmost everywhere\b"),
    ("independent", r"相互独立|互相独立|\bindependent\b"),
    ("distinct", r"互不相同|两两不同|\b(?:pairwise )?distinct\b"),
    ("measurable", r"可测|\bmeasurable\b"),
    ("continuous", r"连续|\bcontinuous\b"),
    ("compact", r"紧致|紧集|\bcompact\b"),
)

_METHOD_PATTERNS = (
    ("Newton iteration", r"牛顿(?:迭代)?法|Newton(?:'s)? method|Newton iteration"),
    ("bisection", r"二分法|bisection"),
    ("secant method", r"割线法|secant method"),
    ("Euler method", r"欧拉法|Euler(?:'s)? method"),
    ("Runge-Kutta", r"Runge.?Kutta|龙格.?库塔"),
    ("finite differences", r"有限差分|finite differences?"),
    ("residue theorem", r"留数定理|residue theorem"),
    ("Cauchy integral formula", r"柯西积分公式|Cauchy integral formula"),
    ("matrix-tree theorem", r"矩阵树定理|matrix[- ]tree theorem"),
    ("generating functions", r"生成函数|generating functions?"),
    ("inclusion-exclusion", r"容斥(?:原理)?|inclusion[- ]exclusion"),
    ("characteristics", r"特征线法|method of characteristics"),
    ("separation of variables", r"分离变量法|separation of variables"),
    ("maximum likelihood", r"极大似然|maximum likelihood"),
    ("Smith normal form", r"Smith\s*标准形|Smith normal form"),
)

_THEOREM_PATTERNS = (
    ("dominated convergence theorem", r"支配收敛定理|dominated convergence theorem"),
    ("monotone convergence theorem", r"单调收敛定理|monotone convergence theorem"),
    ("Fatou lemma", r"Fatou\s*引理|Fatou(?:'s)? lemma"),
    ("Fubini theorem", r"Fubini\s*定理|Fubini(?:'s)? theorem"),
    ("Tonelli theorem", r"Tonelli\s*定理|Tonelli(?:'s)? theorem"),
    ("Radon-Nikodym theorem", r"Radon.?Nikodym\s*定理|Radon[- ]Nikodym theorem"),
    ("Gauss-Bonnet theorem", r"Gauss.?Bonnet\s*定理|高斯.?博内定理"),
    ("Sylow theorem", r"Sylow\s*定理|西罗定理"),
    ("isomorphism theorem", r"同构定理|isomorphism theorem"),
    ("Hahn-Banach theorem", r"Hahn.?Banach\s*定理"),
    ("open mapping theorem", r"开映射定理|open mapping theorem"),
    ("closed graph theorem", r"闭图像定理|closed graph theorem"),
    ("central limit theorem", r"中心极限定理|central limit theorem"),
    ("law of large numbers", r"大数定律|law of large numbers"),
    ("Stokes theorem", r"Stokes\s*定理|斯托克斯定理"),
    ("Green theorem", r"Green\s*定理|格林定理"),
)


def extract_statement_semantics(
    problem: str,
    target: str,
    *,
    subject_confidence: str = "",
) -> StatementSemantics:
    text = str(problem or "").strip()
    target_text = str(target or "").strip()
    relations = _relations(text)
    initial = tuple(item for item in relations if _is_initial_data(item))
    boundary = tuple(item for item in relations if _is_boundary_data(item, text))
    domains = tuple(name for name, pattern in _DOMAIN_PATTERNS if re.search(pattern, text, re.IGNORECASE))
    methods = tuple(name for name, pattern in _METHOD_PATTERNS if re.search(pattern, text, re.IGNORECASE))
    theorems = tuple(name for name, pattern in _THEOREM_PATTERNS if re.search(pattern, text, re.IGNORECASE))
    quantifiers = _quantifiers(text)
    assumptions = _assumptions(text)
    precision = _precision(text)
    unit = _unit(text)
    variables = _variables((*relations, target_text))
    ambiguities: list[str] = []
    if not target_text:
        ambiguities.append("target_not_explicit")
    if subject_confidence == "low":
        ambiguities.append("low_subject_confidence")
    if len(relations) >= 7:
        ambiguities.append("many_relations")
    if len(domains) >= 3:
        ambiguities.append("multiple_domains")
    return StatementSemantics(
        target=target_text,
        variables=variables,
        domains=domains,
        quantifiers=quantifiers,
        relations=relations,
        initial_data=initial,
        boundary_data=boundary,
        requested_methods=methods,
        named_theorems=theorems,
        assumptions=assumptions,
        precision=precision,
        unit=unit,
        ambiguity_flags=tuple(dict.fromkeys(ambiguities)),
    )


def _relations(text: str) -> tuple[str, ...]:
    fragments: list[str] = []
    for match in re.finditer(
        r"\$([^$\n]{1,240})\$|\\\((.{1,240}?)\\\)|\\\[(.{1,240}?)\\\]",
        text,
        re.DOTALL,
    ):
        fragment = next(group for group in match.groups() if group is not None).strip()
        if _has_relation(fragment):
            fragments.append(fragment)
    for clause in re.split(r"[。！？!?；;\n]+", text):
        candidate = clause.strip()
        if not candidate or len(candidate) > 280 or not _has_relation(candidate):
            continue
        candidate = re.sub(r"^.{0,100}?(?=[A-Za-z\\][^=<>≤≥]{0,50}(?:=|≤|≥|<|>))", "", candidate)
        candidate = candidate.strip(" ，,:：")
        if candidate:
            fragments.append(candidate)
    return tuple(dict.fromkeys(fragments))[:12]


def _has_relation(value: str) -> bool:
    return bool(re.search(r"(?<![<>!])=(?!=)|≤|≥|(?<!\\)[<>]|\\(?:leq|geq|in|subset|sim)\b", value))


def _variables(values: tuple[str, ...]) -> tuple[str, ...]:
    variables: list[str] = []
    for value in values:
        cleaned = re.sub(r"\\(?:mathbb|mathrm|operatorname|text)\s*\{[^{}]*\}", "", value)
        for symbol in re.findall(r"(?<![A-Za-z\\])([A-Za-z])(?:_\{?[A-Za-z0-9]+\}?)?(?![A-Za-z])", cleaned):
            if symbol.casefold() in {"d", "e", "i"}:
                continue
            variables.append(symbol)
    return tuple(dict.fromkeys(variables))[:12]


def _quantifiers(text: str) -> tuple[str, ...]:
    patterns = (
        ("for all", r"任意|所有|对每个|\\forall|\b(?:for all|every|each)\b"),
        ("exists", r"存在|至少一个|\\exists|\b(?:there exists|at least one)\b"),
        ("unique", r"唯一|恰有一个|\\exists\s*!|\bunique(?:ly)?\b|\bexactly one\b"),
        ("if and only if", r"当且仅当|充要条件|\\iff|\bif and only if\b"),
        ("maximum", r"最大|至多|\\max|\b(?:maximum|at most)\b"),
        ("minimum", r"最小|至少|\\min|\b(?:minimum|at least)\b"),
    )
    return tuple(name for name, pattern in patterns if re.search(pattern, text, re.IGNORECASE))


def _assumptions(text: str) -> tuple[str, ...]:
    assumptions: list[str] = []
    for clause in re.split(r"[。！？!?；;\n]+", text):
        candidate = clause.strip()
        if not candidate or len(candidate) > 240:
            continue
        if re.search(
            r"^(?:设|假设|已知|给定|令)|(?:其中|满足|使得|条件为)|"
            r"\b(?:assume|suppose|given|where|such that|subject to)\b",
            candidate,
            re.IGNORECASE,
        ):
            assumptions.append(candidate)
    return tuple(dict.fromkeys(assumptions))[:6]


def _is_initial_data(relation: str) -> bool:
    return bool(re.search(
        r"(?:[yux]\s*(?:['′]{0,3}|\^\{?\(?\d+\)?\}?)?\s*\(\s*(?:0|t_?0|x_?0)\s*\)|"
        r"[xuy]_?\{?0\}?)\s*=",
        relation,
        re.IGNORECASE,
    ))


def _is_boundary_data(relation: str, text: str) -> bool:
    if re.search(r"边界|边值|boundary|\\partial|u\s*\|", relation, re.IGNORECASE):
        return True
    return bool(
        re.search(r"边界条件|边值问题|boundary condition|boundary value", text, re.IGNORECASE)
        and re.search(r"[uUyY]\s*\(", relation)
    )


def _precision(text: str) -> str:
    match = re.search(
        r"(?:保留|精确到)\s*(\d+)\s*位小数|"
        r"(?:误差|精度)(?:不超过|小于|为)?\s*([^，,。；;\s]{1,24})|"
        r"(?:to|give)\s*(\d+)\s*decimal places?|"
        r"(?:absolute|relative)?\s*(?:error|tolerance)\s*(?:of|below|less than|<=?)?\s*([^,.;\s]{1,24})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    groups = match.groups()
    if groups[0] or groups[2]:
        return f"{groups[0] or groups[2]} decimal places"
    return groups[1] or groups[3] or ""


def _unit(text: str) -> str:
    match = re.search(
        r"单位(?:为|是|用)\s*([\u4e00-\u9fffA-Za-z%°]{1,12})|"
        r"以\s*([\u4e00-\u9fffA-Za-z%°]{1,12})(?:为单位|计|表示)|"
        r"\b(?:in|measured in)\s+(seconds?|minutes?|hours?|meters?|centimeters?|degrees?|percent)\b",
        text,
        re.IGNORECASE,
    )
    return next((group for group in match.groups() if group), "") if match else ""
