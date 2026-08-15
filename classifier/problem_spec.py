"""Single-problem solve blueprint and gradable answer contract."""

from __future__ import annotations

from dataclasses import dataclass
import re

from classifier.choice import option_labels
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
            return bool(re.search(r"[-+]?\d|\\frac|\\sqrt|\\pi|π|∞|\\infty", value))
        if self.name == "all_solutions":
            return bool(re.search(r"所有|全部|解集|无解|不存在|\ball\b|solution set|no solutions?", value, re.IGNORECASE)) or len(re.findall(r"(?<![A-Za-z])[xyz]\s*=", value)) >= 1
        if self.name == "judgement":
            return bool(re.search(r"是|否|正确|错误|成立|不成立|可|不可|收敛|发散|true|false|yes|no|holds?|does not", value, re.IGNORECASE))
        if self.name == "reasoning":
            return _has_reasoning(value)
        if self.name == "construction_object":
            return bool(re.search(r"取|令|定义|构造|例如|\b(?:take|let|define|construct|example)\b|(?<![<>!])=(?!=)|[\[{]", value, re.IGNORECASE))
        if self.name == "construction_check":
            return bool(re.search(r"满足|验证|检查|代入|成立|\b(?:satisf|verify|check|substitut|holds?)\w*\b", value, re.IGNORECASE))
        if self.name == "choice_labels":
            return bool(re.search(r"(?<![A-Za-z])[A-E](?![A-Za-z])", value))
        if self.name.startswith("decimal_places_"):
            places = int(self.name.rsplit("_", 1)[-1])
            decimals = re.findall(r"[-+]?\d+\.(\d+)", value)
            return any(len(item) == places for item in decimals)
        if self.name == "unit":
            return any(_compact(term) in compact for alt in self.alternatives for term in alt)
        if self.name == "method_formula":
            return bool(re.search(r"[A-Za-z]_(?:\{?n\}?|k)\s*\+\s*1|迭代公式|recurrence|iteration formula", value, re.IGNORECASE))
        if self.name == "first_iteration":
            return bool(re.search(r"[xuy]_?\{?1\}?\s*=|第一次迭代|first iterate", value, re.IGNORECASE))
        if self.name == "domain_or_conditions":
            return bool(re.search(r"定义域|条件|其中|当且仅当|subject to|domain|provided that|for .* such that", value, re.IGNORECASE))
        if self.name == "exact_and_approximate":
            exact = bool(re.search(r"精确|exact|\\frac|\\sqrt|\\pi", value, re.IGNORECASE))
            approximate = bool(re.search(r"≈|约为|近似|approx", value, re.IGNORECASE) or re.search(r"\d+\.\d+", value))
            return exact and approximate
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

    def trace_content(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "strict": self.strict,
        }


@dataclass(frozen=True)
class AnswerContract:
    language: str
    mode: str
    wrapper: str
    parts: tuple[AnswerPart, ...]
    explicit_support_requirements: tuple[str, ...] = ()
    result_kind: str = "expression"

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
    r"\s*(?:请|务必)?(?:将)?(?:最终)?答案(?:写|填|放|置)(?:在|于)?\s*(?:方框|框内|\\boxed\s*\{\s*\})(?:中|内)?\s*[。.!]?\s*$",
))


def _strip_trailing_answer_instructions(text: str) -> str:
    value = str(text or "").strip()
    for pattern in _TRAILING_FORMAT:
        value = pattern.sub("", value).strip()
    return value or str(text or "").strip()


def build_problem_spec(problem: str) -> ProblemSpec:
    original = str(problem or "").strip()
    text = _strip_trailing_answer_instructions(original)
    profile = classify_profile(text)
    target = extract_target_clause(text) or text
    semantics = extract_statement_semantics(
        text,
        target,
        subject_confidence=profile.subject_confidence,
    )
    requirements = _requirements(text, target, profile)
    kind = _goal_kind(profile)
    goal = Goal("g1", target[:1800], profile.answer_shape, kind, (), tuple(requirements))
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
    contract = AnswerContract(profile.language, mode, wrapper, parts, support, profile.result_kind)
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
    if shape == "choice":
        items.append(Requirement("choice_labels", strict=True))
    if shape == "truth":
        items.append(Requirement("judgement", strict=True))
    if shape == "roots" or re.search(r"全部解|所有解|all solutions?|all roots?", target, re.IGNORECASE):
        items.append(Requirement("all_solutions", strict=True))
    explicit_support = re.search(
        r"(?:要求|必须|须|需|应当)(?:严格|分别|逐项|完整地?)?"
        r"(?:使用|用|从|通过|由|以|作|解|识别|构造|给出|推导|验证|核对|检查|说明|证明|计算|归一化)"
        r"[^。；;\n]{0,220}|"
        r"(?:证明|论证|推导|计算|归一化)(?:须|必须|应当|要求)|"
        r"(?:并|且|同时|还)(?:请|须|需|要|应当)?(?:"
        r"推导|证明|论证|验证|核对|检查|说明[^。；;\n]{0,40}(?:理由|过程|步骤|依据|计算)|"
        r"(?:给出|写出|展示|列出)[^。；;\n]{0,80}"
        r"(?:计算|推导|证明|论证|过程|步骤|依据|留数|节点|权重|精度|控制函数|上界|估计))|"
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,140}"
        r"(?:控制函数|控制量|可积上界|支配函数|dominating function|integrable bound)|"
        r"(?:用|利用|根据|通过|由|从)[^。；;\n]{1,120}"
        r"(?:定理|公式|方程|方法|法|定义|矩阵|变换|基本形式|核|原理|"
        r"theorem|formula|equations?|method|definition|transform|kernel|principle)"
        r"[^。；;\n]{0,80}(?:求|计算|推导|证明|说明|验证|find|compute|derive|prove|explain|verify)|"
        r"\band\s+(?:derive|prove|justify|verify|check|show\s+(?:the\s+)?"
        r"(?:calculation|derivation|proof|steps?|work))\b",
        text,
        re.IGNORECASE,
    )
    if profile.task_kind in {"proof", "derivation", "explanation"} or explicit_support or re.search(
        r"说明理由|给出证明|完整(?:论证|证明|推导)|严格(?:论证|证明|推导)|"
        r"(?:证明|论证|推导)须|须[^。；;\n]{0,40}(?:证明|论证|推导)|"
        r"(?:要求|必须|须|需|应当)[^。；;\n]{0,160}(?:证明|论证|推导|验证|核对|检查|说明)|"
        r"证明.*(?:所有|唯一)|justify|give (?:a )?proof|show your work|"
        r"\bwith\s+(?:a\s+)?(?:complete\s+|rigorous\s+)?"
        r"(?:proof|derivation|argument|justification)\b|"
        r"(?:complete|rigorous)\s+(?:proof|derivation|argument|normalization|calculation|justification)|"
        r"(?:the\s+)?(?:proof|derivation|argument|normalization|calculation|justification)\s+"
        r"(?:must|should|is required)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("reasoning", strict=True, category="support"))
    if profile.task_kind == "construction" or re.search(r"构造|举例|反例|construct|counterexample", target, re.IGNORECASE):
        items.extend((
            Requirement("construction_object", strict=True),
            Requirement("construction_check", strict=True, category="support"),
        ))
    if re.search(r"牛顿法|二分法|割线法|欧拉法|迭代公式|newton|bisection|secant|euler method|iteration formula", text, re.IGNORECASE):
        items.append(Requirement("method_formula", strict=True))
        if re.search(r"第一次迭代|第一步迭代|x_?\{?1\}?|first iterate|first iteration", text, re.IGNORECASE):
            items.append(Requirement("first_iteration", strict=True))
    if re.search(
        r"精确值[^。；;\n]{0,100}近似值|近似值[^。；;\n]{0,100}精确值|"
        r"exact\s+(?:value|form)[^.\n]{0,100}approximate\s+value|"
        r"approximate\s+value[^.\n]{0,100}exact\s+(?:value|form)",
        text,
        re.IGNORECASE,
    ):
        items.append(Requirement("exact_and_approximate", strict=True))
    places = re.search(r"(?:保留|精确到)\s*(\d+)\s*位小数|(?:to|give)\s*(\d+)\s*decimal places?", text, re.IGNORECASE)
    if places:
        count = next(group for group in places.groups() if group)
        items.append(Requirement(f"decimal_places_{count}", strict=True, category="format"))
    unit = _requested_unit(text)
    if unit:
        items.append(Requirement("unit", ((unit,),), strict=True, category="format"))
    if re.search(r"并说明.*条件|写明.*定义域|注明.*范围|state.*conditions?|include.*domain", text, re.IGNORECASE):
        items.append(Requirement("domain_or_conditions", strict=True))
    symbols = _requested_symbols(target)
    for symbol in symbols[:4]:
        items.append(Requirement(
            f"target_{symbol.casefold()}",
            ((f"{symbol}=",), (f"{symbol} =",), (f"{symbol}为",), (f"{symbol} is",)),
            strict=len(symbols) > 1,
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


def _methods(
    profile: ProblemProfile,
    semantics: StatementSemantics | None = None,
) -> tuple[str, str]:
    by_topic = {
        "numerical_method": ("derive the requested iteration and compute with the stated data", "independent residual and error check"),
        "calculus": ("definition or standard theorem with domain checks", "symbolic differentiation, substitution, or boundary check"),
        "equation": ("algebraic reduction with complete branch analysis", "substitute every candidate into the original equation"),
        "linear_algebra": ("row reduction and invariant computation", "independent determinant, rank, or polynomial check"),
        "combinatorics": ("bijection, recurrence, or inclusion-exclusion", "small-case enumeration and symmetry audit"),
        "graph": ("graph invariant or structural theorem", "direct small graph or matrix-tree check"),
        "probability": ("condition on a clear sample space", "normalization and complementary-event check"),
        "optimization": ("derive a sharp bound and attainability", "boundary and equality-case verification"),
        "proof": ("minimal sufficient lemma with hypotheses checked", "counterexample search and converse audit"),
        "construction": ("explicit construction", "verify every condition on the same object"),
        "choice": ("evaluate every option from definitions", "independent option-by-option falsification"),
    }
    if profile.topic in by_topic and profile.topic not in {"proof"}:
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
    if constraints:
        risks.append("domain_constraints")
    if len(text) >= 320:
        risks.append("long_statement")
    if sum(item.strict for item in requirements) >= 3:
        risks.append("multiple_answer_obligations")
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
    return profile.topic in {"calculus", "equation", "linear_algebra", "general"}


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
    return next((group for group in match.groups() if group), "")


def _requested_symbols(target: str) -> tuple[str, ...]:
    matches = re.findall(
        r"(?:求|计算|确定|写出|给出|find|compute|determine|give)\s*"
        r"(?:the\s+value\s+of\s+)?([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)",
        target,
        re.IGNORECASE,
    )
    return tuple(dict.fromkeys(matches))


def _requirement_description(item: Requirement, language: str) -> str:
    zh = {
        "result_present": "明确最终结论",
        "numeric_result": "所求数值",
        "choice_labels": "完整选项标签集合",
        "judgement": "带对象的明确判断",
        "all_solutions": "全部解并执行原条件",
        "reasoning": "必要且完整的论证",
        "construction_object": "明确构造对象",
        "construction_check": "逐条验证构造条件",
        "method_formula": "题目指定的方法或迭代公式",
        "first_iteration": "指定的第一次迭代值",
        "exact_and_approximate": "精确值与近似值",
        "domain_or_conditions": "定义域或适用条件",
        "unit": "题目要求的单位",
    }
    en = {
        "result_present": "an explicit final conclusion",
        "numeric_result": "the requested numeric value",
        "choice_labels": "the complete set of option labels",
        "judgement": "an explicit judgement naming its object",
        "all_solutions": "all solutions under the original conditions",
        "reasoning": "the necessary complete justification",
        "construction_object": "an explicit constructed object",
        "construction_check": "verification of every construction condition",
        "method_formula": "the specified method or iteration formula",
        "first_iteration": "the requested first iterate",
        "exact_and_approximate": "both exact and approximate values",
        "domain_or_conditions": "the domain or applicability conditions",
        "unit": "the required unit",
    }
    table = en if language == "en" else zh
    if item.name.startswith("decimal_places_"):
        places = item.name.rsplit("_", 1)[-1]
        return f"{places} decimal places" if language == "en" else f"保留{places}位小数"
    if item.name.startswith("target_"):
        symbol = item.name[len("target_"):]
        return f"the requested value of {symbol}" if language == "en" else f"所求量 {symbol}"
    return table.get(item.name, item.name)


def _has_reasoning(value: str) -> bool:
    if re.search(
        r"因为|由于|根据|由.*得|所以|故|因此|从而|推出|假设|反设|若.*则|矛盾|"
        r"\b(?:because|since|therefore|hence|thus|by|assume|suppose|contradiction|"
        r"implies?|follows from|if\b.*\bthen)\b",
        value,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    relations = len(re.findall(r"(?<![<>!])=(?!=)|≤|≥|<|>|\\(?:leq|geq|implies)", value))
    return relations >= 2


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").casefold()).replace("−", "-")
