"""Structured, local-only understanding of a mathematics problem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.choice import answer_choice_labels
from classifier.profile import ProblemProfile, classify_profile
from classifier.target import extract_target_clause


_SUPPORT_REQUIREMENT_NAMES = frozenset({
    "reasoning",
    "position_selection",
    "variable_shift",
    "stars_and_bars",
    "counting_method",
    "inclusion_exclusion",
    "pairing_step",
    "case_split",
    "position_compression",
    "distributive_step",
    "two_steps",
    "disjoint_decomposition",
    "combination_calculation",
    "partial_fraction",
    "pole_order",
    "pole_location",
    "cauchy_formula",
    "integrating_factor",
    "characteristic_equation",
    "separation_of_variables",
    "independent_increments",
    "inference_rule",
    "continuity_contradiction",
    "open_cover_step",
    "fixed_point_check",
    "induction",
    "pigeonhole_principle",
    "am_gm",
    "cauchy_schwarz",
    "contradiction",
})

_FORMAT_REQUIREMENT_NAMES = frozenset({"boxed_wrapper", "answer_language", "output_unit"})


@dataclass(frozen=True)
class Requirement:
    """A gradable answer obligation with acceptable local textual anchors."""

    name: str
    alternatives: tuple[tuple[str, ...], ...]
    strict: bool = False
    category: str = "auto"

    def __post_init__(self) -> None:
        if self.category == "auto":
            category = "support" if self.name in _SUPPORT_REQUIREMENT_NAMES else (
                "format" if self.name in _FORMAT_REQUIREMENT_NAMES else "result"
            )
            object.__setattr__(self, "category", category)
        elif self.category not in {"result", "support", "format"}:
            raise ValueError(f"unknown requirement category: {self.category}")

    def matches(self, compact_answer: str) -> bool:
        raw_answer = str(compact_answer or "").lower()
        normalized = _compact(raw_answer)
        latex_flat = (
            raw_answer.replace(r"\,", "")
            .replace(r"\;", "")
            .replace(r"\ ", "")
            .replace(r"\quad", "")
        )
        if self.name == "judgement":
            return bool(re.search(
                r"(?:是|否|可以|不可以|正确|错误|成立|不成立|属于|不属于|收敛|发散|"
                r"不可约|可约|有解|无解|相等|不等|存在|不存在|改变|不变|变化|"
                r"位于.*(?:内|外)|在.*(?:内部|外部)|\b(?:yes|no|true|false|"
                r"converges?|diverges?|irreducible|reducible|exists?|does\s+not\s+exist)\b)",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "reasoning":
            return bool(re.search(
                r"(?:因为|由于|由|依据|根据|所以|故|因此|推出|可得|implies|because|since|therefore|hence|by\s+)",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "iteration_formula":
            return bool(re.search(
                r"x\^?[a-z]\+1=[^=]{1,240}",
                normalized,
                re.IGNORECASE,
            ))
        if self.name == "first_iteration":
            return bool(re.search(
                r"(?:x1=|第一次迭代(?:值|结果)?(?:为|是|=)|"
                r"firstiteration(?:value|result)?(?:is|=))[^=]{1,120}",
                normalized,
                re.IGNORECASE,
            ))
        if self.name == "numeric_result":
            return bool(re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?", raw_answer))
        if self.name == "feasibility_or_numeric":
            return bool(re.search(
                r"是|否|可以|不可以|可行|不可行|存在|不存在|"
                r"\b(?:yes|no|possible|impossible|feasible|infeasible|exists?|does\s+not\s+exist)\b|"
                r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "alternative_result":
            return bool(re.search(
                r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?|"
                r"\b(?:does\s+not|need\s+not|not\s+necessarily)\s+exist\b|"
                r"\bno\s+such\b|不存在|不一定存在",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "field_value":
            return bool(re.search(
                r"(?:\\mathbb\s*\{?Q\}?|\bQ)\s*(?:\(|\[|\\bigl?\()",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "degree_value":
            return bool(re.search(
                r"(?:\[[^\]]+:[^\]]+\]|\bdegree\b|扩张次数|次数)\s*=\s*\d+|"
                r"[,，;；]\s*\d+\s*[,，;；]|^\s*\d+\s*$",
                latex_flat,
                re.IGNORECASE,
            ))
        if self.name == "operator_norm":
            return bool(re.search(
                r"(?:\\lVert|\\Vert|\|\|).*(?:\\rVert|\\Vert|\|\|)\s*=|"
                r"(?:算子)?范数\s*(?:为|是|=)|\boperator\s+norm\s*(?:is|=)",
                raw_answer,
                re.IGNORECASE,
            )) or _is_bare_math_expression(raw_answer)
        if self.name == "galois_verdict":
            return bool(re.search(r"是|否|\b(?:yes|no|true|false)\b", raw_answer, re.IGNORECASE))
        if self.name == "two_items":
            return bool(re.search(r"、|，|;|；|\b(?:and|or)\b|(?:方法|法)\s*(?:与|和)", raw_answer, re.IGNORECASE))
        if self.name == "phrase_decomposition":
            indexed_phrases = len(re.findall(r"\(\s*\d+\s*[,，]\s*[^)（）]+\)", raw_answer)) >= 2
            labelled_phrases = bool(re.search(
                r"(?:phrases?|短语(?:序列|列表)?)\s*"
                r"(?:(?:decomposition|分解)\s*)?(?::|：|\bare\b|为|是)\s*"
                r"(?:[^,，;；\n]+[,，]\s*)+[^;；\n]+",
                raw_answer,
                re.IGNORECASE,
            ))
            described_decomposition = bool(re.search(
                r"(?:短语|phrases?).*(?:分解|decomposition).*(?:[,，;；]|\()",
                raw_answer,
                re.IGNORECASE | re.DOTALL,
            ))
            return indexed_phrases or labelled_phrases or described_decomposition
        if self.name == "all_correct_choices":
            # The public problem does not reveal how many options are correct;
            # this obligation can validate label syntax, while the solve and
            # verification prompts must determine and return the complete set.
            return bool(answer_choice_labels(compact_answer))
        if self.name == "encoded_string":
            return bool(re.search(r"(?:[01]{3,}[\s\\,]*){2,}", raw_answer))
        if self.name == "exhaustive_result":
            return bool(re.search(
                r"(?:所有|全部|仅|只有|唯一|恰为|且无其他|任意|"
                r"\b(?:all|only|exactly|no\s+other|for\s+(?:all|some|each)|where)\b|"
                r"(?:共|合计|总计)\s*(?:为|是|=|[:：])?\s*(?:\$|\\\(|\\boxed\s*\{)?\s*\d+|"
                r"\b(?:a\s+)?total(?:\s+(?:number|count))?\s*(?:of|is|=|:)?\s*"
                r"(?:\$|\\\(|\\boxed\s*\{)?\s*\d+|"
                r"\b\d+\s+(?:items?|values?|solutions?|tuples?)?\s*in\s+total\b|"
                r"\\?\{[^{}]*\}|\.\.\.|\\ldots|\\dots|"
                r"[^,\n]+,[^,\n]+)",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "euler_formula_check":
            # A verification must expose a checkable equality, rather than
            # merely naming Euler's formula or returning the requested count.
            numeric_substitution = bool(re.search(
                r"\d+(?:\.\d+)?-\d+(?:\.\d+)?\+\d+(?:\.\d+)?=2(?:\.0+)?(?:\b|$)",
                normalized,
            ))
            symbolic_chain = (
                "v-e+f=" in normalized
                and normalized.count("=") >= 2
                and bool(re.search(r"=2(?:\.0+)?(?:\b|$)", normalized))
            )
            assigned_values = all(
                re.search(rf"(?:^|[^a-z]){symbol}=\d", normalized)
                for symbol in ("v", "e", "f")
            )
            symbolic_formula = "v-e+f=2" in normalized or "v+f=e+2" in normalized
            return numeric_substitution or symbolic_chain or (assigned_values and symbolic_formula)
        if self.name == "principal_curvatures":
            labelled_pair = all(re.search(
                rf"(?:kappa|κ|k){index}=",
                normalized,
            ) for index in (1, 2))
            named_pair = bool(re.search(
                r"(?:主曲率|principal\s+curvatures?)\s*(?:分别)?\s*(?:为|是|are|=|[:：])?\s*"
                r"[-+]?\d+(?:\.\d+)?\s*(?:[,，、]|和|与|\band\b)\s*"
                r"[-+]?\d+(?:\.\d+)?",
                raw_answer,
                re.IGNORECASE,
            ))
            both_equal = bool(re.search(
                r"(?:主曲率|principal\s+curvatures?)[^。.!?;；\n]{0,40}"
                r"(?:均|都是|相等|both|equal)[^。.!?;；\n]{0,20}[-+]?\d+(?:\.\d+)?",
                raw_answer,
                re.IGNORECASE,
            ))
            return labelled_pair or named_pair or both_equal
        if self.name == "gaussian_curvature":
            labelled_value = bool(re.search(
                r"(?:高斯曲率|gaussian\s+curvature)[^。.!?;；\n]{0,30}"
                r"(?:为|是|is|equals?|=|[:：])\s*(?:[-+]?\d|\\[A-Za-z]+|[A-Za-zκ])",
                raw_answer,
                re.IGNORECASE,
            ))
            symbolic_value = bool(re.search(
                r"(?<![A-Za-z0-9_])k\s*(?:\([^)]*\))?\s*=\s*"
                r"(?:[-+]?\d|\\[A-Za-z]+|[A-Za-zκ])",
                raw_answer,
                re.IGNORECASE,
            ))
            return labelled_value or symbolic_value
        if self.name == "surface_second_derivatives":
            derivative_symbols = any(all(
                f"{prefix}{suffix}" in normalized or f"partial{suffix}{prefix}" in normalized
                for suffix in ("xx", "xy", "yy")
            ) for prefix in ("f", "z", "r", "u"))
            derivative_triplet = derivative_symbols and normalized.count("=") >= 2
            hessian = bool(re.search(
                r"hessian|黑塞|nabla\^?2|d\^?2[a-z]",
                normalized,
                re.IGNORECASE,
            )) and "=" in raw_answer and bool(re.search(r"\d", raw_answer))
            return derivative_triplet or hessian
        if self.name == "recurrence_formula":
            return bool(re.search(r"a\s*_?\s*n\s*=|通项", raw_answer, re.IGNORECASE))
        if self.name == "solution_formula":
            return bool(re.search(r"\by\s*(?:\([^)]*\))?\s*=", raw_answer, re.IGNORECASE))
        if self.name == "parallelogram_identity":
            return all(term in normalized for term in ("u+v", "u-v", "="))
        if self.name == "pde_time_space_derivatives":
            return all(term in normalized for term in ("ut", "uxx"))
        if self.name == "laplace_second_derivatives":
            return all(term in normalized for term in ("uxx", "uyy"))
        if self.name == "first_second_derivatives":
            return bool(re.search(r"(?:一阶|['′]).*(?:二阶|['′]{2})|(?:gamma|γ)\s*['′].*(?:gamma|γ)\s*(?:''|′′)", raw_answer, re.DOTALL | re.IGNORECASE))
        if self.name == "two_steps":
            return all(term in normalized for term in ("y1", "y2")) or "两步" in raw_answer
        if self.name == "domain":
            return bool(re.search(r"(?:定义域|区间|domain|[x-z]\s*[<>≤≥≠]|\([^)]*,[^)]*\)|\[[^]]*,[^]]*\])", raw_answer, re.IGNORECASE))
        if self.name == "integral_value":
            return bool(re.search(
                r"(?:积分|integral|∫).*(?:为|=)|(?:级数|总和|结果|近似值|精确值).*(?:为|=)",
                raw_answer,
                re.IGNORECASE,
            )) or bool(re.fullmatch(
                r"\s*\$?\s*(?:[-+]?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}|"
                r"\\sqrt\{[^{}]+\}|(?:[-+]?\d+(?:\.\d+)?\s*)?(?:π|\\pi|∞|\\infty)(?:\s*[A-Za-z])?(?:/\d+)?)\s*\$?\s*[。.]?\s*",
                raw_answer,
                re.IGNORECASE,
            )) or _is_bare_math_expression(raw_answer)
        return any(all(
            term in raw_answer if term in {"{", "}", "[", "]"} else _compact(term) in normalized
            for term in option
        ) for option in self.alternatives)


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")


def _is_bare_math_expression(value: str) -> bool:
    """Recognize a complete formula without requiring a redundant result label."""
    text = str(value or "").strip().strip("$").strip()
    if text.startswith(r"\boxed{"):
        depth = 1
        closing = -1
        for index, character in enumerate(text[len(r"\boxed{"):], len(r"\boxed{")):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing == len(text) - 1:
            text = text[len(r"\boxed{"):-1].strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    commands = re.findall(r"\\([A-Za-z]+)", text)
    allowed_commands = {
        "frac", "dfrac", "tfrac", "sqrt", "pi", "infty", "log", "ln", "exp",
        "sin", "cos", "tan", "sinh", "cosh", "tanh", "lVert", "rVert", "Vert",
        "operatorname", "mathrm", "mathsf", "mathbb", "cdot",
    }
    if any(command not in allowed_commands for command in commands):
        return False
    return bool(
        re.search(r"\d|\\(?:frac|sqrt|pi|infty|log|ln|exp|sin|cos)", text)
        and re.fullmatch(r"[0-9A-Za-z_+\-*/^().,!{}\[\]\\\s]+", text)
    )


@dataclass(frozen=True)
class Goal:
    id: str
    instruction: str
    answer_shape: str
    kind: str = "result"
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
    """One independently gradable part of the requested final answer."""

    id: str
    instruction: str
    kind: str
    answer_shape: str
    required_terms: tuple[str, ...] = ()
    support_requirements: tuple[str, ...] = ()
    unit: str = ""
    validation_requirements: tuple[str, ...] = ()
    result_requirements: tuple[str, ...] = ()
    format_requirements: tuple[str, ...] = ()

    @property
    def explicit_support_requirements(self) -> tuple[str, ...]:
        return self.support_requirements

    def trace_content(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnswerContract:
    """Internal output contract derived only from the public problem text."""

    language: str
    mode: str
    wrapper: str
    answer_shape: str
    parts: tuple[AnswerPart, ...]
    explicit_support_requirements: tuple[str, ...] = ()
    unit: str = ""

    @property
    def support_requirements(self) -> tuple[str, ...]:
        return self.explicit_support_requirements

    @property
    def shape(self) -> str:
        return self.answer_shape

    def trace_content(self) -> dict:
        return {
            "language": self.language,
            "mode": self.mode,
            "wrapper": self.wrapper,
            "answer_shape": self.answer_shape,
            "parts": [part.trace_content() for part in self.parts],
            "explicit_support_requirements": list(self.explicit_support_requirements),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ProblemSpec:
    profile: ProblemProfile
    goals: tuple[Goal, ...]
    constraints: tuple[str, ...]
    risk_flags: tuple[str, ...]
    primary_method: str
    alternative_method: str
    answer_frame: AnswerFrame
    tool_can_answer_whole: bool
    risk_score: int
    verification_required: bool
    answer_contract: AnswerContract | None = None
    problem_text: str = ""

    def trace_content(self) -> dict:
        return {
            "profile": self.profile.trace_content(),
            "goal_count": len(self.goals),
            "goals": [{
                "id": goal.id,
                "kind": goal.kind,
                "instruction": goal.instruction,
                "required_terms": list(goal.required_terms),
                "requirements": [
                    {"name": item.name, "strict": item.strict, "category": item.category}
                    for item in goal.requirements
                ],
            } for goal in self.goals],
            "risk_flags": list(self.risk_flags),
            "risk_score": self.risk_score,
            "verification_required": self.verification_required,
            "primary_method": self.primary_method,
            "answer_frame": self.answer_frame.trace_content(),
            "answer_contract": self.answer_contract.trace_content() if self.answer_contract else None,
            "tool_can_answer_whole": self.tool_can_answer_whole,
        }


_METHODS = {
    "抽象代数": ("definition_and_structure", "counterexample_or_substructure_check"),
    "高等代数": ("invariant_or_polynomial_method", "direct_computation_check"),
    "线性代数": ("rank_and_dimension_method", "matrix_relation_check"),
    "离散数学": ("case_partition_or_invariant", "direct_enumeration_check"),
    "概率论": ("condition_on_events", "sample_space_check"),
    "数学分析": ("definition_or_standard_theorem", "boundary_condition_check"),
    "常微分方程": ("general_solution_then_conditions", "substitution_check"),
    "复分析": ("singularity_and_residue_method", "local_expansion_check"),
    "拓扑学": ("definition_or_open_cover_method", "counterexample_check"),
    "泛函分析": ("definition_and_completeness_method", "assumption_check"),
    "数论": ("congruence_and_factorization", "valuation_or_descent_check"),
    "初等几何": ("angle_similarity_or_cyclic_method", "coordinate_or_area_check"),
}


def build_problem_spec(problem: str) -> ProblemSpec:
    text = str(problem or "").strip()
    semantic_text = _strip_trailing_answer_instructions(text)
    profile = classify_profile(semantic_text)
    goals = _goals(semantic_text, profile)
    constraints = _constraints(semantic_text)
    risks = _risks(semantic_text, profile, len(goals))
    risk_score = _risk_score(semantic_text, profile, goals, risks)
    primary, alternative = _METHODS.get(profile.subject, ("definition_and_case_analysis", "direct_check"))
    answer_frame = _answer_frame(extract_target_clause(semantic_text), profile)
    contract = _answer_contract(text, semantic_text, profile, goals, answer_frame)
    return ProblemSpec(
        profile, tuple(goals), tuple(constraints), tuple(risks), primary, alternative,
        answer_frame, _tool_can_answer_whole(semantic_text, profile, goals),
        risk_score, risk_score >= 3, contract, semantic_text,
    )


_BOXED_VALUE = r"\\boxed\s*\{(?:[^{}]|\{[^{}]*\})*\}"
_TRAILING_ANSWER_INSTRUCTIONS = tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
    rf"\s*(?:remember\s+to\s+|please\s+)?(?:put|place|write|enter|give|provide|express|report|show|display)\s+(?:your\s+)?(?:final\s+)?answer\s+(?:in|inside|within|using|as)\s+(?:the\s+)?(?:form\s+)?(?:of\s+)?(?:{_BOXED_VALUE}|a\s+box|boxed\s+form)\s*(?:only)?\s*[.!。]?\s*$",
    rf"\s*(?:please\s+)?answer\s+(?:in|inside|within|using|with)\s+(?:the\s+)?(?:form\s+)?(?:of\s+)?(?:{_BOXED_VALUE}|a\s+box|boxed\s+form)\s*(?:only)?\s*[.!。]?\s*$",
    rf"\s*(?:your\s+)?(?:final\s+)?answer\s+(?:must|should)\s+be\s+(?:written\s+|placed\s+)?(?:in|inside|within|as)\s+(?:the\s+)?(?:form\s+)?(?:of\s+)?(?:{_BOXED_VALUE}|a\s+box|boxed)\s*[.!。]?\s*$",
    rf"\s*(?:final\s+answer|answer|最终答案|答案)\s*[:：]\s*(?:{_BOXED_VALUE})\s*[.!。]?\s*$",
    rf"\s*(?:请|务必)?(?:将)?(?:最终)?答案(?:请)?(?:写|填|放|置|表示|给出)(?:在|于|为|成|到|入)?\s*(?:{_BOXED_VALUE}|方框|框内)(?:中|内|形式)?\s*[。.!]?\s*$",
    rf"\s*(?:请|务必)?(?:在|于)\s*(?:{_BOXED_VALUE}|方框|框内)(?:中|内)?(?:作答|填写|写出答案)\s*[。.!]?\s*$",
    rf"\s*(?:最后|最终)(?:请)?(?:用|以)\s*(?:{_BOXED_VALUE}|方框|框内)(?:表示|作答|给出)?\s*[。.!]?\s*$",
))


def _strip_trailing_answer_instructions(text: str) -> str:
    """Remove only formatting-only suffixes, never mathematical answer constraints."""
    value = str(text or "").strip()
    changed = True
    while value and changed:
        changed = False
        for pattern in _TRAILING_ANSWER_INSTRUCTIONS:
            trimmed, count = pattern.subn("", value, count=1)
            if count:
                value = trimmed.rstrip(" \t\r\n；;。.!?")
                changed = True
                break
    return value or str(text or "").strip()


def _answer_frame(text: str, profile: ProblemProfile) -> AnswerFrame:
    if profile.problem_type in {"proof", "derivation", "explanation"}:
        return AnswerFrame("proof", predicate="证明结论", question_kind="proof")
    if profile.answer_shape == "truth" and _asks_for_euler_formula_check(text):
        # Formula verification is graded by the displayed equality, not by a
        # bare yes/no sentence (notably in English, where "verify" is often
        # classified as a truth-shaped request).
        return AnswerFrame("math", predicate="公式核验", question_kind="math")
    age = re.search(r"问\s*([\u4e00-\u9fff]{2,6})(?:多少|几)岁", text)
    if not age:
        age = re.search(r"([\u4e00-\u9fff]{2,6})(?:今年|的年龄|现年)", text)
    if age:
        return AnswerFrame("sentence", subject=age.group(1), predicate="年龄", unit="岁", question_kind="age")
    if re.search(r"概率.*(?:多少|几)|(?:多少|几).*概率|probability", text, re.IGNORECASE):
        return AnswerFrame("sentence", predicate="概率", question_kind="probability")
    if profile.answer_shape == "truth":
        before = re.split(
            r"是否|是不是|能否|可否|\bis\s+it\b|\bwhether\b|判断\s*[:：]?",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        subject = re.sub(r"^(?:求|判断|说明|问)\s*", "", before).strip("，,。 ")[-100:]
        return AnswerFrame("sentence", subject=subject, predicate="判断", question_kind="truth")
    if re.search(r"(?:多少|几)个", text):
        return AnswerFrame("sentence", predicate="数量", unit="个", question_kind="count")
    if re.search(r"\bnumber of\b|\bhow many\b", text, re.IGNORECASE):
        return AnswerFrame("sentence", predicate="数量", question_kind="count")
    return AnswerFrame("math", question_kind="math")

def _answer_contract(
    original_text: str,
    semantic_text: str,
    profile: ProblemProfile,
    goals: list[Goal],
    frame: AnswerFrame,
) -> AnswerContract:
    support = tuple(dict.fromkeys(
        requirement.name
        for goal in goals
        for requirement in goal.requirements
        if requirement.category == "support"
    ))
    formal_proof = (
        not _is_result_or_nonexistence_alternative(semantic_text)
        and profile.task_kind == "proof"
    )
    if formal_proof and "reasoning" not in support:
        support = (*support, "reasoning")
    mode = "proof" if formal_proof else (
        "answer_with_support"
        if support or profile.problem_type in {"derivation", "explanation"}
        else "answer_only"
    )
    unit = frame.unit or _explicit_unit(semantic_text)
    wrapper = _requested_wrapper(original_text)
    parts = tuple(AnswerPart(
        id=goal.id,
        instruction=goal.instruction,
        kind=goal.kind,
        answer_shape=goal.answer_shape,
        required_terms=goal.required_terms,
        support_requirements=tuple(dict.fromkeys((
            *(
                requirement.name
                for requirement in goal.requirements
                if requirement.category == "support"
            ),
            *(("reasoning",) if formal_proof else ()),
        ))),
        unit=_explicit_unit(goal.instruction) or (unit if len(goals) == 1 else ""),
        validation_requirements=tuple(dict.fromkeys((
            *(requirement.name for requirement in goal.requirements),
            *(("reasoning",) if formal_proof else ()),
        ))),
        result_requirements=tuple(dict.fromkeys(
            requirement.name
            for requirement in goal.requirements
            if requirement.category == "result"
        )),
        format_requirements=tuple(dict.fromkeys((
            *(requirement.name for requirement in goal.requirements if requirement.category == "format"),
            *(("boxed_wrapper",) if wrapper == "boxed" else ()),
            *(("output_unit",) if (_explicit_unit(goal.instruction) or (unit if len(goals) == 1 else "")) else ()),
        ))),
    ) for goal in goals)
    return AnswerContract(
        language=_answer_language(semantic_text, profile.language),
        mode=mode,
        wrapper=wrapper,
        answer_shape=profile.answer_shape,
        parts=parts,
        explicit_support_requirements=support,
        unit=unit,
    )


def _answer_language(text: str, fallback: str) -> str:
    """Infer prose language without treating Latin math variables as English prose."""
    value = str(text or "")
    prose = re.sub(
        r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|"
        r"\\begin\{[^{}]+\}.*?\\end\{[^{}]+\}",
        " ",
        value,
        flags=re.DOTALL,
    )
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", prose))
    english_words = len(re.findall(r"\b[A-Za-z]{2,}\b", prose))
    if chinese_chars >= 2 and chinese_chars >= english_words:
        return "zh"
    if english_words >= 2 and english_words > chinese_chars:
        return "en"
    return fallback


def _requested_wrapper(text: str) -> str:
    value = str(text or "")
    if not re.search(_BOXED_VALUE, value, re.IGNORECASE | re.DOTALL):
        return "none"
    if re.search(r"(?:answer|final|write|put|place|enter|express|show|display|答案|作答|填写|写入|放入)", value, re.IGNORECASE):
        return "boxed"
    return "none"


def _explicit_unit(text: str) -> str:
    value = str(text or "")
    chinese = re.search(
        r"(?:单位(?:为|是)|以)\s*(度|弧度|厘米|毫米|米|千米|平方厘米|平方米|立方厘米|秒|分钟|小时|元|岁|个|种|条|人)(?:为单位)?",
        value,
    )
    if chinese:
        return chinese.group(1)
    english = re.search(
        r"(?:measured|expressed|reported)?\s*(?:in|to\s+the\s+nearest)\s+"
        r"(degrees?|radians?|percent|centimeters?|millimeters?|kilometers?|meters?|inches?|feet|"
        r"square\s+(?:units?|centimeters?|meters?)|cubic\s+(?:units?|centimeters?|meters?))\b",
        value,
        re.IGNORECASE,
    )
    if english:
        return english.group(1).lower()
    if re.search(r"(?:百分比|percentage|percent)\b|%", value, re.IGNORECASE):
        return "%"
    return ""


_GOAL_COMMAND = (
    r"求|计算|判断|说明|证明|给出|验证|比较|写出|指出|列出|构造|"
    r"\b(?:prove|show|find|determine|solve|calculate|compute|evaluate|verify|compare|"
    r"explain|justify|derive|construct|classify|state|identify|describe|give|write|list)\b|"
    r"what\s+(?:is|are)|is\s+it\s+possible|if\s+(?:it\s+)?is\s+possible"
)

_SPLIT_GOAL_COMMAND = (
    r"求|计算|判断|给出|写出|指出|列出|构造|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|construct|classify|state|identify|"
    r"describe|give|write|list)\b|what\s+(?:is|are)|is\s+it\s+possible|"
    r"if\s+(?:it\s+)?is\s+possible"
)

_ELABORATION_CLAUSE = re.compile(
    r"^\s*(?:给出|写出|指出|列出|(?:state|identify|write|give)\b).*"
    r"(?:所用|理由|依据|条件|结论|含义|计算式|截断|具体公式|转移矩阵|"
    r"derivation|justification|reason|condition|criterion|conclusion|supporting formula|"
    r"family|construction|example|blocks?|cov\s*\()",
    re.IGNORECASE | re.DOTALL,
)


def _goals(text: str, profile: ProblemProfile) -> list[Goal]:
    selected = _split_goal_text(text)
    conditional_followup = len(selected) > 1 and any(re.search(
        r"if\s+(?:it\s+)?is\s+possible.*(?:minimum|maximum|how\s+many|what\s+is)",
        item,
        re.IGNORECASE | re.DOTALL,
    ) for item in selected)
    goals = []
    for index, item in enumerate(selected[:6]):
        # Keep every option in the answer contract. A verb inside one option
        # (for example "求最小值") is not the question's target.
        target = item if profile.answer_shape == "choice" else extract_target_clause(item)
        part_profile = classify_profile(target)
        part_shape = "choice" if profile.answer_shape == "choice" else part_profile.answer_shape
        if re.search(r"分裂域|splitting\s+field", target, re.IGNORECASE):
            part_shape = "expression"
        elif re.search(r"填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|yes\s+or\s+no", target, re.IGNORECASE):
            part_shape = "truth"
        requirements = list(_requirements(target, part_shape))
        if conditional_followup and part_shape == "truth":
            requirements = [item for item in requirements if item.name != "judgement"]
            requirements.append(Requirement(
                "feasibility_or_numeric",
                (("yes",), ("possible",), ("是",), ("可以",)),
                strict=True,
            ))
        elif conditional_followup and re.search(r"minimum|maximum|how\s+many|what\s+is", target, re.IGNORECASE):
            requirements.append(Requirement("numeric_result", (("number",),), strict=True))
        if len(selected) > 1:
            target_symbol = _independent_goal_target(target)
            if target_symbol and not any(requirement.name == f"target_{target_symbol.lower()}" for requirement in requirements):
                requirements.append(Requirement(
                    f"target_{target_symbol.lower()}",
                    ((f"{target_symbol}=",), (f"{target_symbol} =",), (f"{target_symbol}为",), (f"{target_symbol} is",)),
                    strict=True,
                ))
        instruction = target[:1600]
        if any(requirement.name == "all_correct_choices" for requirement in requirements):
            warning = (
                "Answer requirement: return every correct option label; do not assume "
                "single-choice unless the problem explicitly says so."
                if _answer_language(target, profile.language) == "en" else
                "作答要求：返回全部正确选项标签；除非题目明确说明为单选，不得按单选只保留一个标签。"
            )
            instruction = f"{instruction}\n{warning}"
        goals.append(Goal(
            f"g{index + 1}", instruction, part_shape,
            _goal_kind(target, part_shape),
            _required_terms(target, part_shape), tuple(requirements),
        ))
    return goals


def _independent_goal_target(text: str) -> str:
    """Return a short explicitly requested symbol for a split subproblem."""
    value = str(text or "")
    matches = list(re.finditer(
        r"(?:求|计算|写出|给出|find|determine|solve\s+for|calculate|compute)\s*"
        r"(?:the\s+(?:value|values)\s+of\s+)?"
        r"([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)\b",
        value,
        re.IGNORECASE,
    ))
    return matches[-1].group(1) if matches else ""


def _split_goal_text(text: str) -> list[str]:
    """Split only explicit independent asks; keep conditions attached otherwise."""
    value = str(text or "").strip()
    if not value:
        return [value]

    # A sequence of fill-in lines is an explicit multipart contract even when
    # it contains no imperative verb (for example field, degree, verdict).
    blank_parts: list[str] = []
    blank_pattern = re.compile(r"\(\s*\\quad\s*\)|_{2,}|\(\s*\)|（\s*）")
    for line in value.splitlines():
        stripped = line.strip()
        matches = list(blank_pattern.finditer(stripped)) if stripped else []
        if len(matches) == 1:
            blank_parts.append(stripped)
        elif len(matches) > 1:
            marker = "第{index}空" if re.search(r"[\u4e00-\u9fff]", stripped) else "blank {index}"
            blank_parts.extend(f"{stripped} [{marker.format(index=index + 1)}]" for index in range(len(matches)))
    if len(blank_parts) >= 2:
        return blank_parts[:6]

    numbered = list(re.finditer(
        rf"(?im)(?:^|[；;：:。.!?\n])\s*"
        rf"(?:\([a-f1-6ivx]+\)|[（(][一二三四五六\d]+[）)]|[a-f1-6][.)])\s*"
        rf"(?=(?:{_GOAL_COMMAND}))",
        value,
    ))
    if len(numbered) >= 2:
        shared_stem = value[:numbered[0].start()].strip(" \t\r\n；;：:。.!?")
        parts = []
        for index, match in enumerate(numbered):
            end = numbered[index + 1].start() if index + 1 < len(numbered) else len(value)
            part = value[match.end():end].strip(" \t\r\n；;：:。.!?")
            if part and _contains_goal_command(part):
                parts.append(f"{shared_stem}; {part}" if shared_stem else part)
        if len(parts) >= 2:
            return parts

    pieces = [piece.strip(" \t\r\n，,") for piece in re.split(
        rf"[；;]\s*|(?<=[。.!?])\s+(?=(?:{_SPLIT_GOAL_COMMAND}))|"
        rf"(?:并且|并|且|以及|同时)\s*(?=(?:{_SPLIT_GOAL_COMMAND}))|"
        rf"\b(?:and\s+then|and)\s+(?=(?:{_SPLIT_GOAL_COMMAND})\b)",
        value,
        flags=re.IGNORECASE,
    ) if piece.strip(" \t\r\n，,")]
    merged_pieces: list[str] = []
    for piece in pieces:
        if merged_pieces and _ELABORATION_CLAUSE.search(piece):
            merged_pieces[-1] = f"{merged_pieces[-1]}; {piece}"
        else:
            merged_pieces.append(piece)
    pieces = merged_pieces
    command_indices = [index for index, piece in enumerate(pieces) if _contains_goal_command(piece)]
    commands = [pieces[index] for index in command_indices]
    if len(commands) >= 2:
        first_index = command_indices[0]
        first_match = re.search(rf"(?:{_GOAL_COMMAND})|是否|whether", pieces[first_index], re.IGNORECASE)
        stem_parts = pieces[:first_index]
        if first_match and first_match.start() > 0:
            stem_parts.append(pieces[first_index][:first_match.start()].strip(" \t\r\n；;：:。.!?，,"))
        shared_stem = "; ".join(part for part in stem_parts if part)
        if shared_stem:
            commands = [
                command if index == 0 and first_index == 0 else f"{shared_stem}; {command}"
                for index, command in enumerate(commands)
            ]
    return commands if len(commands) >= 2 else [value]


def _contains_goal_command(text: str) -> bool:
    value = re.sub(
        r"\b(?:can|could|may|might|must|should|will|would)\s+(?:then\s+)?"
        r"(?:give|write|list|show|state|identify|describe)\b",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:wants?|wanted|tries|tried|aims?|attempts?)\s+to\s+"
        r"(?:find|determine|solve|calculate|compute|evaluate|describe|give|write|list)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # These are continuation steps in a construction or game rule, not asks.
    value = re.sub(
        r"(?im)^\s*(?:and\s+then\s+)?(?:write\s+this\s+(?:tuple|number|value)|"
        r"give\s+as\s+many\b)[^。.!?]*[。.!?]?",
        "",
        value,
    )
    command_at_clause_start = re.compile(
        rf"(?im)(?:^|[。.!?，,；;：:]\s*|\n)\s*"
        rf"(?:\([a-f1-6ivx]+\)|[a-f1-6][.)])?\s*"
        rf"(?:(?:再|并|同时|然后)\s*)?"
        rf"(?:{_GOAL_COMMAND}|是否|能否|可否|whether)",
        re.IGNORECASE,
    )
    return bool(command_at_clause_start.search(value))


def _goal_kind(text: str, answer_shape: str) -> str:
    if answer_shape == "choice":
        return "choice_selection"
    if _is_result_or_nonexistence_alternative(text):
        return "alternative_result"
    if re.search(r"证明|说明.*(?:理由|为何|原因)|prove|show|explain|justify", text, re.IGNORECASE):
        return "proof"
    if re.search(r"构造|举例|反例|construct|example|counterexample", text, re.IGNORECASE):
        return "construction"
    if answer_shape == "truth" or re.search(r"是否|能否|可否|真假|正确与否|填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|判断.*是否|验证.*是否|whether|yes\s+or\s+no", text, re.IGNORECASE):
        return "truth_judgement"
    if re.search(r"定义域|存在区间|domain|interval", text, re.IGNORECASE):
        return "domain_or_interval"
    if re.search(
        r"中心差分|central\s+difference|"
        r"(?:近似|数值).*(?:导数|积分)|"
        r"(?:approximate|numerical).*(?:derivative|integral)",
        text,
        re.IGNORECASE,
    ):
        return "scalar_or_result"
    if re.search(r"公式|通解|表达式|矩阵|集合|formula|expression|matrix", text, re.IGNORECASE):
        return "formula"
    if re.search(r"近似|误差|比较|approx|error|compare", text, re.IGNORECASE):
        return "comparison"
    if answer_shape == "roots":
        return "equation_roots"
    return "scalar_or_result"


def _required_terms(text: str, answer_shape: str) -> tuple[str, ...]:
    required: list[str] = []
    lowered = text.lower()
    if re.search(r"牛顿法|newton", text, re.IGNORECASE):
        required.append("x_{n+1}")
        if re.search(r"x_0|初值", text):
            required.append("x_1")
    if re.search(r"导数判据|contraction", text, re.IGNORECASE):
        required.extend(["导数", "收敛"])
    if re.search(r"逐点.*极限|pointwise", text, re.IGNORECASE):
        required.append("逐点")
    if re.search(r"积分.*(?:极限|恒|比较)|integral.*limit", text, re.IGNORECASE):
        required.append("积分")
    if re.search(r"(?<!不)交集|intersection", text, re.IGNORECASE):
        required.append("交集")
    if re.search(r"精确值|exact value", text, re.IGNORECASE):
        required.append("精确")
    if re.search(r"特征值", text) and re.search(r"det|行列式", lowered) and re.search(r"tr|迹", lowered):
        required.extend(["det", "tr"])
    if re.search(r"两个任意函数|two arbitrary functions", text, re.IGNORECASE):
        required.extend(["F(", "G("])
    if re.search(r"优先选择|prefer", text, re.IGNORECASE) and re.search(r"T_1", text):
        required.append("T_1")
    if re.search(r"E,F,G|E，F，G", text):
        required.extend(["E=", "F=", "G="])
    # These are coverage hints, not literal admission gates. Variables and
    # theorem nouns from the question are intentionally not copied here:
    # equivalent answers often omit them while remaining fully gradable.
    return tuple(dict.fromkeys(required))


def _requirements(text: str, answer_shape: str) -> tuple[Requirement, ...]:
    requirements: list[Requirement] = []
    alternative_result = _is_result_or_nonexistence_alternative(text)
    if alternative_result:
        requirements.append(Requirement(
            "alternative_result",
            (("number",), ("does not exist",), ("不存在",)),
            strict=True,
        ))
    if answer_shape == "truth" and re.search(
        r"是否|能否|可否|真假|正确与否|填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|"
        r"判断.*是否|验证.*是否|\bis\s+it\b|\bwhether\b|yes\s+or\s+no",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement("judgement", (("是",), ("否",), ("yes",), ("no",), ("true",), ("false",)), strict=True))
    if re.search(
        r"(?:求|计算)[^。.!?\n]{0,80}(?:范数|\|\|[^\n]{0,30}\|\||\bnorm\b)",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "operator_norm",
            (("||t||", "="), ("范数", "="), ("norm", "=")),
            strict=True,
        ))
        if re.search(r"(?:说明|判断)[^。.!?\n]{0,40}是否|\bwhether\b", text, re.IGNORECASE):
            requirements.append(Requirement(
                "judgement",
                (("是",), ("否",), ("等距",), ("isometry",), ("isometric",)),
                strict=True,
            ))
    if answer_shape == "number" and re.search(r"\bfor\s+which\s+values?\b", text, re.IGNORECASE):
        requirements.append(Requirement("numeric_result", (("number",),), strict=True))
    if answer_shape == "choice" and _requires_all_correct_choice_labels(text):
        requirements.append(Requirement(
            "all_correct_choices",
            (("all", "correct", "labels"), ("全部", "正确", "标签")),
            strict=True,
        ))
    asks_for_degree = bool(re.search(
        r"扩张次数|extension\s+degree|\[[^\]]+:[^\]]+\]",
        text,
        re.IGNORECASE,
    ))
    if re.search(r"分裂域|splitting\s+field", text, re.IGNORECASE) and not asks_for_degree:
        requirements.append(Requirement("field_value", (("mathbb", "q"), ("field",)), strict=True))
    if re.search(r"扩张次数|\[[^\]]+:[^\]]+\]\s*=\s*(?:\(\s*\\quad\s*\)|\(\s*\)|_{2,})|extension\s+degree", text, re.IGNORECASE):
        requirements.append(Requirement("degree_value", (("degree",), ("次数",)), strict=True))
    if re.search(r"galois", text, re.IGNORECASE) and re.search(r"填|是|否|yes|no|\(\s*\)", text, re.IGNORECASE):
        requirements.append(Requirement("galois_verdict", (("galois", "是"), ("galois", "否"), ("galois", "yes"), ("galois", "no")), strict=True))
    if len(re.findall(r"\\quad|_{2,}|\(\s*\)|（\s*）", text)) >= 2:
        requirements.append(Requirement("two_items", (("、",), (",",), ("and",)), strict=True))
    if re.search(r"decomposition\s+into\s+phrases|短语分解", text, re.IGNORECASE):
        requirements.append(Requirement("phrase_decomposition", (("phrases",), ("短语",)), strict=True))
    if re.search(r"encoded\s+string|编码(?:串|结果|得到)", text, re.IGNORECASE):
        requirements.append(Requirement("encoded_string", (("000",), ("编码",)), strict=True))
    if re.search(
        r"(?:求|列出|写出)(?:出)?(?:所有|全部)|"
        r"\b(?:find|determine|list|classify|describe)\s+all\b|"
        r"\b(?:find|determine)\s+all\s+possible\b|"
        r"\b(?:find|determine)\s+the\s+complete\s+set\s+of\b|"
        r"\bfind\s+(?:a\s+)?complete\s+set\s+of\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "exhaustive_result",
            (("all",), ("only",), ("所有",), ("仅",)),
            strict=True,
        ))
    if _asks_for_euler_formula_check(text):
        requirements.append(Requirement(
            "euler_formula_check",
            (("v-e+f", "=2"), ("numeric", "substitution")),
            strict=True,
        ))
    if re.search(r"主曲率|\bprincipal\s+curvatures?\b", text, re.IGNORECASE):
        requirements.append(Requirement(
            "principal_curvatures",
            (("kappa_1", "kappa_2"), ("主曲率",), ("principal curvatures",)),
            strict=True,
        ))
    if re.search(r"高斯曲率|\bgaussian\s+curvature\b", text, re.IGNORECASE):
        requirements.append(Requirement(
            "gaussian_curvature",
            (("k", "="), ("高斯曲率",), ("gaussian curvature",)),
            strict=True,
        ))
    if re.search(
        r"(?:写出|给出|列出|计算|求|说明)[^。.!?\n]{0,30}二阶导数|"
        r"二阶导数[^。.!?\n]{0,12}(?:依据|值|矩阵)|"
        r"\b(?:give|write|state|list|compute|show)\b[^.!?\n]{0,60}"
        r"\bsecond[- ](?:order\s+)?(?:partial\s+)?derivatives?\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "surface_second_derivatives",
            (("f_xx", "f_xy", "f_yy"), ("hessian",)),
            strict=True,
        ))
    if not alternative_result and re.search(
        r"说明.*(?:理由|为何|原因)|证明|prove|show|explain|justify|show\s+your\s+work",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement("reasoning", (("因为",), ("依据",), ("because",), ("since",), ("therefore",)), strict=False))
    if re.search(r"牛顿法|newton", text, re.IGNORECASE):
        requirements.append(Requirement("iteration_formula", (("xn+1",), ("xk+1",), ("迭代公式",))))
        if re.search(r"x_0|初值", text):
            requirements.append(Requirement("first_iteration", (("x1",), ("第一次迭代",))))
    if re.search(r"逐点.*(?:极限|收敛)|pointwise", text, re.IGNORECASE):
        requirements.append(Requirement("pointwise_limit", (("逐点",), ("pointwise",)), strict=True))
    if re.search(r"积分.*(?:极限|恒|比较)|integral.*limit", text, re.IGNORECASE):
        requirements.append(Requirement(
            "integral_result",
            (("积分",), ("integral",), ("∫",), ("近似值", "精确值")),
        ))
    if re.search(r"(?:计算|求).*积分|integral", text, re.IGNORECASE):
        requirements.append(Requirement(
            "integral_value",
            (("积分", "为"), ("积分", "="), ("∫", "="), ("integral", "=")),
            strict=True,
        ))
    if re.search(
        r"(?:说明|解释)[^。.!?\n]{0,50}极点[^。.!?\n]{0,30}是否[^。.!?\n]{0,30}围道|"
        r"(?:explain|state)[^.!?\n]{0,60}(?:pole|singularity)[^.!?\n]{0,40}"
        r"(?:inside|outside)[^.!?\n]{0,30}(?:contour|curve)",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "pole_location",
            (("极点", "内"), ("极点", "外"), ("pole", "inside"), ("pole", "outside")),
            strict=True,
        ))
    if re.search(r"(?:是否|判断).*(?:并|且).*计算.*积分|(?:whether|determine).*(?:and).*integral", text, re.IGNORECASE):
        requirements.append(Requirement(
            "integral_value",
            (("积分", "为"), ("积分", "="), ("∫", "="), ("integral", "=")),
            strict=True,
        ))
    if re.search(r"(?<!不)交集|intersection", text, re.IGNORECASE):
        requirements.append(Requirement("intersection", (("交集",), ("intersection",), ("∩",))))
    if re.search(r"精确值|exact value", text, re.IGNORECASE):
        requirements.append(Requirement("exact_comparison", (("精确",), ("exact",), ("真实值",))))
    if re.search(r"特征值", text) and re.search(r"det|行列式", text, re.IGNORECASE) and re.search(r"tr|迹", text, re.IGNORECASE):
        requirements.extend((
            Requirement("determinant", (("det",), ("行列式",))),
            Requirement("trace", (("tr",), ("迹",))),
        ))
    if re.search(r"两个任意函数|two arbitrary functions", text, re.IGNORECASE):
        requirements.append(Requirement("two_function_form", (("f(", "g("), ("任意函数",))))
    if re.search(r"(?:求|列出).*(?:所有|全部).*生成元|all generators", text, re.IGNORECASE):
        requirements.append(Requirement(
            "generator_enumeration",
            (("生成元为",), ("生成元包括",), ("{",), ("generator", "=")),
        ))
    if re.search(r"递推", text) and re.search(r"通项", text):
        requirements.append(Requirement("recurrence_formula", (("a_n", "="), ("通项",)), strict=True))
    if re.search(r"不动点平移", text):
        requirements.append(Requirement(
            "fixed_point_check",
            (("不动点",), ("a_n-1",), ("平移",)),
            strict=True,
        ))
    if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", text, re.IGNORECASE):
        requirements.append(Requirement("domain", (("区间",), ("domain",)), strict=True))
        if re.search(r"求解|解方程|分离变量", text):
            requirements.append(Requirement("solution_formula", (("y=",),), strict=True))
    if re.search(r"严格凸性等式|平行四边形恒等式", text):
        requirements.append(Requirement(
            "parallelogram_identity",
            (("u+v", "u-v", "="),),
            strict=True,
        ))
    if re.search(r"时间和空间导数|时间.*空间.*导数", text):
        requirements.append(Requirement(
            "pde_time_space_derivatives",
            (("u_t", "u_{xx}"),),
            strict=True,
        ))
    if re.search(r"二阶求导", text) and re.search(r"u\s*=", text):
        requirements.append(Requirement(
            "laplace_second_derivatives",
            (("u_{xx}", "u_{yy}"),),
            strict=True,
        ))
    requirements.extend(_explicit_method_requirements(text))
    return tuple(requirements)


def _asks_for_euler_formula_check(text: str) -> bool:
    return bool(re.search(
        r"(?:验证|检验|核验)[^。.!?\n]{0,40}欧拉公式|"
        r"\b(?:verify|check)\b[^.!?\n]{0,60}\beuler(?:'s)?\s+formula\b",
        str(text or ""),
        re.IGNORECASE,
    ))


def _requires_all_correct_choice_labels(text: str) -> bool:
    """Whether a choice prompt may have several correct options.

    In the absence of an explicit single-choice instruction, wording that asks
    which statements are correct is treated as select-all. This prevents a
    verifier from silently collapsing a complete label set to one label.
    """
    value = str(text or "")
    explicit_single = re.search(
        r"单选|单项选择|仅有一(?:项|个)|只有一(?:项|个)|唯一(?:正确|错误)|"
        r"选(?:择|出)一(?:项|个)|"
        r"\b(?:single[- ]choice|select\s+(?:exactly\s+)?one|choose\s+(?:exactly\s+)?one|"
        r"only\s+one|one\s+(?:correct|best)\s+answer|best\s+answer|most\s+appropriate)\b",
        value,
        re.IGNORECASE,
    )
    if explicit_single:
        return False
    return bool(re.search(
        r"(?:下列|以下)[^。！？!?\n]{0,160}(?:正确|错误|不正确)|"
        r"(?:哪些|哪几项|哪几种)[^。！？!?\n]{0,100}(?:正确|错误|不正确|成立)|"
        r"(?:正确|错误|不正确)(?:的)?(?:选项|说法|结论|命题)?(?:是|有|包括)|"
        r"\b(?:select|choose|check)\s+all(?:\s+that\s+apply)?\b|"
        r"\ball\s+that\s+apply\b|"
        r"\bwhich\b[^?.!\n]{0,120}\b(?:is|are)\s+(?:correct|true|false|incorrect)\b",
        value,
        re.IGNORECASE,
    ))


def _is_result_or_nonexistence_alternative(text: str) -> bool:
    return bool(re.search(
        r"\b(?:determine|find|compute)\b.+\bor\s+prove\b.+"
        r"(?:does\s+not|need\s+not|no\s+such|not\s+necessarily\s+exist)",
        str(text or ""),
        re.IGNORECASE | re.DOTALL,
    ))


def _explicit_method_requirements(text: str) -> list[Requirement]:
    """Capture methods and intermediate values explicitly demanded by the prompt."""
    rules: tuple[tuple[str, str, tuple[tuple[str, ...], ...]], ...] = (
        (r"先选取.*位置", "position_selection", (("位置",), ("组合",), ("binom",), ("c_",))),
        (r"变量平移", "variable_shift", (("变量平移",), ("平移",), ("y_1",), ("令y",))),
        (r"隔板", "stars_and_bars", (("隔板",), ("非负整数解",), ("binom",))),
        (r"容斥或条件计数", "counting_method", (("容斥",), ("条件计数",))),
        (r"容斥(?:原理|公式)?", "inclusion_exclusion", (("容斥",),)),
        (r"先将.*配对|先.*配对", "pairing_step", (("配对",), ("!!",), ("成对",))),
        (r"按\s*[A-Za-z\u4e00-\u9fff]+\s*分类", "case_split", (("分类",), ("分情况",), ("a=",))),
        (r"位置压缩", "position_compression", (("位置压缩",), ("压缩",), ("j-1",))),
        (r"使用分配律", "distributive_step", (("分配律",),)),
        (r"列出两步递推", "two_steps", (("y_1", "y_2"), ("两步",))),
        (r"一阶和二阶导数", "first_second_derivatives", (("一阶", "二阶"), ("γ'", "γ''"))),
        (r"按不交集合分解", "disjoint_decomposition", (("不交",), ("分解",))),
        (r"使用组合数计算", "combination_calculation", (("组合数",), ("binom",), ("c_",))),
        (r"先作部分分式", "partial_fraction", (("部分分式",),)),
        (r"识别极点阶数", "pole_order", (("极点", "阶"), ("一阶极点",))),
        (r"使用柯西积分公式", "cauchy_formula", (("柯西积分公式",),)),
        (r"使用积分因子", "integrating_factor", (("积分因子",),)),
        (r"先写特征方程", "characteristic_equation", (("特征方程",),)),
        (r"通过分离变量", "separation_of_variables", (("分离变量",),)),
        (r"利用独立增量", "independent_increments", (("独立增量",),)),
        (r"说明推理规则", "inference_rule", (("假言推理",), ("假言三段论",), ("modus ponens",))),
        (r"使用极值或连续性反证", "continuity_contradiction", (("反证",), ("连续性",))),
        (r"开覆盖限制到闭子集", "open_cover_step", (("开覆盖", "补集"), ("有限子覆盖",))),
        (r"(?:using|by)\s+(?:mathematical\s+)?induction", "induction", (("induction",), ("base case", "inductive"))),
        (r"(?:using|by)\s+(?:the\s+)?pigeonhole\s+principle", "pigeonhole_principle", (("pigeonhole",),)),
        (r"(?:using|by)\s+(?:the\s+)?(?:am[- ]gm|arithmetic[- ]geometric\s+mean)", "am_gm", (("am-gm",), ("arithmetic", "geometric"))),
        (r"(?:using|by)\s+(?:the\s+)?cauchy(?:[- ]schwarz)?", "cauchy_schwarz", (("cauchy",),)),
        (r"(?:using|by)\s+(?:the\s+)?inclusion[- ]exclusion", "inclusion_exclusion", (("inclusion", "exclusion"),)),
        (r"(?:using|by)\s+contradiction", "contradiction", (("contradiction",),)),
    )
    found: list[Requirement] = []
    seen: set[str] = set()
    for pattern, name, alternatives in rules:
        if name == "inclusion_exclusion" and re.search(r"容斥或条件计数", text):
            continue
        if re.search(pattern, text, re.IGNORECASE) and name not in seen:
            found.append(Requirement(name, alternatives, strict=True))
            seen.add(name)
    return found


def _risk_score(text: str, profile: ProblemProfile, goals: list[Goal], risks: list[str]) -> int:
    score = 0
    if len(goals) > 1:
        score += 2
    if profile.problem_type in {"proof", "derivation", "explanation"}:
        score += 2
    if profile.difficulty == "hard":
        score += 1
    if profile.confidence == "low":
        score += 1
    if any(goal.kind == "construction" for goal in goals):
        score += 2
    if re.search(r"牛顿法|二分法|欧拉法|迭代|近似|误差|newton|bisection|euler|iteration|approx", text, re.IGNORECASE):
        score += 2
    if re.search(r"递推|通项|recurrence", text, re.IGNORECASE):
        score += 3
    if any(requirement.strict for goal in goals for requirement in goal.requirements):
        score += 1
    if getattr(profile, "topic", "").startswith("olympiad_"):
        score += 2
    if set(risks) & {"exhaustiveness_required", "integer_constraints", "functional_equation", "diagram_dependency"}:
        score += 2
    if re.search(r"最大右侧存在区间|maximal right(?:-hand)? interval", text, re.IGNORECASE):
        score += 2
    if set(risks) & {"missing_roots", "domain_or_substitution", "endpoint_error", "quantifier_or_missing_argument"}:
        score += 1
    return min(score, 8)


def _tool_can_answer_whole(text: str, profile: ProblemProfile, goals: list[Goal]) -> bool:
    """Allow model bypass only for a small, auditable symbolic whitelist.

    Tool hints are useful for many other problems, but those hints are only local
    evidence.  In particular, solving an equation embedded in a numerical method
    or an olympiad problem does not satisfy the public answer contract.
    """
    if not profile.tool_eligible or len(goals) != 1:
        return False
    if getattr(profile, "topic", "general").startswith("olympiad_"):
        return False
    if len(text) > 240 or re.search(_TOOL_WHOLE_EXCLUSIONS, text, re.IGNORECASE):
        return False
    return (
        _is_direct_arithmetic(text, profile)
        or _is_unconstrained_single_variable_equation(text, profile)
        or _is_single_direct_calculus(text, profile)
    )


_TOOL_WHOLE_EXCLUSIONS = (
    r"牛顿法|二分法|欧拉法|迭代|插值|近似|误差|收敛|条件数|数值解|"
    r"证明|求证|论证|说明理由|解释|推导|验证|比较|估计|保留\s*\d+\s*位|"
    r"小数|有效数字|科学计数法|四舍五入|"
    r"\b(?:newton|bisection|euler|iteration|iterate|interpolation|approx(?:imate|imation)?|"
    r"numerical(?:ly)?|error bound|convergen\w*|condition number|prove|proof|justify|"
    r"explain|derive|verify|compare|estimate|round(?:ed)?|decimal places?|"
    r"significant figures?|scientific notation)\b"
)

_CALCULUS_FAMILIES = (
    re.compile(r"求导|导数|微分|偏导|\b(?:derivative|differentiate|differentiation)\b", re.IGNORECASE),
    re.compile(r"积分|\\int(?![A-Za-z])|\b(?:integral|integrate|integration)\b", re.IGNORECASE),
    re.compile(r"极限|\\lim(?![A-Za-z])|\blimit\b", re.IGNORECASE),
)


def _is_direct_arithmetic(text: str, profile: ProblemProfile) -> bool:
    if profile.answer_shape != "number" or "=" in text:
        return False
    match = re.fullmatch(
        r"\s*(?:计算|求值|calculate|compute|evaluate)\s*"
        r"(?:(?:下列|the\s+value\s+of)\s*)?[:：]?\s*(.+?)\s*[。.!?？]?\s*",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    payload = match.group(1).strip().strip("$")
    identifiers = {
        item.lower() for item in re.findall(r"[A-Za-z]+", payload)
    }
    if identifiers - {"sin", "cos", "tan", "asin", "acos", "atan", "log", "exp", "sqrt", "pi"}:
        return False
    return bool(re.search(r"\d", payload)) and bool(re.fullmatch(
        r"[0-9A-Za-z_+\-*/^().,\s]+", payload,
    ))


def _is_unconstrained_single_variable_equation(text: str, profile: ProblemProfile) -> bool:
    # Short ``Solve x=...`` prompts are sometimes classified as scalar results
    # when they omit the noun "equation".  The structural checks below are the
    # authority for this narrowly defined route.
    if profile.answer_shape not in {"number", "roots"}:
        return False
    if not re.search(r"方程|求解|(?:^|\b)解\b|\b(?:equation|solve|roots?|zeros?|find\s+[xyz])\b", text, re.IGNORECASE):
        return False
    if len(re.findall(r"(?<![<>!])=(?!=)", text)) != 1:
        return False
    if re.search(r"<=|>=|!=|≠|≤|≥|<|>", text):
        return False
    if re.search(_EQUATION_CONSTRAINTS, text, re.IGNORECASE):
        return False
    variables = _symbolic_variables(text)
    return len(variables) == 1 and variables <= {"x", "y", "z"}


_EQUATION_CONSTRAINTS = (
    r"实数|复数|整数|正数|负数|非负|非正|非零|自然数|有理数|无理数|"
    r"互不相同|定义域|区间|范围|满足条件|在.+?上|"
    r"\b(?:real|complex|integer|positive|negative|nonnegative|nonpositive|nonzero|"
    r"natural|rational|irrational|distinct|domain|interval|range|subject\s+to|under\s+the\s+condition)\b"
)


def _is_single_direct_calculus(text: str, profile: ProblemProfile) -> bool:
    # A definite integral or a limit is often classified as a scalar, while a
    # derivative is classified as an expression; both are valid direct routes.
    if profile.answer_shape not in {"number", "expression"}:
        return False
    families = [pattern for pattern in _CALCULUS_FAMILIES if pattern.search(text)]
    if len(families) != 1:
        return False
    if _CALCULUS_FAMILIES[1] in families:
        # Only bounded integrals can be complete deterministic answers.  A
        # bare antiderivative still needs +C, real-domain/absolute-value
        # handling, and a readability check beyond SymPy's expression object.
        explicitly_indefinite = bool(re.search(
            r"不定积分|原函数|\b(?:indefinite\s+integral|antiderivative)\b",
            text,
            re.IGNORECASE,
        ))
        has_latex_bounds = bool(re.search(r"\\int\s*_", text))
        explicitly_definite = bool(re.search(
            r"定积分|\bdefinite\s+integral\b",
            text,
            re.IGNORECASE,
        ))
        if explicitly_indefinite or not (has_latex_bounds or explicitly_definite):
            return False
    if re.search(
        r"二阶|三阶|高阶|混合偏导|梯度|海森|雅可比|"
        r"\b(?:second|third|higher|n(?:th|-th)\s+derivative|mixed partial|gradient|hessian|jacobian)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"[;；]|(?:并且|并求|并计算|以及|同时)|\b(?:and\s+then|then\s+also)\b", text, re.IGNORECASE):
        return False
    # Two displayed operators represent two requested computations even if the
    # natural-language classifier failed to split them into separate goals.
    if len(re.findall(r"\\int(?![A-Za-z])", text)) > 1 or len(re.findall(r"\\lim(?![A-Za-z])", text)) > 1:
        return False
    variables = _symbolic_variables(text)
    return len(variables) == 1


def _symbolic_variables(text: str) -> set[str]:
    """Conservatively recover variables from short symbolic requests."""
    fragments = re.findall(r"\$([^$]+)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", text, re.DOTALL)
    symbolic = [next((item for item in group if item), "") for group in fragments]
    if not symbolic and "=" in text:
        symbolic = [text]
    if not symbolic:
        calculus = re.search(
            r"(?:求导|导数|微分|偏导|积分|极限|derivative|differentiate|integral|integrate|limit)\s*(?:of\s+)?(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if calculus:
            symbolic = [calculus.group(1)]
    value = " ".join(symbolic)
    value = re.sub(r"[fgh]\s*(?=\()", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\\(?:frac|dfrac|tfrac|sqrt|sin|cos|tan|log|ln|exp|left|right|int|lim|to|infty|pi)(?![A-Za-z])", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:sin|cos|tan|asin|acos|atan|log|ln|exp|sqrt|pi|with|respect|from|to|at|of)\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<![A-Za-z])d(?=[A-Za-z])", "", value)
    variables: set[str] = set()
    for identifier in re.findall(r"[A-Za-z]+", value.lower()):
        if len(identifier) == 1 and identifier not in {"d", "e", "i"}:
            variables.add(identifier)
        elif len(identifier) > 1 and set(identifier) <= set("abcstuvwxyz"):
            variables.update(identifier)
    return variables


def _constraints(text: str) -> list[str]:
    markers = (
        r"实数|整数|正整数|非负|正数|非零|互不相同|可逆|连续|紧致|独立|互斥|初值|"
        r"all|every|unique|positive\s+integers?|nonnegative|nonzero|distinct|integer|real|"
        r"continuous|compact|independent|acute|convex"
    )
    return [match.group(0) for match in re.finditer(markers, text, re.IGNORECASE)][:8]


def _risks(text: str, profile: ProblemProfile, goal_count: int) -> list[str]:
    lowered = text.lower()
    risks = []
    if goal_count > 1:
        risks.append("multiple_goals")
    if profile.answer_shape == "roots":
        risks.extend(["missing_roots", "domain_or_substitution"])
    if profile.answer_shape == "interval":
        risks.append("endpoint_error")
    if profile.problem_type in {"proof", "derivation", "explanation"}:
        risks.extend(["theorem_scope", "quantifier_or_missing_argument"])
    if profile.subject == "离散数学" or re.search(r"组合|排列|计数|count|combin", lowered):
        risks.append("double_counting")
    if profile.subject == "概率论":
        risks.append("probability_range")
    if profile.subject == "抽象代数":
        risks.append("definition_or_structure_conditions")
    if re.search(r"构造|construct", lowered):
        risks.append("construction_validation")
    topic = getattr(profile, "topic", "general")
    if topic.startswith("olympiad_"):
        risks.append("olympiad_problem")
    if re.search(r"(?:求|找出|确定|列出).*所有|\b(?:find|determine|classify)\s+all\b", text, re.IGNORECASE):
        risks.append("exhaustiveness_required")
    if topic == "olympiad_number_theory" or re.search(r"正整数|整数解|positive integers?|integer solutions?", text, re.IGNORECASE):
        risks.append("integer_constraints")
    if topic == "olympiad_functional_equation":
        risks.append("functional_equation")
    if topic == "olympiad_geometry":
        risks.append("diagram_dependency")
    if topic in {"olympiad_combinatorics", "olympiad_sequence"}:
        risks.append("case_analysis")
    return list(dict.fromkeys(risks))
