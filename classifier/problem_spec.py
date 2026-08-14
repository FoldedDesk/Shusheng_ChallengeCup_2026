"""Structured, local-only understanding of a mathematics problem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.advanced_families import (
    DIRECTED_EULER_CIRCUIT_PATTERN,
    LACUNARY_NATURAL_BOUNDARY_PATTERN,
    PLANE_ROOTED_TREE_PATTERN,
    RUNGE_KUTTA_STABILITY_PATTERN,
    SPECIALIZED_TOPICS,
    SPHERICAL_TRIANGLE_AREA_PATTERN,
    TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
    WEIERSTRASS_SINE_PRODUCT_PATTERN,
)
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
    "geometric_distribution_identification",
    "independence_use",
    "iid_variance_scaling",
    "strong_law",
    "central_difference_formula",
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
        raw_answer = re.sub(
            r"\\(?:text|mathrm)\s*\{([^{}]*)\}",
            r"\1",
            raw_answer,
        )
        normalized = _compact(raw_answer)
        latex_flat = (
            raw_answer.replace(r"\,", "")
            .replace(r"\;", "")
            .replace(r"\ ", "")
            .replace(r"\quad", "")
        )
        if self.name.startswith("parameter_dependency_"):
            symbol = self.name.removeprefix("parameter_dependency_")
            math_text = re.sub(r"\\[A-Za-z]+", " ", raw_answer)
            return bool(symbol) and bool(re.search(
                rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])",
                math_text,
                re.IGNORECASE,
            ))
        if self.name == "target_e":
            # ``E[...]`` is expectation notation, not a missing scalar label
            # merely because the target extractor also represents it as E.
            return bool(re.search(r"(?<![A-Za-z])e\s*(?:=|\[|\()", raw_answer))
        if self.name == "exhaustive_result":
            explicit_exhaustion = bool(re.search(
                r"(?:所有|全部|仅|只有|唯一|恰为|且无其他|无其他|不存在其他|任意|"
                r"\b(?:all|only|exactly|no\s+others?)\b|"
                r"(?:共|合计|总计)\s*(?:为|是|=|[:：])?\s*(?:\$|\\\(|\\boxed\s*\{)?\s*\d+|"
                r"\b(?:a\s+)?total(?:\s+(?:number|count))?\s*(?:of|is|=|:)?\s*"
                r"(?:\$|\\\(|\\boxed\s*\{)?\s*\d+|"
                r"\b\d+\s+(?:items?|values?|solutions?|tuples?)?\s*in\s+total\b|"
                r"\\?\{[^{}]*\}|\.\.\.|\\ldots|\\dots|"
                r"[^,\n]+,[^,\n]+)",
                raw_answer,
                re.IGNORECASE,
            ))
            if explicit_exhaustion:
                return True
            # A parameter declaration is exhaustive only when it actually
            # describes a family on the right-hand side.  Bare claims such as
            # "x=1 for some real x" certify one witness, not all solutions.
            family = re.search(
                r"(?P<formula>[A-Za-z](?:\s*_\s*\{?[A-Za-z0-9]+\}?)?"
                r"(?:\s*\([^)]*\))?\s*=.+?)\s+"
                r"(?:for\s+(?:some\s+)?|where\s+)"
                r"(?:(?:an?\s+)?(?:non[- ]?negative\s+)?integers?\s+|"
                r"(?:an?\s+)?(?:arbitrary\s+)?real(?:\s+numbers?)?\s+)?"
                r"(?P<parameters>[a-z](?:\s*[,，]\s*[a-z])*)"
                r"(?:\s*(?:\\in|in|are|is)\s*(?:\\mathbb\s*\{?[RZN]\}?|"
                r"REAL|real|integer|non[- ]?negative))?",
                raw_answer,
                re.IGNORECASE,
            )
            if not family:
                return False
            rhs = family.group("formula").split("=", 1)[-1]
            parameters = re.findall(r"[a-z]", family.group("parameters"), re.IGNORECASE)
            return bool(parameters) and all(re.search(
                rf"(?<![A-Za-z]){re.escape(parameter)}(?![A-Za-z])",
                rhs,
                re.IGNORECASE,
            ) for parameter in parameters)
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
        if self.name == "almost_everywhere_zero":
            zero_statement = bool(re.search(
                r"(?<![A-Za-z])f\s*(?:\([^)]*\))?\s*"
                r"(?:=|\\equiv|为|是)\s*0(?!\s*[<>])",
                raw_answer,
                re.IGNORECASE,
            ))
            ae_qualifier = bool(re.search(
                r"几乎(?:处处|到处)|"
                r"(?:\\?mu\s*[-‐‑–—]?\s*)?a\s*\.?\s*e\s*\.?\b|"
                r"almost\s+everywhere",
                raw_answer,
                re.IGNORECASE,
            ))
            return zero_statement and ae_qualifier
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
            explicit_norm = bool(re.search(
                r"(?:\\lVert|\\Vert|\\\||\|\|).{1,120}?"
                r"(?:\\rVert|\\Vert|\\\||\|\|)"
                r"(?:\s*_\s*\{?\s*[A-Za-z0-9]+\s*\}?)?\s*=\s*"
                r"(?:[-+]?\d|\\(?:frac|sqrt)|[A-Za-z])|"
                r"(?:算子)?范数\s*(?:为|是|=)|\boperator\s+norm\s*(?:is|=)",
                raw_answer,
                re.IGNORECASE,
            ))
            # A bare scalar can be the requested norm value.  An equality such
            # as |L(f)|=1 is only a pointwise bound and must not satisfy the
            # operator-norm obligation.
            bare_value = (
                _is_bare_math_expression(raw_answer)
                and not re.search(
                    r"[=|]|\\(?:lvert|rvert|vert|lVert|rVert|Vert)",
                    raw_answer,
                )
            )
            return explicit_norm or bare_value
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
            compact_phrase_and_bits = bool(re.search(
                r"(?:^|[:：])\s*[A-Za-z0-9]+"
                r"(?:\s*[,，]\s*[A-Za-z0-9]+){2,}\s*[;；]\s*"
                r"[01](?:[01\s]{4,})[01]\s*$",
                raw_answer,
                re.IGNORECASE,
            ))
            return (
                indexed_phrases
                or labelled_phrases
                or described_decomposition
                or compact_phrase_and_bits
            )
        if self.name == "conditional_sample_space":
            ordered_outcomes = re.findall(
                r"\(\s*[^,，()]+\s*[,，]\s*[^()]+\)",
                raw_answer,
            )
            named_space = bool(re.search(
                r"条件样本空间|conditional\s+sample\s+space|"
                r"(?:omega|\\omega)\s*_?\s*\{?[^=\s]*\}?\s*=",
                raw_answer,
                re.IGNORECASE,
            ))
            braced_enumeration = bool(re.search(
                r"(?:\\?\{|\\left\\\{)[^{}\n]*"
                r"\([^()]+[,，][^()]+\)[^{}\n]*(?:\\?\}|\\right\\\})",
                raw_answer,
                re.IGNORECASE,
            ))
            # A conditional sample space must be enumerated, not merely
            # described by its cardinality.  A named singleton is allowed,
            # while an unlabelled list needs at least two ordered outcomes.
            return bool(ordered_outcomes) and (
                named_space or braced_enumeration or len(ordered_outcomes) >= 2
            )
        if self.name == "variance_identification":
            centered_moment = bool(re.search(
                r"(?:^|mathbb|operatorname)e(?:x-p)(?:\^|\*\*)?2",
                normalized,
                re.IGNORECASE,
            ))
            variance_named = "varx" in normalized or bool(re.search(
                r"(?:var(?:iance)?\s*(?:\(|\[)?\s*x|方差)",
                raw_answer,
                re.IGNORECASE,
            ))
            explicitly_linked = "=" in raw_answer or bool(re.search(
                r"(?:即为|就是|等于|识别为)[^。.!?\n]{0,30}方差|"
                r"\b(?:is|equals?|identif(?:y|ied)\s+as)\s+(?:the\s+)?variance\b",
                raw_answer,
                re.IGNORECASE,
            ))
            return centered_moment and variance_named and explicitly_linked
        if self.name in {"distribution_result", "variance_result"}:
            normal_parameters = bool(re.search(
                r"(?:\\mathcal\s*\{?n\}?|(?<![A-Za-z])n)\s*"
                r"\(\s*[^,，()]+\s*[,，]\s*[^()]+\)",
                raw_answer,
                re.IGNORECASE,
            ))
            if self.name == "distribution_result":
                return normal_parameters or bool(re.search(
                    r"分布|\\sim|~|\bdistribution\b",
                    raw_answer,
                    re.IGNORECASE,
                ))
            return normal_parameters or bool(re.search(
                r"方差|\\?operatorname\s*\{?var\}?|\bvariance\b|\bvar\s*\(",
                raw_answer,
                re.IGNORECASE,
            ))
        if self.name == "all_correct_choices":
            # The public problem does not reveal how many options are correct;
            # this obligation can validate label syntax, while the solve and
            # verification prompts must determine and return the complete set.
            return bool(answer_choice_labels(compact_answer))
        if self.name == "encoded_string":
            return bool(re.search(
                r"(?:[01]{3,}[\s\\,]*){2,}|(?<![01])[01]{6,}(?![01])",
                raw_answer,
            ))
        if self.name == "pointwise_limit":
            pointwise_named = bool(re.search(
                r"逐点(?:极限|收敛)?|pointwise(?:\s+(?:limit|convergence))?",
                raw_answer,
                re.IGNORECASE,
            ))
            explicit_value = bool(re.search(
                r"(?:逐点(?:极限|收敛)?|pointwise(?:\s+(?:limit|convergence))?)"
                r"[^;；。.!?\n]{0,80}(?:为|是|于|到|=|\bis\b|\bto\b|"
                r"\btowards?\b|\\to|→)\s*"
                r"(?:恒?零函数|零|zero\s+function|[-+]?\d|[A-Za-z]|\\[A-Za-z]+)",
                raw_answer,
                re.IGNORECASE,
            )) or bool(re.search(
                r"[A-Za-z]\s*_?\s*\{?n\}?\s*(?:\([^)]*\))?\s*(?:\\to|→)\s*"
                r"(?:恒?零函数|零|zero\s+function|[-+]?\d|[A-Za-z]|\\[A-Za-z]+)",
                raw_answer,
                re.IGNORECASE,
            ))
            return pointwise_named and explicit_value
        if self.name == "l1_norm_check":
            norm_object = (
                r"(?:l\s*\^?\s*\{?1\}?\s*(?:范数|norm)|范数|"
                r"\\lVert[^\n]{0,80}?\\rVert(?:\s*_?\s*\{?1\}?)?|"
                r"\|[^\n]{1,80}\|(?:\s*_?\s*\{?1\}?)?|"
                r"\\int[^\n]{1,120}|积分)"
            )
            explicit_norm_value = bool(re.search(
                norm_object
                + r"[^;；。.!?\n]{0,100}(?:=|为|是|\\to|→|\\not\s*\\to|不趋于)\s*"
                r"(?:[-+]?\d|\\(?:frac|infty|infinity)|∞|[A-Za-z]\s*\()",
                raw_answer,
                re.IGNORECASE,
            ))
            return explicit_norm_value
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
            return bool(re.search(
                r"(?:一阶|['′]).*(?:二阶|['′]{2})|"
                r"(?:gamma|γ)\s*['′].*(?:gamma|γ)\s*(?:''|′′)|"
                r"\bfirst(?:[- ]order)?\s+derivative\b.*"
                r"\bsecond(?:[- ]order)?\s+derivative\b",
                raw_answer,
                re.DOTALL | re.IGNORECASE,
            ))
        if self.name == "two_steps":
            return all(term in normalized for term in ("y1", "y2")) or "两步" in raw_answer
        if self.name == "domain":
            return bool(re.search(r"(?:定义域|区间|domain|[x-z]\s*[<>≤≥≠]|\([^)]*,[^)]*\)|\[[^]]*,[^]]*\])", raw_answer, re.IGNORECASE))
        if self.name == "integral_value":
            return bool(re.search(
                r"(?:积分|integral|∫).*(?:为|=)|(?:级数|总和|结果|近似值|精确值).*(?:为|=)",
                raw_answer,
                re.IGNORECASE,
            )) or bool(re.search(
                r"(?:积分值|近似值|精确值)\s*(?:为|是|=|[:：])?\s*"
                r"(?:\$?\s*)?(?:[-+]?\d|\\(?:d?frac|sqrt)\s*\{|π|\\pi)",
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
    topic = getattr(profile, "topic", "general")
    if topic == "directed_euler_circuits" or re.search(
        DIRECTED_EULER_CIRCUIT_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "best_theorem_with_fixed_arc_normalization",
            "directed_matrix_tree_and_exit_ordering_check",
        )
    elif topic == "plane_rooted_tree_enumeration" or re.search(
        PLANE_ROOTED_TREE_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "lukasiewicz_words_and_cycle_lemma",
            "rooted_plane_tree_degree_sequence_formula",
        )
    elif topic == "lacunary_natural_boundary" or re.search(
        LACUNARY_NATURAL_BOUNDARY_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "radius_then_dense_boundary_singularities",
            "fabry_or_hadamard_gap_theorem",
        )
    elif topic == "runge_kutta_stability" or re.search(
        RUNGE_KUTTA_STABILITY_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "order_conditions_then_stability_function",
            "imaginary_axis_modulus_and_infinity_limit",
        )
    elif topic == "spherical_triangle_area" or re.search(
        SPHERICAL_TRIANGLE_AREA_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "spherical_cosine_law_then_girard_excess",
            "gram_matrix_or_vector_angle_area_check",
        )
    elif topic == "weierstrass_sine_product" or re.search(
        WEIERSTRASS_SINE_PRODUCT_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "weierstrass_sine_product_then_imaginary_substitution",
            "zero_set_normalization_and_log_derivative_check",
        )
    elif topic == "two_dimensional_polyharmonic_fundamental_solution" or re.search(
        TWO_DIMENSIONAL_POLYHARMONIC_FUNDAMENTAL_PATTERN,
        semantic_text,
        re.IGNORECASE | re.DOTALL,
    ):
        primary, alternative = (
            "radial_laplacian_recurrence_and_flux_normalization",
            "fourier_symbol_and_distributional_constant_check",
        )
    elif "global_connectivity" in risks:
        primary, alternative = (
            "frontier_state_dp_with_connectivity",
            "exact_small_case_enumeration_and_subtour_check",
        )
    elif re.search(
        r"CW\s*(?:复形|complex(?:es)?)|胞腔(?:同调|链复形|边界(?:映射|算子)?)|同调群|"
        r"附着映射|粘附映射|"
        r"\b(?:cellular\s+(?:homology|chain\s+complex|boundary(?:\s+map)?)|"
        r"homology\s+groups?|attaching\s+maps?)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "cellular_chain_complex_then_smith_normal_form",
            "fundamental_group_abelianization_check",
        )
    elif re.search(
        r"切比雪夫|极小极大(?:多项式|逼近)?|最佳一致逼近|等振荡|交错定理|"
        r"\b(?:chebyshev|minimax\s+(?:polynomial|approximation)|best\s+uniform\s+approximation|"
        r"equioscillation|alternation\s+theorem)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "chebyshev_affine_map_and_normalized_alternation",
            "equioscillation_linear_system_check",
        )
    elif re.search(
        r"拉丁方|拉丁矩阵|行列(?:均|各)(?:为|是)?排列|"
        r"每个符号[^。！？\n]{0,50}每行(?:和|与|、)?每列[^。！？\n]{0,30}(?:恰好|正好)?出现一次|"
        r"\b(?:latin\s+squares?|rows?\s+and\s+columns?\s+(?:are|form)\s+permutations?|"
        r"each\s+row\s+and\s+(?:each\s+)?column\s+(?:is|forms?)\s+a\s+permutation|"
        r"row(?:\s*[/&-]\s*|\s+and\s+)column\s+permutations?|"
        r"each\s+symbol\s+(?:must\s+)?occurs?\s+exactly\s+once\s+in\s+every\s+row\s+"
        r"and\s+(?:in\s+)?every\s+column|"
        r"every\s+row\s+and\s+(?:every\s+)?column\s+contains?\s+each\s+symbol\s+exactly\s+once)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "normalize_symmetry_then_exhaust_structural_cases",
            "exact_enumeration_with_orbit_size_check",
        )
    elif re.search(
        r"无处零流|处处非零流|图流多项式|循环空间|圈空间|Tutte\s*多项式|"
        r"\b(?:nowhere[- ]zero\s+(?:graph\s+)?flows?|flow\s+polynomial|cycle\s+space|"
        r"tutte\s+polynomial)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "cycle_space_coordinate_inclusion_exclusion",
            "tutte_flow_polynomial_or_exact_edge_enumeration",
        )
    elif re.search(
        r"分裂域|伽罗瓦|Galois|splitting\s+field|extension\s+degree",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "irreducible_factor_then_field_tower",
            "generate_all_roots_and_check_normality_degree",
        )
    elif re.search(r"傅里叶变换|Fourier\s*(?:变换|transform)", semantic_text, re.IGNORECASE):
        primary, alternative = (
            "split_into_standard_fourier_transform_pairs",
            "direct_integral_with_shift_and_normalization_check",
        )
    elif re.search(
        r"(?:tiles?|方砖|瓷砖)[\s\S]{0,500}(?:covering|覆盖)[\s\S]{0,500}"
        r"(?:same\s+nonzero\s+number|相同非零数|multiplicity|重数)",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "coverage_multiplicity_double_counting",
            "periodic_construction_and_boundary_residue_check",
        )
    elif re.search(
        r"tournaments?[\s\S]{0,700}(?:arrives?|到达)[\s\S]{0,500}"
        r"(?:departs?|离开)[\s\S]{0,500}(?:hotel|住宿|stay)",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "arrival_departure_interval_lower_bound",
            "explicit_complete_schedule_and_cost_recount",
        )
    elif re.search(
        r"bijection[\s\S]{0,700}(?:x_?\{?1\}?\s*\+\s*1|coordinate\s+translations?)"
        r"|双射[\s\S]{0,700}(?:坐标平移|横纵坐标)",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "translation_invariant_order_and_ideal_bounds",
            "construct_extreme_linear_extensions",
        )
    elif re.search(
        r"(?:flood|green\s+cells?|洪水|绿色格)[\s\S]{0,700}"
        r"(?:grid|neighbou?rhood|boundary|网格|邻域|边界)|"
        r"(?:grid|网格)[\s\S]{0,500}(?:green\s+cells?|绿色格)",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "isoperimetric_boundary_growth_bound",
            "finite_window_simulation_then_extremal_construction",
        )
    elif re.search(
        r"(?:多米诺|多连方|铺砌|铺满)[\s\S]{0,700}(?:矩形|棋盘|最少|最小)|"
        r"\b(?:domino(?:es)?|tromino(?:es)?|tetromino(?:es)?|hexomino(?:es)?|"
        r"polyomino(?:es)?)\b[\s\S]{0,700}\b(?:tile|tiling|rectangle|board|minimum|minimal)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "tiling_coloring_and_cut_invariant",
            "boundary_profile_dp_then_periodic_construction",
        )
    elif re.search(
        r"(?:单位立方体|小立方体)[\s\S]{0,700}(?:截面|薄片|三个方向)[\s\S]{0,500}"
        r"(?:颜色集合|不同颜色)|"
        r"\bunit\s+cubes?\b[\s\S]{0,700}\b(?:slices?|rectangular\s+prisms?)\b"
        r"[\s\S]{0,500}\b(?:sets?\s+of\s+(?:distinct\s+)?colou?rs?|orientations?)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "slice_color_incidence_chain_bound",
            "nested_layer_construction_and_small_order_check",
        )
    elif re.search(
        r"\bn\W*good\s+functions?\b[\s\S]{0,900}\bexotic\s+integers?\b|"
        r"n[- ]?好函数[\s\S]{0,900}奇异整数",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "classify_divisibility_preserving_integer_functions",
            "parity_characterization_then_rank_formula",
        )
    elif re.search(
        r"(?:方程|等式)[\s\S]{0,300}(?:全部|所有)正整数(?:有序)?解|"
        r"\b(?:determine|find|classify)\s+all\s+positive\s+integer"
        r"(?:\s+ordered)?\s+(?:pairs?|solutions?)\b",
        semantic_text,
        re.IGNORECASE,
    ) and re.search(
        r"[xy]\s*\^\s*\{?2\}?[\s\S]{0,180}[xy]|"
        r"(?:Vieta|韦达|二次方程|quadratic)",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "vieta_jumping_descent",
            "pell_discriminant_recurrence_check",
        )
    elif re.search(
        r"轮流(?:选择|取|放|移动)|回合制|"
        r"\b(?:players?[^.!?]{0,80}take\s+turns|take\s+turns|turn[- ]based game|optimal play|moves? first)\b",
        semantic_text,
        re.IGNORECASE,
    ) and not re.search(
        r"概率|随机|期望|方差|掷|骰子|硬币|马尔可夫|"
        r"\b(?:probability|random(?:ly)?|expected|expectation|variance|"
        r"(?:roll|flip)(?:ing|s|ed)?|fair\s+(?:die|dice|coin)|markov)\b",
        semantic_text,
        re.IGNORECASE,
    ):
        primary, alternative = (
            "state_space_minimax_with_terminal_payoff",
            "small_parameter_game_tree_then_invariant_strategy",
        )
    elif topic == "olympiad_inequality":
        primary, alternative = (
            "sharp_inequality_with_equality_or_limit_case",
            "extremal_family_and_parameter_check",
        )
    elif topic == "olympiad_functional_equation":
        primary, alternative = (
            "special_substitutions_then_case_exhaustion",
            "verify_candidates_and_force_injectivity_or_periodicity",
        )
    elif topic == "olympiad_polynomial":
        primary, alternative = (
            "factorization_derivative_or_root_configuration",
            "degree_multiplicity_and_numeric_root_check",
        )
    elif topic == "olympiad_combinatorics":
        primary, alternative = (
            "bijection_recurrence_or_state_dp",
            "exact_small_case_enumeration",
        )
    elif topic == "olympiad_sequence":
        primary, alternative = (
            "invariant_and_recurrence",
            "construct_candidates_then_exclude_others",
        )
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
    if profile.answer_shape == "truth" and re.search(
        r"不可约|可约|\b(?:ir)?reducible\b",
        text,
        re.IGNORECASE,
    ):
        # Reducibility is naturally reported as a mathematical predicate
        # ("f is irreducible"), not as a bare yes/no sentence.  The strict
        # judgement requirement below still enforces an explicit verdict.
        return AnswerFrame("math", predicate="不可约性判断", question_kind="math")
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
    # A proof/justification sentence often follows the result sentence and is
    # intentionally not split into a second external goal. Preserve that
    # global support obligation in the single answer contract.
    if (
        not _is_result_or_nonexistence_alternative(semantic_text)
        and (
            _mandatory_result_support_clause(semantic_text)
            or re.search(
                r"说明.*(?:理由|为何|原因)|解释|证明|论证|推导|归一化|规范化|"
                r"prove|show|explain|justify|derive|normalization|normalisation|"
                r"justification|show\s+your\s+work",
                semantic_text,
                re.IGNORECASE,
            )
        )
        and "reasoning" not in support
    ):
        support = (*support, "reasoning")
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
            *((support) if len(goals) == 1 else ()),
            *(("reasoning",) if formal_proof else ()),
        ))),
        unit=_explicit_unit(goal.instruction) or (unit if len(goals) == 1 else ""),
        validation_requirements=tuple(dict.fromkeys((
            *(requirement.name for requirement in goal.requirements),
            *((support) if len(goals) == 1 else ()),
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
    r"报告|返回|表达(?!式)|陈述|"
    r"\b(?:prove|show|find|determine|solve|calculate|compute|evaluate|verify|compare|"
    r"explain|justify|derive|construct|classify|state|report|return|express|identify|"
    r"describe|give|write|list)\b|"
    r"what\s+(?:is|are)|is\s+it\s+possible|if\s+(?:it\s+)?is\s+possible"
)

_SPLIT_GOAL_COMMAND = (
    r"求|计算|判断|给出|写出|指出|列出|构造|报告|返回|表达(?!式)|陈述|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|construct|classify|state|report|"
    r"return|express|identify|describe|give|write|list)\b|what\s+(?:is|are)|is\s+it\s+possible|"
    r"if\s+(?:it\s+)?is\s+possible"
)

_OUTPUT_DIRECTIVE = (
    r"(?:\b(?:report|return|express)\b|"
    r"\bstate\b(?!\s+(?:space|recursion|transition|equation|variable|vector|diagram|process))|"
    r"报告|返回|表达(?!式)|陈述)"
)


def _output_transform_requirements(text: str) -> tuple[Requirement, ...]:
    """Record post-processing asks that a base mathematical result cannot cover."""
    value = str(text or "")
    requirements: list[Requirement] = []
    if re.search(
        rf"{_OUTPUT_DIRECTIVE}[^\n。！？.!?]{{0,120}}"
        r"(?:\bmodulo\b|\bmod\s+\d|\bremainder\b|取模|模\s*\d|余数)",
        value,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "output_modulo_transform",
            (("mod",), ("remainder",), ("余数",), ("取模",)),
            strict=False,
            category="result",
        ))
    output_clause = re.search(
        rf"{_OUTPUT_DIRECTIVE}[^\n。！？.!?]{{0,180}}",
        value,
        re.IGNORECASE,
    )
    if output_clause and re.search(
        r"\b(?:the\s+)?(?:number|count|cardinality)\s+of\s+"
        r"(?:such|these|those|the\s+(?:above|resulting|valid|possible))\b|"
        r"\bhow\s+many\s+(?:such|these|those|valid|possible)\b|"
        r"(?:这些|上述|所得|符合条件的)[^。！？.!?\n]{0,30}(?:数量|个数)|"
        r"\b(?:rather\s+than|instead\s+of)\b|\bnot\b[^.!?\n]{0,40}\bthemselves\b|"
        r"而不是|而非|不要[^。！？\n]{0,30}本身",
        output_clause.group(0),
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "output_object_transform",
            (("number",), ("count",), ("cardinality",), ("数量",), ("个数",)),
            strict=False,
            category="result",
        ))
    return tuple(requirements)

_ELABORATION_CLAUSE = re.compile(
    r"^\s*(?:给出|写出|指出|列出|(?:state|identify|write|give)\b).*"
    r"(?:所用|理由|依据|条件|结论|含义|计算式|截断|具体公式|转移矩阵|"
        r"证明|论证|derivation|proof|justification|reason|condition|criterion|conclusion|supporting formula|"
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
        result_with_support = bool(
            _mandatory_result_support_clause(item)
            or (
                profile.problem_type == "proof"
                and re.search(
                r"(?:求|确定|计算|find|determine|compute)[\s\S]{0,900}"
                r"(?:证明|论证|\b(?:proof|argument|justification)\b)",
                item,
                re.IGNORECASE,
            )
            )
        )
        target = (
            item
            if profile.answer_shape == "choice" or result_with_support
            else extract_target_clause(item)
        )
        part_profile = classify_profile(target)
        part_shape = "choice" if profile.answer_shape == "choice" else part_profile.answer_shape
        if re.search(r"分裂域|splitting\s+field", target, re.IGNORECASE):
            part_shape = "expression"
        elif re.search(r"填[^。\n]*(?:是[^。\n]*否|否[^。\n]*是)|yes\s+or\s+no", target, re.IGNORECASE):
            part_shape = "truth"
        requirements = list(_requirements(target, part_shape))
        if (
            _requires_almost_everywhere_zero_conclusion(item)
            and not any(
                requirement.name == "almost_everywhere_zero"
                for requirement in requirements
            )
        ):
            requirements.append(Requirement(
                "almost_everywhere_zero",
                (("f=0", "几乎处处"), ("f=0", "almost everywhere")),
                strict=True,
            ))
        if len(selected) == 1:
            existing_names = {requirement.name for requirement in requirements}
            requirements.extend(
                requirement
                for requirement in _parallel_result_requirements(item)
                if requirement.name not in existing_names
            )
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
        r"(?:求|计算|写出|给出|报告|返回|表达(?!式)|陈述|"
        r"find|determine|solve\s+for|calculate|compute|report|return|express|state)\s*"
        r"(?:the\s+(?:value|values)\s+of\s+)?"
        r"([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)\b",
        value,
        re.IGNORECASE,
    ))
    return matches[-1].group(1) if matches else ""


def _mandatory_result_support_clause(text: str) -> bool:
    """Recognize proof or normalization requirements attached to one result."""
    value = str(text or "")
    result = re.search(
        r"(?:求出?|计算|确定|写出|给出|构造|"
        r"\b(?:find|calculate|compute|determine|write|give|construct|evaluate)\b)",
        value,
        re.IGNORECASE,
    )
    if not result:
        return False
    suffix = value[result.end():]
    return bool(re.search(
        r"(?:证明|论证|推导|归一化|规范化|常数校准|系数校准)"
        r"[^。！？\n]{0,50}(?:须|需|必须|应当?|要求)|"
        r"(?:须|需|必须|应当?|要求)[^。！？\n]{0,80}"
        r"(?:证明|论证|推导|说明[^。！？\n]{0,35}(?:归一化|规范化|常数|系数))|"
        r"\b(?:proof|argument|derivation|normalization|normalisation|"
        r"normalizing\s+constant|normalising\s+constant)\b"
        r"[^.!?\n]{0,60}\b(?:must|shall|should|is\s+required\s+to)\b|"
        r"\b(?:must|shall|should|required\s+to)\b[^.!?\n]{0,90}"
        r"\b(?:prove|derive|justify|show|check|fix)\b[^.!?\n]{0,45}"
        r"\b(?:normalization|normalisation|constant|coefficient)?\b",
        suffix,
        re.IGNORECASE,
    ))


def _split_goal_text(text: str) -> list[str]:
    """Split only explicit independent asks; keep conditions attached otherwise."""
    value = str(text or "").strip()
    if not value:
        return [value]

    # A mandatory proof/argument clause specifies how the requested result
    # must be established; it is not an independent answer target.  Keeping
    # the full text also protects notation such as H_1(X; R), whose semicolon
    # is mathematical punctuation rather than a multipart delimiter.
    if _mandatory_result_support_clause(value):
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
        r"(?:give|write|list|show|state|report|return|express|identify|describe)\b",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:wants?|wanted|tries|tried|aims?|attempts?)\s+to\s+"
        r"(?:find|determine|solve|calculate|compute|evaluate|describe|give|write|list|"
        r"report|return|express|state)\b",
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


def _requires_almost_everywhere_zero_conclusion(text: str) -> bool:
    """Recognize the standard zero-integral theorem's requested corollary."""
    return all((
        re.search(
            r"(?<![A-Za-z])f\s*(?:≥|\\geq?|\\ge)\s*0|"
            r"\bnonnegative\b[^。.!?\n]{0,35}\bf\b|"
            r"\bf\b[^。.!?\n]{0,35}\bnonnegative\b",
            text,
            re.IGNORECASE,
        ),
        re.search(
            r"(?:∫|\\int)[^。.!?\n]{0,60}\bf\b[^。.!?\n]{0,60}"
            r"(?:=|等于|为|is|equals?)\s*0",
            text,
            re.IGNORECASE,
        ),
        re.search(
            r"(?:\{\s*f\s*(?:≥|\\geq?|\\ge)|level\s+set)[^。.!?\n]{0,100}"
            r"(?:零测集|测度为零|measure\s+(?:is\s+)?zero|measure\s+0)",
            text,
            re.IGNORECASE,
        ),
        re.search(
            r"写出(?:其|最后|最终)?结论|得出(?:其|最后|最终)?结论|"
            r"\b(?:state|write|give)\s+(?:the\s+)?conclusion\b|"
            r"\band\s+conclude\b",
            text,
            re.IGNORECASE,
        ),
    ))


def _requirements(text: str, answer_shape: str) -> tuple[Requirement, ...]:
    requirements: list[Requirement] = list(_output_transform_requirements(text))
    alternative_result = _is_result_or_nonexistence_alternative(text)
    if alternative_result:
        requirements.append(Requirement(
            "alternative_result",
            (("number",), ("does not exist",), ("不存在",)),
            strict=True,
        ))
    if re.search(LACUNARY_NATURAL_BOUNDARY_PATTERN, text, re.IGNORECASE | re.DOTALL):
        radius_requested = bool(re.search(
            r"(?:求|确定|计算|给出|写出|判定)[^。.!?\n]{0,100}收敛半径|"
            r"\b(?:find|finding|determine|determining|compute|computing|give|state)\b[^.!?\n]{0,100}"
            r"\b(?:the\s+)?radius(?:\s+of\s+convergence)?\b",
            text,
            re.IGNORECASE,
        ))
        domain_requested = bool(re.search(
            r"(?:求|确定|给出|写出|判定)[^。.!?\n]{0,100}(?:收敛域|收敛圆盘|解析域)|"
            r"\b(?:find|determine|give|state)\b[^.!?\n]{0,120}"
            r"\b(?:domain|disk)\s+of\s+(?:convergence|analyticity)\b",
            text,
            re.IGNORECASE,
        ))
        boundary_requested = bool(re.search(
            r"(?:证明|说明|判定|判断|确定|验证)[^。.!?\n]{0,160}自然边界|"
            r"自然边界[^。.!?\n]{0,80}(?:证明|说明|判定|判断|确定|验证)|"
            r"\b(?:prove|show|determine|decide|verify|whether)\b[^.!?\n]{0,180}"
            r"\bnatural\s+boundary\b|"
            r"\bis\b[^.!?\n]{0,120}\ba\s+natural\s+boundary\b|"
            r"(?:解析延拓|全纯延拓)[^。.!?\n]{0,160}"
            r"(?:每(?:一|条|个)?(?:边界)?圆?弧|任意(?:边界)?圆?弧|整个(?:收敛)?圆周)|"
            r"\banalytic\s+continuation\b[^.!?\n]{0,180}"
            r"\b(?:every|each|any)\s+(?:boundary\s+)?arc\b",
            text,
            re.IGNORECASE,
        ))
        if radius_requested:
            requirements.append(Requirement(
                "convergence_radius",
                (("收敛半径",), ("radius", "convergence"), ("r=",)),
                strict=True,
            ))
        if domain_requested:
            requirements.append(Requirement(
                "convergence_domain",
                (
                    ("收敛域",),
                    ("收敛圆盘",),
                    ("domain", "convergence"),
                    ("|z|<",),
                    (r"\lvert", "z", r"\rvert", "<"),
                ),
                strict=True,
            ))
        if boundary_requested:
            requirements.append(Requirement(
                "natural_boundary_classification",
                (
                    ("自然边界",),
                    ("natural", "boundary"),
                    ("不能", "解析延拓"),
                    ("cannot", "analytically", "continue"),
                ),
                strict=True,
            ))
    if re.search(RUNGE_KUTTA_STABILITY_PATTERN, text, re.IGNORECASE | re.DOTALL):
        if re.search(r"(?:求|确定|推导|写出|给出|计算)[^。.!?\n]{0,100}稳定函数|"
                     r"\b(?:find|determine|derive|write|give|compute)\b[^.!?\n]{0,120}"
                     r"\bstability\s+function\b", text, re.IGNORECASE):
            requirements.append(Requirement(
                "stability_function",
                (("稳定函数",), ("stability", "function"), ("r(z)", "=")),
                strict=True,
            ))
        if re.search(r"(?:无穷远|无穷大)[^。.!?\n]{0,50}(?:极限|趋于)|"
                     r"(?:极限|趋于)[^。.!?\n]{0,50}(?:无穷远|无穷大)|"
                     r"\b(?:limit|as\s+z)\b[^.!?\n]{0,80}\b(?:infinity|infty)\b|"
                     r"R\s*\(\s*(?:\\infty|∞)\s*\)", text, re.IGNORECASE):
            requirements.append(Requirement(
                "stability_infinity_limit",
                (("无穷", "极限"), ("lim", "infty"), ("r(infty)",), ("∞",)),
                strict=True,
            ))
    for parameter in _answer_parameters(text):
        requirements.append(Requirement(
            f"parameter_dependency_{parameter.lower()}",
            ((parameter,),),
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
    surface_derivative_context = bool(re.search(
        r"曲面|二阶偏导|黑塞|\\?operatorname\s*\{?Hess|\b(?:surface|hessian)\b|"
        r"\bsecond[- ](?:order\s+)?partial\s+derivatives?\b",
        text,
        re.IGNORECASE,
    ))
    curve_derivative_request = not surface_derivative_context and bool(re.search(
        r"曲线|\\?gamma|γ|\bcurve\b",
        text,
        re.IGNORECASE,
    ))
    if not curve_derivative_request and re.search(
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
    if re.search(
        r"(?:列出|写出|给出|枚举)[^。.!?\n]{0,30}条件样本空间|"
        r"条件样本空间[^。.!?\n]{0,30}(?:列出|写出|给出|枚举)|"
        r"\b(?:list|write|give|state|enumerate)\b[^.!?\n]{0,50}"
        r"\bconditional\s+sample\s+space\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "conditional_sample_space",
            (("条件样本空间",), ("conditional sample space",), ("omega", "=")),
            strict=True,
        ))
    if re.search(
        r"(?:识别|认出|指出)[^。.!?\n]{0,30}(?:其|为|是)?[^。.!?\n]{0,12}方差|"
        r"\b(?:identify|recognize)\b[^.!?\n]{0,50}\bvariance\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "variance_identification",
            (("e[(x-p)^2]", "var(x)"), ("方差",), ("variance",)),
            strict=True,
        ))
    if re.search(
        r"(?:说明|指出|识别|认出|写出)[^。.!?\n]{0,40}几何分布|"
        r"\b(?:identify|recognize|describe|state)\b[^.!?\n]{0,55}"
        r"\bgeometric\s+distribution\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "geometric_distribution_identification",
            (("几何分布",), ("geometric", "distribution"), ("geom",)),
            strict=True,
        ))
    distribution_is_support = bool(re.search(
        r"(?:求|计算)[^。.!?\n]{0,120}概率[^。.!?\n]{0,80}"
        r"(?:说明|指出|识别|认出)[^。.!?\n]{0,40}几何分布|"
        r"\b(?:find|compute|calculate)\b[^.!?\n]{0,120}\bprobability\b"
        r"[^.!?\n]{0,80}\b(?:identify|recognize|describe|state)\b"
        r"[^.!?\n]{0,40}\bgeometric\s+distribution\b",
        text,
        re.IGNORECASE,
    ))
    if answer_shape != "choice" and not distribution_is_support and re.search(
        r"(?:求|确定|写出|给出)[^。.!?\n]{0,80}(?:条件)?(?<!初始)(?<!同)分布(?:律)?|"
        r"\b(?:find|determine|write|give)\s+(?:the\s+)?distribution\s+of\b|"
        r"\b(?:find|determine|write|give)\b[^.!?\n]{0,100}"
        r"\b(?:conditional|joint|marginal|sampling|limiting|stationary|probability)\s+distribution\b|"
        r"\bwhat\s+is\b[^.!?\n]{0,80}\bdistribution\b",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "distribution_result",
            (("分布",), ("distribution",), (r"\sim",), ("poisson",), ("normal",)),
            strict=True,
        ))
    if not alternative_result and (
        _mandatory_result_support_clause(text)
        or re.search(
            r"说明.*(?:理由|为何|原因)|解释|证明|论证|推导|归一化|规范化|"
            r"prove|show|explain|justify|derive|normalization|normalisation|justification|"
            r"show\s+your\s+work",
            text,
            re.IGNORECASE,
        )
    ):
        requirements.append(Requirement("reasoning", (("因为",), ("依据",), ("because",), ("since",), ("therefore",)), strict=False))
    if re.search(r"牛顿法|newton", text, re.IGNORECASE):
        requirements.append(Requirement("iteration_formula", (("xn+1",), ("xk+1",), ("迭代公式",))))
        if re.search(r"x_0|初值", text):
            requirements.append(Requirement("first_iteration", (("x1",), ("第一次迭代",))))
    if re.search(r"逐点.*(?:极限|收敛)|pointwise", text, re.IGNORECASE):
        requirements.append(Requirement(
            "pointwise_limit",
            (
                ("逐点极限",),
                ("pointwise limit",),
                ("逐点", "\\to"),
                ("pointwise", "\\to"),
            ),
            strict=True,
        ))
    if re.search(
        r"(?:是否|判断|能否)[^。.!?\n]{0,160}(?:l\s*\^?\s*1|l\s*_?\s*\{?1\}?)[^。.!?\n]{0,80}(?:收敛|极限)|"
        r"\b(?:whether|determine)\b[^.!?\n]{0,160}\bL\s*\^?\s*1\b[^.!?\n]{0,80}\bconverge",
        text,
        re.IGNORECASE,
    ):
        requirements.append(Requirement(
            "l1_norm_check",
            (
                ("范数",),
                (r"\lvert", "积分"),
                (r"\lVert",),
                ("l1", "norm"),
                ("积分", "1"),
            ),
            strict=True,
        ))
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


def _parallel_result_requirements(text: str) -> tuple[Requirement, ...]:
    """Capture named result components inside one independent problem.

    These obligations deliberately do not create additional goals.  Each rule
    requires an explicit request plus a small, recognizable group of named
    outputs, so ordinary conjunctions in hypotheses remain untouched.
    """
    value = str(text or "")
    request = r"(?:求|问|试问|计算|写出|给出|find|determine|calculate|compute|give)"
    groups: tuple[
        tuple[str, tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]], ...
    ] = (
        (
            rf"{request}[\s\S]{{0,520}}(?:产量|quantity)[\s\S]{{0,260}}(?:利润|profit)|"
            rf"{request}[\s\S]{{0,520}}(?:利润|profit)[\s\S]{{0,260}}(?:产量|quantity)",
            (
                ("production_quantity", (("产量",), ("生产",), ("quantity",), ("q=",))),
                ("profit_value", (("利润",), ("profit",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,520}}(?:产量|quantity)[\s\S]{{0,260}}(?:价格|price)|"
            rf"{request}[\s\S]{{0,520}}(?:价格|price)[\s\S]{{0,260}}(?:产量|quantity)",
            (
                ("production_quantity", (("产量",), ("生产",), ("quantity",), ("q=",))),
                ("price_value", (("价格",), ("price",), ("p=",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:最大高度|maximum\s+height)[\s\S]{{0,220}}(?:所需)?时间|"
            rf"{request}[\s\S]{{0,420}}(?:所需)?时间[\s\S]{{0,220}}(?:最大高度|maximum\s+height)",
            (
                ("maximum_height", (("最大高度",), ("maximum", "height"))),
                ("time_to_peak", (("时间",), ("time",))),
            ),
        ),
        (
            rf"(?:问|求)[\s\S]{{0,420}}折起高度[\s\S]{{0,220}}(?:最大)?截面积",
            (
                ("fold_height", (("折起高度",), ("高度",))),
                ("cross_section_area", (("截面积",), ("cross", "section", "area"))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:长和宽|长、宽|length\s+and\s+width)"
            rf"[\s\S]{{0,220}}(?:最大)?面积",
            (
                ("length_value", (("长",), ("length",))),
                ("width_value", (("宽",), ("width",))),
                ("area_value", (("面积",), ("area",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:位移|displacement)[\s\S]{{0,220}}(?:速度|velocity)|"
            rf"{request}[\s\S]{{0,420}}(?:速度|velocity)[\s\S]{{0,220}}(?:位移|displacement)",
            (
                ("displacement_value", (("位移",), ("displacement",))),
                ("velocity_value", (("速度",), ("velocity",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,520}}(?:边长|side\s+length)[\s\S]{{0,220}}(?:最大)?面积",
            (
                ("side_length", (("边长",), ("side", "length"))),
                ("area_value", (("面积",), ("area",))),
            ),
        ),
        (
            r"(?:市场价格|prices?)[\s\S]{0,300}(?:p_?1|p1)[\s\S]{0,120}(?:p_?2|p2)|"
            r"(?:p_?1|p1)[\s\S]{0,120}(?:p_?2|p2)[\s\S]{0,300}(?:市场价格|prices?)",
            (
                ("market_price_p1", (("p1=",), ("p_1=",), ("甲地", "价格"))),
                ("market_price_p2", (("p2=",), ("p_2=",), ("乙地", "价格"))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:最优解|optimal\s+solution)"
            rf"[\s\S]{{0,180}}(?:最优值|optimal\s+value)",
            (
                ("optimal_solution", (("最优解",), ("optimal", "solution"), ("(x,y)=",))),
                ("optimal_value", (("最优值",), ("optimal", "value"))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}极大元[\s\S]{{0,140}}极小元"
            rf"[\s\S]{{0,180}}最长链",
            (
                ("maximal_element", (("极大元",),)),
                ("minimal_element", (("极小元",),)),
                ("longest_chain", (("最长链",),)),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:平均曲率|mean\s+curvature)"
            rf"[\s\S]{{0,180}}(?:高斯曲率|gaussian\s+curvature)",
            (
                ("mean_curvature", (("平均曲率",), ("mean", "curvature"))),
                ("gaussian_curvature", (("高斯曲率",), ("gaussian", "curvature"), ("k=",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:分布|distribution)"
            rf"[\s\S]{{0,180}}(?:方差|variance)",
            (
                ("distribution_result", (("分布",), ("distribution",), ("x+y", "~"))),
                ("variance_result", (("方差",), ("variance",), ("var",))),
            ),
        ),
        (
            r"(?:展开|写出[^。.!?\n]{0,80}(?:展开式|幂级数)|"
            r"\bexpand\b|\bwrite\b[^.!?\n]{0,80}\bpower\s+series)"
            r"[\s\S]{0,420}(?:幂级数|power\s+series)"
            rf"[\s\S]{{0,180}}(?:收敛半径|radius\s+of\s+convergence)",
            (
                ("series_expansion", (("级数",), ("series",), ("sum",), ("Σ",))),
                ("convergence_radius", (("收敛半径",), ("radius", "convergence"), ("r=",))),
            ),
        ),
        (
            rf"{request}[\s\S]{{0,420}}(?:平衡点|equilibrium)"
            rf"[\s\S]{{0,180}}(?:稳定性|stability)",
            (
                ("equilibrium_point", (("平衡点",), ("原点",), ("equilibrium",), ("origin",))),
                ("stability_type", (("稳定",), ("stability",), ("saddle",), ("鞍点",))),
            ),
        ),
    )
    found: list[Requirement] = []
    seen: set[str] = set()
    for pattern, components in groups:
        if not re.search(pattern, value, re.IGNORECASE):
            continue
        for name, alternatives in components:
            if name not in seen:
                found.append(Requirement(name, alternatives, strict=True))
                seen.add(name)
    return tuple(found)


def _answer_parameters(text: str) -> tuple[str, ...]:
    """Return parameters that the requested result must explicitly retain."""
    value = str(text or "")
    parameters: list[str] = []
    for match in re.finditer(
        r"\b([A-Za-z])\s*=\s*\1\s*\(\s*([A-Za-z](?:\s*,\s*[A-Za-z])*)\s*\)",
        value,
        re.IGNORECASE,
    ):
        parameters.extend(re.findall(r"[A-Za-z]", match.group(2)))
    for match in re.finditer(
        r"\b(?:as\s+a\s+function\s+of|in\s+terms\s+of)\s+"
        r"(?:the\s+parameters?\s+)?\(?\s*\$?"
        r"([A-Za-z](?:\s*(?:,|and)\s*[A-Za-z])*)(?![A-Za-z])",
        value,
        re.IGNORECASE,
    ):
        parameters.extend(re.findall(r"[A-Za-z]", match.group(1)))
    return tuple(dict.fromkeys(parameter.lower() for parameter in parameters))


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
        (
            r"一阶和二阶导数|\bfirst(?:[- ]order)?\s+and\s+second(?:[- ]order)?\s+derivatives?\b",
            "first_second_derivatives",
            (("一阶", "二阶"), ("γ'", "γ''"), ("first", "second", "derivative")),
        ),
        (r"按不交集合分解", "disjoint_decomposition", (("不交",), ("分解",))),
        (r"使用组合数计算", "combination_calculation", (("组合数",), ("binom",), ("c_",))),
        (r"先作部分分式", "partial_fraction", (("部分分式",),)),
        (r"识别极点阶数", "pole_order", (("极点", "阶"), ("一阶极点",))),
        (r"使用柯西积分公式", "cauchy_formula", (("柯西积分公式",),)),
        (r"使用积分因子", "integrating_factor", (("积分因子",),)),
        (r"先写特征方程", "characteristic_equation", (("特征方程",),)),
        (r"通过分离变量", "separation_of_variables", (("分离变量",),)),
        (
            r"(?:利用|使用|说明)[^。.!?\n]{0,30}独立增量|"
            r"\b(?:use|using|explain|show)\b[^.!?\n]{0,45}\bindependent\s+increments?\b",
            "independent_increments",
            (("独立增量",), ("independent", "increment")),
        ),
        (
            r"说明[^。.!?\n]{0,35}独立性[^。.!?\n]{0,25}(?:用在|用途|作用|何处)|"
            r"\b(?:explain|state|show)\b[^.!?\n]{0,60}\b(?:where|how)\b"
            r"[^.!?\n]{0,35}\bindependence\b[^.!?\n]{0,20}\bused\b",
            "independence_use",
            (("独立",), ("independence",)),
        ),
        (
            r"说明[^。.!?\n]{0,30}独立同分布(?:假设)?|"
            r"\b(?:explain|state|use)\b[^.!?\n]{0,45}"
            r"\b(?:i\.?i\.?d\.?|independent\s+and\s+identically\s+distributed)\b",
            "iid_variance_scaling",
            (("独立同分布",), ("iid",), ("i.i.d",), ("independent", "identically distributed")),
        ),
        (
            r"强大数律|\bstrong\s+law(?:\s+of\s+large\s+numbers)?\b",
            "strong_law",
            (("强大数律",), ("strong", "law")),
        ),
        (
            r"(?:使用|利用)中心差分公式|\b(?:use|using)\s+(?:the\s+)?central\s+difference\s+formula\b",
            "central_difference_formula",
            (("中心差分公式",), ("central", "difference", "formula")),
        ),
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
    if getattr(profile, "topic", "general") in SPECIALIZED_TOPICS:
        score += 2
    if set(risks) & {
        "exhaustiveness_required", "integer_constraints", "functional_equation",
        "diagram_dependency", "parameter_dependency", "extremal_two_sided_bound",
        "global_connectivity", "statement_integrity_audit",
    }:
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
    if topic == "directed_euler_circuits":
        risks.append("euler_circuit_normalization")
    elif topic == "plane_rooted_tree_enumeration":
        risks.append("lukasiewicz_prefix_and_cycle_normalization")
    elif topic == "lacunary_natural_boundary":
        risks.extend(["theorem_scope", "analytic_continuation_scope"])
    elif topic == "runge_kutta_stability":
        risks.append("order_and_stability_conditions")
    elif topic == "spherical_triangle_area":
        risks.append("spherical_radius_angle_and_unit_normalization")
    elif topic == "weierstrass_sine_product":
        risks.append("entire_product_zero_set_and_normalization")
    elif topic == "two_dimensional_polyharmonic_fundamental_solution":
        risks.append("distributional_sign_and_flux_normalization")
    if re.search(r"(?:求|找出|确定|列出).*所有|\b(?:find|determine|classify)\s+all\b", text, re.IGNORECASE):
        risks.append("exhaustiveness_required")
    if topic == "olympiad_number_theory" or re.search(r"正整数|整数解|positive integers?|integer solutions?", text, re.IGNORECASE):
        risks.append("integer_constraints")
    if topic == "olympiad_functional_equation":
        risks.append("functional_equation")
    geometry_relations = set(re.findall(
        r"\b(?:angle|parallel|perpendicular|similar|congruent|cyclic|tangent|collinear|"
        r"concurrent|circumcircle|incircle|orthocenter|circumcenter|incenter|"
        r"角|平行|垂直|相似|全等|共圆|切线|共线|共点|外接圆|内切圆)\b",
        text,
        re.IGNORECASE,
    ))
    if topic == "olympiad_geometry" and (
        len(geometry_relations) >= 2
        or re.search(r"\b(?:diagram|figure)\b|如图|图中", text, re.IGNORECASE)
    ):
        risks.append("diagram_dependency")
    if topic in {"olympiad_combinatorics", "olympiad_sequence"}:
        risks.append("case_analysis")
    if _answer_parameters(text):
        risks.append("parameter_dependency")
    if re.search(
        r"\b(?:minimum|maximum|smallest|largest|least|greatest|best)\s+"
        r"(?:possible\s+)?(?:value|constant|integer|number|[A-Za-z])\b|"
        r"最小|最大|最优|最佳常数",
        text,
        re.IGNORECASE,
    ):
        risks.append("extremal_two_sided_bound")
    if re.search(
        r"\bpermutations?\b[\s\S]{0,1800}\b(?:consecutive|exactly\s+one)\b|"
        r"排列[\s\S]{0,1800}(?:相邻|恰有一个)",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        risks.append("global_connectivity")
    if _needs_statement_integrity_audit(text):
        risks.append("statement_integrity_audit")
    return list(dict.fromkeys(risks))


def _needs_statement_integrity_audit(text: str) -> bool:
    """Flag strong signs of a corrupted or semantically detached late clause.

    This flag never removes text. It only asks the solver to compare the
    literal reading with a minimally repaired reading when the literal task is
    internally inconsistent or explicitly says that a clause is irrelevant.
    """
    value = str(text or "")
    tuple_dimensions = {
        int(item)
        for item in re.findall(
            r"\b(\d+)\s*[- ]\s*(?:dimensional\s+)?tuples?\b",
            value,
            re.IGNORECASE,
        )
    }
    incompatible_dimensions = len(tuple_dimensions) >= 2 and bool(re.search(
        r"\b(?:write|generate|obtain|produce|represent|map|transform)\b|"
        r"写出|生成|得到|表示|映射|变换",
        value,
        re.IGNORECASE,
    ))
    explicitly_irrelevant = bool(re.search(
        r"\b(?:additionally|also|in\s+addition)[\s\S]{0,700}?"
        r"(?:does\s+not|do\s+not|cannot)\s+(?:affect|change|influence)\b|"
        r"(?:另外|此外|附加)[\s\S]{0,500}?(?:不影响|不会改变|无关)",
        value,
        re.IGNORECASE,
    ))
    suspicious_addendum = bool(re.search(
        r"\b(?:in\s+addition\s+to\s+(?:the\s+)?(?:given|above)\s+constraints|"
        r"additionally)[\s\S]{0,500}?\b(?:random(?:ly)?|forced|broken|vip\s+lounge|"
        r"mischievous|artifact)\b|"
        r"(?:另外|此外|附加)[\s\S]{0,500}?(?:随机|强制|损坏|贵宾|恶作剧|神器)",
        value,
        re.IGNORECASE,
    ))
    stochastic_strategy_conflict = bool(
        re.search(r"\b(?:coin|die|dice|random(?:ly)?)\b", value, re.IGNORECASE)
        and re.search(r"\b(?:forced|required)\s+to\b", value, re.IGNORECASE)
        and re.search(
            r"\b(?:optimal(?:ly)?|perfect\s+strategy|wants?\s+to\s+(?:maximize|minimize)|"
            r"play(?:s|ed)?\s+optimally)\b",
            value,
            re.IGNORECASE,
        )
    )
    return bool(
        incompatible_dimensions
        or explicitly_irrelevant
        or suspicious_addendum
        or stochastic_strategy_conflict
    )
