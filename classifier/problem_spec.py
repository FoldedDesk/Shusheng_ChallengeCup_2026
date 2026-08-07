"""Structured, local-only understanding of a mathematics problem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from classifier.profile import ProblemProfile, classify_profile


@dataclass(frozen=True)
class Requirement:
    """A gradable answer obligation with acceptable local textual anchors."""

    name: str
    alternatives: tuple[tuple[str, ...], ...]
    strict: bool = False

    def matches(self, compact_answer: str) -> bool:
        return any(all(_compact(term) in compact_answer for term in option) for option in self.alternatives)


def _compact(value: str) -> str:
    return re.sub(r"[\s{}()\[\]\\,，。；;：:_]", "", str(value or "").lower()).replace("−", "-")


@dataclass(frozen=True)
class Goal:
    id: str
    instruction: str
    answer_shape: str
    required_terms: tuple[str, ...] = ()
    requirements: tuple[Requirement, ...] = ()


@dataclass(frozen=True)
class AnswerFrame:
    style: str
    subject: str = ""
    predicate: str = ""
    unit: str = ""
    question_kind: str = "math"

    def trace_content(self) -> dict:
        return {"style": self.style, "predicate": self.predicate, "unit": self.unit, "question_kind": self.question_kind}


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

    def trace_content(self) -> dict:
        return {
            "profile": self.profile.trace_content(),
            "goal_count": len(self.goals),
            "goal_requirements": [list(goal.required_terms) for goal in self.goals],
            "risk_flags": list(self.risk_flags),
            "primary_method": self.primary_method,
            "answer_frame": self.answer_frame.trace_content(),
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
}


def build_problem_spec(problem: str) -> ProblemSpec:
    text = str(problem or "").strip()
    profile = classify_profile(text)
    goals = _goals(text, profile)
    constraints = _constraints(text)
    risks = _risks(text, profile, len(goals))
    primary, alternative = _METHODS.get(profile.subject, ("definition_and_case_analysis", "direct_check"))
    return ProblemSpec(
        profile, tuple(goals), tuple(constraints), tuple(risks), primary, alternative,
        _answer_frame(text, profile), _tool_can_answer_whole(text, profile, goals),
    )


def _answer_frame(text: str, profile: ProblemProfile) -> AnswerFrame:
    if profile.problem_type in {"proof", "derivation", "explanation"}:
        return AnswerFrame("proof", predicate="证明结论", question_kind="proof")
    age = re.search(r"问\s*([\u4e00-\u9fff]{2,6})(?:多少|几)岁", text)
    if not age:
        age = re.search(r"([\u4e00-\u9fff]{2,6})(?:今年|的年龄|现年)", text)
    if age:
        return AnswerFrame("sentence", subject=age.group(1), predicate="年龄", unit="岁", question_kind="age")
    if re.search(r"概率.*(?:多少|几)|(?:多少|几).*概率|probability", text, re.IGNORECASE):
        return AnswerFrame("sentence", predicate="概率", question_kind="probability")
    if re.search(r"是否|是不是|能否|可否|is it|whether", text, re.IGNORECASE):
        before = re.split(r"是否|是不是|能否|可否|\bis it\b|\bwhether\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        subject = re.sub(r"^(?:求|判断|说明|问)\s*", "", before).strip("，,。 ")[-100:]
        return AnswerFrame("sentence", subject=subject, predicate="判断", question_kind="truth")
    if re.search(r"(?:多少|几)个|number of", text, re.IGNORECASE):
        return AnswerFrame("sentence", predicate="数量", unit="个", question_kind="count")
    return AnswerFrame("math", question_kind="math")


def _goals(text: str, profile: ProblemProfile) -> list[Goal]:
    pieces = [piece.strip() for piece in re.split(r"[；;]\s*|(?:并且|且)(?=(?:求|证明|判断|说明|给出|prove|show|find|determine)\b)", text, flags=re.IGNORECASE) if piece.strip()]
    commands = [piece for piece in pieces if re.search(r"求|证明|判断|说明|给出|是否|prove|show|find|determine|solve", piece, re.IGNORECASE)]
    selected = commands or [text]
    return [
        Goal(
            f"g{index + 1}", item[:400], profile.answer_shape,
            _required_terms(item, profile.answer_shape), _requirements(item, profile.answer_shape),
        )
        for index, item in enumerate(selected[:4])
    ]


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
    if re.search(r"主曲率", text) and re.search(r"高斯曲率", text):
        required.extend(["主曲率", "高斯曲率"])
    if re.search(r"E,F,G|E，F，G", text):
        required.extend(["E=", "F=", "G="])
    # These are coverage hints, not literal admission gates. Variables and
    # theorem nouns from the question are intentionally not copied here:
    # equivalent answers often omit them while remaining fully gradable.
    return tuple(dict.fromkeys(required))


def _requirements(text: str, answer_shape: str) -> tuple[Requirement, ...]:
    requirements: list[Requirement] = []
    if re.search(r"牛顿法|newton", text, re.IGNORECASE):
        requirements.append(Requirement("iteration_formula", (("xn+1",), ("xk+1",), ("迭代公式",))))
        if re.search(r"x_0|初值", text):
            requirements.append(Requirement("first_iteration", (("x1",), ("第一次迭代",))))
    if re.search(r"逐点.*极限|pointwise", text, re.IGNORECASE):
        requirements.append(Requirement("pointwise_limit", (("逐点",), ("pointwise",))))
    if re.search(r"积分.*(?:极限|恒|比较)|integral.*limit", text, re.IGNORECASE):
        requirements.append(Requirement("integral_result", (("积分",), ("integral",), ("∫",))))
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
    return tuple(requirements)


def _tool_can_answer_whole(text: str, profile: ProblemProfile, goals: list[Goal]) -> bool:
    """Symbolic calculations may bypass the model only when the result is the whole ask."""
    if not profile.tool_eligible or len(goals) != 1:
        return False
    # These methods ask for a process, iteration, approximation, or interpretation;
    # a symbolic result is evidence for a substep, never the complete response.
    excluded = r"牛顿法|二分法|欧拉法|迭代|插值|近似|误差|收敛|条件数|说明|证明|比较|验证|并|且|newton|bisection|euler|iteration|approx"
    return not re.search(excluded, text, re.IGNORECASE)


def _constraints(text: str) -> list[str]:
    markers = r"实数|整数|非负|正数|可逆|连续|紧致|独立|互斥|初值|all|every|unique|integer|real|continuous|compact|independent"
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
    return risks
