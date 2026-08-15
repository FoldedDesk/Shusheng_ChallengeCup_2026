from __future__ import annotations

import re

from classifier.choice import has_choice_options


_RESULT_COMMAND = re.compile(
    r"(?:^|[。！？!?；;：:，,]\s*|\n)\s*(?:请|试)?\s*"
    r"(?:求(?!证)|计算|求解|解(?:出|方程|不等式)?|判断|验证|比较|构造|列出|写出|给出|指出|确定|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|verify|compare|construct|classify|list|"
    r"what\s+(?:is|are)|how\s+many)\b)",
    re.IGNORECASE,
)

_CONTEXTUAL_RESULT_COMMAND = re.compile(
    r"判断|验证|比较|化简|列出|写出|确定|求(?:决定|判定)系数|"
    r"(?:用|采用|利用|根据|按照?|借助)[^。！？!?\n]{0,160}(?:求(?!证)|计算|确定)|"
    r"说明[^。！？!?\n]{0,100}是否|"
    r"\b(?:find|determine|solve|calculate|compute|evaluate|verify|compare|simplify|classify|list|"
    r"what\s+(?:is|are)|how\s+many)\b",
    re.IGNORECASE,
)


_PRIMARY_PROOF_COMMAND = re.compile(
    r"(?:^|[。！？!?；;：:，,]\s*|\n)\s*(?:请|试)?\s*"
    r"(?:证明|求证|论证|试证|"
    r"\b(?:prove|show(?:\s+that)?|demonstrate(?:\s+that)?|establish(?:\s+that)?)\b)",
    re.IGNORECASE,
)


_PRIMARY_EXPLANATION_COMMAND = re.compile(
    r"(?:^|[。！？!?；;：:，,]\s*|\n)\s*(?:请)?\s*"
    r"(?:解释|说明)[^。！？!?\n]{0,100}(?:为什么|为何)|"
    r"(?:^|[.!?;:,]\s*|\n)\s*(?:please\s+)?explain\s+why\b",
    re.IGNORECASE,
)

_NONEXISTENCE_ALTERNATIVE = re.compile(
    r"(?:"
    r"\b(?:determine|find|compute)\b[\s\S]{0,1000}?\bor\s+prove\b[\s\S]{0,500}?"
    r"(?:does\s+not|need\s+not|no\s+such|not\s+necessarily\s+exist)|"
    r"(?:求|确定)[^。！？!?\n]{0,500}(?:或|否则)[^。！？!?\n]{0,80}"
    r"证明[^。！？!?\n]{0,250}(?:不存在|不一定存在|无此|没有这样的)"
    r")",
    re.IGNORECASE,
)

_RESULT_WITH_NAMED_PROOF = re.compile(
    r"(?:"
    r"\b(?:determine|find)\b[^.!?\n]{0,500}"
    r"(?:\bwith\s+(?:a\s+)?(?:complete\s+|rigorous\s+)?proof\b|"
    r"(?:,|;|\band\b)\s*(?:give|provide|write)\s+"
    r"(?:a\s+|the\s+)?(?:complete\s+|rigorous\s+)?proof\b)|"
    r"\bwith\s+(?:a\s+)?(?:complete\s+|rigorous\s+)?proof\b"
    r"[^.!?\n]{0,100}\b(?:determine|find)\b|"
    r"\b(?:determine|find)\b[^.!?\n]{0,500}[.!;]\s*"
    r"(?:the\s+)?(?:proof|argument|justification)\s+"
    r"(?:must|should|shall|is\s+required\s+to)\b|"
    r"(?:求|确定)[^。！？!?\n]{0,500}(?:并|且|，|,|；|;)\s*"
    r"(?:给出|写出)\s*(?:一个|(?:完整|严格)的?)?\s*(?:证明|论证)|"
    r"(?:求|确定)[^。！？!?\n]{0,500}[。；;]\s*"
    r"(?:并?要求|且要求|要求|须|需)\s*(?:给出|写出)?"
    r"[^。！？!?\n]{0,40}(?:证明|论证)|"
    r"(?:求|确定)[^。！？!?\n]{0,500}[。；;]\s*"
    r"(?:证明|论证)\s*(?:须|需|必须|应当?|要求)"
    r")",
    re.IGNORECASE,
)

_RESULT_AND_PROVE = re.compile(
    r"(?:"
    r"\b(?:determine|find)\b[^.!?\n]{0,500}(?:,|;|\band\b)\s*prove\b|"
    r"(?:求|确定)[^。！？!?\n]{0,500}(?:并|且|，|,|；|;)\s*(?:请)?\s*(?:证明|论证)"
    r")",
    re.IGNORECASE,
)

_EXHAUSTIVENESS_PROOF_SUPPORT = re.compile(
    r"(?:"
    r"\bprove\s+(?:that\s+)?(?:"
    r"there\s+(?:is|are)\s+no\s+(?:other|others|further|additional)\b|"
    r"no\s+(?:other|further|additional)\b[^.!?\n]{0,80}\bexist(?:s)?\b"
    r")|"
    r"(?:证明|论证)\s*(?:"
    r"(?:没有|再无|无)\s*(?:其他|别的|其余)?|"
    r"不存在\s*(?:其他|别的|其余)"
    r")"
    r")",
    re.IGNORECASE,
)


def _explicit_result_proof_required(text: str) -> bool:
    """Return whether a result request makes proof part of the primary task."""
    if _RESULT_WITH_NAMED_PROOF.search(text):
        return True
    return bool(
        _RESULT_AND_PROVE.search(text)
        and not _EXHAUSTIVENESS_PROOF_SUPPORT.search(text)
    )


def _primary_command_precedes_result(text: str, command: re.Pattern[str]) -> bool:
    """Return whether a proof/explanation ask precedes any result-only ask."""
    primary = command.search(text)
    if not primary:
        return False
    result_positions = [
        match.start()
        for pattern in (_RESULT_COMMAND, _CONTEXTUAL_RESULT_COMMAND)
        for match in pattern.finditer(text)
    ]
    return not result_positions or primary.start() < min(result_positions)


def classify_problem_type(problem: str) -> str:
    """Classify the primary task, keeping support requests subordinate.

    For example, ``计算……并说明理由`` and ``compute ... and justify`` are
    calculations. A proof or explanation type is reserved for prompts whose
    primary request is the proof or explanation itself.
    """
    text = str(problem or "")
    if has_choice_options(text) or re.search(
        r"选择题|选项|multiple choice|choose (?:the )?(?:correct|best)",
        text,
        re.IGNORECASE,
    ):
        return "choice"
    if re.search(r"填空|填入|fill (?:in|the blank)", text, re.IGNORECASE):
        return "fill_blank"
    if re.search(
        r"^\s*(?:请|试)?\s*(?:构造|给出[^。！？!?\n]{0,20}构造)|"
        r"^\s*(?:please\s+)?construct\b|"
        r"^\s*(?:please\s+)?(?:give|provide)\b[^.!?\n]{0,30}"
        r"\b(?:construction|example)\b",
        text,
        re.IGNORECASE,
    ):
        return "construction"
    if _NONEXISTENCE_ALTERNATIVE.search(text):
        return "calculation"
    if _explicit_result_proof_required(text):
        return "proof"
    # A leading proof remains the primary task when a trailing clause merely
    # asks the solver to state a class, identity, or other supporting result.
    # Position order keeps "find ... and prove there are no others" as a
    # result task through the dedicated exhaustiveness rule above.
    if _primary_command_precedes_result(text, _PRIMARY_PROOF_COMMAND):
        return "proof"
    if re.search(
        r"^\s*(?:请)?\s*(?:给出|写出)\s*(?:一个|完整的|严格的)?\s*(?:证明|论证)|"
        r"^\s*(?:请)?\s*(?:给出|写出)[^。！？!?\n]{1,100}(?:的证明|的论证)|"
        r"^\s*(?:please\s+)?(?:give|provide|write)\s+(?:a\s+|the\s+)?proof\b",
        text,
        re.IGNORECASE,
    ):
        return "proof"
    if re.search(
        r"^\s*(?:请)?\s*(?:解释|说明)\s*(?:为什么|为何)|"
        r"^\s*(?:请)?\s*给出\s*(?:一个|完整的)?\s*(?:解释|说明)|"
        r"^\s*(?:please\s+)?explain\s+why\b",
        text,
        re.IGNORECASE,
    ):
        return "explanation"
    if _primary_command_precedes_result(text, _PRIMARY_EXPLANATION_COMMAND):
        return "explanation"
    if _RESULT_COMMAND.search(text) or _CONTEXTUAL_RESULT_COMMAND.search(text):
        return "calculation"
    if re.search(r"证明|求证|论证|\b(?:prove|proof|show\s+that)\b", text, re.IGNORECASE):
        return "proof"
    if re.search(r"推导|\b(?:derive|derivation)\b", text, re.IGNORECASE):
        return "derivation"
    if re.search(r"解释|说明|\b(?:explain|justify)\b", text, re.IGNORECASE):
        return "explanation"
    return "calculation"


def classify_task_kind(problem: str) -> str:
    """Explicit name for the primary-task classifier."""
    return classify_problem_type(problem)
