"""Conservative extraction of the clause that states the requested result."""

from __future__ import annotations

import re


_TARGET_COMMAND = re.compile(
    r"证明|求证|论证|推导|解释|说明|判断|求|计算|化简|写出|给出|列出|构造|"
    r"多少|哪些|哪个|是否|定义(?:为|是)|"
    r"\b(?:prove|show|derive|explain|justify|find|determine|compute|calculate|evaluate|solve|simplify|"
    r"construct|classify|state|identify|describe|give|write|list|how\s+many|what\s+(?:is|are)|"
    r"which|for\s+which|whether|is\s+it\s+possible)\b",
    re.IGNORECASE,
)


# Relative question words such as ``for which`` describe the object of a
# leading imperative; they are not a later request that should replace it.
_PRIMARY_TARGET_COMMAND = re.compile(
    r"证明|求证|论证|推导|解释|说明|判断|求|计算|化简|写出|给出|列出|构造|"
    r"\b(?:prove|show|derive|explain|justify|find|determine|compute|calculate|evaluate|solve|simplify|"
    r"construct|classify|state|identify|describe|give|write|list)\b",
    re.IGNORECASE,
)

_SUPPORT_ONLY_SEGMENT = re.compile(
    r"^\s*(?:(?:完整|严格|严谨)(?:证明|论证|推导|计算|归一化)|"
    r"(?:证明|论证|推导|计算|归一化))(?:须|必须|应当|要求)|"
    r"^\s*(?:a|the)?\s*(?:complete|rigorous)\s+"
    r"(?:proof|derivation|argument|calculation|normalization)|"
    r"^\s*(?:the\s+)?(?:proof|derivation|argument|calculation|normalization)\s+"
    r"(?:must|should|is required)|"
    r"^\s*(?:要求|必须|须|需|应当)(?:严格|分别|逐项|完整地?)?"
    r"(?:使用|用|从|通过|由|以|作|解|识别|构造|给出|推导|验证|核对|检查|说明|证明|计算|归一化)",
    re.IGNORECASE,
)


def extract_target_clause(problem: str) -> str:
    """Return the last sentence-like segment containing an explicit ask."""
    text = str(problem or "").strip()
    if not text:
        return ""
    enumeration_marker = "<ENUMERATION_DOT>"
    protected = re.sub(
        r"((?:^|[：:\n])\s*\d+)\.(?=\s*\S)",
        rf"\1{enumeration_marker}",
        text,
        flags=re.MULTILINE,
    )
    segments = [segment.strip() for segment in re.split(
        r"\n+|(?<=[。？！])\s*|(?<=[!?])\s+|(?<=\.)\$\s+(?=[A-Z\u4e00-\u9fff])|"
        r"(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])",
        protected,
    ) if segment.strip()]
    segments = [segment.replace(enumeration_marker, ".") for segment in segments]
    all_candidate_indices = [
        index for index, segment in enumerate(segments)
        if _TARGET_COMMAND.search(segment)
    ]
    candidate_indices = [
        index for index in all_candidate_indices
        if not _SUPPORT_ONLY_SEGMENT.search(segments[index])
    ] or all_candidate_indices
    if candidate_indices:
        candidate_index = candidate_indices[-1]
        candidate = segments[candidate_index].strip(" \t\r\n。.!?？")
        if re.search(
            r"(?:satisfying|satisf(?:y|ies)\s+the\s+following\s+condition|such\s+that|"
            r"subject\s+to|given\s+by|满足(?:以下|下列)?条件|使得|条件为)\s*[:：]?$",
            candidate,
            re.IGNORECASE,
        ):
            for tail in segments[candidate_index + 1:]:
                if re.search(
                    r"remember\s+to\s+put|final\s+answer|请.*(?:答案|作答)|"
                    r"最终答案|答案写入|答案放入",
                    tail,
                    re.IGNORECASE,
                ):
                    break
                if _TARGET_COMMAND.search(tail):
                    break
                candidate += "\n" + tail
        matches = list(_TARGET_COMMAND.finditer(candidate))
        primary_matches = list(_PRIMARY_TARGET_COMMAND.finditer(candidate))
        joined_commands = (
            len(matches) >= 2
            and bool(re.search(
                r"\b(?:or|and)\b|或者|或|并且|并",
                candidate[matches[0].end():matches[-1].start()],
                re.IGNORECASE,
            ))
        )
        # Preserve a short numbering/object prefix. A long prefix is setup or
        # definition prose; starting at its final command prevents setup nouns
        # from becoming answer requirements.
        if matches and matches[0].start() > 40:
            if primary_matches:
                # Keep the final genuine imperative, but never truncate a
                # request at an embedded ``which``/``for which`` clause.
                start = primary_matches[0].start() if joined_commands else primary_matches[-1].start()
            else:
                start = matches[0].start() if joined_commands else matches[-1].start()
            return candidate[start:].strip(" \t\r\n。.!?？")
        return candidate
    return segments[-1].strip(" \t\r\n。.!?？") if segments else text
