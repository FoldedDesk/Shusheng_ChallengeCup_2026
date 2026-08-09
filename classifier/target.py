"""Conservative extraction of the clause that states the requested result."""

from __future__ import annotations

import re


_TARGET_COMMAND = re.compile(
    r"证明|求证|论证|推导|解释|说明|判断|求|计算|写出|给出|列出|构造|"
    r"多少|哪些|哪个|是否|定义(?:为|是)|"
    r"\b(?:prove|show|derive|explain|justify|find|determine|compute|calculate|evaluate|solve|"
    r"construct|classify|state|identify|describe|give|write|list|how\s+many|what\s+(?:is|are)|"
    r"which|for\s+which|whether|is\s+it\s+possible)\b",
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
        r"\n+|(?<=[。？！?!])\s+|(?<=\.)\$\s+(?=[A-Z\u4e00-\u9fff])|"
        r"(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])",
        protected,
    ) if segment.strip()]
    segments = [segment.replace(enumeration_marker, ".") for segment in segments]
    candidate_indices = [
        index for index, segment in enumerate(segments)
        if _TARGET_COMMAND.search(segment)
    ]
    if candidate_indices:
        candidate_index = candidate_indices[-1]
        candidate = segments[candidate_index].strip(" \t\r\n。.!?？")
        if re.search(
            r"(?:satisfying|such\s+that|subject\s+to|given\s+by|满足|使得|条件为)\s*[:：]?$",
            candidate,
            re.IGNORECASE,
        ):
            for tail in segments[candidate_index + 1:]:
                if _TARGET_COMMAND.search(tail) or not _looks_like_math_condition(tail):
                    break
                candidate += "\n" + tail
        matches = list(_TARGET_COMMAND.finditer(candidate))
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
            start = matches[0].start() if joined_commands else matches[-1].start()
            return candidate[start:].strip(" \t\r\n。.!?？")
        return candidate
    return segments[-1].strip(" \t\r\n。.!?？") if segments else text


def _looks_like_math_condition(value: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 2400:
        return False
    if re.search(r"(?:\\\[|\\begin\{|\$\$)", text):
        return True
    return bool(
        re.search(r"[=<>\u2264\u2265]", text)
        and re.search(r"[A-Za-z0-9]", text)
        and not re.search(r"[.!?]", text)
    )
