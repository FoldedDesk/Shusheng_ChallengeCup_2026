"""Classify where a bounded mathematical response was interrupted."""

from __future__ import annotations

import re


_ACTIVE_SEARCH = re.compile(
    r"\b(?:alternative approach|let(?:'s| us) try|try another|"
    r"seems complicated|maybe|wait|we need (?:a different|to find|to avoid)|"
    r"construction fails?|contradiction again|start over)\b|"
    r"换一种|另一种方法|重新|还需要|需要找到|构造失败|再次矛盾|似乎复杂",
    re.IGNORECASE,
)
_FINAL_BOUNDARY = re.compile(
    r"(?im)^\s*(?:FINAL(?:\s+ANSWER)?|最终答案|答案)\s*[:：]"
)
_KEY_LEMMA_COMPLETE = re.compile(
    r"\b(?:key lemma (?:is )?proved|we have proved|this proves|"
    r"which proves|therefore the required|hence the required|thus the required)\b|"
    r"关键引理(?:得证|已证)|已经证明|由此即得|这就证明|命题得证",
    re.IGNORECASE,
)
_METHOD_COMMITTED = re.compile(
    r"\b(?:we (?:use|apply|proceed by)|using|by)\s+"
    r"(?:induction|contradiction|inclusion-exclusion|generating function|"
    r"dynamic programming|residue theorem|fourier transform|energy method|"
    r"lagrange multipliers?|cauchy-schwarz|holder|jensen|compactness)\b|"
    r"采用|使用|应用|利用|归纳法|反证法|容斥原理|生成函数|动态规划|"
    r"留数定理|傅里叶变换|能量法|拉格朗日乘子",
    re.IGNORECASE,
)
_RESTATEMENT = re.compile(
    r"\bthe problem asks\b|\blet(?:'s| us) break down\b|"
    r"题目要求|先来分解题目",
    re.IGNORECASE,
)


def classify_truncated_output(value: str) -> dict[str, str]:
    """Return a content-only recovery diagnostic without retaining the text."""
    text = str(value or "").strip()
    if not text:
        return {
            "phase": "UNKNOWN",
            "recoverability": "UNKNOWN",
            "evidence": "empty_response",
        }

    head = text[:1200]
    tail = text[-2400:]
    if _FINAL_BOUNDARY.search(tail):
        return {
            "phase": "BEFORE_FINAL",
            "recoverability": "HIGH",
            "evidence": "explicit_final_boundary_in_tail",
        }
    if _ACTIVE_SEARCH.search(tail):
        return {
            "phase": "DURING_METHOD_SEARCH",
            "recoverability": "LOW",
            "evidence": "unresolved_search_in_tail",
        }
    if _KEY_LEMMA_COMPLETE.search(tail):
        return {
            "phase": "AFTER_KEY_LEMMA",
            "recoverability": "HIGH",
            "evidence": "completed_key_implication_in_tail",
        }
    if _METHOD_COMMITTED.search(text):
        return {
            "phase": "METHOD_FOUND",
            "recoverability": "MEDIUM",
            "evidence": "committed_method_without_completed_key_lemma",
        }
    if _RESTATEMENT.search(head):
        return {
            "phase": "BEFORE_METHOD",
            "recoverability": "LOW",
            "evidence": "restatement_without_committed_method",
        }
    return {
        "phase": "UNKNOWN",
        "recoverability": "UNKNOWN",
        "evidence": "no_reliable_progress_marker",
    }
