from __future__ import annotations

from dataclasses import dataclass
import re

from tools.latex_parser import find_matching_brace, normalize_latex


@dataclass(frozen=True)
class ExtractionResult:
    answer: str
    method: str
    valid: bool
    rejected_reasons: tuple[str, ...] = ()
    raw_has_meta: bool = False
    explicit_answer: bool = False


class Finalizer:
    """Extract explicit answer candidates without silently repairing malformed text."""

    _LABEL = re.compile(
        r"(?im)^\s*(?:(?:【\s*(?:最终答案|答案|结论)\s*】|(?:最终\s*)?答案|结论)\s*[:：]?|(?:final\s+answer|answer|conclusion)\s*[:：])\s*([^\n]+?)\s*$"
    )
    _BRACKET_LABEL = re.compile(
        r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：]?\s*([^`\n]+)",
        re.IGNORECASE,
    )
    _PLACEHOLDER = re.compile(
        r"^(?:(?:最终答案|完整答案|完整结论|答案|final(?:\s+answer)?|answer|"
        r"check\s+format(?:ting)?|format(?:ting)?|(?:final\s+)?(?:conclusion|response|done))"
        r"\s*[.。!?！:：…]*|[.。…`'\"，,!?！:：]+)$",
        re.IGNORECASE,
    )
    _UNFINISHED_PLACEHOLDER = re.compile(
        r"\bthis\s+is\s+(?:only\s+)?(?:a\s+)?placeholder\b|"
        r"\b(?:actual|full|remaining)\s+(?:calculation|computation|derivation|proof)\b"
        r"[^.!?\n]{0,48}\b(?:requires?|needs?)\s+(?:more|further|additional)\s+"
        r"(?:steps?|work|calculation|computation|derivation)\b|"
        r"\b(?:calculation|computation|derivation|proof)\s+(?:is\s+)?"
        r"(?:not\s+yet|still\s+needs?\s+to\s+be)\s+(?:completed|finished|done)\b|"
        r"占位符|(?:实际|完整|剩余)(?:计算|推导|证明)(?:仍|还)?(?:需要|需)"
        r"(?:更多|进一步|额外)?(?:步骤|工作|计算|推导)?|"
        r"(?:计算|推导|证明)(?:尚未|还未|仍未)(?:完成|做完)",
        re.IGNORECASE,
    )
    _CONCLUSION_AFTER_GAP = re.compile(
        r"(?:^|[.!?。！？]\s*)(?:FINAL(?:\s+ANSWER)?|最终答案|最终结论|"
        r"(?:corrected|revised)\s+(?:final\s+)?answer|(?:更正|修正)后(?:的)?(?:最终)?答案)"
        r"\s*[:：=]",
        re.IGNORECASE,
    )
    _META = re.compile(
        r"(?:<think\b|thinking process|(?im:^\s*(?:analysis|drafting)\s*[:：])|"
        r"(?im:^\s*(?:\*{1,3}|_{1,3})?\s*(?:output\s+)?language\s*[:：])|"
        r"(?im:^\s*(?:\*{1,3}|_{1,3})?\s*drafting(?:\s+the)?\s+final(?:\s+(?:answer|line))?\s*[:：])|"
        r"check formatting|check spacing|"
        r"system prompt|prompt instruction|final answer should|最后一行必须|思考过程|分析过程|推理过程|"
        r"格式检查|检查格式|提示词|(?im:^\s*plan\s*[:：])|\bi\s+(?:plan|intend)\s+to\b|(?im:^\s*(?:structure|count\s+lines?|draft(?:\s+\d+)?)\s*:)|(?:final answer )?content for (?:the )?first line|final answer content|\bi (?:need|will|should)\b|\bthe (?:user|instruction|prompt)\b|"
        r"\b(?:check\s+(?:the\s+)?line\s+count|line\s+count)\b|(?im:^\s*line\s+\d+\s*:)|"
        r"\b(?:looks? compliant|looks? solid|is there any risk|so it is fine)\b|"
        r"(?im:^\s*g\d+\s*[:：])|[（(]\s*g\d+\s*[)）]|\bg\d+\s*\[(?:proof|formula|scalar|truth|construction)|"
        r"\b(?:integral_result|integral_value|pointwise_limit|exact_comparison|first_iteration|iteration_formula)\b|"
        r"\b(?:final answer )?expects? (?:just|only)\b|\bi['’]ll\s+(?:write|provide|output|include)\b|"
        r"\boptional\s+(?:brief\s+)?note\b|"
        r"\[\s*(?:explanation|proof|reasoning|derivation|answer|content)(?:\s+text)?\s*\]|"
        r"\[(?:note|method|check)\.[^\]\n]+\]|必查字段|"
        r"\blet(?:'s| us)\s+(?:assemble|compose)\s+(?:the\s+)?(?:final\s+)?(?:text|answer|response)\b|"
        r"(?im:^\s*first\s+line\s*[:：])|\bthen\s+(?:the\s+)?(?:proof|argument|explanation)\b|"
        r"\bi\s+(?:will|'ll)\s+formulate\b|"
        r"让我(?:验证|确认|组织)|我(?:需要|应该)|输出时|"
        r"但再核对一次|或许题目(?:中|的)|也许题目(?:中|的)|重新审视题目|"
        r"我(?:再|来)?核对|\blet me (?:recheck|check again)\b|\bperhaps the problem\b|"
        r"(?:现在|下面)?(?:组装|组织)(?:最终)?(?:文本|答案|回复)|第一行\s*[:：]|然后写(?:证明|论证|解释))",
        re.IGNORECASE,
    )
    _SELF_RETRACTION = re.compile(
        r"\b(?:my|our)\s+(?:calculation|computation|answer|result|derivation|reasoning)\s+"
        r"(?:is|was|may\s+be|might\s+be)\s+(?:wrong|incorrect|not\s+correct|off)\b|"
        r"\b(?:my|our|the)\s+interpretation\b[^.!?\n]{0,40}\b(?:is|was|seems?)\s+off\b|"
        r"\b(?:this|that|it)\s+needs?\s+(?:a\s+)?correction\b|"
        r"\b(?:this|that|the)\s+(?:check|calculation|computation|argument|proof|derivation)\s+"
        r"needs?\s+(?:a\s+)?correction\b|"
        r"\b(?:this|that|the)\b[^.!?\n]{0,50}\bneeds?\s+(?:a\s+)?correction\b|"
        r"\bi\s+(?:may|might)\s+have\b|"
        r"\b(?:which|what)\s+(?:answer|result|value)\s+is\s+correct\b|"
        r"此处(?:仍)?需(?:要)?修正|这里(?:仍)?需(?:要)?修正|需要修正|"
        r"(?:我的|上述|前述)?(?:计算|推导|答案|结果|结论)(?:可能)?(?:有误|不对)|"
        r"(?:这个|这里|此处)(?:结论|结果|答案|推导)不对|我(?:可能|也许)(?:算错|漏掉|忽略|有误)|"
        r"(?:究竟|到底)(?:哪个|哪一个)?(?:答案|结果|数值)(?:才)?(?:正确|对)",
        re.IGNORECASE,
    )
    _AMBIGUOUS_SELF_RETRACTION = re.compile(
        r"\b(?:this|that|it)\s+(?:is|was)\s+(?:incorrect|not\s+correct|wrong)\b|"
        r"\b(?:this|that|my|our|the)\s+(?:answer|result|value|conclusion)\s+"
        r"(?:is|was)\s+(?:bad|invalid)\b|"
        r"\b[A-Za-z][A-Za-z0-9_]*\s*\$?\s+itself\s+(?:is|was)\s+"
        r"(?:bad|invalid)\b|"
        r"(?:这个|这里|此处)不对",
        re.IGNORECASE,
    )
    _DISCARDED_BRANCH_CONTEXT = re.compile(
        r"\b(?:if|when|assuming|suppose|discarded|rejected|alternative|branch|case|"
        r"candidate|root)\b|若|假设|排除|舍去|另一分支|该分支|此分支|候选根|这种情形",
        re.IGNORECASE,
    )
    _CORRECTION_RESOLUTION = re.compile(
        r"(?im)(?:^\s*(?:FINAL(?:\s+ANSWER)?|【\s*最终答案\s*】|最终答案)\s*[:：=]|"
        r"\bcorrected\s+(?:final\s+)?answer\s*[:：=]|(?:更正|修正)后(?:的)?(?:最终)?答案\s*[:：为=])"
        r"[ \t]*(?:\\boxed\{[^\n]+\}|[^\s\n][^\n]*)"
    )

    @staticmethod
    def extract(candidate: str) -> str:
        result = Finalizer.extract_result(candidate)
        return result.answer if result.valid else ""

    @staticmethod
    def extract_result(candidate: str) -> ExtractionResult:
        text = str(candidate or "").strip()
        if not text:
            return ExtractionResult("", "empty", False, ("empty",))
        text = re.sub(r"<\|(?:assistant|user|system|endoftext)\|>", "", text, flags=re.IGNORECASE).strip()
        raw_has_meta = Finalizer.contains_meta(text)

        explicit_results = Finalizer.extract_explicit_results(text, raw_has_meta=raw_has_meta)
        valid_explicit = [result for result in explicit_results if result.valid]
        if valid_explicit:
            labelled = [
                result for result in valid_explicit
                if result.method in {
                    "label", "label_boxed", "bracket_label",
                    "label_next_line", "label_next_line_boxed",
                }
            ]
            # The public prompt deliberately places FINAL before supporting
            # checks. A later unlabelled box is commonly an intermediate value.
            return labelled[-1] if labelled else valid_explicit[-1]
        if explicit_results:
            # Preserve the most relevant structural error for diagnostics only
            # after every explicit fragment has been attempted.
            explicit_failure = explicit_results[-1]
        else:
            explicit_failure = None
        if raw_has_meta:
            recovered = Finalizer._recover_tail_conclusion(text)
            if recovered:
                return Finalizer._result(recovered, "tail_segment", raw_has_meta=True)
            if explicit_failure is not None:
                return explicit_failure
            return ExtractionResult("", "meta_without_explicit_answer", False, ("meta_without_explicit_answer",), True, False)
        if explicit_failure is not None:
            return explicit_failure
        return Finalizer._result(text, "whole_response")

    @staticmethod
    def extract_explicit_results(candidate: str, *, raw_has_meta: bool | None = None) -> tuple[ExtractionResult, ...]:
        """Extract every labelled or boxed answer fragment in source order."""
        text = str(candidate or "")
        has_meta = Finalizer.contains_meta(text) if raw_has_meta is None else raw_has_meta
        fragments: list[tuple[int, ExtractionResult]] = []
        label_pattern = re.compile(
            r"(?im)^\s*(?:`{1,3})?\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)\s*(?:is|equals|[:：=])|"
            r"(?:最终\s*)?答案\s*[:：为=]|结论\s*[:：为=])"
            r"\s*(?:\*{1,3}|_{1,3})?\s*([^\n]+)"
        )
        for match in label_pattern.finditer(text):
            value = re.sub(r"\s*(?:\*{1,3}|_{1,3})\s*$", "", match.group(1)).strip()
            value = Finalizer._trim_explicit_line(value)
            boxed = Finalizer._last_boxed(value)
            method = "label_boxed" if boxed is not None else "label"
            value = boxed if boxed is not None else value
            fragments.append((
                match.start(),
                Finalizer._result(value, method, raw_has_meta=has_meta, explicit=True),
            ))
        next_line_pattern = re.compile(
            r"(?im)^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)|"
            r"(?:最终\s*)?答案|结论)\s*[:：为=]?\s*"
            r"(?:\*{1,3}|_{1,3})?\s*$\n[ \t]*([^\n]+)"
        )
        for match in next_line_pattern.finditer(text):
            value = Finalizer._trim_explicit_line(match.group(1).strip())
            boxed = Finalizer._last_boxed(value)
            method = "label_next_line_boxed" if boxed is not None else "label_next_line"
            value = boxed if boxed is not None else value
            fragments.append((
                match.start(),
                Finalizer._result(value, method, raw_has_meta=has_meta, explicit=True),
            ))
        for match in Finalizer._BRACKET_LABEL.finditer(text):
            value = Finalizer._trim_explicit_line(match.group(1).strip().strip("` "))
            boxed = Finalizer._last_boxed(value)
            value = boxed if boxed is not None else value
            fragments.append((
                match.start(),
                Finalizer._result(value, "bracket_label", raw_has_meta=has_meta, explicit=True),
            ))
        for position, boxed, complete in Finalizer._boxed_values(text):
            value = boxed if complete else text[position:]
            method = "boxed" if complete else "boxed_unclosed"
            fragments.append((
                position,
                Finalizer._result(value, method, raw_has_meta=has_meta, explicit=True),
            ))

        # The same boxed answer can also be captured through its line label.
        unique: list[ExtractionResult] = []
        seen: set[tuple[str, str, bool]] = set()
        for _, result in sorted(fragments, key=lambda item: item[0]):
            key = (result.answer, result.method, result.valid)
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        return tuple(unique)

    @staticmethod
    def contains_meta(value: str) -> bool:
        return bool(Finalizer._META.search(str(value or "")))

    @staticmethod
    def _trim_explicit_line(value: str) -> str:
        """Keep a labelled conclusion before an inline proof or self-check marker."""
        text = str(value or "").strip()
        if text.endswith("`") and "`" not in text[:-1]:
            text = text[:-1].rstrip()
        marker = re.search(
            r"\s+(?:\*{0,2})?(?:证明|论证|推导|论证过程|推导过程|严格论证|"
            r"但再核对一次|不过再核对一次|"
            r"proof|derivation|justification|verification)\s*[:：]",
            text,
            re.IGNORECASE,
        )
        if marker and text[:marker.start()].strip():
            return text[:marker.start()].strip().rstrip("。.;；")
        return text

    @staticmethod
    def has_unresolved_self_retraction(value: str) -> bool:
        """Detect a candidate that disowns its result without a later final correction."""
        text = str(value or "")
        matches = list(Finalizer._SELF_RETRACTION.finditer(text))
        for match in Finalizer._AMBIGUOUS_SELF_RETRACTION.finditer(text):
            sentence_start = max(
                text.rfind(marker, 0, match.start())
                for marker in (".", "!", "?", "。", "！", "？", "\n")
            ) + 1
            sentence_ends = [
                position
                for marker in (".", "!", "?", "。", "！", "？", "\n")
                if (position := text.find(marker, match.end())) >= 0
            ]
            sentence_end = min(sentence_ends) if sentence_ends else len(text)
            sentence = text[sentence_start:sentence_end]
            if not Finalizer._DISCARDED_BRANCH_CONTEXT.search(sentence):
                matches.append(match)
        if not matches:
            return False
        last_match = max(matches, key=lambda item: item.start())
        return not Finalizer._CORRECTION_RESOLUTION.search(text, last_match.end())

    @staticmethod
    def extract_solution(candidate: str) -> str:
        """Keep a proof's reasoning while applying only display cleanup."""
        return Finalizer._clean(str(candidate or "").strip())

    @staticmethod
    def extract_tagged_submission(candidate: str) -> str:
        """Recover a final tagged answer and its proof body, excluding any preamble."""
        blocks = Finalizer.extract_tagged_submissions(candidate)
        return blocks[-1] if blocks else ""

    @staticmethod
    def extract_tagged_submissions(candidate: str) -> tuple[str, ...]:
        """Return every structurally complete tagged block in source order."""
        text = str(candidate or "").strip()
        matches = list(re.finditer(
            r"(?im)^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)\s*(?:is|equals|[:：=])|"
            r"(?:最终\s*)?答案\s*[:：为=]|结论\s*[:：为=])"
            r"[ \t]*(?:\*{1,3}|_{1,3})?[ \t]*[^\n]*$",
            text,
        ))
        blocks = []
        for match in matches:
            lines = [match.group(0).strip()]
            remainder = text[match.end():].splitlines()
            for line in remainder:
                if Finalizer._proof_meta_boundary(line) or re.match(
                    r"^\s*(?:【\s*(?:最终答案|答案|结论|校验)\s*】\s*[:：]?|"
                    r"(?:(?:最终\s*)?答案|FINAL(?:\s+ANSWER)?|ANSWER|CONCLUSION)\s*[:：])",
                    line,
                    re.IGNORECASE,
                ):
                    break
                lines.append(line)
            cleaned = Finalizer._clean("\n".join(lines).strip())
            if Finalizer.contains_meta(cleaned):
                continue
            if Finalizer.validate_structure(cleaned):
                cleaned = Finalizer._trim_incomplete_suffix(cleaned)
            if cleaned and not Finalizer.validate_structure(cleaned):
                blocks.append(cleaned)
        return tuple(blocks)

    @staticmethod
    def extract_terminal_supported_submissions(candidate: str) -> tuple[str, ...]:
        """Recover a clean proof suffix when its conclusion is written last.

        Some recovery responses emit an internal-analysis preamble, then a
        self-contained ``Proof: ...`` block, and only then the final labelled
        conclusion. The ordinary tagged extractor correctly isolates the
        conclusion but cannot include support that precedes it. This narrow
        suffix parser starts only at an explicit proof/derivation marker and
        still rejects meta text or malformed structure.
        """
        lines = str(candidate or "").strip().splitlines()
        if len(lines) < 3:
            return ()
        conclusion = re.compile(
            r"^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)"
            r"\s*(?:is|equals|[:：=])|"
            r"(?:最终\s*)?答案\s*[:：为=]|结论\s*[:：为=])",
            re.IGNORECASE,
        )
        support = re.compile(
            r"^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"proof|argument|justification|derivation|solution|"
            r"证明|论证|推导|解答)\s*[:：]",
            re.IGNORECASE,
        )
        blocks: list[str] = []
        for conclusion_index in reversed(range(len(lines))):
            if not conclusion.match(lines[conclusion_index]):
                continue
            support_index = next(
                (
                    index
                    for index in reversed(range(conclusion_index))
                    if support.match(lines[index])
                ),
                None,
            )
            if support_index is None:
                continue
            block_lines = lines[support_index:conclusion_index + 1]
            if any(
                Finalizer._proof_meta_boundary(line)
                for line in block_lines[1:]
            ):
                continue
            cleaned = Finalizer._clean("\n".join(block_lines).strip())
            if (
                len(cleaned) >= 80
                and not Finalizer.contains_meta(cleaned)
                and not Finalizer.validate_structure(cleaned)
            ):
                blocks.append(cleaned)
            # Only the last labelled conclusion can be authoritative.
            break
        return tuple(blocks)

    @staticmethod
    def _trim_incomplete_suffix(value: str) -> str:
        """Drop only trailing lines whose delimiters or sentence are visibly cut off."""
        recoverable = {
            "unclosed_code_fence",
            "unclosed_inline_math",
            "unclosed_inline_latex",
            "unclosed_display_latex",
            "unclosed_latex_environment",
            "unclosed_latex_brace",
            "trailing_fragment",
            "truncated_sentence",
        }
        lines = str(value or "").splitlines()
        while len(lines) > 1:
            candidate = "\n".join(lines).strip()
            reasons = set(Finalizer.validate_structure(candidate))
            if not reasons:
                return candidate
            if not reasons <= recoverable:
                return ""
            lines.pop()
        return ""

    @staticmethod
    def _candidate_richness(value: str) -> tuple[int, int, int]:
        text = str(value or "").strip()
        mathematical = len(re.findall(r"[=+\-*/^\\]|\d|∈|⊆|≤|≥", text))
        sentences = len([item for item in re.split(r"[。；;\n]+", text) if item.strip()])
        return (sentences, mathematical, min(len(text), 2000))

    @staticmethod
    def _proof_meta_boundary(line: str) -> bool:
        return bool(re.match(
            r"^\s*(?:[-*]{1,3}\s*)?(?:thinking process|analysis|wait\b|okay\b|"
            r"i\s+(?:need|will|should|can|must)\b|i['’]ll\s+(?:write|provide|output|include)\b|check\b|one\s+(?:detail|more|adjustment)\b|"
            r"final\s+(?:check|plan|polish)\b|double\s+check\b|revised\s+(?:body|draft|proof)\b|plan\s*:|"
            r"count\s+lines?\s*:|check\s+(?:the\s+)?line\s+count\b|line\s+\d+\s*:|draft(?:\s+\d+)?\s*:|"
            r"(?:output\s+)?language\s*:|drafting(?:\s+the)?\s+final(?:\s+(?:answer|line))?\s*:|"
            r"refin(?:e|ing)\b|need\s+to\b|the\s+prompt\b|(?:this\s+)?looks?\s+(?:compliant|solid|concise|complete)\b|"
            r"covers?\s+all\s+(?:the\s+)?requirements\b|"
            r"is\s+there\s+any\s+risk\b|so\s+it\s+is\s+fine\b|g\d+\s*(?:\[|[:：])|必查字段|"
            r"(?:[（(]\s*)?optional\s+(?:brief\s+)?note\b|"
            r"\[\s*(?:explanation|proof|reasoning|derivation|answer|content)(?:\s+text)?\s*\]|"
            r"再检查|格式检查|思考过程|分析过程|提示词)",
            str(line or ""),
            re.IGNORECASE,
        ))

    @staticmethod
    def validate_structure(answer: str) -> tuple[str, ...]:
        value = str(answer or "").strip()
        reasons: list[str] = []
        if not value:
            return ("empty",)
        if Finalizer._PLACEHOLDER.fullmatch(value):
            reasons.append("placeholder")
        unfinished = list(Finalizer._UNFINISHED_PLACEHOLDER.finditer(value))
        if unfinished and not Finalizer._CONCLUSION_AFTER_GAP.search(
            value,
            unfinished[-1].end(),
        ):
            reasons.append("placeholder")
        if re.fullmatch(r"<\s*(?:完整答案|最终答案|答案|完整结论)\s*>", value):
            reasons.append("placeholder")
        if re.fullmatch(r"(?:final\s+)?(?:conclusion|response|done)[.。!?！]?", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.fullmatch(
            r"(?:final\s+)?(?:check|checking)(?:\s+(?:on|the|all|format(?:ting)?|constraints?|answer|result)){0,4}\s*[:：]?",
            value,
            re.IGNORECASE,
        ):
            reasons.append("placeholder")
        if re.search(r"(?:裁决|修正|补齐|重做)后的?(?:完整)?答案|(?:complete|final) answer after (?:review|correction)", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.search(r"完整答案|<\s*完整答案\s*>|\b(?:adjudicated|corrected|complete) (?:final )?answer\b", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.search(
            r"\[\s*(?:(?:explanation|proof|reasoning|derivation|answer|content|conclusion)"
            r"(?:\s+text)?(?:\s*(?:and|&|/|\+|,)\s*(?:explanation|proof|reasoning|"
            r"derivation|answer|content|conclusion)(?:\s+text)?){0,3}|"
            r"insert\s+[^\]]+|(?:解释|证明|推理|推导|答案|内容|结论)(?:文本)?"
            r"(?:\s*(?:和|与|及|、|/|\+)\s*(?:解释|证明|推理|推导|答案|内容|结论)(?:文本)?){0,3})\s*\]",
            value,
            re.IGNORECASE,
        ):
            reasons.append("placeholder")
        if re.search(r"并给出全部结论.*(?:必要依据|必要算式)|给出全部结论.*再写", value):
            reasons.append("placeholder")
        if re.search(r"(?:证明|结论|依据|推导)\s*[:：]\s*(?:\.{2,}|…+)", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.fullmatch(
            r"(?:step|stage|part|case|proof|derivation|solution|answer|"
            r"步骤\s*[一二三四五六七八九十\d]*|"
            r"第\s*[一二三四五六七八九十\d]+\s*步)"
            r"\s*(?:\d+)?\s*[:：.、-]*\s*(?:\.{2,}|…+)",
            value,
            re.IGNORECASE,
        ):
            reasons.append("placeholder")
        if re.search(
            r"(?<![,，A-Za-z0-9])(?:\.{3,}|…{2,})(?![,，A-Za-z0-9])",
            value,
        ):
            reasons.append("omitted_fragment")
        if re.search(
            r"\b(?:or similar|or equivalent|or something(?: similar)?|maybe|perhaps|probably)\b|"
            r"(?:或|及)(?:其他)?类似(?:答案|形式|结果)?|诸如此类",
            value,
            re.IGNORECASE,
        ):
            reasons.append("uncertain_fragment")
        if re.search(
            r"^\s*(?:often|usually|typically|presumably|apparently|"
            r"perhaps|maybe)\b|"
            r"^\s*(?:通常(?:是|为)?|大概|或许|也许|似乎|看起来)",
            value,
            re.IGNORECASE,
        ) or value.rstrip().endswith(("?", "？")):
            reasons.append("uncertain_fragment")
        if re.fullmatch(
            r"\s*(?:the\s+)?(?:set|family|class|collection|value|result|"
            r"answer|number|form)\s+of\s+(?:such|these|those)"
            r"(?:\s+[A-Za-z][A-Za-z'-]*){0,5}\s*[.。]?\s*",
            value,
            re.IGNORECASE,
        ) or re.fullmatch(
            r"\s*(?:(?:满足)?(?:上述|前述|这些|此类)条件的|(?:上述|前述|如上)的?)"
            r"[\u4e00-\u9fff]{0,12}(?:集合|全体|族|值|形式|结果|答案|结论)?"
            r"\s*[.。]?\s*",
            value,
        ):
            reasons.append("referential_fragment")
        if re.search(r"\b(?:this|that) (?:looks|seems) like\b|\bspecific test case\b|\blooks like noise\b", value, re.IGNORECASE):
            reasons.append("meta_text")
        if re.search(r"\bthis (?:phrasing|wording|instruction|prompt)\b", value, re.IGNORECASE):
            reasons.append("meta_text")
        if Finalizer._META.search(value):
            reasons.append("meta_text")
        if Finalizer.has_unresolved_self_retraction(value):
            reasons.append("unresolved_self_retraction")
        if not re.search(r"[\w\u4e00-\u9fff=+\-*/^\\]", value):
            reasons.append("meaningless_fragment")
        if re.fullmatch(r"[\\`'\"\s]+", value):
            reasons.append("meaningless_fragment")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            reasons.append("control_character")
        if re.search(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s+[^<>]*)?>", value):
            reasons.append("markup_fragment")
        if value.count("```") % 2:
            reasons.append("unclosed_code_fence")
        if Finalizer._unescaped_count(value, "$") % 2:
            reasons.append("unclosed_inline_math")
        if value.count(r"\(") != value.count(r"\)"):
            reasons.append("unclosed_inline_latex")
        if value.count(r"\[") != value.count(r"\]"):
            reasons.append("unclosed_display_latex")
        environments = set(re.findall(r"\\(?:begin|end)\{([^}]+)\}", value))
        for environment in environments:
            if len(re.findall(rf"\\end\{{{re.escape(environment)}\}}", value)) != len(
                re.findall(rf"\\begin\{{{re.escape(environment)}\}}", value)
            ):
                reasons.append("unclosed_latex_environment")
                break
        if not Finalizer._balanced_braces(value):
            reasons.append("unclosed_latex_brace")
        if not Finalizer._balanced_group_delimiters(value):
            reasons.append("unclosed_group_delimiter")
        if Finalizer._unescaped_count(value, '"') % 2 or value.count("“") != value.count("”"):
            reasons.append("unclosed_quote")
        if re.search(r"[,，:：;；=+*/^\\]\s*$", value):
            reasons.append("trailing_fragment")
        if re.search(
            r"(?:[,，;；。]\s*(?:对应|等于|趋于|得到|可得|说明|证明|推出)|"
            r"(?:答案|结果|值|概率|系数|极限|解|结论)\s*(?:分别)?\s*为)\s*$",
            value,
        ):
            reasons.append("truncated_sentence")
        if re.search(
            r"(?:^|[\s,，;；。:：\-–—])(?:若|如果|当|因为|由于|由|则|且|并|或|"
            r"因此|所以|故|从而|可得|"
            r"if|when|because|since|therefore|hence|then|and|or|where|with|by|"
            r"on|of|to|from|into|about|as)"
            r"\s*[:：\-–—]?\s*$",
            value,
            re.IGNORECASE,
        ):
            reasons.append("truncated_sentence")
        last_line = next((line.strip() for line in reversed(value.splitlines()) if line.strip()), "")
        if (
            last_line
            and not re.search(r"[。.!?！？；;\])}$]$", last_line)
            and (
                re.match(r"^(?:设|取|令)(?:\s|[A-Za-z\u4e00-\u9fff])", last_line)
                or re.search(r"(?:^|[，,；;。])\s*(?:取|设|令|由|因为|若|则|其中|并|且)\s*$", last_line)
            )
        ):
            reasons.append("truncated_sentence")
        return tuple(reasons)

    @staticmethod
    def _result(
        value: str,
        method: str,
        *,
        raw_has_meta: bool = False,
        explicit: bool = False,
    ) -> ExtractionResult:
        answer = Finalizer._clean(value)
        reasons = Finalizer.validate_structure(answer)
        return ExtractionResult(answer if not reasons else "", method, not reasons, reasons, raw_has_meta, explicit)

    @staticmethod
    def _clean(answer: str) -> str:
        value = answer.strip()
        fenced = re.fullmatch(
            r"```(?:latex|text|markdown)?\s*\n?(.*?)\n?\s*```",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced:
            value = fenced.group(1).strip()
        value = normalize_latex(value).strip().strip('"“”')
        # Remove balanced Markdown emphasis without touching exponent syntax
        # such as ``x**2`` or identifier underscores.
        value = re.sub(r"(?<!\*)\*\*([^*\n]+)\*\*(?!\*)", r"\1", value)
        value = re.sub(r"(?<!_)__([^_\n]+)__(?!_)", r"\1", value)
        # Models occasionally put a complete display inside ``\boxed{}``,
        # leaving ``$...$.`` as the extracted payload.  Strip only a balanced
        # wrapper around the entire answer; internal math delimiters are data.
        outer_math = re.fullmatch(r"\s*\$(.*?)\$\s*[。.]?\s*", value, re.DOTALL)
        if outer_math and len(re.findall(r"(?<!\\)\$", value)) == 2:
            value = outer_math.group(1).strip()
        return value

    @staticmethod
    def _last_boxed(text: str) -> str | None:
        values = Finalizer._boxed_values(text)
        if not values:
            return None
        _, value, complete = values[-1]
        return value if complete else ""

    @staticmethod
    def _boxed_values(text: str) -> list[tuple[int, str, bool]]:
        marker = re.compile(r"\\boxed\s*\{")
        values: list[tuple[int, str, bool]] = []
        offset = 0
        while True:
            match = marker.search(text, offset)
            if match is None:
                break
            position = match.start()
            brace = match.end() - 1
            start = brace + 1
            end = find_matching_brace(text, brace)
            if end >= 0:
                values.append((position, text[start:end].strip(), True))
                offset = end + 1
            else:
                values.append((position, text[start:].strip(), False))
                offset = len(text)
        return values

    @staticmethod
    def _balanced_braces(value: str) -> bool:
        depth = 0
        escaped = False
        for char in value:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == "{" and not escaped:
                depth += 1
            elif char == "}" and not escaped:
                depth -= 1
                if depth < 0:
                    return False
            escaped = False
        return depth == 0

    @staticmethod
    def _balanced_group_delimiters(value: str) -> bool:
        """Balance ordinary grouping while allowing half-open interval notation."""
        depth = 0
        escaped = False
        for char in value:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if not escaped and char in "([":
                depth += 1
            elif not escaped and char in ")]":
                depth -= 1
                if depth < 0:
                    return False
            escaped = False
        return depth == 0

    @staticmethod
    def _unescaped_count(value: str, marker: str) -> int:
        count = 0
        backslashes = 0
        for char in str(value or ""):
            if char == "\\":
                backslashes += 1
                continue
            if char == marker and backslashes % 2 == 0:
                count += 1
            backslashes = 0
        return count

    @staticmethod
    def _recover_tail_conclusion(candidate: str) -> str:
        meta = re.compile(
            r"(?:thinking|analysis|draft|check|constraint|instruction|prompt|format|plan|content for (?:the )?first line|final answer content|"
            r"i (?:will|should|need)|i['’]ll\s+(?:write|provide|output|include)|expects? (?:just|only)|思考|分析|草稿|检查|提示|格式)",
            re.IGNORECASE,
        )
        for paragraph in reversed(re.split(r"\n\s*\n+", str(candidate or ""))):
            value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", paragraph.strip())
            if not 2 <= len(value) <= 800 or meta.search(value):
                continue
            conclusion = re.match(r"^(?:因此|所以|故|综上|从而|可得|结论|即)", value)
            formula = len(value) <= 240 and bool(re.fullmatch(
                r"[$\\A-Za-z0-9_{}()[\].,+\-*/^=<>≤≥∈\s]+", value
            )) and "=" in value
            if not (conclusion or formula):
                continue
            answer = Finalizer._clean(value)
            if not Finalizer.validate_structure(answer):
                return answer
        return ""
