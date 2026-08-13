from __future__ import annotations

from dataclasses import dataclass
import re

from tools.latex_parser import normalize_latex


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
        r"^(?:最终答案|完整答案|完整结论|答案|final(?:\s+answer)?|answer|check\s+format(?:ting)?|format(?:ting)?|(?:final\s+)?(?:conclusion|response|done)|[.。…`'\"，,]+)$",
        re.IGNORECASE,
    )
    _META = re.compile(
        r"(?:<think\b|thinking process|(?im:^\s*(?:analysis|drafting)\s*[:：])|check formatting|check spacing|"
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
        r"让我(?:验证|确认|组织)|我(?:需要|应该)|输出时)",
        re.IGNORECASE,
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
                if result.method in {"label", "label_boxed", "bracket_label"}
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
            r"(?im)^\s*(?:\*{1,3}|_{1,3})?\s*(?:"
            r"【\s*(?:最终答案|答案|结论)\s*】\s*[:：为=]?|"
            r"(?:the\s+)?(?:final(?:\s+answer)?|answer|conclusion)\s*(?:is|equals|[:：=])|"
            r"(?:最终\s*)?答案\s*[:：为=]|结论\s*[:：为=])"
            r"\s*(?:\*{1,3}|_{1,3})?\s*([^\n]+)"
        )
        for match in label_pattern.finditer(text):
            value = re.sub(r"\s*(?:\*{1,3}|_{1,3})\s*$", "", match.group(1)).strip()
            boxed = Finalizer._last_boxed(value)
            method = "label_boxed" if boxed is not None else "label"
            value = boxed if boxed is not None else value
            fragments.append((
                match.start(),
                Finalizer._result(value, method, raw_has_meta=has_meta, explicit=True),
            ))
        for match in Finalizer._BRACKET_LABEL.finditer(text):
            value = match.group(1).strip().strip("` ")
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
            r"^\s*(?:[-*]\s*)?(?:thinking process|analysis|wait\b|okay\b|"
            r"i\s+(?:need|will|should|can|must)\b|i['’]ll\s+(?:write|provide|output|include)\b|check\b|one\s+(?:detail|more|adjustment)\b|"
            r"final\s+(?:check|plan|polish)\b|double\s+check\b|revised\s+(?:body|draft|proof)\b|plan\s*:|"
            r"count\s+lines?\s*:|check\s+(?:the\s+)?line\s+count\b|line\s+\d+\s*:|draft(?:\s+\d+)?\s*:|"
            r"refin(?:e|ing)\b|need\s+to\b|the\s+prompt\b|(?:this\s+)?looks?\s+(?:compliant|solid)\b|"
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
            r"\[\s*(?:(?:explanation|proof|reasoning|derivation|answer|content)(?:\s+text)?|"
            r"insert\s+[^\]]+|(?:解释|证明|推理|推导|答案|内容)(?:文本)?)\s*\]",
            value,
            re.IGNORECASE,
        ):
            reasons.append("placeholder")
        if re.search(r"并给出全部结论.*(?:必要依据|必要算式)|给出全部结论.*再写", value):
            reasons.append("placeholder")
        if re.search(r"(?:证明|结论|依据|推导)\s*[:：]\s*(?:\.{2,}|…+)", value, re.IGNORECASE):
            reasons.append("placeholder")
        if re.search(
            r"\b(?:or similar|or equivalent|or something(?: similar)?|maybe|perhaps|probably)\b|"
            r"(?:或|及)(?:其他)?类似(?:答案|形式|结果)?|诸如此类",
            value,
            re.IGNORECASE,
        ):
            reasons.append("uncertain_fragment")
        if re.search(r"\b(?:this|that) (?:looks|seems) like\b|\bspecific test case\b|\blooks like noise\b", value, re.IGNORECASE):
            reasons.append("meta_text")
        if re.search(r"\bthis (?:phrasing|wording|instruction|prompt)\b", value, re.IGNORECASE):
            reasons.append("meta_text")
        if Finalizer._META.search(value):
            reasons.append("meta_text")
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
        if re.search(r"[,，:：;；=+*/^\\]\s*$", value):
            reasons.append("trailing_fragment")
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
        marker = r"\boxed{"
        values: list[tuple[int, str, bool]] = []
        offset = 0
        while True:
            position = text.find(marker, offset)
            if position < 0:
                break
            start = position + len(marker)
            depth = 1
            for index in range(start, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        values.append((position, text[start:index].strip(), True))
                        offset = index + 1
                        break
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
