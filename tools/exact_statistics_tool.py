from __future__ import annotations

from fractions import Fraction
import re
from typing import Optional


class ExactStatisticsTool:
    """Conservative exact handlers for small, fully specified probability laws."""

    def hints_for(self, problem: str) -> list[str]:
        return [
            hint
            for hint in (
                self._independent_event_union(problem),
                self._independent_standard_normal_sum(problem),
                self._brownian_covariance(problem),
                self._sample_mean_variance(problem),
                self._renewal_rate_limit(problem),
            )
            if hint
        ]

    @staticmethod
    def _normalize(problem: str) -> str:
        text = str(problem or "")
        text = re.sub(
            r"\\(?:d?frac)\s*\{\s*(-?\d+)\s*\}\s*\{\s*(\d+)\s*\}",
            r"\1/\2",
            text,
        )
        replacements = {
            r"\left": "",
            r"\right": "",
            r"\,": "",
            r"\;": "",
            r"\mathbb{P}": "P",
            r"\mathbf{P}": "P",
            r"\Pr": "P",
            r"\cup": "∪",
            r"\cap": "∩",
            r"\leq": "≤",
            r"\le": "≤",
            r"\bar{X}": "Xbar",
            r"\bar X": "Xbar",
            "−": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fraction(value: str) -> Optional[Fraction]:
        try:
            parsed = Fraction(str(value).strip())
        except (ValueError, ZeroDivisionError):
            return None
        return parsed

    @staticmethod
    def _render(value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"

    @staticmethod
    def _english(problem: str) -> bool:
        text = str(problem or "")
        return len(re.findall(r"[A-Za-z]{2,}", text)) > len(re.findall(r"[\u4e00-\u9fff]", text))

    @classmethod
    def _independent_event_union(cls, problem: str) -> Optional[str]:
        text = cls._normalize(problem)
        if not re.search(
            r"(?:事件\s*)?A\s*(?:,|，|与|和)\s*(?:事件\s*)?B[^。.!?]{0,50}独立|"
            r"\bA\s+and\s+B\s+are\s+independent(?:\s+events?)?\b|"
            r"\bindependent\s+events?\s+A\s+and\s+B\b",
            text,
            re.IGNORECASE,
        ):
            return None
        assignments = {}
        for event in ("A", "B"):
            values = re.findall(
                rf"P\s*\(\s*{event}\s*\)\s*=\s*(\d+(?:\.\d+|/\d+)?)",
                text,
                re.IGNORECASE,
            )
            if len(values) != 1:
                return None
            parsed = cls._fraction(values[0])
            if parsed is None or not 0 <= parsed <= 1:
                return None
            assignments[event] = parsed
        if not re.search(
            r"P\s*\(\s*A\s*∪\s*B\s*\)|P\s*\(\s*B\s*∪\s*A\s*\)|"
            r"(?:A|事件A)[^。.!?]{0,10}(?:与|和)(?:B|事件B)[^。.!?]{0,20}并事件[^。.!?]{0,20}概率|"
            r"\bprobability\s+of\s+(?:A\s+union\s+B|A\s+or\s+B)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"条件概率|互斥|不独立|补事件|期望|方差|证明|推导|"
            r"\b(?:conditional|mutually\s+exclusive|not\s+independent|complement|"
            r"expectation|variance|prove|derive)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        intersection_targets = re.findall(r"P\s*\(\s*[AB]\s*∩\s*[AB]\s*\)", text, re.IGNORECASE)
        if intersection_targets:
            return None

        pa, pb = assignments["A"], assignments["B"]
        intersection = pa * pb
        union = pa + pb - intersection
        pa_text, pb_text = cls._render(pa), cls._render(pb)
        intersection_text, union_text = cls._render(intersection), cls._render(union)
        if cls._english(problem):
            support = (
                rf"Independence gives \(P(A\cap B)=P(A)P(B)={intersection_text}\). "
                rf"Therefore \(P(A\cup B)={pa_text}+{pb_text}-{intersection_text}={union_text}\)."
            )
        else:
            support = (
                rf"独立性用于 \(P(A\cap B)=P(A)P(B)={intersection_text}\)。因此 "
                rf"\(P(A\cup B)={pa_text}+{pb_text}-{intersection_text}={union_text}\)。"
            )
        return f"本地独立事件并概率: {support}"

    @classmethod
    def _independent_standard_normal_sum(cls, problem: str) -> Optional[str]:
        text = cls._normalize(problem)
        match = re.search(
            r"([A-Z])\s*(?:,|，|、|与|和)\s*([A-Z])[^。.!?]{0,50}"
            r"独立[^。.!?]{0,40}(?:均|都)服从标准正态分布|"
            r"\b([A-Z])\s+and\s+([A-Z])\s+are\s+independent"
            r"(?:\s+and)?\s+(?:both\s+|each\s+)?standard\s+normal",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        first, second = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        if not first or not second or first.upper() == second.upper():
            return None
        first, second = first.upper(), second.upper()
        if not re.search(rf"{first}\s*\+\s*{second}|{second}\s*\+\s*{first}", text):
            return None
        asks_distribution = bool(re.search(r"分布|\bdistribution\b", text, re.IGNORECASE))
        asks_variance = bool(re.search(r"方差|\bvariance\b|Var\s*\(", text, re.IGNORECASE))
        if not (asks_distribution and asks_variance):
            return None
        if re.search(
            r"不独立|相关|条件|协方差|相关系数|期望|概率|加权|"
            r"\b(?:not\s+independent|correlated|conditional|covariance|correlation|"
            r"expectation|probability|weighted)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        result = (
            rf"{first}+{second}\sim N(0,2),\quad "
            rf"\operatorname{{Var}}({first}+{second})=2"
        )
        if cls._english(problem):
            support = (
                rf"Independent normal variables add their means and variances. Hence \({result}\)."
            )
        else:
            support = rf"由独立性，正态变量之和仍为正态且方差相加，故 \({result}\)。"
        return f"本地独立标准正态和: {support}"

    @classmethod
    def _brownian_covariance(cls, problem: str) -> Optional[str]:
        text = cls._normalize(problem)
        if not re.search(r"标准布朗运动|\bstandard\s+Brownian\s+motion\b", text, re.IGNORECASE):
            return None
        if re.search(
            r"布朗桥|分数布朗|漂移|缩放|条件|相关系数|"
            r"\b(?:Brownian\s+bridge|fractional\s+Brownian|drift|scaled|conditional|correlation)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        covariance = re.search(
            r"Cov\s*\(\s*B\s*\(\s*([a-z])\s*\)\s*,\s*B\s*\(\s*([a-z])\s*\)\s*\)",
            text,
            re.IGNORECASE,
        )
        order = re.search(r"(?:0\s*≤\s*)?([a-z])\s*≤\s*([a-z])", text, re.IGNORECASE)
        if not covariance or not order:
            return None
        cov_variables = {covariance.group(1).lower(), covariance.group(2).lower()}
        lower, upper = order.group(1).lower(), order.group(2).lower()
        if lower == upper or cov_variables != {lower, upper}:
            return None
        if not re.search(r"求|计算|写出|\b(?:find|compute|calculate|what\s+is)\b", text, re.IGNORECASE):
            return None
        result = rf"\operatorname{{Cov}}(B({lower}),B({upper}))={lower}"
        if cls._english(problem):
            support = (
                rf"Since \(B({upper})=B({lower})+[B({upper})-B({lower})]\) and the increment is "
                rf"independent of \(B({lower})\), \({result}\)."
            )
        else:
            support = (
                rf"由 \(B({upper})=B({lower})+[B({upper})-B({lower})]\) 且增量与 "
                rf"\(B({lower})\) 独立，得到 \({result}\)。"
            )
        return f"本地布朗运动协方差: {support}"

    @classmethod
    def _sample_mean_variance(cls, problem: str) -> Optional[str]:
        text = cls._normalize(problem)
        if not re.search(r"样本均值|Xbar|\bsample\s+mean\b", text, re.IGNORECASE):
            return None
        if not re.search(r"独立同分布|\bi\.?i\.?d\.?\b|independent\s+and\s+identically\s+distributed", text, re.IGNORECASE):
            return None
        variance_match = re.search(
            r"(?:总体)?方差(?:已知)?\s*(?:为|是|等于|=)\s*(\d+(?:\.\d+|/\d+)?)|"
            r"\bpopulation\s+variance\s*(?:is|equals?|=)?\s*(\d+(?:\.\d+|/\d+)?)",
            text,
            re.IGNORECASE,
        )
        size_match = re.search(
            r"样本量\s*(?:n\s*=\s*)?(\d+)|"
            r"\bsample\s+size\s*(?:n\s*=|is|equals?|=)?\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        if not variance_match or not size_match:
            return None
        variance_raw = next(group for group in variance_match.groups() if group is not None)
        size_raw = next(group for group in size_match.groups() if group is not None)
        variance = cls._fraction(variance_raw)
        size = int(size_raw)
        if variance is None or variance < 0 or size <= 0:
            return None
        if not re.search(
            r"(?:样本均值|Xbar)[^。.!?]{0,35}方差|方差[^。.!?]{0,35}(?:样本均值|Xbar)|"
            r"\bvariance\s+of\s+(?:the\s+)?sample\s+mean\b",
            text,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"不放回|有限总体修正|相关|不独立|未知方差|估计|标准误|"
            r"\b(?:without\s+replacement|finite\s+population\s+correction|correlated|"
            r"not\s+independent|unknown\s+variance|estimate|standard\s+error)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        result_value = variance / size
        result = rf"\operatorname{{Var}}(\bar X)={cls._render(result_value)}"
        variance_text = cls._render(variance)
        if cls._english(problem):
            support = (
                rf"For an i.i.d. sample, variances add and scaling by \(1/{size}\) gives "
                rf"\(\operatorname{{Var}}(\bar X)={variance_text}/{size}={cls._render(result_value)}\)."
            )
        else:
            support = (
                rf"由独立同分布，方差可相加，样本均值再乘缩放因子，故 "
                rf"\(\operatorname{{Var}}(\bar X)={variance_text}/{size}={cls._render(result_value)}\)。"
            )
        return f"本地样本均值方差: {support}"

    @classmethod
    def _renewal_rate_limit(cls, problem: str) -> Optional[str]:
        text = cls._normalize(problem)
        if not re.search(r"更新过程|\brenewal\s+process\b", text, re.IGNORECASE):
            return None
        if re.search(
            r"延迟更新|平衡更新|更新报酬|非独立|无限均值|"
            r"\b(?:delayed|equilibrium|renewal[- ]reward|dependent|infinite\s+mean)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        mean_match = re.search(
            r"(?:更新|到达)?间隔(?:的)?均值\s*(?:为|是|等于|=)\s*(\d+(?:\.\d+|/\d+)?)|"
            r"\bmean\s+(?:renewal|interarrival)\s+(?:interval|time)\s*(?:is|equals?|=)?\s*"
            r"(\d+(?:\.\d+|/\d+)?)|"
            r"\b(?:renewal|interarrival)\s+(?:interval|time)\s+has\s+mean\s*"
            r"(\d+(?:\.\d+|/\d+)?)",
            text,
            re.IGNORECASE,
        )
        if not mean_match:
            return None
        mean_raw = next(group for group in mean_match.groups() if group is not None)
        mean = cls._fraction(mean_raw)
        if mean is None or mean <= 0:
            return None
        if not re.search(
            r"N\s*\(\s*t\s*\)\s*/\s*t|N\s*\(\s*t\s*\)\s*\\?over\s*t|"
            r"\\frac\s*\{\s*N\s*\(\s*t\s*\)\s*\}\s*\{\s*t\s*\}",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(r"极限|强大数律|\blimit\b|\bstrong\s+law\b", text, re.IGNORECASE):
            return None
        if re.search(
            r"中心极限定理|方差|分布|期望|概率|证明|"
            r"\b(?:central\s+limit|variance|distribution|expectation|probability|prove)\b",
            text,
            re.IGNORECASE,
        ):
            return None

        rate = 1 / mean
        rate_text = cls._render(rate)
        result = rf"\lim_{{t\to\infty}}\frac{{N(t)}}{{t}}={rate_text}"
        if cls._english(problem):
            support = (
                rf"The renewal strong law gives \(N(t)/t\to1/\mu\). With "
                rf"\(\mu={mean_raw}\), \({result}\)."
            )
        else:
            support = (
                rf"由更新过程的强大数律，\(N(t)/t\to1/\mu\)。代入 "
                rf"\(\mu={mean_raw}\)，得到 \({result}\)。"
            )
        return f"本地更新过程强大数律: {support}"
