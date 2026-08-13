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
                self._cauchy_location_fisher_information(problem),
                self._one_dimensional_wald_statistic(problem),
                self._diagonal_gls_estimate(problem),
                self._normal_variance_confidence_interval(problem),
                self._independent_event_union(problem),
                self._independent_standard_normal_sum(problem),
                self._brownian_covariance(problem),
                self._sample_mean_variance(problem),
                self._renewal_rate_limit(problem),
                self._two_state_markov_entropy_rate(problem),
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

    @classmethod
    def _two_state_markov_entropy_rate(cls, problem: str) -> Optional[str]:
        """Return the stationary binary Markov entropy rate in exact H_2 form."""
        text = re.sub(r"\s+", " ", str(problem or "")).strip()
        match = re.fullmatch(
            r"A stationary two-state Markov information source has transition matrix\s*"
            r"\$?\\begin\{pmatrix\}\s*([^&\\]+)\s*&\s*([^&\\]+)\s*\\\\\s*"
            r"([^&\\]+)\s*&\s*([^&\\]+)\s*\\end\{pmatrix\}\$?\.\s*"
            r"Using logarithms to base\s*\$?2\$?\s*,?\s*determine its entropy rate\.",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        try:
            first, a, b, second = (Fraction(item.strip()) for item in match.groups())
        except (ValueError, ZeroDivisionError):
            return None
        if (
            any(value < 0 or value > 1 for value in (first, a, b, second))
            or first + a != 1
            or b + second != 1
            or a + b == 0
        ):
            return None

        first_weight = b / (a + b)
        second_weight = a / (a + b)

        def entropy_term(weight: Fraction, probability: Fraction) -> str:
            if not weight or probability in {Fraction(0), Fraction(1)}:
                return ""
            if probability == Fraction(1, 2):
                return cls._render(weight)
            entropy = rf"H_2\!\left({cls._render(probability)}\right)"
            if weight == 1:
                return entropy
            return rf"{cls._render(weight)}{entropy}"

        terms = [
            term
            for term in (
                entropy_term(first_weight, a),
                entropy_term(second_weight, b),
            )
            if term
        ]
        result = "+".join(terms) if terms else "0"
        return f"本地二状态Markov熵率: {result}"

    @classmethod
    def _cauchy_location_fisher_information(cls, problem: str) -> Optional[str]:
        """Recognize the normalized unit-scale Cauchy location family."""
        text = re.sub(r"\s+", "", str(problem or ""))
        normalized = (
            text.replace(r"\left", "")
            .replace(r"\right", "")
            .replace(r"\,", "")
            .replace("[", "(")
            .replace("]", ")")
            .replace(r"\theta", "theta")
            .replace(r"\pi", "pi")
            .replace(r"\{", "{")
            .replace(r"\}", "}")
        )
        iid = bool(re.search(
            r"独立同分布|i\.?i\.?d\.?|independentandidenticallydistributed",
            normalized,
            re.IGNORECASE,
        ))
        density = bool(re.search(
            r"f\(x;?theta\)=\{?pi\(?1\+\(x-theta\)\^\{?2\}?\)?\}?\^\{?-1\}?|"
            r"f\(x;?theta\)=1/\{?pi\(?1\+\(x-theta\)\^\{?2\}?\)?\}?",
            normalized,
            re.IGNORECASE,
        ))
        target = bool(re.search(
            r"(?:样本|整个样本).{0,20}(?:位置参数)?\$?theta\$?.{0,20}Fisher信息|"
            r"(?:sample|observations?).{0,30}Fisherinformation.{0,30}(?:locationparameter)?theta|"
            r"Fisherinformation.{0,30}(?:sample|observations?).{0,30}(?:locationparameter)?theta",
            normalized,
            re.IGNORECASE,
        ))
        if not (iid and density and target):
            return None
        sample_sequences = re.findall(
                r"([A-Z])_?\{?1\}?(?:,|，)"
                r"(?:\\+(?:ldots|dots|cdots)|\.{3}|…)"
                r"(?:,|，)\1_?\{?([A-Za-z]|[1-9]\d*)\}?",
                normalized,
                re.IGNORECASE,
            )
        if len(sample_sequences) != 1:
            return None
        if re.search(
            r"尺度参数|未知尺度|信息矩阵|渐近方差|证明|推导|截断|删失|条件密度|"
            r"另一组|两组样本|合并样本|联合样本|倒数|逆Fisher|每个观测|单个观测|"
            r"只观察.{0,12}符号|仅观察.{0,12}符号|符号观测|"
            r"scaleparameter|unknownscale|informationmatrix|asymptoticvariance|prove|derive|"
            r"truncated|censored|conditionaldensity|another(?:iid)?sample|"
            r"two(?:independent)?samples|pooledsample|combinedsample|reciprocal|"
            r"inverse(?:of)?(?:the)?Fisherinformation|per[- ]?observation|"
            r"informationforeachobservation|sign[- ]?only|observe(?:only)?thesign",
            normalized,
            re.IGNORECASE,
        ):
            return None
        sample_size = sample_sequences[0][1]
        subscript = sample_size if len(sample_size) == 1 else "{" + sample_size + "}"
        if sample_size.isdigit():
            information = cls._render(Fraction(int(sample_size), 2))
        else:
            information = rf"\frac{{{sample_size}}}{{2}}"
        return (
            rf"本地Cauchy位置族Fisher信息: "
            rf"I_{subscript}(\theta)={information}"
        )

    @classmethod
    def _wald_covariance_scale(cls, raw_prefix: str) -> Optional[Fraction]:
        """Parse only a direct positive scalar multiplying a covariance matrix."""
        token = str(raw_prefix or "").replace("$", "").strip()
        token = re.sub(r"\\+(?:left|right)", "", token)
        token = re.sub(r"\\+[\[(]", "", token)
        token = re.sub(
            r"^(?:(?:为|是|等于)|(?:is|equals?|given\s+by)|=|:)\s*",
            "",
            token,
            flags=re.IGNORECASE,
        ).strip()
        token = re.sub(
            r"\s*(?:\\+(?:cdot|times)|\*|×|乘以|倍的?|times?)\s*$",
            "",
            token,
            flags=re.IGNORECASE,
        ).strip()
        token = token.strip("()[] ")
        if not token:
            return Fraction(1)

        latex_fraction = re.fullmatch(
            r"\\+(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*"
            r"\{\s*(\d+)\s*\}",
            token,
        )
        if latex_fraction:
            try:
                return Fraction(int(latex_fraction.group(1)), int(latex_fraction.group(2)))
            except ZeroDivisionError:
                return None
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", token):
            return None
        return cls._fraction(token)

    @classmethod
    def _one_dimensional_wald_statistic(cls, problem: str) -> Optional[str]:
        """Compute a scalar Wald statistic for the explicit beta1+beta2 contrast."""
        text = str(problem or "")
        number = r"[-+]?\d+(?:\.\d+)?(?:/\d+)?"
        estimate = re.search(
            rf"\\widehat\s*\{{?\\?beta\}}?\s*=\s*\(\s*({number})\s*[,，]\s*"
            rf"({number})\s*\)\s*\^\s*\{{?[^}}\s]*(?:T|top)[^}}]*\}}?",
            text,
            re.IGNORECASE,
        )
        covariance = re.search(
            r"(?:协方差矩阵|covariance\s+matrix)"
            r"(?P<prefix>[^\n。.!?]{0,60}?)"
            r"\\+begin\{(?:p|b)?matrix\}(?P<body>.+?)"
            r"\\+end\{(?:p|b)?matrix\}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        target = bool(re.search(
            r"一维\s*Wald\s*(?:卡方|χ\s*\^?\s*2).{0,12}统计量|"
            r"one[- ]dimensional\s+Wald\s+(?:chi[- ]square|statistic)|"
            r"scalar\s+Wald\s+(?:chi[- ]square|statistic)",
            text,
            re.IGNORECASE,
        ))
        constraint = bool(re.search(
            r"(?:H_?\{?0\}?\s*:\s*)?\\?beta_?\{?1\}?\s*\+\s*"
            r"\\?beta_?\{?2\}?\s*=\s*0",
            text,
            re.IGNORECASE,
        ))
        if not (estimate and covariance and target and constraint):
            return None
        if re.search(
            r"多个约束|非线性约束|渐近分布|渐近协方差|p\s*[- ]?value|证明|推导|"
            r"带符号|未平方|Wald\s*[zZ]|[zZ]\s*[- ]?统计量|标准化统计量|"
            r"\\sqrt\s*\{?\s*n\s*\}?|sqrt\s*\(\s*n\s*\)|"
            r"multiple\s+constraints|nonlinear\s+constraint|asymptotic\s+(?:distribution|covariance)|"
            r"signed|unsquared|Wald\s*[zZ]|[zZ][ -]?statistic|standardized\s+statistic|prove|derive",
            text,
            re.IGNORECASE,
        ):
            return None
        covariance_scale = cls._wald_covariance_scale(covariance.group("prefix"))
        if covariance_scale is None or covariance_scale <= 0:
            return None
        rows = [
            row.strip()
            for row in re.split(r"\\{2,}", covariance.group("body"))
            if row.strip()
        ]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if len(cells) != 2 or any(len(row) != 2 for row in cells):
            return None
        try:
            beta = (Fraction(estimate.group(1)), Fraction(estimate.group(2)))
            matrix = tuple(
                tuple(covariance_scale * Fraction(cell) for cell in row)
                for row in cells
            )
        except (ValueError, ZeroDivisionError):
            return None
        if matrix[0][1] != matrix[1][0]:
            return None
        contrast_variance = matrix[0][0] + 2 * matrix[0][1] + matrix[1][1]
        if contrast_variance <= 0:
            return None
        statistic = (beta[0] + beta[1]) ** 2 / contrast_variance
        return f"本地一维Wald统计量: {cls._render(statistic)}"

    @classmethod
    def _diagonal_gls_estimate(cls, problem: str) -> Optional[str]:
        """Solve a two-parameter GLS problem with an explicit diagonal Omega."""
        text = str(problem or "")
        matrix = re.search(
            r"(?<![A-Za-z])X\s*=\s*\\begin\{(?:p|b)?matrix\}(.+?)"
            r"\\end\{(?:p|b)?matrix\}",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        response = re.search(
            r"(?<![A-Za-z])y\s*=\s*\(([^()]+)\)\s*\^\s*"
            r"\{?[^}\s]*(?:T|top)[^}]*\}?",
            text,
            re.IGNORECASE,
        )
        diagonal = re.search(
            r"\\?Omega\s*=\s*\\operatorname\s*\{diag\}\s*\(([^()]+)\)",
            text,
            re.IGNORECASE,
        )
        covariance = bool(re.search(
            r"误差协方差矩阵.{0,30}(?:与|为)\s*\$?\\?Omega.{0,80}成比例|"
            r"error\s+covariance(?:\s+matrix)?.{0,50}proportional\s+to.{0,20}\\?Omega",
            text,
            re.IGNORECASE,
        ))
        target = bool(re.search(
            r"求\s*GLS\s*估计|find\s+(?:the\s+)?GLS\s+estimate|"
            r"compute\s+(?:the\s+)?GLS\s+estimator",
            text,
            re.IGNORECASE,
        ))
        if not (matrix and response and diagonal and covariance and target):
            return None
        if re.search(
            r"拟合值|残差|协方差.{0,20}(?:估计|的协方差)|标准误|证明|推导|"
            r"逆协方差|精度矩阵|\\?Omega\s*\^\s*\{?\s*-1\s*\}?|"
            r"参数约束|约束条件|仅求|只求|斜率|截距|\\?beta_?\{?[012]\}?\s*=|"
            r"fitted\s+values?|residuals?|covariance\s+of\s+(?:the\s+)?estimate|"
            r"standard\s+errors?|inverse\s+covariance|precision\s+matrix|prove|derive|"
            r"constrained|subject\s+to|only\s+(?:the\s+)?(?:slope|intercept|first|second)\b|"
            r"(?:slope|intercept)\s+(?:component|estimate)",
            text,
            re.IGNORECASE,
        ):
            return None

        rows = [row.strip() for row in re.split(r"\\\\", matrix.group(1)) if row.strip()]
        x_cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        y_cells = [cell.strip() for cell in re.split(r"[,，]", response.group(1)) if cell.strip()]
        omega_cells = [cell.strip() for cell in re.split(r"[,，]", diagonal.group(1)) if cell.strip()]
        if len(x_cells) < 2 or any(len(row) != 2 for row in x_cells):
            return None
        if len(y_cells) != len(x_cells) or len(omega_cells) != len(x_cells):
            return None
        try:
            x = tuple(tuple(Fraction(cell) for cell in row) for row in x_cells)
            y = tuple(Fraction(cell) for cell in y_cells)
            omega = tuple(Fraction(cell) for cell in omega_cells)
        except (ValueError, ZeroDivisionError):
            return None
        if any(value <= 0 for value in omega):
            return None
        weights = tuple(1 / value for value in omega)
        normal = tuple(tuple(
            sum((weights[k] * x[k][i] * x[k][j] for k in range(len(x))), Fraction())
            for j in range(2)
        ) for i in range(2))
        rhs = tuple(
            sum((weights[k] * x[k][i] * y[k] for k in range(len(x))), Fraction())
            for i in range(2)
        )
        determinant = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
        if determinant == 0:
            return None
        beta0 = (normal[1][1] * rhs[0] - normal[0][1] * rhs[1]) / determinant
        beta1 = (-normal[1][0] * rhs[0] + normal[0][0] * rhs[1]) / determinant
        result = (
            rf"\begin{{pmatrix}}{cls._render(beta0)}\\{cls._render(beta1)}"
            rf"\end{{pmatrix}}"
        )
        return f"本地对角协方差GLS估计: {result}"

    @classmethod
    def _normal_variance_confidence_interval(cls, problem: str) -> Optional[str]:
        """Invert two supplied chi-square quantiles for a normal variance."""
        text = str(problem or "")
        normal = bool(re.search(
            r"正态总体\s*\$?N\s*\(\s*\\?mu\s*[,，]\s*\\?sigma\s*\^\s*\{?2\}?\s*\)|"
            r"normal\s+population\s*\$?N\s*\(\s*\\?mu\s*[,，]\s*\\?sigma\s*\^\s*\{?2\}?\s*\)",
            text,
            re.IGNORECASE,
        ))
        size = re.search(
            r"样本量(?:为|=)?\s*\$?(\d+)\$?|sample\s+size(?:\s+is|\s*=)?\s*\$?(\d+)",
            text,
            re.IGNORECASE,
        )
        sum_square = re.search(
            r"\\sum_?\{?i=1\}?\^\{?(\d+)\}?\s*"
            r"\(\s*X_?\{?i\}?\s*-\s*\\bar\s*X\s*\)\s*\^\s*\{?2\}?\s*=\s*"
            r"(\d+(?:\.\d+)?(?:/\d+)?)",
            text,
            re.IGNORECASE,
        )
        quantiles = re.findall(
            r"\\chi\s*\^\s*\{?2\}?_\{?\s*(0\.\d+)\s*[,，]\s*(\d+)\s*\}?\s*=\s*"
            r"(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        level = re.search(
            r"双侧\s*\$?(\d+(?:\.\d+)?)\s*\\?%\$?\s*置信区间|"
            r"two[- ]sided\s*\$?(\d+(?:\.\d+)?)\s*\\?%\$?\s+confidence\s+interval",
            text,
            re.IGNORECASE,
        )
        target = bool(re.search(
            r"(?:求|计算|给出|构造)[^。！？!?\n]{0,40}"
            r"\\?sigma\s*\^\s*\{?2\}?[^。！？!?\n]{0,40}置信区间|"
            r"\b(?:find|compute|give|construct)\b[^.!?\n]{0,50}"
            r"(?:confidence\s+interval[^.!?\n]{0,30}\\?sigma\s*\^\s*\{?2\}?|"
            r"\\?sigma\s*\^\s*\{?2\}?[^.!?\n]{0,30}confidence\s+interval)",
            text,
            re.IGNORECASE,
        ))
        if not (normal and size and sum_square and level and target and len(quantiles) == 2):
            return None
        if re.search(
            r"均值已知|单侧|标准差|证明|推导|p\s*[- ]?value|"
            r"区间(?:的)?(?:宽度|长度|中点)|保留\s*(?:\d+|[一二两三四五六七八九十]+)位|精确到|小数位|"
            r"只给近似|仅给近似|只给精确|仅给精确|无需近似|不需要近似|开区间|"
            r"known\s+mean|one[- ]sided|standard\s+deviation|prove|derive|"
            r"interval\s+(?:width|length|midpoint)|round(?:ed|ing)?\s+to|decimal\s+places?|"
            r"only\s+(?:an?\s+)?approximate|approximate\s+values?\s+only|"
            r"only\s+(?:an?\s+)?exact|exact\s+values?\s+only|without\s+(?:an?\s+)?approximation|"
            r"do\s+not\s+(?:give|include|report)\s+(?:an?\s+)?approximation|open\s+interval",
            text,
            re.IGNORECASE,
        ):
            return None
        sample_size = int(next(group for group in size.groups() if group))
        sum_count = int(sum_square.group(1))
        if sample_size < 2 or sum_count != sample_size:
            return None
        confidence = Fraction(next(group for group in level.groups() if group)) / 100
        alpha = 1 - confidence
        expected_probabilities = {alpha / 2, 1 - alpha / 2}
        parsed = []
        for probability_raw, degrees_raw, value_raw in quantiles:
            probability = Fraction(probability_raw)
            degrees = int(degrees_raw)
            value = Fraction(value_raw)
            if degrees != sample_size - 1 or value <= 0:
                return None
            parsed.append((probability, value, value_raw))
        if {item[0] for item in parsed} != expected_probabilities:
            return None
        high = next(item for item in parsed if item[0] == 1 - alpha / 2)
        low = next(item for item in parsed if item[0] == alpha / 2)
        sum_value = Fraction(sum_square.group(2))
        if sum_value <= 0:
            return None
        lower = sum_value / high[1]
        upper = sum_value / low[1]
        if not 0 < lower < upper:
            return None
        exact = rf"\left[\frac{{{sum_square.group(2)}}}{{{high[2]}}},\frac{{{sum_square.group(2)}}}{{{low[2]}}}\right]"
        approximation = rf"\approx[{float(lower):.3f},{float(upper):.3f}]"
        return f"本地正态总体方差置信区间: {exact}{approximation}"

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
