"""Local theorem, method, and checklist cards for the public solve path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.problem_spec import ProblemSpec


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    kind: str
    domain: str
    text: str
    keywords: tuple[str, ...]
    topics: tuple[str, ...] = ()
    text_en: str = ""

    def render(self, language: str) -> str:
        """Render in the contract language, falling back to the source note."""
        if language == "en" and self.text_en:
            return self.text_en
        return self.text

    def supports(self, language: str) -> bool:
        """Whether this card has prose suitable for the requested language."""
        if language == "en":
            return bool(self.text_en) or not re.search(r"[\u4e00-\u9fff]", self.text)
        return bool(re.search(r"[\u4e00-\u9fff]", self.text)) or not re.search(
            r"[A-Za-z]{2,}", self.text
        )


@dataclass(frozen=True)
class RetrievalBundle:
    solve_cards: tuple[KnowledgeCard, ...]
    review_cards: tuple[KnowledgeCard, ...]
    solve_scores: tuple[int, ...] = ()
    review_scores: tuple[int, ...] = ()
    language: str = "zh"

    def solve_context(self) -> str:
        return _render(self.solve_cards, self.language)

    def review_context(self) -> str:
        return _render(self.review_cards, self.language)

    def trace_content(self) -> dict:
        return {
            "solve_card_ids": [card.id for card in self.solve_cards],
            "review_card_ids": [card.id for card in self.review_cards],
            "solve_card_scores": list(self.solve_scores),
            "review_card_scores": list(self.review_scores),
            "language": self.language,
        }


_DOMAIN_BY_NAME = {
    "回归": "线性回归",
    "抽象代数": "抽象代数", "数论": "数论", "数值线性": "数值分析", "线性代数": "线性代数",
    "离散": "离散数学", "图论": "离散数学", "组合": "离散数学", "概率": "概率论",
    "统计": "统计推断", "实分析": "数学分析", "微积分": "数学分析", "测度": "测度积分",
    "常微分": "常微分方程", "偏微分": "偏微分方程", "PDE": "偏微分方程", "复分析": "复分析", "拓扑": "拓扑学",
    "泛函": "泛函分析", "优化": "运筹学", "几何": "微分几何", "答案": "answer",
}


_METHOD_CARDS = (
    KnowledgeCard(
        "method.proof.direct", "method", "proof",
        "证明时先写清题设与目标；引用定理前核对前提，再给出从条件到结论的关键推导。",
        ("证明", "prove", "show"),
        text_en="State the hypotheses and target first. Check every theorem's assumptions, then give the key deductions from the hypotheses to the conclusion.",
    ),
    KnowledgeCard(
        "method.count.inclusion", "method", "离散数学",
        "计数题先明确对象是否有序、是否允许重复；容斥法必须定义违例事件并检查交集层数。",
        ("计数", "组合", "排列", "count"),
        text_en="For counting, determine whether order matters and repetition is allowed. For inclusion-exclusion, define the bad events and check every required intersection level.",
    ),
    KnowledgeCard(
        "method.algebra.structure", "method", "抽象代数",
        "代数结构题先展开定义，再检查运算、子结构、同态或商结构的必要条件。",
        ("群", "环", "域", "同态", "ideal"),
        text_en="Expand the definitions first, then check every required condition on the operation, substructure, homomorphism, or quotient.",
    ),
    KnowledgeCard(
        "method.analysis.conditions", "method", "数学分析",
        "分析题使用定理前检查连续、可导、可积、收敛或支配等前提，并单独检查端点。",
        ("极限", "积分", "级数", "limit", "integral"),
        text_en="Before applying an analysis theorem, verify continuity, differentiability, integrability, convergence, or domination assumptions, and check endpoints separately.",
    ),
    KnowledgeCard(
        "method.olympiad.geometry", "method", "初等几何",
        "几何题先记录共线、共圆、平行和切线等结构；优先检查相似、圆周角、点幂或面积比，并验证退化情形。",
        ("triangle", "circle", "cyclic", "tangent", "collinear", "angle", "geometry"),
        ("olympiad_geometry",),
        "Record collinear, cyclic, parallel, and tangent structures first. Test similarity, inscribed angles, power of a point, or area ratios, and verify degenerate cases.",
    ),
    KnowledgeCard(
        "method.olympiad.number_theory", "method", "数论",
        "整数题先固定整除与互素条件；结合模运算、因式分解、估值或无限递降，并证明所得解已穷尽。",
        ("integer", "divisible", "congruence", "modulo", "prime", "gcd", "diophantine"),
        ("olympiad_number_theory",),
        "Fix the divisibility and coprimality conditions first. Use congruences, factorization, valuations, or descent, and prove that the resulting solutions are exhaustive.",
    ),
    KnowledgeCard(
        "method.olympiad.combinatorics", "method", "离散数学",
        "组合题先明确对象、有序性和重复规则。固定宽度、局部转移或网格路径计数应优先定义有限状态与转移矩阵，并用小规模边界核验；其余再考虑双计数、鸽巢、不变量、极端原理或容斥。",
        ("coloring", "arrangement", "pigeonhole", "counting", "subset", "tournament"),
        ("olympiad_combinatorics",),
        "Specify the objects, ordering, and repetition rules. For fixed-width grids, local transitions, or path counts, define a finite state and transfer recurrence first and verify small boundary cases. Otherwise use double counting, pigeonhole, an invariant, an extremal argument, or inclusion-exclusion.",
    ),
    KnowledgeCard(
        "method.olympiad.functional_equation", "method", "高等代数",
        "函数方程先代入特殊值寻找常数项、零点和对称性，再检验单射、满射或加法性；最后逐一回代全部候选。",
        ("function", "functional", "substitution", "injective", "surjective"),
        ("olympiad_functional_equation",),
        "Substitute special values to find constants, zeros, and symmetries. Test injectivity, surjectivity, or additivity, then substitute every candidate back into the original equation.",
    ),
    KnowledgeCard(
        "method.olympiad.inequality", "method", "高等代数",
        "不等式先检查齐次性、定义域和等号条件，再选择 AM-GM、Cauchy、凸性或变量代换；极值结论必须验证可达。",
        ("inequality", "positive", "minimum", "maximum", "cauchy", "am-gm"),
        ("olympiad_inequality",),
        "Check homogeneity, domain, and equality conditions before choosing AM-GM, Cauchy, convexity, or a substitution. Verify that any claimed extremum is attained.",
    ),
    KnowledgeCard(
        "method.olympiad.polynomial", "method", "高等代数",
        "多项式题结合根与系数、整除性、重根和插值约束；列出候选后检查次数、首项系数及所有根。",
        ("polynomial", "roots", "vieta", "monic", "divisibility"),
        ("olympiad_polynomial",),
        "Use relations between roots and coefficients, divisibility, repeated roots, and interpolation constraints. Check the degree, leading coefficient, and every root for each candidate.",
    ),
    KnowledgeCard(
        "method.olympiad.sequence", "method", "离散数学",
        "数列题先列初值并寻找单调性、不变量或周期；递推变形后用归纳法验证通项和全部边界。",
        ("sequence", "recurrence", "recursive", "induction", "invariant"),
        ("olympiad_sequence",),
        "List initial values and look for monotonicity, invariants, or periodicity. After transforming the recurrence, verify the closed form and all boundary cases by induction.",
    ),
    KnowledgeCard(
        "fact.lz78.encoding", "theorem", "离散数学",
        "LZ78 从只含空串的字典出发；每次输出（最长已有前缀的索引，下一个新字符），再把“前缀+新字符”加入字典。编码串必须同时编码索引和新字符，不能只串联新字符的代码；应明确索引起点与字段宽度。",
        ("lempel", "ziv", "lz78", "phrases", "decomposition", "encoded", "string", "dictionary"),
        text_en=(
            "LZ78 starts with a dictionary containing only the empty string. At each step, output "
            "(the index of the longest dictionary prefix, the next new character), then insert prefix+character. "
            "The encoded bit string must encode both the index and the new character; concatenating only the "
            "new-character codes is incomplete. State the index origin and fixed field width used by the exercise."
        ),
    ),
    KnowledgeCard(
        "check.all_goals", "check", "answer",
        "最终答案必须覆盖题目的全部所求对象；多问题按题目顺序分别作答，不能只给中间量。",
        ("所有", "分别", "证明", "求"),
        text_en="The final answer must cover every requested object. Answer multipart questions in order, and do not substitute an intermediate quantity for the requested result.",
    ),
    KnowledgeCard(
        "check.roots", "check", "answer",
        "方程题必须列出全部根、检查定义域和伪根；离散根不能写成区间。",
        ("方程", "根", "solve", "equation"),
        text_en="List all roots, enforce the domain, and reject extraneous solutions. Do not express a discrete root set as an interval.",
    ),
    KnowledgeCard(
        "check.proof", "check", "proof",
        "证明必须具备关键条件、依据、推导链和结论；仅写定理名称或结论不构成证明。",
        ("证明", "prove", "show"),
        text_en="A proof needs the key hypotheses, justification, deduction chain, and conclusion. A theorem name or bare conclusion alone is not a proof.",
    ),
    KnowledgeCard(
        "check.counterexample", "check", "proof",
        "审查全称命题或逆命题时，尝试检查边界情形、反例以及量词方向。",
        ("是否", "当且仅当", "every", "if and only if"),
        text_en="For universal or converse claims, test boundary cases, possible counterexamples, and the direction of every quantifier.",
    ),
    KnowledgeCard(
        "check.olympiad.exhaustiveness", "check", "answer",
        "竞赛题复核时必须检查全部分支、整数或几何边界、等号情形，并确认没有把必要条件误当充分条件。",
        ("all", "integer", "boundary", "equality", "所有", "穷尽"),
        (
            "olympiad_geometry", "olympiad_number_theory", "olympiad_combinatorics",
            "olympiad_functional_equation", "olympiad_inequality", "olympiad_polynomial",
            "olympiad_sequence", "olympiad_general",
        ),
        "Independently check every branch, integer or geometric boundary, and equality case. Confirm that no necessary condition was used as though it were sufficient.",
    ),
)


class CardRetriever:
    """A dependency-free diversified retriever over bundled knowledge notes."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or Path("rag") / "knowledge"
        self.cards = [*_METHOD_CARDS, *self._load_cards()]

    def retrieve(self, spec: "ProblemSpec") -> RetrievalBundle:
        contract = getattr(spec, "answer_contract", None)
        raw_language = getattr(contract, "language", None) or getattr(spec.profile, "language", "zh")
        language = "en" if raw_language == "en" else "zh"
        scored = self._score(spec, language)
        solve = self._top_confident(scored, include_kinds={"theorem", "method"}, min_score=9)
        solve_ids = {card.id for _, card in solve}
        review_scored = sorted(
            ((score + (3 if card.kind == "check" else 0), card) for score, card in scored),
            key=lambda item: (item[0], item[1].id),
            reverse=True,
        )
        review = self._top_confident(
            review_scored,
            include_kinds={"check", "method"},
            min_score=9,
            excluded_ids=solve_ids,
        )
        return RetrievalBundle(
            tuple(card for _, card in solve),
            tuple(card for _, card in review),
            tuple(score for score, _ in solve),
            tuple(score for score, _ in review),
            language,
        )

    def _load_cards(self) -> list[KnowledgeCard]:
        if not self.knowledge_dir.is_dir():
            return []
        cards = []
        for path in sorted(self.knowledge_dir.glob("*.txt")):
            domain = self._domain(path.stem)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                text = line.strip()
                if not text or text.startswith("主题："):
                    continue
                cards.append(KnowledgeCard(
                    id=f"note.{path.stem}.{index}",
                    kind="theorem",
                    domain=domain,
                    text=text[:420],
                    keywords=tuple(_tokens(text)),
                ))
        return cards

    @staticmethod
    def _domain(name: str) -> str:
        for marker, domain in _DOMAIN_BY_NAME.items():
            if marker in name:
                return domain
        return "进阶数学"

    def _score(self, spec: "ProblemSpec", language: str = "zh") -> list[tuple[int, KnowledgeCard]]:
        topic = getattr(spec.profile, "topic", "general")
        contract = getattr(spec, "answer_contract", None)
        support = tuple(getattr(contract, "explicit_support_requirements", ()))
        query = set(_tokens(" ".join([
            spec.profile.subject, spec.profile.problem_type, topic, spec.primary_method,
            spec.alternative_method, *spec.constraints, *spec.risk_flags,
            *support,
            *(goal.instruction for goal in spec.goals),
        ])))
        domain_tokens = set(_tokens(spec.profile.subject))
        proof_goal = any(goal.kind == "proof" for goal in spec.goals)
        scored = []
        for card in self.cards:
            if not card.supports(language):
                continue
            if card.topics and topic not in card.topics:
                continue
            # Subject agreement is the strongest retrieval signal. Long notes
            # naturally contain generic n-grams such as "what method" and can
            # otherwise outrank a shorter operation-specific card from the
            # actual mathematical domain.
            score = 12 if card.domain == spec.profile.subject else 0
            if card.domain == "proof" and proof_goal:
                score += 6
            if card.domain == "answer":
                score += 2
            if topic != "general" and topic in card.topics:
                score += 9
            # Domain membership already has its own score.  Counting every
            # overlapping n-gram of the domain name made generic cards outrank
            # a card matching the actual requested operation.
            score += 2 * len(query.intersection(card.keywords) - domain_tokens)
            if card.kind == "method" and (
                card.domain == spec.profile.subject or (card.domain == "proof" and proof_goal)
            ):
                score += 3
            if card.kind == "check" and any(flag in {"missing_roots", "theorem_scope", "double_counting", "multiple_goals"} for flag in spec.risk_flags):
                score += 2
            if score:
                scored.append((score, card))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)

    @staticmethod
    def _top_confident(
        scored: list[tuple[int, KnowledgeCard]],
        include_kinds: set[str],
        min_score: int,
        excluded_ids: set[str] | None = None,
    ) -> list[tuple[int, KnowledgeCard]]:
        excluded = excluded_ids or set()
        for score, card in scored:
            if score < min_score:
                break
            if card.kind in include_kinds and card.id not in excluded:
                return [(score, card)]
        return []

    @staticmethod
    def _diverse(
        scored: list[tuple[int, KnowledgeCard]],
        limit: int,
        include_kinds: set[str],
        min_score: int,
    ) -> list[tuple[int, KnowledgeCard]]:
        selected: list[tuple[int, KnowledgeCard]] = []
        domains = set()
        kinds = set()
        for score, card in scored:
            if score < min_score:
                continue
            if card.kind not in include_kinds:
                continue
            if card.id in {item.id for _, item in selected}:
                continue
            if card.domain in domains and card.kind in kinds and len(selected) >= 2:
                continue
            selected.append((score, card))
            domains.add(card.domain)
            kinds.add(card.kind)
            if len(selected) == limit:
                break
        return selected


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{2,}|\d+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(
            run[index:index + width]
            for width in (2, 3, 4)
            for index in range(max(0, len(run) - width + 1))
        )
    return tokens


def _render(cards: tuple[KnowledgeCard, ...], language: str = "zh") -> str:
    return "\n".join(f"- {card.render(language)}" for card in cards)
