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


@dataclass(frozen=True)
class RetrievalBundle:
    solve_cards: tuple[KnowledgeCard, ...]
    review_cards: tuple[KnowledgeCard, ...]

    def solve_context(self) -> str:
        return _render(self.solve_cards)

    def review_context(self) -> str:
        return _render(self.review_cards)

    def trace_content(self) -> dict:
        return {
            "solve_card_ids": [card.id for card in self.solve_cards],
            "review_card_ids": [card.id for card in self.review_cards],
        }


_DOMAIN_BY_NAME = {
    "抽象代数": "抽象代数", "数论": "数论", "线性代数": "线性代数", "数值线性": "线性代数",
    "离散": "离散数学", "图论": "离散数学", "组合": "离散数学", "概率": "概率论",
    "统计": "统计推断", "实分析": "数学分析", "微积分": "数学分析", "测度": "测度积分",
    "常微分": "常微分方程", "偏微分": "偏微分方程", "复分析": "复分析", "拓扑": "拓扑学",
    "泛函": "泛函分析", "优化": "运筹学", "几何": "微分几何", "答案": "answer",
}


_METHOD_CARDS = (
    KnowledgeCard("method.proof.direct", "method", "proof", "证明时先写清题设与目标；引用定理前核对前提，再给出从条件到结论的关键推导。", ("证明", "prove", "show")),
    KnowledgeCard("method.count.inclusion", "method", "离散数学", "计数题先明确对象是否有序、是否允许重复；容斥法必须定义违例事件并检查交集层数。", ("计数", "组合", "排列", "count")),
    KnowledgeCard("method.algebra.structure", "method", "抽象代数", "代数结构题先展开定义，再检查运算、子结构、同态或商结构的必要条件。", ("群", "环", "域", "同态", "ideal")),
    KnowledgeCard("method.analysis.conditions", "method", "数学分析", "分析题使用定理前检查连续、可导、可积、收敛或支配等前提，并单独检查端点。", ("极限", "积分", "级数", "limit", "integral")),
    KnowledgeCard("check.all_goals", "check", "answer", "最终答案必须覆盖题目的全部所求对象；多问题按题目顺序分别作答，不能只给中间量。", ("所有", "分别", "证明", "求")),
    KnowledgeCard("check.roots", "check", "answer", "方程题必须列出全部根、检查定义域和伪根；离散根不能写成区间。", ("方程", "根", "solve", "equation")),
    KnowledgeCard("check.proof", "check", "proof", "证明必须具备关键条件、依据、推导链和结论；仅写定理名称或结论不构成证明。", ("证明", "prove", "show")),
    KnowledgeCard("check.counterexample", "check", "proof", "审查全称命题或逆命题时，尝试检查边界情形、反例以及量词方向。", ("是否", "当且仅当", "every", "if and only if")),
)


class CardRetriever:
    """A dependency-free diversified retriever over bundled knowledge notes."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or Path("rag") / "knowledge"
        self.cards = [*_METHOD_CARDS, *self._load_cards()]

    def retrieve(self, spec: "ProblemSpec") -> RetrievalBundle:
        scored = self._score(spec)
        solve = self._diverse(scored, limit=2, include_kinds={"theorem", "method", "check"})
        review = self._diverse(
            [(score + (3 if card.kind == "check" else 0), card) for score, card in scored],
            limit=1,
            include_kinds={"check", "method"},
        )
        return RetrievalBundle(tuple(solve), tuple(review))

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

    def _score(self, spec: "ProblemSpec") -> list[tuple[int, KnowledgeCard]]:
        query = set(_tokens(" ".join([
            spec.profile.subject, spec.profile.problem_type, spec.primary_method,
            spec.alternative_method, *spec.constraints, *spec.risk_flags,
            *(goal.instruction for goal in spec.goals),
        ])))
        scored = []
        for card in self.cards:
            score = 4 if card.domain == spec.profile.subject else 0
            if card.domain == "answer":
                score += 1
            score += 2 * len(query.intersection(card.keywords))
            if card.kind == "method" and card.domain in {spec.profile.subject, "proof"}:
                score += 3
            if card.kind == "check" and any(flag in {"missing_roots", "theorem_scope", "double_counting", "multiple_goals"} for flag in spec.risk_flags):
                score += 2
            if score:
                scored.append((score, card))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)

    @staticmethod
    def _diverse(scored: list[tuple[int, KnowledgeCard]], limit: int, include_kinds: set[str]) -> list[KnowledgeCard]:
        selected = []
        domains = set()
        kinds = set()
        for _, card in scored:
            if card.kind not in include_kinds:
                continue
            if card.id in {item.id for item in selected}:
                continue
            if card.domain in domains and card.kind in kinds and len(selected) >= 2:
                continue
            selected.append(card)
            domains.add(card.domain)
            kinds.add(card.kind)
            if len(selected) == limit:
                break
        return selected


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{2,}|\d+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run[index:index + width] for width in (2, 3) for index in range(max(0, len(run) - width + 1)))
    return tokens


def _render(cards: tuple[KnowledgeCard, ...]) -> str:
    return "\n".join(f"[{card.id}] {card.text}" for card in cards)
