"""Auditable retrieval over general theorem, method, and checking cards."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    domains: tuple[str, ...] = ()
    theorem_name: str = ""
    provenance: str = ""

    @property
    def effective_domains(self) -> tuple[str, ...]:
        return self.domains or (self.domain,)

    def render(self, language: str) -> str:
        return self.text_en if language == "en" and self.text_en else self.text


@dataclass(frozen=True)
class RetrievalBundle:
    solve_cards: tuple[KnowledgeCard, ...]
    review_cards: tuple[KnowledgeCard, ...]
    solve_scores: tuple[int, ...] = ()
    review_scores: tuple[int, ...] = ()
    language: str = "zh"
    primary_subject: str = ""
    secondary_subject: str = ""
    subject_confidence: str = "low"

    def solve_context(self) -> str:
        return "\n".join(f"- {card.render(self.language)}" for card in self.solve_cards)

    def review_context(self) -> str:
        return "\n".join(f"- {card.render(self.language)}" for card in self.review_cards)

    def verification_fact_context(self) -> str:
        # Independent reviewers receive theorem facts only when the theorem is
        # explicitly named by the public statement and therefore not a hidden
        # answer-bearing hint.
        for card in self.review_cards:
            if card.kind == "theorem" and card.theorem_name:
                return card.render(self.language)
        return ""

    def trace_content(self) -> dict:
        return {
            "solve_card_ids": [card.id for card in self.solve_cards],
            "review_card_ids": [card.id for card in self.review_cards],
            "solve_scores": list(self.solve_scores),
            "review_scores": list(self.review_scores),
            "language": self.language,
            "primary_subject": self.primary_subject,
            "secondary_subject": self.secondary_subject,
            "subject_confidence": self.subject_confidence,
        }


class CardRetriever:
    """Retrieve a small, diverse context without storing solved problems."""

    def __init__(self, knowledge_path: Path | None = None) -> None:
        self.knowledge_path = knowledge_path or Path(__file__).parent / "knowledge" / "cards.json"
        self.cards = tuple(self._load_cards())

    def retrieve(self, spec: "ProblemSpec") -> RetrievalBundle:
        language = getattr(spec.answer_contract, "language", spec.profile.language)
        primary = getattr(spec.profile, "primary_subject", spec.profile.subject)
        secondary = getattr(spec.profile, "secondary_subject", "")
        confidence = getattr(spec.profile, "subject_confidence", "low")
        scored = self._score(spec)

        solve: list[tuple[int, KnowledgeCard]] = []
        used_families: set[tuple[str, str]] = set()
        theorem_count = 0
        for score, card in scored:
            if score < 9 or card.kind not in {"method", "theorem"}:
                continue
            family = (
                card.kind,
                card.topics[0] if card.topics else card.id.split(".")[1],
            )
            if family in used_families or (card.kind == "theorem" and theorem_count >= 1):
                continue
            solve.append((score, card))
            used_families.add(family)
            theorem_count += card.kind == "theorem"
            if len(solve) == 3:
                break

        solve_ids = {card.id for _, card in solve}
        review_candidates = [
            (score + (4 if card.kind == "check" else 0), card)
            for score, card in scored
            if card.id not in solve_ids and card.kind in {"check", "method"}
        ]
        review_candidates.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        review: list[tuple[int, KnowledgeCard]] = []
        review_kinds: set[str] = set()
        for score, card in review_candidates:
            if score < 9 or card.kind in review_kinds:
                continue
            review.append((score, card))
            review_kinds.add(card.kind)
            if len(review) == 2:
                break

        return RetrievalBundle(
            tuple(card for _, card in solve),
            tuple(card for _, card in review),
            tuple(score for score, _ in solve),
            tuple(score for score, _ in review),
            language,
            primary,
            secondary,
            confidence,
        )

    def _load_cards(self) -> list[KnowledgeCard]:
        try:
            payload = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        raw_cards = payload.get("cards", []) if isinstance(payload, dict) else []
        cards: list[KnowledgeCard] = []
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            identifier = str(raw.get("id", "")).strip()
            kind = str(raw.get("kind", "")).strip()
            domains = tuple(str(item).strip() for item in raw.get("domains", []) if str(item).strip())
            text = str(raw.get("text_zh", "")).strip()
            if not identifier or kind not in {"theorem", "method", "check"} or not domains or not text:
                continue
            cards.append(KnowledgeCard(
                id=identifier,
                kind=kind,
                domain=domains[0],
                text=text,
                keywords=tuple(str(item).casefold() for item in raw.get("keywords", []) if str(item).strip()),
                topics=tuple(str(item) for item in raw.get("topics", []) if str(item).strip()),
                text_en=str(raw.get("text_en", "")).strip(),
                domains=domains,
                theorem_name=str(raw.get("theorem_name", "")).strip(),
                provenance=str(raw.get("provenance", "")).strip(),
            ))
        return cards

    def _score(self, spec: "ProblemSpec") -> list[tuple[int, KnowledgeCard]]:
        text = str(getattr(spec, "problem_text", "")).casefold()
        query_tokens = set(_tokens(text))
        semantics = getattr(spec, "semantics", None)
        named_theorems = {
            item.casefold() for item in getattr(semantics, "named_theorems", ())
        }
        requested_methods = {
            item.casefold() for item in getattr(semantics, "requested_methods", ())
        }
        primary = getattr(spec.profile, "primary_subject", spec.profile.subject)
        secondary = getattr(spec.profile, "secondary_subject", "")
        confidence = getattr(spec.profile, "subject_confidence", "low")
        topic = getattr(spec.profile, "topic", "general")
        proof = getattr(spec.profile, "task_kind", "") in {"proof", "derivation", "explanation"}
        scored: list[tuple[int, KnowledgeCard]] = []
        for card in self.cards:
            overlap = len(query_tokens.intersection(card.keywords))
            phrase_overlap = sum(1 for keyword in card.keywords if " " in keyword and keyword in text)
            score = 0
            if primary in card.effective_domains:
                score += {"high": 10, "medium": 7, "low": 2}.get(confidence, 0)
            if secondary and secondary in card.effective_domains:
                score += 4
            score += min(9, overlap * 3 + phrase_overlap * 2)
            if topic in card.topics:
                score += 5
            searchable = " ".join((card.theorem_name, card.text, card.text_en)).casefold()
            if any(name in searchable or searchable in name for name in named_theorems):
                score += 12
            if any(method in searchable or searchable in method for method in requested_methods):
                score += 8
            if card.kind == "check" and proof and card.id == "check.proof.counterexample":
                score += 7
            if card.kind == "check" and spec.profile.answer_shape == "choice" and card.id == "check.choice.polarity":
                score += 8
            if card.id == "check.output.contract" and any(
                requirement.strict
                for goal in spec.goals
                for requirement in goal.requirements
            ):
                score += 5
            if card.kind == "theorem":
                theorem_key = card.theorem_name.casefold()
                named = bool(
                    theorem_key
                    and (
                        theorem_key in text
                        or any(
                            name in searchable or searchable in name
                            for name in named_theorems
                        )
                    )
                )
                if not named and not (confidence == "high" and overlap >= 2):
                    continue
            if confidence == "low" and overlap == 0 and card.kind != "check":
                continue
            if score:
                scored.append((score, card))
        return sorted(scored, key=lambda item: (item[0], item[1].id), reverse=True)


def _tokens(value: str) -> tuple[str, ...]:
    latin = re.findall(r"[a-z][a-z0-9_-]{1,}", value.casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", value)
    grams: list[str] = []
    for phrase in chinese:
        grams.extend(phrase[index:index + 2] for index in range(max(1, len(phrase) - 1)))
        grams.append(phrase)
    return tuple(dict.fromkeys((*latin, *grams)))
