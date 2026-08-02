from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import List


class LocalRetriever:
    """Tiny offline TF-IDF retriever over bundled mathematical notes."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or Path("rag") / "knowledge"
        self.documents = self._load_documents()

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.documents:
            return []
        tokens = self._tokens(query)
        if not tokens:
            return []
        counts = Counter(token for _, text in self.documents for token in set(self._tokens(text)))
        total = len(self.documents)
        scored = []
        for name, text in self.documents:
            words = self._tokens(text)
            frequencies = Counter(words)
            score = sum(
                frequencies[token] * math.log((total + 1) / (counts[token] + 1))
                for token in tokens
            )
            title_tokens = set(self._tokens(Path(name).stem))
            score += sum(2.0 for token in tokens if token in title_tokens)
            if score:
                scored.append((score, name, text))
        return [f"{name}: {text[:600]}" for _, name, text in sorted(scored, reverse=True)[:min(top_k, 5)]]

    def _load_documents(self) -> List[tuple[str, str]]:
        if not self.knowledge_dir.is_dir():
            return []
        documents = []
        for path in sorted(self.knowledge_dir.glob("*.txt")):
            try:
                documents.append((path.name, path.read_text(encoding="utf-8")))
            except OSError:
                continue
        return documents

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens: List[str] = []
        for chinese in re.findall(r"[\u4e00-\u9fff]+", text):
            # Chinese text has no whitespace word boundaries. Overlapping
            # bigrams preserve lightweight offline retrieval while matching
            # terms such as "牛顿法" and "条件概率" inside a full question.
            for width in (2, 3):
                if len(chinese) >= width:
                    tokens.extend(
                        chinese[index:index + width]
                        for index in range(len(chinese) - width + 1)
                    )
        tokens.extend(re.findall(r"[A-Za-z]{2,}|\d+", text.lower()))
        return tokens
