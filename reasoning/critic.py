from __future__ import annotations

from typing import List

from core.client_adapter import ClientAdapter
from core.exception_handler import retry_once


class Critic:
    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def review(self, problem: str, candidate: str) -> str:
        return self.review_all(problem, [candidate])[0]

    def review_all(self, problem: str, candidates: List[str], attempts: int = 2) -> List[str]:
        if not candidates:
            return []
        combined = "\n\n".join(
            f"候选{i}：\n{candidate[:6000]}" for i, candidate in enumerate(candidates)
        )
        result = retry_once(lambda: self.client_adapter.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"题目：\n{problem}\n\n{combined}"},
            ],
            temperature=0.0,
            max_tokens=1024,
        ), attempts=attempts)
        return [str(result or "审核不可用")]
