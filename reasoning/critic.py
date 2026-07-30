from __future__ import annotations

from core.client_adapter import ClientAdapter
from core.exception_handler import retry_once


class Critic:
    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def review(self, problem: str, candidate: str) -> str:
        result = retry_once(lambda: self.client_adapter.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"题目：\n{problem}\n\n候选：\n{candidate}"},
            ],
            temperature=0.0,
            max_tokens=512,
        ))
        return str(result or "审核不可用")
