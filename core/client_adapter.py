from __future__ import annotations

from typing import Any, Dict, List


class ClientAdapter:
    """The only runtime boundary to the official public client API."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        return self.client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
