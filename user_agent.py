from __future__ import annotations

from typing import Dict

from core.client_adapter import ClientAdapter
from core.math_agent import MathAgent


class ReasoningAgent:
    """Official competition entry point."""

    def __init__(self, client, *args, **kwargs) -> None:
        del args, kwargs
        self.client = client
        self.agent = MathAgent(ClientAdapter(client))

    def solve(self, problem: str, metadata: Dict) -> Dict:
        return self.agent.solve(problem, metadata)
