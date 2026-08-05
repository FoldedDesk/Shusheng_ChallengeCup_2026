from __future__ import annotations

from core.submission_agent import SubmissionAgent


class ReasoningAgent:
    """Official competition entry point."""

    def __init__(self, client, *args, **kwargs) -> None:
        del args, kwargs
        self.agent = SubmissionAgent(client)

    def solve(self, problem: str, metadata: dict) -> dict:
        return self.agent.solve(problem, metadata)
