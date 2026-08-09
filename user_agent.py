from __future__ import annotations

import re

from core.submission_agent import SubmissionAgent


class ReasoningAgent:
    """Official competition entry point."""

    def __init__(self, client, *args, **kwargs) -> None:
        del args, kwargs
        self.agent = SubmissionAgent(client)

    def solve(self, problem: str, metadata: dict) -> dict:
        try:
            result = self.agent.solve(problem, metadata)
            if isinstance(result, dict) and str(result.get("final_response", "")).strip():
                return result
            return {
                "final_response": self._safe_fallback(problem),
                "trace": [{"step": "entrypoint", "content": {
                    "status": "invalid_internal_response", "degraded": True,
                }}],
            }
        except Exception as exc:
            # The platform invokes solve one item at a time. Never let an
            # internal parser or tool failure invalidate the entire batch.
            return {
                "final_response": self._safe_fallback(problem),
                "trace": [{"step": "entrypoint", "content": {
                    "status": "failed", "type": type(exc).__name__, "degraded": True,
                }}],
            }

    @staticmethod
    def _safe_fallback(problem: str) -> str:
        boxed = bool(re.search(
            r"(?:within|inside)\s+\\boxed\s*\{\s*\}|put.*final answer.*\\boxed|\\boxed\s*\{\s*\}",
            str(problem or ""),
            re.IGNORECASE,
        ))
        return r"\boxed{0}" if boxed else "0"
