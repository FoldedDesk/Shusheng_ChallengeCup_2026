from __future__ import annotations

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
                "final_response": "未能生成可验证的数学答案。",
                "trace": [{"step": "entrypoint", "content": {"status": "invalid_internal_response"}}],
            }
        except Exception as exc:
            # The platform invokes solve one item at a time. Never let an
            # internal parser or tool failure invalidate the entire batch.
            return {
                "final_response": "未能生成可验证的数学答案。",
                "trace": [{"step": "entrypoint", "content": {"status": "failed", "type": type(exc).__name__}}],
            }
