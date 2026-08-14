from __future__ import annotations

import re

from core.runtime_failure import is_recoverable_runtime_failure
from core.serializer import safe_json
from core.submission_agent import SubmissionAgent


class ReasoningAgent:
    """Official competition entry point."""

    def __init__(self, client, *args, **kwargs) -> None:
        del args, kwargs
        self.agent = SubmissionAgent(client)

    def solve(self, problem: str, metadata: dict) -> dict:
        try:
            result = self.agent.solve(problem, metadata)
            if isinstance(result, dict):
                raw_answer = result.get("final_response")
                answer = "" if raw_answer is None else str(raw_answer).strip()
                if answer:
                    payload = {"final_response": answer}
                    if "trace" in result:
                        payload["trace"] = self._normalize_trace(result["trace"])
                    return safe_json(payload)
            return self._degraded_result(problem, "invalid_internal_response")
        except BaseException as exc:
            if not is_recoverable_runtime_failure(exc):
                raise
            # The platform invokes solve one item at a time. Never let an
            # internal parser or tool failure invalidate the entire batch.
            return self._degraded_result(problem, "failed", type(exc).__name__)

    @staticmethod
    def _normalize_trace(trace: object) -> list:
        if not isinstance(trace, list):
            return [{
                "step": "entrypoint",
                "content": {"status": "trace_normalized"},
            }]
        try:
            normalized = safe_json(trace)
        except BaseException as exc:
            if not is_recoverable_runtime_failure(exc):
                raise
            return [{
                "step": "entrypoint",
                "content": {
                    "status": "trace_serialization_failed",
                    "type": type(exc).__name__,
                },
            }]
        return normalized if isinstance(normalized, list) else []

    @classmethod
    def _degraded_result(
        cls,
        problem: str,
        status: str,
        failure_type: str = "",
    ) -> dict:
        content = {"status": status, "degraded": True}
        if failure_type:
            content["type"] = failure_type
        result = safe_json({
            "final_response": cls._safe_fallback(problem),
            "trace": [{"step": "entrypoint", "content": content}],
        })
        return result

    @staticmethod
    def _safe_fallback(problem: str) -> str:
        boxed = bool(re.search(
            r"(?:within|inside)\s+\\boxed\s*\{\s*\}|put.*final answer.*\\boxed|\\boxed\s*\{\s*\}",
            str(problem or ""),
            re.IGNORECASE,
        ))
        return r"\boxed{0}" if boxed else "0"
