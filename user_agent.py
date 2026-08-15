from __future__ import annotations

from core.runtime_failure import is_recoverable_runtime_failure
from core.serializer import safe_json
from core.submission_agent import SubmissionAgent


class ReasoningAgent:
    """Fixed public entry point used by the official runner."""

    def __init__(self, client, *args, **kwargs) -> None:
        del args, kwargs
        self.agent = SubmissionAgent(client)

    def solve(self, problem: str, metadata: dict) -> dict:
        try:
            result = self.agent.solve(problem, metadata)
            normalized = self._normalize_result(result)
            if normalized is not None:
                return normalized
        except BaseException as error:
            if not is_recoverable_runtime_failure(error):
                raise
            primary_failure = type(error).__name__
        else:
            primary_failure = "InvalidInternalResponse"

        try:
            recovered = self._normalize_result(self.agent.emergency_solve(problem))
            if recovered is not None:
                recovered.setdefault("trace", []).append({
                    "step": "entrypoint",
                    "content": {
                        "status": "recovered",
                        "primary_failure_type": primary_failure,
                    },
                })
                return safe_json(recovered)
        except BaseException as error:
            if not is_recoverable_runtime_failure(error):
                raise
            recovery_failure = type(error).__name__
        else:
            recovery_failure = "InvalidRecoveryResponse"

        # Reached only after both bounded model paths fail to return text.
        return safe_json({
            "final_response": "0",
            "trace": [{
                "step": "entrypoint",
                "content": {
                    "status": "degraded_after_recovery_failure",
                    "primary_failure_type": primary_failure,
                    "recovery_failure_type": recovery_failure,
                },
            }],
        })

    @staticmethod
    def _normalize_result(result: object) -> dict | None:
        if not isinstance(result, dict):
            return None
        raw_answer = result.get("final_response")
        answer = "" if raw_answer is None else str(raw_answer).strip()
        if not answer:
            return None
        trace = result.get("trace", [])
        if not isinstance(trace, list):
            trace = [{"step": "entrypoint", "content": {"status": "trace_normalized"}}]
        return safe_json({"final_response": answer, "trace": trace})
