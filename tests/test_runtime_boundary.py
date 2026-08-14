import asyncio
from concurrent.futures import CancelledError as FuturesCancelledError
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_failure import is_recoverable_runtime_failure
from core.serializer import safe_json
from core.stage_budget import StageBudget
from core.submission_agent import SubmissionAgent
from user_agent import ReasoningAgent


class FunctionTimedOut(BaseException):
    """Compatible stand-in for func_timeout.FunctionTimedOut."""


def _agent_with_internal(internal) -> ReasoningAgent:
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.agent = internal
    return agent


@pytest.mark.parametrize("error", [RuntimeError("failed"), TimeoutError("late")])
def test_ordinary_exceptions_are_recoverable(error):
    assert is_recoverable_runtime_failure(error)


def test_func_timeout_compatible_base_exception_is_recoverable():
    assert is_recoverable_runtime_failure(FunctionTimedOut("late"))


def test_memory_error_is_not_recoverable():
    assert not is_recoverable_runtime_failure(MemoryError("out of memory"))


def test_grouped_ordinary_client_errors_are_recoverable():
    error = ExceptionGroup(
        "client failures",
        [RuntimeError("bad response"), TimeoutError("provider timeout")],
    )
    assert is_recoverable_runtime_failure(error)


@pytest.mark.parametrize(
    "fatal",
    [MemoryError("out of memory"), asyncio.CancelledError(), SystemExit(1)],
)
def test_group_with_fatal_or_cancellation_is_not_recoverable(fatal):
    error = BaseExceptionGroup("mixed failures", [RuntimeError("client"), fatal])
    assert not is_recoverable_runtime_failure(error)


def test_nested_group_with_cancellation_is_not_recoverable():
    error = BaseExceptionGroup(
        "outer",
        [
            RuntimeError("client"),
            BaseExceptionGroup("inner", [asyncio.CancelledError()]),
        ],
    )
    assert not is_recoverable_runtime_failure(error)


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        SystemExit(),
        GeneratorExit(),
        asyncio.CancelledError(),
        FuturesCancelledError(),
    ],
)
def test_fatal_and_cancellation_failures_are_not_recoverable(error):
    assert not is_recoverable_runtime_failure(error)


def test_entrypoint_degrades_func_timeout_to_json_safe_answer():
    class TimedOutAgent:
        def solve(self, problem, metadata):
            raise FunctionTimedOut("simulated provider timeout")

    result = _agent_with_internal(TimedOutAgent()).solve(
        r"Compute the value. Put the final answer within \boxed{}.",
        {"idx": 1},
    )

    assert result["final_response"] == r"\boxed{0}"
    assert result["trace"][0]["content"] == {
        "status": "failed",
        "degraded": True,
        "type": "FunctionTimedOut",
    }
    json.dumps(result, ensure_ascii=False)


def test_entrypoint_serializes_internal_trace_and_normalizes_answer_type():
    class Diagnostic:
        def __str__(self):
            return "diagnostic"

    class InternalAgent:
        def solve(self, problem, metadata):
            return {"final_response": 7, "trace": [Diagnostic()]}

    result = _agent_with_internal(InternalAgent()).solve("求值。", {"idx": 2})

    assert result == {"final_response": "7", "trace": ["diagnostic"]}
    json.dumps(result, ensure_ascii=False)


def test_entrypoint_normalizes_non_list_trace_to_platform_contract():
    class InternalAgent:
        def solve(self, problem, metadata):
            return {"final_response": "  4  ", "trace": {"private": "shape"}}

    result = _agent_with_internal(InternalAgent()).solve("求值。", {"idx": 3})

    assert result["final_response"] == "4"
    assert result["trace"] == [{
        "step": "entrypoint",
        "content": {"status": "trace_normalized"},
    }]
    json.dumps(result, ensure_ascii=False)


def test_entrypoint_preserves_answer_when_trace_is_recursive():
    trace = []
    trace.append(trace)

    class InternalAgent:
        def solve(self, problem, metadata):
            return {"final_response": "42", "trace": trace}

    result = _agent_with_internal(InternalAgent()).solve("求值。", {"idx": 31})

    assert result["final_response"] == "42"
    assert result["trace"] == ["<recursive reference>"]
    json.dumps(result, ensure_ascii=False)


def test_entrypoint_preserves_answer_when_trace_value_cannot_stringify():
    class BadDiagnostic:
        def __str__(self):
            raise ValueError("cannot stringify")

    class InternalAgent:
        def solve(self, problem, metadata):
            return {
                "final_response": "9",
                "trace": [{"diagnostic": BadDiagnostic()}],
            }

    result = _agent_with_internal(InternalAgent()).solve("求值。", {"idx": 32})

    assert result["final_response"] == "9"
    assert result["trace"][0]["diagnostic"] == "<unserializable BadDiagnostic>"
    json.dumps(result, ensure_ascii=False)


def test_safe_json_handles_recursive_mapping():
    value = {}
    value["self"] = value

    assert safe_json(value) == {"self": "<recursive reference>"}


def test_entrypoint_replaces_empty_or_none_answer_with_safe_fallback():
    class EmptyAgent:
        def solve(self, problem, metadata):
            return {"final_response": None, "trace": []}

    result = _agent_with_internal(EmptyAgent()).solve("求值。", {"idx": 4})

    assert result["final_response"] == "0"
    assert result["trace"][0]["content"]["status"] == "invalid_internal_response"
    json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        SystemExit(),
        GeneratorExit(),
        MemoryError(),
        asyncio.CancelledError(),
        FuturesCancelledError(),
    ],
)
def test_entrypoint_reraises_fatal_and_cancellation_failures(error):
    class FatalAgent:
        def solve(self, problem, metadata):
            raise error

    with pytest.raises(type(error)):
        _agent_with_internal(FatalAgent()).solve("求值。", {"idx": 5})


def test_model_call_converts_provider_timeout_into_empty_stage_result():
    class TimedOutClient:
        def chat(self, **kwargs):
            raise FunctionTimedOut("provider request exceeded its window")

    trace = []
    answer, truncated = SubmissionAgent(TimedOutClient())._call(
        "Compute 2+2.", "verify", 128, trace, 100.0
    )

    assert answer == ""
    assert not truncated
    assert trace[-1]["content"]["status"] == "failed"
    assert trace[-1]["content"]["failure_kind"] == "provider_timeout"


def test_model_call_does_not_swallow_fatal_or_cancellation_failure():
    class FatalClient:
        def __init__(self, error):
            self.error = error

        def chat(self, **kwargs):
            raise self.error

    for error in (KeyboardInterrupt(), asyncio.CancelledError()):
        with pytest.raises(type(error)):
            SubmissionAgent(FatalClient(error))._call(
                "Compute 2+2.", "verify", 128, [], 100.0
            )


def test_optional_call_requires_full_request_window_and_reserve():
    budget = StageBudget(
        solve_tokens=100,
        review_tokens=100,
        repair_tokens=100,
        review_min_remaining_seconds=0,
        repair_min_remaining_seconds=0,
        allow_review=True,
        allow_repair=True,
        max_calls=4,
    )
    trace = [{"step": "model_call_solve", "content": {"status": "completed"}}]

    with patch("core.submission_agent.monotonic", return_value=235.0):
        assert SubmissionAgent._can_call(trace, budget, 100.0, 135)
    with patch("core.submission_agent.monotonic", return_value=235.001):
        assert not SubmissionAgent._can_call(trace, budget, 100.0, 135)


def test_call_count_limit_still_precedes_time_admission():
    budget = StageBudget(
        solve_tokens=100,
        review_tokens=100,
        repair_tokens=100,
        review_min_remaining_seconds=0,
        repair_min_remaining_seconds=0,
        allow_review=True,
        allow_repair=True,
        max_calls=1,
    )
    trace = [{"step": "model_call_solve", "content": {"status": "failed"}}]

    with patch("core.submission_agent.monotonic", return_value=100.0):
        assert not SubmissionAgent._can_call(trace, budget, 100.0, 135)


def test_slow_truncated_first_stage_still_admits_required_continuation():
    agent = SubmissionAgent.__new__(SubmissionAgent)
    budget = StageBudget(
        solve_tokens=100,
        review_tokens=100,
        repair_tokens=100,
        review_min_remaining_seconds=0,
        repair_min_remaining_seconds=0,
        allow_review=True,
        allow_repair=True,
        max_calls=4,
    )
    spec = type("Spec", (), {"profile": type("Profile", (), {})()})()

    with patch("core.submission_agent.monotonic", return_value=236.0):
        mode, reason = agent._review_decision(
            spec,
            [],
            "unfinished derivation",
            budget,
            100.0,
            provider_truncated=True,
        )

    assert (mode, reason) == (
        "continue",
        "truncated_without_complete_result",
    )


def test_slow_complete_first_stage_skips_only_optional_review():
    agent = SubmissionAgent.__new__(SubmissionAgent)
    budget = StageBudget(
        solve_tokens=100,
        review_tokens=100,
        repair_tokens=100,
        review_min_remaining_seconds=0,
        repair_min_remaining_seconds=0,
        allow_review=True,
        allow_repair=True,
        require_independent_review=True,
        max_calls=4,
    )
    candidate = type("Candidate", (), {
        "validation_tier": "complete",
        "shape_valid": True,
        "formatting_valid": True,
        "coverage_uncertain": False,
    })()
    spec = type("Spec", (), {"profile": type("Profile", (), {})()})()

    with patch("core.submission_agent.monotonic", return_value=236.0):
        mode, reason = agent._review_decision(
            spec,
            [candidate],
            r"FINAL: \boxed{4}",
            budget,
            100.0,
        )

    assert (mode, reason) == ("", "insufficient_optional_review_time")
