from __future__ import annotations

import requests

import llm_client
from llm_client import InternChatClient


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{
                "message": {"content": r"FINAL: \boxed{2}"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 8},
        }


def test_transport_failure_is_retried_once(monkeypatch) -> None:
    monkeypatch.setenv("INTERN_API_KEY", "test-key")
    attempts = iter((requests.ConnectionError("temporary"), _Response()))
    sleeps: list[int] = []

    def fake_post(*args, **kwargs):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)

    result = InternChatClient(retry=2).chat_result([
        {"role": "user", "content": "Compute 1+1."},
    ])

    assert result.content == r"FINAL: \boxed{2}"
    assert result.finish_reason == "stop"
    assert result.usage == {"completion_tokens": 8}
    assert sleeps == [1]


def test_transport_retry_is_strictly_bounded(monkeypatch) -> None:
    monkeypatch.setenv("INTERN_API_KEY", "test-key")
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.ConnectionError("still unavailable")

    monkeypatch.setattr(llm_client.requests, "post", fail)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    try:
        InternChatClient(retry=99).chat_result([
            {"role": "user", "content": "Compute 1+1."},
        ])
    except RuntimeError as error:
        assert "after 2 attempts" in str(error)
    else:
        raise AssertionError("expected a bounded transport failure")

    assert calls == 2
