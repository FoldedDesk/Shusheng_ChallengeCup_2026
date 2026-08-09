"""Compatibility helpers for platform and local chat-client responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelCallResult:
    content: str
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def provider_truncated(self) -> bool:
        return self.finish_reason.lower() in {"length", "max_tokens", "token_limit"}


def coerce_model_response(response: Any) -> ModelCallResult:
    """Accept the documented string response plus common structured variants."""
    if isinstance(response, ModelCallResult):
        return response
    if isinstance(response, str) or response is None:
        return ModelCallResult(str(response or ""))
    if isinstance(response, Mapping):
        content = response.get("content")
        finish_reason = response.get("finish_reason", "")
        usage = response.get("usage", {})
        if content is None:
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, Mapping):
                    message = choice.get("message", {})
                    if isinstance(message, Mapping):
                        content = message.get("content", "")
                    finish_reason = choice.get("finish_reason", finish_reason)
        return ModelCallResult(str(content or ""), str(finish_reason or ""), usage if isinstance(usage, Mapping) else {})

    content = getattr(response, "content", None)
    finish_reason = getattr(response, "finish_reason", "")
    usage = getattr(response, "usage", {})
    return ModelCallResult(
        str(content if content is not None else response),
        str(finish_reason or ""),
        usage if isinstance(usage, Mapping) else {},
    )
