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


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an SDK/Pydantic response object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text_content(value: Any) -> str:
    """Normalize string content and the text blocks used by some chat SDKs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for part in value:
            text = _read_field(part, "text", None)
            if isinstance(text, Mapping):
                text = text.get("value", "")
            if text is None and isinstance(part, str):
                text = part
            if text is not None:
                parts.append(str(text))
        if parts:
            return "".join(parts)
    return str(value)


def _coerce_usage(value: Any) -> Mapping[str, Any]:
    """Keep usage metadata when an SDK returns a model object instead of a dict."""
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:  # pragma: no cover - defensive SDK compatibility
                continue
            if isinstance(dumped, Mapping):
                return dumped
    return {}


def _first_choice(value: Any) -> Any:
    choices = _read_field(value, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        return choices[0]
    return None


def coerce_model_response(response: Any) -> ModelCallResult:
    """Accept the documented string response plus common structured variants."""
    if isinstance(response, ModelCallResult):
        return response
    if isinstance(response, str) or response is None:
        return ModelCallResult(str(response or ""))
    if isinstance(response, Mapping):
        content = response.get("content")
        finish_reason = response.get("finish_reason", "")
        usage = _coerce_usage(response.get("usage", {}))
        if content is None:
            choice = _first_choice(response)
            if choice is not None:
                message = _read_field(choice, "message", None)
                content = (
                    _read_field(message, "content", "")
                    if message is not None
                    else _read_field(choice, "content", "")
                )
                finish_reason = _read_field(choice, "finish_reason", finish_reason)
        return ModelCallResult(_text_content(content), str(finish_reason or ""), usage)

    content = getattr(response, "content", None)
    finish_reason = getattr(response, "finish_reason", "")
    usage = _coerce_usage(getattr(response, "usage", {}))
    if content is None:
        choice = _first_choice(response)
        if choice is not None:
            message = _read_field(choice, "message", None)
            content = (
                _read_field(message, "content", "")
                if message is not None
                else _read_field(choice, "content", "")
            )
            finish_reason = _read_field(choice, "finish_reason", finish_reason)
    return ModelCallResult(
        _text_content(content if content is not None else response),
        str(finish_reason or ""),
        usage,
    )
