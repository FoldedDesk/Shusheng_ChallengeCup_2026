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
    if isinstance(value, Mapping):
        if "value" in value:
            return _text_content(value["value"])
        if "text" in value:
            return _text_content(value["text"])
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for part in value:
            text = _read_field(part, "text", None)
            if text is None:
                text = _read_field(part, "value", None)
            if text is None and isinstance(part, str):
                text = part
            if text is not None:
                parts.append(_text_content(text))
        if parts:
            return "".join(parts)
    value_field = _read_field(value, "value", None)
    if value_field is not None and value_field is not value:
        return _text_content(value_field)
    return str(value)


def _string_value(value: Any, default: str = "") -> str:
    """Normalize enum-like SDK fields without exposing their object repr."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    scalar = _read_field(value, "value", None)
    return str(scalar if scalar is not None else value)


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


def _tool_call(value: Any) -> dict[str, Any] | None:
    """Normalize one SDK or mapping tool call to the chat wire shape."""
    function = _read_field(value, "function", None)
    name = _read_field(function, "name", None) if function is not None else None
    arguments = _read_field(function, "arguments", None) if function is not None else None
    if name is None:
        return None
    return {
        "id": str(_read_field(value, "id", "") or ""),
        "type": _string_value(_read_field(value, "type", "function"), "function"),
        "function": {
            "name": _string_value(name),
            "arguments": _text_content(arguments),
        },
    }


def _tool_message(value: Any) -> dict[str, Any] | None:
    """Return a canonical assistant message when tool calls are present."""
    calls = _read_field(value, "tool_calls", None)
    if not isinstance(calls, (list, tuple)):
        return None
    normalized = [item for call in calls if (item := _tool_call(call)) is not None]
    if not normalized:
        return None
    return {
        "role": _string_value(_read_field(value, "role", "assistant"), "assistant"),
        "content": _text_content(_read_field(value, "content", "")),
        "tool_calls": normalized,
    }


def coerce_tool_call_message(response: Any) -> dict[str, Any] | None:
    """Extract tool calls from mapping and SDK-style chat responses."""
    candidates: list[Any] = [response]
    choice = _first_choice(response)
    if choice is not None:
        candidates.append(choice)
        message = _read_field(choice, "message", None)
        if message is not None:
            candidates.append(message)
    for candidate in candidates:
        message = _tool_message(candidate)
        if message is not None:
            return message
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
        return ModelCallResult(_text_content(content), _string_value(finish_reason), usage)

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
        _string_value(finish_reason),
        usage,
    )
