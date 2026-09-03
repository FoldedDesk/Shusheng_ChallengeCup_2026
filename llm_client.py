import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Union

import requests

from core.execution_limits import REQUEST_TIMEOUT_SECONDS
from core.model_response import ModelCallResult, coerce_model_response


DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = "intern-s2-preview"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096

ChatMessage = Dict[str, Any]
ChatResponse = Union[str, ChatMessage]


class InternChatClient:
    """Small OpenAI-compatible chat client for the competition sample."""

    def __init__(
        self,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        retry: int = 2,
        default_args: Optional[Mapping[str, Any]] = None,
        **request_args: Any,
    ) -> None:
        raw_api_key = os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        self.api_base = os.environ.get("INTERN_API_BASE", DEFAULT_API_BASE)
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = max(1, min(int(timeout), REQUEST_TIMEOUT_SECONDS))
        # One bounded transport retry absorbs a transient connection failure.
        # Mathematical attempts remain separately visible and budgeted by the
        # SubmissionAgent state machine.
        self.retry = max(1, min(int(retry), 2))
        self.default_args = dict(default_args or {})
        self.default_args.update(request_args)

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        *,
        thinking_mode: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **request_args: Any,
    ) -> ChatResponse:
        message, _, _ = self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
            tools=tools,
            request_args=request_args,
        )
        if "tool_calls" in message:
            return message
        return coerce_model_response(message).content

    def chat_result(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        *,
        thinking_mode: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **request_args: Any,
    ) -> ModelCallResult:
        message, finish_reason, usage = self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
            tools=tools,
            request_args=request_args,
        )
        content = message.get("content", "")
        if "tool_calls" in message:
            content = json.dumps(message, ensure_ascii=False)
        return coerce_model_response({
            "content": content,
            "finish_reason": finish_reason,
            "usage": usage,
        })

    def _request(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        thinking_mode: Optional[bool],
        tools: Optional[List[Dict[str, Any]]],
        request_args: Mapping[str, Any],
    ) -> tuple[ChatMessage, str, Mapping[str, Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        payload.update(self.default_args)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking_mode is not None:
            payload["thinking_mode"] = bool(thinking_mode)
        if tools is not None:
            payload["tools"] = tools
        payload.update(request_args)
        payload["messages"] = messages
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }

        last_error = None
        for attempt in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                if not isinstance(message, dict):
                    raise ValueError("API message is not an object")
                usage = data.get("usage", {})
                return (
                    message,
                    str(choice.get("finish_reason", "") or ""),
                    usage if isinstance(usage, dict) else {},
                )
            except Exception as exc:  # noqa: BLE001 - keep sample robust and simple.
                last_error = exc
                if attempt + 1 < self.retry:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Chat completion failed after {self.retry} attempts: {last_error}")
