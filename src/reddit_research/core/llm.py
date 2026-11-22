"""Minimal Anthropic Messages API client shared by synthesis, rerank and
query expansion.

One blocking call, text in / text out. Every failure raises :class:`LLMError`
with a specific, redacted reason so callers can degrade gracefully (skip the
LLM step, keep the lexical result) and tell the user why.
"""

from __future__ import annotations

import json

import httpx

from .errors import redact

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMError(Exception):
    """An LLM call could not produce text; message is redacted."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))


def complete(
    *,
    api_key: str | None,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    effort: str | None = None,
    timeout: float = 60.0,
) -> tuple[str, bool]:
    """Return ``(text, truncated)`` for one system+user prompt.

    ``truncated`` is True when the model stopped on ``max_tokens``. ``effort``
    ("low" | "medium" | "high" | ...) trades thoroughness for speed/cost; leave
    ``None`` for the model's default.
    """

    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is empty")
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if effort:
        payload["output_config"] = {"effort": effort}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    try:
        resp = httpx.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise LLMError(f"request to Anthropic failed: {type(exc).__name__}: {exc}") from exc
    if resp.status_code >= 400:
        raise LLMError(
            f"Anthropic API HTTP {resp.status_code} (model {model!r}): {_api_error_message(resp)}"
        )
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise LLMError("Anthropic API returned non-JSON response") from exc
    stop_reason = data.get("stop_reason")
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise LLMError(f"Anthropic API returned no text (stop_reason={stop_reason!r})")
    return text, stop_reason == "max_tokens"


def _api_error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f"{err.get('type', 'error')}: {err.get('message', '')}".strip()
    except (json.JSONDecodeError, AttributeError):
        return resp.text[:300]
