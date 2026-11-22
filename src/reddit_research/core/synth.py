"""Optional, provider-agnostic answer synthesis.

Synthesis is deliberately decoupled from retrieval: it takes ranked evidence and
produces a cited answer in the spec's preferred shape (short answer / agreement
/ disagreements / caveats / sources). It never asserts Reddit comments as fact
and always cites permalinks. If no provider is configured it returns ``None`` so
callers can fall back to returning evidence only.
"""

from __future__ import annotations

import json

import httpx

from .config import SynthConfig
from .errors import redact
from .models import Evidence

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You synthesize answers from Reddit discussion evidence for a personal research tool.

Rules:
- Treat Reddit comments as opinions and anecdotes, NOT verified facts.
- Distinguish broad consensus from isolated anecdotes.
- Surface notable disagreement and uncertainty.
- Note when evidence is thin, outdated, or specific to one subreddit.
- Cite claims using the bracketed source numbers provided, e.g. [3].

Output this structure with Markdown headings:
## Short answer
## What Reddit users seem to agree on
## Notable disagreements
## Caveats
## Sources
(Under Sources, list each cited [n] with its permalink.)"""


def _build_user_prompt(question: str, evidence: list[Evidence]) -> str:
    lines = [f"Question: {question}", "", "Evidence:"]
    for i, ev in enumerate(evidence, start=1):
        header = f"[{i}] r/{ev.subreddit} — {ev.kind} (reddit score {ev.reddit_score})"
        lines.append(header)
        lines.append(f"    permalink: {ev.permalink}")
        lines.append(f"    text: {ev.text.strip()[:1200]}")
        lines.append("")
    return "\n".join(lines)


def synthesize(
    question: str,
    evidence: list[Evidence],
    config: SynthConfig,
    timeout: float = 60.0,
) -> str | None:
    """Return a synthesized cited answer, or ``None`` if synthesis is disabled.

    Returns ``None`` only when no provider is configured (synthesis off by
    choice). Any other reason synthesis can't run -- unsupported provider,
    missing key, API/network failure -- raises :class:`SynthesisError` with a
    redacted, specific reason so the caller can surface it as a warning.
    """

    provider = (config.provider or "").lower()
    if not provider:
        return None
    if not evidence:
        raise SynthesisError("no evidence to synthesize from")
    if provider != "anthropic":
        raise SynthesisError(f"unsupported provider {provider!r} (only 'anthropic' is supported)")
    if not config.api_key:
        raise SynthesisError("provider 'anthropic' is set but ANTHROPIC_API_KEY is empty")

    model = config.model or DEFAULT_ANTHROPIC_MODEL
    payload = {
        "model": model,
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_user_prompt(question, evidence)}],
    }
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        resp = httpx.post(ANTHROPIC_ENDPOINT, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise SynthesisError(f"request to Anthropic failed: {type(exc).__name__}: {exc}") from exc
    if resp.status_code >= 400:
        raise SynthesisError(
            f"Anthropic API HTTP {resp.status_code} (model {model!r}): {_api_error_message(resp)}"
        )
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise SynthesisError("Anthropic API returned non-JSON response") from exc
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise SynthesisError(
            f"Anthropic API returned no text (stop_reason={data.get('stop_reason')!r})"
        )
    return text


def _api_error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", {})
        return f"{err.get('type', 'error')}: {err.get('message', '')}".strip()
    except (json.JSONDecodeError, AttributeError):
        return resp.text[:300]


class SynthesisError(Exception):
    """Synthesis was enabled but could not produce an answer; message is redacted."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))
