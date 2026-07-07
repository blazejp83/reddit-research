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
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

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

    Raises no network errors to the caller by design: on failure it returns
    ``None`` and leaves the orchestrator to attach a warning.
    """

    provider = (config.provider or "").lower()
    if not provider or not evidence:
        return None
    if provider != "anthropic":
        # Only Anthropic is wired up in the MVP; other providers no-op for now.
        return None
    if not config.api_key:
        return None

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
        resp.raise_for_status()
        data = resp.json()
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return "".join(parts).strip() or None
    except httpx.HTTPError:
        return None
    except (KeyError, json.JSONDecodeError):
        return None


class SynthesisError(Exception):
    """Raised only when the caller explicitly requires synthesis and it fails."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))
