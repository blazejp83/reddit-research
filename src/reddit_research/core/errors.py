"""Structured warnings and credential redaction.

The pipeline prefers returning actionable warnings over raising, so partial
results still reach the caller. All user-facing error text is passed through
:func:`redact` so secrets pulled from the environment never leak into logs.
"""

from __future__ import annotations

import os
import re

# Warning codes surfaced to callers (mirrors the spec's Error Handling list).
SEARCH_UNAVAILABLE = "search_provider_unavailable"
REDDIT_AUTH_FAILED = "reddit_oauth_failed"
RATE_LIMITED = "rate_limit_reached"
THREAD_UNAVAILABLE = "thread_unavailable"
COMMENTS_TRUNCATED = "comments_partially_loaded"
FEW_RESULTS = "too_few_results"
SYNTH_UNAVAILABLE = "llm_synthesis_unavailable"
RERANK_UNAVAILABLE = "llm_rerank_unavailable"
EXPANSION_UNAVAILABLE = "query_expansion_unavailable"


class RedditResearchError(Exception):
    """Base class for library errors; message is redacted on init."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))


class ConfigError(RedditResearchError):
    """Missing or invalid configuration (e.g. absent API credentials)."""


class SearchError(RedditResearchError):
    """Search provider failed or returned no usable results."""


class RedditError(RedditResearchError):
    """Reddit authentication or fetch failure."""


# Env var names whose *values* must never appear in output.
_SECRET_ENV_VARS = (
    "REDDIT_CLIENT_SECRET",
    "REDDIT_PASSWORD",
    "REDDIT_REFRESH_TOKEN",
    "BRAVE_SEARCH_API_KEY",
    "TAVILY_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
)

# Bearer/API-key-ish tokens embedded in provider error strings.
_TOKEN_RE = re.compile(r"(?i)(bearer\s+|api[_-]?key[=:\s]+|token[=:\s]+)([A-Za-z0-9._\-]{8,})")


def redact(text: str) -> str:
    """Remove known secret values and token-shaped substrings from ``text``."""

    if not text:
        return text
    for name in _SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value and len(value) >= 4 and value in text:
            text = text.replace(value, "***redacted***")
    text = _TOKEN_RE.sub(lambda m: f"{m.group(1)}***redacted***", text)
    return text


def warning(code: str, message: str) -> str:
    """Format a redacted, code-prefixed warning string."""

    return f"[{code}] {redact(message)}"
